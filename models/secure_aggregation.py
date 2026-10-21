"""完整模型布局、两组聚合及同批量化明文检查；不在此处执行训练。"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from models.Fed import FedAvg


# 原曲线 secp256k1（Charm/OpenSSL ID 714）的标量阶。
# 明文模式无需导入 Charm；TRS-FL 启动时核对实际群阶，避免两条路径使用不同模数。
DEFAULT_MODULUS = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class AggregationError(RuntimeError):
    pass


class ModelLayout:
    """按名称排序展开整个 state_dict，保存每段形状、类型和位置以严格恢复。"""
    def __init__(self, template):
        if not template:
            raise AggregationError("Empty model state")
        self.keys = sorted(template)
        self.entries, self.dimension = [], 0
        self.dtypes = {}
        for key in self.keys:
            tensor = template[key]
            if not tensor.is_floating_point():
                # 当前 CNN 没有整数 buffer；新增 BatchNorm 等模型时需明确另定策略。
                raise AggregationError("Unsupported nonfloating model state: " + key)
            size = tensor.numel()
            self.entries.append(dict(name=key, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                     offset=self.dimension, size=size))
            self.dtypes[key] = tensor.dtype
            self.dimension += size
        if self.dimension <= 0:
            raise AggregationError("Empty parameter vector")

    def manifest(self):
        return dict(order="sorted-state-dict-keys", dimension=self.dimension, entries=self.entries)

    def flatten(self, state):
        if set(state) != set(self.keys):
            raise AggregationError("Model state keys differ from registered layout")
        values = []
        for entry in self.entries:
            tensor = state[entry['name']]
            if list(tensor.shape) != entry['shape'] or tensor.dtype != self.dtypes[entry['name']]:
                raise AggregationError("Model shape or dtype changed: " + entry['name'])
            values.append(tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy())
        vector = np.concatenate(values)
        if vector.size != self.dimension or not np.isfinite(vector).all():
            raise AggregationError("Invalid dimension or nonfinite model coordinate")
        return vector

    def restore(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise AggregationError("Invalid decoded complete-model vector")
        state = OrderedDict()
        for entry in self.entries:
            start, size = entry['offset'], entry['size']
            tensor = torch.from_numpy(vector[start:start+size].copy()).reshape(entry['shape'])
            tensor = tensor.to(dtype=self.dtypes[entry['name']])
            if not torch.isfinite(tensor).all().item():
                raise AggregationError("Decoded value is not representable in model dtype")
            state[entry['name']] = tensor
        return state


class FixedPointCodec:
    """负数映射为 z mod q；先恢复有符号整数和，再在实数域除尺度和人数。"""
    def __init__(self, dimension, scale=1000000, max_abs=100.0, modulus=DEFAULT_MODULUS, participants=20):
        if type(scale) is not int or scale <= 0 or not math.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("Encoding scale must be a positive integer and max_abs must be finite and positive")
        if dimension <= 0 or participants <= 0 or modulus <= 2 or modulus % 2 == 0:
            raise ValueError("Invalid dimension, population, or odd modulus")
        # float64 乘尺度后按 ties-to-even 舍入；仅小范围明文编码暂用 int64。
        # 映射为密文后始终使用 Python int，绝不让曲线阶数值进入 int64/浮点张量。
        if not math.isfinite(max_abs * scale) or max_abs * scale > 2 ** 52:
            raise ValueError("max_abs * scale must be <= 2**52; no automatic rescaling")
        self.dimension, self.scale, self.max_abs = dimension, scale, max_abs
        self.modulus, self.participants = modulus, participants
        self.coordinate_bound = math.ceil(max_abs * scale) + 1
        if participants * self.coordinate_bound >= modulus // 2:
            raise ValueError("Configured integer sum can wrap past the signed decoding interval")

    def manifest(self):
        return dict(scale=self.scale, max_abs=self.max_abs, modulus=str(self.modulus),
                    rounding="float64 multiply, nearest integer, ties to even (numpy.rint)",
                    signed_representation="z mod q; centered sum decoded before real division",
                    participants=self.participants, coordinate_bound=self.coordinate_bound)

    def encode(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise AggregationError("Encoding requires the complete finite model vector")
        if np.any(np.abs(vector) > self.max_abs):
            raise AggregationError("Model parameter exceeds configured max_abs; refusing to clip")
        return np.rint(vector * self.scale).astype(np.int64).tolist()

    def residues(self, signed):
        if len(signed) != self.dimension:
            raise AggregationError("Incorrect integer model dimension")
        return [value % self.modulus for value in signed]

    def decode_sum(self, residues, count):
        if count != self.participants or len(residues) != self.dimension:
            raise AggregationError("Incorrect sum dimension or effective participant count")
        signed = []
        for value in residues:
            if type(value) is not int or not 0 <= value < self.modulus:
                raise AggregationError("Noncanonical modular aggregate")
            value = value if value <= self.modulus // 2 else value - self.modulus
            if abs(value) > count * self.coordinate_bound:
                raise AggregationError("Aggregate outside the configured signed sum bound")
            signed.append(value)
        # 不能把有限域平均直接转浮点；负数与不能整除人数的和尤其会产生错误。
        return np.asarray([value / (self.scale * count) for value in signed], dtype=np.float64)


def load_protocol(path):
    path = Path(path).resolve()
    module_name = "trsfl_protocol_v2"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if Path(module.__file__).resolve() != path:
            raise AggregationError("Another protocol source is already loaded")
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AggregationError("Cannot load the configured TRS-FL source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass 处理类型注解时需要可查到所属模块。
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass
class AggregateResult:
    state: dict
    reference_state: dict
    diagnostics: dict


class ModelAggregator:
    """保留同批量化参照用于检查，不提供独立量化明文训练组。"""
    def __init__(self, template, args):
        self.mode = args.aggregation
        if self.mode not in ('float', 'trsfl'):
            raise AggregationError("Only float and trsfl training modes are supported")
        self.layout = ModelLayout(template)
        self.codec = FixedPointCodec(self.layout.dimension, args.encoding_scale,
                                    args.max_abs, args.modulus, args.num_users)
        self.binding = dict(layout=self.layout.manifest(), encoding=self.codec.manifest())
        self.layout_id = hashlib.sha256(json.dumps(self.binding, sort_keys=True).encode()).digest()
        self.protocol = None
        if self.mode == "trsfl":
            module = load_protocol(args.protocol_path)
            self.protocol = module.TRSFLProtocol(
                self.layout.dimension, self.layout_id, args.modulus, args.num_users,
                args.num_es, args.threshold, args.coefficient_bits, args.max_message_age,
                args.curve_id)

    def aggregate(self, states, round_id):
        if len(states) != self.codec.participants:
            raise AggregationError("Fresh local models from all configured clients are required")
        signed_sum = [0] * self.layout.dimension
        if self.protocol is not None:
            self.protocol.begin_round(round_id)
        for client, state in enumerate(states):
            # 输入来自 LocalUpdate 的完整本地模型；无随机替代、裁剪或补齐。
            encoded = self.codec.encode(self.layout.flatten(state))
            signed_sum = [a + b for a, b in zip(signed_sum, encoded)]
            if self.protocol is not None:
                self.protocol.submit(client, self.codec.residues(encoded))
        if any(abs(value) > self.codec.participants * self.codec.coordinate_bound for value in signed_sum):
            raise AggregationError("Exact integer sum exceeds configured bound")
        reference_sum = self.codec.residues(signed_sum)
        # 参照路径属于实验正确性检查，不能作为 AS 正常运行所需的明文侧信道。
        reference_vector = self.codec.decode_sum(reference_sum, len(states))
        reference_state = self.layout.restore(reference_vector)
        float_state = FedAvg(states)
        float_vector = self.layout.flatten(float_state)
        restored_reference = self.layout.flatten(reference_state)
        quantization_error = np.abs(restored_reference - float_vector)
        integer_equal = None
        if self.protocol is not None:
            recovered_sum = self.protocol.finalize()
            integer_equal = recovered_sum == reference_sum
            if not integer_equal:
                raise AggregationError("TRS-FL integer aggregate differs from the same-input plaintext reference")
            state = self.layout.restore(self.codec.decode_sum(recovered_sum, len(states)))
            if any(not torch.equal(state[key], reference_state[key]) for key in self.layout.keys):
                raise AggregationError("Restored TRS-FL model differs from the quantized reference")
        elif self.mode == "float":
            state = float_state
        else:
            raise AggregationError("Unknown aggregation mode")
        difference = np.abs(self.layout.flatten(state) - restored_reference)
        diagnostics = dict(
            participants=len(states), dimension=self.layout.dimension,
            integer_sum_equal=integer_equal,
            protocol_verified=True if self.protocol is not None else None,
            parameter_max_abs_diff=float(difference.max()),
            parameter_mean_abs_diff=float(difference.mean()),
            quantization_max_abs_diff=float(quantization_error.max()),
            quantization_mean_abs_diff=float(quantization_error.mean()),
            layout_id=self.layout_id.hex())
        return AggregateResult(state, reference_state, diagnostics)
