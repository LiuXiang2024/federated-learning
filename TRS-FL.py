"""接收真实模型编码向量的 TRS-FL；TA/AS/ES/User 在单进程内模拟。

密文坐标使用 Python 任意精度整数模 q 运算，标量签名仍使用 Charm 的 ZR。
原随机向量演示和计时入口已移入 backups 原文件副本，不进入联邦训练流程。
"""
from dataclasses import dataclass, replace
import hashlib
import hmac
import math
import secrets
import time

from charm.toolbox.ecgroup import ECGroup, ZR, G
from Crypto.Cipher import ChaCha20
import sympy


class ProtocolError(RuntimeError):
    """出现此异常后不得提交本轮模型。"""


def pack(*parts):
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def uint(value):
    if type(value) is not int or value < 0:
        raise ProtocolError("Expected a nonnegative Python integer")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def sharing_parameters(n, threshold, secret_bits=256):
    """构造公开的 Asmuth–Bloom 模数；门限和 ES 数量彼此独立。"""
    if not 1 <= threshold <= n:
        raise ValueError("Invalid reconstruction threshold")
    m0 = int(sympy.nextprime(1 << secret_bits))
    moduli, current = [], 2 * m0
    for _ in range(n):
        current = int(sympy.nextprime(current))
        moduli.append(current)
    if math.prod(moduli[:threshold]) <= m0 * math.prod(moduli[n-threshold+1:]):
        raise ValueError("Asmuth-Bloom parameter inequality is not satisfied")
    return m0, tuple(moduli)


def share_secret(secret, n, threshold, parameters):
    # 公开模数可以跨轮复用；秘密和随机提升量每轮必须重新产生。
    m0, moduli = parameters
    if len(moduli) != n or not 0 <= secret < m0:
        raise ValueError("Invalid sharing parameters or secret")
    upper = math.prod(moduli[:threshold])
    max_v = (upper - 1 - secret) // m0
    beta = secret + secrets.randbelow(max_v + 1) * m0
    return tuple(beta % modulus for modulus in moduli)


def reconstruct_secret(pairs, m0, threshold):
    if len(pairs) < threshold or len({mod for _, mod in pairs}) != len(pairs):
        raise ProtocolError("Insufficient or duplicate secret shares")
    pairs = pairs[:threshold]
    product = math.prod(mod for _, mod in pairs)
    value = 0
    for share, modulus in pairs:
        if not 0 <= share < modulus:
            raise ProtocolError("Invalid secret share")
        partial = product // modulus
        value += share * partial * pow(partial, -1, modulus)
    return (value % product) % m0


@dataclass(frozen=True)
class Upload:
    """客户端上传：不包含组秘密、轮密钥、秘密份额或私钥。"""
    user_id: int
    es_id: int
    round_id: int
    PID: bytes
    Com: bytes
    Y: object
    timestamp: int
    C: tuple
    tau: bytes
    sigma: object


@dataclass(frozen=True)
class Batch:
    es_id: int
    round_id: int
    uploads: tuple
    C_sum: tuple
    digest: bytes


@dataclass(frozen=True)
class Challenge:
    round_id: int
    batch_digests: tuple
    coefficients: tuple


@dataclass(frozen=True)
class Response:
    es_id: int
    batch_digest: bytes
    sigma: object


class TA:
    """可信机构：系统参数、长期注册密钥及统一哈希/编码转录规则。"""
    def __init__(self, dimension, layout_id, curve_id=714, expected_modulus=None):
        if dimension <= 0 or not isinstance(layout_id, bytes):
            raise ValueError("A complete positive model layout is required")
        self.group = ECGroup(curve_id)
        self.q = int(self.group.order())
        if expected_modulus is not None and self.q != expected_modulus:
            raise ValueError("Configured encoding modulus differs from EC scalar order")
        self.width = (self.q.bit_length() + 7) // 8
        if (dimension + 4096) * self.width >= 1 << 38:
            raise ValueError("Model exceeds the ChaCha20 stream limit")
        self.dimension, self.layout_id = dimension, layout_id
        self.session_id = secrets.token_bytes(32)
        self.P = self.group.random(G)
        if self.P == self.P ** 0:
            raise ProtocolError("Identity generator")
        self.x = self.scalar()
        self.P_pub = self.P ** self.x
        self.salt = secrets.token_bytes(32)

    def scalar(self):
        # 密钥不使用 Python/NumPy/Torch 的训练随机源，训练种子不能复现协议秘密。
        return self.group.init(ZR, secrets.randbelow(self.q - 1) + 1)

    @staticmethod
    def kdf(material, info):
        prk = hmac.new(b"TRS-FL-v2", material, hashlib.sha256).digest()
        return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()

    def hash_to_zq(self, *parts):
        return self.group.hash(hashlib.sha256(pack(*parts)).digest(), ZR)

    def context(self, round_id):
        # layout_id 同时绑定参数名、形状、dtype、编码尺度、范围和模数。
        # session_id 独立随机生成，避免同一训练种子在不同运行中复用掩码上下文。
        return pack(b"TRS-FL-v2", self.session_id, uint(round_id),
                    self.layout_id, uint(self.dimension), uint(self.q))

    def H3(self, PID, X, R):
        return self.hash_to_zq(b"H3", PID, self.group.serialize(self.P_pub),
                              self.group.serialize(X), self.group.serialize(R))

    def H6(self, secret, PID, shared, round_id):
        return self.hash_to_zq(b"H6", self.context(round_id), uint(secret), PID,
                              self.group.serialize(shared))

    def check_vector(self, vector):
        if len(vector) != self.dimension:
            raise ProtocolError("Incorrect complete-model vector dimension")
        if any(type(value) is not int or not 0 <= value < self.q for value in vector):
            raise ProtocolError("Coordinates must be canonical modular integers")

    def cipher_digest(self, vector):
        # 分块仅用于控制序列化内存；哈希覆盖完整向量，不截断至 64 维。
        self.check_vector(vector)
        digest = hashlib.sha256(uint(self.dimension))
        for start in range(0, self.dimension, 4096):
            digest.update(b"".join(value.to_bytes(self.width, "big")
                                   for value in vector[start:start+4096]))
        return digest.digest()

    def metadata(self, upload):
        return pack(self.context(upload.round_id), uint(upload.user_id),
                    uint(upload.es_id), upload.PID, upload.Com,
                    self.group.serialize(upload.Y), uint(upload.timestamp))

    def H(self, upload, shared):
        key = self.kdf(self.group.serialize(shared), b"ES-authentication")
        return hmac.new(key, pack(self.metadata(upload), self.cipher_digest(upload.C)),
                        hashlib.sha256).digest()

    def H7(self, upload, public_key):
        X, R = public_key
        return self.hash_to_zq(b"H7", self.metadata(upload), self.cipher_digest(upload.C),
                              self.group.serialize(X), self.group.serialize(R), upload.tau)

    def mask_vector(self, key_scalar, PID, Y, round_id):
        """双方按相同会话/轮次/布局重现完整维度的 ChaCha20 掩码。"""
        context = pack(self.context(round_id), PID, self.group.serialize(Y))
        material = int(key_scalar).to_bytes(self.width, "big")
        key = self.kdf(material, pack(b"model-mask", context))
        nonce = hashlib.sha256(pack(b"nonce", context)).digest()[:12]
        cipher = ChaCha20.new(key=key, nonce=nonce)
        # 拒绝采样消除直接取模的分布偏差；不会改变真实模型的长度或取值范围。
        ceiling = 1 << (8 * self.width)
        limit = ceiling - ceiling % self.q
        result = []
        while len(result) < self.dimension:
            count = min(self.dimension - len(result) + 16, 4096)
            stream = cipher.encrypt(bytes(count * self.width))
            for offset in range(0, len(stream), self.width):
                value = int.from_bytes(stream[offset:offset+self.width], "big")
                if value < limit:
                    result.append(value % self.q)
                    if len(result) == self.dimension:
                        break
        return result

    def batch_digest(self, batch):
        digest = hashlib.sha256(pack(b"batch", self.context(batch.round_id), uint(batch.es_id)))
        for upload in batch.uploads:
            digest.update(pack(self.metadata(upload), self.cipher_digest(upload.C), upload.tau,
                               self.group.serialize(upload.sigma)))
        digest.update(self.cipher_digest(batch.C_sum))
        return digest.digest()


class User:
    """客户端：持有长期私钥，只接收外部真实 SGD 输出的整数编码，不负责训练。"""
    def __init__(self, ta, user_id, es_id):
        self.ta, self.id, self.es_id = ta, user_id, es_id
        self.puf = secrets.token_bytes(32)  # 原型中的软件注册秘密，不代表实体 PUF。
        self.Com = ta.kdf(self.puf, ta.salt)
        self.x = ta.scalar()
        self.X = ta.P ** self.x
        self.PID = ta.kdf(pack(self.Com, ta.group.serialize(self.X),
                               ta.group.serialize(ta.P_pub)), b"registration-PID")
        r = ta.scalar()
        self.R = ta.P ** r
        self.d = ta.x * ta.H3(self.PID, self.X, self.R) + r
        self.round_id, self.signed, self.group_seed = 0, True, None

    def reconstruct_seed(self, own_pair, other_pairs, m0, threshold):
        self.group_seed = reconstruct_secret([own_pair] + list(other_pairs), m0, threshold)

    def prepare_round(self, round_id, pk_AS):
        # 长期身份与密钥保留；动态假名、临时密钥和轮密钥每轮更新。
        if round_id <= self.round_id or self.group_seed is None:
            raise ProtocolError("Round reuse or missing reconstructed secret")
        self.round_id = round_id
        self.Com_t = self.ta.kdf(secrets.token_bytes(32), pack(b"round-commitment", self.Com))
        self.PID_t = self.ta.kdf(pack(self.PID, self.ta.context(round_id), self.Com_t), b"PID")
        self.y = self.ta.scalar()
        self.Y = self.X ** self.y
        self.k = self.ta.H6(self.group_seed, self.PID_t, pk_AS ** (self.x * self.y), round_id)
        self.signed = False

    def signcrypt(self, encoded_vector, pk_ES):
        # 同一轮密钥只允许签密一次，避免 ChaCha20 的 key/nonce 复用。
        if self.signed:
            raise ProtocolError("A round key may signcrypt exactly one model")
        self.ta.check_vector(encoded_vector)
        self.signed = True
        mask = self.ta.mask_vector(self.k, self.PID_t, self.Y, self.round_id)
        C = tuple((value + noise) % self.ta.q for value, noise in zip(encoded_vector, mask))
        upload = Upload(self.id, self.es_id, self.round_id, self.PID_t, self.Com_t,
                        self.Y, int(time.time()), C, b"", None)
        upload = replace(upload, tau=self.ta.H(upload, pk_ES ** (self.x * self.y)))
        xi = self.ta.H7(upload, (self.X, self.R))
        # 保留原签密代数：sigma = k + (x*y + d)*xi，xi 绑定完整密文和元数据。
        return replace(upload, sigma=self.k + (self.x * self.y + self.d) * xi)


class ES:
    """边缘服务器：时间戳/MAC 检查、固定全员批次、按 AS 系数加权签名。"""
    def __init__(self, ta, es_id, member_ids, max_message_age):
        self.ta, self.id = ta, es_id
        self.member_ids = frozenset(member_ids)
        self.max_message_age = max_message_age
        self.sk = ta.scalar()
        self.pk = ta.P ** self.sk
        self.round_id, self.batch, self.pending = 0, None, {}

    def begin_round(self, round_id):
        if round_id <= self.round_id:
            raise ProtocolError("ES round replay")
        self.round_id, self.batch, self.pending = round_id, None, {}

    def receive(self, upload):
        if self.batch is not None or upload.round_id != self.round_id:
            raise ProtocolError("ES batch already frozen or wrong round")
        if (upload.es_id != self.id or upload.user_id not in self.member_ids
                or upload.user_id in self.pending):
            raise ProtocolError("Unexpected or duplicate client upload")
        age = int(time.time()) - upload.timestamp
        if not 0 <= age <= self.max_message_age:
            raise ProtocolError("ES timestamp check failed")
        if upload.Y == self.ta.P ** 0:
            raise ProtocolError("Invalid ephemeral public key")
        if not hmac.compare_digest(upload.tau, self.ta.H(upload, upload.Y ** self.sk)):
            raise ProtocolError("ES authentication tag check failed")
        # 检查失败直接抛错；只有成功认证的上传才能进入聚合集合。
        self.pending[upload.user_id] = upload

    def edge_verify_aggregate(self):
        if self.batch is not None or set(self.pending) != self.member_ids:
            raise ProtocolError("ES requires the complete fresh client set before freezing")
        uploads = tuple(self.pending[client] for client in sorted(self.pending))
        total = [0] * self.ta.dimension
        for upload in uploads:
            total = [(a + b) % self.ta.q for a, b in zip(total, upload.C)]
        batch = Batch(self.id, self.round_id, uploads, tuple(total), b"")
        self.batch = replace(batch, digest=self.ta.batch_digest(batch))
        self.pending = {}
        return self.batch

    def weighted_response(self, challenge):
        if (self.batch is None or challenge.round_id != self.round_id
                or dict(challenge.batch_digests).get(self.id) != self.batch.digest):
            raise ProtocolError("Challenge does not refer to this frozen ES batch")
        coefficients = dict(challenge.coefficients)
        sigma = self.ta.group.init(ZR, 0)
        for upload in self.batch.uploads:
            sigma += self.ta.group.init(ZR, coefficients[upload.user_id]) * upload.sigma
        return Response(self.id, self.batch.digest, sigma)


class AS:
    """聚合服务器：注册信息与组秘密自持，固定批次后挑战，验证后才能去掩码。"""
    def __init__(self, ta, users, threshold, coefficient_bits, max_message_age):
        self.ta = ta
        self.sk = ta.scalar()
        self.pk = ta.P ** self.sk
        self.threshold, self.coefficient_bits = threshold, coefficient_bits
        self.max_message_age = max_message_age
        if not 1 <= coefficient_bits < ta.q.bit_length():
            raise ValueError("Batch coefficient bit length must be below scalar order bit length")
        # 从可信注册状态取公钥和身份；上传字段不包含用户对象、私钥或组秘密 alpha。
        self.registry = {u.id: (u.PID, u.X, u.R, u.es_id) for u in users}
        self.group_seeds, self.round_id, self.challenge = {}, 0, None
        self.es_ids = {entry[3] for entry in self.registry.values()}

    @staticmethod
    def group_users_by_es(users, num_es):
        if num_es <= 0 or not users or len(users) % num_es:
            raise ValueError("Clients must divide evenly among positive ES count")
        group_size = len(users) // num_es
        return [users[start:start+group_size] for start in range(0, len(users), group_size)]

    def setup_group_secrets(self, groups, parameters, round_id):
        if round_id <= self.round_id:
            raise ProtocolError("AS round replay")
        self.round_id, self.challenge, self.group_seeds = round_id, None, {}
        for es_id, users in enumerate(groups):
            secret = secrets.randbits(256)
            m0, moduli = parameters
            shares = share_secret(secret, len(users), self.threshold, parameters)
            # AS 保留自己生成的秘密，后续验签与去掩码不依赖客户端回传秘密。
            self.group_seeds[es_id] = secret
            for position, user in enumerate(users):
                others = [(shares[index], moduli[index]) for index in range(len(users))
                          if index != position][:self.threshold-1]
                user.reconstruct_seed((shares[position], moduli[position]), others,
                                      m0, self.threshold)
                # 仅用于软件模拟的重构一致性检查，不形成秘密上传消息。
                if user.group_seed != secret:
                    raise ProtocolError("Group secret reconstruction failed")
                user.prepare_round(round_id, self.pk)

    def issue_challenge(self, batches):
        # 先校验并固定全部 ES 批次（含每用户签名和聚合密文），再生成独立随机系数。
        if (self.challenge is not None or len(batches) != len(self.es_ids)
                or {batch.es_id for batch in batches} != self.es_ids):
            raise ProtocolError("AS requires one new frozen batch from every registered ES")
        accepted, now = set(), int(time.time())
        for batch in batches:
            if batch.round_id != self.round_id or batch.digest != self.ta.batch_digest(batch):
                raise ProtocolError("Frozen batch digest or round mismatch")
            expected = {c for c, entry in self.registry.items() if entry[3] == batch.es_id}
            ids = [upload.user_id for upload in batch.uploads]
            if len(ids) != len(expected) or set(ids) != expected:
                raise ProtocolError("Incomplete or duplicate batch membership")
            total = [0] * self.ta.dimension
            for upload in batch.uploads:
                if (upload.user_id in accepted or upload.es_id != batch.es_id
                        or upload.round_id != self.round_id):
                    raise ProtocolError("Duplicate client or mismatched upload context")
                if not 0 <= now - upload.timestamp <= self.max_message_age:
                    raise ProtocolError("AS timestamp check failed; check the explicit freshness window")
                if upload.Y == self.ta.P ** 0:
                    raise ProtocolError("Identity ephemeral public key")
                self.ta.check_vector(upload.C)
                accepted.add(upload.user_id)
                total = [(a + b) % self.ta.q for a, b in zip(total, upload.C)]
            if tuple(total) != batch.C_sum:
                raise ProtocolError("ES aggregate differs from individual ciphertext sum")
        if accepted != set(self.registry):
            raise ProtocolError("AS requires all registered clients")
        coefficients = tuple((client, secrets.randbelow((1 << self.coefficient_bits) - 1) + 1)
                             for client in sorted(accepted))
        self.challenge = Challenge(self.round_id, tuple((b.es_id, b.digest) for b in batches), coefficients)
        return self.challenge

    def unsigncrypt_aggregate(self, batches, challenge, responses):
        # 按同一系数验证 P^(sum a*sigma) = P^(sum a*k) * product(Y*R*Ppub^h)^(a*xi)。
        # 随机加权仅用于验签；模型密文和去掩码结果仍是不加权的普通求和。
        if challenge is not self.challenge or challenge is None:
            raise ProtocolError("Unknown, consumed, or replayed AS challenge")
        self.challenge = None
        expected_digests = dict(challenge.batch_digests)
        if (len(batches) != len(self.es_ids) or len(responses) != len(self.es_ids)
                or {r.es_id for r in responses} != self.es_ids
                or {b.es_id for b in batches} != self.es_ids):
            raise ProtocolError("Missing or duplicate ES response")
        coefficients = dict(challenge.coefficients)
        sigma, k_total = self.ta.group.init(ZR, 0), self.ta.group.init(ZR, 0)
        right, recovery = self.ta.P ** 0, []
        for batch in batches:
            if (batch.digest != expected_digests.get(batch.es_id)
                    or self.ta.batch_digest(batch) != batch.digest):
                raise ProtocolError("ES batch changed after challenge generation")
            response = next(r for r in responses if r.es_id == batch.es_id)
            if response.batch_digest != batch.digest:
                raise ProtocolError("Response belongs to another batch")
            sigma += response.sigma
            for upload in batch.uploads:
                PID, X, R, es_id = self.registry[upload.user_id]
                # 组秘密取自 AS 本轮本地状态；PID/Y 是上传的公开派生上下文。
                key = self.ta.H6(self.group_seeds[es_id], upload.PID,
                                 upload.Y ** self.sk, upload.round_id)
                coefficient = self.ta.group.init(ZR, coefficients[upload.user_id])
                xi, h = self.ta.H7(upload, (X, R)), self.ta.H3(PID, X, R)
                k_total += coefficient * key
                right *= (upload.Y * R * (self.ta.P_pub ** h)) ** (coefficient * xi)
                recovery.append((upload, key))
        if self.ta.P ** sigma != (self.ta.P ** k_total) * right:
            raise ProtocolError("AS randomized aggregate signature check failed")
        # 验签失败已抛出异常；本段不会产生可被误提交的失败模型。
        total = [0] * self.ta.dimension
        for batch in batches:
            total = [(a + b) % self.ta.q for a, b in zip(total, batch.C_sum)]
        for upload, key in recovery:
            mask = self.ta.mask_vector(key, upload.PID, upload.Y, upload.round_id)
            total = [(a - b) % self.ta.q for a, b in zip(total, mask)]
        return total  # 返回模 q 整数和；实数平均交给模型适配层完成。


class TRSFLProtocol:
    """对训练层的接口：begin_round → num_clients 次 submit → finalize；失败即终止。"""
    def __init__(self, dimension, layout_id, modulus, num_clients=20, num_es=2,
                 threshold=3, coefficient_bits=128, max_message_age=3600, curve_id=714):
        if num_clients <= 0 or num_es <= 0 or num_clients % num_es:
            raise ValueError("Clients must divide evenly among positive ES count")
        group_size = num_clients // num_es
        if not 1 <= threshold <= group_size:
            raise ValueError("Threshold must be between 1 and the ES group size")
        if max_message_age <= 0:
            raise ValueError("Freshness window must be positive")
        # 人数由配置传入，分组、门限与全员验证使用同一拓扑。
        self.num_clients = num_clients
        self.ta = TA(dimension, layout_id, curve_id, modulus)
        self.users = [User(self.ta, client, client // group_size) for client in range(num_clients)]
        self.as_server = AS(self.ta, self.users, threshold, coefficient_bits, max_message_age)
        self.groups = self.as_server.group_users_by_es(self.users, num_es)
        self.parameters = sharing_parameters(group_size, threshold)
        self.edges = [ES(self.ta, index, [u.id for u in group], max_message_age)
                      for index, group in enumerate(self.groups)]
        self.round_id, self.state, self.submitted = 0, "ready", set()

    def begin_round(self, round_id):
        if self.state not in ("ready", "complete") or round_id <= self.round_id:
            raise ProtocolError("Invalid protocol state or round reuse")
        self.state = "failed"
        self.as_server.setup_group_secrets(self.groups, self.parameters, round_id)
        for edge in self.edges:
            edge.begin_round(round_id)
        self.round_id, self.submitted, self.state = round_id, set(), "collecting"

    def submit(self, client_id, encoded_vector):
        if self.state != "collecting" or client_id not in range(self.num_clients) or client_id in self.submitted:
            self.state = "failed"
            raise ProtocolError("Unexpected or duplicate protocol submission")
        try:
            user = self.users[client_id]
            edge = self.edges[user.es_id]
            edge.receive(user.signcrypt(encoded_vector, edge.pk))
            self.submitted.add(client_id)
        except Exception:
            self.state = "failed"
            raise

    def finalize(self):
        if self.state != "collecting" or self.submitted != set(range(self.num_clients)):
            self.state = "failed"
            raise ProtocolError("Cannot finalize without all configured clients")
        self.state = "failed"
        batches = tuple(edge.edge_verify_aggregate() for edge in self.edges)
        challenge = self.as_server.issue_challenge(batches)
        responses = tuple(edge.weighted_response(challenge) for edge in self.edges)
        total = self.as_server.unsigncrypt_aggregate(batches, challenge, responses)
        # 大模型密文无需跨轮保留，及时释放以减少下一轮本地训练的内存占用。
        for edge in self.edges:
            edge.batch = None
        self.state = "complete"
        return total


if __name__ == "__main__":
    raise SystemExit("Use main_fed.py; no synthetic model input is generated here.")
