"""仅供服务器手动运行的接口验证，不是训练实验，也不输出准确率结论。

这里使用可核算的确定性整数/小模型状态验证边界；正式入口始终上传真实 SGD 输出。
codec 用例无需密码学依赖；protocol 用例需要 Charm EC 与 PyCryptodome。
"""
import argparse
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.Nets import CNNMnist
from models.secure_aggregation import (AggregationError, DEFAULT_MODULUS, FixedPointCodec,
                                       ModelLayout, ModelAggregator, load_protocol)
from utils.sampling import balanced_partition


def arguments(mode):
    return SimpleNamespace(aggregation=mode, encoding_scale=1000000, max_abs=100.0,
                           modulus=DEFAULT_MODULUS, num_users=20, num_es=2,
                           threshold=3, coefficient_bits=128, max_message_age=3600,
                           curve_id=714, protocol_path=Path(__file__).resolve().parent.parent / 'TRS-FL.py')


class CodecChecks(unittest.TestCase):
    def test_default_experiment_is_cnn_iid(self):
        from utils.options import args_parser
        with patch('sys.argv', ['main_fed.py']):
            args = args_parser()
        self.assertEqual(args.model, 'cnn')
        self.assertIs(args.iid, True)

    def test_repeat_seeds_and_boundary_validation(self):
        from main_fed import validate_args
        args = SimpleNamespace(num_users=20, num_es=2, frac=1.0, threshold=3,
                               epochs=100, local_ep=3, local_bs=32, bs=128,
                               lr=0.01, momentum=0.5, dataset='mnist', model='cnn', iid=True,
                               num_classes=10, num_channels=1, target_accuracy=None,
                               seed=1, repeats=1)
        self.assertEqual(validate_args(args), [1])
        args.iid = False
        with self.assertRaises(ValueError):
            validate_args(args)
        args.iid = True
        args.model = 'unsupported'
        with self.assertRaises(ValueError):
            validate_args(args)
        args.model = 'cnn'
        args.repeats = 0
        with self.assertRaises(ValueError):
            validate_args(args)
        args.seed, args.repeats = 2**32 - 1, 2
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_layout_uses_complete_model_and_roundtrips(self):
        state = CNNMnist(SimpleNamespace(num_channels=1, num_classes=10)).state_dict()
        layout = ModelLayout(state)
        self.assertGreater(layout.dimension, 64)
        restored = layout.restore(layout.flatten(state))
        self.assertEqual(set(restored), set(state))
        for name in state:
            self.assertTrue(torch.equal(restored[name], state[name]))

    def test_signed_sum_then_real_average(self):
        codec = FixedPointCodec(1, scale=10, max_abs=10)
        # 10 个 +0.1 和 10 个 -0.2；整数和 -10，实数平均 -0.05。
        value = codec.decode_sum([(-10) % codec.modulus], 20)
        self.assertAlmostEqual(value[0], -0.05)

    def test_rounding_ties_to_even_and_negative_residues(self):
        codec = FixedPointCodec(4, scale=2)
        signed = codec.encode([0.25, 0.75, -0.25, -0.75])
        self.assertEqual(signed, [0, 2, 0, -2])
        self.assertEqual(codec.residues(signed)[-1], codec.modulus - 2)

    def test_rejects_nonfinite_out_of_range_and_wrong_dimension(self):
        codec = FixedPointCodec(2, max_abs=1)
        for vector in ([0, np.nan], [0, np.inf], [0, 1.01], [0]):
            with self.subTest(vector=vector), self.assertRaises(AggregationError):
                codec.encode(vector)

    def test_rejects_wrap_and_unsafe_scaling(self):
        with self.assertRaises(ValueError):
            FixedPointCodec(1, scale=10, max_abs=10, modulus=101)
        with self.assertRaises(ValueError):
            FixedPointCodec(1, scale=2**53, max_abs=1)

    def test_rejects_layout_mismatch_and_integer_buffers(self):
        layout = ModelLayout({'weight': torch.zeros(65)})
        with self.assertRaises(AggregationError):
            layout.flatten({'weight': torch.zeros(64)})
        with self.assertRaises(AggregationError):
            ModelLayout({'counter': torch.tensor(1)})

    def test_internal_quantized_reference_averages_20_complete_states(self):
        template = {'weight': torch.zeros(65)}
        states = [{'weight': torch.full((65,), 0.1 if client < 10 else -0.2)} for client in range(20)]
        result = ModelAggregator(template, arguments('float')).aggregate(states, 1)
        self.assertTrue(torch.equal(result.reference_state['weight'], torch.full((65,), -0.05)))
        self.assertIsNone(result.diagnostics['protocol_verified'])

    def test_iid_partition_covers_balanced_data(self):
        class Dataset:
            targets = np.repeat(np.arange(10), 60)

            def __len__(self):
                return len(self.targets)
        partition = balanced_partition(Dataset(), 20, 7)
        self.assertEqual(partition, balanced_partition(Dataset(), 20, 7))
        self.assertEqual({len(indices) for indices in partition.values()}, {30})
        self.assertEqual(sorted(index for indices in partition.values() for index in indices), list(range(600)))
        self.assertNotEqual(partition, balanced_partition(Dataset(), 20, 8))
        self.assertTrue(all(len(np.unique(Dataset.targets[indices])) > 2
                            for indices in partition.values()))

    def test_iid_partition_rejects_uneven_sample_count(self):
        with self.assertRaises(ValueError):
            balanced_partition(range(601), 20, 7)


class ProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_protocol(arguments('trsfl').protocol_path)

    def new_protocol(self):
        return self.module.TRSFLProtocol(65, b'server-interface-check-layout', DEFAULT_MODULUS)

    def populate(self, protocol):
        protocol.begin_round(1)
        for client in range(20):
            signed = client - 10
            protocol.submit(client, [signed % DEFAULT_MODULUS] * 65)

    def test_full_path_and_second_round(self):
        protocol = self.new_protocol()
        self.populate(protocol)
        expected = [sum(range(-10, 10)) % DEFAULT_MODULUS] * 65
        self.assertEqual(protocol.finalize(), expected)
        old_pid = protocol.users[0].PID_t
        protocol.begin_round(2)
        self.assertNotEqual(protocol.users[0].PID_t, old_pid)
        for client in range(20):
            protocol.submit(client, [1] * 65)
        self.assertEqual(protocol.finalize(), [20] * 65)

    def test_as_uses_its_own_round_secrets_without_client_secret_upload(self):
        protocol = self.new_protocol()
        self.populate(protocol)
        # 上传只携带公开上下文、密文和认证信息，不携带组秘密、轮密钥或份额。
        self.assertEqual({field.name for field in fields(self.module.Upload)},
                         {'user_id', 'es_id', 'round_id', 'PID', 'Com', 'Y',
                          'timestamp', 'C', 'tau', 'sigma'})
        for user in protocol.users:
            self.assertEqual(user.group_seed, protocol.as_server.group_seeds[user.es_id])
            # 上传完成后清除客户端秘密；AS 必须仍能独立验签和去掩码。
            user.group_seed = user.k = user.x = user.y = user.d = None
        protocol.users.clear()
        protocol.groups.clear()
        expected = [sum(range(-10, 10)) % DEFAULT_MODULUS] * 65
        self.assertEqual(protocol.finalize(), expected)

    def test_es_rejects_tampering_at_last_coordinate(self):
        protocol = self.new_protocol()
        protocol.begin_round(1)
        user, edge = protocol.users[0], protocol.edges[0]
        upload = user.signcrypt([0] * 65, edge.pk)
        changed = list(upload.C)
        changed[-1] = (changed[-1] + 1) % DEFAULT_MODULUS
        with self.assertRaises(self.module.ProtocolError):
            edge.receive(replace(upload, C=tuple(changed)))
        self.assertEqual(edge.pending, {})

    def test_as_signature_failure_never_returns_a_sum(self):
        protocol = self.new_protocol()
        self.populate(protocol)
        edge = protocol.edges[0]
        upload = edge.pending[0]
        edge.pending[0] = replace(upload, sigma=upload.sigma + protocol.ta.group.init(self.module.ZR, 1))
        with self.assertRaises(self.module.ProtocolError):
            protocol.finalize()
        self.assertEqual(protocol.state, 'failed')

    def test_as_checks_individual_ciphertext_sum(self):
        protocol = self.new_protocol()
        self.populate(protocol)
        batches = [edge.edge_verify_aggregate() for edge in protocol.edges]
        changed = list(batches[0].C_sum)
        changed[-1] = (changed[-1] + 1) % DEFAULT_MODULUS
        batch = replace(batches[0], C_sum=tuple(changed))
        batches[0] = replace(batch, digest=protocol.ta.batch_digest(batch))
        with self.assertRaises(self.module.ProtocolError):
            protocol.as_server.issue_challenge(tuple(batches))

    def test_freeze_precedes_challenge_and_changed_batches_are_rejected(self):
        protocol = self.new_protocol()
        self.populate(protocol)
        batches = tuple(edge.edge_verify_aggregate() for edge in protocol.edges)
        challenge = protocol.as_server.issue_challenge(batches)
        self.assertEqual(len(challenge.coefficients), 20)
        self.assertTrue(all(0 < coefficient < 2**128 for _, coefficient in challenge.coefficients))
        responses = tuple(edge.weighted_response(challenge) for edge in protocol.edges)
        changed = replace(batches[0], digest=b'changed')
        with self.assertRaises(self.module.ProtocolError):
            protocol.as_server.unsigncrypt_aggregate((changed,) + batches[1:], challenge, responses)

    def test_no_key_reuse_and_no_silent_missing_participant(self):
        protocol = self.new_protocol()
        protocol.begin_round(1)
        protocol.submit(0, [0] * 65)
        with self.assertRaises(self.module.ProtocolError):
            protocol.users[0].signcrypt([1] * 65, protocol.edges[0].pk)
        with self.assertRaises(self.module.ProtocolError):
            protocol.finalize()

    def test_distinct_shares_are_required(self):
        with self.assertRaises(self.module.ProtocolError):
            self.module.reconstruct_secret([(1, 7), (1, 7), (2, 11)], 5, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['codec', 'protocol', 'all'], default='codec')
    selected = parser.parse_args().suite
    suite = unittest.TestSuite()
    for case, name in ((CodecChecks, 'codec'), (ProtocolChecks, 'protocol')):
        if selected in (name, 'all'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
