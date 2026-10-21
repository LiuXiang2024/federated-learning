"""并行训练的调度与小样本回归检查；不运行正式训练。"""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.Nets import CNNMnist
from models.Update import LocalUpdate
from utils.datasets import load_datasets
from utils.experiment import derived_seed, resolve_device, set_training_seed
from utils.options import args_parser


class ParallelTrainingChecks(unittest.TestCase):
    def test_parallel_default_and_shared_data_path(self):
        with patch('sys.argv', ['main_fed.py']):
            args = args_parser()
        self.assertEqual(getattr(args, 'client_workers', None), 10)
        self.assertEqual(args.num_users, 50)
        expected = Path(__file__).resolve().parent / 'data'
        self.assertEqual(args.data_dir, expected)
        self.assertEqual(args.protocol_path, Path(__file__).resolve().parent / 'TRS-FL.py')

    def test_fifty_clients_partition_and_group_threshold(self):
        from main_fed import validate_args
        from utils.sampling import balanced_partition
        with patch('sys.argv', ['main_fed.py', '--num_users', '50']):
            args = args_parser()
        self.assertEqual(validate_args(args), [1])
        partition = balanced_partition(range(60000), args.num_users, args.seed)
        self.assertEqual(len(partition), 50)
        self.assertEqual({len(indices) for indices in partition.values()}, {1200})
        self.assertEqual(len({index for indices in partition.values() for index in indices}), 60000)
        args.threshold = 25
        self.assertEqual(validate_args(args), [1])
        args.threshold = 26
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_concurrency_cannot_exceed_ten(self):
        from main_fed import validate_args
        from utils.client_training import ClientTrainer
        with patch('sys.argv', ['main_fed.py']):
            args = args_parser()
        args.client_workers = 11
        with self.assertRaises(ValueError):
            validate_args(args)
        with self.assertRaises(ValueError):
            ClientTrainer(args)

    def test_results_keep_client_order_when_completed_out_of_order(self):
        from utils.client_training import collect_results
        first, second = Future(), Future()
        second.set_result((1, {'weight': torch.tensor([2.0])}, 0.2))
        completion_order = []

        def complete_first(client_id):
            completion_order.append(client_id)
            if client_id == 1:
                first.set_result((0, {'weight': torch.tensor([1.0])}, 0.1))

        states, losses = collect_results({second: 1, first: 0}, 2, complete_first)
        self.assertEqual(completion_order, [1, 0])
        self.assertEqual([state['weight'].item() for state in states], [1.0, 2.0])
        self.assertEqual(losses, [0.1, 0.2])

    def test_worker_failure_retains_client_id_and_cancels_pending(self):
        from utils.client_training import ClientTrainingError, collect_results
        failed, pending = Future(), Future()
        failed.set_exception(RuntimeError('local training failed'))
        with self.assertRaises(ClientTrainingError) as error:
            collect_results({failed: 0, pending: 1}, 2)
        self.assertEqual(error.exception.client_id, 0)
        self.assertTrue(pending.cancelled())

    def test_wrong_client_result_is_rejected(self):
        from utils.client_training import ClientTrainingError, collect_results
        future = Future()
        future.set_result((1, {'weight': torch.zeros(1)}, 0.1))
        with self.assertRaises(ClientTrainingError):
            collect_results({future: 0}, 1)

    def test_failed_client_does_not_aggregate_or_commit(self):
        import main_fed
        from utils.client_training import ClientTrainingError
        with patch('sys.argv', ['main_fed.py', '--device', 'cpu', '--aggregation', 'float']):
            args = args_parser()
        args.device = resolve_device(args)
        model = CNNMnist(args)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary) / 'float'
            with patch('main_fed.ClientTrainer') as trainer, \
                    patch('main_fed.ModelAggregator.aggregate') as aggregate:
                trainer.return_value.train_round.side_effect = ClientTrainingError(7, 'test failure')
                with self.assertRaises(ClientTrainingError):
                    main_fed.run_mode(args, 1, model, initial, {}, None, None, directory, {})
                aggregate.assert_not_called()
                trainer.return_value.close.assert_called_once()
            failure = json.loads((directory / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['client_id'], 7)
            self.assertEqual(failure['phase'], 'local_training')
            self.assertEqual(failure['committed_round'], 0)
            checkpoint = torch.load(directory / 'last_good_model.pt', weights_only=True)
            self.assertTrue(all(torch.equal(initial[key], checkpoint[key]) for key in initial))

    def test_real_spawn_two_rounds_match_serial_local_training(self):
        from utils.client_training import ClientTrainer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            data = Path(temporary)
            raw = data / 'mnist' / 'raw'
            raw.mkdir(parents=True)
            rng = np.random.default_rng(7)
            for prefix, count in (('train', 100), ('t10k', 10)):
                images = rng.integers(0, 256, size=(count, 28, 28), dtype=np.uint8)
                labels = (np.arange(count) % 10).astype(np.uint8)
                (raw / (prefix + '-images-idx3-ubyte')).write_bytes(
                    struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes())
                (raw / (prefix + '-labels-idx1-ubyte')).write_bytes(
                    struct.pack('>II', 2049, count) + labels.tobytes())
            with patch('sys.argv', ['main_fed.py', '--device', 'cpu']):
                args = args_parser()
            args.data_dir, args.client_workers = data, 2
            args.local_ep, args.local_bs = 1, 2
            args.device = resolve_device(args)
            train, _, _ = load_datasets(args)
            partition = {client: [client * 2, client * 2 + 1] for client in range(args.num_users)}
            set_training_seed(derived_seed(1, 0, 0))
            model = CNNMnist(args)
            initial = {key: value.clone() for key, value in model.state_dict().items()}
            trainer = ClientTrainer(args)
            try:
                worker_pids = None
                for round_id in (1, 2):
                    actual, losses = trainer.train_round(initial, partition, 1, round_id)
                    current_pids = {process.pid for process in trainer.executor._processes.values()}
                    self.assertEqual(len(current_pids), 2)
                    if worker_pids is not None:
                        self.assertEqual(current_pids, worker_pids)
                    worker_pids = current_pids
                    self.assertEqual(len(actual), 50)
                    for client in range(args.num_users):
                        seed = derived_seed(1, round_id, client)
                        set_training_seed(seed)
                        local_model = copy.deepcopy(model)
                        local_model.load_state_dict(initial)
                        expected, loss = LocalUpdate(args, train, partition[client], seed).train(local_model)
                        self.assertEqual(losses[client], loss)
                        for key in expected:
                            self.assertTrue(torch.equal(actual[client][key], expected[key]), (round_id, client, key))
                    initial = {key: torch.stack([state[key] for state in actual]).mean(0)
                               for key in initial}
            finally:
                trainer.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
