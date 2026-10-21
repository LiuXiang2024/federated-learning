"""Non-IID 划分与配对材料检查，不启动训练。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class NonIIDChecks(unittest.TestCase):
    def test_label_shards_are_balanced_complete_and_reproducible(self):
        from utils.sampling import label_shard_partition
        class Dataset:
            targets = np.repeat(np.arange(10), 6000)
            def __len__(self):
                return len(self.targets)
        dataset = Dataset()
        split = label_shard_partition(dataset, 50, 1)
        self.assertEqual(split, label_shard_partition(dataset, 50, 1))
        self.assertNotEqual(split, label_shard_partition(dataset, 50, 2))
        self.assertEqual(set(split), set(range(50)))
        self.assertEqual({len(x) for x in split.values()}, {1200})
        self.assertEqual(sorted(i for indices in split.values() for i in indices), list(range(60000)))
        self.assertTrue(all(len(np.unique(dataset.targets[indices])) <= 2 for indices in split.values()))
        with self.assertRaises(ValueError):
            label_shard_partition(range(601), 50, 1)

    def test_cli_selects_noniid_and_validation_accepts_it(self):
        from main_fed import validate_args
        from utils.options import args_parser
        with patch('sys.argv', ['main_fed.py', '--noniid']):
            args = args_parser()
        self.assertIs(args.iid, False)
        self.assertEqual(validate_args(args), [1])
        with patch('sys.argv', ['main_fed.py', '--iid']):
            self.assertIs(args_parser().iid, True)

    def test_pairing_saves_and_reuses_noniid_partition(self):
        import torch
        from utils.experiment import prepare_pairing
        class Dataset:
            targets = np.repeat(np.arange(10), 60)
            def __len__(self):
                return len(self.targets)
        config = dict(num_users=50, seed=1, iid=False)
        state = {'weight': torch.zeros(1)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            first, _ = prepare_pairing(temporary, Dataset(), state, config)
            second, _ = prepare_pairing(temporary, Dataset(), state, config)
            self.assertEqual(first, second)
            saved = json.loads((Path(temporary) / 'split.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['method'], 'label-sorted-shards-2')
            self.assertTrue(all(sum(n > 0 for n in counts) <= 2
                                for counts in saved['label_histograms'].values()))
            with self.assertRaises(ValueError):
                prepare_pairing(temporary, Dataset(), state, dict(config, iid=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
