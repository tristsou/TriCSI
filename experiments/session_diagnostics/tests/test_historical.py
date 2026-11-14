"""Software equivalence checks; synthetic fixtures are not research data."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.session_robustness.data import save_windows
from experiments.session_robustness.model import load_anchor
from experiments.session_robustness.study import ROOT, sha
from experiments.session_robustness.tests.test_study import fixture
from experiments.session_diagnostics.historical import SOURCE, run


class HistoricalTests(unittest.TestCase):
    def test_original_trainer_equivalence_and_scoring(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = {s: fixture(s, i+1, occupied_trials=6 if s=='S8' else 4 if s=='S9' else 1,
                             empty_trials=5) for i, s in enumerate(SOURCE+['S8','S9'])}
            for name, data in raw.items():
                save_windows(root/f'{name}.npz', data)
            hashes = {s: sha(root/f'{s}.npz') for s in raw}
            args = argparse.Namespace(project_root=ROOT, cache_dir=root, output=root/'run',
                epochs=1, seed=26082026, batch_size=16, lr=.001, threads=1)
            run(args)
            saved = torch.load(root/'run/checkpoints/historical_reconstruction.pt', weights_only=True)
            anchor = load_anchor(ROOT)
            source = anchor.combine([anchor.prepare_source_session(raw[s]) for s in SOURCE])
            direct, norm = anchor.train_model(source, epochs=1, batch_size=16, lr=.001, seed=args.seed)
            for key, tensor in direct.state_dict().items():
                self.assertTrue(torch.equal(tensor, saved['state_dict'][key]), key)
            for key in saved['norm']:
                np.testing.assert_array_equal(getattr(norm, key), saved['norm'][key])
            m = pd.read_csv(root/'run/reconstruction_metrics.csv')
            self.assertEqual(m.trials.tolist(), [40, 40])
            pred = pd.read_csv(root/'run/reconstruction_predictions.csv')
            self.assertFalse(pred.group.str.contains(r'\|EMPTY\|1\|').any())
            config = json.loads((root/'run/run_config.json').read_text())
            self.assertEqual(config['source_trials'], sum(len(set(raw[s].group)) for s in SOURCE))
            self.assertEqual(hashes, {s: sha(root/f'{s}.npz') for s in raw})
            with self.assertRaises(FileExistsError):
                run(args)

    def test_invalid_configuration_stops_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(project_root=ROOT, cache_dir=root, output=root/'run',
                epochs=0, seed=1, batch_size=16, lr=.001, threads=1)
            with self.assertRaises(ValueError):
                run(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
