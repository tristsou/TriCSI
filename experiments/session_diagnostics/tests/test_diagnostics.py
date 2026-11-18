"""Synthetic software fixtures only: none of these accuracies are research data."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from experiments.session_robustness.tests.test_study import fixture
from experiments.session_robustness.data import save_windows
from experiments.session_robustness.study import ROOT, develop, sha
from experiments.session_diagnostics.core import prepare_arm, prediction_reports
from experiments.session_diagnostics.__main__ import audit, compare


class DiagnosticTests(unittest.TestCase):
    def test_relative_shape_removes_static_spectral_background_without_mutation(self):
        raw = fixture('A', 1)
        original = raw.X.copy()
        changed = fixture('A', 1)
        background = np.sin(np.arange(64) / 3)[None, None, :]
        changed.X[:, [0, 2]] += background
        absolute = changed.X[:, [0, 2]]
        changed.X[:, [1, 3]] = absolute - np.median(absolute, axis=-1, keepdims=True)
        legacy, _ = prepare_arm(raw)
        legacy_shifted, _ = prepare_arm(changed)
        relative, _ = prepare_arm(raw, 'relative_shape')
        shifted, _ = prepare_arm(changed, 'relative_shape')
        np.testing.assert_allclose(relative.X, shifted.X, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(legacy.X[:, [1,3]] - legacy_shifted.X[:, [1,3]]))), .5)
        np.testing.assert_array_equal(raw.X, original)
        np.testing.assert_array_equal(legacy.X[:, [0,2]], relative.X[:, [0,2]])
        np.testing.assert_array_equal(legacy.rssi, relative.rssi)
        np.testing.assert_allclose(np.median(relative.X[:, [1,3]], axis=-1), 0, atol=1e-7)
        self.assertFalse(np.any((relative.y_text == 'EMPTY') & (relative.trial_number == 1)))
        with self.assertRaises(ValueError):
            prepare_arm(raw, 'unknown')

    def test_vote_and_mismatched_trials_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = []
            for method, guesses in [('existing',['Z1','Z3']), ('generic',['Z1','Z2']), ('session',['Z2','Z4'])]:
                for i, guess in enumerate(guesses):
                    rows.append(dict(method=method, seed=1, session='DEV', group=str(i),
                                     true=['Z1','Z2'][i], pred=guess, orientation='N'))
            frame = pd.DataFrame(rows)
            prediction_reports(frame, output)
            vote = pd.read_csv(output/'majority_vote_predictions.csv')
            self.assertEqual(vote.pred.tolist(), ['Z1','Z3'])  # three-way tie => existing
            with self.assertRaises(ValueError):
                prediction_reports(frame.iloc[:-1], output)
            with self.assertRaises(ValueError):
                prediction_reports(pd.concat([frame,frame.iloc[:1]]), output)
            wrong = frame.copy()
            wrong.loc[0,'true'] = 'Z9'
            with self.assertRaises(ValueError):
                prediction_reports(wrong, output)
            wrong_orientation = frame.copy()
            wrong_orientation.loc[0, 'orientation'] = 'S'
            with self.assertRaises(ValueError):
                prediction_reports(wrong_orientation, output)

    def test_matched_training_and_read_only_frozen_diagnostics(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root/'cache'
            cache.mkdir()
            for i, name in enumerate(['A','B','DEV']):
                save_windows(cache/f'{name}.npz', fixture(name, i+1))
            before = {p.name:sha(p) for p in cache.glob('*.npz')}
            run = root/'original'
            common = dict(project_root=ROOT, cache_dir=cache, threads=1)
            develop(argparse.Namespace(**common, train=['A','B'], dev=['DEV'], output=run,
                methods=['existing','session'], strengths=[1.], seeds=[2026], epochs=1,
                eval_every=1, batch_size=16, lr=.001, exclude_dev_occupied_trials=[]))
            weights_before = {p.name:sha(p) for p in (run/'checkpoints').glob('*.pt')}
            audit_args = argparse.Namespace(**common, development_run=run, methods=['existing'],
                                          raw_dir=None, output=root/'audit')
            audit(audit_args)
            receipt = json.loads((root/'audit/diagnostic_receipt.json').read_text())
            self.assertTrue(receipt['complete'])
            self.assertTrue(receipt['saved_predictions_reproduced'])
            self.assertFalse(receipt['fitting_performed'])
            with zipfile.ZipFile(root/'audit/diagnostic_results.zip') as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertIn('diagnostic_receipt.json', zipped.namelist())
                self.assertFalse(any(p.endswith(('.pt', '.zip')) for p in zipped.namelist()))
            self.assertTrue((root/'audit/csi_relative_shape.png').is_file())
            frame = pd.read_csv(root/'audit/diagnostic_predictions.csv')
            np.testing.assert_allclose(frame[[f'p_Z{i}' for i in range(1,10)]].sum(axis=1), 1)
            self.assertEqual({p.name:sha(p) for p in (run/'checkpoints').glob('*.pt')}, weights_before)
            with self.assertRaises(FileExistsError):
                audit(audit_args)
            comp = root/'comparison'
            compare(argparse.Namespace(**common, train=['A','B'], dev=['DEV'], output=comp,
                seeds=[2026], epochs=1, eval_every=1, batch_size=16, lr=.001,
                exclude_dev_occupied_by_session=[]))
            index = json.loads((comp/'checkpoint_index.json').read_text())
            self.assertEqual(len(index), 2)
            self.assertEqual(len({r['updates'] for r in index}), 1)
            old = torch.load(run/'checkpoints/existing_strength0_seed2026.pt', weights_only=True)
            new = torch.load(comp/'checkpoints/legacy_seed2026.pt', weights_only=True)
            for key in old['state_dict']:
                self.assertTrue(torch.equal(old['state_dict'][key],new['state_dict'][key]), key)
            original_predictions = pd.read_csv(run/'development_predictions.csv')
            comparison_predictions = pd.read_csv(comp/'comparison_predictions.csv')
            expected = original_predictions[original_predictions.method=='existing'].set_index('group').pred
            actual = comparison_predictions[comparison_predictions.arm=='legacy'].set_index('group').pred
            pd.testing.assert_series_equal(expected.sort_index(), actual.sort_index())
            self.assertEqual({p.name:sha(p) for p in cache.glob('*.npz')}, before)
            with self.assertRaises(FileExistsError):
                compare(argparse.Namespace(**common, train=['A','B'], dev=['DEV'], output=comp,
                    seeds=[2026], epochs=1, eval_every=1, batch_size=16, lr=.001,
                    exclude_dev_occupied_by_session=[]))


if __name__ == '__main__':
    unittest.main()
