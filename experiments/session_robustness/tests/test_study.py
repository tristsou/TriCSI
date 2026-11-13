"""Software checks on artificial fixtures. No experimental accuracies claimed."""
import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.session_robustness.data import (LABELS, Windows, assert_disjoint,
    calibrate, combine, load_windows, read_sessions, save_windows)
from experiments.session_robustness.generator import SessionGenerator
from experiments.session_robustness.study import ROOT, audit_trial, develop, evaluate, freeze, prepare


def fixture(session, seed):
    rng = np.random.default_rng(seed)
    records = []
    for label in LABELS:
        trials = [1, 2] if label == "EMPTY" else [1]
        for trial in trials:
            for window in range(2):
                profile = 2 + 0.1 * np.sin(np.arange(64) / 8 + seed)
                absolute = np.stack([profile, profile + .1])[:, None, :] + rng.normal(0, .01, (2, 32, 64))
                if label != "EMPTY":
                    absolute += .03 * int(label[1:]) * np.cos(np.arange(64) / 7)[None, None, :]
                centered = absolute - np.median(absolute, axis=-1, keepdims=True)
                x = np.stack([absolute[0], centered[0], absolute[1], centered[1]]).astype(np.float32)
                records.append((x, label, trial, window))
    n = len(records)
    return Windows(X=np.stack([r[0] for r in records]),
        rssi=np.tile(np.asarray([-40., 1., -42., 1.], np.float32), (n, 1)),
        y_text=np.array([r[1] for r in records]), session=np.array([session]*n),
        trial_number=np.array([r[2] for r in records]), orientation=np.array(["N"]*n),
        group=np.array([f"{session}|{r[1]}|{r[2]}|N" for r in records]),
        source_file=np.array([f"{session}/{r[1]}_{r[2]}.csv" for r in records]),
        window_index=np.array([r[3] for r in records]),
        window_start_s=np.array([r[3]*.25 for r in records]),
        left_packets=np.full(n, 32), right_packets=np.full(n, 32)).validate()


class DataTests(unittest.TestCase):
    def test_cache_roundtrip_and_session_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.npz"
            raw = fixture("A", 1)
            save_windows(path, raw)
            np.testing.assert_array_equal(load_windows(path).X, raw.X)
            save_windows(Path(folder) / "B.npz", raw)
            with self.assertRaises(ValueError):
                read_sessions(folder, ["B"])

    def test_empty_reference_excluded_and_no_fallback(self):
        raw = fixture("A", 1)
        scored, reference = calibrate(raw)
        self.assertFalse(np.any((scored.y_text == "EMPTY") & (scored.trial_number == 1)))
        self.assertTrue(np.all(reference.y_text == "EMPTY"))
        self.assertTrue(np.any(scored.y_text == "EMPTY"))
        with self.assertRaises(ValueError):
            calibrate(scored)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint(fixture("A", 1), fixture("A", 2))

    def test_raw_width_and_iq_range_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trial.csv"
            names = ["device_time_s", "session", "zone", "trial", "orientation", "link", "rssi"] + [f"iq_{i:03d}" for i in range(128)]
            row = ["0", "A", "Z1", "1", "N", "left", "-40"] + ["1"]*128
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(names)
                writer.writerow(row)
                row[5] = "right"
                writer.writerow(row)
            self.assertEqual(audit_trial(path)["status"], "ok")
            with path.open("a", newline="") as handle:
                csv.writer(handle).writerow(row + ["EXTRA"])
            self.assertEqual(audit_trial(path)["status"], "invalid_rows")

    def test_original_csv_extractor_to_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = root / "raw/A"
            session.mkdir(parents=True)
            names = ["device_time_s", "session", "zone", "trial", "orientation", "link", "rssi", "x_m", "y_m"] + [f"iq_{i:03d}" for i in range(128)]
            for label, trial in [("EMPTY", 1), ("EMPTY", 2), ("Z1", 1)]:
                path = session / f"{label}_{trial}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(names)
                    for t in np.arange(0, 2.01, .025):
                        for link in ("left", "right"):
                            writer.writerow([t, "A", label, trial, "N", link, -40, 0, 0] + [1, 2]*64)
            prepare(argparse.Namespace(project_root=ROOT, raw_dir=root/"raw", sessions=["A"], cache_dir=root/"cache"))
            raw = load_windows(root / "cache/A.npz")
            self.assertEqual(len(set(raw.group)), 3)
            self.assertEqual(raw.X.shape[1:], (4, 32, 64))


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        a, ar = calibrate(fixture("A", 1))
        b, br = calibrate(fixture("B", 2))
        self.source = combine([a, b])
        self.reference = combine([ar, br])
        self.generator = SessionGenerator.fit(self.source, self.reference)

    def test_determinism_identity_and_no_input_mutation(self):
        x = self.source.X[0].copy()
        style = self.generator.sample(np.random.default_rng(8))
        a = self.generator.transform(x, style, np.random.default_rng(9))
        b = self.generator.transform(x, style, np.random.default_rng(9))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(x, self.source.X[0])
        zero = self.generator.sample(np.random.default_rng(8), 0.)
        np.testing.assert_array_equal(x, self.generator.transform(x, zero, np.random.default_rng(9)))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(np.median(a[[1, 3]], axis=-1), 0, atol=1e-6)

    def test_source_only_calibration_enforced(self):
        with self.assertRaises(ValueError):
            SessionGenerator.fit(self.source, calibrate(fixture("C", 3))[1])


class PipelineTests(unittest.TestCase):
    def test_full_neural_development_freeze_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            for seed, name in enumerate(["A", "B", "DEV", "TEST"]):
                save_windows(cache / f"{name}.npz", fixture(name, seed+1))
            common = dict(project_root=ROOT, cache_dir=cache, threads=1)
            dev = root / "development"
            develop(argparse.Namespace(**common, train=["A", "B"], dev=["DEV"],
                output=dev, methods=["none", "existing", "generic", "session"],
                strengths=[1.], seeds=[2026], epochs=1, batch_size=16, lr=.001,
                exclude_dev_occupied_trials=[]))
            config = json.loads((dev / "run_config.json").read_text())
            # 10 scored trials per source; all 20 remain in every condition.
            self.assertEqual(config["source_trials"], 20)
            index = json.loads((dev / "checkpoint_index.json").read_text())
            self.assertEqual(len({r["updates"] for r in index}), 1)
            bundle = root / "bundle"
            freeze(argparse.Namespace(project_root=ROOT, development_run=dev, output=bundle))
            result = root / "evaluation"
            evaluate(argparse.Namespace(**common, bundle=bundle, sessions=["TEST"],
                output=result, exclude_occupied_trials=[]))
            receipt = json.loads((result / "evaluation_receipt.json").read_text())
            self.assertFalse(receipt["fitting_performed"])
            self.assertTrue((result / "paired_changes.csv").is_file())
            with self.assertRaises(ValueError):
                evaluate(argparse.Namespace(**common, bundle=bundle, sessions=["DEV"],
                    output=root / "bad", exclude_occupied_trials=[]))


if __name__ == "__main__":
    unittest.main()
