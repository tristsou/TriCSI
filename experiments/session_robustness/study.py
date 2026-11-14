"""All-source session-robustness study. No real-training-data subsampling."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import (LABELS, assert_disjoint, calibrate, combine, read_sessions,
                   require_all_classes, safe_session, save_windows)
from .generator import SessionGenerator

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("none", "existing", "generic", "session")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def code_hashes(root):
    files = ["train_tricsi_anchor_final.py", "train_csi_cnn.py", "csi_io.py", "project_config.py"]
    files += [str(p.relative_to(root)).replace("\\", "/") for p in sorted((root / "experiments/session_robustness").glob("*.py"))]
    return {name: sha(root / name) for name in files}


def audit_trial(path):
    """Fail closed on wrong width, bad IQ ranges, or mixed trial identities."""
    info = {"file": str(path), "rows": 0, "bad_rows": 0, "left": 0, "right": 0,
            "status": "ok", "session": "", "zone": "", "trial": "", "orientation": ""}
    needed = {"device_time_s", "session", "zone", "trial", "orientation", "link", "rssi"}
    iq = [f"iq_{i:03d}" for i in range(128)]
    identity = None
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not (needed | set(iq)) <= set(reader.fieldnames)):
                info["status"] = "unsupported_columns"
                return info
            for row in reader:
                info["rows"] += 1
                try:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError("row width")
                    this = tuple(row[k].strip() for k in ("session", "zone", "trial", "orientation"))
                    if identity is None:
                        identity = this
                    if this != identity or this[1] not in LABELS:
                        raise ValueError("mixed/unknown identity")
                    if not np.isfinite([float(row["device_time_s"]), float(row["rssi"])]).all():
                        raise ValueError("non-finite timestamp or RSSI")
                    int(row["trial"])
                    values = [int(row[k]) for k in iq]
                    if min(values) < -128 or max(values) > 127:
                        raise ValueError("signed IQ range")
                    link = row["link"].strip().lower()
                    if link in ("left", "right"):
                        info[link] += 1
                except (ValueError, KeyError, TypeError):
                    info["bad_rows"] += 1
        if identity:
            info.update(zip(("session", "zone", "trial", "orientation"), identity))
        if info["bad_rows"]:
            info["status"] = "invalid_rows"
        elif not info["left"] or not info["right"]:
            info["status"] = "missing_two_link_data"
    except (OSError, UnicodeError, csv.Error) as exc:
        info["status"] = f"read_error:{type(exc).__name__}"
    return info


def inventory(args):
    raw = args.raw_dir.resolve()
    if not raw.is_dir():
        raise ValueError(f"Raw directory not found: {raw}. Supply your local data/raw directory.")
    paths = sorted(p for p in raw.rglob("*.csv") if not p.name.startswith("schedule_"))
    records = [audit_trial(p) for p in paths]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "inventory.json", {"raw_dir": str(raw), "files": records,
               "note": "Read-only inventory. Does not fit a model or modify recordings."})
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "inventory.csv", index=False)
    if len(frame):
        print(frame.groupby(["session", "status"], dropna=False).size().to_string())
    else:
        print("No CSV recordings found.")
    print(f"Saved {args.output / 'inventory.csv'}")


def prepare(args):
    from .model import load_anchor
    load_anchor(args.project_root)
    import train_csi_cnn as cnn
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sessions:
        safe_session(name)
        destination = args.cache_dir / f"{name}.npz"
        metadata = args.cache_dir / f"{name}.json"
        if destination.exists() or metadata.exists():
            raise ValueError(f"Cache already exists for {name}; use a new cache directory.")
        paths = cnn.trial_paths_for_session(args.raw_dir, name)
        reports = [audit_trial(p) for p in paths]
        bad = [r for r in reports if r["status"] != "ok" or r["session"] != name]
        if bad:
            raise ValueError(f"Raw audit failed; inspect inventory before preparing: {bad[:3]}")
        raw = cnn.gather_raw_windows(paths, window_seconds=1., hop_seconds=0.25,
                                    target_steps=32, min_packets_per_link=12)
        if set(raw.session.astype(str)) != {name}:
            raise ValueError("Filename session and recording metadata disagree.")
        input_groups = {(r["zone"], int(r["trial"]), r["orientation"]) for r in reports}
        output_groups = set(zip(raw.y_text.astype(str), raw.trial_number.tolist(), raw.orientation.astype(str)))
        if len(input_groups) != len(paths):
            raise ValueError("Duplicate trial identities across files.")
        if input_groups != output_groups:
            raise ValueError(f"Some recordings produced no windows: {input_groups - output_groups}")
        save_windows(destination, raw)
        write_json(metadata, {"session": name, "cache_sha256": sha(destination),
                   "trials": len(input_groups), "windows": len(raw.X),
                   "sources": {str(p): sha(p) for p in paths},
                   "extractor": "Existing two-link CNN extractor: 4x32x64; no augmentation or fitting",
                   "source_code_sha256": code_hashes(args.project_root)})
        print(f"Prepared {name}: {len(input_groups)} trials, {len(raw.X)} windows")


def metrics(rows):
    frame = pd.DataFrame(rows)
    y, pred = frame["true"].to_numpy(), frame.pred.to_numpy()
    occupied, empty = y != "EMPTY", y == "EMPTY"
    if not occupied.any() or not empty.any():
        raise ValueError("Evaluation requires occupied and independent Empty trials.")
    return {"trial_accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "occupied_accuracy": float(np.mean(y[occupied] == pred[occupied])),
            "empty_recall": float(np.mean(pred[empty] == "EMPTY")), "trials": len(y)}


def summaries(frame):
    # First aggregate sessions within each seed, then aggregate algorithm seeds.
    per_seed = frame.groupby(["method", "strength", "seed"]).agg(
        mean_session_accuracy=("trial_accuracy", "mean"),
        worst_session_accuracy=("trial_accuracy", "min"),
        mean_session_macro_f1=("macro_f1", "mean")).reset_index()
    return per_seed.groupby(["method", "strength"]).agg(
        mean_session_accuracy=("mean_session_accuracy", "mean"),
        worst_session_accuracy=("worst_session_accuracy", "mean"),
        mean_session_macro_f1=("mean_session_macro_f1", "mean"),
        seed_sd_mean_accuracy=("mean_session_accuracy", lambda s: float(s.std(ddof=0))),
        seeds=("seed", "nunique")).reset_index()


def resolve_exclusions(sessions, shared=(), per_session=()):
    """Resolve explicit protocol exclusions without assuming equal trial numbering."""
    if shared and per_session:
        raise ValueError("Use shared exclusions OR per-session exclusions, not both.")
    if any(n <= 0 for n in shared) or len(shared) != len(set(shared)):
        raise ValueError("Excluded trial numbers must be unique positive integers.")
    result = {s: list(shared) for s in sessions}
    seen = set()
    for spec in per_session:
        try:
            name, numbers = spec.split(":", 1)
            values = [int(n) for n in numbers.split(",")] if numbers else []
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid exclusion {spec!r}; use e.g. S8:1,2") from exc
        if name not in result or name in seen:
            raise ValueError(f"Unknown or duplicate exclusion session {name!r}")
        if any(n <= 0 for n in values) or len(values) != len(set(values)):
            raise ValueError(f"Invalid excluded trial numbers in {spec!r}")
        result[name] = values
        seen.add(name)
    return result


def checkpoint_rank(rows):
    """One joint score over development sessions; never pick an epoch per session."""
    if not rows:
        raise ValueError("No development session scores.")
    if len({r["session"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development sessions in checkpoint scoring.")
    accuracies = [float(r["trial_accuracy"]) for r in rows]
    macro = [float(r["macro_f1"]) for r in rows]
    if not np.isfinite(accuracies + macro).all():
        raise ValueError("Non-finite development score.")
    return (min(accuracies), float(np.mean(accuracies)), float(np.mean(macro)))


def develop(args):
    from .model import load_anchor, predict_trials, save_checkpoint, source_prototypes, train
    import torch
    torch.set_num_threads(args.threads)
    if set(args.train) & set(args.dev):
        raise ValueError("Training and development sessions must be disjoint.")
    if len(args.train) < 2:
        raise ValueError("Supply at least two source sessions.")
    if len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        raise ValueError("Duplicate seeds or methods.")
    if "existing" not in args.methods or "session" not in args.methods:
        raise ValueError("Include existing and session for the primary paired comparison.")
    if len(args.strengths) != len(set(args.strengths)) or any(not np.isfinite(s) or s <= 0 for s in args.strengths):
        raise ValueError("Strengths must be unique positive numbers.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative.")
    anchor = load_anchor(args.project_root)
    raw = read_sessions(args.cache_dir, args.train + args.dev)
    source_sets, calibration_sets = zip(*(calibrate(raw[s]) for s in args.train))
    source, calibration = combine(source_sets), combine(calibration_sets)
    require_all_classes(source)
    exclusions = resolve_exclusions(args.dev, args.exclude_dev_occupied_trials,
                                    getattr(args, "exclude_dev_occupied_by_session", []))
    targets = {}
    for session in args.dev:
        assert_disjoint(source, raw[session])
        targets[session], _ = calibrate(raw[session], exclusions[session])
        require_all_classes(targets[session])
        print(f"Scoring {session}: {len(set(targets[session].group.astype(str)))} whole trials; "
              f"occupied exclusions={exclusions[session]}; Empty trial 1 excluded", flush=True)
    generator = SessionGenerator.fit(source, calibration)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    config = {"objective": "Improve real cross-session robustness using ALL eligible source trials",
              "train_sessions": args.train, "development_sessions": args.dev,
              "methods": args.methods, "strengths": args.strengths, "seeds": args.seeds,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "eval_every": getattr(args, "eval_every", 5),
              "epoch_selection": "Per run: worst development-session accuracy, then mean accuracy, then mean macro-F1; exact ties retain earlier epoch. One checkpoint for all sessions.",
              "exclude_dev_occupied_trials": args.exclude_dev_occupied_trials,
              "development_trial_exclusions": exclusions,
              "source_trials": len(set(source.group.astype(str))), "source_windows": len(source.X),
              "source_calibration_trials": len(set(calibration.group.astype(str))),
              "cache_sha256": {s: sha(args.cache_dir / f"{s}.npz") for s in raw},
              "source_code_sha256": code_hashes(args.project_root),
              "selection_rule": "Highest mean-seed worst-session accuracy, then mean accuracy, then macro-F1; ties prefer lower strength",
              "interpretation": "Development only. Previously inspected S8/S9 are not new confirmatory tests.",
              "generator_type": "Empirical residual perturbation generator; not electromagnetic simulation"}
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "generator.json", generator.to_dict())
    rows, predictions, records, learning_curve = [], [], [], []
    for method in args.methods:
        strengths = args.strengths if method in ("generic", "session") else [0.]
        for strength in strengths:
            for seed in args.seeds:
                started = time.monotonic()

                def score_epoch(candidate, candidate_norm, epoch, update_count, train_loss):
                    candidate_prototypes = source_prototypes(candidate, candidate_norm, source, anchor)
                    session_metrics = []
                    for session_name, target in targets.items():
                        predicted = predict_trials(candidate, candidate_norm, candidate_prototypes, target, anchor)
                        scored = dict(session=session_name, **metrics(predicted))
                        session_metrics.append(scored)
                        learning_curve.append(dict(method=method, strength=strength, seed=seed,
                            epoch=epoch, updates=update_count, training_loss=train_loss, **scored))
                    pd.DataFrame(learning_curve).to_csv(args.output / "learning_curves.csv", index=False)
                    rank = checkpoint_rank(session_metrics)
                    detail = " ".join(f"{r['session']}={r['trial_accuracy']:.3f}" for r in session_metrics)
                    print(f"CHECKPOINT {method} strength={strength:g} seed={seed} epoch={epoch}: "
                          f"{detail} worst={rank[0]:.3f} mean={rank[1]:.3f}", flush=True)
                    return {"rank": list(rank), "sessions": session_metrics}

                model, norm, updates, selection = train(source, anchor, generator, method=method,
                    strength=strength, seed=seed, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, on_epoch=score_epoch,
                    eval_every=getattr(args, "eval_every", 5))
                prototypes = source_prototypes(model, norm, source, anchor)
                name = f"{method}_strength{strength:g}_seed{seed}.pt"
                path = checkpoints / name
                save_checkpoint(path, model, norm, prototypes, method=method, strength=strength,
                                seed=seed, updates=updates, source_sessions=args.train, **selection)
                records.append({"method": method, "strength": strength, "seed": seed,
                                "path": f"checkpoints/{name}", "sha256": sha(path), "updates": updates,
                                "selected_epoch": selection["selected_epoch"],
                                "selected_updates": selection["selected_updates"]})
                for session, target in targets.items():
                    predicted = predict_trials(model, norm, prototypes, target, anchor)
                    for row in predicted:
                        predictions.append(dict(method=method, strength=strength, seed=seed,
                                                selected_epoch=selection["selected_epoch"], **row))
                    row = dict(method=method, strength=strength, seed=seed, session=session,
                               updates=updates, selected_epoch=selection["selected_epoch"],
                               selected_updates=selection["selected_updates"],
                               train_seconds=time.monotonic()-started, **metrics(predicted))
                    rows.append(row)
                    print(f"DEVELOPMENT {method} strength={strength:g} seed={seed} "
                          f"selected_epoch={selection['selected_epoch']} {session}: {row['trial_accuracy']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(args.output / "development_metrics.csv", index=False)
                pd.DataFrame(predictions).to_csv(args.output / "development_predictions.csv", index=False)
                write_json(args.output / "checkpoint_index.json", records)
    if len({r["updates"] for r in records}) != 1:
        raise ValueError("Training update counts differ across conditions.")
    summary = summaries(pd.DataFrame(rows))
    summary.to_csv(args.output / "development_summary.csv", index=False)
    write_json(args.output / "completed.json", {"complete": True, "fits": len(records)})
    print(summary.to_string(index=False))


def freeze(args):
    run = args.development_run
    if not (run / "completed.json").exists():
        raise ValueError("Only a completed development run can be frozen.")
    config = json.loads((run / "run_config.json").read_text())
    if config["source_code_sha256"] != code_hashes(args.project_root):
        raise ValueError("Code changed after development. Rerun development before freezing.")
    summary = pd.read_csv(run / "development_summary.csv")
    ranking = ["worst_session_accuracy", "mean_session_accuracy", "mean_session_macro_f1", "strength"]
    ranked = summary.sort_values(ranking, ascending=[False, False, False, True], kind="stable")
    choices = ranked.groupby("method", sort=False).head(1)
    selected = str(ranked.iloc[0]["method"])
    index = json.loads((run / "checkpoint_index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "models").mkdir()
    frozen = []
    for choice in choices.to_dict("records"):
        for record in index:
            if record["method"] != choice["method"] or not np.isclose(record["strength"], choice["strength"]):
                continue
            source = run / record["path"]
            if sha(source) != record["sha256"]:
                raise ValueError("Checkpoint changed since development.")
            name = f"models/{source.name}"
            shutil.copy2(source, args.output / name)
            frozen.append(dict(record, path=name))
    for name in ("run_config.json", "generator.json", "development_summary.csv"):
        shutil.copy2(run / name, args.output / name)
    write_json(args.output / "frozen_manifest.json", {"config": config, "models": frozen,
        "selected_method": selected, "freeze_source": str(run.resolve()),
        "note": "Frozen fitted checkpoints; evaluate never retrains or estimates generator parameters."})
    print(f"Frozen {len(frozen)} checkpoints. Development-selected method: {selected}")


def paired_comparisons(predictions):
    output = []
    for (session, seed), frame in predictions.groupby(["session", "seed"]):
        base = frame[frame.method == "existing"].set_index("group")
        if base.empty:
            raise ValueError("Missing existing-augmentation baseline.")
        for method, other in frame.groupby("method"):
            if method == "existing":
                continue
            other = other.set_index("group")
            if set(base.index) != set(other.index):
                raise ValueError("Paired methods have different trial sets.")
            other = other.loc[base.index]
            if not np.array_equal(base["true"], other["true"]):
                raise ValueError("Paired labels disagree.")
            b = base.pred.to_numpy() == base["true"].to_numpy()
            o = other.pred.to_numpy() == other["true"].to_numpy()
            corrected, harmed = int(np.sum(~b & o)), int(np.sum(b & ~o))
            discordant = corrected + harmed
            p = float(binomtest(corrected, discordant, p=.5).pvalue) if discordant else 1.
            output.append({"session": session, "seed": int(seed), "method": method,
                "corrected": corrected, "harmed": harmed, "accuracy_gain_pp": float(100*np.mean(o.astype(float)-b)),
                "mcnemar_exact_p_exploratory": p,
                "note": "Within-session exploratory test; seeds are not independent participant/session replications."})
    return pd.DataFrame(output)


def evaluate(args):
    from .model import load_anchor, load_checkpoint, predict_trials
    import torch
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.bundle / "frozen_manifest.json").read_text())
    config = manifest["config"]
    if set(args.sessions) & set(config["train_sessions"] + config["development_sessions"]):
        raise ValueError("Final evaluation sessions overlap training/development.")
    if config["source_code_sha256"] != code_hashes(args.project_root):
        raise ValueError("Code changed after development/freeze; use the original version.")
    raw = read_sessions(args.cache_dir, args.sessions)
    for session in raw:
        if sha(args.cache_dir / f"{session}.npz") in config["cache_sha256"].values():
            raise ValueError("Evaluation cache duplicates a development/source cache.")
    exclusions = resolve_exclusions(args.sessions, args.exclude_occupied_trials,
                                    getattr(args, "exclude_occupied_by_session", []))
    targets = {s: calibrate(r, exclusions[s])[0] for s, r in raw.items()}
    for target in targets.values():
        require_all_classes(target)
    anchor = load_anchor(args.project_root)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, predicted = [], []
    for record in manifest["models"]:
        path = args.bundle / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("Frozen checkpoint hash mismatch.")
        model, norm, prototypes, fitted_metadata = load_checkpoint(path, anchor)
        for session, target in targets.items():
            trial_rows = predict_trials(model, norm, prototypes, target, anchor)
            rows.append(dict(method=record["method"], strength=record["strength"],
                             seed=record["seed"], session=session,
                             selected_epoch=fitted_metadata.get("selected_epoch", config["epochs"]),
                             **metrics(trial_rows)))
            predicted.extend(dict(method=record["method"], seed=record["seed"], **r) for r in trial_rows)
    frame, predictions = pd.DataFrame(rows), pd.DataFrame(predicted)
    frame.to_csv(args.output / "test_metrics.csv", index=False)
    predictions.to_csv(args.output / "test_predictions.csv", index=False)
    summary = summaries(frame)
    summary.to_csv(args.output / "test_summary.csv", index=False)
    paired_comparisons(predictions).to_csv(args.output / "paired_changes.csv", index=False)
    for (session, method, seed), group in predictions.groupby(["session", "method", "seed"]):
        cm = confusion_matrix(group["true"], group.pred, labels=LABELS)
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(args.output / f"confusion_{session}_{method}_{seed}.csv")
    predictions["correct"] = predictions["true"] == predictions.pred
    predictions.groupby(["session", "method", "seed", "orientation"]).correct.agg(["mean", "count"]).to_csv(args.output / "orientation_accuracy.csv")
    write_json(args.output / "evaluation_receipt.json", {"fitting_performed": False,
        "bundle_manifest_sha256": sha(args.bundle / "frozen_manifest.json"),
        "sessions": args.sessions, "exclude_occupied_trials": args.exclude_occupied_trials,
        "evaluation_trial_exclusions": exclusions,
        "cache_sha256": {s: sha(args.cache_dir / f"{s}.npz") for s in raw},
        "note": "Report each day. Seeds and overlapping windows are not independent recording sessions."})
    plot_results(frame, args.output)
    print(summary.to_string(index=False))
    print("Evaluation complete. No fitting performed.")


def plot_results(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grouped = frame.groupby(["session", "method"]).trial_accuracy.mean().unstack("method")
    ax = grouped.plot.bar(figsize=(9, 4), ylim=(0, 1), rot=0)
    ax.set_ylabel("Whole-trial accuracy (mean across training seeds)")
    ax.set_xlabel("Real evaluation session")
    ax.set_title("Cross-session localization: matched model and training budget")
    ax.legend(title="Training augmentation", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output / "session_accuracy.png", dpi=220)
    plt.savefig(output / "session_accuracy.svg")
    plt.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="Read-only audit; run this first on your PC")
    inv.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    inv.add_argument("--output", type=Path, default=ROOT / "results/session_robustness_inventory")
    inv.set_defaults(func=inventory)
    prep = sub.add_parser("prepare", help="Cache existing two-link recordings without fitting")
    prep.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    prep.add_argument("--sessions", nargs="+", required=True)
    prep.add_argument("--cache-dir", type=Path, required=True)
    prep.set_defaults(func=prepare)
    dev = sub.add_parser("develop", help="All-source training; development sessions select strength")
    dev.add_argument("--cache-dir", type=Path, required=True)
    dev.add_argument("--train", nargs="+", required=True)
    dev.add_argument("--dev", nargs="+", required=True)
    dev.add_argument("--output", type=Path, required=True)
    dev.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    dev.add_argument("--strengths", nargs="+", type=float, default=[.5, 1., 1.5])
    dev.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    dev.add_argument("--epochs", type=int, default=150,
                     help="Maximum epochs, run equally for every condition")
    dev.add_argument("--eval-every", type=int, default=5,
                     help="Development checkpoint interval; also evaluates first and final epochs")
    dev.add_argument("--batch-size", type=int, default=128)
    dev.add_argument("--lr", type=float, default=.001)
    dev.add_argument("--threads", type=int, default=2)
    dev.add_argument("--exclude-dev-occupied-trials", nargs="*", type=int, default=[])
    dev.add_argument("--exclude-dev-occupied-by-session", nargs="+", default=[],
                     help="Session-specific exclusions, e.g. 'S8:1,2'; unspecified sessions keep every occupied trial")
    dev.set_defaults(func=develop)
    fr = sub.add_parser("freeze", help="Copy selected fitted models; no retraining")
    fr.add_argument("--development-run", type=Path, required=True)
    fr.add_argument("--output", type=Path, required=True)
    fr.set_defaults(func=freeze)
    ev = sub.add_parser("evaluate", help="Evaluate frozen models with Empty-only calibration")
    ev.add_argument("--bundle", type=Path, required=True)
    ev.add_argument("--cache-dir", type=Path, required=True)
    ev.add_argument("--sessions", nargs="+", required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--threads", type=int, default=2)
    ev.add_argument("--exclude-occupied-trials", nargs="*", type=int, default=[])
    ev.add_argument("--exclude-occupied-by-session", nargs="+", default=[],
                    help="Session-specific exclusions, e.g. 'S8:1,2'; unspecified sessions keep every occupied trial")
    ev.set_defaults(func=evaluate)
    return p


def main():
    args = parser().parse_args()
    for name in ("epochs", "batch_size", "threads", "eval_every"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be positive.")
    if hasattr(args, "lr") and (not np.isfinite(args.lr) or args.lr <= 0):
        raise SystemExit("lr must be finite and positive.")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"STOP: {exc}") from exc
