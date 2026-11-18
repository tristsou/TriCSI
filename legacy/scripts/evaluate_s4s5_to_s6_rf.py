from pathlib import Path
import argparse, csv, json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import CLASS_ORDER, RAW_DATA_DIR, RESULTS_DIR, MODEL_DIR, WINDOW_SECONDS, HOP_SECONDS
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session

def load_rows(raw_dir, session, window, hop):
    paths = trial_paths_for_session(raw_dir, session)
    if not paths:
        raise SystemExit(f"No trial files found for {session}")
    rows = gather_windows(paths, window_seconds=window, hop_seconds=hop)
    for r in rows:
        r["_session"] = session
    print(f"{session}: {len(paths)} trials, {len(rows)} windows")
    return rows

def matrix(rows, names):
    X = np.asarray([[float(r["features"][n]) for n in names] for r in rows], dtype=float)
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
    }

def aggregate_trials(rows, probs, classes):
    groups = np.asarray(
        [f"{r['_session']}|{r['zone']}|{trial_number_from_row(r)}|{r['orientation']}" for r in rows],
        dtype=object,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    yt, yp, details = [], [], []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        true = str(y[idx[0]])
        mean_prob = probs[idx].mean(axis=0)
        pred = str(classes[int(np.argmax(mean_prob))])
        sample = rows[int(idx[0])]
        yt.append(true)
        yp.append(pred)
        details.append({
            "trial_group": g,
            "zone_true": true,
            "trial": int(trial_number_from_row(sample)),
            "orientation": str(sample["orientation"]),
            "n_windows": int(len(idx)),
            "pred_mean_probability": pred,
            "correct": int(pred == true),
            "confidence": float(np.max(mean_prob)),
        })
    return np.asarray(yt, dtype=object), np.asarray(yp, dtype=object), details

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\n" + "="*78)
    print("FINAL STANDARDIZED-GEOMETRY RF: S4+S5 -> UNSEEN S6")
    print("="*78)

    s4 = load_rows(args.raw_dir, "S4", args.window, args.hop)
    s5 = load_rows(args.raw_dir, "S5", args.window, args.hop)
    s6 = load_rows(args.raw_dir, "S6", args.window, args.hop)

    train_rows = s4 + s5
    test_rows = s6
    names = sorted(train_rows[0]["features"])
    if sorted(test_rows[0]["features"]) != names:
        raise RuntimeError("S6 feature set does not match S4/S5.")

    Xtr, ytr = matrix(train_rows, names)
    Xte, yte = matrix(test_rows, names)

    print(f"\nTrain windows: {len(train_rows)}")
    print(f"Test windows : {len(test_rows)}")
    print(f"Features     : {len(names)}")

    print("\nTraining on all S4+S5...")
    rf = make_rf(args.seed)
    rf.fit(Xtr, ytr)

    print("Evaluating once on untouched S6...")
    pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)

    wm = metrics(yte, pred)
    trial_true, trial_pred, trial_rows = aggregate_trials(test_rows, prob, rf.classes_)
    tm = metrics(trial_true, trial_pred)

    zone_rows = []
    for zone in CLASS_ORDER:
        idx = np.where(yte == zone)[0]
        recall = float(np.mean(pred[idx] == zone)) if len(idx) else None
        zone_rows.append({"zone": zone, "n_windows": int(len(idx)), "recall": recall})

    out_dir = RESULTS_DIR / "rf_s4_s5_to_s6"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rf_s4_s5_to_s6.joblib"

    joblib.dump({
        "model": rf,
        "feature_names": names,
        "train_sessions": ["S4", "S5"],
        "test_session": "S6",
        "seed": args.seed,
    }, model_path)

    write_csv(out_dir / "trial_predictions.csv", trial_rows)
    write_csv(out_dir / "per_zone_recall.csv", zone_rows)

    summary = {
        "train_sessions": ["S4", "S5"],
        "test_session": "S6",
        "train_windows": len(train_rows),
        "test_windows": len(test_rows),
        "n_features": len(names),
        "window_metrics": wm,
        "trial_mean_probability_metrics": tm,
        "per_zone_recall": zone_rows,
        "note": "Untouched unseen-session evaluation only if S6 was not used for tuning first.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*78)
    print("S4+S5 -> S6 UNSEEN-SESSION RF RESULTS")
    print("="*78)
    print(f"Window level: accuracy={wm['accuracy']:.3f} macroF1={wm['macro_f1']:.3f}")
    print(f"Trial mean probability: accuracy={tm['accuracy']:.3f} macroF1={tm['macro_f1']:.3f}")
    print("\nPer-zone window recall:")
    for r in zone_rows:
        v = "NA" if r["recall"] is None else f"{r['recall']:.3f}"
        print(f"  {r['zone']:5s}: {v}")
    print(f"\nSaved model  : {model_path}")
    print(f"Saved results: {out_dir}")

if __name__ == "__main__":
    main()
