from pathlib import Path
import argparse, json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import CLASS_ORDER, RAW_DATA_DIR, RESULTS_DIR, WINDOW_SECONDS, HOP_SECONDS
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session


def load_rows(raw_dir, session, window, hop):
    paths = trial_paths_for_session(raw_dir, session)
    rows = gather_windows(paths, window_seconds=window, hop_seconds=hop)
    for r in rows:
        r["_session"] = session
    print(f"{session}: {len(paths)} trials, {len(rows)} windows")
    return rows


def mat(rows, names):
    X = np.asarray([[float(r["features"][n]) for n in names] for r in rows], dtype=float)
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=CLASS_ORDER, average="macro", zero_division=0)),
    }


def trial_metrics(rows, probs, classes):
    groups = np.asarray(
        [f"{r['_session']}|{r['zone']}|{trial_number_from_row(r)}|{r['orientation']}" for r in rows],
        dtype=object,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    yt, yp = [], []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        yt.append(str(y[idx[0]]))
        yp.append(str(classes[int(np.argmax(probs[idx].mean(axis=0)))]))
    return metrics(np.asarray(yt, dtype=object), np.asarray(yp, dtype=object))


def fit_eval(train_rows, test_rows, names, seed):
    Xtr, ytr = mat(train_rows, names)
    Xte, yte = mat(test_rows, names)
    rf = make_rf(seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)
    wm = metrics(yte, pred)
    tm = trial_metrics(test_rows, prob, rf.classes_)
    return {
        "window_accuracy": wm["accuracy"],
        "window_macro_f1": wm["macro_f1"],
        "trial_accuracy": tm["accuracy"],
        "trial_macro_f1": tm["macro_f1"],
    }


def north(rows):
    return [r for r in rows if int(trial_number_from_row(r)) in (1, 5)]


def pooled_cv(s4, s5, names, seed):
    rng = np.random.default_rng(seed)
    assign = {}
    for session in ("S4", "S5"):
        for zone in CLASS_ORDER:
            trials = np.arange(1, 6)
            rng.shuffle(trials)
            for fold, t in enumerate(trials, 1):
                assign[(session, zone, int(t))] = fold

    rows = s4 + s5
    out = []
    print("\n=== POOLED S4+S5 WHOLE-TRIAL 5-FOLD CV ===")
    for fold in range(1, 6):
        test = [
            r for r in rows
            if assign[(str(r["_session"]), str(r["zone"]), int(trial_number_from_row(r)))] == fold
        ]
        test_ids = {id(r) for r in test}
        train = [r for r in rows if id(r) not in test_ids]
        m = fit_eval(train, test, names, seed + 100 + fold)
        m["fold"] = fold
        out.append(m)
        print(f"Fold {fold}: window={m['window_accuracy']:.3f} F1={m['window_macro_f1']:.3f} trial={m['trial_accuracy']:.3f}")

    a = np.asarray([m["window_accuracy"] for m in out])
    f = np.asarray([m["window_macro_f1"] for m in out])
    t = np.asarray([m["trial_accuracy"] for m in out])
    return {
        "folds": out,
        "window_accuracy_mean": float(a.mean()),
        "window_accuracy_std": float(a.std(ddof=1)),
        "window_macro_f1_mean": float(f.mean()),
        "window_macro_f1_std": float(f.std(ddof=1)),
        "trial_accuracy_mean": float(t.mean()),
        "trial_accuracy_std": float(t.std(ddof=1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    s4 = load_rows(args.raw_dir, "S4", args.window, args.hop)
    s5 = load_rows(args.raw_dir, "S5", args.window, args.hop)

    names = sorted(s4[0]["features"])
    if sorted(s5[0]["features"]) != names:
        raise RuntimeError("S4/S5 feature mismatch")

    print("\n=== DIRECT CROSS-SESSION ===")
    a = fit_eval(s4, s5, names, args.seed + 1)
    b = fit_eval(s5, s4, names, args.seed + 2)
    print(f"S4 -> S5: window={a['window_accuracy']:.3f} F1={a['window_macro_f1']:.3f} trial={a['trial_accuracy']:.3f}")
    print(f"S5 -> S4: window={b['window_accuracy']:.3f} F1={b['window_macro_f1']:.3f} trial={b['trial_accuracy']:.3f}")

    print("\n=== NORTH-ONLY CROSS-SESSION ===")
    an = fit_eval(north(s4), north(s5), names, args.seed + 11)
    bn = fit_eval(north(s5), north(s4), names, args.seed + 12)
    print(f"S4 N -> S5 N: window={an['window_accuracy']:.3f} F1={an['window_macro_f1']:.3f} trial={an['trial_accuracy']:.3f}")
    print(f"S5 N -> S4 N: window={bn['window_accuracy']:.3f} F1={bn['window_macro_f1']:.3f} trial={bn['trial_accuracy']:.3f}")

    pooled = pooled_cv(s4, s5, names, args.seed)

    print("\n" + "="*72)
    print("S4/S5 STANDARDIZED-GEOMETRY SUMMARY")
    print("="*72)
    print(f"S4 -> S5        : window={a['window_accuracy']:.3f} | trial={a['trial_accuracy']:.3f}")
    print(f"S5 -> S4        : window={b['window_accuracy']:.3f} | trial={b['trial_accuracy']:.3f}")
    print(f"S4 N -> S5 N    : window={an['window_accuracy']:.3f} | trial={an['trial_accuracy']:.3f}")
    print(f"S5 N -> S4 N    : window={bn['window_accuracy']:.3f} | trial={bn['trial_accuracy']:.3f}")
    print(f"Pooled S4+S5 CV : window={pooled['window_accuracy_mean']:.3f} ± {pooled['window_accuracy_std']:.3f} | trial={pooled['trial_accuracy_mean']:.3f} ± {pooled['trial_accuracy_std']:.3f}")

    out_dir = RESULTS_DIR / "s4_s5_standardized"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "s4_to_s5": a,
            "s5_to_s4": b,
            "s4_n_to_s5_n": an,
            "s5_n_to_s4_n": bn,
            "pooled_cv": pooled,
        }, f, indent=2)

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
