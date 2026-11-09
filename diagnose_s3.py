from pathlib import Path
import argparse, json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import CLASS_ORDER, RAW_DATA_DIR, RESULTS_DIR, WINDOW_SECONDS, HOP_SECONDS
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session


def load_session(raw_dir, session, window, hop):
    paths = trial_paths_for_session(raw_dir, session)
    if not paths:
        raise SystemExit(f"No trials found for {session}")
    rows = gather_windows(paths, window_seconds=window, hop_seconds=hop)
    for r in rows:
        r["_session"] = session
    print(f"{session}: {len(paths)} trials, {len(rows)} windows")
    return rows


def feature_names(rows):
    return sorted(rows[0]["features"])


def xy(rows, names):
    X = np.asarray([[float(r["features"][n]) for n in names] for r in rows], dtype=float)
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def score(y, pred):
    return (
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
    )


def trial_score(rows, probs, classes):
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
    return score(np.asarray(yt, dtype=object), np.asarray(yp, dtype=object))


def fit_test(train_rows, test_rows, names, seed):
    Xtr, ytr = xy(train_rows, names)
    Xte, yte = xy(test_rows, names)
    rf = make_rf(seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)
    wa, wf = score(yte, pred)
    ta, tf = trial_score(test_rows, prob, rf.classes_)
    return {
        "window_accuracy": wa,
        "window_macro_f1": wf,
        "trial_accuracy": ta,
        "trial_macro_f1": tf,
        "train_windows": len(train_rows),
        "test_windows": len(test_rows),
    }


def standard_cv(rows, names, seed):
    out = []
    print("\n=== S3 STANDARD TRIAL-NUMBER 5-FOLD CV ===")
    for fold in range(1, 6):
        test = [r for r in rows if int(trial_number_from_row(r)) == fold]
        train = [r for r in rows if int(trial_number_from_row(r)) != fold]
        m = fit_test(train, test, names, seed + fold)
        m["fold"] = fold
        out.append(m)
        print(f"Fold {fold}: window={m['window_accuracy']:.3f} F1={m['window_macro_f1']:.3f} trial={m['trial_accuracy']:.3f}")
    return out


def randomized_cv(rows, names, seed):
    rng = np.random.default_rng(seed)
    assignment = {}
    for zone in CLASS_ORDER:
        trials = np.arange(1, 6)
        rng.shuffle(trials)
        for fold, trial in enumerate(trials, start=1):
            assignment[(zone, int(trial))] = fold

    out = []
    print("\n=== S3 RANDOMIZED WHOLE-TRIAL 5-FOLD CV ===")
    for fold in range(1, 6):
        test = [
            r for r in rows
            if assignment[(str(r["zone"]), int(trial_number_from_row(r)))] == fold
        ]
        test_ids = {id(r) for r in test}
        train = [r for r in rows if id(r) not in test_ids]
        m = fit_test(train, test, names, seed + 100 + fold)
        m["fold"] = fold
        out.append(m)
        print(f"Fold {fold}: window={m['window_accuracy']:.3f} F1={m['window_macro_f1']:.3f} trial={m['trial_accuracy']:.3f}")
    return out


def n_repeat(rows, names, seed):
    n1 = [r for r in rows if int(trial_number_from_row(r)) == 1]
    n5 = [r for r in rows if int(trial_number_from_row(r)) == 5]

    print("\n=== S3 NORTH-ONLY REPEATABILITY ===")
    a = fit_test(n1, n5, names, seed + 201)
    b = fit_test(n5, n1, names, seed + 202)
    print(f"N1 -> N5: window={a['window_accuracy']:.3f} F1={a['window_macro_f1']:.3f} trial={a['trial_accuracy']:.3f}")
    print(f"N5 -> N1: window={b['window_accuracy']:.3f} F1={b['window_macro_f1']:.3f} trial={b['trial_accuracy']:.3f}")
    return {"N1_to_N5": a, "N5_to_N1": b}


def direct_transfer(source_rows, target_rows, names, source, seed):
    m = fit_test(source_rows, target_rows, names, seed)
    print(f"{source} -> S3: window={m['window_accuracy']:.3f} F1={m['window_macro_f1']:.3f} trial={m['trial_accuracy']:.3f}")
    return m


def summary_cv(rows):
    a = np.asarray([r["window_accuracy"] for r in rows], dtype=float)
    f = np.asarray([r["window_macro_f1"] for r in rows], dtype=float)
    t = np.asarray([r["trial_accuracy"] for r in rows], dtype=float)
    return {
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

    s1 = load_session(args.raw_dir, "S1", args.window, args.hop)
    s2 = load_session(args.raw_dir, "S2", args.window, args.hop)
    s3 = load_session(args.raw_dir, "S3", args.window, args.hop)

    names = feature_names(s1)
    if feature_names(s2) != names or feature_names(s3) != names:
        raise RuntimeError("Feature mismatch between sessions")

    standard = standard_cv(s3, names, args.seed)
    randomized = randomized_cv(s3, names, args.seed)
    north = n_repeat(s3, names, args.seed)

    print("\n=== DIRECT TRANSFER TO S3 ===")
    s1s3 = direct_transfer(s1, s3, names, "S1", args.seed + 301)
    s2s3 = direct_transfer(s2, s3, names, "S2", args.seed + 302)

    std_sum = summary_cv(standard)
    rnd_sum = summary_cv(randomized)

    print("\n" + "=" * 72)
    print("S3 DIAGNOSTIC SUMMARY")
    print("=" * 72)
    print(f"S3 standard CV   : window={std_sum['window_accuracy_mean']:.3f} ± {std_sum['window_accuracy_std']:.3f} | trial={std_sum['trial_accuracy_mean']:.3f}")
    print(f"S3 randomized CV : window={rnd_sum['window_accuracy_mean']:.3f} ± {rnd_sum['window_accuracy_std']:.3f} | trial={rnd_sum['trial_accuracy_mean']:.3f}")
    print(f"S3 N1 -> N5      : window={north['N1_to_N5']['window_accuracy']:.3f} | trial={north['N1_to_N5']['trial_accuracy']:.3f}")
    print(f"S3 N5 -> N1      : window={north['N5_to_N1']['window_accuracy']:.3f} | trial={north['N5_to_N1']['trial_accuracy']:.3f}")
    print(f"S1 -> S3         : window={s1s3['window_accuracy']:.3f} | trial={s1s3['trial_accuracy']:.3f}")
    print(f"S2 -> S3         : window={s2s3['window_accuracy']:.3f} | trial={s2s3['trial_accuracy']:.3f}")

    out_dir = RESULTS_DIR / "s3_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "s3_standard_cv": std_sum,
                "s3_randomized_cv": rnd_sum,
                "s3_north_repeatability": north,
                "s1_to_s3": s1s3,
                "s2_to_s3": s2s3,
                "note": "S3 is development data after inspection of the untouched S1+S2->S3 result.",
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
