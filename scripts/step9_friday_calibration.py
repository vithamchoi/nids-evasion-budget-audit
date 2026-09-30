"""Step 9: honest out-of-sample calibration on the Friday partition.

The published calibration (step 2) fitted isotonic regression on a 20% fold of
the 50k TRAINING rows and then measured expected calibration error on that same
fold. Two things follow: the numbers describe the training distribution, where
the detector is about 99.8% accurate, not the Friday partition where it is about
70.5%; and the post-calibration ECE of exactly 0.0000 is an in-sample artefact.

This script measures the quantity the paper actually needs: the calibration of
the posterior the attacker reads, on Friday, with the isotonic map fitted on
data the evaluation never sees. No API key is needed.

Outputs results/revision_v8/step9_friday_calibration.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (  # noqa: E402
    load_temporal_frames,
    prepare_xy,
    project_root,
    train_rf,
)

N_BINS = 10


def ece(conf, y, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    total, details = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        mc, fp = float(conf[m].mean()), float(y[m].mean())
        total += (n / len(conf)) * abs(mc - fp)
        details.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                        "mean_conf": round(mc, 4), "frac_pos": round(fp, 4),
                        "gap": round(abs(mc - fp), 4)})
    return float(total), details


def block(conf, y, label):
    e, bins = ece(conf, y)
    brier = float(np.mean((conf - y) ** 2))
    pred = (conf >= 0.5).astype(int)
    return {
        "label": label,
        "n": int(len(y)),
        "positive_rate_pct": round(100 * float(y.mean()), 2),
        "accuracy_pct": round(100 * float((pred == y).mean()), 3),
        "ece": round(e, 4),
        "brier": round(brier, 4),
        "bins": bins,
    }


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("STEP 9 - out-of-sample calibration on the Friday partition")
    print("=" * 68)

    print("\n[1/4] Load temporal frames ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X, y, _ = prepare_xy(df_tr)
    X_te, y_te, _ = prepare_xy(df_te)

    # Fit the forest on 80% of train, the isotonic map on the remaining 20%.
    # Neither touches Friday, so every Friday number below is out-of-sample.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[2/4] Train RF on {len(X_fit)} rows, fit isotonic on {len(X_cal)} ...")
    model = train_rf(X_fit, y_fit)
    p_cal = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)

    print("[3/4] Score on Friday (never seen by either fit) ...")
    p_te_raw = model.predict_proba(X_te)[:, 1]
    p_te_iso = iso.predict(p_te_raw)
    yv = y_te.values

    out = {
        "note": ("ECE and Brier on the Friday partition, with the isotonic map "
                 "fitted only on a held-out fold of the training partition. "
                 "Compare against step2, which reported in-sample numbers on "
                 "the training distribution."),
        "n_train_fit": int(len(X_fit)),
        "n_isotonic_fit": int(len(X_cal)),
        "in_sample_reference": {
            "train_fold_raw": block(p_cal, y_cal.values, "train fold, raw"),
            "train_fold_isotonic_in_sample": block(
                iso.predict(p_cal), y_cal.values, "train fold, isotonic (in-sample)"),
        },
        "friday_out_of_sample": {
            "raw": block(p_te_raw, yv, "Friday, raw"),
            "isotonic": block(p_te_iso, yv, "Friday, isotonic (out-of-sample)"),
        },
    }

    # The quantity the attack actually consumes: the posterior on detected
    # attack flows, which is where the cohort is drawn from.
    tp = (yv == 1) & (p_te_raw >= 0.5)
    if tp.sum() > 0:
        out["friday_true_positives"] = {
            "n": int(tp.sum()),
            "raw": block(p_te_raw[tp], yv[tp], "Friday TPs, raw"),
            "isotonic": block(p_te_iso[tp], yv[tp], "Friday TPs, isotonic"),
            "stratum_counts": {
                "High(>0.8)": int(((p_te_raw > 0.8) & tp).sum()),
                "Medium(0.6 - 0.8)": int(((p_te_raw > 0.6) & (p_te_raw <= 0.8) & tp).sum()),
                "Boundary(0.5 - 0.6)": int(((p_te_raw >= 0.5) & (p_te_raw <= 0.6) & tp).sum()),
            },
        }

    path = os.path.join(out_dir, "step9_friday_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n[4/4] Result")
    fr = out["friday_out_of_sample"]
    ins = out["in_sample_reference"]
    print(f"  train fold  raw      : acc={ins['train_fold_raw']['accuracy_pct']:.2f}%  "
          f"ECE={ins['train_fold_raw']['ece']:.4f}  Brier={ins['train_fold_raw']['brier']:.4f}")
    print(f"  Friday      raw      : acc={fr['raw']['accuracy_pct']:.2f}%  "
          f"ECE={fr['raw']['ece']:.4f}  Brier={fr['raw']['brier']:.4f}")
    print(f"  Friday      isotonic : acc={fr['isotonic']['accuracy_pct']:.2f}%  "
          f"ECE={fr['isotonic']['ece']:.4f}  Brier={fr['isotonic']['brier']:.4f}")
    if "friday_true_positives" in out:
        s = out["friday_true_positives"]["stratum_counts"]
        print(f"\n  Friday true-positive strata (the population the cohort samples):")
        for k, v in s.items():
            print(f"    {k:20s} {v}")
        print("  These counts are what a prevalence-weighted evasion rate needs.")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
