"""Bước 2: Fit isotonic calibration on train; ECE/Brier + threshold sweep on attack-val."""

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (
    load_temporal_frames,
    prepare_xy,
    train_rf,
    project_root,
    clopper_pearson_ci,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


def ece(conf, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    details = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        mc, fp = float(conf[m].mean()), float(y[m].mean())
        gap = abs(mc - fp)
        total += (n / len(conf)) * gap
        details.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n, "mean_conf": round(mc, 4), "frac_pos": round(fp, 4), "gap": round(gap, 4)})
    return float(total), details


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("BƯỚC 2 — Isotonic calibration (RF on temporal train)")
    print("=" * 65)

    df_tr, _ = load_temporal_frames(te_subsample=None)
    X, y, _ = prepare_xy(df_tr)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n[1/3] Train RF on {len(X_fit)} rows ...")
    model = train_rf(X_fit, y_fit)
    p_cal_raw = model.predict_proba(X_cal)[:, 1]

    print("[2/3] Fit isotonic on calibration fold ...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)
    p_cal = iso.predict(p_cal_raw)

    ece_raw, bins_raw = ece(p_cal_raw, y_cal.values)
    ece_iso, bins_iso = ece(p_cal, y_cal.values)
    brier_raw = float(np.mean((p_cal_raw - y_cal.values) ** 2))
    brier_iso = float(np.mean((p_cal - y_cal.values) ** 2))

    print(f"  ECE raw: {ece_raw:.4f}  -> isotonic: {ece_iso:.4f}")
    print(f"  Brier raw: {brier_raw:.4f} -> isotonic: {brier_iso:.4f}")

    # attack-only subset on cal fold
    atk = y_cal.values == 1
    if atk.sum() > 0:
        ece_a_raw, _ = ece(p_cal_raw[atk], y_cal.values[atk])
        ece_a_iso, _ = ece(p_cal[atk], y_cal.values[atk])
        print(f"  ECE attack-only (n={atk.sum()}): raw={ece_a_raw:.4f} iso={ece_a_iso:.4f}")

    print("[3/3] Save artifacts ...")
    joblib.dump(model, os.path.join(out_dir, "rf_model.joblib"))
    joblib.dump(iso, os.path.join(out_dir, "isotonic.joblib"))
    joblib.dump(list(X.columns), os.path.join(out_dir, "feature_columns.joblib"))

    out = {
        "calibration_rows": len(X_cal),
        "ece_raw": round(ece_raw, 4),
        "ece_isotonic": round(ece_iso, 4),
        "brier_raw": round(brier_raw, 4),
        "brier_isotonic": round(brier_iso, 4),
        "bins_raw": bins_raw,
        "bins_isotonic": bins_iso,
    }
    path = os.path.join(out_dir, "step2_isotonic_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved -> {path}")


if __name__ == "__main__":
    main()
