"""Bước 1: Load HF temporal split, train RF, verify v6 index alignment."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (
    load_temporal_frames,
    prepare_xy,
    train_rf,
    top10_features,
    prob_attack,
    project_root,
    select_stratified_v6,
    load_v6_sample_indices,
)
from sklearn.metrics import accuracy_score


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("BƯỚC 1 — HF temporal load + RF train + v6 index check")
    print("=" * 65)

    print("\n[1/4] Loading lacg030175/CICIDS2017 temporal (test=FULL split) ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, y_te, lbl_te = prepare_xy(df_te)
    print(f"  train rows={len(X_tr)}, test rows={len(X_te)}, n_features={X_tr.shape[1]}")

    print("\n[2/4] Training RF ...")
    model = train_rf(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred) * 100
    print(f"  Test Accuracy: {acc:.3f}%")

    print("\n[3/4] Top-10 features ...")
    top10 = top10_features(model, X_tr, y_tr)
    print(f"  {top10}")

    print("\n[4/4] v6 index alignment ...")
    v6_path = os.path.join(root, "results", "evasion_v6", "evasion_v6_results.json")
    v6_ids = load_v6_sample_indices(v6_path)
    probs = model.predict_proba(X_te)[:, 1]
    tp_idx = np.where((y_te.values == 1) & (y_pred == 1))[0]
    reproduced = select_stratified_v6(tp_idx, probs)
    overlap = len(set(v6_ids) & set(reproduced))
    print(f"  v6 stored ids: {len(v6_ids)}, reproduced stratified: {len(reproduced)}")
    print(f"  overlap: {overlap}/{len(v6_ids)}")

    with open(v6_path, encoding="utf-8") as f:
        raw = json.load(f)["raw"]
    mism = 0
    for r in raw[:10]:
        idx = int(r["id"])
        p = round(prob_attack(model, X_te.iloc[idx], X_tr.columns), 4)
        if abs(p - r["p_orig"]) > 0.02:
            mism += 1
            print(f"  WARN id={idx} p_recalc={p} p_v6={r['p_orig']}")
    print(f"  p_orig mismatches in first 10: {mism}")

    meta = {
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
        "test_accuracy_pct": round(acc, 3),
        "top10_features": top10,
        "v6_overlap_reproduced": overlap,
        "v6_n": len(v6_ids),
    }
    out_path = os.path.join(out_dir, "step1_temporal_verify.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✓ Saved -> {out_path}")


if __name__ == "__main__":
    main()
