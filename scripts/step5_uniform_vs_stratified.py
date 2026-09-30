"""Bước 5: Stratified vs uniform TP sampling (greedy only). Reuses top10 from step3."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (
    load_temporal_frames,
    prepare_xy,
    train_rf,
    prob_attack,
    project_root,
    select_stratified_v6,
    aggregate_esr_dp,
    enforce_budget,
    MAX_K,
    BUDGET_PCT,
    NEAR_ZERO_ABS,
)
from sklearn.metrics import accuracy_score


def greedy_k20(model, orig_vec, feat_dict, cols, feature_order):
    vec = orig_vec.copy()
    fd = dict(feat_dict)
    p0 = prob_attack(model, vec, cols)
    for step in range(MAX_K):
        f = feature_order[step % len(feature_order)]
        v = fd[f]
        if abs(v) < NEAR_ZERO_ABS:
            lo, hi = max(0, v - NEAR_ZERO_ABS), v + NEAR_ZERO_ABS
        else:
            lo, hi = max(0, v * (1 - BUDGET_PCT)), v * (1 + BUDGET_PCT)
        best_p, best_val = prob_attack(model, vec, cols), v
        for cand in (lo, hi):
            trial = vec.copy()
            trial[f] = cand
            p = prob_attack(model, trial, cols)
            if p < best_p:
                best_p, best_val = p, cand
        vec[f] = best_val
        fd[f] = best_val
        if best_p < 0.5:
            break
    pf = prob_attack(model, vec, cols)
    return pf, pf - p0, pf < 0.5


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    with open(os.path.join(out_dir, "v8_sample_indices.json"), encoding="utf-8") as f:
        top10 = json.load(f)["top10"]

    print("=" * 65)
    print("BƯỚC 5 — Uniform vs stratified (Greedy, N=50 each arm)")
    print("=" * 65)

    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, y_te, _ = prepare_xy(df_te)
    print("[1/3] Train RF ...")
    model = train_rf(X_tr, y_tr)
    y_pred = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    tp_idx = np.where((y_te.values == 1) & (y_pred == 1))[0]
    print(f"  Test acc={accuracy_score(y_te, y_pred)*100:.3f}% TPs={len(tp_idx)}")

    N_CMP = 50
    strat = select_stratified_v6(tp_idx, probs, n_total=N_CMP, n_per=25, seed=42)
    rng = np.random.default_rng(42)
    pool = [int(i) for i in tp_idx if probs[i] > 0.5]
    uniform = rng.choice(pool, size=min(N_CMP, len(pool)), replace=False).tolist()

    results = {"N_per_arm": N_CMP, "top10": top10}
    print("[2/3] Greedy attacks ...")
    for name, indices in [("stratified", strat), ("uniform", uniform)]:
        rows = []
        t0 = time.time()
        for j, idx in enumerate(indices):
            orig = X_te.iloc[idx].copy()
            feat = {f: float(orig[f]) for f in top10}
            p, d, ev = greedy_k20(model, orig, feat, X_tr.columns, top10)
            rows.append({"greedy": {"evaded": ev, "prob": round(p, 4), "delta": round(d, 4), "queries": MAX_K}})
            if (j + 1) % 10 == 0:
                print(f"    {name} {j+1}/{len(indices)}")
        results[name] = {"greedy": aggregate_esr_dp(rows, "greedy"), "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {name}: ESR={results[name]['greedy']['ESR']}% mean_dP={results[name]['greedy']['mean_dP']}")

    path = os.path.join(out_dir, "step5_sampling_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved -> {path}")


if __name__ == "__main__":
    main()
