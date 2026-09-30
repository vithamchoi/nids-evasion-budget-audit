"""Step 3: strong baselines on unified v8 cohort."""

import json
import math
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (  # noqa: E402
    BUDGET_PCT,
    MAX_K,
    NEAR_ZERO_ABS,
    aggregate_esr_dp,
    enforce_budget,
    load_temporal_frames,
    prepare_xy,
    prob_attack,
    project_root,
    select_stratified_v6,
    stratum,
    top10_features,
    train_rf,
)


def greedy_k20(model, orig_vec, feat_dict, cols, feature_order):
    vec = orig_vec.copy()
    fd = dict(feat_dict)
    p0 = prob_attack(model, vec, cols)
    for step in range(MAX_K):
        feat = feature_order[step % len(feature_order)]
        value = fd[feat]
        if abs(value) < NEAR_ZERO_ABS:
            lo, hi = max(0, value - NEAR_ZERO_ABS), value + NEAR_ZERO_ABS
        else:
            lo, hi = max(0, value * (1 - BUDGET_PCT)), value * (1 + BUDGET_PCT)
        best_p, best_val = prob_attack(model, vec, cols), value
        for cand in (lo, hi):
            trial = vec.copy()
            trial[feat] = cand
            p = prob_attack(model, trial, cols)
            if p < best_p:
                best_p, best_val = p, cand
        vec[feat] = best_val
        fd[feat] = best_val
        if best_p < 0.5:
            break
    pf = prob_attack(model, vec, cols)
    return vec, pf, pf - p0, pf < 0.5


def best_of_k_random(model, orig_vec, feat_dict, cols, rng):
    p0 = prob_attack(model, orig_vec, cols)
    best_vec, best_p = orig_vec.copy(), p0
    for _ in range(MAX_K):
        raw = {}
        for key, value in feat_dict.items():
            if abs(value) < NEAR_ZERO_ABS:
                raw[key] = max(0.0, value + rng.uniform(-NEAR_ZERO_ABS, NEAR_ZERO_ABS))
            else:
                raw[key] = max(0.0, value * (1 + rng.uniform(-BUDGET_PCT, BUDGET_PCT)))
        clip, _ = enforce_budget(feat_dict, raw)
        trial = orig_vec.copy()
        for feat, val in clip.items():
            trial[feat] = val
        p = prob_attack(model, trial, cols)
        if p < best_p:
            best_p, best_vec = p, trial
    pf = prob_attack(model, best_vec, cols)
    return best_vec, pf, pf - p0, pf < 0.5


def nes_attack(model, orig_vec, feat_dict, cols, rng, sigma_frac=0.05, n_perturb=8):
    p0 = prob_attack(model, orig_vec, cols)
    vec = orig_vec.copy()
    fd = dict(feat_dict)
    best_p = p0
    for _ in range(MAX_K):
        grads = {feat: 0.0 for feat in fd}
        for _ in range(n_perturb):
            noise = {}
            for feat, value in fd.items():
                scale = max(abs(value), NEAR_ZERO_ABS) * sigma_frac
                noise[feat] = rng.normal(0, scale)
            prop = {feat: fd[feat] + noise[feat] for feat in fd}
            clip, _ = enforce_budget(feat_dict, prop)
            trial = vec.copy()
            for feat, val in clip.items():
                trial[feat] = val
            p = prob_attack(model, trial, cols)
            sign = 1.0 if p < best_p else -1.0
            for feat in fd:
                grads[feat] += sign * noise[feat]
        step_prop = {}
        for feat, value in fd.items():
            grad = grads[feat] / n_perturb
            if abs(value) < NEAR_ZERO_ABS:
                step_prop[feat] = value - np.sign(grad) * NEAR_ZERO_ABS * 0.5
            else:
                step_prop[feat] = value - np.sign(grad) * abs(value) * BUDGET_PCT * 0.25
        clip, _ = enforce_budget(feat_dict, step_prop)
        for feat, val in clip.items():
            vec[feat] = val
            fd[feat] = val
        p = prob_attack(model, vec, cols)
        if p < best_p:
            best_p = p
        if p < 0.5:
            break
    pf = prob_attack(model, vec, cols)
    return vec, pf, pf - p0, pf < 0.5


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    os.makedirs(out_dir, exist_ok=True)
    n_cases = int(os.getenv("P02_LLM_CASES", "2000"))
    n_per = max(1, math.ceil(n_cases / 3))

    print("=" * 65)
    print(f"STEP 3 - Baselines: Greedy, NES, Best-of-K Random (N={n_cases}, K=20)")
    print("=" * 65)
    t0 = time.time()

    print("\n[1/5] Load data + train RF (50k train, full test) ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, y_te, lbl_te = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred) * 100
    print(f"  Test Accuracy: {acc:.3f}%")

    top10 = top10_features(model, X_tr, y_tr)
    print(f"  Top-10: {top10}")

    print(f"\n[2/5] Select N={n_cases} stratified TPs (seed=42) ...")
    probs = model.predict_proba(X_te)[:, 1]
    tp_idx = np.where((y_te.values == 1) & (y_pred == 1))[0]
    selected = select_stratified_v6(tp_idx, probs, n_total=n_cases, n_per=n_per, seed=42)
    print(f"  Selected {len(selected)} samples")

    idx_path = os.path.join(out_dir, "v8_sample_indices.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"indices": selected, "top10": top10}, f, indent=2)

    print("\n[3/5] Run attacks (CPU queries to RF) ...")
    rng = np.random.default_rng(99)
    rows = []
    for i, idx in enumerate(selected):
        orig = X_te.iloc[idx].copy()
        p0 = prob_attack(model, orig, X_tr.columns)
        feat = {feat: float(orig[feat]) for feat in top10}
        atype = str(lbl_te.iloc[idx]) if lbl_te is not None else "unknown"

        _, p_g, d_g, e_g = greedy_k20(model, orig, feat, X_tr.columns, top10)
        feat_n = {feat: float(orig[feat]) for feat in top10}
        _, p_n, d_n, e_n = nes_attack(model, orig.copy(), feat_n, X_tr.columns, rng)
        feat_r = {feat: float(orig[feat]) for feat in top10}
        _, p_r, d_r, e_r = best_of_k_random(model, orig.copy(), feat_r, X_tr.columns, rng)

        rows.append(
            {
                "id": int(idx),
                "type": atype,
                "stratum": stratum(p0),
                "p_orig": round(p0, 4),
                "greedy": {"evaded": e_g, "prob": round(p_g, 4), "delta": round(d_g, 4), "queries": MAX_K},
                "nes": {"evaded": e_n, "prob": round(p_n, 4), "delta": round(d_n, 4), "queries": MAX_K},
                "random_k": {"evaded": e_r, "prob": round(p_r, 4), "delta": round(d_r, 4), "queries": MAX_K},
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1:4d}/{len(selected)}] {atype[:10]:<10} p0={p0:.3f} "
                f"G={'Y' if e_g else 'N'} NES={'Y' if e_n else 'N'} Rk={'Y' if e_r else 'N'}"
            )

    print("\n[4/5] Aggregate ...")
    summary = {
        "config": {
            "N": len(rows),
            "K": MAX_K,
            "budget_pct": BUDGET_PCT,
            "features": top10,
            "test_accuracy_pct": round(acc, 3),
            "cohort": "v8_unified_hf_temporal",
        },
        "greedy": aggregate_esr_dp(rows, "greedy"),
        "nes": aggregate_esr_dp(rows, "nes"),
        "random_k": aggregate_esr_dp(rows, "random_k"),
        "wilcoxon": {
            "NES_vs_random_k": float(
                stats.wilcoxon([r["nes"]["delta"] for r in rows], [r["random_k"]["delta"] for r in rows]).pvalue
            ),
            "Greedy_vs_NES": float(
                stats.wilcoxon([r["greedy"]["delta"] for r in rows], [r["nes"]["delta"] for r in rows]).pvalue
            ),
        },
        "raw": rows,
    }

    out_path = os.path.join(out_dir, "step3_baselines_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[5/5] Results")
    print(f"  Greedy:   ESR={summary['greedy']['ESR']}% CI={summary['greedy']['CI']} dP={summary['greedy']['mean_dP']}")
    print(f"  NES:      ESR={summary['nes']['ESR']}% CI={summary['nes']['CI']} dP={summary['nes']['mean_dP']}")
    print(f"  Random K: ESR={summary['random_k']['ESR']}% CI={summary['random_k']['CI']} dP={summary['random_k']['mean_dP']}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
