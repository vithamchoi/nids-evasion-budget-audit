"""
Dự án 02 - Evasion Attack v4 (Final Live Run)
=============================================
Fixes:
  1. True Temporal Split: Load from `temporal/train` and `temporal/test`.
  2. N=50 samples for robust CI.
  3. Clopper-Pearson Exact CI for ESR.
  4. Greedy Heuristic baseline (coordinate-wise).
  5. Wilcoxon signed-rank test for paired prob_delta comparisons.
  6. Permutation Importance computed *strictly* on Train set.
  7. Explicit RF hyperparameters documented.
"""

import os, json, time, requests
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env"))

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_URL     = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME       = "llama3.1-8b"
BUDGET_PCT       = 0.10        # ±10% relative budget
NEAR_ZERO_ABS    = 1e-3        # absolute budget when |original| < this threshold
N_ATTACK_SAMPLES = 50          # Increased from 10 to 50
N_BOOTSTRAP      = 1000
DATASET_TR_SIZE  = 50000       # Subsample train for speed
DATASET_TE_SIZE  = 10000       # Subsample test

# ── Budget enforcement ────────────────────────────────────────────────────────
def enforce_budget(original: dict, perturbed: dict, budget_pct: float = BUDGET_PCT,
                   near_zero: float = NEAR_ZERO_ABS) -> tuple[dict, dict]:
    clipped, compliance = {}, {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        if abs(orig_val) < near_zero:
            lo, hi = orig_val - near_zero, orig_val + near_zero
            budget_label = f"abs ±{near_zero}"
            violated = abs(pert_val - orig_val) > near_zero + 1e-9
        else:
            lo, hi = orig_val * (1 - budget_pct), orig_val * (1 + budget_pct)
            budget_label = f"rel ±{budget_pct*100:.0f}%"
            violated = abs(pert_val - orig_val) / abs(orig_val) * 100 > (budget_pct * 100 + 0.01)
        clipped_val = float(np.clip(pert_val, lo, hi))
        clipped[k] = clipped_val
        compliance[k] = {"original": orig_val, "proposed": pert_val, "clipped": clipped_val,
                         "violated": violated, "type": budget_label}
    return clipped, compliance

# ── LLM call ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an expert cybersecurity attacker AI. Output raw JSON only — no markdown, no explanation."
USER_TEMPLATE = """\
You are a network adversary trying to evade a machine-learning NIDS classifier.

Task: Modify the network flow features below so the classifier outputs BENIGN instead of ATTACK.

STRICT RULES:
1. Modify each feature value by AT MOST ±{budget_pct}% of its original value.
2. All values must remain non-negative (traffic features cannot be negative).
3. Return ONLY a JSON object with exactly the same keys. No text outside the JSON.

Original Features (feature_name: original_value):
{features_json}
"""

def call_llm(features_dict: dict, budget_pct: float = BUDGET_PCT) -> dict:
    user_prompt = USER_TEMPLATE.format(budget_pct=int(budget_pct * 100), features_json=json.dumps(features_dict, indent=2))
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "temperature": 0.3}
    for attempt in range(4):
        try:
            r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 429:
                print(f"    [!] Rate limited. Waiting 30s (attempt {attempt+1}/4)...")
                time.sleep(30); continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"):   text = text[3:]
            if text.endswith("```"):       text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            time.sleep(5)
    raise Exception("Max retries exceeded or parsing failed")

# ── Baselines ─────────────────────────────────────────────────────────────────
def random_perturb(features_dict: dict, budget_pct: float = BUDGET_PCT, rng=None) -> dict:
    if rng is None: rng = np.random.default_rng(42)
    perturbed = {}
    for k, v in features_dict.items():
        if abs(v) < NEAR_ZERO_ABS: delta = rng.uniform(-NEAR_ZERO_ABS, NEAR_ZERO_ABS)
        else: delta = v * rng.uniform(-budget_pct, budget_pct)
        perturbed[k] = max(0.0, v + delta)
    return perturbed

def greedy_heuristic_perturb(orig_vec_full, feat_dict, model, all_cols, budget_pct=BUDGET_PCT):
    """
    Coordinate-wise greedy search:
    For each target feature, test +budget and -budget.
    Keep the modification that yields the lowest P(Attack).
    """
    current_vec = orig_vec_full.copy()
    perturbed = {}
    
    for f, v in feat_dict.items():
        if abs(v) < NEAR_ZERO_ABS: lo, hi = max(0, v - NEAR_ZERO_ABS), v + NEAR_ZERO_ABS
        else: lo, hi = max(0, v * (1 - budget_pct)), v * (1 + budget_pct)
        
        # Test lo
        vec_lo = current_vec.copy()
        vec_lo[f] = lo
        p_lo = model.predict_proba(pd.DataFrame([vec_lo], columns=all_cols))[0][1]
        
        # Test hi
        vec_hi = current_vec.copy()
        vec_hi[f] = hi
        p_hi = model.predict_proba(pd.DataFrame([vec_hi], columns=all_cols))[0][1]
        
        # Test orig
        p_orig = model.predict_proba(pd.DataFrame([current_vec], columns=all_cols))[0][1]
        
        best_p, best_val = p_orig, v
        if p_lo < best_p: best_p, best_val = p_lo, lo
        if p_hi < best_p: best_p, best_val = p_hi, hi
            
        current_vec[f] = best_val
        perturbed[f] = best_val
        
    return perturbed

# ── Statistical Testing ───────────────────────────────────────────────────────
def clopper_pearson_ci(k, n, alpha=0.05):
    """Exact binomial confidence interval."""
    if n == 0: return 0.0, 0.0
    lo = stats.beta.ppf(alpha/2, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(1-alpha/2, k+1, n-k) if k < n else 1.0
    return lo, hi

def bootstrap_mean_ci(values, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def compute_pvalue(deltas_A, deltas_B):
    """Wilcoxon signed-rank test for paired samples."""
    try:
        if np.allclose(deltas_A, deltas_B): return 1.0
        _, pval = stats.wilcoxon(deltas_A, deltas_B)
        return float(pval)
    except Exception:
        return 1.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Dự án 02 v4 — N=50, True Temporal Split, Greedy Baseline")
    print("="*65)
    
    project_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir  = os.path.join(project_dir, "datasets", "CICIDS2017_hf", "temporal")
    results_dir  = os.path.join(project_dir, "results", "evasion_v4")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load True Temporal Split
    print("\n[1/6] Loading True Temporal Dataset ...")
    df_train = pd.read_parquet(os.path.join(dataset_dir, "train-00000-of-00001.parquet")).sample(n=DATASET_TR_SIZE, random_state=42)
    df_test  = pd.read_parquet(os.path.join(dataset_dir, "test-00000-of-00001.parquet")).sample(n=DATASET_TE_SIZE, random_state=42)
    
    X_tr = df_train.drop(columns=["Label","label"]).replace([np.inf,-np.inf], np.nan).fillna(0)
    y_tr = df_train["label"]
    X_te = df_test.drop(columns=["Label","label"]).replace([np.inf,-np.inf], np.nan).fillna(0)
    y_te = df_test["label"]
    lbl_te = df_test["Label"]

    # 2. Train RF
    print("\n[2/6] Training Random Forest (n_estimators=100, min_samples_split=5) ...")
    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_te)
    metrics = {
        "accuracy":  accuracy_score(y_te, y_pred)*100,
        "precision": precision_score(y_te, y_pred, zero_division=0)*100,
        "recall":    recall_score(y_te, y_pred, zero_division=0)*100,
        "f1":        f1_score(y_te, y_pred, zero_division=0)*100,
    }
    print(f"  Test Accuracy: {metrics['accuracy']:.3f}%, F1: {metrics['f1']:.3f}%")

    # 3. Feature Importance (Strictly on Train set!)
    print("\n[3/6] Computing Feature Importances on Train set ...")
    gini_imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    
    X_perm_sub = X_tr.sample(n=min(5000, len(X_tr)), random_state=42)
    y_perm_sub = y_tr.loc[X_perm_sub.index]
    perm_res = permutation_importance(model, X_perm_sub, y_perm_sub, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm_res.importances_mean, index=X_tr.columns).sort_values(ascending=False)

    DISCRETE = ["Destination Port", "Source Port", "Protocol"]
    cont_cols = [c for c in X_tr.columns if c not in DISCRETE]
    top10_gini = [f for f in gini_imp.index if f in cont_cols][:10]
    top10_perm = [f for f in perm_imp.index if f in cont_cols][:10]
    
    top10_final = [f for f in top10_gini if f in top10_perm][:10]
    if len(top10_final) < 10:
        top10_final += [f for f in top10_gini if f not in top10_final][:10-len(top10_final)]
    print(f"  Final attack features: {top10_final}")

    # 4. Stratified sample selection (N=50)
    print(f"\n[4/6] Selecting {N_ATTACK_SAMPLES} stratified TP samples from Test set ...")
    tp_mask = (y_te.values == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    tp_types = lbl_te.iloc[tp_indices].values
    
    rng_sel = np.random.default_rng(seed=42)
    selected_idx, selected_types = [], []
    series_grps = pd.Series(tp_indices, index=tp_types).groupby(level=0)
    
    # Stratified sampling round robin
    while len(selected_idx) < N_ATTACK_SAMPLES:
        added = 0
        for atype, grp in series_grps:
            avail = [x for x in grp.values if x not in selected_idx]
            if avail:
                selected_idx.append(rng_sel.choice(avail))
                selected_types.append(atype)
                added += 1
                if len(selected_idx) == N_ATTACK_SAMPLES: break
        if added == 0: break # Exhausted

    # 5. Attack Loop
    print(f"\n[5/6] Executing Attacks (LLM vs Greedy vs Random) on {len(selected_idx)} samples ...")
    rng_rand = np.random.default_rng(seed=99)
    results = []

    for i, (idx, atype) in enumerate(zip(selected_idx, selected_types)):
        orig_vec = X_te.iloc[idx].copy()
        prob_orig = model.predict_proba(pd.DataFrame([orig_vec], columns=X_tr.columns))[0]
        feat_dict = {f: float(orig_vec[f]) for f in top10_final}

        # LLM
        try:
            llm_raw = call_llm(feat_dict)
            clipped, compliance = enforce_budget(feat_dict, llm_raw)
            viol = sum(1 for v in compliance.values() if v["violated"])
            vec_llm = orig_vec.copy()
            for f, v in clipped.items(): vec_llm[f] = v
            p_llm = model.predict_proba(pd.DataFrame([vec_llm], columns=X_tr.columns))[0][1]
            e_llm = bool(p_llm < 0.5)
            llm_res = {"evaded": e_llm, "prob": round(float(p_llm),4), "delta": round(float(p_llm-prob_orig[1]),4), "viols": viol}
        except Exception as e:
            llm_res = {"error": str(e), "evaded": False, "delta": 0.0}

        # Greedy
        greedy_raw = greedy_heuristic_perturb(orig_vec, feat_dict, model, X_tr.columns)
        g_clip, _ = enforce_budget(feat_dict, greedy_raw)
        vec_g = orig_vec.copy()
        for f, v in g_clip.items(): vec_g[f] = v
        p_g = model.predict_proba(pd.DataFrame([vec_g], columns=X_tr.columns))[0][1]
        e_g = bool(p_g < 0.5)
        greedy_res = {"evaded": e_g, "prob": round(float(p_g),4), "delta": round(float(p_g-prob_orig[1]),4)}

        # Random
        rand_raw = random_perturb(feat_dict, rng=rng_rand)
        r_clip, _ = enforce_budget(feat_dict, rand_raw)
        vec_r = orig_vec.copy()
        for f, v in r_clip.items(): vec_r[f] = v
        p_r = model.predict_proba(pd.DataFrame([vec_r], columns=X_tr.columns))[0][1]
        e_r = bool(p_r < 0.5)
        rand_res = {"evaded": e_r, "prob": round(float(p_r),4), "delta": round(float(p_r-prob_orig[1]),4)}

        results.append({
            "id": i+1, "type": atype, "p_orig": round(float(prob_orig[1]),4),
            "llm": llm_res, "greedy": greedy_res, "random": rand_res
        })
        
        print(f"  [{i+1:2d}/{len(selected_idx)}] {atype[:15]:<15} | P0={prob_orig[1]:.3f} "
              f"| LLM[{'+' if e_llm else '-'}]={p_llm:.3f} "
              f"| Greedy[{'+' if e_g else '-'}]={p_g:.3f} "
              f"| Rand[{'+' if e_r else '-'}]={p_r:.3f}")

    # 6. Metrics & Stats
    print("\n[6/6] Computing Strict CIs and Statistical Tests ...")
    valid = [r for r in results if "error" not in r["llm"]]
    n_v = len(valid)
    
    succ_l = sum(r["llm"]["evaded"] for r in valid)
    succ_g = sum(r["greedy"]["evaded"] for r in valid)
    succ_r = sum(r["random"]["evaded"] for r in valid)
    
    d_l = [r["llm"]["delta"] for r in valid]
    d_g = [r["greedy"]["delta"] for r in valid]
    d_r = [r["random"]["delta"] for r in valid]
    
    # Clopper-Pearson
    lo_l, hi_l = clopper_pearson_ci(succ_l, n_v)
    lo_g, hi_g = clopper_pearson_ci(succ_g, n_v)
    lo_r, hi_r = clopper_pearson_ci(succ_r, n_v)
    
    # Bootstrap mean delta
    md_l, mdlo_l, mdhi_l = bootstrap_mean_ci(d_l)
    md_g, mdlo_g, mdhi_g = bootstrap_mean_ci(d_g)
    md_r, mdlo_r, mdhi_r = bootstrap_mean_ci(d_r)
    
    # Wilcoxon Tests
    pval_L_vs_R = compute_pvalue(d_l, d_r)
    pval_L_vs_G = compute_pvalue(d_l, d_g)

    summary = {
        "config": {"N": n_v, "budget": BUDGET_PCT, "features": top10_final},
        "model": {"RF_params": "n_est=100, min_split=5", "test_acc": metrics["accuracy"]},
        "results": {
            "llm": {
                "ESR": round(succ_l/n_v*100,1), "ESR_CI": [round(lo_l*100,1), round(hi_l*100,1)],
                "Mean_Delta": md_l, "Delta_CI": [mdlo_l, mdhi_l],
                "Mean_Violations": float(np.mean([r["llm"].get("viols",0) for r in valid]))
            },
            "greedy": {
                "ESR": round(succ_g/n_v*100,1), "ESR_CI": [round(lo_g*100,1), round(hi_g*100,1)],
                "Mean_Delta": md_g, "Delta_CI": [mdlo_g, mdhi_g]
            },
            "random": {
                "ESR": round(succ_r/n_v*100,1), "ESR_CI": [round(lo_r*100,1), round(hi_r*100,1)],
                "Mean_Delta": md_r, "Delta_CI": [mdlo_r, mdhi_r]
            }
        },
        "stats": {
            "pval_LLM_vs_Random": pval_L_vs_R,
            "pval_LLM_vs_Greedy": pval_L_vs_G
        },
        "raw": valid
    }

    out_p = os.path.join(results_dir, "evasion_v4_results.json")
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "="*65)
    print("FINAL RESULTS (v4, N=50)")
    print("="*65)
    print(f"LLM Attack:    ESR={summary['results']['llm']['ESR']}% (CI: {summary['results']['llm']['ESR_CI']}) | ΔP={md_l:.4f}")
    print(f"Greedy Base:   ESR={summary['results']['greedy']['ESR']}% (CI: {summary['results']['greedy']['ESR_CI']}) | ΔP={md_g:.4f}")
    print(f"Random Base:   ESR={summary['results']['random']['ESR']}% (CI: {summary['results']['random']['ESR_CI']}) | ΔP={md_r:.4f}")
    print("\nStatistical Tests (Wilcoxon on ΔP):")
    print(f"  LLM vs Random: p = {pval_L_vs_R:.5f}")
    print(f"  LLM vs Greedy: p = {pval_L_vs_G:.5f}")
    print(f"\n✓ Saved -> {out_p}")

if __name__ == "__main__":
    main()
