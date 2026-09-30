"""
Dự án 02 - Evasion Attack v5 (Final Iterative & Cross-Model)
============================================================
Fixes:
  1. Cross-Model: Evaluates on both Random Forest and XGBoost.
  2. Iterative Agent Loop: LLM receives feedback and adjusts up to 3 times.
  3. Problem-Space Constraints: Packet lengths and counts can only INCREASE (padding/dummy).
  4. Sensitivity Analysis: Tests budgets ±10% and ±25%.
"""

import os, json, time, requests
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env"))

# Using fast model for iterative calls
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_URL     = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME       = "llama3.1-8b"
N_ATTACK_SAMPLES = 50
DATASET_TR_SIZE  = 50000
DATASET_TE_SIZE  = 10000

# ── Budget & Constraints ──────────────────────────────────────────────────────
def enforce_budget(original: dict, perturbed: dict, budget_pct: float) -> tuple[dict, dict]:
    clipped, compliance = {}, {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        
        # Problem-space constraints: Packets and sizes can generally only increase (via padding or dummy injection)
        # We enforce that if the feature implies a count or length, it cannot go below original
        if any(x in k.lower() for x in ["length", "size", "count", "total"]):
            lo = orig_val
            hi = orig_val * (1 + budget_pct) if orig_val > 0 else 1.0
        else:
            lo = orig_val * (1 - budget_pct)
            hi = orig_val * (1 + budget_pct) if orig_val > 0 else 1.0
            
        if orig_val == 0: hi = max(hi, 1.0)
        lo = max(0.0, lo) # strictly non-negative
        
        clipped_val = float(np.clip(pert_val, lo, hi))
        violated = (pert_val < lo - 1e-5) or (pert_val > hi + 1e-5)
        
        clipped[k] = clipped_val
        compliance[k] = {"violated": violated}
    return clipped, compliance

# ── LLM Agent ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an expert cybersecurity attacker AI. Output raw JSON only — no markdown, no explanation."
USER_TEMPLATE = """\
You are a network adversary trying to evade a machine-learning NIDS classifier.
Modify the network flow features below so the classifier outputs BENIGN instead of ATTACK.

RULES:
1. Modify each feature value by AT MOST ±{budget_pct}% of its original value.
2. PROBLEM-SPACE CONSTRAINT: Features representing Packet Lengths, Sizes, or Total Counts can ONLY INCREASE (simulating payload padding or dummy packet injection). They CANNOT decrease.
3. Return ONLY a JSON object mapping feature names to their new numerical values.

Original Features:
{features_json}
"""

FEEDBACK_TEMPLATE = """\
Your previous proposed features FAILED to evade the classifier.
The classifier still predicts ATTACK with a confidence probability of {prob:.2f} (Needs to be < 0.50).
Please propose a NEW set of features following the same rules to reduce the confidence further.

Previous Proposal:
{prev_json}
"""

def call_llm(messages: list) -> dict:
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.3}
    for attempt in range(3):
        try:
            r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 429: time.sleep(10); continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"):   text = text[3:]
            if text.endswith("```"):       text = text[:-3]
            return json.loads(text.strip())
        except Exception:
            time.sleep(2)
    return {}

def agent_loop(model, orig_vec, feat_dict, all_cols, budget_pct, max_turns=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user", 
        "content": USER_TEMPLATE.format(budget_pct=int(budget_pct*100), features_json=json.dumps(feat_dict, indent=2))
    })
    
    best_p = 1.0; best_vec = orig_vec.copy(); viol_count = 0
    t0 = time.time()
    
    for turn in range(max_turns):
        llm_raw = call_llm(messages)
        if not llm_raw: break
        
        clipped, compliance = enforce_budget(feat_dict, llm_raw, budget_pct)
        viol_count += sum(1 for v in compliance.values() if v["violated"])
        
        vec_llm = orig_vec.copy()
        for f, v in clipped.items(): vec_llm[f] = v
        
        p_llm = model.predict_proba(pd.DataFrame([vec_llm], columns=all_cols))[0][1]
        
        if p_llm < best_p:
            best_p = p_llm
            best_vec = vec_llm
            
        if p_llm < 0.5: # Evaded successfully!
            break
            
        # Provide feedback
        messages.append({"role": "assistant", "content": json.dumps(clipped)})
        messages.append({
            "role": "user",
            "content": FEEDBACK_TEMPLATE.format(prob=p_llm, prev_json=json.dumps(clipped, indent=2))
        })
        
    t_elap = time.time() - t0
    return {"evaded": bool(best_p < 0.5), "prob": round(float(best_p), 4), "viols": viol_count, "time": round(t_elap,2)}

# ── Baselines ─────────────────────────────────────────────────────────────────
def greedy_heuristic(model, orig_vec, feat_dict, all_cols, budget_pct):
    t0 = time.time()
    current_vec = orig_vec.copy()
    for f, v in feat_dict.items():
        if any(x in f.lower() for x in ["length", "size", "count", "total"]):
            lo = v; hi = v * (1 + budget_pct) if v > 0 else 1.0
        else:
            lo = max(0.0, v * (1 - budget_pct)); hi = v * (1 + budget_pct) if v > 0 else 1.0
        
        vec_lo = current_vec.copy(); vec_lo[f] = lo
        p_lo = model.predict_proba(pd.DataFrame([vec_lo], columns=all_cols))[0][1]
        vec_hi = current_vec.copy(); vec_hi[f] = hi
        p_hi = model.predict_proba(pd.DataFrame([vec_hi], columns=all_cols))[0][1]
        p_orig = model.predict_proba(pd.DataFrame([current_vec], columns=all_cols))[0][1]
        
        best_p, best_val = p_orig, v
        if p_lo < best_p: best_p, best_val = p_lo, lo
        if p_hi < best_p: best_p, best_val = p_hi, hi
        current_vec[f] = best_val
        
    p_final = model.predict_proba(pd.DataFrame([current_vec], columns=all_cols))[0][1]
    t_elap = time.time() - t0
    return {"evaded": bool(p_final < 0.5), "prob": round(float(p_final), 4), "time": round(t_elap,2)}

# ── Statistics ────────────────────────────────────────────────────────────────
def clopper_pearson_ci(k, n):
    if n == 0: return 0.0, 0.0
    lo = stats.beta.ppf(0.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k+1, n-k) if k < n else 1.0
    return lo, hi

def bootstrap_mean_ci(values, n=1000):
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Dự án 02 v5 — Iterative LLM, Cross-Model, Problem-Space Constraints")
    print("="*65)
    
    project_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir  = os.path.join(project_dir, "datasets", "CICIDS2017_hf", "temporal")
    os.makedirs(os.path.join(project_dir, "results", "evasion_v5"), exist_ok=True)

    print("\n[1/4] Loading True Temporal Dataset ...")
    df_train = pd.read_parquet(os.path.join(dataset_dir, "train-00000-of-00001.parquet")).sample(n=DATASET_TR_SIZE, random_state=42)
    df_test  = pd.read_parquet(os.path.join(dataset_dir, "test-00000-of-00001.parquet")).sample(n=DATASET_TE_SIZE, random_state=42)
    X_tr = df_train.drop(columns=["Label","label"]).replace([np.inf,-np.inf], np.nan).fillna(0)
    y_tr = df_train["label"]
    X_te = df_test.drop(columns=["Label","label"]).replace([np.inf,-np.inf], np.nan).fillna(0)
    y_te = df_test["label"]
    lbl_te = df_test["Label"]

    print("\n[2/4] Training Models (RF & XGBoost) ...")
    models = {
        "RF": RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    for m_name, m in models.items():
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f"  {m_name} Test Accuracy: {acc*100:.2f}%")

    # Features (Static Top 10)
    top10_final = ['Bwd Packet Length Std', 'Average Packet Size', 'Packet Length Variance', 'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Flow Bytes/s', 'Max Packet Length']

    print(f"\n[3/4] Selecting {N_ATTACK_SAMPLES} stratified TP samples ...")
    tp_mask = (y_te.values == 1)
    tp_indices = np.where(tp_mask)[0]
    tp_types = lbl_te.iloc[tp_indices].values
    rng_sel = np.random.default_rng(seed=42)
    
    selected_idx, selected_types = [], []
    for atype, grp in pd.Series(tp_indices, index=tp_types).groupby(level=0):
        avail = list(grp.values)
        if avail:
            selected_idx.append(rng_sel.choice(avail))
            selected_types.append(atype)
            if len(selected_idx) >= N_ATTACK_SAMPLES: break
    while len(selected_idx) < N_ATTACK_SAMPLES:
        selected_idx.append(rng_sel.choice(tp_indices))
        selected_types.append(lbl_te.iloc[selected_idx[-1]])

    print("\n[4/4] Executing Iterative Agent Loop ...")
    all_results = {}
    
    for m_name, m in models.items():
        all_results[m_name] = {}
        for budget in [0.10, 0.25]:
            print(f"\n>>> Model: {m_name} | Budget: {budget*100:.0f}%")
            results = []
            
            for i, (idx, atype) in enumerate(zip(selected_idx, selected_types)):
                orig_vec = X_te.iloc[idx].copy()
                prob_orig = float(m.predict_proba(pd.DataFrame([orig_vec], columns=X_tr.columns))[0][1])
                feat_dict = {f: float(orig_vec[f]) for f in top10_final}
                
                # If model doesn't predict attack, skip
                if prob_orig < 0.5: continue
                
                llm_res = agent_loop(m, orig_vec, feat_dict, X_tr.columns, budget)
                g_res   = greedy_heuristic(m, orig_vec, feat_dict, X_tr.columns, budget)
                
                llm_res["delta"] = prob_orig - llm_res["prob"]
                g_res["delta"]   = prob_orig - g_res["prob"]
                
                results.append({
                    "id": i+1, "type": atype, "p_orig": prob_orig,
                    "llm": llm_res, "greedy": g_res
                })
                print(f"  [{i+1}/{len(selected_idx)}] {atype[:15]:<15} | P0={prob_orig:.3f} | LLM={llm_res['prob']:.3f} ({llm_res['time']}s) | Greedy={g_res['prob']:.3f} ({g_res['time']}s)")
                
            n_v = len(results)
            if n_v > 0:
                s_l = sum(r["llm"]["evaded"] for r in results)
                s_g = sum(r["greedy"]["evaded"] for r in results)
                md_l, mdlo_l, mdhi_l = bootstrap_mean_ci([r["llm"]["delta"] for r in results])
                md_g, mdlo_g, mdhi_g = bootstrap_mean_ci([r["greedy"]["delta"] for r in results])
                lo_l, hi_l = clopper_pearson_ci(s_l, n_v)
                lo_g, hi_g = clopper_pearson_ci(s_g, n_v)
                
                all_results[m_name][f"budget_{budget}"] = {
                    "N": n_v,
                    "llm": {"ESR": s_l/n_v, "ESR_CI": [lo_l, hi_l], "Mean_Delta": md_l, "Time_Avg": np.mean([r["llm"]["time"] for r in results])},
                    "greedy": {"ESR": s_g/n_v, "ESR_CI": [lo_g, hi_g], "Mean_Delta": md_g, "Time_Avg": np.mean([r["greedy"]["time"] for r in results])}
                }

    out_p = os.path.join(project_dir, "results", "evasion_v5", "evasion_v5_results.json")
    with open(out_p, "w") as f: json.dump(all_results, f, indent=4)
    print(f"\n✓ Saved -> {out_p}")

if __name__ == "__main__":
    main()
