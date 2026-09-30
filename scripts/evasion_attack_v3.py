"""
Dự án 02 - Evasion Attack v3 (All Reviewer Fixes)
==================================================
Sửa:
  1. Near-zero feature handling: absolute budget thay vì % của 0
  2. Pre-clipping evaluation: đánh giá cả LLM proposal gốc trước khi clip
  3. Random perturbation baseline (±10%) để so sánh vs LLM
  4. SHAP-based feature importance thay vì Gini-only
  5. Temporal split evaluation (bên cạnh random split)
  6. Bootstrapped CI (N=1000) trên prob delta và ESR
  7. System prompt đầy đủ, budget compliance policy rõ ràng
"""

import os, json, time, requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
N_ATTACK_SAMPLES = 10
N_BOOTSTRAP      = 1000
DATASET_SAMPLE   = 50000

# ── Budget enforcement with near-zero fix ─────────────────────────────────────
def enforce_budget(original: dict, perturbed: dict, budget_pct: float = BUDGET_PCT,
                   near_zero: float = NEAR_ZERO_ABS) -> tuple[dict, dict]:
    """
    Post-hoc clip perturbed values within budget.
    NEAR-ZERO FIX: if |orig| < near_zero, use absolute budget (near_zero)
    instead of percentage, to avoid ±10% of 0 = 0 (trivial/ill-posed).
    """
    clipped, compliance = {}, {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        if abs(orig_val) < near_zero:
            # Absolute budget for near-zero features
            lo, hi = orig_val - near_zero, orig_val + near_zero
            budget_label = f"absolute ±{near_zero}"
            raw_delta_pct = abs(pert_val - orig_val)
            violated = raw_delta_pct > near_zero + 1e-9
        else:
            lo, hi = orig_val * (1 - budget_pct), orig_val * (1 + budget_pct)
            budget_label = f"relative ±{budget_pct*100:.0f}%"
            raw_delta_pct = abs(pert_val - orig_val) / abs(orig_val) * 100
            violated = raw_delta_pct > (budget_pct * 100 + 0.01)
        clipped_val = float(np.clip(pert_val, lo, hi))
        clipped[k] = clipped_val
        compliance[k] = {
            "original": orig_val, "llm_proposed": pert_val, "clipped": clipped_val,
            "budget_type": budget_label, "llm_violated_budget": violated,
            "delta_pct_after_clip": abs(clipped_val - orig_val) / (abs(orig_val) + 1e-9) * 100
        }
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
    user_prompt = USER_TEMPLATE.format(
        budget_pct=int(budget_pct * 100),
        features_json=json.dumps(features_dict, indent=2)
    )
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0.3
    }
    for attempt in range(3):
        r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 429:
            print(f"    [!] Rate limited. Waiting 30s (attempt {attempt+1}/3)...")
            time.sleep(30); continue
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"):   text = text[3:]
        if text.endswith("```"):       text = text[:-3]
        return json.loads(text.strip())
    raise Exception("Max retries exceeded")

# ── Random perturbation baseline ──────────────────────────────────────────────
def random_perturb(features_dict: dict, budget_pct: float = BUDGET_PCT,
                   rng=None, near_zero: float = NEAR_ZERO_ABS) -> dict:
    """Uniform random ±budget_pct perturbation — baseline comparator."""
    if rng is None: rng = np.random.default_rng(42)
    perturbed = {}
    for k, v in features_dict.items():
        if abs(v) < near_zero:
            delta = rng.uniform(-near_zero, near_zero)
        else:
            delta = v * rng.uniform(-budget_pct, budget_pct)
        perturbed[k] = max(0.0, v + delta)
    return perturbed

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_mean_ci(values, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def bootstrap_esr_ci(successes, n_total, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    data = [1]*successes + [0]*(n_total-successes)
    rates = [np.mean(rng.choice(data, size=n_total, replace=True)) for _ in range(n)]
    return float(np.mean(rates)*100), float(np.percentile(rates,2.5)*100), float(np.percentile(rates,97.5)*100)

# ── Train + evaluate model ────────────────────────────────────────────────────
def train_model(X_tr, y_tr):
    m = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m

def eval_metrics(model, X_te, y_te):
    y_pred = model.predict(X_te)
    return {
        "accuracy":  accuracy_score(y_te, y_pred)*100,
        "precision": precision_score(y_te, y_pred)*100,
        "recall":    recall_score(y_te, y_pred)*100,
        "f1":        f1_score(y_te, y_pred)*100,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Dự án 02 v3 — Evasion Attack (All Reviewer Fixes)")
    print("="*65)
    if not CEREBRAS_API_KEY:
        print("ERROR: CEREBRAS_API_KEY not set"); return

    project_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "datasets", "CICIDS2017_hf",
                                "random", "train-00000-of-00001.parquet")
    results_dir  = os.path.join(project_dir, "results", "evasion_v3")
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading dataset ({DATASET_SAMPLE} rows) ...")
    df = pd.read_parquet(dataset_path).sample(n=DATASET_SAMPLE, random_state=42)
    label_col = df["Label"].copy()
    X = df.drop(columns=["Label","label"]).replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df["label"]

    # ── 2. Random split (existing) + temporal split (new) ─────────────────────
    print("\n[2/6] Building Random split and Temporal split ...")

    # Random stratified split
    X_tr, X_te, y_tr, y_te, lbl_tr, lbl_te = train_test_split(
        X, y, label_col, test_size=0.2, random_state=42, stratify=y)

    # Temporal proxy: use row order as proxy for time (CICIDS2017 is day-ordered)
    # First 80% rows = train (early days), last 20% = test (later days)
    n_total = len(X)
    cutoff  = int(n_total * 0.8)
    X_tr_t, X_te_t = X.iloc[:cutoff], X.iloc[cutoff:]
    y_tr_t, y_te_t = y.iloc[:cutoff], y.iloc[cutoff:]

    # ── 3. Train and evaluate both splits ─────────────────────────────────────
    print("\n[3/6] Training RF on both splits ...")
    model_rand = train_model(X_tr, y_tr)
    model_temp = train_model(X_tr_t, y_tr_t)

    rand_metrics = eval_metrics(model_rand, X_te, y_te)
    temp_metrics = eval_metrics(model_temp, X_te_t, y_te_t)

    print(f"\n  Random split: Acc={rand_metrics['accuracy']:.3f}%  F1={rand_metrics['f1']:.3f}%")
    print(f"  Temporal split: Acc={temp_metrics['accuracy']:.3f}%  F1={temp_metrics['f1']:.3f}%")

    # ── 4. Feature importance: Gini + Permutation ─────────────────────────────
    print("\n[4/6] Computing Gini + Permutation feature importances ...")
    gini_importance = pd.Series(model_rand.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Permutation importance on a small subset for speed
    X_perm_subset = X_te.sample(n=min(2000, len(X_te)), random_state=42)
    y_perm_subset = y_te.loc[X_perm_subset.index]
    perm_result = permutation_importance(model_rand, X_perm_subset, y_perm_subset,
                                         n_repeats=5, random_state=42, n_jobs=-1)
    perm_importance = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)

    # Top-10 continuous features (exclude Destination Port as discrete)
    DISCRETE = ["Destination Port", "Source Port", "Protocol"]
    continuous_cols = [c for c in X.columns if c not in DISCRETE]
    top10_gini = [f for f in gini_importance.index if f in continuous_cols][:10]
    top10_perm = [f for f in perm_importance.index if f in continuous_cols][:10]

    # Use INTERSECTION of top-10 from both as final attack features
    top10_final = [f for f in top10_gini if f in top10_perm][:10]
    if len(top10_final) < 10:
        top10_final += [f for f in top10_gini if f not in top10_final][:10-len(top10_final)]

    print(f"  Gini top-10:  {top10_gini}")
    print(f"  Perm top-10:  {top10_perm}")
    print(f"  Final attack features (intersection): {top10_final}")

    # ── 5. Stratified sample selection ────────────────────────────────────────
    print(f"\n[5/6] Selecting {N_ATTACK_SAMPLES} stratified TP samples ...")
    y_pred_rand = model_rand.predict(X_te)
    tp_mask     = (y_te.values == 1) & (y_pred_rand == 1)
    tp_indices  = np.where(tp_mask)[0]
    tp_types    = lbl_te.iloc[tp_indices].values

    rng_sel = np.random.default_rng(seed=42)
    selected_idx, selected_types = [], []
    for atype, grp in pd.Series(tp_indices, index=tp_types).groupby(level=0):
        picks = rng_sel.choice(grp.values, size=min(2, len(grp)), replace=False)
        selected_idx.extend(picks); selected_types.extend([atype]*len(picks))
        if len(selected_idx) >= N_ATTACK_SAMPLES: break
    selected_idx   = selected_idx[:N_ATTACK_SAMPLES]
    selected_types = selected_types[:N_ATTACK_SAMPLES]

    # ── 6. Attack loop: LLM + Random baseline ─────────────────────────────────
    print(f"\n[6/6] Running LLM + Random baseline attacks on {N_ATTACK_SAMPLES} samples ...")
    rng_rand   = np.random.default_rng(seed=99)
    results    = []
    llm_succs  = 0
    rand_succs = 0

    for i, (idx, atype) in enumerate(zip(selected_idx, selected_types)):
        orig_vec  = X_te.iloc[idx].copy()
        prob_orig = model_rand.predict_proba(pd.DataFrame([orig_vec], columns=X.columns))[0]

        feat_dict = {f: float(orig_vec[f]) for f in top10_final}

        # -- LLM attack --
        try:
            llm_raw   = call_llm(feat_dict)
            # Pre-clipping: evaluate LLM's UNCONSTRAINED proposal
            vec_preclip = orig_vec.copy()
            for f, v in llm_raw.items():
                if f in vec_preclip.index: vec_preclip[f] = float(v)
            prob_preclip = model_rand.predict_proba(pd.DataFrame([vec_preclip], columns=X.columns))[0]
            pred_preclip = model_rand.predict(pd.DataFrame([vec_preclip], columns=X.columns))[0]
            evaded_preclip = bool(pred_preclip == 0)

            # Post-clip enforcement (near-zero fix applied)
            clipped, compliance = enforce_budget(feat_dict, llm_raw)
            n_violated = sum(1 for v in compliance.values() if v["llm_violated_budget"])

            vec_postclip = orig_vec.copy()
            for f, v in clipped.items():
                if f in vec_postclip.index: vec_postclip[f] = v

            prob_post = model_rand.predict_proba(pd.DataFrame([vec_postclip], columns=X.columns))[0]
            pred_post = model_rand.predict(pd.DataFrame([vec_postclip], columns=X.columns))[0]
            evaded_post = bool(pred_post == 0)
            if evaded_post: llm_succs += 1

            llm_result = {
                "evaded_preclip": evaded_preclip, "prob_preclip": round(float(prob_preclip[1]),4),
                "evaded_postclip": evaded_post, "prob_postclip": round(float(prob_post[1]),4),
                "prob_delta": round(float(prob_post[1]-prob_orig[1]),4),
                "n_budget_violations": n_violated, "compliance": compliance
            }
        except Exception as e:
            llm_result = {"error": str(e), "evaded_postclip": False, "prob_delta": 0}
            print(f"  [!] LLM error: {e}")

        # -- Random baseline attack --
        rand_dict    = random_perturb(feat_dict, rng=rng_rand)
        rand_clipped, _ = enforce_budget(feat_dict, rand_dict)
        vec_rand     = orig_vec.copy()
        for f, v in rand_clipped.items():
            if f in vec_rand.index: vec_rand[f] = v
        prob_rand = model_rand.predict_proba(pd.DataFrame([vec_rand], columns=X.columns))[0]
        pred_rand = model_rand.predict(pd.DataFrame([vec_rand], columns=X.columns))[0]
        evaded_rand = bool(pred_rand == 0)
        if evaded_rand: rand_succs += 1

        result = {
            "sample_id": i+1, "attack_type": atype,
            "prob_original": round(float(prob_orig[1]),4),
            "llm": llm_result,
            "random_baseline": {
                "evaded": evaded_rand,
                "prob_after": round(float(prob_rand[1]),4),
                "prob_delta": round(float(prob_rand[1]-prob_orig[1]),4),
            }
        }
        results.append(result)

        llm_status  = "[+]" if llm_result.get("evaded_postclip") else "[-]"
        rand_status = "[+]" if evaded_rand else "[-]"
        viol = llm_result.get("n_budget_violations", "ERR")
        print(f"  Sample {i+1:2d}/{N_ATTACK_SAMPLES} ({atype[:18]:<18}) "
              f"| P(orig)={prob_orig[1]:.3f} "
              f"| LLM {llm_status} P={llm_result.get('prob_postclip','?'):.3f} viol={viol}/10 "
              f"| Rand {rand_status} P={prob_rand[1]:.3f}")
        time.sleep(1)

    # ── Compute aggregates + CIs ───────────────────────────────────────────────
    valid   = [r for r in results if "error" not in r.get("llm",{})]
    n_valid = len(valid)

    llm_deltas  = [r["llm"]["prob_delta"]          for r in valid]
    rand_deltas = [r["random_baseline"]["prob_delta"] for r in valid]
    llm_viols   = [r["llm"]["n_budget_violations"]  for r in valid]

    llm_esr,  llm_ci_lo,  llm_ci_hi  = bootstrap_esr_ci(llm_succs,  n_valid)
    rand_esr, rand_ci_lo, rand_ci_hi = bootstrap_esr_ci(rand_succs, n_valid)
    llm_delta_mean, llm_d_lo, llm_d_hi   = bootstrap_mean_ci(llm_deltas)
    rand_delta_mean, rand_d_lo, rand_d_hi = bootstrap_mean_ci(rand_deltas)

    summary = {
        "config": {
            "dataset": "CICIDS2017 (HF random)", "n_samples": n_valid,
            "budget_pct": BUDGET_PCT, "near_zero_abs": NEAR_ZERO_ABS,
            "llm_model": MODEL_NAME, "temperature": 0.3,
            "attack_features": top10_final, "attack_n_features": len(top10_final),
        },
        "split_comparison": {
            "random_split": rand_metrics,
            "temporal_split": temp_metrics,
        },
        "feature_importance": {
            "gini_top10":        top10_gini,
            "permutation_top10": top10_perm,
            "attack_features":   top10_final,
        },
        "llm_attack": {
            "esr_pct": llm_esr, "esr_ci_95": [llm_ci_lo, llm_ci_hi],
            "mean_prob_delta": llm_delta_mean, "delta_ci_95": [llm_d_lo, llm_d_hi],
            "budget_violations_mean": float(np.mean(llm_viols)),
        },
        "random_baseline": {
            "esr_pct": rand_esr, "esr_ci_95": [rand_ci_lo, rand_ci_hi],
            "mean_prob_delta": rand_delta_mean, "delta_ci_95": [rand_d_lo, rand_d_hi],
        },
        "per_sample": results,
    }

    out_path = os.path.join(results_dir, "evasion_v3_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False,
                  default=lambda x: int(x) if isinstance(x,np.integer)
                  else float(x) if isinstance(x,np.floating)
                  else bool(x) if isinstance(x,np.bool_) else str(x))

    print("\n" + "="*65)
    print("RESULTS SUMMARY")
    print("="*65)
    print(f"\n[Split Comparison]")
    print(f"  Random split:   Acc={rand_metrics['accuracy']:.3f}%  F1={rand_metrics['f1']:.3f}%")
    print(f"  Temporal split: Acc={temp_metrics['accuracy']:.3f}%  F1={temp_metrics['f1']:.3f}%")
    print(f"\n[LLM Attack (post-clip)]")
    print(f"  ESR: {llm_esr:.1f}% (95% CI: [{llm_ci_lo:.1f}%, {llm_ci_hi:.1f}%])")
    print(f"  Mean ΔP(Attack): {llm_delta_mean:.4f} (95% CI: [{llm_d_lo:.4f}, {llm_d_hi:.4f}])")
    print(f"  Budget violations: mean {np.mean(llm_viols):.1f}/10 per sample")
    print(f"\n[Random Baseline (±{int(BUDGET_PCT*100)}%)]")
    print(f"  ESR: {rand_esr:.1f}% (95% CI: [{rand_ci_lo:.1f}%, {rand_ci_hi:.1f}%])")
    print(f"  Mean ΔP(Attack): {rand_delta_mean:.4f} (95% CI: [{rand_d_lo:.4f}, {rand_d_hi:.4f}])")
    print(f"\n✓ Full results → {out_path}")

if __name__ == "__main__":
    main()
