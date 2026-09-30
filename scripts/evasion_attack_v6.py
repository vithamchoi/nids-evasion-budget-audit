"""
Dự án 02 - Evasion Attack v6 (Primary: K=20 matched, N=107)
============================================================
- True temporal split (Mon-Thu train / Fri test)
- LLM iterative gray-box (Groq llama-3.1-8b-instant), max K=20, patience=3
- Greedy coordinate-wise iterative, K=20 queries
- Random open-loop baseline
- Stratified sample selection by RF confidence (High / Medium / Boundary)
- Output: results/evasion_v6/evasion_v6_results.json
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

BUDGET_PCT = 0.10
NEAR_ZERO_ABS = 1e-3
MAX_K = 20
PATIENCE = 3
N_ATTACK_SAMPLES = 107
N_PER_STRATUM = 50
N_BOOTSTRAP = 1000
DATASET_TR_SIZE = 50000
DATASET_TE_SIZE = 10000


def enforce_budget(original: dict, perturbed: dict, budget_pct: float = BUDGET_PCT,
                   near_zero: float = NEAR_ZERO_ABS):
    clipped, compliance = {}, {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        if abs(orig_val) < near_zero:
            lo, hi = orig_val - near_zero, orig_val + near_zero
            violated = abs(pert_val - orig_val) > near_zero + 1e-9
        else:
            lo, hi = orig_val * (1 - budget_pct), orig_val * (1 + budget_pct)
            violated = abs(pert_val - orig_val) / abs(orig_val) * 100 > (budget_pct * 100 + 0.01)
        clipped_val = float(np.clip(pert_val, lo, hi))
        clipped[k] = clipped_val
        compliance[k] = {"violated": violated}
    return clipped, compliance


SYSTEM_PROMPT = "You are an expert cybersecurity attacker AI. Output raw JSON only — no markdown."
USER_TEMPLATE = """Modify flow features so the NIDS outputs BENIGN (P(Attack) < 0.50).
Rules: each feature may change by at most ±{budget_pct}% of its original value; non-negative.
Return ONLY JSON with the same keys.

Features:
{features_json}
"""
FEEDBACK_TEMPLATE = """Previous proposal still classified ATTACK with P(Attack)={prob:.4f}.
Propose NEW values (same rules) to lower P(Attack) further.

Previous:
{prev_json}
"""


def call_llm(messages: list) -> dict:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.3}
    for attempt in range(4):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception:
            time.sleep(5)
    return {}


def prob_attack(model, vec, cols):
    return float(model.predict_proba(pd.DataFrame([vec], columns=cols))[0][1])


def random_perturb(features_dict, rng):
    out = {}
    for k, v in features_dict.items():
        if abs(v) < NEAR_ZERO_ABS:
            out[k] = max(0.0, v + rng.uniform(-NEAR_ZERO_ABS, NEAR_ZERO_ABS))
        else:
            out[k] = max(0.0, v * (1 + rng.uniform(-BUDGET_PCT, BUDGET_PCT)))
    return out


def greedy_one_step(vec, feat_dict, model, cols, feature_order, step_idx):
    """One coordinate query: try lo/hi for feature at step_idx % len(features)."""
    feats = list(feature_order)
    f = feats[step_idx % len(feats)]
    v = feat_dict[f]
    if abs(v) < NEAR_ZERO_ABS:
        lo, hi = max(0, v - NEAR_ZERO_ABS), v + NEAR_ZERO_ABS
    else:
        lo, hi = max(0, v * (1 - BUDGET_PCT)), v * (1 + BUDGET_PCT)
    current_p = prob_attack(model, vec, cols)
    best_p, best_val = current_p, v
    for cand in (lo, hi):
        trial = vec.copy()
        trial[f] = cand
        p = prob_attack(model, trial, cols)
        if p < best_p:
            best_p, best_val = p, cand
    vec[f] = best_val
    feat_dict[f] = best_val
    return vec, feat_dict, best_p


def llm_iterative(model, orig_vec, feat_dict, cols):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(
            budget_pct=int(BUDGET_PCT * 100),
            features_json=json.dumps(feat_dict, indent=2),
        ),
    })
    vec = orig_vec.copy()
    best_p = prob_attack(model, vec, cols)
    queries = 0
    no_improve = 0
    total_viol = 0

    for _ in range(MAX_K):
        raw = call_llm(messages)
        queries += 1
        if not raw:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
            continue
        clipped, comp = enforce_budget(feat_dict, raw)
        total_viol += sum(1 for v in comp.values() if v["violated"])
        for f, val in clipped.items():
            vec[f] = val
            feat_dict[f] = val
        p = prob_attack(model, vec, cols)
        if p < 0.5:
            return vec, queries, p, best_p, total_viol, True
        if p < best_p - 1e-6:
            best_p = p
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
        messages.append({"role": "assistant", "content": json.dumps(clipped)})
        messages.append({
            "role": "user",
            "content": FEEDBACK_TEMPLATE.format(prob=p, prev_json=json.dumps(clipped, indent=2)),
        })
    p_final = prob_attack(model, vec, cols)
    return vec, queries, p_final, best_p, total_viol, bool(p_final < 0.5)


def greedy_iterative(model, orig_vec, feat_dict, cols, feature_order):
    vec = orig_vec.copy()
    fd = dict(feat_dict)
    p0 = prob_attack(model, vec, cols)
    for step in range(MAX_K):
        vec, fd, _ = greedy_one_step(vec, fd, model, cols, feature_order, step)
        if prob_attack(model, vec, cols) < 0.5:
            break
    p_final = prob_attack(model, vec, cols)
    return vec, MAX_K, p_final, p0, bool(p_final < 0.5)


def clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def bootstrap_mean_ci(values, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stratum(p):
    if p > 0.8:
        return "High(>0.8)"
    if p > 0.6:
        return "Medium(0.6 - 0.8)"
    return "Boundary(0.5 - 0.6)"


def select_stratified(tp_indices, probs, labels, n_total=N_ATTACK_SAMPLES, n_per=N_PER_STRATUM):
    buckets = {"High(>0.8)": [], "Medium(0.6 - 0.8)": [], "Boundary(0.5 - 0.6)": []}
    for idx in tp_indices:
        p = probs[idx]
        if p <= 0.5:
            continue
        buckets[stratum(p)].append(idx)
    rng = np.random.default_rng(42)
    chosen = []
    for name in buckets:
        pool = buckets[name]
        if not pool:
            continue
        k = min(n_per, len(pool), n_total - len(chosen))
        if k > 0:
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    if len(chosen) < n_total:
        rest = [i for i in tp_indices if i not in chosen and probs[i] > 0.5]
        rng.shuffle(rest)
        for idx in rest:
            if len(chosen) >= n_total:
                break
            chosen.append(int(idx))
    return chosen[:n_total]


def aggregate_method(rows, key):
    n = len(rows)
    succ = sum(1 for r in rows if r[key]["evaded"])
    dps = [r[key]["delta"] for r in rows]
    q = [r[key]["queries"] for r in rows]
    lo, hi = clopper_pearson_ci(succ, n)
    md, mdlo, mdhi = bootstrap_mean_ci(dps)
    return {
        "ESR": round(succ / n * 100, 1),
        "CI": [round(lo * 100, 1), round(hi * 100, 1)],
        "mean_dP": round(md, 4),
        "dP_CI": [round(mdlo, 4), round(mdhi, 4)],
        "mean_queries": round(float(np.mean(q)), 1),
    }


def main():
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY missing in workspace .env")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_dir, "datasets", "CICIDS2017_hf", "temporal")
    results_dir = os.path.join(project_dir, "results", "evasion_v6")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("Dự án 02 v6 — N=107, K=20 matched, Groq iterative LLM")
    print("=" * 65)

    df_train = pd.read_parquet(os.path.join(dataset_dir, "train-00000-of-00001.parquet")).sample(
        n=DATASET_TR_SIZE, random_state=42
    )
    df_test = pd.read_parquet(os.path.join(dataset_dir, "test-00000-of-00001.parquet")).sample(
        n=DATASET_TE_SIZE, random_state=42
    )

    X_tr = df_train.drop(columns=["Label", "label"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    y_tr = df_train["label"]
    X_te = df_test.drop(columns=["Label", "label"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    y_te = df_test["label"]
    lbl_te = df_test["Label"]

    model = RandomForestClassifier(
        n_estimators=100, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"Test Accuracy: {accuracy_score(y_te, y_pred) * 100:.2f}%")

    gini_imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    X_perm = X_tr.sample(n=min(5000, len(X_tr)), random_state=42)
    perm = permutation_importance(model, X_perm, y_tr.loc[X_perm.index], n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False)
    discrete = {"Destination Port", "Source Port", "Protocol"}
    cont = [c for c in X_tr.columns if c not in discrete]
    top10 = [f for f in gini_imp.index if f in cont and f in perm_imp.index[:20]][:10]
    print("Top features:", top10)

    probs = model.predict_proba(X_te)[:, 1]
    tp_mask = (y_te.values == 1) & (y_pred == 1)
    tp_idx = np.where(tp_mask)[0]
    selected = select_stratified(tp_idx, probs, lbl_te.values)
    print(f"Selected {len(selected)} attack samples")

    rng_rand = np.random.default_rng(99)
    rows = []
    for i, idx in enumerate(selected):
        orig = X_te.iloc[idx].copy()
        p0 = prob_attack(model, orig, X_tr.columns)
        feat = {f: float(orig[f]) for f in top10}
        atype = str(lbl_te.iloc[idx])

        vec_l, q_l, p_l, _, viol_l, ev_l = llm_iterative(model, orig, feat, X_tr.columns)
        d_l = p_l - p0

        feat_g = {f: float(orig[f]) for f in top10}
        vec_g, q_g, p_g, _, ev_g = greedy_iterative(model, orig.copy(), feat_g, X_tr.columns, top10)
        d_g = p_g - p0

        feat_r = {f: float(orig[f]) for f in top10}
        r_raw = random_perturb(feat_r, rng_rand)
        r_clip, _ = enforce_budget(feat_r, r_raw)
        vec_r = orig.copy()
        for f, v in r_clip.items():
            vec_r[f] = v
        p_r = prob_attack(model, vec_r, X_tr.columns)
        ev_r = p_r < 0.5
        d_r = p_r - p0

        rows.append({
            "id": i + 1,
            "type": atype,
            "stratum": stratum(p0),
            "p_orig": round(p0, 4),
            "llm": {"evaded": ev_l, "prob": round(p_l, 4), "delta": round(d_l, 4), "queries": q_l, "viols": viol_l},
            "greedy": {"evaded": ev_g, "prob": round(p_g, 4), "delta": round(d_g, 4), "queries": q_g},
            "random": {"evaded": ev_r, "prob": round(p_r, 4), "delta": round(d_r, 4), "queries": 0},
        })
        print(
            f"  [{i+1:3d}/{len(selected)}] {atype[:12]:<12} {stratum(p0):<18} "
            f"P0={p0:.3f} LLM={'Y' if ev_l else 'N'} Greedy={'Y' if ev_g else 'N'}"
        )

    overall = {
        "N": len(rows),
        "llm": aggregate_method(rows, "llm"),
        "greedy": aggregate_method(rows, "greedy"),
        "random": aggregate_method(rows, "random"),
        "wilcoxon": {
            "LLM_vs_Random": float(stats.wilcoxon(
                [r["llm"]["delta"] for r in rows], [r["random"]["delta"] for r in rows]
            ).pvalue) if len(rows) else 1.0,
            "Greedy_vs_LLM": float(stats.wilcoxon(
                [r["greedy"]["delta"] for r in rows], [r["llm"]["delta"] for r in rows]
            ).pvalue) if len(rows) else 1.0,
        },
    }

    per_stratum = {}
    for sname in ["High(>0.8)", "Medium(0.6 - 0.8)", "Boundary(0.5 - 0.6)"]:
        sub = [r for r in rows if r["stratum"] == sname]
        if sub:
            per_stratum[sname] = {"N": len(sub), **{m: aggregate_method(sub, m) for m in ("llm", "greedy", "random")}}

    out = {
        "config": {
            "budget_pct": BUDGET_PCT,
            "budget_space": "raw feature values",
            "max_queries_K": MAX_K,
            "N_total": len(rows),
            "N_per_stratum_target": N_PER_STRATUM,
            "features": top10,
            "temporal_split": "HuggingFace pre-split by day label (Mon-Thu=train, Fri=test).",
            "llm_model": MODEL_NAME,
        },
        "overall": overall,
        "per_stratum": per_stratum,
        "raw": rows,
    }

    out_path = os.path.join(results_dir, "evasion_v6_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    print("\n" + "=" * 65)
    print(f"LLM ESR={overall['llm']['ESR']}% | Greedy={overall['greedy']['ESR']}% | Random={overall['random']['ESR']}%")
    print(f"✓ Saved -> {out_path}")


if __name__ == "__main__":
    main()
