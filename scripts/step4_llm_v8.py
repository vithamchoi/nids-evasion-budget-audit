"""Bước 4: LLM iterative attack on v8 cohort (same indices as step3)."""

import json
import os
import sys
import time

import joblib
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (
    load_temporal_frames,
    prepare_xy,
    train_rf,
    enforce_budget,
    prob_attack,
    project_root,
    aggregate_esr_dp,
    MAX_K,
    BUDGET_PCT,
)

load_dotenv(os.path.join(project_root(), "..", ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
PATIENCE = 3
MAX_CASES = int(os.getenv("P02_LLM_CASES", "2000"))

SYSTEM = "You are an expert cybersecurity attacker AI. Output raw JSON only."
USER = """Modify features to evade NIDS (P(Attack)<0.5). Max ±{bp}% per feature. Non-negative. JSON only.
{feat}"""
FB = "Still ATTACK P={p:.4f}. Propose new JSON:\n{prev}"""


def call_llm(messages):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.3}
    for _ in range(4):
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


def llm_attack(model, orig, feat, cols):
    messages = [{"role": "system", "content": SYSTEM}]
    messages.append({
        "role": "user",
        "content": USER.format(bp=int(BUDGET_PCT * 100), feat=json.dumps(feat, indent=2)),
    })
    vec = orig.copy()
    p0 = prob_attack(model, vec, cols)
    best_p = p0
    queries = 0
    stall = 0
    for _ in range(MAX_K):
        raw = call_llm(messages)
        queries += 1
        if not raw:
            stall += 1
            if stall >= PATIENCE:
                break
            continue
        clip, _ = enforce_budget(feat, raw)
        for f, v in clip.items():
            vec[f] = v
            feat[f] = v
        p = prob_attack(model, vec, cols)
        if p < 0.5:
            return p0, p, p - p0, True, queries
        if p < best_p - 1e-6:
            best_p, stall = p, 0
        else:
            stall += 1
        if stall >= PATIENCE:
            break
        messages.append({"role": "assistant", "content": json.dumps(clip)})
        messages.append({"role": "user", "content": FB.format(p=p, prev=json.dumps(clip))})
    pf = prob_attack(model, vec, cols)
    return p0, pf, pf - p0, pf < 0.5, queries


def load_or_train_model(out_dir):
    cache = os.path.join(out_dir, "v8_rf_cache.joblib")
    if os.path.exists(cache):
        print("  Using cached RF from v8_rf_cache.joblib")
        return joblib.load(cache)
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, y_te, lbl_te = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr)
    joblib.dump({"model": model, "X_te": X_te, "X_tr_cols": list(X_tr.columns), "lbl_te": lbl_te}, cache)
    return {"model": model, "X_te": X_te, "X_tr_cols": list(X_tr.columns), "lbl_te": lbl_te}


def main():
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY missing")

    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    with open(os.path.join(out_dir, "v8_sample_indices.json"), encoding="utf-8") as f:
        meta = json.load(f)
    selected = meta["indices"]
    top10 = meta["top10"]
    if MAX_CASES > 0:
        selected = selected[: min(MAX_CASES, len(selected))]

    print("=" * 65)
    print("BƯỚC 4 — LLM attack on v8 cohort (Groq API)")
    print("=" * 65)

    print("\n[1/3] Load RF + test frame ...")
    cache = load_or_train_model(out_dir)
    model = cache["model"]
    X_te = cache["X_te"]
    cols = cache["X_tr_cols"]

    out_path = os.path.join(out_dir, "step4_llm_results.json")
    rows = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            partial = json.load(f)
        rows = partial.get("raw", [])
        done_ids = {r["id"] for r in rows}
        selected = [i for i in selected if i not in done_ids]
        print(f"  Resume: {len(rows)} done, {len(selected)} remaining")

    total_cases = len(selected)
    print(f"\n[2/3] LLM on {total_cases} samples ...")
    for idx in selected:
        orig = X_te.iloc[idx].copy()
        feat = {f: float(orig[f]) for f in top10}
        p0, pf, dp, ev, q = llm_attack(model, orig, feat, cols)
        rows.append({
            "id": int(idx),
            "p_orig": round(p0, 4),
            "llm": {"evaded": ev, "prob": round(pf, 4), "delta": round(dp, 4), "queries": q},
        })
        print(f"  [{len(rows):3d}/{total_cases}] id={idx} p0={p0:.3f} p={pf:.3f} ev={'Y' if ev else 'N'} q={q}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"llm": aggregate_esr_dp(rows, "llm"), "raw": rows}, f, indent=2)
        time.sleep(0.3)

    agg = aggregate_esr_dp(rows, "llm")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"llm": agg, "raw": rows}, f, indent=2)
    print(f"\n[3/3] LLM ESR={agg['ESR']}% CI={agg['CI']} mean_dP={agg['mean_dP']} mean_q={agg['mean_queries']}")
    print(f"✓ Saved -> {out_path}")


if __name__ == "__main__":
    main()
