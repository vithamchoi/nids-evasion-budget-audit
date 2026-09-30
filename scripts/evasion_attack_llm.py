import os
import json
import time
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env"))

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "llama3.1-8b"

# Top-10 features from RF feature importance (gray-box attacker knowledge)
# Excludes Destination Port as it is discrete and semantically invalid for ±10% perturbation
TOP_FEATURES_CONTINUOUS = [
    'Bwd Packet Length Std',
    'Packet Length Std',
    'Bwd Packet Length Max',
    'Avg Bwd Segment Size',
    'Bwd Packet Length Mean',
    'Fwd IAT Std',
    'Average Packet Size',
    'Packet Length Variance',
    'Flow IAT Max',
    'Packet Length Mean'  # replacing Destination Port with next ranked continuous feature
]

def parse_llm_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def enforce_budget(original: dict, perturbed: dict, budget_pct: float = 0.10) -> tuple[dict, dict]:
    """Post-hoc clip perturbed values to be within budget_pct of originals.
    Returns (clipped_dict, compliance_dict) where compliance shows % deviation per feature."""
    clipped = {}
    compliance = {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        lo = orig_val * (1 - budget_pct)
        hi = orig_val * (1 + budget_pct)
        raw_delta_pct = abs(pert_val - orig_val) / (abs(orig_val) + 1e-9) * 100
        clipped_val = float(np.clip(pert_val, lo, hi))
        clipped[k] = clipped_val
        compliance[k] = {
            "original": orig_val,
            "llm_proposed": pert_val,
            "clipped": clipped_val,
            "llm_violated_budget": raw_delta_pct > (budget_pct * 100 + 0.01),
            "delta_pct_after_clip": abs(clipped_val - orig_val) / (abs(orig_val) + 1e-9) * 100
        }
    return clipped, compliance

def attack_sample_with_llm(features_dict):
    prompt = f"""You are a cybersecurity adversarial attacker. You must modify the following network flow features of a malicious packet so it gets classified as BENIGN by a machine learning model.

Rules:
1. You can only modify the values by a maximum of +/- 10% from their original values.
2. All values are continuous floats.
3. Return ONLY a valid JSON object with the identical keys. DO NOT output any explanation, markdown formatting, or text outside the JSON block.

Original Features:
{json.dumps(features_dict, indent=2)}
"""
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an AI that outputs raw JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 429:
            print(f"    [!] Rate limited (429). Waiting 30s (attempt {attempt+1}/{max_retries})...")
            time.sleep(30)
            continue
        response.raise_for_status()
        output_text = response.json()["choices"][0]["message"]["content"]
        return parse_llm_json(output_text)
    raise Exception("Max retries exceeded for 429 Rate Limit")

def main():
    print("=== LLM Evasion Attack (v2 — Budget-Enforced, Stratified, Gray-box) ===")
    if not CEREBRAS_API_KEY:
        print("ERROR: CEREBRAS_API_KEY not found in .env")
        return

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "datasets", "CICIDS2017_hf", "random", "train-00000-of-00001.parquet")

    print("1. Loading dataset...")
    df = pd.read_parquet(dataset_path).sample(n=50000, random_state=42)

    # Keep Label for stratified sampling, then drop
    label_col = df['Label'].copy()
    X = df.drop(columns=['Label', 'label']).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['label']

    X_train, X_test, y_train, y_test, label_train, label_test = train_test_split(
        X, y, label_col, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Training Random Forest Baseline (gray-box: attacker uses feature importances)...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Feature importances used by attacker → gray-box setting
    feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"   Top-10 features used (gray-box): {list(feat_importance.head(10).index)}")

    y_pred = model.predict(X_test)

    # Stratified sample: pick true positive attack samples across attack types
    tp_mask = (y_test.values == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    tp_attack_types = label_test.iloc[tp_indices].values

    print(f"\n   Attack types in test set TPs:")
    type_counts = pd.Series(tp_attack_types).value_counts()
    for t, c in type_counts.items():
        print(f"     {t}: {c} samples")

    # Sample up to 2 from each attack type (stratified), max 10 total
    rng = np.random.default_rng(seed=42)
    selected_indices = []
    selected_types = []
    for attack_type, group_indices in pd.Series(tp_indices, index=tp_attack_types).groupby(level=0):
        picks = rng.choice(group_indices.values, size=min(2, len(group_indices)), replace=False)
        selected_indices.extend(picks)
        selected_types.extend([attack_type] * len(picks))
        if len(selected_indices) >= 10:
            break

    selected_indices = selected_indices[:10]
    selected_types = selected_types[:10]
    NUM_ATTACKS = len(selected_indices)

    print(f"\n3. Starting LLM Evasion Attack on {NUM_ATTACKS} stratified attack samples...")
    print(f"   Attacker model: {MODEL_NAME} (Cerebras) | temperature: 0.3")
    print(f"   Attacker knowledge: GRAY-BOX (uses victim RF feature importances)")
    print(f"   Budget: ±10% (post-hoc enforced via clipping)")
    print(f"   Features perturbed: {len(TOP_FEATURES_CONTINUOUS)}/78 (all continuous, no discrete ports)")

    results = []
    success_count = 0
    budget_violations = 0

    for i, (idx, attack_type) in enumerate(zip(selected_indices, selected_types)):
        original_vector = X_test.iloc[idx].copy()
        prob_before = model.predict_proba(pd.DataFrame([original_vector], columns=X.columns))[0]

        target_features = {feat: float(original_vector[feat]) for feat in TOP_FEATURES_CONTINUOUS}

        try:
            llm_output = attack_sample_with_llm(target_features)
            clipped, compliance = enforce_budget(target_features, llm_output, budget_pct=0.10)

            # Count budget violations
            n_violated = sum(1 for v in compliance.values() if v["llm_violated_budget"])
            if n_violated > 0:
                budget_violations += 1

            modified_vector = original_vector.copy()
            for feat, val in clipped.items():
                if feat in modified_vector.index:
                    modified_vector[feat] = val

            modified_df = pd.DataFrame([modified_vector], columns=X.columns)
            new_pred = model.predict(modified_df)[0]
            prob_after = model.predict_proba(modified_df)[0]

            evaded = bool(new_pred == 0)
            if evaded:
                success_count += 1

            result = {
                "sample_id": i + 1,
                "attack_type": attack_type,
                "evaded": evaded,
                "n_budget_violations": n_violated,
                "prob_attack_before": round(float(prob_before[1]), 4),
                "prob_attack_after": round(float(prob_after[1]), 4),
                "prob_delta": round(float(prob_after[1] - prob_before[1]), 4),
                "compliance": compliance
            }
            results.append(result)

            status = "[+] EVADED" if evaded else "[-] FAILED"
            print(f"  {status} Sample {i+1}/{NUM_ATTACKS} ({attack_type}): "
                  f"P(Attack) {prob_before[1]:.4f} → {prob_after[1]:.4f} | "
                  f"Budget violations: {n_violated}/10")

        except Exception as e:
            print(f"  [!] Sample {i+1}/{NUM_ATTACKS}: ERROR - {str(e)}")
            results.append({"sample_id": i+1, "attack_type": attack_type, "evaded": False, "error": str(e)})

        time.sleep(1)

    # Save detailed results
    results_dir = os.path.join(project_dir, "results", "evasion")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "evasion_detailed.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else bool(x) if isinstance(x, np.bool_) else str(x)) + "\n")

    valid_results = [r for r in results if "error" not in r]
    esr = (success_count / len(valid_results)) * 100 if valid_results else 0

    print("\n=== EVASION ATTACK RESULTS (v2) ===")
    print(f"Attacker Model: {MODEL_NAME} (Cerebras, temp=0.3)")
    print(f"Attacker Knowledge: GRAY-BOX (victim RF feature importances used)")
    print(f"Total Samples: {NUM_ATTACKS}")
    print(f"Valid LLM Responses: {len(valid_results)}")
    print(f"Evasion Successes: {success_count}")
    print(f"Evasion Success Rate (ESR): {esr:.2f}%")
    print(f"Robust Accuracy: {100 - esr:.2f}%")
    print(f"Samples where LLM violated ±10% budget: {budget_violations}/{len(valid_results)}")

    print("\n--- Per-Sample Detail ---")
    print(f"{'ID':<4} {'Attack Type':<30} {'P(Atk) Before':<16} {'P(Atk) After':<15} {'Delta':<10} {'Evaded':<8} {'Budget Viol.'}")
    for r in valid_results:
        print(f"{r['sample_id']:<4} {r['attack_type']:<30} {r['prob_attack_before']:<16} "
              f"{r['prob_attack_after']:<15} {r['prob_delta']:<10} {str(r['evaded']):<8} {r['n_budget_violations']}/10")

if __name__ == "__main__":
    main()
