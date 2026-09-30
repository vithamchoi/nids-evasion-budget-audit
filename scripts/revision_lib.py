"""Shared loaders and metrics for project 02 revision runs."""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

BUDGET_PCT = 0.10
NEAR_ZERO_ABS = 1e-3
MAX_K = 20
DATASET_TR_SIZE = 50000
DATASET_TE_SIZE = 10000
RF_PARAMS = dict(n_estimators=100, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=1)

DISCRETE = {"Destination Port", "Source Port", "Protocol"}


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_temporal_frames(tr_n=DATASET_TR_SIZE, te_subsample=None, seed=42):
    """Train: subsample for speed. Test: full split by default (v6 sample ids index full test)."""
    from datasets import load_dataset

    tr = load_dataset("lacg030175/CICIDS2017", "temporal", split="train")
    te = load_dataset("lacg030175/CICIDS2017", "temporal", split="test")
    df_train = tr.to_pandas().sample(n=min(tr_n, len(tr)), random_state=seed)
    df_test = te.to_pandas()
    if te_subsample is not None and te_subsample < len(df_test):
        df_test = df_test.sample(n=te_subsample, random_state=seed)
    return df_train, df_test


def prepare_xy(df):
    drop = [c for c in ("Label", "label") if c in df.columns]
    X = df.drop(columns=drop).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].astype(int)
    lbl = df["Label"] if "Label" in df.columns else None
    return X, y, lbl


def train_rf(X_tr, y_tr):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    return model


def top10_features(model, X_tr, y_tr):
    gini = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    Xp = X_tr.sample(n=min(5000, len(X_tr)), random_state=42)
    perm = permutation_importance(model, Xp, y_tr.loc[Xp.index], n_repeats=3, random_state=42, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False)
    cont = [c for c in X_tr.columns if c not in DISCRETE]
    inter = [f for f in gini.index if f in cont and f in perm_imp.index[:20]]
    return inter[:10]


def enforce_budget(original: dict, perturbed: dict, budget_pct=BUDGET_PCT, near_zero=NEAR_ZERO_ABS):
    clipped, compliance = {}, {}
    for k, orig_val in original.items():
        pert_val = perturbed.get(k, orig_val)
        if abs(orig_val) < near_zero:
            lo, hi = orig_val - near_zero, orig_val + near_zero
            violated = abs(pert_val - orig_val) > near_zero + 1e-9
        else:
            lo, hi = orig_val * (1 - budget_pct), orig_val * (1 + budget_pct)
            violated = abs(pert_val - orig_val) / abs(orig_val) * 100 > (budget_pct * 100 + 0.01)
        clipped[k] = float(np.clip(pert_val, lo, hi))
        compliance[k] = {"violated": bool(violated)}
    return clipped, compliance


def prob_attack(model, vec, cols):
    return float(model.predict_proba(pd.DataFrame([vec], columns=cols))[0][1])


def clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def aggregate_esr_dp(rows, method_key):
    n = len(rows)
    succ = sum(1 for r in rows if r[method_key]["evaded"])
    dps = [r[method_key]["delta"] for r in rows]
    lo, hi = clopper_pearson_ci(succ, n)
    return {
        "ESR": round(succ / n * 100, 1) if n else 0.0,
        "CI": [round(lo * 100, 1), round(hi * 100, 1)],
        "mean_dP": round(float(np.mean(dps)), 4) if dps else 0.0,
        "mean_queries": round(float(np.mean([r[method_key]["queries"] for r in rows])), 1),
    }


def stratum(p):
    if p > 0.8:
        return "High(>0.8)"
    if p > 0.6:
        return "Medium(0.6 - 0.8)"
    return "Boundary(0.5 - 0.6)"


def select_stratified_v6(tp_indices, probs, n_total=2000, n_per=None, seed=42):
    buckets = {"High(>0.8)": [], "Medium(0.6 - 0.8)": [], "Boundary(0.5 - 0.6)": []}
    for idx in tp_indices:
        p = probs[idx]
        if p <= 0.5:
            continue
        buckets[stratum(p)].append(int(idx))
    rng = np.random.default_rng(seed)
    chosen = []
    if n_per is None:
        n_per = max(1, int(np.ceil(n_total / max(len(buckets), 1))))
    for name in buckets:
        pool = buckets[name]
        if not pool:
            continue
        k = min(n_per, len(pool), n_total - len(chosen))
        if k > 0:
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    if len(chosen) < n_total:
        rest = [int(i) for i in tp_indices if int(i) not in chosen and probs[i] > 0.5]
        rng.shuffle(rest)
        for idx in rest:
            if len(chosen) >= n_total:
                break
            chosen.append(idx)
    return chosen[:n_total]


def load_v6_sample_indices(v6_json_path):
    with open(v6_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [int(r["id"]) for r in data["raw"]]
