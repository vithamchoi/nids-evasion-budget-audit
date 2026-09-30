"""Step 8: re-run the baselines under a TRUE oracle-call budget.

The published run capped MAX_K search *iterations*. One iteration costs a
different number of detector calls in each strategy (1 for random-K, 3 for
greedy, n_perturb+1 = 9 for NES), so the comparison was matched in iterations
and unmatched in oracle access. This script caps the number of calls to the
detector directly, at ORACLE_BUDGET per flow, identically for every strategy.

The LLM arm needs no re-run: its audited cost is one call per iteration with a
maximum of 18 observed, so it already satisfies a 20-call budget and its
published numbers carry over unchanged. No API key is needed here.

Outputs results/revision_v8/step8_oracle_capped.json. Never overwrites an
existing result file from an earlier step. Safe to stop and restart: finished
flows are reloaded and skipped.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (  # noqa: E402
    BUDGET_PCT,
    NEAR_ZERO_ABS,
    clopper_pearson_ci,
    enforce_budget,
    load_temporal_frames,
    prepare_xy,
    prob_attack,
    project_root,
    stratum,
    train_rf,
)

ORACLE_BUDGET = int(os.getenv("P02_ORACLE_BUDGET", "20"))
TAU = 0.5


class Oracle:
    """Counts detector calls and refuses to serve more than the budget."""

    class Exhausted(Exception):
        pass

    def __init__(self, model, cols, budget):
        self.model, self.cols, self.budget, self.used = model, cols, budget, 0

    def __call__(self, vec):
        if self.used >= self.budget:
            raise Oracle.Exhausted()
        self.used += 1
        return prob_attack(self.model, vec, self.cols)


def _box(value):
    if abs(value) < NEAR_ZERO_ABS:
        return max(0.0, value - NEAR_ZERO_ABS), value + NEAR_ZERO_ABS
    return max(0.0, value * (1 - BUDGET_PCT)), value * (1 + BUDGET_PCT)


def greedy(oracle, orig_vec, feat, order, p0):
    """Two calls per coordinate step; the incumbent probability is cached."""
    vec, fd, best_p = orig_vec.copy(), dict(feat), p0
    step = 0
    try:
        while True:
            name = order[step % len(order)]
            step += 1
            lo, hi = _box(fd[name])
            for cand in (lo, hi):
                trial = vec.copy()
                trial[name] = cand
                p = oracle(trial)
                if p < best_p:
                    best_p, vec = p, trial
                    fd[name] = cand
                if best_p < TAU:
                    return vec, best_p
    except Oracle.Exhausted:
        return vec, best_p


def random_k(oracle, orig_vec, feat, rng, p0):
    vec, best_p = orig_vec.copy(), p0
    try:
        while True:
            raw = {}
            for k, v in feat.items():
                if abs(v) < NEAR_ZERO_ABS:
                    raw[k] = max(0.0, v + rng.uniform(-NEAR_ZERO_ABS, NEAR_ZERO_ABS))
                else:
                    raw[k] = max(0.0, v * (1 + rng.uniform(-BUDGET_PCT, BUDGET_PCT)))
            clip, _ = enforce_budget(feat, raw)
            trial = orig_vec.copy()
            for k, v in clip.items():
                trial[k] = v
            p = oracle(trial)
            if p < best_p:
                best_p, vec = p, trial
            if best_p < TAU:
                return vec, best_p
    except Oracle.Exhausted:
        return vec, best_p


def nes(oracle, orig_vec, feat, rng, p0, n_perturb, sigma_frac=0.05, step_frac=0.25):
    """One step costs n_perturb probes plus one evaluation."""
    vec, fd, best_p = orig_vec.copy(), dict(feat), p0
    try:
        while True:
            grads = {k: 0.0 for k in fd}
            for _ in range(n_perturb):
                noise = {k: rng.normal(0, max(abs(v), NEAR_ZERO_ABS) * sigma_frac)
                         for k, v in fd.items()}
                prop = {k: fd[k] + noise[k] for k in fd}
                clip, _ = enforce_budget(feat, prop)
                trial = vec.copy()
                for k, v in clip.items():
                    trial[k] = v
                p = oracle(trial)
                sign = 1.0 if p < best_p else -1.0
                for k in fd:
                    grads[k] += sign * noise[k]
                if p < TAU:
                    return trial, p
            prop = {}
            for k, v in fd.items():
                g = grads[k] / n_perturb
                if abs(v) < NEAR_ZERO_ABS:
                    prop[k] = v - np.sign(g) * NEAR_ZERO_ABS * 0.5
                else:
                    prop[k] = v - np.sign(g) * abs(v) * BUDGET_PCT * step_frac
            clip, _ = enforce_budget(feat, prop)
            for k, v in clip.items():
                vec[k], fd[k] = v, v
            p = oracle(vec)
            if p < best_p:
                best_p = p
            if p < TAU:
                return vec, p
    except Oracle.Exhausted:
        return vec, best_p


ARMS = [
    ("greedy", lambda o, v, f, r, p, order: greedy(o, v, f, order, p)),
    ("random_k", lambda o, v, f, r, p, order: random_k(o, v, f, r, p)),
    ("nes8", lambda o, v, f, r, p, order: nes(o, v, f, r, p, n_perturb=8)),
    ("nes4", lambda o, v, f, r, p, order: nes(o, v, f, r, p, n_perturb=4)),
    ("nes2", lambda o, v, f, r, p, order: nes(o, v, f, r, p, n_perturb=2)),
]


def summarise(rows, key):
    n = len(rows)
    k = sum(1 for r in rows if r[key]["evaded"])
    lo, hi = clopper_pearson_ci(k, n)
    return {
        "evaded": k, "n": n,
        "ESR": round(100 * k / n, 1) if n else 0.0,
        "CI": [round(100 * lo, 1), round(100 * hi, 1)],
        "mean_dP": round(float(np.mean([r[key]["delta"] for r in rows])), 4),
        "mean_oracle_calls": round(float(np.mean([r[key]["calls"] for r in rows])), 1),
        "max_oracle_calls": int(max(r[key]["calls"] for r in rows)),
    }


def main():
    root = project_root()
    out_dir = os.path.join(root, "results", "revision_v8")
    out_path = os.path.join(out_dir, "step8_oracle_capped.json")

    with open(os.path.join(out_dir, "v8_sample_indices.json"), encoding="utf-8") as f:
        meta = json.load(f)
    selected, top10 = meta["indices"], meta["top10"]

    print("=" * 68)
    print(f"STEP 8 - true oracle-call budget = {ORACLE_BUDGET} calls per flow")
    print("=" * 68)
    print("\n[1/3] Load data + train RF (same 50k rows, same seed) ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, _, lbl_te = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr)
    cols = X_tr.columns

    rows = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            rows = json.load(f).get("raw", [])
        done = {r["id"] for r in rows}
        selected = [i for i in selected if i not in done]
        print(f"  Resume: {len(rows)} done, {len(selected)} remaining")

    print(f"\n[2/3] Attack {len(selected)} flows, {len(ARMS)} arms ...")
    t0 = time.time()
    for n_done, idx in enumerate(selected, 1):
        orig = X_te.iloc[idx].copy()
        feat = {f: float(orig[f]) for f in top10}
        p0 = prob_attack(model, orig, cols)
        rec = {"id": int(idx), "p_orig": round(p0, 4), "stratum": stratum(p0),
               "type": str(lbl_te.iloc[idx]) if lbl_te is not None else "unknown"}
        for name, fn in ARMS:
            rng = np.random.default_rng(99 + idx)
            oracle = Oracle(model, cols, ORACLE_BUDGET)
            vec, pf = fn(oracle, orig.copy(), dict(feat), rng, p0, top10)
            rec[name] = {"evaded": bool(pf < TAU), "prob": round(float(pf), 4),
                         "delta": round(float(pf - p0), 4), "calls": oracle.used}
        rows.append(rec)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"config": {"oracle_budget": ORACLE_BUDGET, "tau": TAU,
                                  "features": top10},
                       "arms": {k: summarise(rows, k) for k, _ in ARMS},
                       "raw": rows}, f, indent=2)
        if n_done % 10 == 0 or n_done == 1:
            el = time.time() - t0
            print(f"  [{len(rows):3d}] id={idx} p0={p0:.3f} " +
                  " ".join(f"{k}={'Y' if rec[k]['evaded'] else 'N'}" for k, _ in ARMS) +
                  f"  ({el:.0f}s)")

    print("\n[3/3] Summary")
    for k, _ in ARMS:
        s = summarise(rows, k)
        print(f"  {k:9s} ESR={s['ESR']:5.1f}% CI={s['CI']} "
              f"dP={s['mean_dP']:+.4f} calls={s['mean_oracle_calls']:.1f}/{ORACLE_BUDGET}")
    print(f"\n  LLM arm: not re-run. Its audited cost is <= 18 calls per flow,")
    print(f"  already within the {ORACLE_BUDGET}-call budget, so its published")
    print(f"  ESR=33.6% carries over unchanged.")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
