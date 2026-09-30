"""Bước 7: Threshold sweep + ECE on v8 step3 raw (+ step4 llm if present)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import project_root, clopper_pearson_ci


def ece(conf, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        mc, fp = float(conf[m].mean()), float(y[m].mean())
        total += (n / len(conf)) * abs(mc - fp)
    return float(total)


def main():
    d = os.path.join(project_root(), "results", "revision_v8")
    with open(os.path.join(d, "step3_baselines_results.json"), encoding="utf-8") as f:
        rows = json.load(f)["raw"]
    llm_path = os.path.join(d, "step4_llm_results.json")
    if os.path.exists(llm_path):
        by_id = {r["id"]: r["llm"] for r in json.load(open(llm_path, encoding="utf-8"))["raw"]}
        for r in rows:
            if r["id"] in by_id:
                r["llm"] = by_id[r["id"]]

    p_orig = np.array([r["p_orig"] for r in rows])
    y = np.ones(len(rows))
    out = {
        "N": len(rows),
        "ece_p_orig": round(ece(p_orig, y), 4),
        "brier_p_orig": round(float(np.mean((p_orig - y) ** 2)), 4),
        "threshold_sweep": {},
    }
    methods = ["greedy", "nes", "random_k"]
    if "llm" in rows[0]:
        methods.append("llm")
    for t in [0.35, 0.4, 0.45, 0.5, 0.55]:
        out["threshold_sweep"][str(t)] = {}
        for m in methods:
            if m not in rows[0]:
                continue
            ev = sum(1 for r in rows if r[m]["prob"] < t)
            lo, hi = clopper_pearson_ci(ev, len(rows))
            out["threshold_sweep"][str(t)][m] = {
                "esr": round(ev / len(rows) * 100, 1),
                "ci": [round(lo * 100, 1), round(hi * 100, 1)],
            }

    path = os.path.join(d, "calibration_v8_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("=" * 65)
    print("BƯỚC 7 — v8 calibration post-hoc")
    print(f"  N={out['N']} ECE={out['ece_p_orig']} Brier={out['brier_p_orig']}")
    for t in ["0.45", "0.5"]:
        print(f"  thresh {t}: {out['threshold_sweep'][t]}")
    print(f"✓ Saved -> {path}")


if __name__ == "__main__":
    main()
