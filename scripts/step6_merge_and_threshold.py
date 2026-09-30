"""Bước 6: Merge v8 results + threshold sweep on greedy/nes from step3 raw."""

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import project_root, clopper_pearson_ci


def esr_thresh(rows, method, t):
    ev = sum(1 for r in rows if r[method]["prob"] < t)
    n = len(rows)
    lo, hi = clopper_pearson_ci(ev, n)
    return {"threshold": t, "evaded": ev, "esr": round(ev / n * 100, 1), "ci": [round(lo * 100, 1), round(hi * 100, 1)]}


def main():
    root = project_root()
    d = os.path.join(root, "results", "revision_v8")
    with open(os.path.join(d, "step3_baselines_results.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    rows = s3["raw"]

    llm_path = os.path.join(d, "step4_llm_results.json")
    llm_agg = None
    if os.path.exists(llm_path):
        with open(llm_path, encoding="utf-8") as f:
            s4 = json.load(f)
        llm_agg = s4.get("llm")
        by_id = {r["id"]: r["llm"] for r in s4.get("raw", [])}
        for r in rows:
            if r["id"] in by_id:
                r["llm"] = by_id[r["id"]]

    thresh = {}
    for t in [0.4, 0.45, 0.5, 0.55]:
        thresh[str(t)] = {
            "greedy": esr_thresh(rows, "greedy", t),
            "nes": esr_thresh(rows, "nes", t),
            "random_k": esr_thresh(rows, "random_k", t),
        }
        if any("llm" in r for r in rows):
            thresh[str(t)]["llm"] = esr_thresh(
                [r for r in rows if "llm" in r], "llm", t
            )

    merged = {
        "baselines": {
            "greedy": s3["greedy"],
            "nes": s3["nes"],
            "random_k": s3["random_k"],
            "wilcoxon": s3["wilcoxon"],
        },
        "llm": llm_agg,
        "threshold_sweep": thresh,
        "config": s3["config"],
    }
    with open(os.path.join(d, "step2_isotonic_calibration.json"), encoding="utf-8") as f:
        merged["isotonic_train_cal"] = json.load(f)

    if os.path.exists(os.path.join(d, "step5_sampling_comparison.json")):
        with open(os.path.join(d, "step5_sampling_comparison.json"), encoding="utf-8") as f:
            merged["sampling_comparison"] = json.load(f)

    out = os.path.join(d, "step6_merged_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("=" * 65)
    print("BƯỚC 6 — Merge v8 results")
    print("=" * 65)
    print(f"  Greedy ESR={s3['greedy']['ESR']}% NES={s3['nes']['ESR']}% RandomK={s3['random_k']['ESR']}%")
    if llm_agg:
        print(f"  LLM ESR={llm_agg['ESR']}% (n={len(s4.get('raw',[]))} samples)")
    else:
        print("  LLM: step4 not complete yet")
    print(f"\n✓ Saved -> {out}")


if __name__ == "__main__":
    main()
