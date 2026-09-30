"""
Post-hoc calibration + ESR threshold sensitivity on v6 raw results.
No API / no re-training — reads existing evasion_v6_results.json only.
"""

import json
import os
import numpy as np
from scipy import stats


def ece(conf, y_true, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    details = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            details.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0})
            continue
        mean_conf = float(conf[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(mean_conf - frac_pos)
        ece_val += (n / len(conf)) * gap
        details.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": n,
            "mean_conf": round(mean_conf, 4),
            "frac_pos": round(frac_pos, 4),
            "gap": round(gap, 4),
        })
    return float(ece_val), details


def brier(conf, y_true):
    return float(np.mean((conf - y_true) ** 2))


def clopper(k, n):
    if n == 0:
        return 0.0, 0.0
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def esr_at_threshold(rows, method, thresh):
    probs = [r[method]["p"] for r in rows]
    evaded = sum(1 for p in probs if p < thresh)
    n = len(rows)
    lo, hi = clopper(evaded, n)
    return {
        "threshold": thresh,
        "evaded": evaded,
        "esr": round(evaded / n * 100, 1) if n else 0.0,
        "ci": [round(lo * 100, 1), round(hi * 100, 1)],
    }


def main():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(project, "results", "evasion_v6", "evasion_v6_results.json")
    out_path = os.path.join(project, "results", "evasion_v6", "calibration_v6_analysis.json")

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["raw"]
    n = len(rows)
    p_orig = np.array([r["p_orig"] for r in rows], dtype=float)
    y_true = np.ones(n)  # all attack TPs

    ece_val, bins = ece(p_orig, y_true)
    brier_val = brier(p_orig, y_true)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresh_sens = {}
    for t in thresholds:
        key = f"thresh_{int(t * 100)}"
        thresh_sens[key] = {
            "llm": esr_at_threshold(rows, "llm", t),
            "greedy": esr_at_threshold(rows, "greedy", t),
            "random": esr_at_threshold(rows, "random", t),
        }

    per_type = {}
    for atype in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == atype]
        ns = len(sub)
        per_type[atype] = {
            "n": ns,
            "p_orig_mean": round(float(np.mean([r["p_orig"] for r in sub])), 4),
            "llm_esr_50": round(sum(1 for r in sub if r["llm"]["evaded"]) / ns * 100, 1) if ns else 0,
            "greedy_esr_50": round(sum(1 for r in sub if r["greedy"]["evaded"]) / ns * 100, 1) if ns else 0,
        }

    out = {
        "source": in_path,
        "N": n,
        "rf_calibration_attack_tps": {
            "ece_p_orig": round(ece_val, 4),
            "brier_p_orig_vs_y1": round(brier_val, 4),
            "p_orig_min": round(float(p_orig.min()), 4),
            "p_orig_max": round(float(p_orig.max()), 4),
            "p_orig_mean": round(float(p_orig.mean()), 4),
            "calibration_bins": bins,
            "note": "All v6 samples are attack TPs (y=1). ECE measures confidence vs. true label.",
        },
        "threshold_sensitivity_v6": thresh_sens,
        "per_attack_type": per_type,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    print("=" * 60)
    print("v6 post-hoc calibration (N=%d)" % n)
    print("=" * 60)
    print(f"ECE (p_orig): {ece_val:.4f}")
    print(f"Brier:      {brier_val:.4f}")
    print(f"p_orig:     [{p_orig.min():.4f}, {p_orig.max():.4f}] mean={p_orig.mean():.4f}")
    print("\nESR threshold sweep (LLM):")
    for t in thresholds:
        e = esr_at_threshold(rows, "llm", t)
        print(f"  thresh={t:.2f}  ESR={e['esr']}%  CI={e['ci']}")
    print(f"\n✓ Saved -> {out_path}")


if __name__ == "__main__":
    main()
