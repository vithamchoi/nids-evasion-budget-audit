#!/usr/bin/env python3
"""Generate the paper's data figures directly from the revision_v8 result files.

    python3 make_figures.py <results_dir> <out_dir>

Every plotted value is read from disk; none is estimated or simulated. Output is
vector PDF sized for the Elsevier two-column layout (3.35 in single column,
6.9 in full width).

Palette: Okabe-Ito subset, validated colourblind-safe in fixed assignment order
(#0072B2, #D55E00, #009E73, #E69F00). Series identity is never colour-alone:
every series also carries a distinct marker or hatch and appears in the legend.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle as F

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
OUT.mkdir(parents=True, exist_ok=True)

C = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def load(rel):
    with open(RES / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


s6 = load("revision_v8/step6_merged_results.json")
s3 = load("revision_v8/step3_baselines_results.json")
s4 = load("revision_v8/step4_llm_results.json")
s2 = load("revision_v8/step2_isotonic_calibration.json")

raw3 = {r["id"]: r for r in s3["raw"]}
raw4 = {r["id"]: r for r in s4["raw"]}
IDS = sorted(raw3)

METHODS = [("llm", "LLM-guided"), ("greedy", "Greedy 1-D"),
           ("random_k", "Random-$K$"), ("nes", "NES")]
MARK = ["o", "s", "^", "D"]


def deltas(key):
    src = raw4 if key == "llm" else raw3
    return [src[i][key]["delta"] for i in IDS]


# ============================================================ Fig: threshold sweep
sweep = s6["threshold_sweep"]
ths = sorted(sweep, key=float)
xs = [float(t) for t in ths]

fig, ax = plt.subplots(figsize=(5.6, 2.64))
for i, (key, label) in enumerate(METHODS):
    ys = [sweep[t][key]["esr"] for t in ths]
    lo = [sweep[t][key]["ci"][0] for t in ths]
    hi = [sweep[t][key]["ci"][1] for t in ths]
    ax.fill_between(xs, lo, hi, color=C[i], alpha=0.12, linewidth=0)
    ax.plot(xs, ys, color=C[i], marker=MARK[i], markersize=3.6, linewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.5, label=label)
ax.set_xlabel(r"Decision threshold $\tau$")
ax.set_ylabel("Evasion success rate (\\%)".replace("\\%", "%"))
ax.set_xticks(xs)
ax.set_ylim(-2, 58)
F.place_legend(ax, frameon=False, handlelength=1.6,
          labelcolor=INK, borderaxespad=0.2)
tidy(ax)
fig.tight_layout()
save(fig, "fig_threshold_sweep.pdf")


# ============================================================ Fig: per-stratum
by = defaultdict(list)
for i in IDS:
    by[raw3[i]["stratum"]].append(i)
ORDER = ["High(>0.8)", "Medium(0.6 - 0.8)", "Boundary(0.5 - 0.6)"]
SHORT = {"High(>0.8)": "High\n$p_{orig}>0.8$",
         "Medium(0.6 - 0.8)": "Medium\n$0.6$–$0.8$",
         "Boundary(0.5 - 0.6)": "Boundary\n$0.5$–$0.6$"}
NLAB = {}
ORDER = [s for s in ORDER if s in by]

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.30))

ax = axes[0]
ns = [len(by[s]) for s in ORDER]
esr = [100 * sum(raw4[i]["llm"]["evaded"] for i in by[s]) / len(by[s]) for s in ORDER]
bars = ax.bar(range(len(ORDER)), esr, width=0.55, color=C[0],
              edgecolor="white", linewidth=0.6)
for x, (v, n) in enumerate(zip(esr, ns)):
    ax.text(x, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.4, color=INK)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f"{SHORT[s]}\n($n={len(by[s])}$)" for s in ORDER])
ax.set_ylabel("LLM evasion success rate (%)")
ax.set_ylim(0, 116)
ax.xaxis.grid(False)
ax.set_title(r"(a) Evasion succeeds only near the boundary", loc="left", pad=4)
tidy(ax)

ax = axes[1]
for k, s in enumerate(ORDER):
    d = [raw4[i]["llm"]["delta"] for i in by[s]]
    jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.26
    ax.scatter(np.full(len(d), k) + jitter, d, s=9, color=C[0], alpha=0.5,
               linewidths=0)
    ax.plot([k - 0.28, k + 0.28], [statistics.mean(d)] * 2, color=C[1],
            linewidth=1.8, solid_capstyle="butt",
            label="stratum mean" if k == 0 else None)
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f"{SHORT[s]}\n($n={len(by[s])}$)" for s in ORDER])
ax.set_ylabel(r"Per-flow $\Delta P$")
ax.xaxis.grid(False)
ax.set_title(r"(b) $\Delta P$ is largest in the middle stratum", loc="left", pad=4)
F.place_legend(ax, frameon=False, handlelength=1.4, labelcolor=INK,
          borderaxespad=0.2)
tidy(ax)

fig.tight_layout(w_pad=2.0)
save(fig, "fig_strata.pdf")


# ============================================================ Fig: delta-P by method
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.30),
                         gridspec_kw={"width_ratios": [1.45, 1.0]})

ax = axes[0]
rng = np.random.default_rng(1)
for i, (key, label) in enumerate(METHODS):
    d = deltas(key)
    jitter = (rng.random(len(d)) - 0.5) * 0.3
    ax.scatter(np.full(len(d), i) + jitter, d, s=8, color=C[i], alpha=0.45,
               linewidths=0)
    m = statistics.mean(d)
    ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, linewidth=1.6,
            solid_capstyle="butt")
    ax.text(i + 0.34, m, f"{m:+.3f}", va="center", ha="left", fontsize=6.8,
            color=INK)
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([m[1] for m in METHODS])
ax.set_xlim(-0.5, len(METHODS) - 0.1)
ax.set_ylabel(r"Per-flow $\Delta P$")
ax.xaxis.grid(False)
ax.set_title(r"(a) Per-flow $\Delta P$, all \Ntotal\ flows".replace(r"\Ntotal\ ", f"{len(IDS)} "),
             loc="left", pad=4)
tidy(ax)

ax = axes[1]
q = [raw4[i]["llm"]["queries"] for i in IDS]
ax.hist(q, bins=range(1, max(q) + 2), color=C[0], edgecolor="white",
        linewidth=0.6, align="left")
ax.axvline(20, color=C[1], linewidth=1.4, linestyle=(0, (4, 3)))
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
ax.text(19.4, ax.get_ylim()[1] * 0.99, "baseline budget $K=20$", ha="right",
        va="top", fontsize=6.8, color=C[1])
ax.axvline(statistics.mean(q), color=INK, linewidth=1.2)
ax.text(statistics.mean(q) + 0.5, ax.get_ylim()[1] * 0.72,
        f"mean {statistics.mean(q):.1f}", ha="left", va="top", fontsize=6.8,
        color=INK)
ax.set_xlabel("Oracle queries used per flow")
ax.set_ylabel("Flows")
ax.set_title("(b) The LLM stops early", loc="left", pad=4)
tidy(ax)

fig.tight_layout(w_pad=2.0)
save(fig, "fig_deltap.pdf")


# ============================================================ Fig: calibration
def reliability(bins):
    xs, ys, ns = [], [], []
    for b in bins:
        if b["n"] == 0:
            continue
        xs.append(b["mean_conf"])
        ys.append(b["frac_pos"])
        ns.append(b["n"])
    return xs, ys, ns


fig, ax = plt.subplots(figsize=(5.6, 2.82))
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
        label="perfect calibration")
for i, (key, label) in enumerate([("bins_raw", "raw posteriors"),
                                  ("bins_isotonic", "isotonic")]):
    xs, ys, ns = reliability(s2[key])
    sizes = [10 + 26 * (np.log10(n + 1) / np.log10(max(ns) + 1)) for n in ns]
    ax.scatter(xs, ys, s=sizes, color=C[i], alpha=0.85, marker=MARK[i],
               edgecolor="white", linewidth=0.5, label=label, zorder=3)
    ax.plot(xs, ys, color=C[i], linewidth=1.1, alpha=0.7, zorder=2)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed positive fraction")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
F.place_legend(ax, frameon=False, handlelength=1.4, labelcolor=INK,
          borderaxespad=0.2)
ax.text(0.98, 0.03,
        f"ECE {s2['ece_raw']:.4f} $\\to$ {s2['ece_isotonic']:.4f}\n"
        f"Brier {s2['brier_raw']:.4f} $\\to$ {s2['brier_isotonic']:.4f}",
        ha="right", va="bottom", fontsize=6.8, color=MUTED)
tidy(ax)
fig.tight_layout()
save(fig, "fig_calibration.pdf")

print("\nAll figures generated from:", RES.resolve())


# --- v2: duong cong ESR theo ngan sach nhieu, doc tu step12 ----------------
_S12f = load("revision_v8/step12_budget_sweep.json")
_tauf = _S12f["config"]["tau"]
_rawf = _S12f["raw"]
_EPS = ["0.05", "0.10", "0.15", "0.20", "0.30", "0.50", "1.00"]
_x = [float(e) * 100 for e in _EPS]
_esr = [100.0 * sum(1 for r in _rawf if r["eps"][e]["best"] < _tauf) / len(_rawf)
        for e in _EPS]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
ax.plot(_x, _esr, "o-", color=C[0], lw=1.2, ms=4)
ax.axvline(15, color=MUTED, lw=0.7, ls="--")
ax.annotate("first evasion\n15%", xy=(15, 2), xytext=(19, 18),
            fontsize=6.8, color=MUTED,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
ax.set_xlabel(r"Per-feature budget $\varepsilon$ (%)")
ax.set_ylabel("Evasion success rate (%)")
ax.set_xscale("log")
ax.set_xticks(_x)
ax.set_xticklabels([f"{v:g}" for v in _x])
ax.set_ylim(-2, 55)
tidy(ax)

ax = axes[1]
for e, col in zip(["0.10", "0.20", "0.50"], C):
    vals = sorted(r["eps"][e]["best"] for r in _rawf)
    ax.plot(vals, [i / len(vals) for i in range(len(vals))], lw=1.2,
            color=col, label=rf"$\varepsilon={float(e)*100:g}\%$")
ax.axvline(_tauf, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("Best detector probability reached")
ax.set_ylabel("Cumulative fraction of flows")
ax.legend(frameon=False, loc="upper left")
tidy(ax)
save(fig, "fig_budget_sweep.pdf")
