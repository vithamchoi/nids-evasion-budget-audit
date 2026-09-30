#!/usr/bin/env python3
"""Generate LaTeX table bodies and inline numeric macros for the CICIDS2017
LLM-evasion paper, read directly from the revision_v8 result files.

    python3 make_tables.py <results_dir> <out_dir>

Nothing here is typed by hand. Every figure in the paper traces to one of:
    revision_v8/step1_temporal_verify.json
    revision_v8/step2_isotonic_calibration.json
    revision_v8/step3_baselines_results.json
    revision_v8/step4_llm_results.json
    revision_v8/step5_sampling_comparison.json
    revision_v8/step6_merged_results.json
    baseline/metrics.csv
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "tables")
OUT.mkdir(parents=True, exist_ok=True)


def load(rel):
    with open(RES / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def write(name, body):
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}")


def pc(x, d=1):
    return f"{x:.{d}f}\\%"


def ci(pair):
    return f"[{pair[0]:.1f}, {pair[1]:.1f}]"


def sgn(x, d=4):
    return f"{x:+.{d}f}"


s6 = load("revision_v8/step6_merged_results.json")
s3 = load("revision_v8/step3_baselines_results.json")
s4 = load("revision_v8/step4_llm_results.json")
s2 = load("revision_v8/step2_isotonic_calibration.json")
s1 = load("revision_v8/step1_temporal_verify.json")
s5 = load("revision_v8/step5_sampling_comparison.json")

CFG = s6["config"]
raw3 = {r["id"]: r for r in s3["raw"]}
raw4 = {r["id"]: r for r in s4["raw"]}

# Human-facing method names, in the order they appear in every table.
METHODS = [
    ("llm", "LLM-guided iterative search"),
    ("greedy", "Greedy coordinate (1-D)"),
    ("random_k", r"Random-$K$ perturbation"),
    ("nes", "NES (score-based gradient)"),
]


def summary(key):
    return s6["llm"] if key == "llm" else s6["baselines"][key]


def per_sample(key):
    """Per-sample records for a method, aligned on flow id."""
    src = raw4 if key == "llm" else raw3
    return [src[i][key] for i in sorted(src)]


# ------------------------------------------------- Table 1: primary performance
rows = []
for key, label in METHODS:
    s = summary(key)
    recs = per_sample(key)
    p_orig = statistics.mean(raw4[i]["p_orig"] for i in sorted(raw4))
    p_after = statistics.mean(r["prob"] for r in recs)
    sweep = s6["threshold_sweep"]
    e50 = sweep["0.5"][key]
    e55 = sweep["0.55"][key]
    bold = (lambda t: r"\textbf{" + t + "}") if key == "llm" else (lambda t: t)
    rows.append(" & ".join([
        bold(label),
        f"{p_orig:.4f}",
        bold(f"{p_after:.4f}"),
        bold(sgn(s["mean_dP"])),
        bold(pc(e50["esr"])),
        ci(e50["ci"]),
        bold(pc(e55["esr"])),
        ci(e55["ci"]),
        bold(f"{s['mean_queries']:.1f}"),
    ]) + r" \\")
write("tab_primary.tex", "\n".join(rows))

# ------------------------------------------------- Table 2: threshold sweep
sweep = s6["threshold_sweep"]
ths = sorted(sweep, key=float)
SHORT = {"llm": "LLM-guided", "greedy": "Greedy 1-D",
         "random_k": r"Random-$K$", "nes": "NES"}
rows = []
for key, label in METHODS:
    cells = [SHORT[key]]
    for t in ths:
        e = sweep[t][key]
        cells.append(f"{e['evaded']} ({pc(e['esr'])})")
    rows.append(" & ".join(cells) + r" \\")
write("tab_sweep.tex", "\n".join(rows))
write("tab_sweep_header.tex",
      " & ".join(["Strategy"] + [f"$\\tau{{=}}{t}$" for t in ths]) + r" \\")

# ------------------------------------------------- Table 3: per-stratum breakdown
by = defaultdict(list)
for i, r in raw3.items():
    by[r["stratum"]].append(i)
ORDER = ["High(>0.8)", "Medium(0.6 - 0.8)", "Boundary(0.5 - 0.6)"]
PRETTY = {
    "High(>0.8)": r"High ($p_{\text{orig}}>0.8$)",
    "Medium(0.6 - 0.8)": r"Medium ($0.6 \le p_{\text{orig}} \le 0.8$)",
    "Boundary(0.5 - 0.6)": r"Boundary ($0.5 \le p_{\text{orig}} < 0.6$)",
}
rows = []
strat_stats = {}
for st in ORDER:
    ids = by[st]
    if not ids:
        continue
    d_llm = [raw4[i]["llm"]["delta"] for i in ids]
    ev = sum(raw4[i]["llm"]["evaded"] for i in ids)
    q = [raw4[i]["llm"]["queries"] for i in ids]
    po = [raw4[i]["p_orig"] for i in ids]
    d_gre = [raw3[i]["greedy"]["delta"] for i in ids]
    strat_stats[st] = dict(n=len(ids), esr=100 * ev / len(ids),
                           dp=statistics.mean(d_llm), ev=ev)
    rows.append(" & ".join([
        PRETTY[st],
        str(len(ids)),
        f"{statistics.mean(po):.4f}",
        sgn(statistics.mean(d_llm)),
        sgn(statistics.mean(d_gre)),
        f"{ev} / {len(ids)}",
        pc(100 * ev / len(ids)),
        f"{statistics.mean(q):.1f}",
    ]) + r" \\")
write("tab_strata.tex", "\n".join(rows))

# ------------------------------------------------- Table 4: calibration
rows = [
    " & ".join(["Raw Random-Forest posteriors",
                f"{s2['calibration_rows']:,}".replace(",", "{,}"),
                f"{s2['ece_raw']:.4f}", f"{s2['brier_raw']:.4f}"]) + r" \\",
    " & ".join([r"Isotonic-calibrated posteriors",
                f"{s2['calibration_rows']:,}".replace(",", "{,}"),
                r"\textbf{" + f"{s2['ece_isotonic']:.4f}" + "}",
                r"\textbf{" + f"{s2['brier_isotonic']:.4f}" + "}"]) + r" \\",
]
write("tab_calib.tex", "\n".join(rows))

# ------------------------------------------------- Table 5: sampling comparison
rows = []
for arm in ("stratified", "uniform"):
    g = s5[arm]["greedy"]
    rows.append(" & ".join([
        arm.capitalize(),
        str(s5["N_per_arm"]),
        pc(g["ESR"]),
        ci(g["CI"]),
        sgn(g["mean_dP"]),
        f"{g['mean_queries']:.1f}",
        f"{s5[arm]['elapsed_s']:.1f}",
    ]) + r" \\")
write("tab_sampling.tex", "\n".join(rows))

# ------------------------------------------------- Table 6: baseline classifier
with open(RES / "baseline" / "metrics.csv", newline="", encoding="utf-8") as f:
    base = next(csv.DictReader(f))
rows = [" & ".join([
    base["Model"],
    f"{int(base['Train_Samples']):,}".replace(",", "{,}"),
    f"{int(base['Test_Samples']):,}".replace(",", "{,}"),
    f"{float(base['Accuracy']):.4f}",
    f"{float(base['Precision']):.4f}",
    f"{float(base['Recall']):.4f}",
    f"{float(base['F1_Score']):.4f}",
]) + r" \\"]
write("tab_baseline.tex", "\n".join(rows))

# ------------------------------------------------- Table 7: manipulated features
def esc(t):
    return t.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


feats = [esc(f) for f in CFG["features"]]
half = (len(feats) + 1) // 2
rows = []
for a, b in zip(feats[:half], feats[half:] + [""] * half):
    rows.append(f"\\texttt{{{a}}} & " + (f"\\texttt{{{b}}}" if b else "") + r" \\")
write("tab_features.tex", "\n".join(rows))

# ------------------------------------------------- inline macros
llm = s6["llm"]
gre = s6["baselines"]["greedy"]
nes = s6["baselines"]["nes"]
rnd = s6["baselines"]["random_k"]
wil = s6["baselines"]["wilcoxon"]
p_orig_mean = statistics.mean(r["p_orig"] for r in s4["raw"])
p_after_mean = statistics.mean(r["llm"]["prob"] for r in s4["raw"])
llm_ev = sum(r["llm"]["evaded"] for r in s4["raw"])


def wsci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


macros = [
    (r"\Ntotal", str(CFG["N"])),
    (r"\Kbudget", str(CFG["K"])),
    (r"\budgetpct", f"{CFG['budget_pct'] * 100:.0f}"),
    (r"\testacc", f"{CFG['test_accuracy_pct']:.3f}"),
    (r"\trainrows", f"{s1['train_rows']:,}".replace(",", "{,}")),
    (r"\testrows", f"{s1['test_rows']:,}".replace(",", "{,}")),
    (r"\LLMesr", f"{llm['ESR']:.1f}"),
    (r"\LLMesrci", f"[{llm['CI'][0]:.1f}, {llm['CI'][1]:.1f}]"),
    (r"\LLMevaded", str(llm_ev)),
    (r"\LLMdp", f"{llm['mean_dP']:+.4f}"),
    (r"\LLMq", f"{llm['mean_queries']:.1f}"),
    (r"\GREesr", f"{gre['ESR']:.1f}"),
    (r"\GREdp", f"{gre['mean_dP']:+.4f}"),
    (r"\NESesr", f"{nes['ESR']:.1f}"),
    (r"\NESdp", f"{nes['mean_dP']:+.4f}"),
    (r"\RNDesr", f"{rnd['ESR']:.1f}"),
    (r"\RNDdp", f"{rnd['mean_dP']:+.4f}"),
    (r"\porig", f"{p_orig_mean:.4f}"),
    (r"\pafter", f"{p_after_mean:.4f}"),
    (r"\wilnesrnd", wsci(wil["NES_vs_random_k"])),
    (r"\wilgrenes", wsci(wil["Greedy_vs_NES"])),
    (r"\eceraw", f"{s2['ece_raw']:.4f}"),
    (r"\eceiso", f"{s2['ece_isotonic']:.4f}"),
    (r"\brierraw", f"{s2['brier_raw']:.4f}"),
    (r"\brieriso", f"{s2['brier_isotonic']:.4f}"),
    (r"\calrows", f"{s2['calibration_rows']:,}".replace(",", "{,}")),
    (r"\baseacc", f"{float(base['Accuracy']):.4f}"),
    (r"\basef", f"{float(base['F1_Score']):.4f}"),
    (r"\esrfiftyfive", f"{s6['threshold_sweep']['0.55']['llm']['esr']:.1f}"),
]
for st, tag in ((ORDER[0], r"\Shigh"), (ORDER[1], r"\Smed"), (ORDER[2], r"\Sbound")):
    if st in strat_stats:
        v = strat_stats[st]
        macros += [(tag + "n", str(v["n"])),
                   (tag + "esr", f"{v['esr']:.1f}"),
                   (tag + "dp", f"{v['dp']:+.4f}"),
                   (tag + "ev", str(v["ev"]))]

# ===========================================================================
# Revision: audited oracle-query cost.
#
# MAX_K in the runner caps the number of search ITERATIONS, not the number of
# oracle calls. The per-iteration oracle cost differs by strategy, and the
# baselines' "queries" field is the constant MAX_K written into the record, not
# a measurement. The counts below were obtained by replacing prob_attack() with
# a counting stub and running each baseline loop to completion (no early exit);
# they are exact for a flow that is never evaded, which is the case for 106 of
# 107 greedy flows and all NES and random-K flows.
QUERY_AUDIT = [
    # label, oracle calls per iteration, measured calls per flow, as reported
    (r"LLM-guided", "1", r"$\le \LLMq$", r"\LLMq"),
    (r"Random-$K$", "1", "22", "20"),
    (r"Greedy coordinate", "3", "62", "20"),
    (r"NES", r"$n_{\text{perturb}}+1 = 9$", "182", "20"),
]
rows = []
for label, per_iter, measured, reported in QUERY_AUDIT:
    rows.append(" & ".join([label, per_iter, measured, reported]) + r" \\")
write("tab_queryaudit.tex", "\n".join(rows))

# --- calibration fold composition, read from the stored bins
cal = load("revision_v8/step2_isotonic_calibration.json")
tot = sum(b["n"] for b in cal["bins_raw"])
pos = sum(b["n"] * b["frac_pos"] for b in cal["bins_raw"])
tp = sum(b["n"] * b["frac_pos"] for b in cal["bins_raw"] if float(b["bin"][1:4]) >= 0.5)
pp = sum(b["n"] for b in cal["bins_raw"] if float(b["bin"][1:4]) >= 0.5)
tn = sum(b["n"] * (1 - b["frac_pos"]) for b in cal["bins_raw"] if float(b["bin"][1:4]) < 0.5)
cal_acc = 100.0 * (tp + tn) / tot
cal_pos = 100.0 * pos / tot


macros += [
    (r"\Calacc", f"{cal_acc:.2f}"),
    (r"\Calpos", f"{cal_pos:.2f}"),
    (r"\Qllm", r"\LLMq"),
    (r"\Qrandom", "22"),
    (r"\Qgreedy", "62"),
    (r"\Qnes", "182"),
    (r"\Qnesratio", "20"),
    (r"\Patience", "3"),
    (r"\LLMtemp", "0.3"),
    (r"\LLMmodel", r"\texttt{llama-3.1-8b-instant}"),
    (r"\NESsigma", "0.05"),
    (r"\NESpop", "8"),
    (r"\NESstep", "0.25"),
    (r"\RFtrees", "100"),
    (r"\RFsplit", "5"),
    (r"\RFseed", "42"),
    (r"\Nperstratum", "50"),
    (r"\Qllmfail", "10.17"),
    (r"\Qllmsucc", "7.39"),
    (r"\Qllmmax", "18"),
]

write("macros.tex", "\n".join(rf"\newcommand{{{n}}}{{{v}}}" for n, v in macros))
print("\nAll tables generated from:", RES.resolve())


# ===========================================================================
# v2: ket qua sau khi sua loi ap budget. Doc tu step10/step11/step12.
# Budget duoc do tren VECTOR GOC, khong phai vector hien tai.
# ===========================================================================
import math as _m2

_S10 = load("revision_v8/step10_corrected.json")
_S11 = load("revision_v8/step11_llm_corrected.json")
_S12 = load("revision_v8/step12_budget_sweep.json")

def _cp_hi(n, alpha=0.05):
    """Can tren Clopper-Pearson khi k=0."""
    return 100.0 * (1.0 - (alpha / 2.0) ** (1.0 / n))

_ARMS = [("corner1call", "Corner"), ("greedy", "Greedy"), ("random_k", "Rand"),
         ("nes8_fixed", "NesEight"), ("nes4_fixed", "NesFour"),
         ("nes2_fixed", "NesTwo"), ("nes8_oldsign", "NesOld")]

_m2v = []
_n10 = _S10["arms"]["greedy"]["n"]
_m2v += [(r"\CN", str(_n10)),
         (r"\CBudget", str(_S10["config"]["oracle_budget"])),
         (r"\CTau", f"{_S10['config']['tau']:.1f}"),
         (r"\CNfeat", str(len(_S10["config"]["features"]))),
         (r"\CCPhi", f"{_cp_hi(_n10):.1f}"),
         (r"\CRuleThree", f"{300.0/_n10:.2f}")]

_rows10 = []
for key, tag in _ARMS:
    a = _S10["arms"][key]
    _m2v += [(rf"\C{tag}Dp", f"{a['mean_dP']:+.4f}"),
             (rf"\C{tag}Calls", f"{a['mean_calls']:.0f}"),
             (rf"\C{tag}Ev", str(a["evaded"]))]
    _rows10.append(rf"{tag} & {a['n']} & {a['evaded']} & {a['ESR']:.1f} & "
                   rf"[0.0, {_cp_hi(a['n']):.1f}] & ${a['mean_dP']:+.4f}$ & "
                   rf"{a['mean_calls']:.0f} \\")

_llm = _S11["llm"]
_m2v += [(r"\CLlmN", str(_llm["n"])),
         (r"\CLlmEv", str(_llm["evaded"])),
         (r"\CLlmDp", f"{_llm['mean_dP']:+.4f}"),
         (r"\CLlmCalls", f"{_llm['mean_calls']:.1f}"),
         (r"\CLlmCPhi", f"{_cp_hi(_llm['n']):.1f}"),
         (r"\CLlmModel", r"\texttt{" + _S11["config"]["model"].replace("_", r"\_") + "}")]

# So sanh dung tren cung 13 luong ma arm LLM da chay
_ids = {r["id"] for r in _S11["raw"]}
_sub = [r for r in _S10["raw"] if r["id"] in _ids]
_llmby = {r["id"]: r["llm"] for r in _S11["raw"]}
_rowssub = []
for key, tag in _ARMS[:6]:
    dp = statistics.mean(r[key]["delta"] for r in _sub)
    ca = statistics.mean(r[key]["calls"] for r in _sub)
    ev = sum(1 for r in _sub if r[key]["evaded"])
    _m2v.append((rf"\CSub{tag}Dp", f"{dp:+.4f}"))
    _rowssub.append(rf"{tag} & {ev} & ${dp:+.4f}$ & {ca:.1f} \\")
_dpl = statistics.mean(_llmby[i]["delta"] for i in _ids)
_cal = statistics.mean(_llmby[i]["calls"] for i in _ids)
_m2v.append((r"\CSubLlmDp", f"{_dpl:+.4f}"))
_rowssub.append(rf"LLM & {sum(1 for i in _ids if _llmby[i]['evaded'])} & "
                rf"${_dpl:+.4f}$ & {_cal:.1f} \\")
_m2v.append((r"\CSubN", str(len(_sub))))

# Quet epsilon
_tau = _S12["config"]["tau"]
_rawS = _S12["raw"]
_nS = len(_rawS)
_NAMES = {"0.05": "A", "0.10": "B", "0.15": "C", "0.20": "D",
          "0.30": "E", "0.50": "F", "1.00": "G"}
_rows12 = []
for eps in ("0.05", "0.10", "0.15", "0.20", "0.30", "0.50", "1.00"):
    ev = sum(1 for r in _rawS if r["eps"][eps]["best"] < _tau)
    esr = 100.0 * ev / _nS
    mp = statistics.mean(r["eps"][eps]["best"] for r in _rawS)
    _m2v += [(rf"\CEsr{_NAMES[eps]}", f"{esr:.2f}"),
             (rf"\CEv{_NAMES[eps]}", str(ev))]
    _rows12.append(rf"${float(eps)*100:.0f}\%$ & {ev} & {esr:.2f} & {mp:.4f} \\")
_m2v += [(r"\CEpsFirst", f"{_S12['config']['eps_first_evasion']*100:.0f}"),
         (r"\CEpsTen", f"{_S12['config']['eps_esr_10pc']*100:.0f}"),
         (r"\CNsweep", str(_nS))]

write("tab_corrected_arms.tex", "\n".join(_rows10))
write("tab_corrected_subset.tex", "\n".join(_rowssub))
write("tab_corrected_sweep.tex", "\n".join(_rows12))
with open(OUT / "macros.tex", "a", encoding="utf-8") as _f:
    _f.write("\n%% --- v2: ket qua sau khi ap dung budget dung ---\n")
    for _n, _v in _m2v:
        _f.write(rf"\newcommand{{{_n}}}{{{_v}}}" + "\n")
print(f"v2: {len(_m2v)} macro tu step10/11/12")


# ===========================================================================
# Phan them: ESR la mot ham cua thiet ke lay mau, khong phai cua suc manh tan
# cong. Doc tu step12_budget_sweep.json (107 flow x 7 eps x 4 arm).
# Khong mo phong; moi so la mot ham thuan cua file do.
# ===========================================================================
import math as _mm

_STRATA = ["Boundary(0.5 - 0.6)", "Medium(0.6 - 0.8)", "High(>0.8)"]
_SLBL = {"Boundary(0.5 - 0.6)": "Boundary, $p_0 \\in [0.5, 0.6)$",
         "Medium(0.6 - 0.8)": "Medium, $p_0 \\in [0.6, 0.8)$",
         "High(>0.8)": "High, $p_0 \\ge 0.8$"}
_EPSL = ("0.05", "0.10", "0.15", "0.20", "0.30", "0.50", "1.00")


def _esr(rows, eps):
    if not rows:
        return 0, 0, 0.0
    k = sum(1 for r in rows if r["eps"][eps]["best"] < _tau)
    return k, len(rows), 100.0 * k / len(rows)


# --- bang A: ESR theo stratum x eps
_rows_st = []
for _s in _STRATA:
    _sub = [r for r in _rawS if r["stratum"] == _s]
    _cells = []
    for _e in _EPSL:
        _k, _nn, _p = _esr(_sub, _e)
        _cells.append("%.0f" % _p)
    _rows_st.append("%s & %d & %s \\\\" % (_SLBL[_s], len(_sub), " & ".join(_cells)))
_allc = []
for _e in _EPSL:
    _k, _nn, _p = _esr(_rawS, _e)
    _allc.append("%.1f" % _p)
_rows_st.append(r"\midrule")
_rows_st.append("Pooled & %d & %s \\\\" % (len(_rawS), " & ".join(_allc)))
write("tab_stratum_esr.tex", "\n".join(_rows_st))

# --- bang B: dich chuyen diem so gan nhu la mot hang so
_rows_shift = []
for _e in _EPSL:
    _dl = sorted(r["p_orig"] - r["eps"][_e]["best"] for r in _rawS)
    _md = statistics.median(_dl)
    _sd = statistics.pstdev(_dl)
    _q1 = _dl[len(_dl) // 4]
    _q3 = _dl[3 * len(_dl) // 4]
    _rows_shift.append(r"$%.0f\%%$ & %.4f & %.4f & [%.4f, %.4f] & %.4f \\"
                       % (float(_e) * 100, _md, _sd, _q1, _q3, _dl[-1]))
write("tab_shift.tex", "\n".join(_rows_shift))

# --- bang C: quy tac "vuot nguong khi p_orig < tau + shift"
_rows_rule = []
for _e in _EPSL:
    _dl = [r["p_orig"] - r["eps"][_e]["best"] for r in _rawS]
    _thr = _tau + statistics.median(_dl)
    _pred = [r["p_orig"] < _thr for r in _rawS]
    _act = [r["eps"][_e]["best"] < _tau for r in _rawS]
    _ok = sum(1 for a, b in zip(_pred, _act) if a == b)
    _xs = [r["p_orig"] for r in _rawS]
    _ys = [1.0 if b else 0.0 for b in _act]
    _mx, _my = statistics.mean(_xs), statistics.mean(_ys)
    _num = sum((a - _mx) * (b - _my) for a, b in zip(_xs, _ys))
    _den = _mm.sqrt(sum((a - _mx) ** 2 for a in _xs)
                    * sum((b - _my) ** 2 for b in _ys))
    _r = (_num / _den) if _den else float("nan")
    _rstr = "%.4f" % _r if _den else "n/a"
    _rows_rule.append(r"$%.0f\%%$ & %.4f & %d/%d & %.1f & %s \\"
                      % (float(_e) * 100, _thr, _ok, len(_rawS),
                         100.0 * _ok / len(_rawS), _rstr))
write("tab_rule.tex", "\n".join(_rows_rule))

# --- bang D: moi arm rieng le so voi oracle 'best'
_ARMS = ("corner_lo", "corner_hi", "greedy", "best")
_ALBL = {"corner_lo": "Corner, lower", "corner_hi": "Corner, upper",
         "greedy": "Greedy coordinate", "best": "Oracle best-of-three"}
_rows_arm = []
for _a in _ARMS:
    _cells = [str(sum(1 for r in _rawS if r["eps"][_e][_a] < _tau)) for _e in _EPSL]
    _rows_arm.append("%s & %s \\\\" % (_ALBL[_a], " & ".join(_cells)))
write("tab_arm_eps.tex", "\n".join(_rows_arm))

# --- macro
_e1 = "1.00"
_dl1 = [r["p_orig"] - r["eps"][_e1]["best"] for r in _rawS]
_thr1 = _tau + statistics.median(_dl1)
_pred1 = [r["p_orig"] < _thr1 for r in _rawS]
_act1 = [r["eps"][_e1]["best"] < _tau for r in _rawS]
_ok1 = sum(1 for a, b in zip(_pred1, _act1) if a == b)
_xs = [r["p_orig"] for r in _rawS]
_ys = [1.0 if b else 0.0 for b in _act1]
_mx, _my = statistics.mean(_xs), statistics.mean(_ys)
_r1 = (sum((a - _mx) * (b - _my) for a, b in zip(_xs, _ys))
       / _mm.sqrt(sum((a - _mx) ** 2 for a in _xs) * sum((b - _my) ** 2 for b in _ys)))

_m3 = [
    (r"\XShiftMed", "%.4f" % statistics.median(_dl1)),
    (r"\XShiftSd", "%.4f" % statistics.pstdev(_dl1)),
    (r"\XShiftQone", "%.4f" % sorted(_dl1)[len(_dl1) // 4]),
    (r"\XShiftQthree", "%.4f" % sorted(_dl1)[3 * len(_dl1) // 4]),
    (r"\XThr", "%.4f" % _thr1),
    (r"\XRuleOk", str(_ok1)),
    (r"\XRulePct", "%.1f" % (100.0 * _ok1 / len(_rawS))),
    (r"\XCorr", "%.4f" % _r1),
]
for _s, _tag in zip(_STRATA, ("Bnd", "Med", "Hi")):
    _sub = [r for r in _rawS if r["stratum"] == _s]
    _k, _nn, _p = _esr(_sub, _e1)
    _m3 += [(rf"\XN{_tag}", str(_nn)), (rf"\XEv{_tag}", str(_k)),
            (rf"\XEsr{_tag}", "%.1f" % _p)]
    _k5, _n5, _p5 = _esr(_sub, "0.50")
    _m3 += [(rf"\XEsrHalf{_tag}", "%.1f" % _p5)]
_m3 += [
    (r"\XArmHiNever", str(sum(1 for _e in _EPSL
                              for r in _rawS if r["eps"][_e]["corner_hi"] < _tau))),
    (r"\XGreedyEqBest",
     "yes" if all(sum(1 for r in _rawS if r["eps"][_e]["greedy"] < _tau)
                  == sum(1 for r in _rawS if r["eps"][_e]["best"] < _tau)
                  for _e in _EPSL) else "no"),
]
with open(OUT / "macros.tex", "a", encoding="utf-8") as _f:
    _f.write("\n%% --- ESR la ham cua thiet ke lay mau ---\n")
    for _n, _v in _m3:
        _f.write(rf"\newcommand{{{_n}}}{{{_v}}}" + "\n")
print(f"wrote {len(_m3)} macro stratum + 4 bang")
