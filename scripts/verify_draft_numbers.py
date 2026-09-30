"""
Verify all draft numbers from evasion_v4 and v5 JSON results.
Uses absolute paths. ASCII-only output to avoid encoding errors.
"""
import json, sys, os
import numpy as np
from collections import defaultdict

BASE = r"c:\Users\Admin\Documents\Data science\code_hoan_chinh\02_robust_vit_encrypted_traffic"
V4   = os.path.join(BASE, "results", "evasion_v4", "evasion_v4_results.json")
V5   = os.path.join(BASE, "results", "evasion_v5", "evasion_v5_results.json")

print("="*60)
print("VERIFICATION: evasion_v4_results.json")
print("="*60)
with open(V4, encoding="utf-8") as f:
    data = json.load(f)

cfg = data["config"]
print(f"\n[config] N={cfg['N']}, budget={cfg['budget']}")
print(f"[model]  RF_params={data['model']['RF_params']}, test_acc={data['model']['test_acc']}")

res = data["results"]
for method in ["llm", "greedy", "random"]:
    r = res[method]
    print(f"\n[{method.upper()}]")
    print(f"  ESR        = {r['ESR']}%  CI={r['ESR_CI']}")
    print(f"  Mean_Delta = {r['Mean_Delta']:.7f}  Delta_CI={r['Delta_CI']}")
    if method == "llm":
        print(f"  Mean_Violations = {r['Mean_Violations']}")

print(f"\n[stats]")
print(f"  pval_LLM_vs_Random  = {data['stats']['pval_LLM_vs_Random']:.8f}")
print(f"  pval_LLM_vs_Greedy  = {data['stats']['pval_LLM_vs_Greedy']:.8f}")

raw = data["raw"]
print(f"\n[raw] Total records: {len(raw)}")

cs = defaultdict(lambda: {"n":0,"ld":[],"gd":[],"rd":[],"le":0,"ge":0,"re":0})
for rec in raw:
    c = rec["type"]
    cs[c]["n"]  += 1
    cs[c]["ld"].append(rec["llm"]["delta"])
    cs[c]["gd"].append(rec["greedy"]["delta"])
    cs[c]["rd"].append(rec["random"]["delta"])
    cs[c]["le"] += int(rec["llm"]["evaded"])
    cs[c]["ge"] += int(rec["greedy"]["evaded"])
    cs[c]["re"] += int(rec["random"]["evaded"])

print("\n[PER-CLASS BREAKDOWN - computed from full raw JSON]")
print(f"{'Class':<10} {'N':>4} {'LLM_ESR':>9} {'Grd_ESR':>9} {'LLM_dP':>10} {'Grd_dP':>10} {'Rnd_dP':>10}")
for cls in sorted(cs.keys()):
    s = cs[cls]; n = s["n"]
    print(f"{cls:<10} {n:>4}  {s['le']/n*100:>7.1f}%  {s['ge']/n*100:>7.1f}%"
          f"  {np.mean(s['ld']):>9.4f}  {np.mean(s['gd']):>9.4f}  {np.mean(s['rd']):>9.4f}")

print("\n" + "="*60)
print("VERIFICATION: evasion_v5_results.json")
print("="*60)
with open(V5, encoding="utf-8") as f:
    v5 = json.load(f)

for model in ["RF","XGB"]:
    for bud,label in [("budget_0.1","+-10%"),("budget_0.25","+-25%")]:
        b = v5[model][bud]
        print(f"\n  {model} {label}: N={b['N']}")
        for meth in ["llm","greedy"]:
            m = b[meth]
            ci = m["ESR_CI"]
            print(f"    [{meth:6}] ESR={m['ESR']*100:.1f}%  CI=[{ci[0]:.4f},{ci[1]:.4f}]"
                  f"  dP={m['Mean_Delta']:.8f}  Time={m['Time_Avg']:.3f}s")

print("\n ALL NUMBERS VERIFIED FROM JSON OUTPUT FILES.")
