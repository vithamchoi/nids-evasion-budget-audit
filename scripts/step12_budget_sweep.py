"""Bai 02, buoc 12: quet NGAN SACH.

Buoc 10 cho ESR = 0% tren MOI nhanh, ke ca corner1call va greedy. Do khong
phai vi cac tan cong yeu, ma vi hop +-10% quanh vector goc khong chua diem
nao vuot nguong. Gia tri p thap nhat dat duoc tren 107 flow la 0.507, thieu
0.007 nua la cham nguong 0.5.

Mot ket qua "khong cai nao chay duoc" kem tran duy nhat la hieu ung san, va
reviewer se noi dung nhu vay. Cach xu ly la bien no thanh mot duong cong:
quet epsilon va tim nguong ma tan cong bat dau kha thi.

Toan bo chay bang CPU, khong goi LLM, khong ton API. Vai phut la xong.

Ghi: results/revision_v8/step12_budget_sweep.json
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_lib import (NEAR_ZERO_ABS, clopper_pearson_ci,
                          load_temporal_frames, prepare_xy, prob_attack,
                          project_root, stratum, train_rf)

TAU = 0.5
ORACLE_BUDGET = int(os.getenv("P02_ORACLE_BUDGET", "20"))
SWEEP = [float(x) for x in os.getenv(
    "P02_SWEEP", "0.05,0.10,0.15,0.20,0.30,0.50,1.00").split(",")]


def box(orig, pct):
    if abs(orig) < NEAR_ZERO_ABS:
        return orig - NEAR_ZERO_ABS, orig + NEAR_ZERO_ABS
    lo, hi = orig * (1 - pct), orig * (1 + pct)
    return (lo, hi) if lo <= hi else (hi, lo)


class Oracle:
    class Done(Exception): pass
    def __init__(self, model, cols, budget):
        self.m, self.c, self.b, self.used = model, cols, budget, 0
    def __call__(self, vec):
        if self.used >= self.b: raise Oracle.Done()
        self.used += 1
        return prob_attack(self.m, vec, self.c)


def corner(oracle, orig_vec, original, p0, pct, lower=True):
    """Dat het feature ve mot goc cua hop. 1 lan goi."""
    t = orig_vec.copy()
    for k, o in original.items():
        t[k] = box(o, pct)[0 if lower else 1]
    try:
        return min(p0, oracle(t))
    except Oracle.Done:
        return p0


def greedy(oracle, orig_vec, original, p0, pct):
    """Quet toa do: moi feature thu ca hai dau, giu dau nao ha p."""
    vec, best = orig_vec.copy(), p0
    step = 0
    try:
        while True:
            name = list(original)[step % len(original)]; step += 1
            lo, hi = box(original[name], pct)
            for cand in (lo, hi):
                t = vec.copy(); t[name] = cand
                p = oracle(t)
                if p < best:
                    best, vec = p, t
                if best < TAU:
                    return best
    except Oracle.Done:
        return best


def summarise(vals, key):
    n = len(vals); k = sum(1 for v in vals if v[key] < TAU)
    lo, hi = clopper_pearson_ci(k, n)
    probs = [v[key] for v in vals]
    return {"evaded": k, "n": n, "ESR": round(100*k/n, 1),
            "CI": [round(100*lo, 1), round(100*hi, 1)],
            "p_min": round(min(probs), 4),
            "p_median": round(float(np.median(probs)), 4)}


def main():
    root = project_root()
    out_dir = Path(root) / "results" / "revision_v8"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step12_budget_sweep.json"
    meta = json.loads((out_dir / "v8_sample_indices.json").read_text(encoding="utf-8"))
    selected, top10 = meta["indices"], meta["top10"]

    print("=" * 70)
    print("BAI 02 - BUOC 12: quet ngan sach, tim nguong epsilon cho evasion")
    print("=" * 70)
    print("\n[1/3] Tai du lieu + huan luyen RF ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr); X_te, _, lbl = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr); cols = X_tr.columns

    # Doc lai tien trinh cu neu co, de tat may giua chung khong mat gi.
    raw = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("config", {}).get("sweep") == SWEEP:
                raw = prev.get("raw", [])
                done = {r["id"] for r in raw}
                selected = [i for i in selected if i not in done]
                print(f"\n  Tiep tuc: da xong {len(raw)} flow, con {len(selected)}")
            else:
                print("\n  File cu dung muc ngan sach khac, chay lai tu dau.")
        except Exception:
            raw = []

    print(f"\n[2/3] {len(selected)} flow x {len(SWEEP)} muc ngan sach, chi CPU ...")
    t0 = time.time()
    per_eps = {}

    def flush():
        pe = {}
        for pct in SWEEP:
            key = f"{pct:.2f}"
            vals = [r["eps"][key] for r in raw]
            pe[key] = {arm: summarise(vals, arm)
                       for arm in ("corner_lo", "corner_hi", "greedy", "best")}
        fa = ft = None
        for pct in SWEEP:
            e = pe[f"{pct:.2f}"]["best"]["ESR"]
            if fa is None and e > 0: fa = pct
            if ft is None and e >= 10: ft = pct
        out_path.write_text(json.dumps(
            {"config": {"tau": TAU, "oracle_budget": ORACLE_BUDGET, "sweep": SWEEP,
                        "features": top10, "n_flows": len(raw),
                        "eps_first_evasion": fa, "eps_esr_10pc": ft,
                        "note": "budget luon do tren vector GOC bat bien"},
             "per_eps": pe, "raw": raw}, indent=2), encoding="utf-8")
        return pe

    for idx in selected:
        orig = X_te.iloc[idx].copy()
        original = {f: float(orig[f]) for f in top10}
        p0 = prob_attack(model, orig, cols)
        rec = {"id": int(idx), "p_orig": round(p0, 4), "stratum": stratum(p0),
               "type": str(lbl.iloc[idx]) if lbl is not None else "unknown", "eps": {}}
        for pct in SWEEP:
            pl = corner(Oracle(model, cols, 1), orig.copy(), original, p0, pct, True)
            pu = corner(Oracle(model, cols, 1), orig.copy(), original, p0, pct, False)
            pg = greedy(Oracle(model, cols, ORACLE_BUDGET), orig.copy(), original, p0, pct)
            rec["eps"][f"{pct:.2f}"] = {
                "corner_lo": round(float(pl), 4),
                "corner_hi": round(float(pu), 4),
                "greedy": round(float(pg), 4),
                "best": round(float(min(pl, pu, pg)), 4)}
        raw.append(rec)
        per_eps = flush()          # ghi ngay sau moi flow, tat may khong mat
        if len(raw) % 20 == 0:
            print(f"  {len(raw)} flow xong ({time.time()-t0:.0f}s)")

    per_eps = flush()
    first_any = json.loads(out_path.read_text(encoding="utf-8"))["config"]["eps_first_evasion"]
    first_ten = json.loads(out_path.read_text(encoding="utf-8"))["config"]["eps_esr_10pc"]

    print("\n[3/3] Ket qua: ESR theo muc ngan sach (nhanh 'best' = tot nhat trong 3)")
    print(f"\n  {'eps':>6s} {'corner_lo':>10s} {'corner_hi':>10s} {'greedy':>8s} "
          f"{'best':>8s} {'p_min':>8s}")
    for pct in SWEEP:
        e = per_eps[f"{pct:.2f}"]
        print(f"  {pct*100:5.0f}% {e['corner_lo']['ESR']:9.1f}% {e['corner_hi']['ESR']:9.1f}% "
              f"{e['greedy']['ESR']:7.1f}% {e['best']['ESR']:7.1f}% {e['best']['p_min']:8.3f}")

    print()
    if first_any is None:
        print(f"  Khong muc nao trong {SWEEP} cho evasion. Detector rat vung,")
        print("  hoac feature duoc phep sua khong du de vuot nguong.")
    else:
        print(f"  Evasion bat dau kha thi tu eps = {first_any*100:.0f}%")
        if first_ten:
            print(f"  ESR dat 10% tu eps = {first_ten*100:.0f}%")
    print("\n  So sanh corner (1 lan goi) voi greedy (20 lan goi) o tung muc:")
    for pct in SWEEP:
        e = per_eps[f"{pct:.2f}"]
        c = max(e["corner_lo"]["ESR"], e["corner_hi"]["ESR"])
        g = e["greedy"]["ESR"]
        verdict = "greedy hon" if g > c + 1e-9 else ("bang nhau" if abs(g-c) < 1e-9 else "corner hon")
        print(f"    eps={pct*100:3.0f}%  corner={c:5.1f}%  greedy={g:5.1f}%  -> {verdict}")

    print(f"\nDa ghi -> {out_path}")


if __name__ == "__main__":
    main()
