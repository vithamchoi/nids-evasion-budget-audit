"""Bai 02, buoc 10: chay lai baseline voi NGAN SACH DUNG va NES SUA DAU.

Hai loi trong ban cu:
  1. step4_llm_v8.py goi enforce_budget(feat, raw) voi `feat` la dict ma chinh
     vong lap do sua (feat[f] = v), nen hop +-10% DOI TAM moi vong. Sau 18 vong
     nhanh LLM di duoc toi 5,56x gia tri goc, trong khi NES va random-K bi giu
     dung 0,9x-1,1x. So sanh cua ca bai bao khong ton tai.
  2. nes_attack cong don `grads` theo huong LAM GIAM xac suat (huong xuong doc)
     roi buoc di `value - sign(grad)*delta`, tuc di NGUOC len doc. Du lieu xac
     nhan: NES lam TANG xac suat o 72/107 flow.

Script nay: moi arm do ngan sach tren vector GOC bat bien, tran tinh theo SO LAN
GOI DETECTOR (khong phai so vong lap), NES sua dau va them do nhay n_perturb,
va them mot phep thu 1 LAN GOI: dat ca 10 feature ve goc -10%.

Chi can CPU va dataset. KHONG can GROQ_API_KEY.
Ghi ra results/revision_v8/step10_corrected.json, khong de len file cu.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (BUDGET_PCT, NEAR_ZERO_ABS, clopper_pearson_ci,
                          load_temporal_frames, prepare_xy, prob_attack,
                          project_root, stratum, train_rf)

ORACLE_BUDGET = int(os.getenv("P02_ORACLE_BUDGET", "20"))
TAU = 0.5


def box(orig):
    """Hop +-10% quanh gia tri GOC. Xu ly ca gia tri am (ban cu bi lo>hi)."""
    if abs(orig) < NEAR_ZERO_ABS:
        return orig - NEAR_ZERO_ABS, orig + NEAR_ZERO_ABS
    lo, hi = orig * (1 - BUDGET_PCT), orig * (1 + BUDGET_PCT)
    return (lo, hi) if lo <= hi else (hi, lo)          # <- sua loi gia tri am


def clip_to_original(original, proposal):
    out = {}
    for k, o in original.items():
        lo, hi = box(o)
        out[k] = float(np.clip(proposal.get(k, o), lo, hi))
    return out


class Oracle:
    class Done(Exception): pass
    def __init__(self, model, cols, budget):
        self.m, self.c, self.b, self.used = model, cols, budget, 0
    def __call__(self, vec):
        if self.used >= self.b: raise Oracle.Done()
        self.used += 1
        return prob_attack(self.m, vec, self.c)


def corner(oracle, orig_vec, original, p0):
    """Phep thu re nhat co the: dat het 10 feature ve goc duoi cua hop. 1 lan goi."""
    trial = orig_vec.copy()
    for k, o in original.items():
        trial[k] = box(o)[0]
    try:
        p = oracle(trial)
    except Oracle.Done:
        return p0
    return min(p0, p)


def greedy(oracle, orig_vec, original, p0):
    vec, best = orig_vec.copy(), p0
    cur = dict(original)
    step = 0
    try:
        while True:
            name = list(original)[step % len(original)]; step += 1
            lo, hi = box(original[name])              # <- luon quanh GOC
            for cand in (lo, hi):
                t = vec.copy(); t[name] = cand
                p = oracle(t)
                if p < best:
                    best, vec = p, t; cur[name] = cand
                if best < TAU: return vec, best
    except Oracle.Done:
        return vec, best


def random_k(oracle, orig_vec, original, rng, p0):
    vec, best = orig_vec.copy(), p0
    try:
        while True:
            prop = {}
            for k, o in original.items():
                lo, hi = box(o); prop[k] = rng.uniform(lo, hi)
            t = orig_vec.copy()
            for k, v in clip_to_original(original, prop).items(): t[k] = v
            p = oracle(t)
            if p < best: best, vec = p, t
            if best < TAU: return vec, best
    except Oracle.Done:
        return vec, best


def nes(oracle, orig_vec, original, rng, p0, n_perturb, sigma=0.05, step_frac=0.25):
    """Dau da SUA: di THEO huong uoc luong, khong nguoc lai."""
    vec, cur, best = orig_vec.copy(), dict(original), p0
    try:
        while True:
            grads = {k: 0.0 for k in cur}
            for _ in range(n_perturb):
                noise = {k: rng.normal(0, max(abs(v), NEAR_ZERO_ABS) * sigma)
                         for k, v in cur.items()}
                t = vec.copy()
                for k, v in clip_to_original(original,
                        {a: cur[a] + noise[a] for a in cur}).items(): t[k] = v
                p = oracle(t)
                sign = 1.0 if p < best else -1.0
                for k in cur: grads[k] += sign * noise[k]
                if p < TAU: return t, p
            prop = {}
            for k, v in cur.items():
                g = grads[k] / n_perturb
                d = (NEAR_ZERO_ABS * 0.5 if abs(v) < NEAR_ZERO_ABS
                     else abs(v) * BUDGET_PCT * step_frac)
                prop[k] = v + np.sign(g) * d          # <- SUA DAU: cong, khong tru
            for k, v in clip_to_original(original, prop).items():
                vec[k] = v; cur[k] = v
            p = oracle(vec)
            if p < best: best = p
            if best < TAU: return vec, best
    except Oracle.Done:
        return vec, best


ARMS = [
    ("corner1call",  lambda o, v, f, r, p: (None, corner(o, v, f, p))),
    ("greedy",       lambda o, v, f, r, p: greedy(o, v, f, p)),
    ("random_k",     lambda o, v, f, r, p: random_k(o, v, f, r, p)),
    ("nes8_fixed",   lambda o, v, f, r, p: nes(o, v, f, r, p, 8)),
    ("nes4_fixed",   lambda o, v, f, r, p: nes(o, v, f, r, p, 4)),
    ("nes2_fixed",   lambda o, v, f, r, p: nes(o, v, f, r, p, 2)),
    ("nes8_oldsign", lambda o, v, f, r, p: nes_oldsign(o, v, f, r, p, 8)),
]


def nes_oldsign(oracle, orig_vec, original, rng, p0, n_perturb):
    """Ban CU sai dau, giu lai de chung minh loi la co that."""
    vec, cur, best = orig_vec.copy(), dict(original), p0
    try:
        while True:
            grads = {k: 0.0 for k in cur}
            for _ in range(n_perturb):
                noise = {k: rng.normal(0, max(abs(v), NEAR_ZERO_ABS) * 0.05)
                         for k, v in cur.items()}
                t = vec.copy()
                for k, v in clip_to_original(original,
                        {a: cur[a] + noise[a] for a in cur}).items(): t[k] = v
                p = oracle(t)
                sign = 1.0 if p < best else -1.0
                for k in cur: grads[k] += sign * noise[k]
                if p < TAU: return t, p
            prop = {k: (v - np.sign(grads[k] / n_perturb) *
                        (NEAR_ZERO_ABS * 0.5 if abs(v) < NEAR_ZERO_ABS
                         else abs(v) * BUDGET_PCT * 0.25))
                    for k, v in cur.items()}
            for k, v in clip_to_original(original, prop).items():
                vec[k] = v; cur[k] = v
            p = oracle(vec)
            if p < best: best = p
            if best < TAU: return vec, best
    except Oracle.Done:
        return vec, best


def summarise(rows, key):
    n = len(rows); k = sum(1 for r in rows if r[key]["evaded"])
    lo, hi = clopper_pearson_ci(k, n)
    return {"evaded": k, "n": n, "ESR": round(100*k/n, 1),
            "CI": [round(100*lo, 1), round(100*hi, 1)],
            "mean_dP": round(float(np.mean([r[key]["delta"] for r in rows])), 4),
            "mean_calls": round(float(np.mean([r[key]["calls"] for r in rows])), 1)}


def main():
    root = project_root()
    out_dir = Path(root) / "results" / "revision_v8"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step10_corrected.json"
    meta = json.loads((out_dir / "v8_sample_indices.json").read_text(encoding="utf-8"))
    selected, top10 = meta["indices"], meta["top10"]

    print("=" * 68)
    print(f"BAI 02 - BUOC 10: ngan sach do tren vector GOC, tran {ORACLE_BUDGET} lan goi detector")
    print("=" * 68)
    print("\n[1/3] Tai du lieu + huan luyen RF (lan dau se tai dataset tu HuggingFace) ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr); X_te, _, lbl = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr); cols = X_tr.columns

    rows = []
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8")).get("raw", [])
        done = {r["id"] for r in rows}
        selected = [i for i in selected if i not in done]
        print(f"  Tiep tuc: da xong {len(rows)}, con {len(selected)}")

    print(f"\n[2/3] Tan cong {len(selected)} flow, {len(ARMS)} nhanh ...")
    t0 = time.time()
    for n_done, idx in enumerate(selected, 1):
        orig = X_te.iloc[idx].copy()
        original = {f: float(orig[f]) for f in top10}
        p0 = prob_attack(model, orig, cols)
        rec = {"id": int(idx), "p_orig": round(p0, 4), "stratum": stratum(p0),
               "type": str(lbl.iloc[idx]) if lbl is not None else "unknown"}
        for name, fn in ARMS:
            rng = np.random.default_rng(99 + idx)
            o = Oracle(model, cols, 1 if name == "corner1call" else ORACLE_BUDGET)
            _, pf = fn(o, orig.copy(), dict(original), rng, p0)
            rec[name] = {"evaded": bool(pf < TAU), "prob": round(float(pf), 4),
                         "delta": round(float(pf - p0), 4), "calls": o.used}
        rows.append(rec)
        out_path.write_text(json.dumps(
            {"config": {"oracle_budget": ORACLE_BUDGET, "tau": TAU, "features": top10,
                        "note": "budget measured against ORIGINAL vector; NES sign fixed"},
             "arms": {k: summarise(rows, k) for k, _ in ARMS}, "raw": rows},
            indent=2), encoding="utf-8")
        if n_done % 10 == 0 or n_done == 1:
            print(f"  [{len(rows):3d}] id={idx} p0={p0:.3f} " +
                  " ".join(f"{k}={'Y' if rec[k]['evaded'] else 'N'}" for k, _ in ARMS) +
                  f"  ({time.time()-t0:.0f}s)")

    print("\n[3/3] Ket qua")
    for k, _ in ARMS:
        s = summarise(rows, k)
        print(f"  {k:13s} ESR={s['ESR']:5.1f}% CI={s['CI']} dP={s['mean_dP']:+.4f} "
              f"calls={s['mean_calls']:.1f}/{ORACLE_BUDGET}")
    print(f"\nDa ghi -> {out_path}")
    print("\nSO SANH VOI BAI CU: greedy 0,9%  NES 0,0%  random 0,0%  LLM 33,6%")
    print("Neu nes8_fixed > nes8_oldsign thi loi dau la co that.")
    print("Neu corner1call da danh bai ca ba baseline thi bai bao can viet lai.")


if __name__ == "__main__":
    main()
