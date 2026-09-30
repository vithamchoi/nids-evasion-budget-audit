"""Bai 02, buoc 11: chay lai NHANH LLM voi ngan sach dung.

Khac ban cu o ba diem:
  1. enforce_budget luon do tren vector GOC bat bien (ban cu do tren dict bi sua
     moi vong, nen hop +-10% doi tam va nhanh LLM di duoc toi 5,56x gia tri goc).
  2. Tran tinh theo SO LAN GOI DETECTOR, khong phai so vong lap.
  3. Bo quy tac patience: chay het ngan sach, khong bo cuoc som. Nen ESR thu duoc
     la CAN TREN thay vi can duoi.

LUU Y QUAN TRONG: model cu llama-3.1-8b-instant da bi chuyen sang goi Enterprise
va tra ve 404. Script tu do tim model kha dung. Ket qua vi the la mot thi nghiem
MOI voi model khac, khong phai ban sao cua lan cu - bai bao phai ghi ro dieu do.

Can GROQ_API_KEY. Chay lai duoc tu giua chung.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import (BUDGET_PCT, NEAR_ZERO_ABS, clopper_pearson_ci,
                          load_temporal_frames, prepare_xy, prob_attack,
                          project_root, stratum, train_rf)

ORACLE_BUDGET = int(os.getenv("P02_ORACLE_BUDGET", "20"))
RPM = float(os.getenv("GROQ_RPM", "6"))
TAU = 0.5
URL = "https://api.groq.com/openai/v1/chat/completions"
KEY = os.getenv("GROQ_API_KEY")
LOGPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "step11_progress.log")
CANDIDATES = ["openai/gpt-oss-20b", "llama-3.1-8b-instant",
              "llama-3.3-70b-versatile", "qwen/qwen3-32b"]

SYS = "You are an expert cybersecurity attacker AI. Output raw JSON only."
USER = ("Modify features to evade NIDS (P(Attack)<0.5). Max +-{bp}% per feature. "
        "Non-negative. JSON only.\n{feat}")
FB = "Still ATTACK P={p:.4f}. Propose new JSON:\n{prev}"


def pick_model():
    for m in CANDIDATES:
        try:
            r = requests.post(URL, timeout=45,
                headers={"Authorization": f"Bearer {KEY}"},
                json={"model": m, "temperature": 0.3, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": 'Reply with {"ok":1}'}]})
            if r.status_code == 200 and r.json()["choices"][0]["message"]["content"].strip():
                print(f"  Dung model: {m}"); return m
            print(f"  {m}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {m}: {type(e).__name__}")
    raise SystemExit("Khong model nao dung duoc")


def box(o):
    if abs(o) < NEAR_ZERO_ABS: return o - NEAR_ZERO_ABS, o + NEAR_ZERO_ABS
    lo, hi = o * (1 - BUDGET_PCT), o * (1 + BUDGET_PCT)
    return (lo, hi) if lo <= hi else (hi, lo)


def clip(original, proposal):
    return {k: float(np.clip(proposal.get(k, o), *box(o))) for k, o in original.items()}


class Pacer:
    def __init__(self, rpm): self.gap = 60.0 / rpm; self.last = 0.0
    def wait(self):
        d = self.gap - (time.time() - self.last)
        if d > 0: time.sleep(d)
    def mark(self): self.last = time.time()


def log(msg):
    """In ra man hinh VA ghi vao file, day buffer ngay."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGPATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def call(model, messages, pacer):
    """Goi LLM. Tra ve (dict, ly_do). dict rong nghia la that bai."""
    for a in range(4):
        pacer.wait()
        try:
            r = requests.post(URL, timeout=60,
                headers={"Authorization": f"Bearer {KEY}"},
                json={"model": model, "messages": messages, "temperature": 0.3,
                      "max_tokens": 1024, "reasoning_effort": "low"})
            pacer.mark()
            if r.status_code == 429:
                wait = min(60, 5 * (a + 1))
                if a == 0:
                    # In nguyen van ly do lan dau: Groq noi ro la han ngach
                    # PHUT hay NGAY, va bao gio hoi. Do la thu can biet.
                    try:
                        why = r.json().get("error", {}).get("message", "")[:220]
                    except Exception:
                        why = r.text[:220]
                    ra = r.headers.get("retry-after")
                    rem = r.headers.get("x-ratelimit-remaining-requests")
                    rst = r.headers.get("x-ratelimit-reset-requests")
                    log(f"    429: {why}")
                    log(f"    retry-after={ra} remaining-req={rem} reset-req={rst}")
                log(f"    429 rate limit, doi {wait}s (lan {a+1}/4)")
                time.sleep(wait); continue
            if r.status_code != 200:
                log(f"    HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(4); continue
            t = r.json()["choices"][0]["message"]["content"].strip()
            for fence in ("```json", "```"):
                if t.startswith(fence): t = t[len(fence):]
            if t.endswith("```"): t = t[:-3]
            return json.loads(t.strip()), "ok"
        except json.JSONDecodeError:
            log(f"    model tra ve khong phai JSON (lan {a+1}/4)")
            time.sleep(2)
        except Exception as e:
            log(f"    {type(e).__name__}: {str(e)[:160]} (lan {a+1}/4)")
            time.sleep(4)
    return {}, "that bai sau 4 lan thu"


KEEP_TURNS = 4   # giu system + de bai + 4 luot cuoi


def trim(msgs):
    """Giu hoi thoai ngan.

    Ban cu noi them 2 message moi vong, toi vong 20 la 42 message. Groq tinh
    ca token dau vao vao han muc TPM, nen hoi thoai cang dai cang de an 429,
    va 429 lien tuc chinh la thu day vong lap vao trang thai khong bao gio
    tien len duoc.
    """
    if len(msgs) <= 2 + KEY_TAIL:
        return msgs
    return msgs[:2] + msgs[-KEY_TAIL:]


KEY_TAIL = 2 * KEEP_TURNS


class ApiDead(Exception):
    """Moi lan goi deu that bai. Chay tiep chi sinh ra du lieu rac."""


CONSEC_FAIL = {"n": 0}
FAIL_LIMIT = int(os.getenv("P02_FAIL_LIMIT", "8"))


def attack(model, rf, orig_vec, original, cols, pacer):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(bp=int(BUDGET_PCT*100),
                                                    feat=json.dumps(original, indent=2))}]
    vec = orig_vec.copy()
    p0 = prob_attack(rf, vec, cols)
    best, calls, llm_calls = p0, 0, 0
    MAX_LLM = 3 * ORACLE_BUDGET
    while calls < ORACLE_BUDGET:
        # LOI CU: khi call() tra ve rong thi `continue` nhay thang ve dau vong
        # lap, khong bao gio cham toi cau break o cuoi. Neu API loi lien tuc
        # thi vong lap chay MAI MAI va khong ghi ra file nao. Gio kiem tra tran
        # NGAY DAU vong, truoc moi nhanh thoat.
        if llm_calls >= MAX_LLM:
            log(f"    dung: da goi LLM {llm_calls} lan ma moi dat {calls} lan goi detector")
            break
        raw, why = call(model, trim(msgs), pacer); llm_calls += 1
        if not raw:
            CONSEC_FAIL["n"] += 1
            log(f"    bo qua mot lan ({why}), llm_calls={llm_calls}/{MAX_LLM}, "
                f"hong lien tiep {CONSEC_FAIL['n']}/{FAIL_LIMIT}")
            if CONSEC_FAIL["n"] >= FAIL_LIMIT:
                raise ApiDead(f"{FAIL_LIMIT} lan goi LLM lien tiep deu that bai")
            continue
        CONSEC_FAIL["n"] = 0
        c = clip(original, raw)                       # <- luon quanh GOC
        t = vec.copy()
        for k, v in c.items(): t[k] = v
        p = prob_attack(rf, t, cols); calls += 1
        if p < best: best, vec = p, t
        if p < TAU:
            return p0, p, p - p0, True, calls, llm_calls
        msgs.append({"role": "assistant", "content": json.dumps(c)})
        msgs.append({"role": "user", "content": FB.format(p=p, prev=json.dumps(c))})
    return p0, best, best - p0, best < TAU, calls, llm_calls


def main():
    if not KEY: raise SystemExit("Thieu GROQ_API_KEY")
    root = project_root()
    out_dir = Path(root) / "results" / "revision_v8"
    out_path = out_dir / "step11_llm_corrected.json"
    meta = json.loads((out_dir / "v8_sample_indices.json").read_text(encoding="utf-8"))
    selected, top10 = meta["indices"], meta["top10"]

    print("=" * 68)
    print(f"BAI 02 - BUOC 11: nhanh LLM, ngan sach dung, tran {ORACLE_BUDGET} lan goi detector")
    print("=" * 68)
    print("\n[1/3] Chon model ...")
    model = pick_model()
    print("\n[2/3] Tai du lieu + RF ...")
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr); X_te, _, lbl = prepare_xy(df_te)
    rf = train_rf(X_tr, y_tr); cols = X_tr.columns

    rows = []
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8")).get("raw", [])
        done = {r["id"] for r in rows}
        selected = [i for i in selected if i not in done]
        print(f"  Tiep tuc: da xong {len(rows)}, con {len(selected)}")

    pacer = Pacer(RPM); t0 = time.time()
    print(f"\n[3/3] {len(selected)} flow, toi da {ORACLE_BUDGET} lan goi detector moi flow")
    print(f"      Nhip {RPM:.0f} lan goi LLM/phut. Uoc tinh "
          f"{len(selected)*ORACLE_BUDGET/RPM/60:.1f} gio neu moi flow dung het tran.\n")
    log(f"=== bat dau: {len(selected)} flow can lam, model={model}, rpm={RPM:.0f} ===")
    for i, idx in enumerate(selected, 1):
        orig = X_te.iloc[idx].copy()
        original = {f: float(orig[f]) for f in top10}
        log(f"flow {i}/{len(selected)} (id={idx}) bat dau ...")
        t_flow = time.time()
        try:
            p0, pf, dp, ev, calls, lc = attack(model, rf, orig, original, cols, pacer)
        except ApiDead as e:
            log("")
            log("=" * 62)
            log(f"DUNG LAI: {e}")
            log("API dang tu choi moi request. Khong ghi them dong nao de tranh")
            log("sinh ra du lieu rac. Chay 'python kiem_tra_groq.py' de xem han")
            log("ngach con lai va bao gio hoi, roi bam lai file .bat nay.")
            log(f"Da hoan tat {len(rows)} flow, tat ca deu con nguyen trong file.")
            log("=" * 62)
            break
        log(f"flow {i}/{len(selected)} xong sau {time.time()-t_flow:.0f}s: "
            f"p {p0:.3f} -> {pf:.3f}, {calls} lan goi detector, {lc} lan goi LLM")
        rows.append({"id": int(idx), "p_orig": round(p0, 4), "stratum": stratum(p0),
                     "llm": {"evaded": ev, "prob": round(pf, 4), "delta": round(dp, 4),
                             "calls": calls, "llm_calls": lc}})
        k = sum(1 for r in rows if r["llm"]["evaded"])
        lo, hi = clopper_pearson_ci(k, len(rows))
        out_path.write_text(json.dumps(
            {"config": {"model": model, "oracle_budget": ORACLE_BUDGET, "rpm": RPM,
                        "note": "budget vs ORIGINAL vector; no patience rule"},
             "llm": {"evaded": k, "n": len(rows), "ESR": round(100*k/len(rows), 1),
                     "CI": [round(100*lo, 1), round(100*hi, 1)],
                     "mean_dP": round(float(np.mean([r["llm"]["delta"] for r in rows])), 4),
                     "mean_calls": round(float(np.mean([r["llm"]["calls"] for r in rows])), 1)},
             "raw": rows}, indent=2), encoding="utf-8")
        el = time.time() - t0
        print(f"  [{len(rows):3d}/{len(rows)+len(selected)-i}] id={idx} p0={p0:.3f} "
              f"-> {pf:.3f} {'EVADED' if ev else '      '} calls={calls} "
              f"| ESR={100*k/len(rows):.1f}% | {el/60:.0f}m")

    print(f"\nDa ghi -> {out_path}")
    print("SO SANH: ban cu bao 33,6% voi ngan sach bi loi. Con so nay moi la that.")


if __name__ == "__main__":
    main()
