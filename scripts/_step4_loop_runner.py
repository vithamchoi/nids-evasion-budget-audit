import json, subprocess, sys, time, os
from datetime import datetime, timedelta

os.chdir(r"c:\Users\Admin\Documents\Data science\code_hoan_chinh\02_robust_vit_encrypted_traffic")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS = "results/revision_v8/step4_llm_results.json"
DEADLINE = datetime.now() + timedelta(hours=2)

last_run_lines = []

def get_count():
    try:
        with open(RESULTS, "r", encoding="utf-8") as f:
            d = json.load(f)
        return len(d.get("raw", []))
    except Exception as e:
        print("COUNT_ERROR", repr(e), file=sys.stderr, flush=True)
        return 0

while datetime.now() < DEADLINE:
    p = subprocess.run(
        [sys.executable, "sandbox/step4_llm_v8.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = p.stdout or ""
    last_run_lines = out.splitlines()[-30:]
    sys.stdout.write(out)
    sys.stdout.flush()

    n = get_count()
    print("count", n, flush=True)
    if n >= 107:
        break
    if datetime.now() >= DEADLINE:
        break
    time.sleep(30)

final_count = get_count()
print("FINAL_COUNT", final_count, flush=True)

p6 = subprocess.run(
    [sys.executable, "sandbox/step6_merge_and_threshold.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)
print("===STEP6===", flush=True)
print(p6.stdout or "", end="", flush=True)

try:
    with open(RESULTS, "r", encoding="utf-8") as f:
        esr = json.load(f).get("llm")
except Exception as e:
    esr = {"error": repr(e)}
print("===ESR===", esr, flush=True)

print("===STEP4_TAIL===", flush=True)
for line in last_run_lines:
    print(line, flush=True)
