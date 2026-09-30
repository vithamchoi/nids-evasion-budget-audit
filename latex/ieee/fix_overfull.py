#!/usr/bin/env python3
"""Doc log LaTeX cua ban IEEE, tim bang nao bi tran khi de mot cot, va ghi
nhan cua no vao ieee/force_wide.txt de lan build sau day bang do ra ca trang.

    python3 fix_overfull.py <duong_dan_thu_muc_ieee>

Cach nay do thay vi doan: nguong do dai chuoi khong biet font, khong biet
noi dung o, con log thi noi dung so diem tran cua tung dong.
Tra ve so nhan MOI them vao; 0 nghia la khong con gi phai sua.
"""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
log, tex, out = d / "main_ieee.log", d / "main_ieee.tex", d / "force_wide.txt"
if not (log.exists() and tex.exists()):
    print("0")
    sys.exit(0)

ranges = [(int(a), int(b)) for a, b in
          re.findall(r"Overfull \\hbox \([\d.]+pt too wide\) in paragraph at lines (\d+)--(\d+)",
                     log.read_text(encoding="utf-8", errors="replace"))]

lines = tex.read_text(encoding="utf-8").splitlines()
blocks = []          # (dong_bat_dau, dong_ket_thuc, nhan)
start = None
for i, l in enumerate(lines, 1):
    if re.match(r"\s*\\begin\{table\}", l):
        start = i
    elif re.match(r"\s*\\end\{table\}", l) and start:
        seg = "\n".join(lines[start - 1:i])
        lb = re.search(r"\\label\{([^}]*)\}", seg)
        blocks.append((start, i, lb.group(1) if lb else None))
        start = None

have = set()
if out.exists():
    have = {x.strip() for x in out.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.startswith("#")}

new = set()
for a, b in ranges:
    for s, e, lb in blocks:
        if lb and s <= a <= e:
            if lb not in have:
                new.add(lb)

if new:
    out.write_text("# Bang do duoc la tran khi de mot cot -- ghi tu dong.\n"
                   + "\n".join(sorted(have | new)) + "\n", encoding="utf-8")
print(len(new))
