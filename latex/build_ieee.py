#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the IEEE two-column version of this paper.

Run it after build.py, from the latex/ directory:

    python3 build.py ../results        # writes main.tex
    python3 build_ieee.py              # writes ieee/main_ieee.tex

to_ieee.py rewrites the journal-source manuscript into IEEEtran: it drops the
CRediT section (IEEE has no such field), lifts the funding statement into a
page-one \\thanks, rebuilds the author block in IEEE form, and widens only the
tables that overflow a column. The keyword list and the author-block variant
below are the ones used for the submitted version and are not inferred.
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYWORDS = 'Intrusion detection, adversarial machine learning, evasion attacks, black-box optimization, CICIDS2017, reproducibility'
AUTHORS  = 'huong'

def main():
    src = HERE / "main.tex"
    if not src.exists():
        sys.exit("main.tex not found. Run build.py first.")
    dst = HERE / "ieee" / "main_ieee.tex"
    dst.parent.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(HERE / "ieee" / "to_ieee.py"),
                        str(src), str(dst), KEYWORDS, AUTHORS])
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
