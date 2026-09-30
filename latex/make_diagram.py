#!/usr/bin/env python3
"""Sinh Figure 1 cua bai 02 bang TikZ, so lieu doc tu file ket qua.

    python3 make_diagram.py <results_dir> <out_dir>

Khong co con so nao go tay: ty le evasion va duong cong epsilon deu lay tu
step10_corrected.json va step12_budget_sweep.json.
"""
import json
import subprocess
import sys
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
STYLE = Path(__file__).resolve().parents[2] / "tikz" / "style.tex"
sys.path.insert(0, str(HERE.parents[1] / "tikz"))
from scale import widen                                        # noqa: E402


S10 = json.loads((RES / "revision_v8" / "step10_corrected.json").read_text())
S12 = json.loads((RES / "revision_v8" / "step12_budget_sweep.json").read_text())

K = S10["config"]["oracle_budget"]
N = S10["arms"]["greedy"]["n"]
TAU = S12["config"]["tau"]
raw = S12["raw"]
EPS = ["0.05", "0.10", "0.15", "0.20", "0.30", "0.50", "1.00"]
esr = {e: 100.0 * sum(1 for r in raw if r["eps"][e]["best"] < TAU) / len(raw)
       for e in EPS}
first = next(e for e in EPS if esr[e] > 0)
n_arms = len([a for a in S10["arms"] if not a.endswith("oldsign")])
grow = 1.1 ** K          # he so bung ra sau K vong
shrink = 0.9 ** K

# toa do duong cong: truc x deu tay, truc y theo ESR
xs = [0.50 + k * 1.25 for k in range(7)]
ymax = 55.0
pts = " ".join(f"({x:.2f},{0.28 + esr[e] / ymax * 1.29:.3f})"
               for x, e in zip(xs, EPS))
ticks = "".join(
    f"\\draw[draw=mutedc,line width=.3pt] ({x:.2f},0.28) -- ({x:.2f},0.20);\n"
    f"\\node[lbl,anchor=north,inner sep=2pt] at ({x:.2f},0.20) "
    f"{{{float(e) * 100:g}\\%}};\n" for x, e in zip(xs, EPS))

TEX = r"""\documentclass[border=3pt]{standalone}
\input{style}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]

%% ================= A: ngan sach bi ap sai =================
\node[ttl] at (0,6.55) {A\ \ The budget was enforced against the wrong vector};

%% -- hang 1: y dinh
\node[lbl,anchor=west,text=okc,text width=1.3cm] at (0,6.12) {\textsc{intended}};
\node[lbl,anchor=west,text width=2.9cm] at (1.40,6.12)
  {one fixed box around the original flow};
\draw[draw=okc,line width=.7pt,fill=oksoft,rounded corners=1.5pt]
  (4.55,5.88) rectangle (7.95,6.36);
\foreach \i in {0,...,5}{\fill[okc] ({4.85+\i*0.52},6.12) circle (0.038);}
\node[lbl,anchor=north,text=okc,inner sep=2pt] at (6.25,5.86)
  {$|x_t-x_0|\le\varepsilon|x_0|$ at every step};

%% -- hang 2: thuc te
\node[lbl,anchor=west,text=critc,text width=1.3cm] at (0,5.05) {\textsc{implem-\\ented}};
\node[lbl,anchor=west,text width=2.9cm] at (1.40,5.05)
  {the box is recomputed from wherever the search stands};
\draw[dash] (4.55,4.81) rectangle (5.55,5.29);
\foreach \i in {0,1}{\fill[critc] ({4.78+\i*0.44},5.05) circle (0.038);}
\draw[arc] (5.62,5.05) -- (6.02,5.05);
\foreach \i in {0,...,4}{
  \fill[critc,opacity={0.30+\i*0.17}] ({6.18+\i*0.42},5.05) circle ({0.038+\i*0.016});}
\node[lbl,anchor=north,text=critc,inner sep=2pt] at (6.25,4.79)
  {$|x_{t+1}-x_t|\le\varepsilon|x_t|$ compounds};

\node[critbox,anchor=north west,text width=2.75cm] at (0,4.32)
  {\faExclamationTriangle\ \ after $K{=}%%K%%$ iterations a\\ feature reaches
   $%%GROW%%\times x_0$,\\ not $1.1\times x_0$};
\node[lbl,anchor=north west,text width=4.75cm,align=left] at (3.10,4.32)
  {The reachable set grows with the iteration count, so the arm that moved most
   collected the most budget. That is the arm the first version of this work
   reported as the winner.};

\draw[draw=linec,line width=.4pt] (0,3.46) -- (8.0,3.46);

%% ================= B: sau khi sua =================
\node[ttl] at (0,3.16) {B\ \ With the box fixed to the original flow, nothing evades};

\node[okbox,anchor=west,text width=1.55cm] at (0,2.64)
  {\faCrosshairs\ corner\\1 call\\[1.5pt]\textbf{0/%%N%%}};
\node[okbox,anchor=west,text width=1.55cm] at (2.05,2.64)
  {\faCogs\ greedy\\%%K%% calls\\[1.5pt]\textbf{0/%%N%%}};
\node[okbox,anchor=west,text width=1.55cm] at (4.10,2.64)
  {\faProjectDiagram\ NES\\three sizes\\[1.5pt]\textbf{0/%%N%%}};
\node[okbox,anchor=west,text width=1.55cm] at (6.15,2.64)
  {\faRobot\ LLM\\13 flows\\[1.5pt]\textbf{0/13}};
\node[lbl,anchor=west] at (0,2.04)
  {%%NARMS%% arms on the same flows, exact Clopper--Pearson upper bound $3.4\%$};

%% -- duong cong epsilon
\node[lbl,anchor=west,text=inkc] at (0,1.72)
  {\textsc{what does move the rate is the perturbation allowed}};
\draw[draw=mutedc,line width=.4pt] (0.50,0.28) -- (8.0,0.28);
\draw[draw=mutedc,line width=.4pt] (0.50,0.28) -- (0.50,1.70);
\node[lbl,rotate=90,anchor=south] at (-0.02,0.99) {evasion (\%)};
%%TICKS%%
\foreach \v/\y in {0/0.28, 25/0.925, 50/1.57}{
  \draw[draw=mutedc,line width=.3pt] (0.50,\y) -- (0.40,\y);
  \node[lbl,anchor=east,inner sep=2pt] at (0.40,\y) {\v};}
\draw[dash] (%%XFIRST%%,0.28) -- (%%XFIRST%%,1.36);
\node[lbl,anchor=west,text=critc,inner sep=2pt,text width=1.7cm] at (%%XFIRST%%+0.08,1.18)
  {nothing evades below %%FIRST%%\%};
\draw[draw=accc,line width=.9pt] plot[smooth,tension=.4] coordinates {%%PTS%%};
\foreach \p in {%%PTS%%}{\fill[accc] \p circle (0.040);}

\end{tikzpicture}
\end{document}
"""

TEX = (TEX.replace("%%K%%", str(K))
          .replace("%%GROW%%", f"{grow:.2f}")
          .replace("%%N%%", str(N))
          .replace("%%NARMS%%", str(n_arms))
          .replace("%%TICKS%%", ticks)
          .replace("%%PTS%%", pts)
          .replace("%%XFIRST%%", f"{xs[EPS.index(first)]:.2f}")
          .replace("%%FIRST%%", f"{float(first) * 100:g}"))

work = OUT / "_tikz"
work.mkdir(exist_ok=True)
(work / "style.tex").write_text(STYLE.read_text(), encoding="utf-8")
(work / "fig_threat_model.tex").write_text(widen(TEX), encoding="utf-8")
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig_threat_model"],
                   cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
src = work / "fig_threat_model.pdf"
if not src.exists():
    log = (work / "fig_threat_model.log").read_text(errors="ignore")
    print("\n".join(l for l in log.splitlines() if l.startswith("!"))[:800])
    raise SystemExit("TikZ khong bien dich duoc")
(OUT / "fig_threat_model.pdf").write_bytes(src.read_bytes())
print(f"wrote {OUT / 'fig_threat_model.pdf'}  (K={K}, n={N}, nguong={first})")
