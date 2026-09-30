# NIDS evasion under a correctly bounded budget

Result files, experiment scripts and LaTeX sources for the paper

> S. X. Ha, P. T. Tran-Truong, X.-B. Le, L. H. Huong, T. Q. Nguyen, T. G. Huy, N. N. T. Kha, T. N. Minh and N. N. Phien,
> "The Budget Was Never Enforced: A Reported 33% Evasion Rate Against a Random-Forest NIDS Falls to Zero Under a Correctly Bounded Perturbation",
> submitted to IEEE Transactions on Network and Service Management, 2026.

## What is in here

```
results/     the measured result files. Every number, table and figure in the
             paper is computed from these and from nothing else.
scripts/     the experiment code that produced those files.
latex/       the generators that read results/ and emit the table bodies and
             figures, plus the paper source they are substituted into.
```

## What you can reproduce, and what you cannot

**From this repository alone** you can regenerate every table, figure and
inline number in the paper:

```bash
pip install -r requirements.txt
cd latex
python3 make_tables.py  ../results  tables
python3 make_figures.py ../results  figures
python3 make_diagram.py ../results  figures
python3 build.py        ../results          # writes main.tex
python3 build_ieee.py                      # writes ieee/main_ieee.tex
```

`build.py` substitutes the generated values into `paper_template.tex`. No number
in the paper is typed by hand, so a mismatch between the paper and a fresh run
of these scripts is a bug and we would like to hear about it.

`build_ieee.py` then rewrites that manuscript into the IEEE two-column form that
was actually submitted: it drops the CRediT section, which IEEE has no field for,
lifts the funding statement into a page-one footnote, rebuilds the author block
in IEEE style, and widens only the tables that overflow a column.
`latex/ieee/main_ieee.tex` is checked in even though it is generated, because it
is the exact manuscript we submitted. Regenerating it must produce the same
bytes; that diff is the artifact's own self-check, and we ran it on all nine
papers before publishing.

**You cannot regenerate `results/` from this repository alone.** Doing that needs
the flow-level intrusion-detection dataset the detector was trained on, a scikit-learn environment to refit the random forest, and a Groq API key for the LLM attacker arm. The dataset is not redistributed here; see `THIRD_PARTY.md`.
The scripts in `scripts/` are the code we ran; they are published so the
procedure can be inspected and re-executed by anyone who assembles that
environment.

## Layout of `results/`

| File | Used for |
|---|---|
| `revision_v8/step10_corrected.json` | the corrected-budget re-run. The headline result of the paper is computed from this file |
| `revision_v8/step12_budget_sweep.json` | the budget sweep: evasion rate as a function of the allowed perturbation, from which the epsilon curve is drawn |
| `revision_v8/step11_llm_corrected.json` | the LLM attacker arm under the corrected budget. Only the first flows received the full detector budget; the paper reports that and does not average over the truncated ones |
| `revision_v8/step8_oracle_capped.json` | the oracle attacker under a capped query budget |
| `revision_v8/step9_friday_calibration.json` | calibration measured on the held-out later-day traffic |
| `revision_v8/step1_temporal_verify.json` | the check that the train and test splits are separated in time |
| `revision_v8/step2_isotonic_calibration.json` | isotonic recalibration of the detector scores |
| `revision_v8/step3_baselines_results.json` | the NES and other attack baselines |
| `revision_v8/step4_llm_results.json` | the LLM arm of the v8 round, before the budget correction |
| `revision_v8/step5_sampling_comparison.json` | uniform versus stratified flow sampling |
| `revision_v8/step6_merged_results.json` | merged scores and the decision threshold |
| `revision_v8/v8_sample_indices.json` | the indices of the sampled flows, so the selection can be inspected rather than trusted |
| `evasion_v6/evasion_v6_results.json` | the earlier round whose reported rate this paper retracts |
| `evasion_v6/calibration_v6_analysis.json` | calibration analysis of that earlier round |
| `baseline/metrics.csv` | the random-forest detector's clean-traffic performance |

## Paths in `scripts/`

`scripts/_step4_loop_runner.py` and `scripts/verify_draft_numbers.py` still contain the absolute path of the machine the experiments ran on. We have
not rewritten them. These files are published as a record of what was executed,
and editing a path inside a script after the fact would make the record less
reliable rather than more. A reader who wants to re-run them should change that
one line to point at their own checkout.

## Licence

Code in `scripts/` and `latex/` is MIT. The result files in `results/` are
CC BY 4.0. See `LICENSE`.
