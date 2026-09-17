# ICC Implementation

Implementation of **Individual Confidential Computing (ICC) of Polynomials over
Non-Uniform Information** — Tarnopolsky, Deng, Ramkumar, Raviv, Cohen (arXiv:2501.15645).

Master's thesis work, Daniel Palmer (d.m.palmer@wustl.edu), advised by Netanel Raviv.

## Layout

```
icc/        implementation (config, client, server, worker, utils, quantize)
tests/      test_icc.py (validation suite + report tables), test_regression.py (fast checks)
tools/      feasibility.py (parameter feasibility), quantization_study.py (quantisation measurements)
results/    saved suite output and raw CSV rows — inputs to the write-up
docs/       PLAN.md (semester plan), WEEK1_FINDINGS.md, QUANTIZATION.md (Week 3 design doc)
Written_Resources/   the papers and the Spring 2026 report, plus extracted/ plain text
main.py     one end-to-end demo run
```

`icc/` is a plain directory rather than an installed package; entry points add it to
`sys.path` so modules import each other by plain name.

## Running

Requires the project venv (Python 3.11). `icc/` itself needs only `galois` and `numpy`;
`tools/` and `tests/` additionally use `scipy`, `scikit-learn` and `pypdf`. See
`requirements.txt`. Run from the repo root:

```bash
.venv/bin/python main.py                       # end-to-end demo
.venv/bin/python tests/test_regression.py      # ~8 s  — fast safety net
.venv/bin/python tests/test_icc.py             # ~9 s  — full suite, prints report tables
.venv/bin/python tests/test_icc.py --fast      # ~2.5 s
.venv/bin/python tools/feasibility.py --csv results/feasibility_sweep.csv
.venv/bin/python tools/quantization_study.py --csv results/quantization_budget.csv
```

## Where the work stands

Weeks 1–2 (performance and Theorem 1 correctness) and Week 3 (quantisation of real-valued
data into `F_q`) are done; see `docs/WEEK1_FINDINGS.md` and `docs/QUANTIZATION.md`. Open
questions for Raviv are collected at the end of `docs/PLAN.md`.
