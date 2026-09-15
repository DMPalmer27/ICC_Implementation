# ICC Implementation

Implementation of **Individual Confidential Computing (ICC) of Polynomials over
Non-Uniform Information** — Tarnopolsky, Deng, Ramkumar, Raviv, Cohen (arXiv:2501.15645).

Master's thesis work, Daniel Palmer (d.m.palmer@wustl.edu), advised by Netanel Raviv.

## Layout

```
icc/        implementation (config, client, server, worker, utils)
tests/      test_icc.py (validation suite + report tables), test_regression.py (fast checks)
tools/      feasibility.py (parameter feasibility instrument)
results/    saved suite output and raw CSV rows — inputs to the write-up
docs/       PLAN.md (semester plan), WEEK1_FINDINGS.md (optimization pass write-up)
Written_Resources/   the paper and the Spring 2026 report
main.py     one end-to-end demo run
```

`icc/` is a plain directory rather than an installed package; entry points add it to
`sys.path` so modules import each other by plain name.

## Running

Requires the project venv (Python 3.11, `galois` + `numpy`). Run from the repo root:

```bash
.venv/bin/python main.py                       # end-to-end demo
.venv/bin/python tests/test_regression.py      # ~8 s  — fast safety net
.venv/bin/python tests/test_icc.py             # ~9 s  — full suite, prints report tables
.venv/bin/python tests/test_icc.py --fast      # ~2.5 s
.venv/bin/python tools/feasibility.py --csv results/feasibility_sweep.csv
```
