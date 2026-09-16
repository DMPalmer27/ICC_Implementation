# ICC Implementation — Master's Thesis (Daniel Palmer, advised by Netanel Raviv)

Implementation of **Individual Confidential Computing (ICC) of Polynomials over
Non-Uniform Information** — Tarnopolsky, Deng, Ramkumar, Raviv, Cohen (arXiv:2501.15645).

Timeline: started Jan 2026 as an independent study; the Spring 2026 report
(`Written_Resources/`) is the deliverable for that semester. Continuing through
~mid-2027 as the MS thesis.

## What the scheme does

A user holds `x ∈ F_q^n` from an **unknown, non-uniform** distribution. They want a
cloud provider (admin + workers) to compute a polynomial `f` of total degree `≤ d`
over `x`, without the provider — *as a whole*, not just colluding workers — learning
anything about any `r`-subset of `x`.

- **Storage:** user picks `k ~ Unif(F_q^m)`, random `G ∈ F_q^{m×n}`, uploads `x̃ = x + kG`,
  keeps only `k`. `G` is a *random* linear code (not MDS/Vandermonde — that was the
  uniform-data predecessor scheme).
- **Distribution:** admin builds `x̃ − tG` for each `t` in an information set
  `I_{d,m}` of `RM_q(d,m)`, one per worker.
- **Computation:** workers return `f(x̃ − tG)`. Define `g(t) ≜ f(x̃ − tG)`; then
  `g(k) = f(x + kG − kG) = f(x)`.
- **Decoding:** user interpolates `g` from its evaluations on the information set
  (solve `Mc = A` over `F_q`), then evaluates `g(k)`.

The non-uniform extension replaces *zero* leakage with *negligible* leakage via
**distribution smoothing** (Pathegama et al.): a random linear code with large enough
`m` makes `x̃` nearly uniform, so the uniform-case privacy argument carries through.

## File map

Implementation lives in `icc/`, which is a plain directory (not an installed package).
Entry points put it on `sys.path` via `tests/conftest.py`, `tools/_path.py`, or an inline
shim in `main.py`, so modules keep importing each other by plain name (`from utils import`).

| File | Role |
|---|---|
| `icc/config.py` | `SystemContext` dataclass — all scheme parameters in one place |
| `icc/client.py` | `Client.encode_data` (storage), `Client.decode_result` (interpolate `g`, evaluate at `k`); caches the interpolation matrix `M` |
| `icc/server.py` | `Server` = the admin: shards `x̃` to workers, aggregates results |
| `icc/worker.py` | `Worker` — deliberately trivial, just `f(its share)` |
| `icc/utils.py` | `G` generation, entropy, Theorem 1 `m` bound, leakage bound, `build_monomial_matrix`, information (super)set enumeration |
| `icc/quantize.py` | Fixed-point map reals ↔ `F_q`, balanced lift, hard no-wraparound bounds, field-size choice, rigorous entropy bounds for quantised data |
| `main.py` | One end-to-end demo run with printed diagnostics |
| `tests/test_icc.py` | 6-section validation suite mapped to paper claims; its printed tables are report output |
| `tests/test_regression.py` | Fast equivalence/invariant checks; keeps pre-optimization implementations as oracles |
| `tests/conftest.py` | `sys.path` shim; imported explicitly by both suites |
| `tools/feasibility.py` | Parameter feasibility instrument — reports `m`, `λ`, and **measured** build/solve cost |
| `tools/quantization_study.py` | Seeded measurements behind `docs/QUANTIZATION.md`: entropy-bound validation, field-size budget, residue entropy, end-to-end decodability |
| `results/full_test_pre_entropy_fix.txt` | Full `test_icc.py` run before the Week 1 entropy fix (baseline) |
| `results/full_test_post_fix.txt` | Full `test_icc.py` run after it |
| `results/feasibility_sweep.csv` | Raw rows from `tools/feasibility.py` |
| `results/quantization_study.txt` / `quantization_budget.csv` | Saved output of `tools/quantization_study.py` |
| `docs/PLAN.md` | Fall 2026 semester plan |
| `docs/WEEK1_FINDINGS.md` | What the Week 1 optimization pass changed and why |
| `docs/QUANTIZATION.md` | Week 3 design doc: survey of real→`F_q` embeddings, field-size derivation, the entropy bounds, recommended parameters |
| `Written_Resources/` | The paper + the Spring 2026 report (PDFs, plus `extracted/` plain text) |

## Notation: paper ↔ code

`q` field size · `n` data length · `m` key length · `d` max total degree of `f` ·
`r` privacy/security parameter · `p` entropy order (`p ≥ 2`) · `ε` smoothing budget ·
`a` confidence parameter (guarantee holds w.p. `≥ 1 − 1/a`) · `ε_c` mutual-information
leakage bound · `λ = |I_{d,m}| = C(m+d, d)` = number of workers = download cost.

Theorem 1: `m ≥ n + p + log_q(1/ε) − H_p(X) + max_R H_p(X_R)`, giving
`ε_c = p/(p−1) · log_q(1 + a·2^((2p−1)/p)·(1 + q^(−max_R H_p(X_R)))·ε^(1/p))`.

## Running things

Use the project venv (Python 3.11, `galois` + `numpy`). Run from the repo root:

```bash
.venv/bin/python main.py
.venv/bin/python tests/test_regression.py          # ~8 s, run this first
.venv/bin/python tests/test_icc.py                 # full suite, ~9 s
.venv/bin/python tests/test_icc.py --fast          # ~2.5 s
.venv/bin/python tools/feasibility.py --csv results/feasibility_sweep.csv
.venv/bin/python tools/quantization_study.py --csv results/quantization_budget.csv
```

The full suite took ~81 min before the Week 1 pass and now takes ~9 s, so it is cheap to
run. `--fast` genuinely reduces work (trial counts 10/5/50 → 2/2/5, drops the 500k
dataset). To spot-check, import from `tests/test_icc.py` and call a single `test_*`.

Cost drivers, in order:
1. Decoding solves a `λ × λ` `GF(q)` system, `λ = C(m+d, d)`. The binding constraint is
   **memory**, not time: the dense matrix is 1.7 GB at `n=250`. `M` is only ~0.5% nonzero
   (`M[t,a] ≠ 0 ⟺ supp(a) ⊆ supp(t)`, so `C(2d,d)` nonzeros per row), so a sparse
   representation is the unlock — see `docs/PLAN.md`.
2. `build_monomial_matrix` — vectorized, but still the largest term at `n ≥ 150`
   (16 s build vs 2.4 s solve). `Client` caches `M` across GD steps, so it is paid once.
3. `get_information_superset` is brute force over all `C(λ+S, λ)` subsets — a prototype
   only, effectively unrunnable. Out of scope for Fall 2026 (see `docs/PLAN.md`).

## Known open issues — flag these, don't silently "fix" them

These are thesis-math decisions. Raise them; let Daniel decide.

1. **`ε_c` exponent.** Code uses `q^(−max_R H_p(X_R))`, matching the Theorem 1
   statement. The derivation at eq. (9)–(10) of the paper carries `q^(−max_R H_p(X_R)/p)`.
   Worth resolving against the authors.
2. **The i.i.d. source assumption is now load-bearing.** `compute_max_subset_p_entropy`
   returns `r·h` in closed form, which is exact only when coordinates are independent.
   Rényi entropy of order `p ≥ 2` is **not** subadditive (unlike Shannon), so for
   correlated features `Σ hᵢ` bounds `H_p(X)` in neither direction — the error is
   unsigned. Theorem 1 needs a *lower* bound on `H_p(X)` and an *upper* bound on
   `max_R H_p(X_R)`; either one backwards under-estimates `m` and breaks privacy.
   Needs a ruling from Raviv before real (correlated) data. See `docs/WEEK1_FINDINGS.md` §4.
3. **`quantize.entropy_bounds` is a proposed third route, and it computes BOUNDS not
   estimates.** It lower-bounds `H_p(X)` by min-entropy under a sup-density assumption and
   upper-bounds `max_R H_p(X_R)` by `r·log_q(levels)` — the directions Theorem 1 needs. The
   upper bound is assumption-free; the lower bound survives correlation and is a candidate
   fix for issue 2. Do not mix it with the two estimators below. Needs Raviv's ruling —
   see `docs/QUANTIZATION.md` §8 and open questions 7–9 in `docs/PLAN.md`.
4. **Two entropy estimators still coexist, now documented rather than mixed.**
   `compute_max_subset_p_entropy` uses the source-model closed form;
   `_compute_max_subset_p_entropy_empirical` is the old `C(n,r)` brute force, kept for
   reference and explicitly **not** an oracle for the former. Don't use it to compute `m`.

Resolved in the Week 1 pass (`week1-2-optimizations`) — do not re-report as bugs:
per-symbol vs vector entropy scaling, the two entropy paths disagreeing, the
`max(r, …)` justification (now Remark 1, not Singleton), the missing `d < m(q−1)`
assertion, and the unused `generate_vandermonde_G` import.

## Working preferences

- **Straggler resilience (`S`) is out of scope for existing code** — the current
  implementation is deliberately the no-straggler simplification. Information
  super-sets are the next major feature, not an assumption to code against.
- Match the existing style: module docstring with `File:` / `Author:` / `Description:`,
  `:param:` / `:return:` docstrings, type hints, plain NumPy + `galois`.
- Keep every equation traceable to the paper — cite theorem/definition/appendix
  numbers in comments, as the current code does.
- This is thesis work that gets written up in LaTeX. Numerical results that land in
  tables need to be reproducible: prefer seeded runs and save raw output.
