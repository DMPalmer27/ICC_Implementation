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

| File | Role |
|---|---|
| `config.py` | `SystemContext` dataclass — all scheme parameters in one place |
| `client.py` | `Client.encode_data` (storage), `Client.decode_result` (interpolate `g`, evaluate at `k`) |
| `server.py` | `Server` = the admin: shards `x̃` to workers, aggregates results |
| `worker.py` | `Worker` — deliberately trivial, just `f(its share)` |
| `utils.py` | `G` generation, entropy, Theorem 1 `m` bound, leakage bound, information (super)set enumeration |
| `main.py` | One end-to-end demo run with printed diagnostics |
| `test_icc.py` | 6-section validation suite mapped to paper claims |
| `full_test.txt` | Saved output of the last full `test_icc.py` run |
| `Written_Resources/` | The paper + the Spring 2026 report (PDFs, plus `extracted/` plain text) |

## Notation: paper ↔ code

`q` field size · `n` data length · `m` key length · `d` max total degree of `f` ·
`r` privacy/security parameter · `p` entropy order (`p ≥ 2`) · `ε` smoothing budget ·
`a` confidence parameter (guarantee holds w.p. `≥ 1 − 1/a`) · `ε_c` mutual-information
leakage bound · `λ = |I_{d,m}| = C(m+d, d)` = number of workers = download cost.

Theorem 1: `m ≥ n + p + log_q(1/ε) − H_p(X) + max_R H_p(X_R)`, giving
`ε_c = p/(p−1) · log_q(1 + a·2^((2p−1)/p)·(1 + q^(−max_R H_p(X_R)))·ε^(1/p))`.

## Running things

Use the project venv (Python 3.11, `galois` + `numpy`):

```bash
.venv/bin/python main.py
.venv/bin/python test_icc.py
```

**The full suite takes ~81 minutes.** Do not run it casually. `--fast` is parsed but
currently skips nothing (`FAST_MODE` at `test_icc.py:58` is only used to print a
banner) — fixing that is a good small task. To spot-check, import from `test_icc.py`
and call a single `test_*` function.

Cost drivers, in order:
1. `compute_max_subset_p_entropy` is `C(n, r)` — superlinear blowup, caps `n` at ~12.
2. Decoding solves a dense `λ × λ` `GF(q)` system; `λ = C(m+d, d)` grows fast in `d`
   (`d=2 → λ≈171`, `d=4 → λ≈3876`, i.e. 1.4 s vs 448 s per trial).
3. `get_information_superset` is brute force over all `C(λ+S, λ)` subsets — a
   prototype only, effectively unrunnable. Replacing it is the main open work item.

## Known open issues — flag these, don't silently "fix" them

These are thesis-math decisions. Raise them; let Daniel decide.

1. **Entropy is computed per-symbol, not over the length-`n` vector.**
   Theorem 1 wants `H_p(X)` for `X ∈ F_q^n` (`≈ n·h` for i.i.d. data, `= n` when
   uniform). `compute_p_entropy` instead estimates the empirical distribution over the
   `q` alphabet symbols and returns `h ≈ 0.38`, not `n·h ≈ 3.8`. Same for
   `compute_max_subset_p_entropy` (returns `≈ h`, not `r·h`). Consequence: the
   `−H_p(X) + max_R H_p(X_R)` terms nearly cancel, so `m ≈ n + p + log_q(1/ε)` and
   `m > n` in every small test — the opposite of the scheme's `m ≪ n` selling point.
   Tell: the code can produce `max_R H_p(X_R) > H_p(X)`, which is impossible for the
   true quantities. `compute_fast_iid_entropy` in `test_icc.py` (used only in Section 6)
   *does* scale correctly by `n` and `r`, which is why only Section 6 shows `m ≪ n`.
2. **Two entropy paths disagree.** The `test_icc.py` i.i.d. shortcut computes entropy
   from the *true* sampling distribution; `utils.py` computes it from the *empirical*
   distribution of one realization. Both are defensible, but they are not the same
   estimator and should not be silently mixed.
3. **`ε_c` exponent.** Code uses `q^(−max_R H_p(X_R))`, matching the Theorem 1
   statement. The derivation at eq. (9)–(10) of the paper carries `q^(−max_R H_p(X_R)/p)`.
   Worth resolving against the authors.
4. **`max(r, ...)` floor in `compute_required_m`** is justified in the report by the
   Singleton bound, which came from the MDS/uniform predecessor scheme. Under ICC the
   code is random, so the real justification is Remark 1 (key of size `≥ r` is necessary).
5. **`d < m(q−1)` is never asserted** anywhere, though the scheme requires it.
6. `main.py:12` still imports `generate_vandermonde_G`, which is unused.

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
