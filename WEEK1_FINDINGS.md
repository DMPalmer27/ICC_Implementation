# Week 1 — what changed and why

Baseline: `full_test_pre_entropy_fix.txt` (pre-fix) vs `full_test_post_fix.txt` (post-fix).
Both are full `test_icc.py` runs. **0 failures in both**, so nothing regressed; the numbers
that moved are corrections, and they are the finding.

## 1. Runtime: 4859.8 s → 8.8 s (552×)

| | pre | post |
|---|---:|---:|
| Full suite | 4859.8 s | **8.8 s** |
| `--fast` | n/a (flag did nothing) | **2.5 s** |
| Section 1a, per trial | ~1380 ms | **~18 ms** |

Two causes, both removed:

- `Client._evaluate_monomials` was a triple Python loop over scalar `galois` objects,
  `λ²·m` interpreted operations. At `m=48, d=2` it cost 64.9 s while the linear solve it
  feeds cost 0.15 s. Replaced by one vectorised pass per variable (power table + gather):
  0.18 s, elementwise identical.
- `compute_max_subset_p_entropy` enumerated all `C(n,r)` subsets. At the suite's sizes that
  was the dominant cost; at the case-study size `n=60, r=5` it is 5.4M subsets and would not
  have run at all.

`M` is now cached on the `Client` and `get_information_set` is memoised. Since `M` depends
only on the information set and the exponents — not on the data, the polynomial, or the key
— one build serves every gradient component and every GD step in Weeks 5–6.

## 2. Entropy: the pre-fix table was internally impossible

Section 2, `H_p(X)` and `max_R H_p(X_R)` columns:

| Params (q,n,r,p,ε) | pre `H_p(X)` | pre `max_R` | post `H_p(X)` | post `max_R` | pre `m` | post `m` |
|---|---:|---:|---:|---:|---:|---:|
| (31,8,3,2,1e-04) | 0.241 | **0.320** | 1.926 | 0.722 | 13 | **12** |
| (31,10,5,2,1e-06) | 0.253 | **0.371** | 2.526 | 1.263 | 17 | **15** |
| (31,12,4,2,1e-06) | 0.345 | **0.404** | 4.143 | 1.381 | 19 | **16** |
| (31,8,3,3,1e-05) | 0.124 | **0.320** | 0.995 | 0.373 | 15 | **14** |
| (37,10,4,2,1e-06) | 0.215 | **0.272** | 2.151 | 0.860 | 16 | **15** |

**In every pre-fix row `max_R H_p(X_R) > H_p(X)`** — impossible for the true quantities, an
`r`-subset cannot carry more entropy than the whole vector. That was the tell that
`compute_p_entropy` returned per-symbol `h` where Theorem 1 wants the length-`n` vector
quantity. Because the `−H_p(X)` and `+max_R H_p(X_R)` terms then nearly cancelled, the bound
collapsed to roughly `m ≈ n + p + log_q(1/ε)`, forcing `m > n` everywhere — the opposite of
the scheme's `m ≪ n` claim.

Post-fix, `max_R ≤ H_p(X) ≤ n` holds throughout, and `m` drops in every configuration.

`compute_required_m` now raises on both impossible conditions, so this class of mistake
cannot silently return a too-small `m` again.

## 3. Section 6 is byte-identical, and that is the point

Section 6 already used the correct convention via its i.i.d. shortcut
(`H_p(X) = n·h`, `max_R H_p(X_R) = r·h`), which is why it was the only section showing
`m ≪ n`. Its output is unchanged. **`utils.py` now agrees with it**, so the two entropy
paths that `CLAUDE.md` flagged as silently disagreeing have been reconciled onto the
Section 6 convention.

## 4. Modelling decision made explicit

`max_R H_p(X_R) = r·h` in closed form assumes an **i.i.d. source**: every `r`-subset then
has the same distribution, so the maximum is attained everywhere. This both fixes the
scaling and removes the `C(n,r)` blowup.

The old brute force is retained as `_compute_max_subset_p_entropy_empirical` and documented
as a **different estimator, not an oracle**: it measures the empirical entropy of one
`r`-element realization, which for small `r` is badly biased (5 values cannot resolve a
31-ary distribution) and is exactly why it could exceed `H_p(X)`.

**Open with Raviv** (blocks Weeks 3–4): the closed form is exact only when coordinates are
independent. Note that Rényi entropy of order `p ≥ 2` is **not subadditive** (verified
numerically: a binary pair with `H_2(X,Y) = 0.6200 > H_2(X) + H_2(Y) = 0.5837`), unlike
Shannon. So for correlated features `Σ hᵢ` is not an upper *or* lower bound on `H_p(X)` —
the error is unsigned. Theorem 1 needs a **lower** bound on `H_p(X)` (it enters with a minus
sign) and an **upper** bound on `max_R H_p(X_R)`; getting either backwards under-estimates
`m`, which breaks the privacy guarantee. The i.i.d. model is exact for the synthetic control
but needs a ruling for `load_diabetes`. Fallback if it is rejected: per-feature `h_j` estimated from the `n_s`
samples in column `j`, giving `H_p(X) = n_s·Σ_j h_j` and `max_R H_p(X_R) = r·max_j h_j`.

## 5. Other fixes

- `d < m(q−1)` asserted in `compute_required_m`; the scheme requires it and nothing checked it.
- `max(r, …)` floor re-justified in the docstring by **Remark 1** (a key of size `≥ r` is
  necessary) rather than the Singleton bound, which came from the MDS/uniform predecessor.
- `--fast` now actually reduces work (trial counts 10/5/50 → 2/2/5, skips the 500k dataset)
  instead of only printing a banner.
- Unused `generate_vandermonde_G` import dropped from `main.py`.
- New `test_regression.py`: 19 checks in 8.4 s.

## Carried into Week 2

The `λ³` decode model should not appear in the paper. Measured: the interpolation matrix is
`~0.5%` nonzero at `m=48`, because `M[t,a] ≠ 0 ⟺ supp(a) ⊆ supp(t)` gives exactly `C(2d,d)`
nonzeros per row independent of `m`. Real solve at `λ=1225` is 0.16 s against 17.6 s for a
dense random matrix of the same size. `feasibility.py` must measure, not model.
