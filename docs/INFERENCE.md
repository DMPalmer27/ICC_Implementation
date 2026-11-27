# Inference through the scheme

Design notes for the Fall 2026 paper: a client stores a dataset with the ICC scheme once,
then a model trained *without* the scheme scores the stored rows through it. This
document sets out the decisions, the math the code relies on, and how the study is run.
Results are in `results/inference_{S,F,C,R}.csv`, and the figures are in
`results/figures/inference/`. The paper sources are in `paper/`.

## 1. Setting

- **Stored data.** `N` rows, each with `P` features, are quantised to fixed point and
  uploaded once as `x̃ = x + kG`, with `n = N·P` values. The client keeps the key `k` and
  nothing else.
- **Model.** A linear model `(W, b)` with `K` outputs (`K = 1` for linear and binary
  logistic, `K = 3` for softmax). It is fitted in floating point on *other* rows, then
  quantised. The model is **public** because its weights are the coefficients of the
  query (decided 2026-11-25).
- **Inference.** For every stored row `i` and output `k`, the client asks for the score
  `z_ik = w_k·x_i + b_k`. This is a degree-1 polynomial in the stored data. The server
  applies it to each worker's share, and the client decodes it. The link function
  (identity, sigmoid or softmax) runs on the client after decoding, so it never has to be
  a polynomial.
- **Requests are independent (non-adaptive).** Which queries are sent does not depend on
  any earlier result.

Live gradient descent is *not* covered by this setting: each query there depends on the
previous decoded gradient, so the weights published as query coefficients form a channel
that depends on the data (`docs/GD_LINEAR_REGRESSION.md` §5). That work is paused.

## 2. Why the privacy guarantee carries over

Theorem 1 bounds `I(X_R; x̃) ≤ ε_c` for every `r`-subset `R`. It says nothing about
queries. Under the three rules below, inference adds nothing to that bound.

**Argument (Lemma A, no new math).** Everything the provider sees is `x̃`, `G` and the
public query polynomials `f_1, …, f_Q`. Each worker's output `f_j(x̃ − tG)` is a
deterministic function of those. If the queries do not depend on `X`, then the provider's
whole view is a function of `(x̃, G, f_1..f_Q)`, where `(f_1..f_Q)` is independent of
`(X, k)`. Data processing then gives `I(X_R; view) = I(X_R; x̃, G) ≤ ε_c`.

The three rules make "the queries do not depend on `X`" true:

1. **The model is not fitted on the stored rows.** Training rows and stored rows are
   disjoint: a train/store split, or fresh synthetic draws. A model fitted on `X` is a
   function of `X`, and publishing it as query coefficients would leak.
2. **Preprocessing statistics come from the training rows.** Standardisation means and
   standard deviations, and the label scale, are computed on the training split, for the
   same reason.
3. **No adaptivity.** Which rows are scored and with which model is fixed in advance.

What the client learns (the scores) is not part of the provider's view. Whether the
*client* should learn them is outside the threat model.

## 3. The query, its field and its error

**Scale.** Data codes are `x̂ = round(2^{f_X}·clip(x, ±A_X))`. Weight codes are
`ŵ = round(2^{f_w}·w)`, and the intercept is coded the same way with `f_w` bits. The query
for row `i`, output `k` is

    ẑ_ik = Σ_j ŵ_kj · x̂_ij + 2^{f_X} · b̂_k            (scale 2^{f_X + f_w})

and it is built by `query.linear_score`. The intercept is a constant term, multiplied by
`2^{f_X}` so that both terms share one scale.

**Field.** No reduction modulo `q` may wrap around. The worst case is

    B = P · X̂max · Ŵmax + 2^{f_X} · B̂max

(`quantize.score_bound_worst_case`, `ScoreBudget`), and the field needs `q > 2B`. This is
the existing gate `check_query` evaluated at the ceiling. Example: `P = 10`,
`f_X = f_w = 8`, `A_X = 4`, `A_w = 2`, `A_b = 1` gives `B = 10·1024·512 + 256·256 ≈ 5.3M`,
so `q ≈ 2^{23.3}`. Each score's integer is then exact, and the tests check it against the
plaintext integer oracle on every stored row of every run.

**Error.** With `x̂/2^{f_X} = clip(x) + δ`, `|δ| ≤ 2^{−f_X−1}`, and `ŵ/2^{f_w} = w + γ`,
`|γ| ≤ 2^{−f_w−1}`:

    |z_scheme − z_float| ≤ Σ_j |w_j| (|x_j − clip(x_j)| + 2^{−f_X−1})
                         + 2^{−f_w−1} Σ_j |clip(x_j)| + P·2^{−f_X−f_w−2} + 2^{−f_w−1}

(`inference_setup.score_error_bound`). The first term is clipping, and it is zero unless a
feature falls outside `±A_X`. Every other term halves with each extra bit. Example: `P = 10`,
`f = 8/8`, `|w_j| ≈ 0.3`, `|x_j| ≈ 0.8`: `10·0.3·2^{-9} + 2^{-9}·8 + … ≈ 0.024`. A decision
can flip only where the float score is within the bound of a tie (column `flip_possible`).

The scheme is **exact** on the integers. All error is quantisation error, which the client
fixes by choosing the format before upload.

## 4. Key length

`m ≥ n + p + log_q(1/ε) − H_p(X) + max_R H_p(X_R)` (Theorem 1), with `λ = m + 1` workers at
`d = 1`. The settings are those of the case study: `r = 5`, `p = 2`, `ε = 10⁻⁶`, `a = 5`.

The entropy per value in `log_q` units is `h = H_p(X)/(n·log2 q)`. At `d = 1` the field is
sized by `P·X̂max·Ŵmax`, so `log2 q` includes the weight bits, while the entropy depends on
the data alone. For example, a Gaussian at `f_X = 8, A_X = 4σ` has `H_2 ≈ 9.8` bits per code
against `log2 q ≈ 22`, so `h ≈ 0.45` and `m ≈ 0.55·n`. Across the synthetic families,
`h ≈ 0.25–0.55`. The spiked source (most mass on one value) has `h ≈ 0.03`, so `m ≈ n`.

**Synthetic entropy is exact.** For i.i.d. sources, `H_p(X) = n·h_code` from the exact
code pmf, and `max_R = min(r, n)·h_code` (`inference_source.FeatureSource.entropies`). For
the equicorrelated Gaussian (`ρ = 0.5, 0.9`) the bounds route of `quantize.entropy_bounds`
is used.

## 5. The study (`tools/inference_study.py`)

Every scored row goes through the real scheme: upload, one query per (row, output), one
decode. Every decoded integer is compared with the plaintext oracle (`exact`, `all_exact`).

| exp | what varies | fixed | runs |
|---|---|---|---|
| S | 9 families + Gaussian ρ ∈ {0.5, 0.9}; task ∈ {linear, logistic, softmax}; P ∈ {2, 5, 10, 20}; 3 seeds | N = 100, f = 8/8, A_X = 4 | 396 |
| F | f_X, f_w ∈ {2, 4, …, 12}; Gaussian and lognormal; linear and logistic; 2 seeds | P = 10, N = 50 | 288 |
| C | N ∈ {10, 25, 50, 100, 200, 300, 442} (442 = diabetes size), plus the d = 2 training plan for the same N | Gaussian, linear, P = 10, f = 8/8 | 7 |
| R | diabetes, wine; f_X, f_w ∈ {4, 6, 8}; A_X ∈ {4σ, 8σ} | single upload of the stored half | 36 |

Synthetic labels: linear `y = w*·x + b* + noise` at SNR 4; logistic and softmax use logits
`2.5·(direction·x)` with unit-variance directions. Training uses 2,000 fresh rows, and the
models are OLS or `LogisticRegression(C=1)`. Seeds: run seed → `SeedSequence` → independent
train, store, labels and scheme streams.

Timing hygiene: one run at a time, on AC power under `caffeinate`. Each row records `power`
and `sleep_events`.

## 6. Real data and the provisional entropy bound

`tools/real_data.py`: shuffle with a public seed (wine is stored sorted by class), split
50/50 (diabetes 221/221, wine 89/89), and standardise with training statistics.

**Entropy is PROVISIONAL (pending Raviv; CLAUDE.md known issues 2 and 3).** Real features
are correlated, and Rényi entropy of order `p ≥ 2` is not subadditive, so the product formula
is wrong in an unknown direction. The `entropy_bounds` route is used instead:

- `H_p(X) ≥ N_store · (Σ_continuous min(f_X, b_j) − log2 M)`, the min-entropy lower bound.
  `M` is the sup of the joint density, from a **Gaussian fit on the training half**.
- `max_R H_p(X_R) ≤ r·log2(levels)`. This needs no model.
- **Discrete features** (≤ 10 distinct values, e.g. diabetes `sex`) are credited 0 bits.
  The density is fitted within each discrete group, and the largest is kept.
- **Recorded precision:** a feature recorded on a grid of step `s_j` carries at most
  `b_j = log2(σ_j/s_j)` bits, however large `f_X` is. Diabetes and wine are recorded only
  about 5–6 bits finer than σ, so raising `f_X` past that *lowers* `h`.

The Gaussian `M` is a model, not a guarantee. That is the open point Raviv has to rule on.

## 7. Cost

- `d = 1`: `λ = m + 1`, linear in `n`. At diabetes size (`n = 4,420`), `λ ≈ 2,200`.
- `d = 2` (the training gradient, cost only, nothing uploaded): `λ = C(m+2, 2)`. At the same
  `N = 442, P = 10` the plan gives `λ ≈ 6.6M`, about 3,000× more.
- galois threshold: the share product `T @ G` falls back to Python integers when
  `max(shape)·q² > 2^63`. Measured at `n = 4,420`, `λ = 2,211`: store took 28.8 s at
  `log2 q = 24` against 872.9 s at `log2 q = 27`. The default formats keep `log2 q ≲ 24`.
  F is run at `N = 50` so that its larger formats stay cheap.

## 7a. Results of the full run (2026-11-26/27)

727 runs (S 396, F 288, C 7, R 36), 90,311 scores, all on AC power with 0 sleep events,
about 76 minutes of compute in total.

- **Exact:** all 727 runs decoded every score equal to the integer oracle.
- **Error bound:** 0 violations. Without clipping, the max error was 0.041, at most 0.73 of the bound
  (mean 0.16). With clipping the bound is nearly tight. The worst case was 261, from two
  Student-t₃ values far outside ±4σ, in a run whose unclipped rows had error 0.011.
- **Metrics:**
  - Without clipping, the |ΔR²| was ≤ 9.9e-4, and accuracy differed by ≤ 0.01 (one decision in 100).
  - With clipping, the largest gaps were ΔR² 0.41 (Student-t₃) and accuracy 0.02–0.03.
  - All 81 flips were on rows the bound flagged as near a tie.
- **Format (F):**
  - Error halves per bit until the other operand's bits limit it.
  - `m/n` rises with `f_w` and falls with `f_X`.
  - Gaussian 6/6 gives ΔR² < 1e-3 at log2 q ≈ 18; 8/8 gives err ≈ 0.01 at log2 q ≈ 22.
  - Lognormal is dominated by clipping at every format.
- **Key length:** across continuous sources `h` = 0.31–0.49 and `m/n` = 0.54–0.69. Discrete3
  (`h` 0.07) and spiked (`h` 0.03) need `m/n` ≈ 0.94–0.98.
- **Cost (C, N = 442, P = 10):** λ = 2,483 (d = 1) vs 6,564,876 for the d = 2 plan (×2,644).
  A run takes 152 s: store 37 s, build M 56 s, eval 54 s, decode 3 s.
- **Real data (f = 8/8, ±4σ):**
  - Diabetes: R² 0.5173 vs 0.5156, m = 1,656.
  - Wine: acc 0.955 vs 0.944, m = 825.
  - Both "gains" come from clipping 2 values each. At ±8σ there is no clipping: max error 0.007 and
    0.020, R² 0.5158, and wine decisions identical.

## 8. Deferred and flagged

- **Blocking** (several independent uploads, each sized alone) caps replication. It needs
  an additivity lemma, so it is **deferred**; there are no new lemmas this semester.
  Everything here is a single upload.
- **Breast cancer** is deferred. Its features are strongly correlated (max |corr| 0.998,
  roughly 51 bits/row Gaussian penalty). `load('breast_cancer')` works, to run overnight
  later.
- **Known issue 1** (`ε_c` exponent): the CSVs carry both `eps_c` (theorem statement) and
  `eps_c_alt` (the derivation's `/p`).
- **Core-code change for review:** `utils.get_information_set` was rewritten from a
  depth-first recursion, which hit Python's recursion limit at `m ≈ 990`, to a direct
  enumeration. It was verified identical, order included, on 240 cases
  (`tests/test_inference.py` §4).

## 9. Reproduce

```bash
.venv/bin/python tests/test_inference.py                          # 21 checks, ~5 s
.venv/bin/python tools/inference_study.py --exp all --dry-run     # grid sizes
caffeinate -dimsu .venv/bin/python tools/inference_study.py --exp all --resume   # ~2-2.5 h
.venv/bin/python tools/plot_inference.py                          # -> results/figures/inference/
```
