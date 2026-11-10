# Linear regression by gradient descent under ICC: implementation plan

Plan for `docs/PLAN.md` Weeks 5–6, written 2026-10-20 on branch `gd-linear-regression`.
Scope of this round: **least-squares linear regression, trained live by gradient descent through
the scheme, on synthetic data drawn from a known distribution, with the information leakage
measured directly** rather than only bounded. Logistic regression and `load_diabetes` come
later.

Everything privacy-relevant is written by hand: the training loop, the data source and its
exact entropies, and the leakage computation. No scikit-learn, no autograd, no MI-estimator
library. NumPy is used for array arithmetic, and `galois` for `F_q` arithmetic as before.

Status: **plan only, nothing implemented yet.** Decisions D1–D4 (§9) were **settled
2026-10-21, all as recommended**: live GD, with the weight channel measured; `m` from the exact
entropy; exact leakage at toy scale plus an attack at `n = 100`; the default source and
schedule.
**Phase 1 (2026-11-10): convergence across many synthetic distributions, encoded vs unencoded,
in §10**, which adds an intercept (C3) and a per-distribution `η` (C4). Leakage is Phase 2.

---

## 0. The short version

1. **Correctness is not in doubt.** Study D already shows that one decoded gradient equals
   the integer oracle exactly. The new correctness claim to verify is stronger: the whole
   training trajectory through the scheme equals plaintext integer GD **bit for bit**.
2. **Theorem 1 does not cover live GD.** The ICC guarantee (Definition 1) bounds
   `I(X_R; X̃)`, which is the stored data only. In live GD, the weight codes `ŵ_t` are
   coefficients of the query. The provider reads them, and they are a function of the data.
   So the provider's real view is `(X̃, ŵ_1, …, ŵ_T)`, and the extra channel `ŵ_{1:T}` has
   no bound in either paper. §2 states it precisely, and §5 measures it exactly at toy scale.
   This is the most important thing in the plan. See **D1**.
3. **The leakage can be measured exactly, not estimated.** Two reductions (§2.2) collapse the
   provider's whole view to `(syndrome of X, ŵ_{1:T})`. Both are deterministic functions of
   `X`. So whenever the support of `X` can be enumerated, every mutual information in the
   problem is an exact finite sum. That requires a toy size (`n ≈ 6–10` symbols, 5–9 levels);
   at the case-study size (`n = 100`) it is impossible, and §5.3 says what is measured there
   instead.
4. **The synthetic source is built so that its true Rényi entropies are computable in closed
   form** (§3). The labels are correlated with the features, which makes the repo's i.i.d.
   entropy estimator wrong even on synthetic data (open issue 2). The exact values let `m` be
   set from the truth, and also let `quantize.entropy_bounds` be **checked** against the
   truth for the first time.

---

## 1. Symbols, with the case-study numbers

These are the same as `docs/QUANTIZATION.md` §2. They are repeated here with numbers, because
the plan leans on them throughout.

| symbol | meaning in words | case-study value |
|---|---|---|
| `n_s`, `P` | number of samples, number of features | 25, 3 |
| `n` | length of the stored vector: the feature matrix flattened, then the labels | `25·(3+1) = 100` |
| `f_X` | fractional bits of the stored data: real `x` is stored as the integer `round(x·2^f_X)` | 7, so `0.7071 → 91` |
| `A_X` | clip bound on the data, in standard deviations | 4, so codes lie in `[−512, 512]` |
| `X̂max` | largest stored code, `⌊A_X·2^f_X⌋`; the server's `max_input` | 512 |
| `f_w`, `A_w` | fractional bits and clip of the weights | 4, 2, so `w = 0.6 → ŵ = 10` |
| `ŵ_t` | the integer weight codes at step `t`; **coefficients of the query, sent in the clear** | e.g. `[8, −2, 10]` |
| `q` | prime field size, from the public worst-case bound `B` | 1,468,006,403 (`log2 q = 30.45`) |
| `m` | key length, from Theorem 1 | ≈ 77, recomputed from exact entropy in §3 |
| `λ` | workers = `C(m+2, 2)` | ≈ 3,081 |
| `η` | learning rate; the update is `w ← w − (η/n_s)·g` | 0.5, public (§4.3) |
| `T` | number of GD steps; fixed in advance and public | 200 |
| `ε_c` | Theorem 1 per-subset leakage bound, in `log_q` units | 0.0013, i.e. **0.04 bits** |

One gradient step costs `P = 3` queries. Measured at this size (`results/q_sweep_C.csv`):
**1.7 s per step** warm and 3.4 s of one-time setup. So `T = 200` takes about 6 minutes.

---

## 2. What the provider sees, and what Theorem 1 covers

### 2.1 The view

During a training run, the provider (admin and all workers, colluding) holds:

| item | public? | depends on the data? |
|---|---|---|
| `q, n, m, d, r`, the formats `f_X, A_X, f_w, A_w`, `n_s, P` | yes | no, they are set from the public format (`QUANTIZATION.md` §7.2) |
| `G` | yes | no |
| `X̃ = X + kG` | yes | yes, but this is the channel Theorem 1 bounds |
| the queries `f_{ŵ_1}, …, f_{ŵ_T}`, i.e. **the weight codes `ŵ_t`** | yes | **yes, because `ŵ_{t+1}` is computed from the decoded gradient at step `t`** |
| the worker outputs `f_{ŵ_t}(X̃ − tG)` for every `t ∈ I_{d,m}` | yes | adds nothing (Lemma A) |
| the refusal statuses, and `T` itself | yes | only through `ŵ_t`; `T` is fixed in advance |
| `η`, `w_0`, the rounding rule | effectively yes: they are recoverable from the `ŵ_t` sequence, so treat them as public | no, if they are set from the design (§4.3) |

### 2.1a Why the weights are not hidden by the key

Only the **data** is encoded. The key `k` masks `x` at upload (`x̃ = x + kG`), and it is used
once more, at decode, to evaluate the interpolated `g` at `k`. The query polynomial is never
encoded. The workers have to evaluate it on their shares, so the admin receives it in the
clear: "they share the identity of `f` with the system admin" (ICC §II). In live GD the
weights are the coefficients of that polynomial.

A concrete step at the case study. The client holds `w_t = [0.5, −0.125, 0.625]`, so
`ŵ_t = [8, −2, 10]` (×16, since `f_w = 4`). Component 0 of the gradient query is

```
g_0(z) = Σ_i ( 8·z_{i,0} − 2·z_{i,1} + 10·z_{i,2} − 16·z_{y_i} ) · z_{i,0}
```

and the admin reads `8, −2, 10` directly. The decoded gradient goes only to the client, and that
part is private. But the client then sends step `t+1` with, say, `ŵ_{t+1} = [9, −2, 10]`, and
`ŵ_{t+1} − ŵ_t ≈ −(η/n_s)·g_t·16`. **So each decoded gradient reaches the provider one step later,
rounded to the weight grid, through the next query.** At the end, the provider holds `ŵ_T`,
which is the fitted model.

The weights cannot be moved under the key without changing the protocol. They would have to
become data, which means a fresh upload every step (a re-upload, known issue 6) and a degree-3
query `x·x·w`. The alternative that does work is §2.4: make the queries independent of the
data.

### 2.2 Three reductions

These are elementary, and each one gets a numerical check in §7.

**Lemma A (worker outputs are free).** Each worker output is `f_{ŵ_t}(X̃ − tG)`. The provider
holds `X̃`, `G`, `t` and `f_{ŵ_t}`, so it can compute every output itself. Repeated queries
against one stored `X̃` therefore leak nothing beyond `X̃` and the query polynomials. This
answers half of PLAN open question 6: *reusing the key is not the problem; choosing the
queries adaptively is.*

**Lemma B (`X̃` is equivalent to the syndrome).** Fix `G` and let `C` be its row space. Given
`X = x`, the encoding `X̃` is uniform on the coset `x + C`. That distribution depends on `x`
only through the coset, i.e. the syndrome `S = x·Hᵀ`, where `H` is a parity-check matrix of
`C` (`H` has `n − rank G` rows). So for any random variable `Z` that is a function of `X`,

```
I(X_R ; X̃, Z) = I(X_R ; S, Z)        with S = X·Hᵀ ∈ F_q^{n − rank G}
```

The key `K` drops out of the computation entirely. Use `rank G`, not `m`, because a random
`G` over a small field can be rank-deficient. If `rank G = n`, then `S` is empty and the
storage leakage is exactly 0 (the one-time pad).

**Lemma C (GD only sees the sufficient statistics).** The integer gradient is
`ĝ_t = X̂ᵀX̂·ŵ_t − 2^{f_w}·X̂ᵀŷ`. The update is a fixed public function of `ĝ_t` and the client's
float `w_t`. So the whole sequence `ŵ_{1:T}` is a function of `(X̂ᵀX̂, X̂ᵀŷ)`, together with
public constants and any client-secret rounding randomness, which is independent of `X`. By
data processing,

```
I(X_R ; ŵ_{1:T})  ≤  I(X_R ; X̂ᵀX̂, X̂ᵀŷ)        for every T, η, f_w
```

So more steps cannot leak more than the `P(P+1)/2 + P = 9` sufficient statistics (6 entries of
`X̂ᵀX̂` and 3 of `X̂ᵀŷ`). They can leak less, because quantising `ŵ` and the fixed point the
iteration stalls at both lose information.

### 2.3 Why this matters numerically (probe)

The probe is a throwaway script, not committed. E5 will regenerate these numbers properly.

Setup: `n_s = 3`, `P = 1` (so `n = 6`), 9 levels (`f_X = 1`, `A_X = 2`), Gaussian features,
labels `y = 0.75·x + noise`. The leakage is computed exactly by enumerating all `9^6`
configurations.

| quantity | bits |
|---|---:|
| `H(x_1)`, everything there is to know about one feature value | 2.97 |
| `I(x_1 ; X̂ᵀX̂, X̂ᵀŷ)` | **0.94** |
| `I(y_1 ; stats)` | 0.33 |
| `I((x_1, y_1) ; stats)` | 1.57 |
| Theorem 1 budget at the case study, `ε_c` in bits | **0.04** |

At `n_s = 3` the statistics are sums of only 3 terms, so this is the worst case. At `n_s = 25`
the leakage will be smaller, but nothing makes it negligible. A second probe at the case-study
size used a naive linear attack on the public `ŵ` sequence: it recovered `X̂ᵀŷ` to 20% relative
error. It did not recover `X̂ᵀX̂`, because `ŵ` stalls on its `1/16` grid. And the final public
`ŵ_T = [0.5, −0.125, 0.4375]` is the OLS solution `[0.51, −0.15, 0.41]`, rounded to that grid.
A smarter attack would recover more. These numbers are **lower bounds** on what leaks.

### 2.4 The mitigation exists, and it is cheap (flag; not in scope unless D1 says so)

The least-squares gradient is **affine in `w`**: `∇(w) = X̂ᵀX̂·w − X̂ᵀŷ`. So the client can query
the 9 sufficient statistics once, using fixed polynomials that **do not depend on the data**,
and then run all of GD locally. The query channel then carries exactly zero bits, the total
leakage is `I(X_R; X̃) ≤ ε_c`, and the queries have degree 2 in the data with no weight
factor, so `q` is smaller. This is the "one-shot" variant of `PLAN.md` Week 6, but it changes
what the comparison is about. It is not just "the price of generality". **Live GD is strictly
leakier than the one-shot variant, and the gap is exactly the query channel measured in §5.**
The same trick covers logistic regression with a polynomial sigmoid: the gradient is then a
polynomial in `w` whose coefficients are data moments of degree up to `d`. It needs more
moments but stays data-independent.

---

## 3. The synthetic source: exact entropies by construction

### 3.1 Design

The source generates integer codes directly from known probability mass functions (pmfs).
This avoids two problems. Sample standardisation is a public transform that depends on the
data (open question 4), and the entropies would otherwise have to be estimated.

```
features   X̂_ij  i.i.d. ~ π_X        π_X = N(0,1) discretised at step 2^-f_X, clipped at ±A_X
noise      Ê_i   i.i.d. ~ π_E        N(0, σ_E²) discretised the same way
labels     ŷ_i = clip( round(Σ_k w*_k · X̂_ik) + Ê_i ,  ±X̂max )
```

- The true weights `w*` and the noise level `σ_E` are fixed so that `Σ w*_k² + σ_E² = 1`. The
  labels then have unit variance by design, and nothing is standardised from the sample.
  Default `w* = [0.5, −0.25, 0.625]`, which gives `σ_E = 0.545`.
- The pmfs are computed by hand from `math.erf`: the mass of each lattice cell, with each
  clipped tail collected on its boundary code. This is the same construction as
  `quantization_study.discretised_gaussian_pmf`, but without SciPy, and moved into a module
  that the tools and tests can import.
- The samples are independent, so the data vector is `n_s` i.i.d. **blocks** of `P+1`
  coordinates (one sample's features and its label). Inside a block the label is correlated
  with the features.
- Extra families for the leakage sweeps, so that entropy can be varied: uniform over the
  codes (maximum entropy), and a "spiked" pmf with extra mass on 0 (low entropy, like the
  skew distribution in `test_icc.py`).

### 3.2 Exact Rényi entropies (order `p = 2`, and any order)

`H_p(X)`: the blocks are independent, so `H_p(X) = n_s · H_p(block)`. Rényi entropy is additive
over **independent** blocks, which is exact here. It is not additive over correlated
coordinates, which is why the block is the unit.

`H_p(block)`: `Σ_{x,y} P(x,y)^p = Σ_x π(x)^p · c(u(x))`, where `u(x) = round(w*·x)` and
`c(u) = Σ_y P(y | u)^p`. `P(y | u)` is `π_E` shifted by `u`, plus the clip atoms, so `c`
depends only on the scalar `u`. Then the sum over `x` is a sum over `u` under the *tilted*
product measure `π(x_k)^p`. Its distribution is a 1-D convolution of the scaled, tilted
marginals. **Exact. No independence assumption between `x` and `y`.**

Rounding inside `u`: put `w*` on a dyadic grid (multiples of `2^-a`) so that `w*·x` is exact on
the `2^-a` lattice before it is rounded. The default above is a multiple of `2^-3`, so `a = 3`.

`max_R H_p(X_R)`: an `r`-subset touches some blocks. Its entropy is the sum, over those blocks,
of the entropy of the coordinates it takes from each one. For each of the `2^{P+1} = 16`
patterns `S ⊆ {x_1, …, x_P, y}`, compute `H_p` of that marginal exactly with the same tilted
convolution. Then a small knapsack over blocks (size `c = |S|`, value `H_p`, total size `r`)
gives the exact maximum. Under the uniform distribution the best pattern is presumably "one
coordinate from each of `r` different samples", but the knapsack does not assume it.

Unit conversion: all of this is in bits, so divide by `log2 q` to get `log_q` units
(`QUANTIZATION.md` §2). Clamp to `≤ n`, as `entropy_bounds` does.

### 3.3 What the exact values buy

- `m` comes from the true entropies, which is the cleanest possible input to Theorem 1 (D2).
- **The first real check of `quantize.entropy_bounds`** (open issue 3 / open question 9): its
  lower bound on `H_p(X)` must sit below the exact value, and its upper bound on
  `max_R H_p(X_R)` must sit above. Report the slack in bits. The labels are correlated, so
  this tests the joint-density form, which is the form whose rigour actually matters.
- **A signed measurement of the i.i.d. estimator's error** (open issue 2):
  `utils.compute_p_entropy` applied to one sample versus the truth. Report it, and keep it
  well away from `m`.
- Validation: at tiny alphabets, the convolution code must agree with brute-force enumeration
  of the joint pmf to `1e-12`.

---

## 4. The training protocol

### 4.1 Roles in code, kept apart on purpose

To keep the guarantee checkable, the code is split by **who holds what**:

| role | holds | code |
|---|---|---|
| client (trainer) | `k`, its float `w_t`, the public schedule, decoded results | `icc/training.py`; it receives a `Client` and a `Server` and **is never given `X`** |
| provider | `X̃, G`, queries, worker outputs | the existing `Server`, wrapped by a recorder that logs the provider's view verbatim |
| harness (experimenter) | the plaintext data, `w*`, and the oracles | `tools/gd_experiment.py` |

The recorder (`ProviderTranscript`) is what the leakage code consumes. Leakage is computed
from **what was actually sent**, not from a model of what is sent. It does not modify
`Server`. It wraps `compute_request` and stores each polynomial's integer coefficients and
the returned status.

### 4.2 Upload (once per run)

```
1. format     x_spec = (f_X=7, A_X=4), w_spec = (f_w=4, A_w=2), n_s=25, P=3, d=2
2. q          QueryBudget(...).magnitude_bound() → choose_field_size(public-format) → 1,468,006,403
3. entropy    exact H_p(X), max_R H_p(X_R) from §3 (D2), in log_q units
4. m, ε_c     compute_required_m, compute_leakage_bound      (report both ε_c exponents, issue 1)
5. data       harness samples codes from the source (rng stream "data")
6. G, k       generate_random_G(rng stream "G"); Client(rng stream "key")
7. upload     codes_to_field → client.encode_data (checks |code| ≤ X̂max) → server.store_data
8. warm M     client._interpolation_matrix(...)   (built once, cached for all T·P decodes)
```

Steps 1–4 are the pending `SystemContext.from_budget` item from the CLAUDE.md handoff. Build
it here as a small helper, `tools/gd_setup.py::upload_context`, since GD needs exactly this
chain. Promote it to `config.py` only if Daniel wants it there.

### 4.3 One step (repeated `T` times, `T` fixed)

```
ŵ_t  = to_integer(clip(w_t, ±A_w), w_spec)                          # public from here on
f_j  = least_squares_gradient(n_s, P, ŵ_t, f_w, j)   for j = 0..P-1
r_j  = server.compute_request(f_j)                                   # gate: OK expected
ĝ    = lift_balanced(client.decode_results([r_0..r_{P-1}]), q)       # one solve, P columns
g    = ĝ / 2^(2 f_X + f_w)                                           # real gradient, Σ over samples
w_{t+1} = clip(w_t − (η / n_s) · g, ±A_w)                            # float, client-side
```

Things this fixes deliberately:

- **The update uses the float `w_t`, not `ŵ_t`.** Quantisation error then does not accumulate
  (`PLAN.md` design decision 1). The provider sees only `ŵ_t`.
- **The gradient is a sum, with no `1/n_s`.** `η/n_s` absorbs it. Both are public.
- **Hyperparameters come from the design, never from the data.** `w_0 = 0`. `η` is chosen from
  the *population* covariance (`I` for this source). The Hessian of the per-sample loss is
  `XᵀX/n_s`, and its largest eigenvalue for `n_s = 25`, `P = 3` is about
  `(1 + √(P/n_s))² ≈ 1.81` (the Marchenko–Pastur edge), so stability needs `η < 1.1`; use
  `η = 0.5`. `A_w = 2` is the public commitment from `QUANTIZATION.md` §6.1. Choosing any of
  these from the sample would be a leak, through the same mechanism as `q` (§7.1 there).
- **`T` is fixed, with no early stopping.** A stopping rule that reads the decoded gradient
  would publish one more data-dependent number, the stopping time.
- **Rounding of `ŵ`:** nearest by default. Stochastic rounding uses a separate client-secret
  rng stream; it is an option, not the default. Error feedback stays out of scope (Week 7).
- **A refusal is a bug.** Clipping `w` to `A_w` puts every query inside the budget, so any
  status other than OK raises with the server's reason. It is never skipped silently.
- **No loss queries.** The harness computes the loss in plaintext for the plots. A loss query
  (`Σ(x·w − y)²`, degree 2, with a `y²` term) would be one more data-dependent public channel,
  and the client does not need it.

### 4.4 References the run is checked against (harness)

1. **Plaintext integer GD**: the same loop, with `ĝ` computed exactly in Python integers.
   The scheme trajectory must equal it **bit for bit**: the same `ŵ_t` and the same float
   `w_t` at every step. This is the strongest end-to-end correctness check available.
2. Float GD on the dequantised codes: isolates the effect of quantising `w`.
3. Float GD on the real-valued data (before quantisation): isolates the effect of quantising
   the data.
4. The OLS optimum, from the normal equations solved by hand (Gaussian elimination on a 3×3
   system), plus the population optimum `w*`.

Metrics per step: loss (plaintext), `‖w_t − w_OLS‖`, `‖w_t − w*‖`, gradient error against
reference 3, the integer `max |ĝ|` against the gate's `B(f)`, and timing (eval, decode).

---

## 5. Measuring the leakage

### 5.1 Tier 0: correctness of everything the leakage code assumes

Every lemma in §2.2 gets a brute-force check at a tiny size (§7). Without those checks, the
exact numbers in Tier 1 rest on algebra nobody has run.

### 5.2 Tier 1: exact, at toy scale

**Size.** The support of `X` must be enumerable. With `L` levels per symbol that is `L^n`
configurations: `9^6 = 531k` (`n_s = 3, P = 1`) or `5^10 = 9.8M` (`n_s = 5, P = 1`). The field
`q` does **not** need to be small. Enumeration is over `X`, not over `F_q`, so `q` stays the
true `next_prime(2B+1)` for the toy format, and the gradient queries remain decodable.

**Computation**, all in bits and also converted to `log_q` units:

```
for each configuration x (vectorised):  prob P(x),  syndrome S(x) = x·Hᵀ mod q,
                                       trajectory ŵ_{1:T}(x) from plaintext integer GD,
                                       statistics (X̂ᵀX̂, X̂ᵀŷ)(x)
I(X_R ; Z) = H(X_R) + H(Z) − H(X_R, Z)    by grouping identical keys (np.unique on rows)
```

This is valid because the trajectory through the scheme equals plaintext integer GD
(§4.4 check 1, verified on a sample of configurations through the real `Client`/`Server`).
Report every `r`-subset `R` and the maximum over `R`, which is what Definition 1 bounds.

**Experiments:**

- **E4: storage leakage vs `m` (a direct test of Theorem 1).** Sweep `m` from `r` to `n`. For
  each `m`, draw many `G` (e.g. 200, seeded) and compute `max_R I(X_R; S)`. Plot the empirical
  CDF over `G` and mark `ε_c` and Theorem 1's `m`. Theorem 1 says that at its `m` the leakage
  is `≤ ε_c` with probability `≥ 1 − 1/a`. The measured *smallest* `m` that achieves this
  shows how conservative the theorem is. This repo has never measured `I(X̃; X_R)` directly;
  `test_icc.py` §3 only evaluates the formula. Report both `ε_c` exponents (issue 1).
  - Expected caveat, to state honestly: at toy `n`, Theorem 1's `p + log_q(1/ε)` overhead
    already pushes its `m` to `≥ n`. At `n = 6`, `q = 1163`, `ε = 1e-6` it gives `m ≈ 9`. At
    that `m` the storage leakage is exactly 0, so the informative part is the sweep below the
    bound. An optional field-native variant (small `q`, non-uniform pmf over `F_q`, no GD) is
    the regime where `m < n` is reachable, and it tests the theorem in the paper's own setting.
- **E5: query-channel leakage.** `max_R I(X_R; ŵ_{1:t})` as a function of `t`. Expected: it
  rises over the first few steps and saturates at or below `I(X_R; stats)` (Lemma C). Sweep
  `f_w ∈ {2, 4, 6}` and `η`.
- **E6: the full view.** `max_R I(X_R; S, ŵ_{1:T})`, against `ε_c` in bits. This is the number
  that answers "is live GD under ICC private in the sense of Definition 1?"
- **E6b: the one-shot view, for comparison.** The queries are data-independent, so the view
  is `S` alone and the answer is E4's number. With E6, this gives the price of adaptivity.

**Numerics.** `P(x)` can be as small as about `1e-20`. Accumulate `−Σ p·log2 p` in float64 with
`math.fsum` over the grouped masses. Zero-leakage cases must come out exactly 0
(`H(Z|X_R) = H(Z)` structurally), so treat anything below `1e-12` bits as the precision floor
and say so. Check against `fractions.Fraction` arithmetic once, at the tiniest size.

### 5.3 Tier 2: case-study scale (`n = 100`), what can and cannot be measured

- **Storage channel:** `I(X_R; X̃)` cannot be computed (the support has `1025^100` points), and
  no estimator is trustworthy in that dimension. Report Theorem 1's `ε_c` from the exact
  entropies, plus the E4 evidence on how tight the theorem is. Say plainly that this is a
  bound, not a measurement.
- **Query channel (E7):** run the provider's attack on the recorded transcript, by hand:
  reconstruct `X̂ᵀŷ` and `X̂ᵀX̂` from the `ŵ` sequence using the public `η`, `n_s` and rounding
  rule. Report the reconstruction error and how close the final `ŵ_T` is to `w_OLS`. This is
  a **lower bound** on the leakage.
- **Bridging estimate (optional, label it as an approximation):** a Gaussian/CLT
  approximation of `I(X_R; X̂ᵀX̂, X̂ᵀŷ)` at `n_s = 25`, calibrated by checking it against the
  exact Tier 1 value at the toy sizes, where both exist. Keep it only if it tracks the exact
  value there.

---

## 6. Code layout

New files only. `icc/` keeps depending on nothing but `galois` and `numpy`.

| file | contents |
|---|---|
| `icc/training.py` | `GDSchedule` (public: `η, T, w_0`, rounding, `f_w, A_w`); `train_linear_regression(client, server, schedule, n_s, P, f_X) → Trajectory` (the `ŵ_t`, float `w_t`, decoded `ĝ_t`, statuses, timings). It never receives the plaintext. |
| `tools/synthetic_source.py` | `LinearGaussianSource` (pmfs from `math.erf`, `sample(rng)`); `exact_block_entropy`, `exact_entropies(p, r) → (H_p(X), max_R H_p(X_R))` via tilted convolution and the knapsack; brute-force twins for testing |
| `tools/leakage.py` | `parity_check(G)` (row-reduce over `GF(q)`; `rank G`); `syndromes(X_codes, H)`; `exact_mi(prob, keys_a, keys_b)`; `enumerate_support(source)`; `ProviderTranscript` |
| `tools/gd_setup.py` | `upload_context(format, source, seeds)`: the §4.2 chain in one place |
| `tools/gd_experiment.py` | E1–E7, each one seeded and writing CSV to `results/gd_*.csv` plus a printed report to `results/gd_*.txt`; `--fast`; `--exp` |
| `tools/gd_sweep.py` | Phase 1 distribution sweep S1–S4 (§10.6), resumable |
| `tools/plot_gd.py` | §10.7 figures, loss curves, the leakage-vs-`m` CDF, leakage vs `t`; PDF and PNG into `results/figures/` |
| `tests/test_gd.py` | the §7 checks; a fast subset is also added to `tests/test_regression.py` |

Seeds: one `np.random.SeedSequence` per run, spawned into the named streams `data`, `G`, `key`,
`round`. Changing one stream must not move the others. Never seed a real client's key; the
seeded key is for experiments only (`Client` docstring).

---

## 7. Tests (all fast, brute force where possible)

1. **Trajectory exactness:** scheme GD equals plaintext integer GD bit for bit (`ŵ_t` and `w_t`)
   at `n_s = 4, P = 2`, 20 steps. And at every step, the decoded `ĝ` equals the integer oracle.
2. **Gate:** every step's queries return OK. A deliberately over-budget `ŵ` returns
   `MAY_OVERFLOW` and the trainer raises.
3. **Lemma A:** the provider's own computation of `f(X̃ − tG)` equals the worker outputs.
4. **Lemma B:** at `q = 5, n = 4, m = 2`, `I(X_R; X̃)` computed by enumerating `(x, k)` jointly
   equals `I(X_R; S)` from the syndrome. Also check a rank-deficient `G`.
5. **Lemma C:** `I(X_R; ŵ_{1:T}) ≤ I(X_R; stats)` on the toy source, for several `T`.
6. **Exact entropies:** the convolution route equals brute-force enumeration of the block pmf
   (tiny alphabet, `P = 1, 2`), for `p = 2, 3`. The knapsack maximum equals brute force over
   all `C(n, r)` subsets at `n = 6`.
7. **Sanity bounds:** `rank G = n` gives leakage exactly 0; `m = 0` gives `I = H(X_R)`;
   `entropy_bounds` lower ≤ exact ≤ `n`, and exact `max_R` ≤ upper bound.
8. **Hygiene:** `train_linear_regression` has no parameter through which the plaintext can be
   passed, and the `Client` holds no attribute derived from `x` after `encode_data`.

---

## 8. Order of work, with checkpoints (superseded by §10.9 for Phase 1)

1. `tools/synthetic_source.py` with exact entropies and tests 6–7. **Checkpoint:** print the
   case-study `m` from exact entropy next to the `entropy_bounds` and i.i.d.-estimator values.
2. `icc/training.py`, the reference loops, and tests 1–2. **Checkpoint (E1):** one step at the
   case study, giving decoded vs integer oracle vs float gradient (the `PLAN.md` W5 claim).
3. **E2:** full training at the case study (`T = 200`, about 6 min), loss curves for the
   scheme and the three references, with the bit-for-bit check. **This is the `PLAN.md` W6
   checkpoint.**
4. `tools/leakage.py` and tests 3–5. **E4** (storage vs `m`), **E5** (query channel),
   **E6/E6b** (full view, one-shot comparison).
5. **E7** (case-study attack). Then figures, and a results write-up appended to this file.

The timing in E2 follows the house rules: `--jobs 1`, on AC power, under `caffeinate`. The
leakage experiments do not time anything. The `5^10` enumeration with a 200-step plaintext
trajectory per configuration is the heaviest piece (minutes, vectorised over configurations).
Start that run only after checking that the laptop is on power.

---

## 9. Decisions (D1–D4: settled 2026-10-21, all as recommended) and things flagged rather than decided

- **D1. Threat model for the weights.** Is the provider allowed to learn `ŵ_t`, i.e. the
  model? Recommended: implement live GD as requested, measure the query channel (E5–E7) and
  report it as a finding, with the one-shot variant (§2.4) as the comparison. The other
  options are to switch to data-independent queries now, or to build both from the start.
- **D2. Which entropy sets `m` for the synthetic runs.** Recommended: the exact values from §3,
  with `entropy_bounds` and the i.i.d. estimator reported alongside and never used for `m`.
- **D3. Leakage scope.** Recommended: exact Tier 1 at toy scale, plus the Tier 2 attack lower
  bound at the case study. The alternative is a general-purpose MI estimator at `n = 100`,
  which I would not trust.
- **D4. Source shape.** Recommended: Gaussian features, linear-Gaussian labels with
  `w* = [0.5, −0.25, 0.625]`, no intercept (the source has zero mean by construction),
  `η = 0.5`, `T = 200`, nearest rounding. **Superseded for the distribution sweep by §10
  (C3: an intercept is added; C4: `η` and `T` per distribution, from the design).** The
  defaults still apply to E1/E2 at the case study.

**For Raviv** (add to `PLAN.md` open questions as 10–11 once Daniel agrees):

10. Definition 1 bounds `I(X_R; X̃)`. An adaptive protocol publishes queries that depend on
    earlier decoded results. Is the intended guarantee `I(X_R; X̃, f_1, …, f_T) ≤ ε_c`? If so,
    is live GD acceptable only with data-independent queries (the moments)?
11. Is the model itself (`w_T ≈ w_OLS`) meant to be private from the provider? If it is not,
    the query channel is "allowed" leakage, but it still has to be accounted for per subset.

**Known issues this work touches, without changing them:** issue 1 (`ε_c` exponent: report
both), issue 2 (i.i.d. entropy: measured, not used), issue 3 (`entropy_bounds`: checked against
the truth), issue 6 (re-upload: not used; a single upload serves every step). Logistic
regression, `load_diabetes`, CRT and stragglers remain out of scope for this round.

---

## 10. Phase 1: convergence across synthetic distributions (added 2026-11-10)

**Goal.** Train by live GD on many synthetic distributions, both through the scheme and
unencoded, and compare convergence by data type. Convergence comes first; the leakage work
(§5, E4–E7) becomes Phase 2.

**Decisions (2026-11-10):**
- **C1. Convergence is measured both in steps and in wall-clock time, kept separate.**
- **C2. Every run goes through the real scheme.** The plaintext integer loop is kept only as
  the bit-for-bit check.
- **C3. An intercept is added**, folded into the query as a public coefficient. This replaces
  D4's "no intercept".
- **C4. The learning rate is set per distribution, from the design.**

### 10.1 What "encoded vs unencoded" measures

Encoding adds **zero** arithmetic error: the scheme's trajectory equals plaintext integer GD bit
for bit (§4.4 check 1, asserted in every run). So the comparison splits cleanly into two parts:

| axis | encoded | unencoded | what the difference is |
|---|---|---|---|
| steps to converge | scheme GD on fixed-point codes | float GD on the real-valued data (§4.4 ref. 3) | **quantisation only**: `f_X = 7` data and the `1/16` weight grid |
| wall-clock to converge | steps × measured (eval + decode) per step, plus setup (field, `M`, encode, store) | steps × float step time | **the scheme's overhead**, about 1.7 s/step at `n = 100` against microseconds |

Float GD on the dequantised codes (ref. 2) sits between the two and separates the data's
quantisation from the weights'. The unencoded loop is hand-written NumPy: a matrix–vector
product per step, with no library solver.

### 10.2 The intercept (C3)

The model is `y ≈ x·w + b`, and the bias `b` is quantised like a weight: `b̂ = round(b·2^{f_w})`,
clipped at a public `A_b` (default 4). The residual of sample `i`, at the common scale
`2^{f_X + f_w}`:

```
r_i(z) = Σ_k ŵ_k·z_{X(i,k)}  +  2^{f_X}·b̂  −  2^{f_w}·z_{y(i)}        # 2^{f_X}·b̂ is a constant term
g_j(z) = Σ_i r_i(z) · z_{X(i,j)}          j = 0..P-1   (degree 2)
g_b(z) = Σ_i r_i(z) · 2^{f_X}                           (degree 1)
```

All `P + 1` components land at the scale `2^{2f_X + f_w}`, so they descale together. `Polynomial`
already supports constant terms. The server's gate is computed per query, so it needs no
change.

Changes to `icc/`:
- `query.least_squares_gradient` gains an optional `bias_code` and a `j = P` bias component.
- `quantize.gradient_bound_worst_case` and `QueryBudget` gain `A_b`, and the residual bound
  becomes `P·X̂max·Ŵmax + 2^{f_X}·B̂max + 2^{f_w}·Ŷmax`. The bias component's bound is
  `n_s·2^{f_X}·(…)`, which is no larger than the feature components' bound, since
  `2^{f_X} ≤ X̂max`.
- With the defaults (`X̂max = 512, Ŵmax = 32, B̂max = 64`) this collapses to **`B = n·2^23`**, i.e.
  `log2 q ≈ 24 + log2 n`. That is 30.6 at `n = 100` and stays native up to `n ≈ 180`. So **one
  format (`f_X = 7, A_X = 4, f_w = 4, A_w = 2, A_b = 4`) serves every size in the grid**, which
  keeps the comparison fair.
- Tests: the bias component decodes exactly; the new `B` matches `worst_case_bound` with every
  weight at its clip; and the old no-intercept path is unchanged (regression oracle).

The intercept does **not** remove the conditioning cost of a non-zero mean. It moves it into the
augmented second-moment matrix `E[[x;1][x;1]ᵀ]`. For one feature with mean 1 and variance 1 that
matrix is `[[2,1],[1,1]]`, with `κ = 6.9`, against 1 for centred data. This is a real effect and
the sweep should report it (shape family `exponential` below).

### 10.3 Distribution catalogue

Every family is defined by its exact CDF, written by hand (`math.erf`, `math.exp`,
`math.atan`), and discretised onto the code lattice cell by cell, with clipped tails collected on
the boundary codes. That keeps §3's exact entropy available for independent features. Unless
stated otherwise, features have **population variance 1**. Their means are left as they come out,
and the intercept handles them.

| family | definition | mean | what it probes |
|---|---|---|---|
| `gauss` | `N(0,1)` | 0 | baseline |
| `uniform` | `U[−√3, √3]` | 0 | maximum entropy for a bounded range; light tails |
| `laplace` | scale `1/√2` | 0 | moderately heavy tails |
| `student3` | Student-t with 3 degrees of freedom, scaled by `1/√3` (its CDF has a closed form) | 0 | heavy tails; clip saturation at `±A_X` |
| `exponential` | `Exp(1)` | 1 | skew, positive support, non-zero mean |
| `lognormal` | `exp(N(0, 0.75²))`, rescaled to variance 1 | > 0 | strong skew plus a heavy right tail |
| `spiked` | 80% exactly 0, 20% `N(0, 5)` | 0 | a large atom: low entropy, so a larger `m` and a costlier scheme |
| `bimodal` | `½N(−0.9, 0.436²) + ½N(0.9, 0.436²)` | 0 | multimodal |
| `discrete3` | uniform on `{−1, 0, 1}·√1.5` | 0 | few-valued, categorical-like; very low entropy |
| `corr(ρ)` | Gaussian, pairwise correlation `ρ ∈ {0, 0.5, 0.9, 0.95}` | 0 | **conditioning**: `κ = (1 + (P−1)ρ)/(1−ρ)`, i.e. 28 at `ρ = 0.9`, `P = 3` |

**Labels:** `y = x·w* + b* + e`, with `e ~ N(0, σ_E²)` by default and Laplace noise as a variant.
`w*` and `b*` are fixed per run from the design. The noise is set by a signal-to-noise ratio
(SNR) `∈ {10, 1, 0.1}`; at the default SNR `y` has unit variance. Labels are clipped at `±A_X`
like the features.

**Sizes** `(n_s, P)`, all with `n = n_s(P+1) ≤ 110` so they stay within the scheme's practical
range:

| `(n_s, P)` | `n` | regime |
|---|---:|---|
| (25, 3) | 100 | the case study |
| (50, 1) | 100 | many samples, one feature |
| (10, 9) | 100 | many features, few samples |
| (6, 10) | 66 | more features than samples: `XᵀX` is singular, and GD converges to the minimum-norm solution, so `κ` is reported on the range |
| (10, 3) | 40 | small and fast; the scaling reference |

`load_diabetes` and `n_s ≫ 50` remain out of scope for the scheme. They can run unencoded only,
if wanted.

### 10.4 Learning rate and run length, from the design (C4)

- `η = 1/λ_max(H_pop)`, where `H_pop` is the **population** augmented second-moment matrix of
  the family. It is computed exactly from the family's parameters, never from the sample. The
  update stays `w ← w − (η/n_s)·g`.
- `T` is fixed per configuration from the design: `T = max(200, ⌈3·κ_pop·ln(1/tol_min)⌉)` with
  `tol_min = 1e-3`, capped at 2,000. That is about 600 steps at `ρ = 0.9` and the cap at
  `ρ = 0.95`, where `κ = 58`. Both are public functions of the design, so a fixed `T` still
  holds (§4.3).
- The sample `λ_max` can exceed the population value. If it passes `2λ_pop` the run oscillates,
  and clipping at `A_w` keeps that bounded. Record it as `diverged`; never retune on the sample.

### 10.5 Convergence metrics (read off the trajectory afterwards)

- `excess_t = (L(w_t) − L(w_OLS)) / L(w_OLS)`: the relative excess loss on the **real-valued**
  data, where `w_OLS` (with intercept) is solved by hand. The same reference is used for the
  encoded and the unencoded runs.
- **Steps to tolerance**, `τ(tol)`: the first `t` after which `excess` stays `≤ tol`, for
  `tol ∈ {1e-1, 1e-2, 1e-3}`.
- **Plateau**: the median `excess` over the last 10% of steps. For the encoded runs this is the
  quantisation floor. If `tol` is below the floor, `τ` is reported as "not reached", never as
  `T`.
- **Wall-clock** `= setup + τ × per-step time` (C1), with per-step eval and decode measured in
  every run.
- Context per run: the sample `κ` and `κ_pop`, the clip saturation fraction (data and weights),
  `‖w_T − w*‖`, `m`, `λ` and `h`.

### 10.6 Grid and cost (C2: every run through the scheme)

Five seeds per cell. Each seed is a fresh dataset, `G` and key, drawn from its own spawned stream.

| sweep | cells | runs |
|---|---|---:|
| S1 shape | 9 families at (25, 3), SNR 1 | 45 |
| S2 conditioning | `corr(ρ)`, 4 values, at (25, 3) | 20 |
| S3 size | 5 sizes × {`gauss`, `exponential`, `spiked`} | 75 |
| S4 noise | SNR 3 values × {Gaussian, Laplace} noise, `gauss` features | 30 |

Estimated cost:
- At `n = 100` a step costs about 1.7 s, so a 200-step run takes about 6 min.
- Low-entropy families (`spiked`, `discrete3`) push `m` toward `n`. `λ` then rises to about
  5,000, roughly doubling the cost per step.
- The `ρ = 0.95` cells run 2,000 steps, about an hour per seed.
- **Total: about 20–25 h.** That means overnight runs, on AC power under `caffeinate` with
  `--jobs 1` (the house rules for reported timings), or on the DGX Spark. There the step
  counts are unaffected, but the timings should be re-measured one cell at a time.

The driver writes one CSV row per run plus a per-step trajectory file, and supports `--resume`
like `q_sweep.py`, so a sleep or a battery stop loses at most one run.

### 10.7 Figures (`tools/plot_gd.py`)

1. Excess loss vs step, encoded vs unencoded, one panel per family (S1).
2. **Steps to `tol = 1e-2` vs `κ`**, every run as a point coloured by family, with the theory
   line `κ·ln(1/tol)`. I expect this to be the central figure.
3. Plateau (quantisation floor) by family: what fixed point costs in accuracy.
4. Wall-clock to converge, encoded vs unencoded (log scale), split into setup and per-step.
5. Per-step cost and `λ` by family: low-entropy data is more expensive to protect.
6. Steps and wall-clock vs size (S3); excess loss vs SNR (S4).

### 10.8 Entropy for `m` in this phase

D2 still holds where it can: the exact entropy (§3) is used for independent features with any
of the marginals above, since the tilted-convolution method only needs the marginal pmfs. For
`corr(ρ)` the features are not independent. Those runs use the joint-density form of
`quantize.entropy_bounds`, recorded in a `m_source` column. For this phase `m` affects only `λ`
and so the wall-clock time; it never changes the step count.

### 10.9 Revised order of work

1. `tools/synthetic_source.py`: the catalogue above, its exact pmfs, and sampling. Exact
   entropies for independent features (tests 6–7).
2. The intercept in `query.py` / `quantize.py`, with its tests.
3. `icc/training.py` (`P+1` components), the plaintext integer and float references, and
   tests 1–2. Then E1 and E2 at the case study as before.
4. `tools/gd_sweep.py`: the S1–S4 driver (resumable, one row per run), on top of the §4.2 upload
   chain. A `--fast` grid with 1 seed and short `T` for smoke tests.
5. `tools/plot_gd.py`. **Phase 1 checkpoint:** figures 1–6 from a full run.
6. Phase 2: leakage (§5, E4–E7), unchanged.

### 10.10 Implementation status (2026-11-10): ready to run

Phase 1 is implemented and tested. Only the full sweep itself remains to be run.

| file | what it does |
|---|---|
| `icc/query.py`, `icc/quantize.py` | the intercept (§10.2): `least_squares_gradient(..., bias_code=, f_X=)`, and `QueryBudget(b_spec=)` / `gradient_bound_worst_case(b_spec=)`. The old no-intercept path is unchanged; `test_regression.py` still passes 77/77 |
| `icc/training.py` | `GDSchedule`, `run_gd` (the loop), `scheme_gradient`, and `train_linear_regression(client, server, …)`. The trainer has no input through which the plaintext could arrive |
| `tools/synthetic_source.py` | the catalogue (§10.3): CDFs and samplers written by hand, exact discretised pmfs, the design's `w*`, `b*` and `σ_E`, and exact Rényi entropies (tilted convolution plus a knapsack), with a brute-force twin |
| `tools/gd_reference.py` | plaintext integer GD (the bit-for-bit oracle), float GD (the unencoded baseline), a Jacobi eigen-solver, the minimum-norm least-squares solve, and the metrics |
| `tools/gd_setup.py` | the §4.2 upload chain, and the design schedule (`η`, `T`) |
| `tools/gd_sweep.py` | the S1–S4 driver. `--dry-run`, `--fast`, `--resume`, `--sweep`, `--seeds` |
| `tools/plot_gd.py` | figures 1–6 into `results/figures/gd_fig*.pdf/.png` |
| `tests/test_gd.py` | 38 checks, about 2 s |

**Changes from §10.4–10.6 made during implementation:**

1. **The step size has a Marchenko–Pastur factor.** It is now
   `η = 1/(λ_max(H_pop)·(1+√((P+1)/n_s))²)`, and `T` scales by the same factor. With
   `η = 1/λ_max(H_pop)` alone, Gaussian data diverged at `(n_s, P) = (10, 3)`: the sample's
   top eigenvalue sits up to about that factor above the population one. The factor depends
   only on `(n_s, P)`, so `η` is still public and design-only. Measured with plaintext float GD
   over the whole grid, 5 seeds per cell:
   - No cell diverges except `student3`: 2 of 5 seeds at (25, 3) and 1 of 5 at (6, 10), because
     heavy tails push the sample spectrum past the design's prediction. Those runs are recorded
     as `diverged`, which is a finding.
   - The square design (10, 9), where `n_s ≈ P+1`, often fails to reach `1e-2` in 200 steps,
     because the smallest sample eigenvalue collapses toward 0. That is real and is reported
     as "not reached".
2. **The metric is the normalised suboptimality** `(L(w_t) − L*)/(L(w_0) − L*)`, not
   `(L − L*)/L*`. The latter is undefined when `P ≥ n_s`, where `L* = 0`. The new metric starts
   at 1, and a tolerance of `1e-2` means "99% of the initial excess removed". It is also
   recorded against the dequantised data's own optimum (`scheme_vs_deq_opt`), which separates
   the floor set by the weight grid from the bias that quantising the data introduces.
3. **Duplicate cells run once.** `ρ = 0` (S2), size (25, 3) (S3) and SNR 1 with Gaussian
   noise (S4) are all the S1 `gauss` cell, and `plot_gd.py` reuses it. The grid is therefore
   **29 cells × 5 seeds = 145 runs**, not 170.
4. **The cost is about 58 h, not 20–25 h.** `--dry-run` uses a per-step model calibrated
   against `q_sweep` C, and it matched three direct measurements on this machine to within 15%:
   - (25, 3), λ = 3,003: 2.16 s/step measured, 2.19 s estimated.
   - (10, 9), λ = 3,003: 4.78 s measured, 5.48 s estimated.
   - spiked (10, 9), λ = 5,151: 11.45 s measured, 12.45 s estimated.

   The 20–25 h figure in §10.6 left out two things: the intercept's extra query, and the long
   `T` that uncentred and correlated families need. The total is dominated by a few cells:

   | cell | `T` | minutes per seed | hours for 5 seeds |
   |---|---:|---:|---:|
   | exponential (10, 9) | 2,000 | 205 | 17 |
   | ρ = 0.95 | 2,000 | 89 | 7.4 |
   | lognormal (25, 3) | 1,622 | 67 | 5.6 |
   | ρ = 0.9 | 1,138 | 49 | 4.1 |
   | exponential (6, 10) | 2,000 | 46 | 3.8 |
   | exponential (25, 3) | 989 | 41 | 3.4 |

   The other 23 cells together take about 17 h. Options, none applied yet: run the grid as is
   (about 2.5 days, resumable); cap `T` lower; give the expensive cells fewer seeds; or run on
   the DGX Spark. Either way, `--resume` makes it safe to run in chunks: one sweep per night
   with `--sweep S1` and so on.

**How to run** (on AC power, lid open; reported timings need one run at a time, which is the
only mode the driver has):

```bash
.venv/bin/python tests/test_gd.py                                  # 38 checks, ~2 s
.venv/bin/python tools/gd_sweep.py --dry-run                       # grid, T, m, λ, estimated hours
caffeinate -dimsu .venv/bin/python tools/gd_sweep.py --sweep S1    # ~15 h
caffeinate -dimsu .venv/bin/python tools/gd_sweep.py --resume      # everything left
.venv/bin/python tools/plot_gd.py                                  # figures
```
