# Fall 2026 Semester Plan — ICC Implementation

**Goal:** a conference-style paper (~6pp, IEEE two-column) demonstrating that the ICC
scheme can be used to *train* a machine learning model over quantized real-valued data.

**Scope decisions (locked):**
- Model: **linear regression only**, trained by **iterative gradient descent**
  (not the closed form — GD is the general protocol that logistic will need later)
- **Straggler tolerance / information super-sets are OUT OF SCOPE this semester.**
  The scheme runs in the no-straggler regime throughout. Note it as future work.
- Deliverable: conference-style paper. **Results frozen by end of Week 10.**
- Next semester: privacy analysis of *repeated* queries. Design now so that work is possible.

**What replaces straggler tolerance as the semester's second contribution:**
the *quantization ↔ entropy ↔ key-size* chain, and the parameter-efficiency
techniques that follow from it (Weeks 3–4, 7–8). This is closer to the ICC paper's
own contribution (non-uniform data) than super-sets were, and it is the thing that
decides whether the scheme is usable on real data at all.

---

## The finding that shapes everything

Per-symbol entropy `h` (in `log_q` units, `0` = deterministic, `1` = uniform over `F_q`)
controls the key size through Theorem 1:

```
m ≥ n(1 − h) + p + log_q(1/ε) + r·h        λ = C(m+d, d) = workers = download cost
```

The `n(1−h)` term is the whole game: **`h` must be near 1 or `m` scales with `n`.**

### The tension

Two requirements pull `q` in opposite directions:

| Requirement | Pushes `q` |
|---|---|
| **Correctness** — no wraparound. A GD step computes `Σᵢ Σₖ Xᵢⱼ Xᵢₖ wₖ − Σᵢ Xᵢⱼ yᵢ`, a product of **three** quantized factors summed over `n_s·p` terms | **up**, to `q > 2·n_s·p·2^{3f}` |
| **Entropy** — `h = (bits of Rényi-2 entropy per data symbol)/log₂ q`. Large `q` with small data range means small `h` | **down**, toward `2^f` |

With a common scale `f` for all three factors, `h ≈ (f+2)/(3f + log₂(2·n_s·p))` →
**`h ≈ 1/3`**, essentially independent of `f`. Not a tuning problem; a structural one.

> **Correction to an earlier version of this plan.** It claimed "more precision costs
> worker count." That is not what the algebra says. Write `h(f) = (f+c)/(3f+k)`, where
> `c` is the per-symbol entropy overhead above `f` bits and `k = log₂(2·n_s·P)`. Then
>
> ```
> dh/df  =  (k − 3c) / (3f + k)²        →  sign(dh/df) = sign(k − 3c)
> ```
>
> so the effect is **second-order and its sign is data-dependent**, not a law in either
> direction. For the case study (`n_s=15, P=3`): `k = 6.49`, crossover at `c* = k/3 = 2.16`.
> Measured `c` for a discretized standard Gaussian is **1.83, stable across `f`** (`f=4→
> H₂=5.83`; `f=8→9.82`; `f=12→13.77`), i.e. just below the crossover. Net effect over
> `f = 4→32`: `h` moves `0.324 → 0.332`. **Flat.**
>
> The defensible claim for the paper is therefore: **`h` is essentially independent of `f`;
> precision is close to free, and the sign of the residual effect depends on `c` vs
> `log₂(2·n_s·P)/3`, which for our parameters are 1.83 vs 2.16 — nearly equal.** Do not
> claim precision costs workers, and do not over-claim the reverse either; the honest
> result is the near-cancellation, and it is more interesting than either slogan.
>
> The real lever is not `f`. It is the **ratio of quantized factors to data symbols**
> in the denominator — see Weeks 7–8.

### Feasibility envelope

Derived at `r = 5`, `p = 2`, `ε = 1e-6`, `d = 2`, `q ≈ 2³⁰`:

| `n` | `h` | `m` | `λ` |
|---:|---:|---:|---:|
| 60 | 0.33 | 45 | 1,081 |
| 60 | 0.46 | 38 | 780 |
| 100 | 0.33 | 73 | 2,775 |
| 150 | 0.33 | 107 | 5,995 |

**Case study targets `n ≈ 60`** — about **15 samples × 3 features** (`n = n_s(p+1) = 60`),
`f = 8` fractional bits, `q` a prime just above `2³⁰`.

This is small for ML, and that is fine. The paper's contribution is *correctness
end-to-end plus an honest characterization of the parameter regime*, not a competitive
benchmark. **Re-verify this envelope in Week 2** — it is derived, not fully measured.

### The cost model in the last version of this plan was wrong (measured)

It assumed decode cost `~λ³` from a dense `GF(q)` solve, and used that to declare
`n = 500` infeasible. Measured on this machine, `q = 2³⁰+3`, `d = 2`:

| `m` | `λ` | build `M` (current `client.py`) | solve, real `M` | solve, dense random `M` |
|---:|---:|---:|---:|---:|
| 20 | 231 | 2.38 s | 0.08 s | 0.10 s |
| 30 | 496 | 10.48 s | 0.04 s | 1.08 s |
| 48 | 1,225 | **64.89 s** | **0.16 s** | 17.60 s |

Two separate errors, both in the conservative direction:

**(a) The solve is not dense.** The interpolation matrix on the standard information set
is **~0.5% nonzero** at `m=48` and gets sparser as `λ` grows. Reason: `M[t, a] =` monomial
`x^a` evaluated at point `t`, which vanishes unless `supp(a) ⊆ supp(t)`; with `d = 2`,
both supports have size `≤ 2`, so each row has `O(1)` nonzeros and `M` has `O(λ)`
nonzeros total. Measured `0.16 s` vs `17.6 s` for a random matrix of the same size —
**110× off**, and the solution was verified exactly (`M @ c == b`).

Worth stating carefully in the paper: the `λ³` model was **not bad arithmetic**. Measured
on dense random `GF(q)` matrices, the solve scales at `1.07 / 7.48 / 72.81 / 653.10` s for
`λ = 500 / 1000 / 2000 / 4000` — ratios of 7.0×, 9.7×, 9.0× per doubling, i.e. cubic, as
predicted. The model was correct and applied to the wrong matrix. That framing is more
useful than "the old estimate was wrong."

**(b) The actual bottleneck is `Client._evaluate_monomials`**, at 65 s — 400× the solve
it was supposedly dominated by. It is a triple Python loop (`λ` points × `λ` exponents ×
`m` variables) over scalar `galois` objects. A vectorized rebuild (per-variable power
tables + gather, batched over all points) runs the same case in **0.18 s**, a **360×
speedup**, verified elementwise against the loop version. And `M` depends only on
`(I_{d,m}, exponents)` — it is *identical across all `p` gradient components and every GD
step*: build once, factor once, reuse forever.

**Consequences.** The wall is far further out than the previous plan assumed, the Week-1
work is what moves it, and `λ³` should not appear in the paper as the decode cost.
Re-derive the go/no-go in Week 2 on measured numbers.

The structure is sharper than "sparse", and it is now **verified**, not conjectured:

- `M[t, a] ≠ 0  ⟺  supp(a) ⊆ supp(t)` — exact iff, checked elementwise at
  `(m,d) = (20,2), (48,2), (12,3), (10,4)`.
- Hence nonzeros per row `= C(2d, d)` **exactly** (6 / 20 / 70 for `d` = 2 / 3 / 4),
  **independent of `m` and `λ`**. Total nnz `= O(λ)` at fixed degree.
- Support containment is a partial order, so ordering rows and columns by
  `(|supp|, supp)` makes `M` **block lower-triangular**, with blocks indexed by exact
  support. Measured violations: **0** at all three `(m,d)`. Max block size **2 / 3 / 6**
  for `d` = 2 / 3 / 4 — i.e. `O(1)`.

So decode is not a linear solve at all; it is forward substitution over `O(1)`-sized
blocks, `O(λ · C(2d,d))` field operations. A prototype block-triangular solver returns
**the identical coefficient vector** as `np.linalg.solve` (verified, plus `M@x == b`) and
is already 2.3× faster at `λ=1891` despite a naive `O(λ²)` inner matvec and a Python loop
over blocks. Exploiting the `C(2d,d)`-nonzero row structure removes the remaining `λ²`.

**Priority note:** decode is *already* 0.34 s at `λ=1891`, so this is not on the critical
path for the case study. Its value is that it removes `λ³` from the paper's complexity
story and replaces it with `O(λ)` — a clean, self-contained contribution. **Scope in W2,
implement in W7–8.** Do not let it displace Weeks 1 and 3–6.

---

## Two design decisions worth locking early

**1. Do the weight update in floating point, client-side.**
Rescaling after fixed-point multiplication is the hard part of finite-field arithmetic
(the same problem HE and MPC fight). Sidestep it entirely: the client already decodes the
gradient and is not resource-constrained for an `O(p)` update.

```
1. client quantizes w_t → w̃_t ∈ F_q^p, sends f_{w̃_t} to admin
2. workers evaluate on their shares, return λ values per gradient component
3. client interpolates, evaluates at k → gradient in F_q
4. client maps to signed integers, descales by 2^{2f} → real gradient
5. client updates w in float, re-quantizes next round
```

Scale never accumulates across iterations; the only error source is quantization of
`X`, `y`, `w`. The gradient is otherwise **exact** — no truncation error — provided no
wraparound. Note that `w_t` enters as *coefficients of `f`*, not as data: the workers'
shares are written once at storage time and never change across the whole training run.
That is the structurally elegant part of this protocol and it should be said plainly in
the paper.

**2. Build and factor the interpolation matrix exactly once per `(m, d)`.**
See the measurement above. `T · p` solves become one build + one factorization +
`T · p` back-substitutions. Cache it on the `Client`.

---

## A concern to state in the paper, not to resolve by changing scope

For linear regression specifically, per-step queries are **strictly worse** than computing
the sufficient statistics `XᵀX` and `Xᵀy` once: `T` leakage events instead of 1, three
quantized factors instead of two (so `h ≈ 1/3` instead of `≈ 1/2`, inflating `m`), and `T×`
the worker cost. A reviewer will raise this immediately.

The justification is that GD is the general protocol required for losses without closed
forms — which is where this work is going. **Implement the one-shot variant too** (cheap
once GD works, ~1 day) so the paper can *quantify* the price of generality rather than
hand-wave it. Measured: one-shot reaches `h ≈ 0.48, λ = 741` against GD's
`h ≈ 0.33, λ = 1081`. That comparison is a stronger result than either path alone.

---

## Week-by-week

### Weeks 1–2 · Make the loop fast and the math right (front-loaded risk)

Everything downstream depends on being able to run the scheme in seconds and on the
feasibility model being trustworthy. Right now neither holds.

**Week 1 — performance and correctness of the existing code**

- [ ] **Vectorize monomial-matrix construction** (`Client._evaluate_monomials` /
      `decode_result`, and the duplicate in `utils.get_information_superset`).
      Measured 64.9 s → 0.18 s at `m=48, d=2`. Verify elementwise against the current
      loop before deleting it.
- [ ] **Cache `M` and its factorization on the `Client`**, keyed by `(m, d, q)`.
      Reused across gradient components and GD steps.
- [ ] **Vectorize `Server.store_data`** — currently `λ` separate `(m,) @ (m,n)` matmuls in
      a Python loop; it is one `(λ×m) @ (m×n)` product.
- [ ] **Entropy scaling.** `compute_p_entropy` returns per-symbol `h`, not `H_p(X)`.
      `compute_max_subset_p_entropy` returns `≈h`, not the `r`-subset total. Fix both.
      Tell that it is currently wrong: the code can produce `max_R H_p(X_R) > H_p(X)`,
      impossible for the true quantities.
- [ ] **Kill the `C(n,r)` enumeration.** `compute_max_subset_p_entropy` is `C(n,r)`;
      at the case-study size `n=60, r=5` that is 5.4M subsets and simply will not run.
      Rényi entropy is additive over independent coordinates, so
      `max_R H_p(X_R) = ` sum of the `r` largest per-coordinate entropies — `O(n log n)`.
      **This is what removes the `n ≤ 12` cap, not a micro-optimization.**
      See Open Question 1 for the correlated-data caveat, which is a real one.
- [ ] Reconcile the two entropy estimators — `utils.py` uses the empirical distribution of
      one realization, `test_icc.py`'s i.i.d. shortcut uses the true sampling distribution.
      Pick one, document why.
- [ ] `ε_c` exponent: code has `q^(−max H)` per the Theorem 1 statement; the derivation at
      eq. (9)–(10) has `q^(−max H / p)`. Resolve with Raviv.
- [ ] Assert `d < m(q−1)`. (Trivially satisfied at `q ≈ 2³⁰`; it bites at the toy `q = 31`.)
- [ ] Make `--fast` actually skip tests; remove the unused `generate_vandermonde_G` import.
- [ ] `pip install scikit-learn` — not currently in the venv, and Week 5 needs it.
- [ ] Re-run the suite. Expect it to drop from ~81 min to minutes, and expect table values
      to move. Archive `full_test.txt` first for comparison.

**Week 2 — feasibility instrument and go/no-go**

- [ ] Build `feasibility.py`: given `(n, q, h, r, p, ε, d)` → `m`, `λ`, **measured**
      build/factor/solve time, verdict. No `λ³` proxy — call the real code.
- [ ] Empirically measure `h` on actually-quantized data and check the `h ≈ 1/3`
      derivation, including the claim that `h` is flat in `f`.
- [ ] Sweep `λ` to find where decode time actually becomes painful. Report the real wall.
- [ ] **Go/no-go on case-study size.** With Week 1 done this may well be larger than
      `n = 60`. Fix it now, not in Week 8.
- [ ] Start the paper's parameter-selection section. This analysis is paper material.

> **Checkpoint (end of W2):** case-study dimensions fixed; one full scheme run at those
> dimensions completes in seconds. Everything downstream depends on this.

### Weeks 3–4 · Quantization (moved up; this is now the critical path)

- **W3 — design**
  - [ ] Fixed-point map `x_real → round(x·2^f) mod q`, signed values in the upper half
        of the field (`[−(q−1)/2, (q−1)/2]`).
  - [ ] Dynamic-range budget: derive `q > 2·n_s·p·2^{f_X+f_X+f_w}` for a GD step; pick `q`
        prime. Keep the three scales **separate** in the derivation from the start —
        Weeks 7–8 depend on them being independent knobs.
  - [ ] Map the `f → h → m → λ` chain empirically, and report it with the correct sign.
  - [ ] Decide standardization: per-feature scaling before quantization improves
        conditioning *and* entropy. Confirm it doesn't leak (public, data-dependent
        transform — flag for next semester's analysis). See Open Question 3.
- **W4 — implement**
  - [ ] `quantize.py`: `to_field` / `from_field`, round-trip tests, overflow detection that
        **fails loudly** rather than silently wrapping. Independent scales per operand.
  - [ ] Empirically confirm no wraparound at the chosen `(q, f_X, f_w, n_s, p)`.
  - [ ] Measure quantization error vs `f` on the target dataset.
  - [ ] Measure real-data entropy: per-feature `h_i`, and how far from i.i.d. the
        coordinates actually are. This feeds Open Question 1.

### Weeks 5–6 · Linear regression under the scheme

- **W5** — One GD step, end to end. Encode `(X, y)`, query the gradient, decode, compare
  against the plaintext float gradient. Success = relative error at the quantization floor.
  Add the vector-valued query path (`p` gradient components share one `M`).
- **W6** — Full training loop. Convergence vs. an unencrypted float baseline on the same
  data; plot loss curves together. Add the one-shot sufficient-statistics variant.

> **Checkpoint (end of W6): the paper's core claim is demonstrated.** Everything after
> this is strengthening. If W1–W6 slip, W7–W8 are what get cut — not the write-up.

### Weeks 7–8 · Parameter efficiency (the slot straggler tolerance used to occupy)

Two ideas that attack `h` directly. Both are cheap to *evaluate* on paper before any
code is written — do that first, in a day, and only implement what survives.

- **W7 — asymmetric operand scales.** Data entropy depends only on `f_X`; the field size
  depends on `f_X + f_X + f_w`. So shrinking the *coefficient* precision `f_w` raises `h`
  at no cost to data fidelity. Derived at `n=60, f_X=12`: `f_w = 12 → h=0.33, λ=1081`;
  `f_w = 4 → h=0.41, λ=903`; `f_w = 0 → h=0.46, λ=780`. Ceiling is `h → 1/2`, i.e. the
  one-shot regime, reached without giving up GD.
  Risk: low-precision weights slow convergence. Mitigation: error feedback on the client
  (it holds full-precision `w` anyway). **Measure convergence vs. `f_w`** — that plot is
  a paper figure whichever way it comes out.

- **W8 — CRT decomposition (higher risk, higher payoff).** Run `L` independent scheme
  instances over small primes `q_1…q_L` with `∏ q_i` exceeding the dynamic range, and
  CRT-reconstruct. Polynomial evaluation commutes with reduction mod `q_i`, so
  correctness is immediate. The point is entropy: if `q_i` is *smaller* than the
  per-symbol entropy, `x mod q_i` is nearly uniform and `h_i ≈ 1`, so
  `m_i ≈ r + p + log_{q_i}(1/ε)` — **independent of `n`**.

  Derived at `n=60, f_X=f_w=8` (total `log₂ Q = 30.5`, `H_sym ≈ 10` bits):

  | prime bits | `L` | `h_i` | `m_i` | `λ_i` | total workers |
  |---:|---:|---:|---:|---:|---:|
  | single `2³⁰` | 1 | 0.33 | 45 | 1,081 | 1,081 |
  | 10 | 4 | ≈1.0 | 9 | 55 | 220 |
  | 8 | 4 | ≈1.0 | 10 | 66 | 264 |

  ~4× fewer workers, and — the real prize — **`m` stops scaling with `n`**, so the
  `n(1−h)` term that dominates Theorem 1 disappears. Decode gets cheaper too, but after
  the sparsity finding above that was never the binding constraint; **argue this on
  worker count and key size, not on decode time.**

  **This is not free and may not be sound.** The provider sees all `L` instances; the
  residues jointly determine `x mod ∏q_i`, which carries the full entropy. Per-instance
  leakage bounds do **not** obviously compose. The honest framing is a single scheme over
  the ring `Z_{∏q_i} ≅ ∏ F_{q_i}`, and the smoothing argument has to be redone there.
  **Ask Raviv before implementing** (Open Question 5). If the analysis doesn't hold up,
  this becomes a "promising but unproven" paragraph in future work — which is still a
  better outcome than a half-built super-set construction would have been.

> **Hard stop at end of W8.** Whatever is working, freeze it. W5–W6 is the result.

### Weeks 9–10 · Experiments

- **W9** — Run the sweeps, seeded and saved to disk:
  - convergence vs. `f_X` and vs. `f_w`
  - worker count / build / factor / solve time vs. `n`, vs. `λ`
  - measured `ε_c` across the sweep
  - GD vs. one-shot: leakage events, workers, wall-clock
  - CRT variant if W8 survived
- **W10** — Figures, tables, reproducibility pass (every number regenerable from a seeded
  script).

> **Results frozen end of W10.** No new experiments after this.

### Weeks 11–12 · Paper

- **W11** — Full draft. Suggested structure: Introduction / Background (ICC + Theorem 1) /
  Quantization for finite-field ML / Parameter selection and the entropy–precision chain /
  Training protocol / Evaluation / Limitations and future work (→ straggler tolerance via
  information super-sets, and next semester's repeated-query leakage analysis).
- **W12** — Revise with Raviv, finalize figures, submit.

**Write incrementally from Week 2.** Each week, append results to the draft. A paper
assembled from twelve weeks of notes is far better than one written in two.

---

## Recommended dataset

**`sklearn.datasets.load_diabetes`** — real, standard regression benchmark, features already
standardized, 442×10. Subsample to **15 samples × 3 features** to hit `n = 60`.
(`scikit-learn` is not yet installed in the venv — Week 1.)

Pair it with a **synthetic i.i.d. control** (known ground-truth `w`, controlled
conditioning, coordinates independent *by construction* so the entropy math is exact) so
quantization error can be isolated from real-data messiness — and so there is a setting
where Open Question 1 does not bite. Report both.

`load_diabetes` is preferable to California housing here: smaller dynamic range after
standardization, better conditioned at low feature counts, and no need to fetch anything.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Correlated real features break the additive-entropy shortcut | **high** | Open Q1 to Raviv in W1; synthetic i.i.d. control always available; consider whitening |
| Writing compressed into W11–12 | **high** | Write from W2 onward |
| Entropy fix invalidates last semester's tables | medium | Expected — re-run and report the correction as a finding |
| CRT leakage analysis doesn't compose | medium | W8 is explicitly optional; degrade to future-work paragraph |
| GD doesn't converge under quantization | medium | W5 single-step check catches it early; fall back to more bits or one-shot |
| Low-precision `f_w` hurts convergence | medium | Error feedback client-side; report the tradeoff curve either way |
| Case study still too slow after W1 | **low** (was medium) | W1 measurements suggest large headroom; W2 go/no-go confirms |

---

## Open questions for Raviv

1. **(New, and the most urgent.)** Theorem 1 needs `H_p(X)` for the whole length-`n`
   vector and `max_R H_p(X_R)`. For independent coordinates these are `Σ h_i` and the sum
   of the `r` largest `h_i`. Real regression features are **correlated**, so `Σ h_i`
   *over*estimates `H_p(X)`, which *under*estimates `m` — unsafe. What is the intended
   practice: a valid lower bound on `H_p(X)`, a decorrelating pre-transform, or something
   in the paper I'm missing?
2. Which entropy definition is intended in `compute_p_entropy` — the empirical distribution
   of the realization, or the true source distribution?
3. `ε_c` exponent: Theorem 1 statement vs. eq. (9)–(10) derivation — which is authoritative?
4. Is standardizing features before quantization acceptable, given it is a public,
   data-dependent transform?
5. **(New.)** Does running `L` independent instances over CRT-coprime primes compose?
   Framed as one scheme over `Z_{∏q_i} ≅ ∏F_{q_i}`, does the smoothing argument of
   Theorem 1 go through, and what is the joint `ε_c`?
6. For repeated GD queries against one key: does the Theorem 1 bound compose over `T`
   observations, or is a fresh key needed per round? (Next semester's core question, but the
   answer shapes this semester's protocol design.)

---

## Deferred to future work (explicitly not this semester)

- **Straggler tolerance / information super-sets.** `utils.get_information_superset` stays
  as the unrunnable prototype it is; do not build against it. The 2024 Deng–Ramkumar–Raviv
  constructions are no longer a Week-1 blocker. Note the limitation in the paper.
- Repeated-query leakage analysis (next semester's core question).
- Logistic regression / losses without closed forms.
