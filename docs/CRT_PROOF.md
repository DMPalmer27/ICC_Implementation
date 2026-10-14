# CRT proof plan: smoothing over `Z_Q`, and leakage from multiple encodings

**Status: a plan with partial proofs, not a proof.** Deferred by decision on 2026-10-13: until
this is done, the implementation stays with a single prime and treats `q > NATIVE_Q_MAX` as
"allowed, but expect much longer running times" (see `docs/Q_SWEEP.md`). Empirical
motivation: `docs/Q_SWEEP.md` §6 and `tools/crt_leakage_demo.py`.

Each result below is marked:

- **[proved here]** — the argument in this file is complete.
- **[sketch]** — the argument is given but has steps to check or a normalisation to match.
- **[conjecture]** — the statement is what we expect; the proof is the open work.

---

## 0. Why this matters, and what has to be proved

Quantising reals forces a large field (`docs/QUANTIZATION.md` §7). A large prime is correct but
slow above the `galois` native limit (`log2 q ≤ 31.5`, `docs/Q_SWEEP.md`). CRT runs the scheme
over `L` small primes `q₁ … q_L` with product `Q`, at native speed, and recombines. Two things
must be proved before CRT can be used:

1. **Smoothing over the product.** The privacy theorem (ICC Theorem 1) holds when arithmetic is
   mod `Q = q₁⋯q_L` instead of mod a prime, with `q` replaced by `Q`.
2. **Leakage from multiple encodings.** The provider receives `L` encodings of the same data.
   We must bound what that reveals — positively for the correct ("joint") key sizing, and
   negatively (a lower bound on leakage) for the tempting wrong ("per-instance") sizing. The
   same analysis covers **re-uploading** the same data under a fresh key.

The plan below reduces (1) to a single lemma about random linear codes over `Z_Q`, and shows
(2) follows from the same computation.

---

## 1. Setup and notation

| symbol | meaning |
|---|---|
| `q₁, …, q_L` | distinct primes |
| `Q = q₁⋯q_L` | the modulus; `Z_Q` is the ring of integers mod `Q` |
| `φ : Z_Q → F_{q₁} × ⋯ × F_{q_L}` | the CRT isomorphism, `a ↦ (a mod q₁, …, a mod q_L)`; applied entrywise to vectors and matrices |
| `X ∈ Z^n` | the quantised data, integer codes with `‖X‖_∞ ≤ A` (`A = X̂max`), distribution `p_X`, support `S` |
| `K ~ Unif(Z_Q^m)`, `G ~ Unif(Z_Q^{m×n})` | key and code, independent of each other and of `X` |
| `X̃ = X + KG mod Q` | the encoding |
| `C = {kG : k ∈ Z_Q^m} ⊆ Z_Q^n` | the code: a `Z_Q`-submodule ("linear code over a ring") |
| `Xᵢ, Kᵢ, Gᵢ, Cᵢ, X̃ᵢ` | the `i`-th CRT component, over `F_{qᵢ}` |
| `H_p(·)` | Rényi entropy of order `p ≥ 2`, in `log_Q` units unless stated (so `H_p(X) ≤ n`) |
| `V_p(·,·)` | the `p`-variational distance of ICC Definition 2 / Pathegama–Barg; target is `Unif(Z_Q^n)` |

**Separation assumption (S).** For all `x ≠ x′` in `S` and every `i`: `x ≢ x′ (mod qᵢ)`.
For quantised data it holds automatically when every `qᵢ > 2A`, since then two distinct code
vectors differ by less than `qᵢ` in some coordinate.

**Rank-preservation assumption (S_k).** For every set of at most `k` difference vectors
`x − x′` (`x, x′ ∈ S`), its rank mod `qᵢ` equals its rank over `Q` for every `i`. (S₁) is (S).
By Hadamard's inequality every `j × j` minor of such vectors is at most `(2A)^j·j^{j/2}` in
absolute value, so (S_k) holds when every `qᵢ > (2A)^k·k^{k/2}`. Section 5.3 explains why
this may be needed for general `p`.

---

## 2. The protocol as implemented, and the ring view

**Implemented.** For each `i`: draw `Kᵢ ~ Unif(F_{qᵢ}^{mᵢ})` and `Gᵢ ~ Unif(F_{qᵢ}^{mᵢ×n})`
independently; upload `X̃ᵢ = (X mod qᵢ) + KᵢGᵢ`. The admin shards each with the same
information set `I_{d,m}` (valid since `d < qᵢ`); workers evaluate `f` on every share; the client
decodes `f(X) mod qᵢ` per instance and recombines by CRT. Correctness is unconditional
(reduction mod `qᵢ` and `φ` are ring homomorphisms; `tools/q_sweep.py` checks it against an
integer oracle).

**Two sizings.**

- *Joint:* `mᵢ = m` for all `i`, with `m` from Theorem 1 in `log_Q` units.
- *Per-instance:* `mᵢ` from Theorem 1 in `log_{qᵢ}` units, as if instance `i` were alone. Smaller
  keys (the 34.45-bit example: `m = 58` vs `80`), and unsafe (§7).

### Proposition P1 (the implementation is the ring scheme) — [proved here]

*With joint sizing, `(X̃₁, G₁, …, X̃_L, G_L)` has the same joint distribution with `X` as
`(φ(X̃), φ(G))`, where `X̃ = X + KG` over `Z_Q` with `K ~ Unif(Z_Q^m)`, `G ~ Unif(Z_Q^{m×n})`.*

*Proof.* `φ` is a bijection `Z_Q^m → ∏ F_{qᵢ}^m`, so `K` uniform ⟺ `φ(K) = (K₁, …, K_L)` uniform
on the product ⟺ the `Kᵢ` are independent and each uniform. The same holds for `G`. `φ` is a ring
homomorphism, so `φ(X + KG) = (X mod qᵢ + KᵢGᵢ)ᵢ`. Since `φ` is a bijection, the provider's view
in the implementation is an invertible function of `(X̃, G)` and conversely, so every
information quantity (`I(X_R; ·)`, divergences) is the same for both. ∎

*Consequence.* Every privacy statement about the implemented `L`-instance protocol with joint
sizing is a statement about one encoding over `Z_Q`. Per-instance sizing is also covered:
different `mᵢ` per component is a `Z_Q`-code whose components have different dimensions.

---

## 3. Results to prove

| id | statement | status |
|---|---|---|
| P1 | implementation ≡ ring scheme | [proved here] §2 |
| L0 | preliminaries: `KG | G` uniform on `C`; `|C| = Q^m` w.h.p.; inclusion probabilities factor over primes | [proved here] §4 |
| L2 | smoothing over `Z_Q` for `p = 2`, under (S) | [sketch] §5.2 |
| LZ | smoothing over `Z_Q` for all integer `p ≥ 2` (Pathegama–Barg Thm 3.1 with `q → Q`), under (S_p) or weaker | [conjecture] §5.3 |
| T1′ | ICC Theorem 1 over `Z_Q` | [sketch, given LZ] §6 |
| U | unified condition: smoothing needs `Σᵢ (n − mᵢ)·log2 qᵢ ≤ H(X)[bits] − log2(1/ε)` (up to the `p`-dependent terms) | [sketch] §7.1 |
| N1 | per-instance sizing reveals `X`: lower bound on leakage | [sketch, rigorous for uniform `X`] §7.2 |
| N2 | re-uploading the same data | [sketch] §7.3 |
| N3 | necessity of separation | [sketch] §7.4 |

---

## 4. Preliminaries (L0) — [proved here]

**(a) `KG` given `G` is uniform on `C`.** `k ↦ kG` is a group homomorphism `Z_Q^m → C`, onto by
definition of `C`. A uniform input to a surjective homomorphism of finite groups gives a uniform
output (every fibre is a coset of the kernel, all of equal size). Hence
`p_{X̃|G} = p_X ∗ U_C` (convolution with the uniform distribution on `C`).

**(b) Product structure.** `φ(C) = C₁ × ⋯ × C_L` with `Cᵢ = rowspace(Gᵢ) ⊆ F_{qᵢ}^n`, because
`φ(kG) = (kᵢGᵢ)ᵢ` and the `kᵢ` range independently. So `|C| = ∏ᵢ |Cᵢ| = ∏ᵢ qᵢ^{rank Gᵢ}`.

**(c) Full rank w.h.p.** For `m ≤ n`, a uniform `m × n` matrix over `F_q` is rank-deficient only
if some row lies in the span of the rows before it. Row `j + 1` is uniform on `F_q^n`, and the span
of `j` rows has at most `q^j` elements, so by a union bound

```
P[rank < m] ≤ Σ_{j=0}^{m−1} q^{j−n} = (q^m − 1)/((q − 1)·q^n) ≤ q^{m−n}/(q − 1).
```

A union bound over `i` gives
`P[some Gᵢ rank-deficient] ≤ Σᵢ qᵢ^{m−n}/(qᵢ − 1)`, negligible because `m < n` in every regime
we use. On the complement, `|C| = Q^m` and each `Cᵢ` is a **uniformly random** `m`-dimensional
subspace of `F_{qᵢ}^n` (a uniform full-rank matrix has a uniform row space).

**(d) Inclusion probabilities factor.** For `v ∈ Z_Q^n`, `v ∈ C ⟺ vᵢ ∈ Cᵢ` for all `i`, and the
`Cᵢ` are independent. For a uniform `m`-dimensional subspace `D ⊆ F_q^n` and `u ≠ 0`,
`P[u ∈ D] = (q^m − 1)/(q^n − 1) ≤ q^{m−n}`. Hence, on the full-rank event,

```
P[v ∈ C] = ∏_{i : vᵢ ≠ 0} (qᵢ^m − 1)/(qᵢ^n − 1)  ≤  ∏_{i : vᵢ ≠ 0} qᵢ^{m−n}.
```

Under (S), every nonzero difference `v = x − x′` has `vᵢ ≠ 0` for all `i`, so
`P[v ∈ C] ≤ Q^{m−n}` — **exactly the field formula with `q → Q`.** This one line is why the
whole programme is plausible.

---

## 5. The smoothing lemma over `Z_Q`

### 5.1 Target

**Lemma LZ [conjecture].** Under (S_p) (or a weaker condition found in §5.3), if

```
m ≥ n + p + log_Q(1/ε) − H_p(X)          (H_p in log_Q units)
```

then `E_G[ V_p(p_X̃, Unif(Z_Q^n)) ] ≤ 2^{(p−1)/p}·((1+ε)^p − 1)^{1/p}` — the bound of
Pathegama–Barg Thm 3.1 / Lem 3.6 as used in ICC eq. (2)–(3), with `q → Q`.

### 5.2 The case `p = 2` via collision probability — [sketch]

Fix `G` on the full-rank event. With `X′, K′` independent copies,

```
CP(X̃ | G) = P[X + KG = X′ + K′G]
          = Σ_{x,x′} p(x)p(x′)·P[x − x′ = (K′ − K)G]
          = (1/|C|)·Σ_{x,x′} p(x)p(x′)·1[x − x′ ∈ C]           (L0a: (K′−K)G ~ U_C)
          = Q^{−m}·( CP(X) + Σ_{x ≠ x′} p(x)p(x′)·1[x − x′ ∈ C] ).
```

Average over `G` with L0d under (S):

```
E_G[CP(X̃)] ≤ Q^{−m}·CP(X) + Q^{−m}·Q^{m−n}·(1 − CP(X))  ≤  Q^{−m}·CP(X) + Q^{−n}.
```

The uniform distribution on `Z_Q^n` has collision probability `Q^{−n}`, and
`‖p_X̃ − U‖₂² = CP(X̃) − Q^{−n}`. With `CP(X) = Q^{−H₂(X)}` (log_Q units):

```
Q^n · E_G‖p_X̃ − U‖₂²  ≤  Q^{n − m − H₂(X)},
```

which is `≤ ε` iff `m ≥ n − H₂(X) + log_Q(1/ε)` — the `p = 2` form of Theorem 1's condition
with `q → Q` (the `+p` slack and the exact constant come from Pathegama–Barg's normalisation of
`V_p`, which must be matched: **to do** — confirm their definition of `V_p` and redo this
paragraph in it).

*Sanity check of the method:* setting `L = 1` gives the field computation, so this is not new
for fields; the new content is entirely L0d.

### 5.3 General `p ≥ 2` — [conjecture]; the real work

Follow Pathegama–Barg (arXiv:2405.04406, Thm 3.1 and Lem 3.6) line by line, and at every use of
field structure substitute its CRT counterpart:

| field-case tool | `Z_Q` counterpart |
|---|---|
| `dim C = m`, `|C| = q^m` | `|C| = Q^m` on the full-rank event (L0c) |
| `P[u ∈ C] ≈ q^{m−n}` | L0d, factors over primes |
| `P[u₁, …, u_k ∈ C] ≈ q^{(m−n)·rank(u₁…u_k)}` | `∏ᵢ qᵢ^{(m−n)·rankᵢ}`, `rankᵢ` = rank mod `qᵢ` |
| dual code `C^⊥`, MacWilliams identities | exist for codes over finite Frobenius rings (`Z_Q` is one; Wood's duality theory) and factor over `φ` |
| additive characters / Fourier on `F_q^n` | characters of the finite abelian group `Z_Q^n` = products of characters of the `F_{qᵢ}^n` |

**The expected snag.** A `p`-th moment typically needs joint inclusion of up to `p − 1`
difference vectors, whose probability depends on their rank. Over `Z_Q` the exponent is
`Σᵢ (m − n)·rankᵢ·log qᵢ`. If the `rankᵢ` agree for all `i` (assumption S_{p−1}), the product is
`Q^{(m−n)·rank}` and the field proof goes through verbatim. Pairwise separation (S) does not
guarantee this: three small integer vectors can be dependent mod one prime and independent
over `Q`. Options, in order of preference:

1. Show the moment bound only needs **upper** bounds that hold regardless of `rankᵢ` (a lower
   `rankᵢ` raises a probability, so this may fail — check which direction each term needs).
2. Assume (S_{p−1}): primes larger than `(2A)^{p−1}(p−1)^{(p−1)/2}`. For `p = 2` this is just
   `qᵢ > 2A`. For larger `p` it costs bigger primes (fewer, larger CRT components).
3. Bound the contribution of rank-deficient tuples separately; they are rare when the primes
   are large relative to `A`, and may be absorbed into `ε`.

**Deliverable of this step:** LZ with an explicit condition, or a counterexample showing a
condition is necessary (the numerical checker of §8 is designed to find one).

---

## 6. From LZ to ICC Theorem 1 over `Z_Q` (T1′) — [sketch, given LZ]

Walk the ICC proof (`Written_Resources/extracted/ICC_Polynomials_Non-Uniform.txt`, §V-A, §V-B,
Appendix B) with `q → Q`:

| ICC step | change |
|---|---|
| eq. (1): triangle inequality for `V` | none — holds on any finite set |
| eq. (2)–(4): smoothing of `X | X_R = z` | apply LZ to each conditional. Its support is a subset of `S`, so (S)/(S_k) are inherited |
| eq. (5)(a): `H_p(X|X_R=z) ≥ H_p(X) − max_R H_p(X_R)` | none — Appendix B is Rényi-entropy algebra over a finite alphabet; only the log base changes |
| eq. (5)(b): `ε′ = ε·q^{−max_R H_p(X_R)}` | `ε′ = ε·Q^{−max_R H_p(X_R)}` |
| eq. (6)–(8): `V ≤ V_p`, smoothing of `X̃` itself | none |
| §V-B: `V_p`–`D_p` equivalence ([22, Prop. 2.1]) and `I(X̃; X_R) ≤ D_p(·)` | none — generic over finite sets; `log_q → log_Q` in eq. (11) |
| the `1 − 1/a` probability over `G` (Markov on `E_G`) | add the full-rank failure probability of L0c |

**Dependency:** CLAUDE.md open issue 1 (the exponent of `q` inside `δ`: Theorem 1 statement vs
eq. (9)–(10)) carries over unchanged. Resolve it for fields first.

**Statement of T1′.** Under (S_p), for `m ≥ n + p + log_Q(1/ε) − H_p(X) + max_R H_p(X_R)`
(log_Q units), with probability at least `1 − 1/a − Σᵢ qᵢ^{m−n}/(qᵢ−1)` over `G`, every `r`-subset
satisfies `I(X̃; X_R) ≤ ε_c` with `ε_c` as in Theorem 1 and `q → Q`. By P1 this is a statement
about the implemented `L`-instance protocol.

---

## 7. Leakage from multiple encodings

### 7.1 The unified condition (U) — [sketch]

Redo §5.2 without assuming equal `mᵢ`. L0d gives, for nonzero `v` under (S),
`P[v ∈ C] ≤ ∏ᵢ qᵢ^{mᵢ−n}` and `|C| = ∏ᵢ qᵢ^{mᵢ}`, so

```
(∏ᵢ qᵢ^n) · E‖p_X̃ − U‖₂²  ≤  ∏ᵢ qᵢ^{n − mᵢ} · 2^{−H₂(X)[bits]},
```

and smoothing to `ε` needs

```
Σᵢ (n − mᵢ)·log2 qᵢ  ≤  H₂(X)[bits] − log2(1/ε).                       (U)
```

Read `(n − mᵢ)·log2 qᵢ` as **the bits of linear information about `X` that instance `i` hands
the provider** (its parity checks). (U) says the total over all instances must stay below the
data's entropy minus a margin. **Joint sizing** makes the left side `(n − m)·log2 Q`, the same as
one field of size `Q`. **Per-instance sizing** makes each term nearly `H₂(X) − log2(1/ε)` on its
own, so the sum is about `L` times too large.

### 7.2 Negative result N1: per-instance sizing reveals `X` — [sketch; rigorous for uniform `X`]

Let `X` be uniform on `S`, `|S| = 2^H`. The provider's candidate set is
`{x′ ∈ S : x′ − X ∈ C}` (the true `X` always belongs). By L0d under (S),

```
E[#(candidates other than X)] = (|S| − 1)·∏ᵢ (qᵢ^{mᵢ} − 1)/(qᵢ^n − 1)  ≤  2^H·∏ᵢ qᵢ^{mᵢ − n}.
```

By Markov, `P[X is the unique candidate] ≥ 1 − 2^H·∏ᵢ qᵢ^{mᵢ−n}`. When `X` is unique the provider
knows `X` exactly, so `I(X_R; view) = H(X_R)` for every `R` — total leakage. Per-instance sizing
sets each `qᵢ^{n−mᵢ} ≈ 2^H·(margin)⁻¹`, so for `L ≥ 2` the bound is `≈ 2^{−(L−1)H}·∏ margins`, and
`X` is recovered with probability close to 1 once `H` exceeds the margins.

**Check against the experiment** (`tools/crt_leakage_demo.py`, `|S| = 3^8 = 6561`):

| case | predicted other candidates `≈ 6561·∏ qᵢ^{mᵢ−n}` | predicted `P[unique] ≥` | measured |
|---|---:|---:|---|
| `q = 11, m = 6` (`n − m = 2`) | 6561/121 ≈ 54 | — | median 54 left |
| `q = 13, m = 6` | 6561/169 ≈ 39 | — | median 40 left |
| both, per-instance (6, 6) | 6561/(121·169) ≈ 0.32 | 0.68 | recovered 24/30 = 0.80 |
| both, joint (7, 7) | 6561/(11·13) ≈ 46 | — | median 44 left |

The heuristic matches the brute force, which is good evidence that (U) is the right condition.
To make N1 a theorem for non-uniform `X`, replace `|S|` by the min-entropy (or `H₂`) and the
uniqueness event by a guessing-probability bound.

### 7.3 Re-uploads (N2) — [sketch]

Two uploads of the same `X` over the **same** prime `q` with independent `(K, G)`, `(K′, G′)` are
one encoding of `(X, X) ∈ F_q^{2n}` by a block-diagonal code. The candidate count above gives the
condition `(n − m₁) + (n − m₂) ≤ H_q(X) − log_q(1/ε)`. If the first upload was sized tight,
`n − m₁ ≈ H_q(X) − log_q(1/ε) − (margin)`, then the second needs `m₂ ≥ n − (margin)`: **essentially a
one-time pad.** Practical rule to state in the protocol: *never re-encode the same data unless
the new key length accounts for every earlier encoding the provider may have kept.* (Note:
the `quantize.QueryBudget` docstring currently says "re-upload with a larger budget" — that
advice must carry this caveat.)

### 7.4 Necessity of separation (N3) — [sketch]

Without (S), L0d keeps factors equal to 1 for primes where `x ≡ x′`. Grouping pairs by the set
`T` of primes on which they agree,

```
E_G[CP(X̃)] ≈ Q^{−m}·Σ_{T ⊆ [L]} CP_T(X)·∏_{i ∉ T} qᵢ^{m−n},     CP_T(X) = P[X ≡ X′ (mod ∏_{i∈T} qᵢ)].
```

Terms with `∅ ≠ T ≠ [L]` are collisions of `X` reduced mod a partial product. When the primes are
smaller than the data range (the abandoned "regime (b)"), `X mod qᵢ` has low entropy and these
terms dominate. That is the formal reason small primes fail, and it gives a lower bound to state
alongside N1.

---

## 8. Numerical checks to run alongside the proof

All exact (enumerate `S`, average over many or all `G`), tiny parameters (`n ≤ 8`,
`Q = 11·13 = 143` vs the prime `139`):

1. **L2 check:** exact `V₂(p_X̃, U)` over `Z_Q` vs over `F_139`, under (S): should agree up to
   the `|Q − 139|` difference.
2. **Rank snag (§5.3):** build `S` whose difference vectors are dependent mod `q₁` but not over
   `Q`, compute exact `V₃`, and compare with the (S₂)-satisfying control. A gap shows the
   stronger condition is needed; no gap suggests option 1 of §5.3 works.
3. **(U):** sweep `(m₁, m₂)` around the boundary `Σ(n−mᵢ)log qᵢ = H − log(1/ε)` and check that the
   measured distance crosses `ε` there.
4. **N3:** primes below `2A`; verify the partial-collision terms appear in the exact `V₂`.

These do not prove anything, but they catch a wrong lemma statement before whiteboard time is
spent on it.

---

## 9. Reading list

- **M. Pathegama and A. Barg, "Rényi divergence guarantees for hashing with linear codes,"
  arXiv:2405.04406, 2024** — Thm 3.1, Lem 3.6, Prop 2.1. *Not yet in `Written_Resources/`; get it
  first, since §5.3 is a line-by-line port of their proof.*
- The ICC paper, §V-A, §V-B, Appendix B (the steps in §6).
- Smoothing references [35] (Micciancio–Regev) and [36] (Debris-Alazard–Ducas–Resch–Tillich)
  of the ICC paper, for the general shape of smoothing arguments.
- Codes over finite rings: J. A. Wood's duality / MacWilliams results for finite Frobenius
  rings (check exact citation), needed only if Pathegama–Barg use the dual code.

---

## 10. Order of work

1. Get Pathegama–Barg; identify every use of field structure (fills the §5.3 table).
2. Resolve CLAUDE.md open issue 1 for fields (it propagates to T1′).
3. Match the `V_p` normalisation and finish L2 (§5.2) rigorously.
4. Run checks 1–2 of §8; decide the condition for LZ.
5. Prove LZ; then T1′ by the substitution table (§6).
6. Prove N1 (start with uniform `X`), N2, N3; run checks 3–4.
7. Only then: implement CRT in `icc/` (the prototype in `tools/q_sweep.py` already decodes
   exactly) with joint sizing enforced in code.
