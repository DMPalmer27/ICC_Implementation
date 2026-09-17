# The two predecessor papers — reading notes

Notes on the two papers ICC builds on, extracted to
`Written_Resources/extracted/` by `tools/extract_pdf_text.py`:

- **[2022]** N. Raviv and Z. Goldfeld, *Perfect Subset Privacy for Data Sharing and
  Learning*, ISIT 2022, pp. 1850–1855.
- **[2024]** Z. Deng, V. Ramkumar and N. Raviv, *Perfect Subset Privacy in Polynomial
  Computation*, ISIT 2024, pp. 933–938 (full version: arXiv:2405.05567).
- **[ICC]** S. Tarnopolsky, Z. Deng, V. Ramkumar, N. Raviv and A. Cohen, *Individual
  Confidential Computing of Polynomials over Non-Uniform Information*, arXiv:2501.15645.

Purpose of this document: most of what the code does comes from [2024], not from [ICC],
and several things in `CLAUDE.md` and `docs/PLAN.md` were attributed to the wrong paper or
stated without their side conditions. Section 5 lists what this reading actually changed.

---

## 1. The lineage in one table

| | [2022] | [2024] | [ICC] |
|---|---|---|---|
| Data assumption | i.i.d., discrete, **`p`-dyadic**, `P_X` **known** | `X ~ Unif(F_q^n)` | **unknown, non-uniform** |
| How uniformity is obtained | explicit **uniformization mechanism** | assumed | **smoothing** by a random linear code |
| Code | Vandermonde / Reed–Solomon (MDS) | any `[n,m]_q` with `dmin(C⊥) ≥ r+1` | **uniformly random** `G` |
| Key size | `m = r` (Singleton met with equality) | `m ≥ r` | `m ≥ n + p + log_q(1/ε) − H_p(X) + max_R H_p(X_R)` |
| Leakage | `I(X_R; X̃) = 0` exactly | `= 0` exactly | `≤ ε_c`, negligible |
| Computation | none — data *sharing* + learning on `X̃` | **polynomial `f`, degree `≤ d`, via workers** | same as [2024] |
| Stragglers | — | **information super-sets**, `S` of them | inherits, not implemented here |

The through-line: each paper relaxes one assumption and pays for it. [2024] drops the field
size restriction `q ≥ n` and adds computation; [ICC] drops uniformity and pays by giving up
`m = r` and exact zero leakage.

**The scheme in `icc/` is [2024]'s computation scheme with [ICC]'s key-size rule.**
`get_information_set`, `λ`, the `x̃ − tG` sharding, `g(t) ≜ f(x̃ − tG)`, the interpolate-then-
evaluate-at-`k` decode — all of that is [2024] §IV. Only `compute_required_m`,
`compute_leakage_bound` and the random `G` come from [ICC].

---

## 2. [2024] — what the implementation actually implements

**The computation model** (§II-A). Storage phase: user sends `x̃`, admin shards to `N`
workers. Computation phase: user sends `f`, workers evaluate, admin aggregates into `A`,
user decodes with `k`. Three metrics: number of workers `N`, download cost `D`, side
information `m`.

> **"The polynomial `f` is not known during the storage phase."**

That sentence is the model, stated outright, and it is the citation for
`docs/QUANTIZATION.md` §6: the field size cannot be chosen against a query, because no query
exists yet. `quantize.QueryBudget` is the consequence.

**`r`-subset privacy** (Def. 1, from [2022, Def. 1]): `I(X_R; X̃) = 0` for all
`R ∈ C(n,r)`. Three distinctions worth keeping straight for the write-up, all made
explicitly in §II-B:

- it is **not** collusion resistance, which is `I(X; X̃_R) = 0` — the subset is on the
  *other* side;
- it is **not** individual security, `I(X̃_R; X_Q) = 0`;
- neither it nor differential privacy implies the other.

Both alternatives are irrelevant here for the same reason: the provider sees `X̃` *in its
entirety*. Also useful as a global statement: `r`-subset privacy implies
`I(X; X̃) ≤ (n−r)·log₂ q` bits.

**The encoding** (§III): `x̃ = x + kG` with `G` a generator of any `[n,m]_q` code satisfying
`dmin(C⊥) ≥ r+1`, which is exactly the condition that `G_R` has full rank for every
`r`-subset. Theorem 1 gives `r`-subset privacy for any field size.

**Remark 2 is where `m ≥ r` comes from**, and the derivation matters — see §5 below.

**Upload cost can be `n − m`, not `n`** (§III). Since `x̃ ∈ x + C`, the user can send a
*syndrome* identifying the coset rather than the whole vector. **Not implemented**:
`Client.encode_data` returns the full `x̃`. This is a real, self-contained efficiency result
sitting unused. Footnote 2 of the same page notes that the quantized-computation-over-the-
reals literature uses the same trick — i.e. it is adjacent to this project's Week 3 work.

**Information sets** (Def. 2, 3, Lemma 2). `I_{d,m} ≜ {(α_{i_1},…,α_{i_m}) : Σ i_k ≤ d,
0 ≤ i_k ≤ q−1}` is an information set for `RM_q(d,m)`. Download drops from `q^m` (the
trivial "try every key" scheme) to `λ(q,d,m) = dim RM_q(d,m)`.

**The degree condition is stated verbatim**: *"The scheme assumes that the total degree of
`f` is less than `m(q−1)`; that is, `d < m(q−1)`. Otherwise, `RM_q(d,m)` is not
well-defined."* The assertion added to `compute_required_m` in Week 1 is therefore exactly
right and now has a citation. Note the model separately requires `d < n(q−1)` for `f` itself
— different variable count, different condition; both appear.

**`λ` has side conditions the repo was quoting without** (Corollary 1) — see §5.

**Information super-sets** (§V, Def. 4): a multiset `T` is an `S`-information super-set if
every `(|T|−S)`-subset of `T` contains an information set. Bounds: `L ≥ λ + S` (Lemma 3),
`L ≤ q^m − dmin + S + 1` (Lemma 4), and `L ≤ (S+1)λ` by repeating an information set. The
`q = 2` constructions are the paper's main technical contribution and live in the full
version. Also: an LCC-based alternative with `N = (λ−1)d + S + 1` workers, requiring
`q ≥ N` — attractive only when `d < S`.

**Multiple datapoints** (Remark 1): the paper assumes a single `x` and says the extension to
a dataset is "straightforward repetition". The ML case study does something different — it
*flattens* `n_s` samples and their labels into one length-`n = n_s(P+1)` vector under one
key. That is cheaper (one key, one `G`, one upload) but the privacy semantics differ:
repetition would protect any `r` coordinates *within each sample* under independent keys,
whereas flattening protects any `r` of the `n` scalars, so a protected subset may straddle
samples and mix features with labels. Deliberate, and worth one sentence in the paper.

---

## 3. [2022] — the definitions, and the road ICC did not take

**Origin of `r`-subset privacy** (Def. 1), generalising perfect *sample* privacy. The
informativeness target is to keep `I(X^n; X̃^n)` high, and `≤ (n−r)H(X)` is the ceiling for
any such mechanism.

**The uniformization mechanism** (§III-A) is the part most relevant to us, and it is the
option [ICC] deliberately replaced. For a `p`-dyadic source, partition `{0,1}^{log p}` into
blocks sized by the probabilities and map each value to a **uniformly random element of its
block**. Proposition 1: the output is *exactly* `Unif({0,1}^d)`. Two properties worth
noting:

- The randomness is **secret and client-side**, so it genuinely adds entropy. This is the
  legitimate version of the "add a dither" idea that `docs/QUANTIZATION.md` §4G records as a
  trap — the trap was making the dither *known to the provider*.
- It is not free, and the price is quantified: Theorem 1 gives
  `I(X^n; X̃^n) ≥ (n−r)H(X) − r·H(X̂|X)`, where `H(X̂|X)` is exactly the injected randomness.
  Theorem 3 shows this penalty is essentially unavoidable within a broad framework.

**It requires knowing `P_X`.** That is the whole reason [ICC] exists: under an unknown
distribution you cannot build the partition, so [ICC] replaces exact uniformization with
smoothing, and pays with `ε_c > 0` and `m > r` instead of `r·H(X̂|X)`. This reframes open
question 8 — see §5.

**Signal preservation** (Def. 3): `f` is signal preserving if for every `(w₁,b₁)` there is
`(w₂,b₂)` with `w₁x − b₁ = w₂f(x) − b₂` for all `x`. The point is *instance encoding*: if
the privatization is signal preserving, an ordinary learning algorithm runs **unaltered on
the privatized data**, since models of the form `h(x) = h̄(Wxᵀ − vᵀ)` — linear and logistic
regression, feedforward nets — only see linear functionals. Any invertible linear map
qualifies, and the property composes. §IV builds a signal-preserving scheme via random
Hadamard encoding.

**This is a genuinely different paradigm from ICC and belongs in the paper's related work.**
Instance encoding trains *on* `X̃` and preserves only linear signals; ICC recovers `f(x)`
*exactly* through a query protocol and supports any polynomial of degree `≤ d`, at the cost
of `λ` workers per query and a round of interaction. Worth noting that fixed-point
quantisation is **not** signal preserving — clipping and rounding are not linear — which
costs us nothing under ICC but would rule the Week 3 pipeline out of the [2022] route.

---

## 4. Things the papers say that we should not re-derive

- `r`-subset privacy ⟹ `I(X; X̃) ≤ (n−r) log₂ q` bits [2024, §II-B].
- `I(X^n; X̃^n) ≤ (n−r)H(X)` for any `r`-subset-private mechanism [2022, §V].
- `λ(q,d,m) = dim RM_q(d,m)`; `= Σ_{i≤d} C(m,i)` at `q = 2`; `= C(m+d,d)` if
  `d < min{q, m(q−1)}` [2024, Cor. 1].
- `L(q,d,m,S) ≥ λ + S`, `≤ q^m − dmin + S + 1`, `≤ (S+1)λ` [2024, Lemmas 3–4].
- Perfect privacy needs a key as large as the data [2024, §II-B], which is why the scheme
  settles for subset privacy.

---

## 5. What this reading changed

**(a) A Week 1 "resolution" was itself wrong — the Singleton note.** `CLAUDE.md` lists under
*resolved, do not re-report* that the `max(r, …)` floor was re-justified "by Remark 1, not
Singleton", on the grounds that "under ICC the code is random, so Singleton does not apply".
That reasoning is incorrect. [2024, Remark 2] derives `m ≥ r` as: the encoding needs
`dmin(C⊥) ≥ r+1`, and Singleton bounds `dmin` of the `[n, n−m]` dual code by `m+1`, so
`m ≥ r`. **Singleton bounds the minimum distance of every linear code**, random ones
included; MDS is only where the bound is met with *equality*, and that is the part specific
to the Vandermonde predecessor. Both citations are valid and give the same floor, so the
code's behaviour was never wrong — only the docstring's reasoning. Corrected in
`utils.compute_required_m`.

**(b) `λ = C(m+d, d)` was being quoted without its side condition.** [2024, Cor. 1] gives it
only for `d < min{q, m(q−1)}`. Verified against the enumeration: `(q,m,d) = (2,10,3)` has
`λ = 176 = Σ C(10,i)` against `C(13,3) = 286`, and `(3,8,4)` has `λ = 423` where *neither*
closed form applies (495 and 163). `get_information_set` was already correct at any `q`
because it caps each entry at `q−1`; it was `tools/feasibility.py` and the docs that used
the unconditional formula. Now `utils.lambda_workers`, which takes the fast path in the
regime the project runs in (`q ≈ 2³⁰`, `d = 2`).

**(c) The storage-before-query ordering has a citation.** `docs/QUANTIZATION.md` §6 derived
it from the code; [2024, §II-A] states it as part of the model.

**(d) Open question 8 (quantile pre-transform) is really a question about lineage.** The
transform I proposed for maximising per-symbol entropy is the continuous analogue of
[2022]'s uniformization mechanism — and [ICC] *deliberately replaced* that mechanism with
smoothing, because uniformization needs `P_X` and ICC assumes it is unknown. A quantile
transform estimates `P_X` from the sample, so it reintroduces exactly the assumption ICC was
built to avoid, on top of the leakage concern already raised. The sharper question for Raviv
is therefore not "is this transform acceptable?" but **"for quantized real data, should one
uniformize (2022) or smooth (ICC), or compose both — and what does `r·H(X̂|X)` become?"**

**(e) Two concrete defects in the `get_information_superset` prototype**, which matter
whenever straggler tolerance comes back into scope:

1. It searches for a super-set of size exactly `λ + S`. That is Lemma 3's *lower bound*, which
   is not known to be achievable — the only general guarantees are `q^m − dmin + S + 1` and
   `(S+1)λ`. So the prototype can exhaust the search and raise `"Field size too small to find
   a valid Information Super-set"`, which misdiagnoses the failure: the target size may
   simply not exist.
2. Definition 4 is stated for **multisets**, and the repetition construction `L ≤ (S+1)λ`
   relies on that. The prototype skips any candidate already present
   (`if any(np.array_equal(gf_pt, sp) …): continue`), so it cannot represent repetition and
   cannot reach the one bound that is always achievable.

**(f) Not implemented, worth a line in the paper's future work:** syndrome-based upload
(`n → n−m`), and the LCC straggler scheme (`N = (λ−1)d + S + 1`, needs `q ≥ N`).
