# Parameter sweep: does a massive `q` make the scheme infeasible?

Week of 2026-10-12. Raw data `results/q_sweep_*.csv`, machine `results/q_sweep_machine.json`,
log `results/q_sweep_log.txt`, figures `results/figures/fig*.pdf`. Regenerate with

```bash
.venv/bin/python tools/q_sweep.py --exp all          # sequential; several hours, dominated by C
.venv/bin/python tools/plot_sweep.py
.venv/bin/python tools/crt_leakage_demo.py           # section 6
```

## 1. Answer

**A massive `q` does not make the scheme explode.** Holding the problem size fixed, cost jumps
once — at the `galois` native-arithmetic limit, `log2 q = 31.5` — and is nearly flat after that,
all the way to `q ≈ 2^521`. What does grow with `q` is the problem size: for quantised data the
key length `m` (hence `λ = C(m+2,2)` workers) grows with `log q` through Theorem 1, but it is
**capped near `n + p`**, so the growth is bounded. Every measured cell decoded exactly.

| question | answer | evidence |
|---|---|---|
| per-step cost vs `q`, problem size fixed | ×4.5 jump at the native limit, then flat from `2^32` to `2^127`; ×1.5 more by `2^521` | §3, fig. 1 |
| is the large-`q` penalty constant in `λ`? | decode: yes (×7.6–7.7). Full step: grows slowly (×3.6 at `λ=496` → ×6.5 at `λ=5151`). Setup: grows faster (×1.2 → ×13.5) | §4, fig. 2 |
| does `λ` depend on `q`? | not at fixed `m`; for quantised data yes, `m ≈ n(1 − h)` with `h = H/log2 q`, capped at `≈ n + p` | §5, fig. 3 |
| what precision stays native? | `f_X ≤ ⌊(31.5 − 1 − log2 n_s − 2·log2 A_X − f_w − log2(P·A_w + 1))/2⌋`: 7 for the case study | §5, fig. 4 |
| can a huge field be built at all? | verified construction factors `q − 1` (hangs at `2^521 − 1`); skipping the unused primitive element takes milliseconds at any size | §5.3, fig. 7 |
| which formats are feasible? | single prime: 23/30 of the grid; all fail only at `n = 200` above 31.5 bits (`λ ≳ 10,000` in large-`q` mode). With CRT (a): 30/30, ×1.7–6.8 faster per step at equal `m` — cost only, privacy unproven | §7, figs. 5–6 |

## 2. Method

`tools/q_sweep.py` runs every cell in a fresh interpreter through the real `Client`/`Server` code
and times each phase: field construction, building the interpolation matrix `M`, encoding,
sharding (**setup**, paid once per upload) and the overflow gate + worker evaluation + decoding of
the `P = 3` gradient queries (**per training step**). Per-step phases are repeated; the first
repeat is reported as cold, the median of the rest as warm. A cell's timings are kept only if
every decoded value equals the field evaluation (A, B) or the exact integer oracle (C).

Controls, each added after it was found to matter:

- **One cell at a time.** Four concurrent cells on the M2 Pro inflated large-`q` timings by up to
  ×3.5 (verified by re-running cells alone). `--jobs 1` is the default.
- **Sleep and power are recorded per cell** (`power`, `sleep_events` columns, from the macOS
  power log), and a cell that straddles a sleep is marked `slept` and re-run. A lid close during
  experiment B produced a 38.7 s cell whose re-measurement gave 26.6 s.
- **Hard per-cell cap** that kills the whole process group (the macOS venv launcher otherwise
  survives a timeout), and a memory pre-check (`3 × 8λ²` bytes per instance) that records a
  cell as `skipped:memory` instead of swapping.
- **Rows are written after every cell**; `--resume` / `--rerun KEY=VALUE` re-measure only what is
  missing or flagged.

Machine: Apple M2 Pro (8 performance + 4 efficiency cores), 16 GB, Python 3.11, `galois` 0.4.10,
NumPy 2.3.5 (`results/q_sweep_machine.json`). Experiments T, A, B, D ran on battery with Low Power
Mode off; C on AC (see the `power` column). **All timings are properties of `galois` on this
machine**, not of finite-field arithmetic in general (§8).

## 3. Experiment A — cost vs `q` at fixed problem size (fig. 1)

`m ∈ {20, 48}` (`λ = 231, 1,225`), `n = 100`, `q` from `2^13` to `2^521`, 28 cells, all exact.

| `log2 q` | 13–31.5 (native) | 32 | 61 | 127 | 256 | 521 |
|---|---:|---:|---:|---:|---:|---:|
| per step, `m = 48` (s) | 0.30–0.40 | 1.73 | 1.79 | 1.82 | 2.06 | 2.75 |
| build `M`, `m = 48` (s) | 0.15–0.18 | 4.63 | 4.63 | 4.62 | 4.68 | 4.55 |
| worker evaluation, `m = 48` (s) | 0.17–0.22 | 0.27 | 0.30 | 0.34 | 0.52 | 1.20 |
| decode, `m = 48` (s) | 0.11–0.23 | 1.46 | 1.50 | 1.48 | 1.54 | 1.55 |

- **The jump is interpreter overhead, paid once.** Above the limit every element is a Python
  integer and every operation goes through the interpreter; that costs the same at 61 bits as at
  127.
- **Only worker evaluation keeps growing** (×4 from `2^61` to `2^521`): big-integer multiplication
  cost grows with the number of 30-bit limbs (3 at 61 bits, 18 at 521). Decode and `M` are
  dominated by interpreter overhead and stay flat.
- **Two things look *better* above the limit; neither is real.** Field construction drops from
  ~0.4 s to milliseconds because native fields are compiled per field when created (§5.3), and at
  `m = 20` sharding looks cheaper because it is timed once and its first native call includes
  compiling the matrix product (25–50 ms cold vs 0.1–8 ms warm; large-`q` mode has no compile step).
- **A second, hidden cliff inside native.** `galois` multiplies prime-field matrices in floats or
  `int64` and reduces mod `q` at the end, choosing the type from the bound
  `max(A.shape)·(q − 1)²` (`galois/_domains/_linalg.py`). Past `2^63` it falls back to Python
  integers. With `A = T` of shape `λ × m` the cutoff is `q ≈ 2^63/λ`: `2^27.6` at `λ = 231`,
  `2^26.4` at `λ = 1,225` (measured: fast at `2^26.25`, slow at `2^26.5`, independent of `m`). So at
  the case study's `q ≈ 2^30.45`, **sharding, encoding and the final step of decoding already run
  in Python integers while "native"**; element-wise operations (worker evaluation, building `M`)
  stay compiled. Not fixable cheaply: even the correct sum length `m` gives `m·q² > 2^63` at
  `q ≈ 2^30`.

## 4. Experiment B — cost vs `λ` (fig. 2)

`m` from 10 to 100 (`λ` to 5,151) at `q = 2^30` (native), `2^61`, `2^127`; 30 cells, all exact. The
`2^127` series was re-measured after the lid-close sleep and now lies on the `2^61` curve
(26.62 s vs 26.59 s per step at `λ = 5,151`).

| `λ` | step native | step `2^61` | ratio | setup native | setup `2^61` | ratio | peak RSS `2^61` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 496 | 0.12 s | 0.42 s | ×3.6 | 0.4 s | 0.5 s | ×1.2 | 244 MB |
| 1,326 | 0.42 s | 2.07 s | ×5.0 | 0.7 s | 5.8 s | ×8.0 | 381 MB |
| 2,556 | 1.21 s | 6.92 s | ×5.7 | 2.3 s | 29.4 s | ×12.7 | 869 MB |
| 5,151 | 4.06 s | 26.59 s | ×6.5 | 14.4 s | 195.5 s | ×13.5 | 2.4 GB |

Log–log slopes over `λ ≥ 400`: per step `λ^1.52` native, `λ^1.79–1.80` at `2^61`/`2^127`; setup
`λ^1.54` native, `λ^2.52` at both large sizes. **Decode alone is a constant ×7.6–7.7 penalty;
the full step's penalty grows slowly (`λ^0.27`), and setup's grows fastest — building `M` is the
first thing to become expensive in large-`q` mode.** `2^61` and `2^127` are indistinguishable
throughout, confirming that above the limit the field size barely matters. Decoding is still a
dense solve here (`λ^~2` in practice, because `M` is ~0.5% nonzero); the block-triangular decoder
of `docs/PLAN.md` would change these exponents for every `q`.

## 5. Theory — how `q` drives `m`, and what precision stays native (figs. 3–4)

### 5.1 `m` and `λ` against `q` (fig. 3)

For quantised data a symbol carries a fixed number of bits (`≈ f_X + 1.325` by the entropy bound
of `docs/QUANTIZATION.md` §8), so `h = H/log2 q` falls as `q` grows and Theorem 1's
`m ≈ n(1 − h) + p + …` rises — but never past `≈ n + p`. At `n = 100`, `f_X = 7`: `m = 77`, `λ = 3,081`
at the case study's `q`; `m = 90`, `λ = 4,186` at `2^61`; `m = 101`, `λ = 5,253` at `2^521`. So
**the problem a larger field forces is at most ×1.7 bigger** at this `n`, and figure 2 prices it.

### 5.2 The field a format needs, and the native precision cap (fig. 4)

From the public worst-case bound (`docs/QUANTIZATION.md` §7),
`log2 q ≈ 1 + log2 n_s + 2(f_X + log2 A_X) + f_w + log2(P·A_w + 1)`. Each extra bit of `f_X` costs
**two** bits of `q` (the gradient multiplies two data values), each ×4 in samples costs one bit of
`f_X`. Largest `f_X` that stays native (`A_X = 4`, `A_w = 2`):

| `n_s` | P=3, `f_w`=0 | **P=3, `f_w`=4** | P=3, `f_w`=8 | P=10, `f_w`=0 | P=10, `f_w`=4 | P=10, `f_w`=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 8 | 6 | 9 | 7 | 5 |
| 25 | 9 | **7** | 5 | 8 | 6 | 4 |
| 100 | 8 | 6 | 4 | 7 | 5 | 3 |
| 442 | 7 | 5 | 3 | 6 | 4 | 2 |
| 1,000 | 6 | 4 | 2 | 6 | 4 | 2 |
| 10,000 | 5 | 3 | 1 | 4 | 2 | 0 |

At `f_X = 7` a standardised feature is stored to within 0.004σ (1,025 levels, ~10 bits) — mild for
the case study; at 10,000 samples the native cap is `f_X = 3` (steps of 0.125σ). Beyond the cap
the scheme stays correct and runs in large-`q` mode (§3–4). The bound is worst-case (~680× loose);
a data-dependent bound would recover ~2 levels of `f_X` but leaks through the public `q`.

### 5.3 Field construction (experiment D, fig. 7)

Default `galois.GF(q)` construction verifies `q` and finds a primitive element, which requires
factoring `q − 1`. Native fields additionally compile their arithmetic per field (~0.3–0.5 s, paid
again for every new field) and, for `q ≤ 2^20`, build lookup tables (2.4 s at `2^20`). Just above
the limit there is nothing to compile, so construction drops to milliseconds — then grows with the
difficulty of factoring `q − 1`, not with `q`: 0.02 s at `2^61 − 1` (largest factor of `q − 1`:
11 bits), 1.2 s at `2^127 − 1` (37 bits), 106 s at `2^256 − 189`, unfinished after 120 s at
`2^521 − 1`. The scheme never uses the primitive element, so `tools/q_sweep.py` passes a placeholder
with `verify=False` above the limit: milliseconds at every size, arithmetic verified exact.


---

## 6. CRT regime (a): how several encodings work, and what they leak

### 6.1 Mechanics

A query result can be as large as `B`, so a single prime needs `q > 2B`. CRT replaces that
prime by `L` distinct primes `q₁ … q_L`, each below the native limit, with
`Q = ∏ qᵢ > 2B`.

| step | single prime `q` | CRT, instance `i = 1 … L` |
|---|---|---|
| data | `x̂ mod q ∈ F_q^n` | `x̂ mod qᵢ ∈ F_{qᵢ}^n` — the **same integer codes**, since `qᵢ > 2·X̂max` |
| key, code | `k ∈ F_q^m`, `G ∈ F_q^{m×n}` | `kᵢ ∈ F_{qᵢ}^m`, `Gᵢ ∈ F_{qᵢ}^{m×n}`, independent and uniform |
| upload | `x̃ = x̂ + kG` | `x̃ᵢ = (x̂ mod qᵢ) + kᵢGᵢ` |
| shard | `x̃ − tG`, `t ∈ I_{d,m}` | `x̃ᵢ − tGᵢ`, the **same** `I_{d,m}` (it depends only on `m, d` once `d < qᵢ`) |
| query | workers return `f(x̃ − tG)` | instance `i` returns `f(x̃ᵢ − tGᵢ) mod qᵢ` |
| decode | `gᵢ(k) = f(x̂) mod q` | `gᵢ(kᵢ) = f(x̂) mod qᵢ` for each `i` |
| lift | balanced lift mod `q` | CRT → `f(x̂) mod Q` → balanced lift mod `Q` |
| overflow gate | `B(f) ≤ (q−1)/2` | `B(f) ≤ (Q−1)/2`, checked once |

Correctness is unconditional: reduction mod each `qᵢ` is a ring homomorphism, and the CRT map
`Z_Q → ∏ F_{qᵢ}` is a ring isomorphism. Every arithmetic operation is native. The `λ` workers
can be the same machines, each holding `L` shares.

**The equivalence that matters.** `k` uniform in `Z_Q^m` is the same thing as
`(k mod q₁, …, k mod q_L)` independent and uniform, and likewise for `G`. So "L instances with
independent keys and codes **of the same length m**" *is* one instance of the scheme over the
ring `Z_Q`: `x̃ = x̂ + kG (mod Q)`. CRT is not a different scheme; it is a way of doing the
big-ring arithmetic with small native numbers. The only real design choice is **how `m` is
chosen**, and that is where privacy enters.

### 6.2 What the provider learns from several encodings

One encoding reveals the coset `x̂ + rowspace(G)`: for the `(n−m) × n` parity-check matrix `H`
of the code, the provider learns `H x̂ = H x̃`, which is `n − m` linear constraints on `x̂`.
Theorem 1 chooses `m` so that, for the data's distribution, enough candidates survive those
constraints that no `r`-subset is pinned down.

With `L` encodings the provider holds **every** constraint set at once, and the true `x̂` must
satisfy all of them. Heuristically, with `x̂` uniform on a set of `2^H` values,

```
candidates ≈ 2^H / ∏ᵢ qᵢ^(n − mᵢ)
```

- **Sizing each `mᵢ` as if its instance were alone** (`log_{qᵢ}` units) makes every factor
  `qᵢ^(n−mᵢ)` nearly `2^H` on its own — that is exactly what one instance is designed to spend.
  Two such factors overshoot `2^H`, and `x̂` is pinned.
- **Sizing every `mᵢ = m(Q)` jointly** (`log_Q` units, as for one field of size `Q`) makes the
  product `Q^(n−m)` — the same as a single big prime.

Measured exactly by brute force (`tools/crt_leakage_demo.py`; `x` uniform on `{−1,0,1}^8`,
6,561 candidates, 30 seeded trials per row; output of the committed tool):

| what the provider sees | candidates left (median) | min | `x` recovered |
|---|---:|---:|---:|
| one encoding, `q = 11`, `m = 6` | 54 | 39 | 0/30 |
| one encoding, `q = 13`, `m = 6` | 40 | 2 | 0/30 |
| **two encodings, each `m` sized alone (6, 6)** | **1** | 1 | **24/30** |
| one encoding over a single prime `≈ Q` (`q = 139`), `m = 7` | 47 | 11 | 0/30 |
| **two encodings, `m` sized jointly for `Q = 143` (7, 7)** | **44** | 13 | **0/30** |

Per-instance sizing hands the data to the provider. Joint sizing leaves as much uncertainty
as one prime of the same size.

### 6.3 Consequences

1. **The sweep uses joint sizing** (`m` from `log_Q`). So CRT here buys native-speed arithmetic
   for an arbitrarily large modulus, **and nothing else**: `m` and `λ` are those of a single
   prime of size `Q`.
2. **CRT gives no entropy advantage.** The "regime (b)" idea — primes below the per-symbol
   entropy so that `hᵢ ≈ 1` and `m` stops depending on `n` — computes each `hᵢ` per instance,
   which is the unsafe sizing above. Under joint sizing `h = H/log₂ Q`, exactly as for one big
   prime. **This contradicts** the Week-8 CRT table in `docs/PLAN.md` and the claims in
   `docs/QUANTIZATION.md` §4E and §11 that RNS "breaks the `h < 1/d` ceiling". Those
   passages are left unedited pending Raviv; see the open questions below.
3. **The same applies to any re-encoding of the same data**, CRT or not. Uploading the same
   `x` again under a fresh key — for example to enlarge `q` after a budget change, which the
   `quantize.QueryBudget` docstring currently suggests ("re-upload with a larger budget") —
   gives the provider two coset constraint sets, and the second upload's `m` must account for
   the first. Deleting the first upload does not help if the provider kept it.

### 6.4 Questions for Raviv

1. Is the ring view right — `L` encodings with independent `(kᵢ, Gᵢ)` of common length `m` ≡ one
   encoding over `Z_Q` — and does the smoothing lemma behind Theorem 1 (Pathegama et al.,
   stated over fields) extend to random linear codes over `Z_Q = ∏ F_{qᵢ}` with entropies in
   `log_Q` units? If yes, regime (a) is exactly as private as one big prime.
2. Is there a rigorous version of 6.2 — `I(X_R; X̃₁, …, X̃_L)` for independently sized
   encodings — that the paper should state as a negative result?
3. Is the `h < 1/d` ceiling (`docs/QUANTIZATION.md` §11) then unavoidable for fixed-point
   embeddings, with CRT only a computational device?
4. Re-uploads (6.3.3): should the protocol forbid re-encoding the same data, or size the new
   `m` against the conditional entropy given earlier uploads?


---

## 7. Experiment C — feasibility of quantisation formats (figs. 5–6)

Every quantisation format of the grid — `n_s ∈ {10, 25, 50}` (`n = 40, 100, 200`),
`f_X ∈ {4, 7, 10, 13, 16}`, `f_w ∈ {4, 8}`, `P = 3`, `A_X = 4`, `A_w = 2` — run end to end with `q`, `m`
and `λ` derived, not chosen: overflow bound → `q` (or CRT primes) → Theorem 1 with the entropy
bounds → `m` → `λ`. 51 cells on AC power, no sleeps, **every completed cell decoded to the exact
integer gradient**. Feasible means ≤ 60 s per training step and ≤ 20 min setup (`plot_sweep.py`
flags; the 20 min also equals the per-cell cap, so a timeout is an infeasible verdict).

### 7.1 Feasibility (fig. 5)

| | feasible formats |
|---|---:|
| single prime (native ≤ 31.5 bits, `galois` large-`q` mode above) | **23 / 30** |
| best of single prime and CRT regime (a) | **30 / 30** |

- **`n = 40` and `n = 100`: everything is feasible either way.** The worst single-prime cell is
  11.7 s per step and 57 s setup (`n = 100`, `f_X = 7`, `f_w = 8`, 34.5 bits).
- **`n = 200` is where the single prime breaks.** Native formats (`f_X = 4`, and `f_X = 7` at
  `f_w = 4`) run at 21.6–27.5 s per step with 2–4 min setup — feasible. Every format that needs
  more than 31.5 bits is infeasible: four time out (setup alone exceeds 20 min at `λ ≥ 10,585`)
  and three finish too slowly (84.5–95.9 s per step, 13–16 min setup at `λ = 9,316–10,011`). The
  boundary is `λ ≈ 10,000` in large-`q` mode on this machine.
- **CRT regime (a) makes all of them feasible:** 12.4–25.0 s per step and 2–4 min setup at
  `n = 200`, with `L = 2` primes in every case (no format in the grid needs more than 53.5 bits).
- **The `λ` values are large:** `n = 200` means 9,316–14,365 workers. That is the cost of Theorem 1
  at `h ≈ 0.2–0.3` (`docs/QUANTIZATION.md` §11's `h < 1/d` ceiling), and it is what pushes
  setup to minutes — a feasibility limit independent of `q`.

### 7.2 CRT vs single prime at equal `m` (fig. 6)

Every pair has the same `m` (verified), because the CRT primes are equal-sized with product
just above `2B + 1` (so `Q ≈ q`). The difference is therefore arithmetic only.

| `n` | per-step speedup of CRT | setup speedup of CRT |
|---:|---|---|
| 40 | ×1.7–1.9 | ×0.2–0.8 (CRT is *slower*: it compiles two native fields, ~0.3 s each, against no compile step in large-`q` mode — §5.3) |
| 100 | ×3.0–4.6 | ×6.0–7.5 |
| 200 | ×3.8–6.8 where the single prime finished; ∞ where it timed out | ×6.4–6.7 |

The speedup grows with `λ`, as experiment B predicts (the large-`q` penalty grows like `λ^0.27`
per step and `λ^1` in setup). Median over all completed pairs: ×3.6 per step, ×6.1 in setup.

**What this does and does not show.** CRT's *cost* advantage is real and grows with problem size;
its *privacy* with joint sizing is the open question of §6.4 and `docs/CRT_PROOF.md`. Until that
is proved, the recommended configuration is the single prime with the large-`q` warning
(`quantize.choose_prime` already prints it), which is feasible for everything up to `n = 100` and
for native formats at `n = 200`.


## 8. Caveats

- **Library- and machine-specific.** Every timing is `galois` 0.4.10 on one M2 Pro. The native
  limit (`q² < 2^63`), the per-field compile cost and the matrix-product fallback (§3) are
  `galois` implementation choices; a Montgomery/128-bit backend or CRT moves the limit to `2^63`
  or removes it. The *shape* of the result (a jump, then flat) is what should generalise.
- **Decoding is a dense solve.** The `λ`-exponents in §4 would change under the block-triangular
  decoder planned in `docs/PLAN.md`.
- **`m` depends on the entropy bound** of `quantize.entropy_bounds` (open questions 1 and 9 in
  `docs/PLAN.md`); if Raviv rules differently, figure 3 and experiment C's `λ` change.
- **CRT privacy is unproven** (§6, `docs/CRT_PROOF.md`): experiment C measures its cost only.
