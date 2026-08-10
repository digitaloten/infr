# nvidia-ada.md — first NVIDIA measurements (RTX 4000 SFF Ada)

`docs/perf/results.md` asks for NVIDIA numbers; this file is that data for one
card. It records **measurements and mechanisms only** — no fix is claimed and
nothing here has been optimized. infr ran fully stock: no `--ubatch`, no config
file, no kernel flags, no tuning of any kind.

**This is the measurements, not the first analysis.**
[`vulkan-review.md`](vulkan-review.md) already reviewed the NVIDIA path by
reading the code, and closed with five items marked *"needs measurement (cannot
settle without profiling)"*. Two of them are about this hardware: the flash
bm=32 tile share (its finding #3) and NVIDIA inheriting AMD's decode int8 dtype
set (its finding #2). The follow-up pass sketched at the end is scoped to answer
the first one on its own stated terms.

All numbers below come from one session, one engine at a time on an idle card,
per `playbook.md` §"Hardware contention".

## Provenance

| item      | value                                                                                    |
| --------- | ---------------------------------------------------------------------------------------- |
| GPU       | NVIDIA RTX 4000 SFF Ada Generation, 20475 MiB, driver 580.173.02, compute capability 8.9 |
| host      | Ubuntu 24.04.4, 20 cores, 62 GiB RAM, single discrete GPU                                |
| model     | `Qwen3.6-27B` Q4_K_M (unsloth GGUF) — GGUF arch `qwen35`, 15.65 GiB, 26.90 B params      |
| infr      | commit `f67e21a`, `cargo build --release`, rustc 1.97.1                                  |
| shaders   | glslc shaderc v2026.3 (Vulkan SDK 1.4.357.1)                                             |
| llama.cpp | Vulkan build `7ba604f1c (10331)`; CUDA build `22dc605c4 (10257)`                         |

⚠ The two llama.cpp builds are **different revisions** (10331 vs 10257), so the
Vulkan and CUDA columns are not strictly comparable to each other. Each is
individually a valid reference for infr; the Vulkan column is the intended
apples-to-apples one (same API, same card, same GGUF).

### Toolchain note — glslc version is load-bearing

The shaders declare `GL_EXT_float_e4m3` and `GL_EXT_bfloat16`. **Tested:**
LunarG's Ubuntu-noble apt SDK 1.4.313 (`glslc` reporting `shaderc v2023.8`,
built 2025-04-29) rejects `GL_EXT_float_e4m3` with
`'#extension' : extension not supported`, failing `quant_f8_row.comp`,
`native_gemm_f8cm_q8_0.comp` and `repack_q8_to_f8.comp`. The 1.4.357.1 tarball
SDK (`shaderc v2026.3`) compiles all three clean and builds the tree.

**Not tested:** Ubuntu 24.04's own `glslc` package. `apt-cache policy` gives its
candidate as `2023.8-1build1`, i.e. the same shaderc series as the LunarG build
that failed, so it is *likely* to fail identically — but it was never installed
and that is an inference, not a measurement.

## Results

Same GGUF, same host, KV f16 on every row, `-ngl 999`, one engine at a time.
infr rows are `infr bench`; llama.cpp rows are `llama-bench` with matching
`-p/-n/-d/-r`. infr spreads are min–max over reps; llama.cpp reports ± stddev.

⚠ **Four deviations from this project's own measurement procedure. None is large
enough to explain a ~2.7× gap, but each bounds how tightly an individual row
should be read:**

1. **`infr compare` was not used**, though it exists for exactly this job. It
   shells out to a *system* `llama-bench`, and this box had none — the only
   llama.cpp available was an upstream container image. The matched
   `-p/-n/-d/-r` flags were therefore driven by hand against the containerised
   `llama-bench`, which is flag-for-flag equivalent but bypasses `compare`'s own
   bookkeeping.
2. **No cooldown between runs.** `playbook.md` rule 6 requires *"60s+ between
   back-to-back"* manual bench runs against thermal drift it puts at 2–8%. This
   session ran engines back-to-back with no sleep. The measured gap is ~2.7×
   (206.4 vs 559.20 t/s at pp2048), far past that band, so the headline
   survives — but individual rows carry an unquantified thermal term, and the
   1.09× CUDA-vs-Vulkan margin in observation 3 is *within* it.
3. **Flash attention was left at each engine's default and never pinned.** Both
   llama.cpp builds used here report `-fa, --flash-attn <on|off|auto> (default:
   auto)`, so the reference rows ran with FA *auto-selected*, not disabled — and
   `infr compare` passes `-fa auto` to `llama-bench` for the same reason
   (`crates/infr-cli/src/main.rs:3212`), so the defaults agree. What is **not**
   recorded is what `auto` resolved to on this card; `llama-bench` only prints
   an `fa` column when the value is non-default, and these logs have none.
   ⚠ Anyone repeating this against an **older** llama.cpp tree should pin it
   explicitly: `llama-bench` used to default flash attention to **off**
   (`cmd_params_defaults.flash_attn = { false }`), and a comparison built from
   such a build would be FA-mismatched against infr, which always runs its flash
   kernel here.
4. **Physical batch width differs at pp8192, and only there.** `llama-bench`'s
   `-ub` default is 512 and does not move with prompt length; infr dropped to
   128 at pp8192 (observation 1). So **pp2048 is a like-for-like 512-vs-512
   comparison and pp8192 is not** — 128 vs 512. The pp8192 row is the weaker of
   the two as a kernel-quality statement; lean on pp2048 (0.37×).

| test            | infr Vulkan             | llama.cpp Vulkan (10331) | llama.cpp CUDA (10257) | infr ÷ llama Vk | infr ÷ llama CUDA |
| --------------- | ----------------------- | ------------------------ | ---------------------- | --------------- | ----------------- |
| pp2048          | 206.4 (spread 0.9%, r3) | 559.20 ± 0.71 (r3)       | 608.75 ± 0.64 (r3)     | **0.37×**       | 0.34×             |
| pp8192          | 185.1 (spread 0.6%, r2) | 541.48 ± 2.93 (r3)       | 591.63 ± 0.78 (r3)     | **0.34×**       | 0.31×             |
| tg128           | 11.4 (spread 0.2%, r3)  | 14.41 ± 0.01 (r3)        | 14.65 ± 0.01 (r3)      | **0.79×**       | 0.78×             |
| tg128 @ d8192 ¹ | 11.1 (spread 0.1%, r2)  | 13.96 ± 0.02 (r2)        | 14.24 ± 0.01 (r2)      | **0.80×**       | 0.78×             |

¹ infr's row needed an explicit `--ctx 8448`; the default session context on
this card is 8271 tokens, 50 short of the 8321 the shape requires (see
observation 6). llama.cpp's row ran at its own default.

Also measured: infr `pp512` = 209.6 t/s (1 rep). ⚠ That figure came from an
interactive smoke run whose output was **not captured to a log file** — unlike
every row in the table above, it cannot be re-read from a saved artifact. It is
reported for shape only; do not build on it.

Per-row infr chunk heights, as reported by `infr bench` itself:

| row           | ubatch  | submit splitter |
| ------------- | ------- | --------------- |
| pp2048        | 512     | split/166       |
| pp8192        | **128** | unlimited       |
| tg128         | 1024    | unlimited       |
| tg128 @ d8192 | 128     | unlimited       |

The prefill spreads here are 0.6–0.9%. `results.md` §"How reproducible is each
column?" reports batch prefill (`pp512`) as stable to **~1.5%** run to run on
the current tree — its older large-variance warning (34.5% worst on pp512, 31.7%
on pp4@d4096) is explicitly **retracted** there ("the old warning in this box no
longer holds"), and the residual softness it still records is confined to
`pp4@d4096`, a 4-token depth-primed shape not measured here. So the ~2.7× gap is
wide against a measurement whose own reproducibility is single-digit percent;
noise does not explain it.

### Device capabilities

infr's own probe line — ⚠ transcribed from the same uncaptured interactive smoke
run as the `pp512` figure, so it is **not re-readable from a saved log** the way
the llama.cpp block below it is:

```
GPU: NVIDIA RTX 4000 SFF Ada Generation | f16:y f16cm:y bf16:y bf16cm:y
     f8:n f8cm:n i8:y i8dot:y i8cm:n subgroup:32-32 sgp:32 shared:48KB
```

llama.cpp's, same card, same session (verbatim from its run log):

```
NVIDIA RTX 4000 SFF Ada Generation (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 |
fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 |
matrix cores: NV_coopmat2
```

**This SKU** exposes neither fp8 nor int8 cooperative matrix (`f8:n`, `i8cm:n`),
so the fp8/int8 coopmat paths `playbook.md` gates off are unavailable here
regardless of their merit; f16 and bf16 coopmat are both available. Whether that
holds for every Ada part is not established by one probe on one card and is not
claimed. The independently-logged llama.cpp line corroborates the two
capabilities that matter most below — `shared memory: 49152` (= 48 KiB, see
observation 5) and `bf16: 1`.

## Observations

These are mechanisms consistent with the data, not verified causes. Each is a
hypothesis with a named test; none has been measured yet.

### 1. Prefill runs at the 128-row chunk floor at pp8192

`crates/infr-llama/src/seam/mod.rs:799-846`: the default prefill chunk is 1024
rows, and a dense placement sweep walks a shrink ladder `512 → 256 → 128` when
the chunk's activation reserve is what tips the model out of VRAM residency. The
ladder's own doc comment states the floor's cost: *"128 is the floor: below it
the per-dispatch launch overhead dominates prefill entirely."*

15.65 GiB of weights on a 20 GiB card is exactly that squeeze, and pp8192
reported `ubatch 128` — the floor — while pp2048 settled at 512. The trade the
ladder makes (chunk height for residency) is right on its own terms; what is
unmeasured on this card is how much prefill throughput it costs, and whether a
taller chunk plus streaming would beat a resident 128-row chunk here.

**Test:** pin `-u 128/256/512/1024` at pp2048 and pp8192, recording per cell
whether placement stayed resident or flipped to streaming.

### 2. The coopmat2 verdict was measured on RADV emulation, not on NVIDIA

`playbook.md:588-603` records coopmat2 (`VK_NV_cooperative_matrix2`) as a
measured dead end — **0.964× at 4096 wg, i.e. slower** — and explicitly scopes
that measurement: it was taken on RDNA4 behind
`radv_cooperative_matrix2_nv=true`, with a standing note to *"re-run it if a
future Mesa stabilizes/optimizes coopmat2"*. `coopmat2_test.rs:59-76` documents
the same gating.

On Ada, coopmat2 is a native driver path rather than a Mesa emulation, and
llama.cpp's Vulkan backend uses it as its primary matmul and attention route:
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:2999` selects `FA_COOPMAT2` when
`device->coopmat2`, with a dedicated shape-driven tuner at
`get_fa_tuning_params_coopmat2()` (`:2970`), plus coopmat2 branches at `:3206`,
`:3465`, `:3481`, `:4681`, `:4747`. infr's shader tree declares
`GL_KHR_cooperative_matrix` only (17 occurrences across `.comp`/`.glsl`);
coopmat2 appears in exactly one file repo-wide — the isolated
`crates/infr-vulkan/examples/coopmat2_test.rs` probe — and in no production
kernel. (Searched set: all `.rs`, `.comp` and `.glsl` under `crates/`.)

The existing dead-end verdict therefore says nothing about this hardware. It
should not be treated as settled for NVIDIA either way until measured natively.

### 3. The gap is not the Vulkan backend

llama.cpp CUDA is only 1.09× its own Vulkan build at pp2048 (608.75 vs 559.20)
and 1.02× at tg128 — and both builds ran under identical defaults, so that
particular ratio is internally clean (though 1.09× is itself within the
uncontrolled thermal band noted above). Vulkan is not meaningfully handicapped
on this card.

⚠ **That rules out one alternative, not all of them.** "Not an API penalty" does
not yet establish "therefore infr's kernels", because a third category sits
between them: measurement asymmetries. The pp8192 batch-width mismatch
(deviation 4), the unlogged flash-attention resolution (deviation 3), and the
submit-splitter regime difference (observation 7) are none of them API penalties
and none of them kernel quality. Kernel difference is the leading hypothesis and
the rest of this document treats it as such — but it is not earned until those
three are controlled for.

### 4. Prefill and decode fail differently

Decode sits near 0.8× and is flat **over the only two depths measured** (0.79×
at tg128, 0.80× at tg128 @ d8192); prefill sits at 0.34–0.37×. Decode is
memory-bandwidth bound — where a weaker matmul primitive costs little — and
prefill is compute bound, where it costs everything. The shape of the gap is
consistent with observations 1 and 2 and inconsistent with, say, a
weight-loading or KV-layout problem.

⚠ **This does not establish that decode is depth-independent, and there is a
specific reason to doubt it.** `results.md:500-505` records infr's AMD decode
falling to **0.65× at d32768**, with `attn_partial` at **59% of decode GPU
time**, and says outright that the published d4096-topped table *"flatters us at
exactly the shape a long-lived agent session actually reaches."* The mechanism
behind that decay is attention cost growing with KV length — the same kernel
family observation 5 says NVIDIA runs at half tile size. If depth compounds with
the tile handicap, this card would be *worse* than AMD at depth, not flat. d8192
is nowhere near where AMD's decay showed up.

**Why it was not measured further:** at 15.65 GiB resident there is only room
for ~8.3k tokens of f16 KV on this card (observation 6). Going deeper requires
quantized KV or a smaller model — either of which changes a variable that must
then be changed on the llama.cpp side too.

### 5. The 48 KB shared-memory ceiling is real on this card

`vulkan-review.md` finding #3 predicted from code that NVIDIA's 48 KB
`maxComputeSharedMemorySize` forces the flash-warp tile from bm=64 (58112 B,
doesn't fit) down to bm=32 (29056 B), and the register-O tile from BR=128 down
to BR=64 (29440 B). ⚠ That review's own line citations for this have gone stale
as the file moved; the current code is `recorder.rs:4818-4835` (bm) and
`:5043-5050` (BR) — verified against this tree, not inherited.

Both engines' probe lines agree the limit is 48 KiB here (`shared:48KB`;
llama.cpp's independently-logged `shared memory: 49152`), so the predicted
fallback is what runs — 2× the workgroups, each doing half the work.

One nuance the review's summary omits and the code states: the bm=64 tile is
taken only when shared memory allows **and** `n >= 64`. Partial tiles below 64
rows take BM=32 on every device including RADV, deliberately
(`recorder.rs:4828-4835`). So the NVIDIA penalty applies to full tiles, not to
every flash dispatch.

That review could not say whether it costs anything, and set an explicit
decision rule (`vulkan-review.md:167-170`, verbatim): *"Profile pp512 attention
time on an NVIDIA GPU with `INFR_PROF_OPS=1`. If attention is <20% of prefill
wall time, the tile gap is not the lever. If it's >40% and the bm=32 WG count
visibly underfills the SM array, a register-O redesign becomes worth scoping."*
Note it names **pp512** specifically. An RTX 4000 SFF Ada is a far smaller part
than the RTX 4090 the review reasoned about, so it tests the small-GPU end of
that argument.

⚠ **The underfill half of that rule cannot be evaluated with what infr and the
usual tools report.** infr's `compute_units` is populated only from
`VK_AMD_shader_core_properties` (`crates/infr-vulkan/src/lib.rs:2093`) and is
therefore `0` on any NVIDIA card; `nvidia-smi -q` reports SM *clocks* but no SM
count; and `vulkaninfo` does not dump `VK_NV_shader_sm_builtins`, the struct
that would carry it. Obtaining the number means writing a probe. Until then the
attention-share half is measurable and the underfill half is not — a `>40%`
reading yields "threshold met, second condition unevaluable", never "redesign
justified".

### 6. Default KV capacity clamps context to 8271 tokens

`infr bench -p 0 -n 128 -d 8192` fails with
`prompt 8193 + gen 128 exceeds the session KV capacity 8271`. The shape needs
8321 tokens (the engine primes `d + 1` = 8193, then generates 128); with
15.65 GiB resident the default session context lands at 8271, **50 short**. The
row in the table was produced with an explicit `--ctx 8448`. Not a performance
finding, but it is a sharp edge on a card this size — the default silently
refuses a shape `llama-bench` runs without comment.

### 7. The submit splitter fired on one row and not the other

The splitter's cap is derived from the *previous* forward's measured
`elapsed_ns / dispatches` (`crates/infr-core/src/backend.rs:288-297`), so it is
not an independent knob — chunk height feeds it. The two prefill rows landed in
different regimes: pp2048 (ubatch 512) reports `submit split/166`, pp8192
(ubatch 128) reports `submit unlimited`.

⚠ **The available evidence argues against this being the main driver.** The row
that *was* split has the **better** ratio (0.37× at pp2048) and the unsplit row
the worse (0.34× at pp8192) — the opposite of "splitting is what hurts infr".
That is n=2 with chunk height confounded into it, so it is weak either way, and
it is why any follow-up sweep must hold the splitter fixed
(`INFR_SUBMIT_DISPATCHES=0`) rather than letting it float with chunk height.

### 8. ⚠ The test model is 75% linear-attention — this weakens observation 5's test

Everything above was measured on one model, `qwen35`, and that architecture is
not a conventional transformer. Per [`qwen35.md`](../qwen35.md): of its 24
layers, **6 are full attention (layers 3, 7, 11, 15, 19, 23) and the other 18
are gated DeltaNet linear-attention** — a different kernel family from the
flash-attention path that observation 5 and `vulkan-review.md` finding #3 are
about.

So the flash tiles under suspicion govern **a quarter of this model's layers**.
A low attention-share reading here would be much weaker evidence that "the tile
gap is not the lever" than the same reading on a dense transformer, because most
of the wall time is going somewhere the hypothesis never claimed.
`vulkan-review` specified pp512 for that test but not an architecture, and this
one is an unlucky choice for it.

Nothing else in this document depends on the architecture — the prefill and
decode ratios are what they are for this model — but they should not be read as
"infr on NVIDIA" in general either.

## Reproducing

```bash
M=/path/to/Qwen3.6-27B-Q4_K_M.gguf

# infr (Vulkan)
infr bench "$M" -p 2048 -n 0 -r 3
infr bench "$M" -p 8192 -n 0 -r 2
infr bench "$M" -p 0 -n 128 -r 3
infr bench "$M" -p 0 -n 128 -d 8192 -r 2 --ctx 8448   # --ctx needed, see observation 6

# llama.cpp reference, same flags
llama-bench -m "$M" -p 2048,8192 -n 128 -r 3 -ngl 999
llama-bench -m "$M" -p 0 -n 128 -d 8192 -r 2 -ngl 999
```

## What would settle this

A follow-up pass on the same card, committing to **no kernel work**, would
attribute the prefill gap into three parts: chunk height, submit splitting, and
the residual per-kernel GPU time that only a kernel change can move. That split
decides whether the real follow-up is a native coopmat2 port or a heuristics
port, and prevents a coopmat2 decision resting on an unmeasured assumption about
chunk height.

What such a pass has to do, ordered by what makes it valid rather than merely
informative:

1. **Sweep chunk height with the splitter pinned.** `-u 128/256/512/1024` at
   pp2048 and pp8192 with `INFR_SUBMIT_DISPATCHES=0` throughout, so chunk height
   and splitter behaviour are separated rather than confounded (observation 7).
   Pinning `-u` skips the placement sweep entirely
   (`crates/infr-llama/src/seam/mod.rs:824-831`), so each cell must record
   whether it stayed resident or fell into dense streaming (`INFR_PAGER_STATS=1`
   is the closest diagnostic, though it is documented for MoE expert pools);
   streaming cells are a separate bucket, not "chunk-height cost".
2. **Profile per kernel** — `INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1` at pp8192
   and at pp512, plus `INFR_SEAM_NO_REPLAY=1` for decode, whose replay tape
   carries no timestamps.
3. **Use a dense control model for the finding-#3 verdict**, not `qwen35`
   (observation 8) — small enough to leave KV room at depth, standard attention
   on every layer.
4. **Sweep decode depth on that control model** — d4096/d16384/d32768, both
   engines — before repeating that decode is flat (observation 4).
5. **Pin flash attention explicitly on both sides** and record the value rather
   than inheriting `auto` (deviation 3), and **match batch width** at pp8192
   (deviation 4).
6. **Respect `playbook.md` rule 6** — 60s+ between runs (deviation 2).

Acceptance is an attribution, not a speed-up: each of the three buckets carries
a number, and the finding-#3 verdict is reported against `vulkan-review`'s own
pre-registered thresholds (<20% / >40% attention share at pp512), measured on
the control model, with the underfill condition recorded as unevaluable per
observation 5. A reading between 20% and 40% is reported as landing between
them, not rounded to whichever side suits a follow-up. If the control model is
not obtained, the verdict is **not established** — never a pass by default.

Deliberately out of scope for that pass: no kernel edits, no coopmat2 bindings,
and **no change to `DENSE_UBATCH_LADDER` or any default** — a large ubatch win
is a *finding*, not a patch, since the ladder exists for RDNA residency reasons
that have not been re-measured and infr's tuning to date is AMD-derived.

Two NVIDIA-relevant items from `vulkan-review.md` stay open, recorded here so
they are not mistaken for settled:

- **Finding #2 / needs-measurement #3** — NVIDIA inherits AMD's decode int8
  dtype set (`{Q4K, Q6K, Q2K, Q4_0, Q5_0, Q5_1, IQ4_NL}`) unmeasured. The pass
  above measures decode *attribution*, not dtype-set alternatives. It is cheap
  to test later (`INFR_MMV_MW=1` A/B) and is the obvious next candidate if
  decode rather than prefill becomes the target.
- **Finding #5** — `with_padded_dst`'s per-forward temp allocation is
  vendor-agnostic and applies here too; the review rates it Low impact and the
  pass above does not isolate it.

Finally, the scope this document does **not** cover: one Ada card with 20 GiB
running a model that occupies 78% of it. A 24 GiB or 48 GiB card would not hit
observation 1 at this model size at all, and no claim is made about other NVIDIA
parts, other quants, or other architectures.
