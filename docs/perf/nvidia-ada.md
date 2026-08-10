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
set (its finding #2). Phase A below is scoped to answer the first one on its own
stated terms.

Everything below was produced in one session on 2026-08-10 (bench window
02:43:08Z–02:50:05Z plus a follow-up Vulkan-reference run), one engine at a time
on an idle card, per `playbook.md` §"Hardware contention".

## Provenance

| item      | value                                                                                      |
| --------- | ------------------------------------------------------------------------------------------ |
| GPU       | NVIDIA RTX 4000 SFF Ada Generation, 20475 MiB, driver 580.173.02, compute capability 8.9   |
| host      | Ubuntu 24.04.4, 20 cores, 62 GiB RAM                                                       |
| model     | `unsloth-Qwen3.6-27B-Q4_K_M.gguf` — GGUF arch `qwen35`, 15.65 GiB, 26.90 B params          |
| infr      | commit `f67e21a`, `cargo build --release`, rustc 1.97.1                                    |
| shaders   | glslc shaderc v2026.3 (Vulkan SDK 1.4.357.1)                                               |
| llama.cpp | Vulkan build `7ba604f1c (10331)`; CUDA build `22dc605c4 (10257)` — container images        |

⚠ The two llama.cpp builds are **different revisions** (10331 vs 10257), so the
Vulkan and CUDA columns are not strictly comparable to each other. Each is
individually a valid reference for infr; the Vulkan column is the intended
apples-to-apples one (same API, same card, same GGUF).

### Toolchain note — glslc version is load-bearing

The shaders declare `GL_EXT_float_e4m3` and `GL_EXT_bfloat16`. Ubuntu 24.04's
`glslc` (2023.8) **and** LunarG's noble apt SDK 1.4.313 (glslc dated 2025-04)
both reject `GL_EXT_float_e4m3` with
`'#extension' : extension not supported`. The 1.4.357.1 tarball SDK (shaderc
v2026.3) compiles the tree clean. A Linux packager following the distro
packages will not be able to build infr.

## Results

Same GGUF, same host, KV f16 on every row, `-ngl 999`, one engine at a time.
infr rows are `infr bench`; llama.cpp rows are `llama-bench` with matching
`-p/-n/-d/-r`. infr spreads are min–max over reps; llama.cpp reports ± stddev.

| test            | infr Vulkan            | llama.cpp Vulkan (10331) | llama.cpp CUDA (10257) | infr ÷ llama Vk | infr ÷ llama CUDA |
| --------------- | ---------------------- | ------------------------ | ---------------------- | --------------- | ----------------- |
| pp2048          | 206.4 (spread 0.9%, r3) | 559.20 ± 0.71 (r3)      | 608.75 ± 0.64 (r3)     | **0.37×**       | 0.34×             |
| pp8192          | 185.1 (spread 0.6%, r2) | 541.48 ± 2.93 (r3)      | 591.63 ± 0.78 (r3)     | **0.34×**       | 0.31×             |
| tg128           | 11.4 (spread 0.2%, r3)  | 14.41 ± 0.01 (r3)       | 14.65 ± 0.01 (r3)      | **0.79×**       | 0.78×             |
| tg128 @ d8192 ¹ | 11.1 (spread 0.1%, r2)  | 13.96 ± 0.02 (r2)       | 14.24 ± 0.01 (r2)      | **0.79×**       | 0.78×             |

¹ infr's row needed an explicit `--ctx 8448`; the default session context on
this card is 8271 tokens, one short of the 8320 the shape requires (see
observation 6). llama.cpp's row ran at its own default.

Also measured: infr `pp512` = 209.6 t/s (1 rep, smoke run).

Per-row infr chunk heights, as reported by `infr bench` itself:

| row           | ubatch | submit splitter |
| ------------- | ------ | --------------- |
| pp2048        | 512    | split/166       |
| pp8192        | **128**| unlimited       |
| tg128         | 1024   | unlimited       |
| tg128 @ d8192 | 128    | unlimited       |

The prefill spreads (0.6–0.9%) are far tighter than the ~30% run-to-run prefill
variance `results.md` warns about, so run-to-run noise is not what produces the
3× gap.

### Device capabilities

infr's own probe line:

```
GPU: NVIDIA RTX 4000 SFF Ada Generation | f16:y f16cm:y bf16:y bf16cm:y
     f8:n f8cm:n i8:y i8dot:y i8cm:n subgroup:32-32 sgp:32 shared:48KB
```

llama.cpp's, same card, same session:

```
NVIDIA RTX 4000 SFF Ada Generation (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 |
fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 |
matrix cores: NV_coopmat2
```

Ada exposes **neither fp8 nor int8 cooperative matrix** (`f8:n`, `i8cm:n`), so
the fp8/int8 coopmat paths `playbook.md` gates off are unavailable here
regardless of their merit. f16 and bf16 coopmat are both available.

## Observations

These are mechanisms consistent with the data, not verified causes. Each is a
hypothesis with a named test; none has been measured yet.

### 1. Prefill runs at the 128-row chunk floor at pp8192

`crates/infr-llama/src/seam/mod.rs:799-846`: the default prefill chunk is 1024
rows, and a dense placement sweep walks a shrink ladder `512 → 256 → 128` when
the chunk's activation reserve is what tips the model out of VRAM residency.
The ladder's own doc comment states the floor's cost: *"128 is the floor: below
it the per-dispatch launch overhead dominates prefill entirely."*

15.65 GiB of weights on a 20 GiB card is exactly that squeeze, and pp8192
reported `ubatch 128` — the floor — while pp2048 settled at 512. The trade the
ladder makes (chunk height for residency) is right on its own terms; what is
unmeasured on this card is how much prefill throughput it costs, and whether a
taller chunk plus streaming would beat a resident 128-row chunk here.

**Test:** `--ubatch 128/256/512/1024` at pp2048 and pp8192, watching whether
placement flips to streaming.

### 2. The coopmat2 verdict was measured on RADV emulation, not on NVIDIA

`playbook.md:588-603` records coopmat2 (`VK_NV_cooperative_matrix2`) as a
measured dead end — **0.964× at 4096 wg, i.e. slower** — and explicitly scopes
that measurement: it was taken on RDNA4 behind
`radv_cooperative_matrix2_nv=true`, with a standing note to *"re-open it if a
future Mesa stabilizes/optimizes coopmat2"*. `coopmat2_test.rs:59-76` documents
the same gating.

On Ada, coopmat2 is a native driver path rather than a Mesa emulation, and
llama.cpp's Vulkan backend uses it as its primary matmul and attention route:
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:2999` selects `FA_COOPMAT2` when
`device->coopmat2`, with a dedicated shape-driven tuner at
`get_fa_tuning_params_coopmat2()` (`:2970`), plus coopmat2 branches at
`:3206`, `:3465`, `:3481`, `:4681`, `:4747`. infr's shader tree declares
`GL_KHR_cooperative_matrix` only (17 occurrences); coopmat2 appears solely in
the two isolated `crates/infr-vulkan/examples/` probes, never in a production
kernel.

The existing dead-end verdict therefore says nothing about this hardware. It
should not be treated as settled for NVIDIA either way until measured natively.

### 3. The gap is not the Vulkan backend

llama.cpp CUDA is only 1.09× its own Vulkan build at pp2048 (608.75 vs 559.20)
and 1.02× at tg128. Vulkan is not meaningfully handicapped on this card, so the
0.37× is an infr-vs-ggml kernel difference, not an API penalty.

### 4. Prefill and decode fail differently

Decode sits at 0.79× and does not degrade with depth (0.79× at both tg128 and
tg128 @ d8192); prefill sits at 0.34–0.37×. Decode is memory-bandwidth bound —
where a weaker matmul primitive costs little — and prefill is compute bound,
where it costs everything. The shape of the gap is consistent with observations
1 and 2 and inconsistent with, say, a weight-loading or KV-layout problem.

### 5. The 48 KB shared-memory ceiling is real on this card

`vulkan-review.md` finding #3 predicted from code that NVIDIA's 48 KB
`maxComputeSharedMemorySize` forces the flash-warp tile from bm=64 (58112 B,
doesn't fit) down to bm=32 (29056 B), and the register-O tile from BR=128 down
to BR=64 (`recorder.rs:4426-4442`, `:4651-4658`). infr's own probe line on this
card reports `shared:48KB`, so the predicted fallback is what actually runs
here — 2× the workgroups, each doing half the work.

That review could not say whether it costs anything, and set an explicit
decision rule: profile attention share of prefill wall time — *"if attention is
<20% of prefill wall time, the tile gap is not the lever; if it's >40% and the
bm=32 WG count visibly underfills the SM array, a register-O redesign becomes
worth scoping."* An RTX 4000 SFF Ada is a far smaller part than the RTX 4090 the
review reasoned about, so it tests the small-GPU end of that argument. Its exact
SM count is not recorded here — `nvidia-smi` does not report it directly, and it
is not guessed; Phase A logs it from the device probe.

### 6. Default KV capacity clamps context to 8271 tokens

`infr bench -p 0 -n 128 -d 8192` fails with
`prompt 8193 + gen 128 exceeds the session KV capacity 8271`. With 15.65 GiB
resident, the default session context lands just below the 8320 tokens that row
needs. The row above was produced with an explicit `--ctx 8448`. Not a
performance finding, but it is a sharp edge on a card this size.

## Reproducing

```bash
M=/path/to/unsloth-Qwen3.6-27B-Q4_K_M.gguf

# infr (Vulkan)
infr bench "$M" -p 2048 -n 0 -r 3
infr bench "$M" -p 8192 -n 0 -r 2
infr bench "$M" -p 0 -n 128 -r 3
infr bench "$M" -p 0 -n 128 -d 8192 -r 2 --ctx 8448   # --ctx needed, see §5

# llama.cpp reference, same flags
llama-bench -m "$M" -p 2048,8192 -n 128 -r 3 -ngl 999
llama-bench -m "$M" -p 0 -n 128 -d 8192 -r 2 -ngl 999
```

## Phase A — attribute the gap (planned, not yet run)

Phase A commits to **no kernel work**. Its output is an attribution of the
prefill gap into three buckets: chunk height, submit splitting, and residual
per-kernel GPU time. That split decides whether the follow-up is a native
coopmat2 port or a heuristics port, and prevents a coopmat2 decision resting on
an unmeasured assumption about chunk height.

### Stages

The card is shared with a long-running llama.cpp server, so measurement time is
an approved exclusive window rather than free. Everything is batched into one
unattended script; a stage that fails logs and continues rather than forfeiting
the window.

| # | stage                                                                             | measures                              |
| - | --------------------------------------------------------------------------------- | ------------------------------------- |
| 0 | record serving state, stop it, assert 0 MiB in use                                | clean, restorable baseline            |
| 1 | `--ubatch 128,256,512,1024` × {pp2048, pp8192}                                    | chunk-height cost (observation 1)     |
| 2 | `INFR_SUBMIT_DISPATCHES=0` at the best ubatch                                      | submit-splitter cost                  |
| 3 | `INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1` at pp8192, best + default ubatch          | per-kernel prefill attribution        |
| 4 | `INFR_SEAM_NO_REPLAY=1` + `INFR_PROF_OPS=1` at tg128                              | decode attribution (replay is untimed)|
| 5 | llama.cpp Vulkan reference rows, same session                                     | same-session baseline, not a recall   |
| 6 | restart serving, verify HTTP 200 + healthy on 3 consecutive polls                 | restoration observed, not assumed     |

Stage 6 runs from a `trap` so it fires even if an earlier stage aborts the
script. Stage 5 is re-run rather than quoted from this document because the two
sets would otherwise come from different sessions.

### Acceptance

Phase A succeeds when each of the three buckets carries a number and the
residual — the part only a kernel change can move — is quantified. It does
**not** succeed on a speed improvement; no default is changed in Phase A.

It must also return a verdict on [`vulkan-review.md`](vulkan-review.md) finding
#3 using that finding's own pre-registered thresholds: attention share of
prefill wall time **<20%** (tile gap is not the lever) or **>40%** with visible
SM underfill (register-O redesign worth scoping). A value between the two is
reported as landing between them, not rounded to whichever side suits a
follow-up. Stage 0 also logs the device's SM count so the underfill half of that
rule can be evaluated rather than asserted.

Every quoted number names its ubatch, engine, build id, and shape, because the
sweep deliberately compares runs whose dispatch populations differ.

### Explicitly out of scope for Phase A

- No kernel edits, no coopmat2 bindings, no new shader family.
- No change to `DENSE_UBATCH_LADDER` or any default. A large ubatch win is a
  *finding*, not a patch: the ladder exists for RDNA residency reasons that
  have not been re-measured, and infr's tuning to date is AMD-derived.
- No claim about other NVIDIA parts. Every number here is one Ada card with
  20 GiB, running a model that occupies 78% of it. A 24 GiB or 48 GiB card
  would not hit observation 1 at this model size at all.
