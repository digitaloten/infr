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

The shaders declare `GL_EXT_float_e4m3` and `GL_EXT_bfloat16`. **Tested on this
host:** LunarG's noble apt SDK 1.4.313 (`glslc` reporting `shaderc v2023.8`,
built 2025-04-29) rejects `GL_EXT_float_e4m3` with
`'#extension' : extension not supported`, failing `quant_f8_row.comp`,
`native_gemm_f8cm_q8_0.comp` and `repack_q8_to_f8.comp`. The 1.4.357.1 tarball
SDK (`shaderc v2026.3`) compiles all three clean and builds the tree.

**Not tested:** Ubuntu 24.04's own `glslc` package. `apt-cache policy` gives its
candidate as `2023.8-1build1`, i.e. the same shaderc series as the LunarG build
that failed, so it is *likely* to fail identically — but it was never installed
on this host and that is an inference, not a measurement.

## Results

Same GGUF, same host, KV f16 on every row, `-ngl 999`, one engine at a time.
infr rows are `infr bench`; llama.cpp rows are `llama-bench` with matching
`-p/-n/-d/-r`. infr spreads are min–max over reps; llama.cpp reports ± stddev.

⚠ **Two deviations from this project's own measurement procedure, both
deliberate, both affecting how tightly these numbers should be read:**

1. **`infr compare` was not used**, though it exists for exactly this job. It
   shells out to the *system* `llama-bench`, and this host has none — the only
   llama.cpp on it is a container image (`/srv/prismml-llama.cpp/tools/llama-bench`
   is a source directory, not a binary). The matched `-p/-n/-d/-r` flags were
   therefore driven by hand against the containerised `llama-bench`, which is
   flag-for-flag equivalent but bypasses `compare`'s own bookkeeping.
2. **No cooldown between runs.** `playbook.md` rule 6 requires *"60s+ between
   back-to-back"* manual bench runs against thermal drift it puts at 2–8%. The
   session that produced this table ran engines back-to-back with no sleep. The
   measured gap is ~2.7× (206.4 vs 559.20 t/s at pp2048), far past that band, so the headline
   conclusion survives — but individual rows carry an unquantified thermal term
   and the 1.09× CUDA-vs-Vulkan margin in observation 3 is *within* it. Phase A
   inserts the cooldown.

Two further asymmetries, both **verified against the container builds actually
used** (`--help` on each image), not against a source tree:

3. **Flash attention was left at each engine's default and never pinned.** Both
   llama.cpp images report `-fa, --flash-attn <on|off|auto> (default: auto)`, so
   the reference rows ran with FA *auto-selected*, not disabled — and infr's own
   `infr compare` passes `-fa auto` to `llama-bench` for the same reason
   (`crates/infr-cli/src/main.rs:3212`), so the defaults agree. What is **not**
   recorded is what `auto` resolved to on this card in these runs; `llama-bench`
   only prints an `fa` column when the value is non-default, and the logs have
   none. Phase A pins `-fa on` explicitly on the llama.cpp side and records it,
   rather than trusting a resolution nobody logged. ⚠ Note for anyone repeating
   this against an older tree: `llama-bench` used to default flash attention to
   **off** (`cmd_params_defaults.flash_attn = { false }` in the vendored
   checkout), so a comparison built from that source *would* be FA-mismatched.
   The builds used here are not.
4. **Physical batch width differs at pp8192, and only there.** `llama-bench`'s
   `-ub` default is 512 in these builds and does not move with prompt length;
   infr dropped to 128 at pp8192 (observation 1). So **pp2048 is a like-for-like
   512-vs-512 comparison, and pp8192 is not** — 128 vs 512. The pp8192 row is
   therefore the *weaker* of the two as a kernel-quality statement, and the
   pp2048 row (0.37×) is the one to lean on until Phase A separates this.

| test            | infr Vulkan            | llama.cpp Vulkan (10331) | llama.cpp CUDA (10257) | infr ÷ llama Vk | infr ÷ llama CUDA |
| --------------- | ---------------------- | ------------------------ | ---------------------- | --------------- | ----------------- |
| pp2048          | 206.4 (spread 0.9%, r3) | 559.20 ± 0.71 (r3)      | 608.75 ± 0.64 (r3)     | **0.37×**       | 0.34×             |
| pp8192          | 185.1 (spread 0.6%, r2) | 541.48 ± 2.93 (r3)      | 591.63 ± 0.78 (r3)     | **0.34×**       | 0.31×             |
| tg128           | 11.4 (spread 0.2%, r3)  | 14.41 ± 0.01 (r3)       | 14.65 ± 0.01 (r3)      | **0.79×**       | 0.78×             |
| tg128 @ d8192 ¹ | 11.1 (spread 0.1%, r2)  | 13.96 ± 0.02 (r2)       | 14.24 ± 0.01 (r2)      | **0.80×**       | 0.78×             |

¹ infr's row needed an explicit `--ctx 8448`; the default session context on
this card is 8271 tokens, 50 short of the 8321 the shape requires (see
observation 6). llama.cpp's row ran at its own default.

Also measured: infr `pp512` = 209.6 t/s (1 rep). ⚠ That figure came from an
interactive smoke run whose output was **not captured to a log file** — unlike
every row in the table above, it cannot be re-read from an archived artifact. It
is reported for shape only; do not build on it. Phase A re-measures pp512 into a
log.

Per-row infr chunk heights, as reported by `infr bench` itself:

| row           | ubatch | submit splitter |
| ------------- | ------ | --------------- |
| pp2048        | 512    | split/166       |
| pp8192        | **128**| unlimited       |
| tg128         | 1024   | unlimited       |
| tg128 @ d8192 | 128    | unlimited       |

The prefill spreads here are 0.6–0.9%. `results.md` §"How reproducible is each
column?" reports batch prefill (`pp512`) as stable to **~1.5%** run to run on
the current tree — its older large-variance warning (34.5% worst on pp512, 31.7% on pp4@d4096) is explicitly
**retracted** there ("the old warning in this box no longer holds"), and the
residual softness it still records is confined to `pp4@d4096`, a 4-token
depth-primed shape not measured here. So the ~2.7× gap is wide
against a measurement whose own reproducibility is single-digit percent; noise
does not explain it.

### Device capabilities

infr's own probe line — ⚠ transcribed from the same uncaptured interactive
smoke run as the `pp512` figure, so it is **not re-readable from an archived
log** the way the llama.cpp block below it is. Phase A re-captures it:

```
GPU: NVIDIA RTX 4000 SFF Ada Generation | f16:y f16cm:y bf16:y bf16cm:y
     f8:n f8cm:n i8:y i8dot:y i8cm:n subgroup:32-32 sgp:32 shared:48KB
```

llama.cpp's, same card, same session (verbatim from the run log):

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
`radv_cooperative_matrix2_nv=true`, with a standing note to *"re-run it if a
future Mesa stabilizes/optimizes coopmat2"*. `coopmat2_test.rs:59-76` documents
the same gating.

On Ada, coopmat2 is a native driver path rather than a Mesa emulation, and
llama.cpp's Vulkan backend uses it as its primary matmul and attention route:
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:2999` selects `FA_COOPMAT2` when
`device->coopmat2`, with a dedicated shape-driven tuner at
`get_fa_tuning_params_coopmat2()` (`:2970`), plus coopmat2 branches at
`:3206`, `:3465`, `:3481`, `:4681`, `:4747`. infr's shader tree declares
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

⚠ **That rules out one alternative, not all of them.** "Not an API penalty"
does not yet establish "therefore infr's kernels", because a third category
sits between them: measurement asymmetries. The pp8192 batch-width mismatch
(deviation 4), the unlogged flash-attention resolution (deviation 3), and the
submit-splitter regime difference (observation 7) are none of them API
penalties and none of them kernel quality. Kernel difference is the leading
hypothesis and the rest of this document treats it as such — but it is not
earned until Phase A controls for those three.

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
the tile handicap, this card would be *worse* than AMD at depth, not flat.
d8192 is nowhere near where AMD's decay showed up.

**Why it was not measured further:** at 15.65 GiB resident there is only room
for ~8.3k tokens of f16 KV on this card (observation 6). Going deeper requires
quantized KV or a smaller model — either of which changes a variable that must
then be changed on the llama.cpp side too. Phase A does this deliberately rather
than by accident.

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
Note it names **pp512** specifically. An RTX 4000 SFF Ada is a far smaller part than the RTX 4090 the
review reasoned about, so it tests the small-GPU end of that argument.

⚠ **The underfill half of that rule cannot be evaluated on this host.** infr's
`compute_units` is populated only from `VK_AMD_shader_core_properties`
(`crates/infr-vulkan/src/lib.rs:2093`) and is therefore `0` on any NVIDIA card;
`nvidia-smi -q` on this host reports SM *clocks* but no SM count; and
`vulkaninfo` does not dump `VK_NV_shader_sm_builtins`, which is the struct that
would carry it. Obtaining the number means writing a probe, which Phase A does
not do. Phase A therefore returns the **attention-share** number only, and the
"visibly underfills the SM array" half stays open with that reason recorded —
it is not satisfied by a guessed SM count.

### 6. Default KV capacity clamps context to 8271 tokens

`infr bench -p 0 -n 128 -d 8192` fails with
`prompt 8193 + gen 128 exceeds the session KV capacity 8271`. The shape needs
8321 tokens (the engine primes `d + 1` = 8193, then generates 128); with
15.65 GiB resident the default session context lands at 8271, **50 short**. The
row in the table was produced with an explicit `--ctx 8448`. Not a performance
finding, but it is a sharp edge on a card this size — the default silently
refuses a shape `llama-bench` runs without comment.

### 7. The submit splitter fired on one row and not the other

Phase A treats submit splitting as one of three cost buckets, so the mechanism
belongs here rather than appearing for the first time in a stage table. The
splitter's cap is derived from the *previous* forward's measured
`elapsed_ns / dispatches` (`crates/infr-core/src/backend.rs:288-297`), so it is
not an independent knob — chunk height feeds it. The two prefill rows landed in
different regimes: pp2048 (ubatch 512) reports `submit split/166`, pp8192
(ubatch 128) reports `submit unlimited`.

⚠ **The available evidence argues against this being the main driver.** The row
that *was* split has the **better** ratio (0.37× at pp2048) and the unsplit row
the worse (0.34× at pp8192) — the opposite of "splitting is what hurts infr".
That is n=2 with chunk height confounded into it, so it is weak either way, and
it is precisely why Phase A stage 1 must hold the splitter fixed rather than
letting it float with chunk height.

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
of the wall time is going somewhere the hypothesis never claimed. `vulkan-review`
specified pp512 for that test but not an architecture, and this one is an
unlucky choice for it.

Nothing else in this document depends on the architecture — the prefill and
decode ratios are what they are for this model — but they should not be read as
"infr on NVIDIA" in general either. Phase A therefore adds a conventional dense
control model rather than re-using this one for the finding-#3 verdict.

## Reproducing

```bash
M=/path/to/unsloth-Qwen3.6-27B-Q4_K_M.gguf

# infr (Vulkan)
infr bench "$M" -p 2048 -n 0 -r 3
infr bench "$M" -p 8192 -n 0 -r 2
infr bench "$M" -p 0 -n 128 -r 3
infr bench "$M" -p 0 -n 128 -d 8192 -r 2 --ctx 8448   # --ctx needed, see observation 6

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

### Preconditions (owner-gated — not the executor's call)

The card is shared with a production llama.cpp server. Phase A stops that
server, so it runs **only** inside a window the service owner has approved, and
the executor must confirm all of the following before stage 0 does anything:

1. The window is approved, and its **duration is agreed and written down**. The
   script aborts remaining stages and jumps to restoration at the deadline
   rather than overrunning.
2. The server is idle: `curl -s http://10.10.5.1:8000/metrics` reports
   `llamacpp:requests_processing 0` **and** `llamacpp:requests_deferred 0`. Poll
   for up to 5 minutes; if it never goes idle, abort the window rather than
   killing live generations.
3. No other GPU consumer: `nvidia-smi --query-compute-apps=pid,name --format=csv`
   lists only the llama.cpp server.

### Survivability (the script must outlive the session and the operator)

- Launch it detached — `setsid nohup … &` or inside `tmux` — never as a
  foreground SSH command. A dropped connection otherwise SIGHUPs the run
  mid-window.
- `trap` on **`EXIT INT TERM HUP`**, not bare `EXIT`, or the restoration step
  does not fire on a hangup.
- `mkdir -p /srv/phaseA` as the **literal first command**, before anything
  redirects into it. Every stage writes there and the directory does not exist
  today; a bare `> /srv/phaseA/...` redirect against a missing directory fails
  *before* its command runs, which would skip stage 0's `docker stop` while
  later stages benchmark against a live server — silently invalid measurements
  on a still-serving box.
- **An independent watchdog, outside the script.** `timeout` is not sufficient
  and the plan must not pretend it is: a GPU wedge leaves a **D-state** process
  that ignores SIGKILL, so `timeout` blocks in its own `waitpid` and the
  "stage fails, log and continue" rule never runs. This host has **no
  `/dev/watchdog`**, `llamacpp` has `RestartPolicy=no`, and nothing self-heals.
  Run a second session that polls the stage's PID state and the window clock,
  and that pages a human when a stage exceeds its budget. Recovery from a real
  wedge is a manual power cycle followed by `docker start llamacpp`.
- Re-check "no other GPU consumer" **before every stage**, not just at stage 0.
  The systemd `Conflicts=` directives on this host only bind `systemctl`
  invocations; a second operator running `docker run --gpus all` directly is
  blocked by nothing and would silently poison every subsequent number.

### Restoration (read before stage 0 — this is the part that can hurt)

Capture the exact live configuration **before** stopping anything:

```bash
docker inspect llamacpp > /srv/llamacpp-state-$(date -u +%Y%m%dT%H%M%SZ).json
```

Stop with `docker stop llamacpp` — **never** `docker rm -f`. A plain stop keeps
the container, so restoration is `docker start llamacpp`, which replays the
exact original arguments with no chance of drift.

⚠ **Two things on this host destroy the container `docker start` depends on.
Both are named here so they are recognised and avoided:**

- **`/root/serve-llamacpp.sh`** — defaults `CTX=49152`, `PARALLEL=4`, while the
  live container runs `-c 90112 -np 6` (plus `--cache-type-k q8_0
  --cache-type-v q8_0 -fa on --metrics`). It begins with `docker rm -f llamacpp`.
  Run unmodified, it brings production back with 45% less context and a third
  fewer slots, silently and without error.
- **`systemctl start|restart|stop llamacpp`** (likewise `vllm`, `sglang`) —
  `/etc/systemd/system/llamacpp.service` exists and is plausibly named, but it
  is a **stale disabled template**: its `ExecStartPre` is
  `-/usr/bin/docker rm -f vllm sglang llamacpp`, and it points at a different
  port and a placeholder model. It is `disabled`/`inactive` and has nothing to
  do with the running container. Reaching for it during a failed restore — the
  exact moment it looks most like the right tool — force-removes the stopped
  container and converts a recoverable outage into a hand-reconstruction job.

**If `docker start llamacpp` itself fails**, do not improvise: rebuild the
`docker run` from the `docker inspect` JSON captured above (`Config.Cmd`,
`Config.Image`, `HostConfig.PortBindings`, `HostConfig.DeviceRequests`,
`HostConfig.Binds`), and verify the resulting `docker inspect` `Cmd` matches the
captured one argument for argument before declaring the service restored.

**If the host reboots mid-window**, nothing brings the service back on its own —
`RestartPolicy=no`, the systemd unit is disabled, and there is no watchdog. The
recovery step is `docker start llamacpp` (the container survives a reboot,
`AutoRemove=false`), then the same verification as stage 6.

### Stages

Everything is batched into one unattended script, because each window costs a
serving outage. Every device-touching command is wrapped in `timeout` — a hung
process, not a crashed one, is the failure that keeps production down
(`playbook.md` §"Hardware contention": *"a job killed mid-submit leaves an
unkillable D-state task holding the GPU until reboot"*). A stage that fails logs
and continues; only the deadline or a restoration failure ends the run early.

Fixed for every stage: `M=/srv/models/gguf/unsloth-Qwen3.6-27B-Q4_K_M.gguf`,
`INFR=/srv/infr/target/release/infr`, one run at a time, each run's stdout+stderr
captured to its own file `/srv/phaseA/<stage>-<shape>-ub<N>.log` and its profiler
output to `INFR_PROF_OUT=/srv/phaseA/<stage>-<shape>-ub<N>.prof`.

Rep counts are pinned per shape and match this document's table so the numbers
stay comparable: `-r 3` at pp2048/pp512/tg128, `-r 2` at pp8192. The profiling
stages (3 and 4) use `-r 1` — `benchmarking.md` runs the profiler that way, the
aggregate report is emitted once at process exit, and warmup runs unprofiled, so
extra reps buy nothing and cost window time.

| #   | stage                                                                                                       | timeout | measures                               |
| --- | ----------------------------------------------------------------------------------------------------------- | ------- | -------------------------------------- |
| 0   | `mkdir -p /srv/phaseA`; `docker inspect` capture; `docker stop llamacpp`; assert no compute apps listed       | 300s    | clean, restorable baseline             |
| 1   | 8 separate runs: `-u N` for N in 128, 256, 512, 1024, each × {`-p 2048 -n 0 -r 3`, `-p 8192 -n 0 -r 2`}      | 900s ea | chunk-height cost (observation 1)      |
| 1b  | `-p 512 -n 0 -r 3` at the stage-1 pp2048 winner, unprofiled                                                  | 900s    | the archived pp512 number §Results owes |
| 2   | `INFR_SUBMIT_DISPATCHES=0`, at pp2048 **and** pp8192, each at that shape's stage-1 winner                    | 900s ea | submit-splitter cost                   |
| 3   | `INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1` at pp8192 (stage-1 winner + unpinned default) and at **pp512**       | 1800s ea| per-kernel prefill attribution         |
| 3b  | stage 3 repeated on the **dense control model** (below)                                                      | 1800s ea| the finding-#3 verdict (observation 8)  |
| 4   | `INFR_SEAM_NO_REPLAY=1 INFR_PROF_OPS=1` at `-p 0 -n 128 -r 3` only (not the d8192 shape)                     | 1800s   | decode attribution (replay is untimed) |
| 4b  | decode depth sweep on the control model: `-p 0 -n 128 -d {4096, 16384, 32768}`, KV f16, both engines          | 1800s ea| whether decode decays with depth (obs 4)|
| 5   | llama.cpp Vulkan reference rows, same session, exact invocation below                                        | 3600s   | same-session baseline, not a recall    |
| 6   | `docker start llamacpp`; verify restoration (below)                                                          | 600s    | restoration observed, not assumed      |

**60s sleep between every run**, per `playbook.md` rule 6 — the thermal
discipline the existing table skipped. It is inside the time budget below, not
an optional extra.

**Stages 3b and 4b need a dense control model, and this host has none.** The
only GGUFs on the box are 27B-class (`unsloth-Qwen3.6-27B-Q4_K_M.gguf`,
`Qwen3.6-27B.q4_k_m.gguf`, `Ternary-Bonsai-27B-Q2_0.gguf`). Obtaining one is a
**precondition, not stage 0** — it needs outbound internet, takes minutes, and
must be finished while the server is still up, so it cannot sit inside the
window:

```bash
# BEFORE the window, server still running:
infr pull unsloth/Qwen3-8B-GGUF:Q4_K_M     # ~5 GiB, dense, standard attention every layer
# infr pull lands in the HF hub cache; the llama.cpp container mounts ONLY
# /srv/models, so the control GGUF must also be reachable there:
ln -sf "$(find /root/.cache/huggingface/hub -name 'Qwen3-8B-Q4_K_M.gguf' | head -1)" \
       /srv/models/gguf/control-Qwen3-8B-Q4_K_M.gguf
ls -lL /srv/models/gguf/control-Qwen3-8B-Q4_K_M.gguf   # verify the link resolves
```

Both engines then read the same file: infr via the cache path, llama.cpp via
`/srv/models/gguf/control-Qwen3-8B-Q4_K_M.gguf` inside the container. Stage 4b's
llama.cpp side is the same container invocation as stage 5 with
`-m /srv/models/gguf/control-Qwen3-8B-Q4_K_M.gguf -p 0 -n 128 -d 4096,16384,32768 -r 2 -fa on -ngl 999`.
Without the control model, stage 3b/4b do not run and the finding-#3 verdict stays
open — which is a valid outcome to report, but not one to discover at 3am.

**Time budget.** Stage 1 is 8 runs; from this document's own measured rates a
2048-token prefill at r3 is seconds of device time but each run pays a ~16 GiB
model load. Budget ~4 min per 27B run including load and cooldown → stage 1
~35 min, stage 2 ~10 min, stage 3 ~15 min, stage 4 ~5 min, stage 5 ~15 min,
plus control-model stages ~30 min → **roughly 2 hours** of GPU-exclusive time,
excluding the pull. Agree the window against that number, not against a guess.

**Stage 1 must record residency, not just throughput.** Passing `-u/--ubatch`
sets `ubatch_specified`, and `ubatch_candidates()`
(`crates/infr-llama/src/seam/mod.rs:824-831`) then collapses the candidate list
to the pinned value — the placement sweep is skipped entirely. Pinning *above*
what placement would have chosen (256/512/1024 at pp8192) can therefore push the
run into dense weight streaming rather than a taller resident chunk, which is a
PCIe-bound regime, not a chunk-height effect. Run each cell with
`INFR_PAGER_STATS=1` and record per cell whether it stayed resident or streamed.
Streaming cells are a **separate bucket** and must not be averaged into
"chunk-height cost". If the stats output turns out not to distinguish the dense
streaming path (it is documented for MoE expert pools), say so in the results
rather than assuming residency.

**"Stage-1 winner" means:** highest t/s among stage 1's *resident* cells for
that same shape. Per shape, never mixed across shapes — the existing data
already shows the two shapes settling at different heights (512 vs 128).

Stage 5's invocations. These differ from the ones that produced this document's
reference column in exactly one way — **`-fa on` is now pinned** rather than
left at the builds' `auto` default (deviation 3), so the resolution is recorded
instead of inferred. Use `-fa on`, not `-fa 1`: these builds take
`<on|off|auto>`, and `-fa 1` is the older tree's syntax.

```bash
timeout 3600 docker run --rm --gpus all -v /srv/models:/srv/models \
  --entrypoint /app/llama-bench ghcr.io/ggml-org/llama.cpp:full-vulkan \
  -m "$M" -p 2048,8192 -n 128 -r 3 -ngl 999 -fa on

# the fourth reference row — omitted from the command above because -d applies
# to every -p/-n pair, so it needs its own run
timeout 3600 docker run --rm --gpus all -v /srv/models:/srv/models \
  --entrypoint /app/llama-bench ghcr.io/ggml-org/llama.cpp:full-vulkan \
  -m "$M" -p 0 -n 128 -d 8192 -r 2 -ngl 999 -fa on
```

Because `-fa` moves between this document's table and Phase A, stage 5's rows
are the new baseline for every Phase A comparison; do **not** compare a Phase A
infr number against this document's `-fa auto` reference column.

⚠ `/srv/prismml-llama.cpp/tools/llama-bench` is a **source directory**, not a
binary; there is no system `llama-bench` on this host. The container above is
the only working path.

Stage 6 runs from a `trap` so it fires on abort as well as on completion, and
verifies rather than assumes: `docker inspect` reports `running` + `healthy`,
`curl -s -o /dev/null -w '%{http_code}' http://10.10.5.1:8000/v1/models`
returns 200, and both hold on **3 consecutive polls** 10s apart. Anything else
is an incident, not a completed window: leave the log, say so, do not report the
window as clean.

Stage 5 is re-run rather than quoted from this document because the two sets
would otherwise come from different sessions.

### Acceptance

Phase A succeeds when each of the three buckets carries a number and the
residual — the part only a kernel change can move — is quantified. It does
**not** succeed on a speed improvement; no default is changed in Phase A.

It must also return a verdict on [`vulkan-review.md`](vulkan-review.md) finding
#3 using that finding's own pre-registered thresholds, measured at **pp512** as
that finding specifies: attention share of prefill wall time **<20%** (tile gap
is not the lever) or **>40%** (register-O redesign worth scoping, *if* the
underfill condition also held). A value between 20% and 40% is reported as
landing between them, not rounded to whichever side suits a follow-up.

Two conditions on that verdict, both pre-registered here rather than argued
after the number lands:

- It is taken from the **dense control model** (stage 3b), not from `qwen35` —
  observation 8 explains why a 6-of-24-attention-layer model cannot settle a
  flash-tile question. If the control model was not obtained, the verdict is
  reported as **not established**, never as a pass by default.
- The **underfill** half stays unevaluated, with its reason recorded: no SM
  count is obtainable on this host (observation 5). A `>40%` reading therefore
  yields "threshold met, second condition unevaluable", not "redesign
  justified".

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

Two NVIDIA-relevant items from `vulkan-review.md` are **deliberately deferred**,
recorded here so they are not mistaken for settled:

- **Finding #2 / needs-measurement #3** — NVIDIA inherits AMD's decode int8
  dtype set (`{Q4K, Q6K, Q2K, Q4_0, Q5_0, Q5_1, IQ4_NL}`) unmeasured. Phase A
  measures decode *attribution*, not dtype-set alternatives, so this stays open.
  It is cheap to test later (`INFR_MMV_MW=1` A/B) and is the obvious Phase B
  candidate if decode rather than prefill becomes the target.
- **Finding #5** — `with_padded_dst`'s per-forward temp allocation is
  vendor-agnostic and applies here too; the review rates it Low impact and
  Phase A does not isolate it.
