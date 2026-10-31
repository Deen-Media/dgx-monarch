# Validation

This page supports the model support and performance claims with measured
results and known limits. For installation, start with [Quickstart](QUICKSTART.md).
For a workflow, use [Model support](MODELS.md); for speed comparisons, use
[Benchmarks](BENCHMARKS.md).

All hardware results below used two DGX Sparks with GB10 GPUs. They cover the
stated models, settings and software, not every combination supported by the
code. The records span different builds and do not constitute a complete
retest of the current release.

- [Installation and recovery](#dependency-and-setup-checks): fresh environments,
  two distributed renders, setup reruns and restoration.
- [Image models](#image-models) and [video/audio models](#video-and-audio-models):
  measured accuracy and untested configurations.
- [Memory calibration](#memory-calibration): measurements behind capacity checks.
- [Lifecycle](#lifecycle): cleanup, interruption and failure recovery.
- [Native RDMA](#native-rdma): selected-rail return checks and full-sampling limits; disabled by default.

## Monarch dependency migration

On October 12, 2026, an isolated official ARM64 TorchMonarch 0.7.0 wheel
loaded its native RDMA bindings with the existing Torch 2.12.0+cu132 stack.
The release wheel was built with Torch 2.14.1/CUDA 13.2; the check retained
the installed Torch version and dependencies. The empty native RDMA action
constructor succeeded without initializing a GPU or fabric manager.

Local CPU actor checks exercised child import identity, concurrent endpoint
dispatch, observer timeout and reobservation, lazy Future execution, terminal
Future timeout, and deferred ProcMesh stop. The GPUWorker dispatch-contract
probe also completed on a local CPU actor: status and cancellation remained
responsive during a busy operation, and the actor survived cancellation.
The production driver preparation and actor bootstrap helpers also launched
a real local CPU child after Monarch had already been imported. Driver and
child reported the same validated transport identity and native readback;
neither process imported Torch. The owned ProcMesh stopped successfully.
An explicit local OTLP check used the actual native provider and a temporary
loopback HTTP collector. The collector received a phase counter value of one
and a histogram with one observation totaling 0.25 seconds. After disabling
DGX metrics, 100 further local records were not exported; the DGX exporter
thread and test collector stopped. This does not claim shutdown of Monarch's
upstream provider or acceptance of a remote collector. Upstream log export
requires its separate explicit opt-in.
The full CPU suite passed 11,236 tests with 29 skips in 209 seconds on the
same overlay. Lint, full-package type checking, protected-test inventory and
refusal-ledger checks passed. Local Comfy CPU canaries checked 248 touchpoints,
86 templates and synthetic video export; these did not load learned models.
These checks used a scratch dependency overlay; the live environment was
unchanged. They establish native load and local semantics, not CUDA execution,
paired-fabric performance, render fidelity, or multi-host recovery acceptance.

### GPU checks for the 0.7.0 migration

**October 12, 2026:** candidate [source `ad199a9`](https://github.com/Deen-Media/dgx-monarch/commit/ad199a9b7d7016cfb106a060bc9a48d5c7d851b9)
with TorchMonarch 0.7.0 was compared with the 0.6.0 baseline at
[source `7ba25b7`](https://github.com/Deen-Media/dgx-monarch/commit/7ba25b7512310cf89dff245b950d8f794c9d233b).
Both used Torch 2.12.0+cu132 and ComfyUI
[`e8665ca`](https://github.com/Comfy-Org/ComfyUI/commit/e8665ca8e4089ab917567cdb491308e1397663d4).
These are checks of the complete upgrade configuration on the two Sparks,
with actor messaging for latent returns. They do not establish native RDMA
acceptance or isolate the effect of the Monarch dependency.

**October 13, 2026 UTC:** the shipped
[`dgx-monarch-krea2-t2i.json`](../example_workflows/dgx-monarch-krea2-t2i.json)
workflow completed without recipe changes. Only UI-to-API conversion and
private config/output locations differed. It used Turbo FP8 scaled, Qwen3VL
4B BF16, the Qwen image VAE, the original mouse prompt and blank negative,
1536×1536, seed 0, 8 steps, CFG 1 and euler_cfg_pp/simple. Auto mode selected
Ulysses 2 and the SAGE_AUTO policy through table row 11. The graph retained
its original first_use Gate and automatic residency settings.

The first-use slab cross-residency Gate passed with zero latent difference
across three two-step proof renders, taking 50.7 seconds. Its exact artifact,
source and capability context passed the public ledger integrity lookup.
After the requested render, both workers reported full loading with stock
cudaMalloc backing. One uncached 1536×1536 PNG decoded completely and showed
the intended mouse in visual review. Typed recycle, owned driver exit and
paired passive idle checks completed with persistent Workers unchanged.

The whole-prompt history interval was 145.737 seconds; the controller measured
146.134 seconds and the main eight-step sampling log reported 38.32 seconds.
These include different phases and establish no comparative speedup. SAGE_AUTO
selection is bound by context and logs, not a per-kernel trace. This is
functional acceptance of the shipped workflow and its recorded Gate; it is
not a stock-fidelity or cross-attention-kernel comparison.

The shipped
[`dgx-monarch-minimax-h3-mp4.json`](../example_workflows/dgx-monarch-minimax-h3-mp4.json)
also completed on October 13 UTC without recipe changes: 1344×768, 124 frames
at 24 fps, seed 0, 20 steps, CFG 1, res_multistep/simple, TORCH_FLASH, pruned INT8 DiT,
NVFP4 Qwen3VL encoder and both original VAEs. Only API conversion, consistent
node identifiers and private config/output locations differed. Auto selected
Ulysses 2; first_use and automatic residency remained enabled. Both workers
reported full loading and stock cudaMalloc backing after sampling.

The resulting MP4 passed complete CPU decoding: 124 H.264 frames at the exact
canvas/rate and AAC 32-kHz stereo audio, with 165,888 decoded sample frames
(165,334 expected plus 554 AAC tail samples). Four preview frames showed a
plausible glassblowing sequence; audio quality was not assessed by listening.
The two first-use proof renders took 133.4 seconds and remained
INCONCLUSIVE/no_material, granting no optimization PASS. Existing Sol-only
consent was carried unchanged; no actual use was observed in retained stamps
or logs, and raw per-rank waiver payloads were not retained. Typed recycle,
owned driver exit and subsequent paired idle checks completed with persistent
Workers unchanged.

The prompt history interval was 857.489 seconds, controller 858.884 seconds and
main sampling log 555.84 seconds; these are observations of different phases,
not a matched 0.6 comparison or a Monarch speedup. The 613 retained in-prompt
head samples peaked at 2,093 MHz, 63.55 W and 80°C, with minimum MemAvailable
43,578,156 KiB. Their largest gap was 205.28 seconds, so neither continuous cap
coverage nor an exact peak-memory claim follows. The post-run boot, cap unit
and service state matched the pre-run record.

The adapted controlled image recipe used Krea2 Turbo BF16, Qwen3VL 4B BF16 and the Qwen
image VAE at 1024×1024, 8 steps, CFG 1, Euler/simple, native shift 1.15 and
explicit TORCH_FLASH. A fixed prompt and zeroed negative conditioning were
used without prompt enhancement, with the identity Gate disabled for these
controlled comparisons. This was not the unchanged shipped Krea2 workflow.
Seeds 10041-10044 covered warmup and three
measured renders. Stock ComfyUI, DGX local and two-GPU Ulysses outputs matched
in decoded PNG pixels. Separate-process null controls covered stock only.
A candidate-only 1040×1040, one-step control also matched across those three
execution modes; it is not an eight-step or cross-version padding result.

The retained core set contains **31 renders and 57 exact pixel comparisons**,
checked with the unchanged [pixel comparator](../benchmark/sweep/compare.py).
Those pairs reuse the same outputs; they are not 57 independent experiments.
The comparisons comprise 18 pairings within the same topology across the
upgrade, 32 pairings across topologies, four stock process-null pairings and
three padding pairings. Decoded-pixel equality does not assert equality of
unsaved intermediate tensors or untested models and settings.

Warm whole-prompt history medians, excluding warmup, padding and status
observation, were:

| Execution mode | Baseline 0.6.0 | Candidate 0.7.0 |
|---|---:|---:|
| Stock ComfyUI | 17.547 s | 15.993 s |
| DGX local | 15.818 s | 15.820 s |
| Two-GPU Ulysses | 10.556 s | 10.627 s |

Each median uses three seeds. These are `execution_start` to
`execution_success` intervals, not kernel times or polling-inclusive controller
wall times. Stock timing also changed. The observations do not establish a
Monarch speedup or a statistically resolved performance regression.

A targeted interruption correctly refused recycling during active work and
interrupted the selected prompt, but retained one abandoned sample and did
not reach clean settlement within the observation budget. **Clean cancellation
acceptance remains unproven.** An explicit typed recycle stopped the owned
actors; fresh actors then completed an uncached recovery render in the same
driver. Its pixels matched the stock, local and Ulysses controls. The final
typed recycle and owned driver close completed, with the persistent Workers
preserved. This is explicit recovery evidence, not a clean-cancellation PASS.

A separate Krea2 RAW BF16 diagnostic at 1024×1024, 8 steps, CFG 1,
Euler CFG++/simple, TORCH_FLASH and `uly2+fsdp` passed the exact-context
`fsdp_clean_reload` Gate with identical latents and maximum absolute
difference 0. Its report and terminal ledger row agreed. This short RAW
recipe establishes that clean-reload proof only; it is not the representative
Turbo recipe, an external stock-pixel comparison or RAW quality acceptance.

Pipeline depths 1/2 and Fleet produced matching batched decoded images for
seeds 10042/10043, but differed from the single-image controls: NRMS
0.005310/0.006332 and maximum 8-bit pixel differences 24/44. A follow-up saved
the FP32 batch and verified each slice against earlier sequential and depth-1
actor hashes. Splitting that latent before VAE decoding matched stock, local
and Ulysses pixels exactly; decoding it together reproduced the earlier
pipeline/Fleet outputs in six comparisons. The difference is downstream of
matching sampled latent bytes, in the decode batch path, shape or associated
state. No specific kernel cause or blanket batched-pixel equality is established.

A reduced-canvas, adapted H3 workflow also completed: pruned INT8 DiT, NVFP4 Qwen3VL encoder,
video/audio VAEs, 512×320, 124 frames, 24 fps, 20 steps, CFG 1,
res_multistep/simple, TORCH_FLASH, Ulysses and seed 42. Complete CPU decoding
verified all H.264 frames and AAC 32-kHz stereo audio. Decoded audio contained
165,888 sample frames, including 554 AAC tail samples beyond ComfyUI's
165,334-sample input. Four preview frames showed plausible glassblowing;
audio quality was not assessed by listening.

The single H3 controller interval was 212.219 seconds, including loading and
two identity checks totaling about 23.2 seconds; observed sampling was
45.32 seconds. Its Gate remained INCONCLUSIVE/no_material with normal stock
residency, granting no optimization PASS. Existing Sol-only consent was carried
unchanged, with no actual use observed; raw actor waiver payloads were not
retained. No consent was created or changed. Head clocks stayed at or below
2,093 MHz in 192 prompt-interval samples against a 2,100-MHz cap; gaps reached
15.068 seconds, so this is not continuous coverage. Typed recycle, owned driver
exit and subsequent passive idle checks completed with persistent Workers
preserved. This is functional AV and cleanup evidence for the adapted
512×320 recipe, not unchanged shipped-workflow acceptance or a matched 0.6.0
fidelity or performance comparison.

Historical H3 timings used substantially different settings:

| Record | Shape and DiT | Pair timing and clock |
|---|---|---|
| September 0.6.0 t2va | 1344×768×124, FP8 scaled | 690.6 s cold / 564.5 s warm, whole-prompt polling |
| September 0.6.0 guide | 1344×768×124, INT8 ConvRot, guide at frame 62 | 772.8 s cold / 636.6 s warm, whole-prompt polling |
| Adapted 0.7.0 MP4 run | 512×320×124, INT8 ConvRot | 211.927 s prompt history / 212.219 s controller; 45.32 s sampling observation |
| Shipped 0.7.0 MP4 workflow | 1344×768×124, INT8 ConvRot, no guide | 857.489 s prompt history / 858.884 s controller; 555.84 s sampling observation |

All used 20 steps, CFG 1, res_multistep/simple and Torch Flash with Ulysses.
The historical rows had 6.3 times as many video pixels per frame as the
adapted 512×320 run. The shipped run matches their canvas, but model/guide,
prompt/seed, software and output paths remain unmatched; both pair records remained CHECK because their
audio comparator was missing. Exact historical graph/prompt/seed and full
model payload identities were not retained in the published records. No ratio
across these rows isolates Monarch or supplies a matched upgrade speedup.
See the [historical video measurements](BENCHMARKS.md#campaign-render-times-video-2026-09-27-to-2026-09-28-head-capped).

[Native return qualification](#native-rdma) has separate scope; the earlier
CPU-only and historical model records retain their original limits.

### Ownership and actor-start review (2026-10-13)

**October 13, 2026:** the review that
[TROUBLESHOOTING #112](TROUBLESHOOTING.md#112-the-torchmonarch-canary-reports-semantic-drift-against-a-new-release)
asks for compared two TorchMonarch 0.7.0 changes with the integration's
ownership rules: RDMA buffer drops that start at once, and shutdown timeouts
that cancel the rest of a stop. Ownership order is unchanged. The driver drops
its registration handles once after a completed read, sends the ACK only after
every drop is confirmed, and never retries a drop that failed or timed out;
since a timed-out drop may still be in flight, a retry could release it twice.
A timed-out `ProcMesh.stop` no longer finishes its remaining steps. In local
CPU runs on the pinned wheel, a zero deadline cancelled the stop before its
native step in 15 of 15 runs, and the procs kept answering until a fresh stop.
With a 10 ms deadline the Future still timed out in 10 of 10 runs, but the
native stop had already started and the procs were gone one second later. The
outcome of a timed-out stop is therefore unknown, as the integration already
assumes: it refuses another stop and any reuse, so procs that outlive a
cancelled stop are left to the actor reaper. The safety ladder stays ordered:
the reaper's 300 s lease grace, less one 30 s renewal interval, still exceeds
the 240 s teardown-token authority. The CPU smoke now asserts the zero-deadline
case on every canary run. It issues a second stop only to clean up its own
procs; production never does.

**October 13, 2026:** actor-start readiness was read against TorchMonarch
0.7.0's 30 s hold on messages to an actor that is still starting
(`HYPERACTOR_PENDING_ACTOR_DELIVERY_TIMEOUT`, upstream default). With a warm
page cache, a fresh driver's first queue built a two-actor fleet, one actor per
Spark, with the head under its 2,100 MHz GPU clock cap. The driver's
`readiness` phase recorded four observations totaling 3.921 s, the longest
3.519 s; `spawn` took at most 0.001 s and `attach` 0.005 s. The head actor's
journal shows about 5 s from its device line to the end of its custom-node
preload. The first held message, the readiness `status` call, answered in
about a tenth of the hold. No policy setting for the hold was added, because
the measured time is not within a factor of two of 30 s. A cold-cache start was
measured the same day, with dev `960d999` deployed. `drop_caches` ran on both
Sparks with the driver down, which left `Cached` at 1.3 GiB on the head and
0.7 GiB on the sibling at launch; the head was again under its clock cap. The
driver was ready after 10 s (5 s warm). The fresh two-actor fleet's
`readiness` phase recorded four observations totaling 4.622 s, the longest
4.151 s (3.519 s warm); `spawn` took at most 0.002 s and `attach` 0.016 s. The
render took 70 s in all, with no re-gate. The actor journals show 5 s on the
head and 6 s on the sibling from `Device: cuda:0` to `comfy bootstrapped`.
The first held message answered in about one seventh of the 30 s hold. The
decision stands: no policy setting for the hold.

## Onboarding and operator acceptance

**2026-10-11, source `09957d971441c59d4605092c064886c66841eb6e`, two DGX
Sparks with GB10 GPUs.** These checks preceded the v1.1.0 release commit;
they are not an exact-release-commit hardware retest.

Verified update completed with Doctor, an owned cluster smoke and exact
two-rank source readback. Worker startup identities and source files were
checked separately; persistent coordinator loaded imports were not directly
inspected. The fresh client reused existing dependencies: this was not a
clean-OS or independent dependency installation.

Optional Sol installation changed only `apache-tvm-ffi`, from 0.1.13.post3 to
0.1.14.post1, on both hosts; protected runtime file hashes remained unchanged.
Both GPUs executed `cute_sm120` on a 4,096-token BF16 dense control, with
relative L2 error **0.00298337** against PyTorch SDPA. This establishes kernel
execution, not sparse-attention or model fidelity.

The MiniMax MP4 workflow completed with **TORCH_FLASH, uly2, 512×320,
124 frames and 24 fps**. H.264 video and AAC audio streams were verified;
complete CPU decoding succeeded. The run carried an existing accuracy waiver;
no new waiver was created. Its identity gate had no LoRA/slab configuration to compare and remained
**INCONCLUSIVE**, not PASS.
This was functional output acceptance without an exact-fidelity comparison.

Headless rendering of actual Textual widgets showed **Health READY /
Activity RENDERING / Maintenance Wait**, then READY / IDLE / None. The combined
launcher's terminal handshake succeeded; native windows were not visually
inspected. Owned mesh recycle and driver/dashboard cleanup completed;
persistent Worker identities stayed unchanged and final idle checks passed.
Native latent-return RDMA was **NOT RUN / HOLD** in this October 11 record;
see the later [selected-rail checks](#native-rdma).

<a id="release-1-4-0-checks"></a>

## Release 1.4.0 checks

**2026-10-28, two DGX Sparks with GB10 GPUs.** The hardware evidence for this
release sits in the family and feature records below, each naming the commit
it ran on. The release commit adds documentation, the version and the SCAIL
refusal code over those commits; it is not an exact-release-commit hardware
retest.

- MOSS-Image text attention bias on ComfyUI `426f5eea`: the 2026-10-27 legs
  ran dgx-monarch `3438874fbf6c`
  ([record](#moss-image-bias-on-comfyui-426f5eea)). On 2026-10-28 the same
  legs ran again on `0511b0ff2142` and reproduced every row: all 40 latent
  and pixel arrays byte-identical to the 2026-10-27 record and the refusal
  text byte-identical, now raised at the driver's model patch preflight; 83
  comparisons, 77 PASS, 3 recorded, 3 harness-equivalent; the first Spark
  capped at 2100 MHz, highest sampled SM clock 2093 MHz.
- Wan SCAIL2 at world 2 on `e11889da5986`
  ([record](#wan-scail2-degree-2-probe-2026-10-28)): uly2 and ring2
  byte-identical to one GPU at one step, cfg2 not. The cfg and ring refusals
  that followed are CPU-tested; no hardware leg has observed them: **NOT RUN**.
- Studio: the 2026-10-28 browser round of 26 rows on the `0511b0ff2142`
  bundle, with the three faults it found fixed in the next commit
  ([STUDIO_INTERFACE_QA](STUDIO_INTERFACE_QA.md#browser-evidence-2026-10-28)).
- Verified updates of both Workers on 2026-10-28 through `9500c57`,
  `c852244`, `0511b0f` and `e11889d`, with `dgxm doctor` at 44 checks and 0
  failures after each.
- **NOT RUN:** the Worker-side bias scan, an all-zero bias, a one-prompt bias
  under uly2+cfg2, SCAIL Preview on cfg2, every layout at world 4, and the
  full sweep runbook; only the cells the records name ran.
- Build and install, on the release commit, in a disposable Python 3.12 venv
  that holds only `build` and `pip`: `python -m build` wrote the sdist and
  the wheel (521 files, the `dgxm` and `dgxm-studio` console scripts and the
  Studio bundle inside). The wheel installed with `--no-deps` and reported
  1.4.0, as did an editable install of a fresh clone of the commit after it;
  `dgxm --help` answered with the runtime dependencies of the rig's Monarch
  venv on the path. The full INSTALL recipe, which rebuilds the xFuser
  compatibility wheel against a ComfyUI interpreter, was **NOT RUN** for this
  release; the rig's own venv already carries that wheel.

## Comparison method

- **Native reference:** an independent stock ComfyUI workflow.
- **Local reference:** the corresponding DGX Monarch workflow on one GPU.
- **DP2 reference:** sample 0 of a two-sample data-parallel run using stock
  model forward calls. This is not automatically an independent native reference.
- **Resident comparison:** FSDP or a memory-saving load compared with ordinary
  residency. Passing this comparison does not establish native-reference fidelity.
- **Cross-rank agreement:** workers returned the same result. They can agree
  while both differ from a reference, so this alone does not prove accuracy.

One-step fidelity tests use normalized RMS difference (NRMS), usually with a
0.10 limit. Latent and decoded-pixel measurements are separate surfaces and
must not be substituted for each other. "Identical" below means equality of
the compared arrays. Full-render completion and warm timing are separate from
one-step accuracy. Multi-step differences are descriptive unless a test
explicitly declares a threshold; no general decoded-video or audio quality
threshold is claimed.

September campaign runs used torch 2.12.0+cu132, torchmonarch 0.6.0 and world 2.
The first Spark was capped at 2100 MHz and the second was unrestricted.
Older records stamped NCCL 2.29.7, the torch build version; the loaded library
was 2.30.7. The pure-Ulysses image-family comparisons used NCCL 2.32.3,
with a separate comparison against 2.30.7. Software and recipe details for
the campaign records are in [Campaign results](CAMPAIGN_RESULTS.md).

A result applies to the named checkpoint, precision, topology, inputs and
comparison. It does not cover every workflow in a family. CPU tests, source
inspection, admission rules and successful loads are not hardware fidelity
results. `CHECK`, `INCONCLUSIVE` and a capacity refusal are not accuracy passes.
Native latent RDMA qualification is recorded [separately](#native-rdma);
model-render comparisons do not qualify that path.

## Campaign summary

The main campaign retained 2,597 primary records and 15 retries from 2,632
selected cases. Another 35 selected cases did not run because their reference
failed. Outcomes were 2,186 PASS, 86 PASS-capacity, 153 CHECK, 116 non-crash
FINDING and 56 crashes. PASS-capacity means an expected refusal, not a render.
A follow-on retained 98 primary records: 27 PASS, 59 PASS-capacity, 3 CHECK,
9 FINDING and no crashes. Its findings were refusals where the matrix expected
a render; its CHECKs lacked usable references.

Coverage was incomplete: 3,009 of 4,560 unselected cases had no record on any
tested version. Only 26 selected video cases and their local references ran;
the full Sol-Attn wave did not run. Physical-output auditing was incomplete,
and historical ownership and cleanup receipts were not recorded. Individual
results below do not remove those limitations or replace the original verdicts.
See [Campaign results](CAMPAIGN_RESULTS.md) for retained measurements.

**2026-10-27, external reproduction.** A four-Spark owner reran the cells of
the published one-GPU and two-node campaign tables on their own Sparks, with
no clock cap, and reported the results on the NVIDIA Developer Forums
([post 17](https://forums.developer.nvidia.com/t/dgx-monarch-dual-spark-rendering-w-comfyui/385450/17)).
They compared 109 of the 117 published cells
([Benchmarks](BENCHMARKS.md#campaign-render-times-stills-2026-09-12-to-2026-09-28-head-capped)):
108 PASS and 1 CHECK, LTX 2.5 two-stage on `auto` at one-step NRMS 0.120
against the 0.10 limit. Their times ran a median 5.5 percent faster than
published, with 94 of the 109 cells within 10 percent and 104 within
15 percent. This project has not reproduced their runs.
[Benchmarks](BENCHMARKS.md#community-four-node-results-2026-10-27) records
them with their four-node results.

## Image models

<a id="evidence-krea2-raw-turbo"></a>

### Krea2 RAW and Turbo

Pure-Ulysses tests on TORCH_FLASH matched DP2 latents exactly:
RAW BF16 at 1536 and 1448 with odd/even text and image rows, batch 2 and the
template; RAW INT8 ConvRot and Turbo FP8-scaled on the template. This does
not cover the SAGE route (row 11), which auto chooses for FP8 above 2.36 MP,
and from 1.2 MP at CFG 1 under a plain sampler. The unchanged
ring2 control read NRMS 0.012292.

Resident-versus-FSDP comparisons passed with identical outputs for RAW
and Turbo BF16, FP8-scaled and, after its wrap repair, INT8 ConvRot. All 60
non-LoRA Krea2 FSDP records in the September campaign passed. Krea2 LoRA/FSDP
is different: the follow-on refused quantized shards and incompatible
FP32-stored/live-BF16 parameters before baking; it did not establish accuracy for those combinations.

NVFP4 Ulysses measured one-step NRMS 0.014 without shared activation scale
and 0.011 with it. A separate 512-square CFG-1 native control measured
0.018844. Padding can still raise a shared scale relative to an unpadded
one-GPU tensor. No hook-cost or non-Krea/non-Chroma NVFP4 control was measured.

Auto has no small-image cfg2 rule; cfg2 is explicit. Plain samplers at CFG 1
have no unconditional batch to split, while cfg++ samplers retain it. Unequal
cfg2 prompts use padding and a key mask; combined sequence/CFG topologies
refuse that asymmetric pair. ConditioningZeroOut is real conditioning, not
padding. Reference-image workflows refuse sequence or CFG splitting and
must use `single`. Padded ring/hybrid requests refuse unless an explicit
accuracy waiver is recorded. INT8 LoRA swaps measured 10.9/9.7/9.7 seconds.

<a id="evidence-kroma"></a>

### Kroma

**2026-10-12, source `328c16779a68feda57544cfc1ee43d1f314d73a4`, ComfyUI
`e8665ca8e4089ab917567cdb491308e1397663d4`.** Driver and actors used source
manifest `9af02423527ceb23064760a15350f81331ec0833797d6d3e95b9deba97619229`.
Pure Ulysses on two NVIDIA GB10 GPUs with TORCH_FLASH was compared with
independent stock one-GPU ComfyUI, using each checkpoint's own matched reference.

The full graphs used batch 1, 1024×1024, 8 Euler/simple steps, CFG 1,
denoise 1, native Flux-style mu 1.15 and ConditioningZeroOut. The BF16
upscale graph then decoded, scaled pixels 2x with Lanczos, re-encoded with
the same VAE and refined at 2048×2048 for 4 steps at denoise 0.2 and CFG 1.

| Graph or control | Stock comparison | Cross-rank latents |
|---|---|---|
| BF16, full 8 steps at 1024×1024 | Identical conditioning, latents, floating decoded images and PNG pixels; max abs 0 | Identical |
| Local INT8, full 8 steps at 1024×1024 | Identical conditioning, latents, floating decoded images and PNG pixels; max abs 0 | Identical |
| BF16 upscale, initial 8-step stage at 1024×1024 | Identical latents, floating decoded images and PNG pixels; max abs 0 | Identical |
| BF16 upscale, 4-step refinement at 2048×2048 | Identical latents, floating decoded images and PNG pixels; max abs 0 | Identical |
| Kroma BF16 padding, one step at 1040×1040 | Identical conditioning, latents, floating decoded images and PNG pixels; PNG NRMS 0 | Identical |
| Krea2 RAW shared-adapter control, one step at 1024×1024 | Identical conditioning, latents, floating decoded images and PNG pixels; max abs 0 | Identical |

The full-workflow criterion is exact equality of the compared tensors and
pixels. The one-step Kroma padding control uses the separately declared
decoded-PNG NRMS limit of 0.10: RMS(pixel difference) / RMS(reference pixels),
with a denominator of 1 for an all-zero reference. The Krea2 RAW control retains its predeclared
exact comparison. The padding threshold does not apply to RAW or full renders.
PNG comparisons use decoded integer pixel arrays, not PNG container hashes.

| Graph or control | Seed | Positive prompt and suffix | Text rows |
|---|---:|---|---:|
| BF16 | 2026101201 | Teapot prompt from the BF16 template plus ` kroma-20261012-bf16.` | 119 |
| Local INT8 | 2026101202 | Teapot prompt from the INT8 template plus ` kroma-20261012-int8.` | 117 |
| BF16 upscale | 2026101203, both stages | Teapot prompt from the upscale template plus ` kroma-20261012-upscale.` | 117 |
| Padding control | 2026101201 | Bicycle prompt below plus ` kroma-20261012-bf16.` | 73 |
| Krea2 RAW control | 2026101204 | Teapot prompt from the BF16 template plus ` kroma-20261012-bf16.` | 119 |

The templates at the tested source commit define the full graph and teapot
prompt. Each reference and candidate used the same prompt, seed, conditioning,
checkpoint and VAE. The padding control had 4,225 image rows and 73 text rows;
it qualifies that one-step comparison, not full 8-step padded rendering.
The independent bicycle prompt, before the table's suffix, was:

> A documentary photograph of a red bicycle leaning against a pale stone wall after the rain. A woven basket on the handlebars holds yellow flowers. Afternoon light casts soft shadows across the wet cobblestones, with crisp metal spokes and natural colors. Fine detail.

The RAW control derived from the BF16 graph by changing only the diffusion
checkpoint to `krea2_raw_bf16.safetensors`, the step count to 1 and the seed
to 2026101204. It checks the shared adapter correction, not a recommended
RAW generation recipe.

On prior source `c988d2c1ae7e7e7fd78cfb116db411a6e898f187`, the first
BF16 8-step comparison differed from stock (decoded-PNG NRMS 0.025169).
Preserving the complete stock row shape through Krea2's final projection
removed that BF16 difference in the fixed-source rerun above.
The earlier result does not qualify the fixed source or other workflows.
All five fixed-source comparisons produced finite outputs of the expected
shape with source-matched ranks. Their scoped identity gates passed and
owned actor cleanup was confirmed; these checks are separate from fidelity.

Tested files and SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| `kroma-v0.3.1-turbo-opd.safetensors` | `3cfb5bc847c804ea8e8f894677f7e16ebe4c9ff0f978896cc5e7eb1e64fc7d82` |
| Local `kromaKrea2_opd.safetensors` | `96043ecf3e920288fbf6f13ca81ba3629fe02e9f8729ed116382ebe4978e7839` |
| Control `krea2_raw_bf16.safetensors` | `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7` |
| Local `qwen3vl_4b_bf16.safetensors` | `341d8e1bdd34d3c5119b5b768badf87ed2a1d663e79d331b85b4c1fc4f7bf498` |
| `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

The [artifact manifest](../example_workflows/artifacts.toml) separately lists
the pinned upstream encoder download, whose SHA-256 is
`36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34`.
Its file hash differs from the tested local encoder's, but a 2026-10-12
per-tensor comparison found all 713 tensors byte-identical, with equal dtypes
and shapes and none missing on either side, after the `model.language_model.`
to `model.` key rename that ComfyUI's `krea2` CLIP loader applies. Only the
header layout and metadata differ, so these results cover the pinned
download's encoder weights. The local INT8 conversion's upstream version and
provenance remain unverified. Comparing it with itself on one GPU does not
measure quantization error against BF16.

These resident results do not qualify LoRA, FSDP, cfg2, ring, hybrid topologies,
other attention kernels, reference-image conditioning, other quantizations
or unlisted dimensions. A Kroma BF16 file reads as family krea2, but Krea2's
cfg2 row 15 excludes Kroma: measured on 2026-10-16, cfg2 on the shipped
graph's zero-conditioning negative did not match one GPU, so `auto` keeps
Ulysses for Kroma at every CFG and sampler ([Kroma cfg2](#kroma-cfg2),
[cfg2 crossover rows](#campaign-k-cfg2)). The shipped 1024×1024 graphs, the
2048×2048 refinement and the INT8 file keep Ulysses as well.

#### Kroma manual FSDP

**2026-10-12, source `c334e4dca68297f7dadbc53e8982680ce04fbd6d`.** The
runtime manifest, ComfyUI revision and artifact hashes are the same as the
resident record above. Two NVIDIA GB10 GPUs with TORCH_FLASH used the shipped
graphs at 1024×1024, 8 Euler/simple steps, CFG 1, denoise 1 and native mu
1.15. BF16 refinement continued at 2048×2048 for 4 steps at denoise 0.2
following the same Lanczos pixel-upscale and VAE re-encode sequence. Each
candidate had an independent, matched stock one-GPU reference at
the same image batch size. The acceptance criterion remained exact equality.

| Checkpoint | Topology / batch | Stock comparison | PNG NRMS | Maximum PNG channel difference | Result |
|---|---|---|---:|---:|---|
| BF16 | `uly2+fsdp` / 1 | Conditioning, latents, floating decoded images and PNG pixels identical | 0 | 0 | Qualified |
| Local INT8 | `uly2+fsdp` / 1 | Conditioning, latents, floating decoded images and PNG pixels identical | 0 | 0 | Qualified |
| BF16 2x refinement, both stages | `uly2+fsdp` / 1 | Conditioning, both sampler latents, floating decoded images and PNG pixels identical | 0 | 0 | Qualified |
| BF16 | `dp2+fsdp` / 2 | Conditioning identical; latents and pixels differ | 0.0733553770 (7.335538%) | 254 | **CHECK** |
| Local INT8 | `dp2+fsdp` / 2 | Conditioning identical; latents and pixels differ | 0.0857061815 (8.570618%) | 232 | **CHECK** |
| BF16 2x refinement | `dp2+fsdp` / initial 2 | First stage completed; native VAE re-encode produced latent batch 1 and DP2 refused refinement | n/a | n/a | **BLOCKED**, no end-to-end comparison |

PNG NRMS is RMS(decoded pixel difference) / RMS(reference pixels), with a
denominator of 1 for an all-zero reference. PNG metadata is excluded. The
nonzero rows failed the exact criterion; no tolerance was substituted.
Both batch-2 T2I rows passed source/artifact checks, correctly assembled their DP
slices, and passed fresh FSDP clean-reload readiness proofs and the normal
first-use Gate. Those checks establish execution, not stock fidelity.
Within-group rank identity passed for `uly2+fsdp`; it is not applicable across
the two independent singleton DP groups in `dp2+fsdp`. Owned actor cleanup
completed for every candidate, including the blocked refinement. The final
paired idle/ownership check passed. The blocked run had a fresh first-stage
FSDP proof, not a passing refinement.

| Checkpoint / image batch | Seed | Suffix appended to its template's teapot prompt | Text rows |
|---|---:|---|---:|
| BF16 / 1 | 2026101201 | ` kroma-fsdp-20261012-bf16-b1.` | 124 |
| Local INT8 / 1 | 2026101202 | ` kroma-fsdp-20261012-int8-b1.` | 122 |
| BF16 2x refinement / 1 | 2026101203, both stages | ` kroma-fsdp-20261012-upscale-b1.` | 122 |
| BF16 / 2 | 2026101201 | ` kroma-fsdp-20261012-bf16-b2.` | 124 |
| Local INT8 / 2 | 2026101202 | ` kroma-fsdp-20261012-int8-b2.` | 122 |
| BF16 2x refinement / initial 2 | 2026101203, both stages requested | ` kroma-fsdp-20261012-upscale-b2.` | 122 |

Conditioning had shape `[1, text_rows, 30720]`; batch 2 refers to generated
images, not two separately encoded prompts. All pair inputs were identical.
The salts specify those inputs; they are not evidence of a fresh Gate ceremony.
The recorded normal first-use `fsdp_clean_reload` proof establishes the fresh
FSDP reload.

A BF16 resident `dp2` control reused the batch-2 inputs and produced latents
bit-identical to `dp2+fsdp` for both images. The stock-batch2 latent mismatch
therefore also occurs without FSDP. A stock-only BF16 diagnostic replayed
the first denoiser call with identical inputs: repeating the full batch of 2
was exact, while concatenating two batch-1 calls differed by maximum absolute
0.03125. The first observed Linear difference was `tmlp.0` on input shape
`[2, 1, 256]`: two values differed, maximum 1.4901161193847656e-8. This is not
a claim that this one layer solely caused the final drift. The complete
8-step diagnostic sampler still matched the original stock latent exactly;
the observer did not replace its output. The native Krea2 batch-shape
sensitivity was reproduced independently of FSDP.

The resident control's decoded PNGs differed from FSDP by NRMS 0.0110233649
(1.102336%), maximum channel difference 58. A separate VAE
diagnostic reproduced the FSDP pixels by decoding the identical latents as
one batch of 2, and reproduced the resident pixels by concatenating two
batch-1 decodes. Repeating the full batch-2 decode was exact. This demonstrates
a decode batch-shape difference; the original memory condition that selected
the historical decode batch sizes was not captured. It does not qualify DP2
or attribute all latent drift to a single operation.

The FSDP result qualifies the batch-1 T2I and BF16 refinement rows above.
Both refinement stages had exact within-group rank outputs; their native VAE
decode batch size was 1. The first-use proof may cover both stages; a separate
reload per stage is not claimed.

The initial batch-2 upscale candidate failed before refinement sampling.
On the pinned ComfyUI revision, native `VAEEncode` sends the IMAGE batch to
WanVAE as video frames. Temporal rounding maps two frames to one, producing
latent shape `[1, 16, 1, 256, 256]`. The stock graph's second sampler had that
same batch-1 shape; DP2 correctly refused a batch it cannot divide between
two groups. The graph is batch 1 only: increasing its initial batch widget
does not create independent-image DP2 refinement. No graph or ComfyUI changes
were made, and no end-to-end batch-2 refinement acceptance is claimed.

LoRA and unlisted topologies or shapes are not qualified by this record.
Existing resident refinement acceptance remains valid. The local INT8
provenance limitation above still applies; the 2026-10-12 per-tensor encoder
comparison above extends these results to the pinned upstream file. Strict actor
module-origin attestation was unavailable under the production policy; the
managed runtime was separately verified and normal source/artifact/generation
checks remained active. Native RDMA was not exercised.

#### Kroma cfg2

**2026-10-16, dgx-monarch `70fb8997e3ea7b4e771eb046a046a4c8e6d6fdf3`, ComfyUI
`29e28e2b488e7739c351c331d73da25aa435da5c`.** Campaign K's method compared
cfg2 and `uly2` with one GPU on two NVIDIA GB10 GPUs with TORCH_FLASH, the
first Spark capped at 2100 MHz: `benchmark/run_matrix.py` in cluster mode, with
the reference taken from batch row 0 of a same-build dp2 render of batch 2.
The cells used the BF16 template's checkpoint, prompt, seed 42, 8 Euler/simple
steps and native mu at CFG 1.5, not the template's CFG 1: plain Euler at CFG 1
drops the unconditional pass, so cfg2 has nothing to split there, and 1.5 is
the top of the model card's range. Each size ran two negatives. The template's
ConditioningZeroOut copies the positive's 103 text rows, so one GPU evaluates
both in one batch-2 call. An empty text negative has 5 rows, so one GPU makes
two batch-1 calls. Walls are warm sampler walls without text encoding or decode.

| Size | Negative | cfg2 against one GPU | cfg2 warm | `uly2` warm | One GPU warm |
|---|---|---|---|---|---|
| 1152×1152 | Zeroed (template) | One-step NRMS 0.0108; full schedule max abs 4.07 | 30.3, 22.3 s | 25.1, 25.1 s | 39.4, 39.0 s |
| 1152×1152 | Empty text | Identical, full schedule and one step | 19.3, 19.4 s | 25.2, 25.4 s | 39.1, 39.0 s |
| 1536×1536 | Zeroed (template) | One-step NRMS 0.0087; full schedule max abs 2.31 | 38.2, 38.4 s | 46.0, 46.2 s | 75.5, 75.8 s |
| 1536×1536 | Empty text | Identical, full schedule and one step | 38.0, 37.9 s | 46.8, 46.9 s | 76.1, 76.1 s |

`uly2` matched one GPU exactly for every pair, full schedule and one step.
On the zeroed pair cfg2 makes two batch-1 calls where one GPU makes one batch-2
call, the stock batch-shape difference the FSDP record above found. The
[cfg2 row bar](DESIGN.md#54-topology--auto-mode) needs cfg2 bit-identical to
one GPU at every measured size, so row 15's record excludes Kroma, which
`auto` recognizes by the merge record in its header, and a Kroma BF16 file
keeps row 12. The records are the cells `kroma-bf16-1152-cfg15-euler` and
`kroma-bf16-1536-cfg15-euler`. A CFG++ sampler at CFG 1, the other route to
row 15, was not run.

<a id="evidence-chroma"></a>

### Chroma

Cfg2 comparisons of the shipped 100/28-token prompt pair at
1024 square matched a one-GPU probe exactly for BF16, FP8-scaled, FP8 mixed,
MXFP8 and INT8 ConvRot. Trimming cfg padding before the stock forward keeps
its original unmasked call. Equal-length prompts take a different path:
BF16 cfg2 read NRMS 0.034 because stock batches both conditions together.

BF16 pure Ulysses matched same-build local references at 1024 and
1536 for 100/28 and 44/44 prompts, and across 26-step controls including
45/45 and 101/29 prompts. FP8-scaled and INT8 ConvRot tests
matched fresh one-GPU references for the one-step probe and both 26-step
renders, including a changed seed. Their warm times were 32.1 versus 46.2
seconds and 34.0 versus 50.0 seconds. Odd 1040-square FP8-scaled and Radiance
controls were exact. The 1536 FP8-mixed SAGE_AUTO probe read 0.072037.
These comparisons do not cover ring, hybrid or FSDP generally.

Ulysses excludes divisibility padding from attention and restores stock token
order. Effective attention masks still refuse under sequence parallelism;
a no-op mask is accepted. Chroma's text encoder does not create a masked
padding tail, so unequal prompts alone do not explain older deviations.
Padded ring/hybrid requests retain the explicit accuracy refusal.

Chroma NVFP4 remains blocked by `shard_quant_scale:chroma` on sharded
layouts. Shared activation scale reduced Ulysses NRMS from 0.131 to 0.120
and cfg2 from 0.134 to 0.129, still above 0.10. A waived probe of
the unequal 100/28 pair was exact, but the folded 44/44 pair remained above
the limit. Neither isolated success removes the guard. An older test had inconsistent token counts and supplies no accuracy evidence.

<a id="evidence-radiance"></a>

### Radiance

The pixel-space FP32 artifact measured one-step NRMS 0.036 on Ulysses,
0.034 on ring2 and 0.043 on cfg2 at 1024 square, CFG 3.5, with a completed
30-step render. The odd-text Ulysses test read 0.017. A native-workflow Ulysses probe matched its one-GPU reference exactly.
FSDP versus resident output was identical: the BF16 core replicated two
FP32 parameters and held 8.9 GiB per rank. Unequal cfg2 prompts refuse because
stock mask upscaling does not support this pixel-space padding path.

<a id="evidence-ideogram4"></a>

### Ideogram4

Shipped dual-model Ulysses probes at CFG 7 matched one-GPU
pixels exactly for BF16, FP8 and INT8. Warm pair/single times were
34.0/56.1, 26.0/36.0 and 28.0/44.2 seconds. Separate conditional-only FP8
CFG-1 tests matched DP2 latents at 1024 and 1040 with 283/284 text rows and
with Ulysses+FSDP at 1024. That single-model FSDP result does not cover the
dual-model FSDP request. Separately, the September campaign has one
BF16, one FP8 and one INT8 dual-model Ulysses+FSDP record each matching its
resident twin, with the automatic gate off. The default first-use clean-reload
proof still refuses dual-model requests; cfg2+FSDP refuses on every setting.

Explicit dual-model cfg2 puts one checkpoint on each rank. The FP8
comparison read NRMS 0.000 against DP2 and completed 20 steps; each rank
held 8,850 MB. Separate cfg2 probes also read 0.0 for BF16, INT8 ConvRot,
MXFP8 and NVFP4-mixed pairs. Auto remains Ulysses.

The older Ulysses quant tests read INT8 0.032, MXFP8 0.069 and NVFP4
0.164, the latter failing the 0.10 limit. Ring2 measured 0.102 and is not
automatic. These are topology-specific results, so cfg2 NVFP4 success does
not establish accuracy for NVFP4 Ulysses. Sharded attention uses TORCH_FLASH because head
dimension 256 exceeds the other supported kernels. `sync_ulysses=off` was
the faster measured setting, 24 versus 26 seconds, with unchanged output.
Those two times are one 2-second polling interval apart, and the setting
has no effect on the pinned xfuser 0.7.0, whose Ulysses all-to-all never
receives it (see [Adapters](ADAPTERS.md#what-an-adapter-provides)).

<a id="evidence-flux-1-dev"></a>

### Flux 1 Dev

BF16 Ulysses matched DP2 latents exactly at 1024 and 1448,
including odd/even image rows and 313/316 text rows. The template probe also
matched one GPU exactly. The 1448-square 20-step run took 40.74 seconds;
stock token ordering added no material cost in that comparison.
BF16 FSDP matched resident output exactly. A two-LoRA FSDP test with
`lora_low_rss` baked 76 keys and also matched its resident bake; this does
not cover arbitrary LoRAs. Use CFG 1 for the distilled model and converted
LoRAs rather than XLabs format. Padded ring/hybrid keeps its refusal.

<a id="evidence-flux-1-schnell"></a>

### Flux 1 Schnell

BF16 at 1024 square measured one-step NRMS 0.048 on Ulysses and
0.057 on ring2; a four-step render completed. FSDP output
matched resident output exactly. Use CFG 1. Schnell shares Flux's padding
implementation but has no separate padded-stream or stock-ordering comparison.

<a id="evidence-flux2"></a>

### Flux2

Resident FP8-mixed Ulysses matched DP2 latents exactly at
1024 and padded 1040 square. The older 1536 comparison read 0.022 and
completed 20 steps. An earlier Ulysses+FSDP comparison read one-step NRMS 0.007 against an
independent stock one-GPU reference.

Three campaign BF16 LoRA/FSDP cold and warm renders completed, but their
resident references refused for capacity. No LoRA/FSDP fidelity result
follows. Comfy-kitchen quantized-shard LoRA and incompatible FP32 island
bakes refuse. The FP8-mixed first-use FSDP proof was inconclusive because its
quantized wrapping transient did not fit beside the driver. On 2026-10-13 the
BF16 first-use FSDP proof refused at its initial price with the text encoder
resident; [FSDP calibration](#fsdp-calibration) has the figures. The credit
that followed has not been rerun on this file.

The 60.02 GiB BF16 artifact needs a stock-load price of about 131.0 GiB
(2.1 times the file plus 5 GiB), beyond one Spark in this test pair. Use
supported slab residency, explicit FSDP, or a smaller checkpoint. CFG2 and
DP2 still place a full copy on each rank. FP8-mixed stock placement was
measured after the placement repair: 77.0 seconds cold, 34.7 warm and a
20.0 GiB available-memory minimum.

<a id="evidence-flux-3-klein"></a>

### FLUX.3 Klein

**2026-10-15 and 2026-10-16, dgx-monarch `70fb8997e3ea`, ComfyUI `29e28e2b4`
on both Sparks.** The hardware ladder ran the release that added the family,
with torch 2.12.0+cu132 and NCCL 2.32.3, on two NVIDIA GB10 GPUs with
TORCH_FLASH; stock ComfyUI on the first Spark ran comfy-kitchen 0.2.37. The
first Spark was capped at 2100 MHz, with a highest sampled SM clock of
2093 MHz; the second was uncapped. Support follows ComfyUI 20b46065e
("Support Flux3-image"), first released in ComfyUI 0.41.0. The six DiT
files matched the manifest's SHA-256 on both Sparks, and the BF16 encoder
matched its pinned hash on the first Spark, which encodes the text. This
record covers both checkpoints;
[FLUX.3 Klein Base](#evidence-flux-3-klein-base) adds the base model's cfg2
and auto-table readings.

The ladder compared each candidate with stock ComfyUI on one GPU, which ran
on the first Spark in fresh processes of one or more graphs each, and with
batch row 0 of a same-build dp2 render of batch 2, which carries one-GPU math.
All 17 dp2 references scored against stock equaled their stock renders, so
the two references agree. Legs used the shipped recipes unless a row says
otherwise: the step-distilled model at 16 Euler/simple steps and CFG 1, the
base model at 25 steps and CFG 3.5, seed 42, the BF16 Qwen3-VL encoder and
the Flux 2 VAE. Three prompts set the text rows: the shipped step-distilled
prompt (80 rows, one block, which batches with the empty negative), the
shipped base prompt (160 rows) and a 355-word vendor-length caption (480
rows). One-step comparisons use the 0.10 NRMS floor. Multi-step comparisons
across topologies are recorded, not gated.

| Phase | What ran | Verdict |
|---|---|---|
| 0, preparation | Header audits, the header gate scan of the 77 files in each Spark's model directories, pin-stripped INT8 copies, encoder and VAE identity | Every check matched the design; of those 77 files the gate refused one, the diffusers-layout `fp8r` |
| 1, first numbers | 2 stock references, 3 candidate cells, 4 regression cells | uly2 identical to one GPU and stock; Anima, Kandinsky 5 and Wan 2.1 latents unchanged from dev `6f2b199a3ada`, and Kroma's `uly2+fsdp` equal to its resident twin |
| 2, stock references | 18 stock jobs, 44 graphs, first Spark only | 42 rendered; the two diffusers-layout NVFP4 files loaded with their scales dropped and failed at the first step |
| 3, run_matrix candidates | 15 cells | Every gated comparison passed or refused as expected; a batch-2 second row and two first-use readings read CHECK, below |
| 4, API legs | 76 legs in three driver sessions | Every leg rendered or refused as expected; four readings are CHECK, below |
| 5, auto-table measurement | 27 cells: 15 base, 10 step-distilled at CFG 1 and 2 step-distilled at CFG 3.5 | uly2 identical to one GPU in every BF16 and NVFP4 cell; no cfg2 row ([FLUX.3 Klein Base](#evidence-flux-3-klein-base)) |
| 6, scoped sweep | The four public templates: a local session of 36 one-GPU reference cells, then a cluster record of 228 cells. 114 are the scoped cells, every preset and quant on Torch Flash with default levers; the other 114 are cells a stopped full run had finished, on cuDNN and Sage attention and with lever variants | Local 36 PASS. Scoped 109 PASS, 3 CHECK and 2 FINDING, below; the other 114 PASS |

The final scoring of 2026-10-16 read 349 comparisons: 299 PASS, 18 RECORDED,
17 harness controls equal to stock, 2 second batch rows equal to their
references, 7 CHECK, 2 MISSING and 4 FAIL. The four FAILs are verdicts on
vendor files, below, and the two MISSING rows are the diffusers-layout NVFP4
files, which left no render to compare. The first base cfg2 cell lost its
one-GPU full-schedule case to a Worker restart (below) and reran on
2026-10-16 with the five base INT8 auto-table cells. In the rerun both rows of
the base model's full 25-step dp2 render equaled stock: row 0 its batch-1
render, and row 1 row 1 of its batch-2 render.

| Comparison | Checkpoints | Result |
|---|---|---|
| uly2, one step, 1024 and 1536 square, BF16 | Both | Identical to one GPU and to stock |
| uly2, one step, odd 1040 and 1552 square grids, BF16 | Both | Identical to one GPU and to stock |
| uly2, one step, 1024 and 1536 square, NVFP4 | Both | Identical to one-GPU NVFP4 and to stock NVFP4 |
| uly2, one step, 1024 and 1536 square, INT8 ConvRot | Both | Identical to stock on the pin-stripped copy; against stock with the pins, NRMS 0.020 to 0.021 (base) and 0.149 to 0.160 (step-distilled), recorded |
| uly2 and cfg2 against a pinned one-GPU render, 1024 to 1536 square, INT8 ConvRot | Base | uly2 one step NRMS 0.020 to 0.022 (one-block prompt) and 0.027 to 0.031 (vendor length), and no full schedule matched; cfg2 identical on the vendor-length prompt, one step 0.015 to 0.016 on the one-block prompt, recorded |
| uly2, full schedule, 1024 to 1536 square, BF16 and NVFP4 | Both | Identical to one GPU at every size, on every prompt |
| uly2, the shipped text-to-image graphs, full schedule | Both | Identical to stock |
| uly2 with the INT8 Qwen3-VL encoder, one step | Base | Identical to stock with that encoder |
| ring2, one step, 1024 square | Base; step-distilled | NRMS 0.0108; 0.0520. Under the floor, not identical |
| ring2, odd 1040 square grid | Both | Refused in the Worker; a base-model API leg showed class K with the ring-pad waiver card |
| `mode=local` on one GPU | Base one step and full, step-distilled one step, base square edit | Identical to stock |
| uly2 edit, one step, one 4:3 or square reference | Base; step-distilled | NRMS 0.019 to 0.021; 0.053 to 0.055. Under the floor, not identical; the fused-norm parity change closes it (below) |
| uly2 edit, one step, two references | Both | Identical to stock |
| dp2 edit, one step, one 4:3 reference, row 0 | Base | Identical to stock |
| cfg2 edit, one step, one 4:3 reference | Base | NRMS 0.020 |
| uly2 edit, four references (20,155 joint rows) and keep resolution (24,656) | Base | Rendered, finite |
| uly2, the shipped edit graphs, full schedule | Base; step-distilled | NRMS 0.028; 0.066, recorded; the same cause as the one-step edit row |
| dp2 batch 2, full schedule | Base | Row 0 identical to stock; row 1 identical to stock's batch-2 row 1 |
| dp2 batch 2, full schedule | Step-distilled | Row 0 identical to stock; row 1 NRMS 0.0083 against stock's batch-2 row 1 (CHECK, batch shape) |
| Determinism, uly2, one step | Step-distilled | Identical across a Recycle and across two driver sessions |
| Comfy-managed residency, one step | Step-distilled uly2 BF16; base uly2 BF16 and INT8; base single | The step-distilled render equaled the resident one; on the base cases a second sample after a prompt change equaled the same prompt rendered fresh |
| `uly2+fsdp`, one step and full schedule | Both, BF16 | Identical to resident uly2 |
| `uly2+fsdp`, one step | Both, INT8 ConvRot | Identical to resident uly2 |
| `cfg2+fsdp`; `dp2+fsdp`, both rows, one step | Base, BF16 | Identical to resident cfg2; identical to resident dp2 |
| `cfg2+fsdp` and `dp2+fsdp`; `uly2+fsdp` | Base INT8; base NVFP4 | Refused class K (the pins); refused class P (no shard layout for block-scaled weights) |
| Pinned INT8 file, one step | Step-distilled | uly2 dropped the pins and asked nothing; dp2 refused class K with the Sol-Attn card, and under the grant row 0 equaled stock's pinned render |
| Probe LoRA pinning the ten early blocks, one step | Step-distilled BF16 | uly2 ignored the pins and equaled the render without it; dp2 refused class K, and under the grant row 0 equaled stock's probe render |
| Synthetic rank-4 LoRA at 0.5, then 0.8, uly2, one step | Both, BF16 and INT8 | Every 0.8 step hot-swapped with no load; the first BF16 use at 0.5 passed the first-use Gate with the LoRA loaded; at 0.8, NRMS 0.0096 (base) and 0.0499 (step-distilled) against stock's LoRA loader |
| The same LoRA on `uly2+fsdp` with `lora_low_rss` | Base BF16; base INT8 | At 0.8, identical to the resident bake, including keys on the replicated early blocks; INT8 refused class P |
| Removing the LoRA (un-bake) | Both on the resident rung; base on the slab and FSDP rungs | The next render equaled the render without a LoRA on every rung; only FSDP restored in place, and the resident and slab rungs reloaded (CHECK) |

The first-use Gate for INT8 dp2 ended on the class K guard and changed no
lever. Two other first-use readings read CHECK and wait on a reading of the
ledger: uly2 BF16 base gained no new row, and `uly2+fsdp` BF16 base logged
first-use PASS rows that do not name the clean-reload proof. The first
two-Spark cells, run earlier on 2026-10-15, recorded the expected LoRA-free
reading, INCONCLUSIVE `no_material`. The slab rung's first use with a LoRA
left no Gate row (CHECK), so the slab rung is not vouched for FLUX.3.

**Quantized files.** The NVFP4 files are a capacity choice. Against stock BF16
at one step, stock NVFP4 read NRMS 0.0589 (base) and 0.4988
(step-distilled); that is the quantization's own difference, recorded and
not a fidelity claim. The INT8 files pin comfy-kitchen's INT8 attention on 33
blocks; stripping those 33 config tensors left every other tensor
byte-identical, and the stripped copy is the exact reference for uly2, which
drops the pins. INT8 renders on a topology without sequence parallelism only
under the Sol-Attn waiver, so INT8 walls and comparisons there are recorded,
never granted. The vendor's diffusers-layout `variants/fp8r` file loaded in
stock ComfyUI with its scales dropped and rendered a non-finite or black
latent. dgx-monarch's header gate refuses it before any load, finding 234
layers with no ComfyUI quantization record
([TROUBLESHOOTING #120](TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
The diffusers-layout `variants/nvfp4` files carry no ComfyUI quantization
record either: stock ComfyUI loaded them with their weight and input scales
dropped, and the first step failed with a matrix shape error. The header gate
refuses both by the same rule, finding 198 packed 4-bit layers in each with no
record. The 77-file scan did not cover them, so that reading comes from a
header read with the release's gate code on 2026-10-16. They hold 557 of 953
(step-distilled) and 555 of 951 (base) tensors byte-identical to the
ComfyUI-layout NVFP4 files, so they are distinct files, not another layout of
the same bytes.

**INT8 on uly2 after the INT8 dispatch fix.** On 2026-10-21, at dgx-monarch
`9caa6aae660b` with ComfyUI `d52a7a5ef` and the first Spark capped at
2100 MHz, the [MOSS-Image](#evidence-moss-image-sft) re-measurement rendered
both INT8 ConvRot checkpoints on uly2 again. Against the pin-stripped
one-GPU render in the same cell, dp2 row 0, uly2 was bit-identical at 1024
and 1536 square, one step and full schedule (16 steps at CFG 1; 25 steps at
CFG 3.5), 8 of 8, and the pinned files dropped their 33 pins and equaled the
pin-stripped renders, 16 of 16. No stock FLUX.3 render exists on that
ComfyUI, so the reference is the one-GPU render; on MOSS-Image that render
equaled stock at every INT8 size measured. No FLUX.3 INT8 uly2 render ran
between the fused-norm parity change and the fix, so that window is
unmeasured.

**The prompt upsampler.** The vendor's upsampler ran on the driver ahead of
CLIP Text Encode for a step-distilled uly2 render and a base cfg2 render; both
were finite. During the uly2 leg the first Spark's lowest sampled available
memory was 44.6 GiB, and it was still falling when sampling stopped, against
54.9 GiB for the shipped step-distilled graph on uly2 without it.

**Incidents that changed the evidence.** The first run of the API
legs, on 2026-10-15, refused 62 of 74 legs with `ComfyManagedResidencyError`.
The leg order changed the `comfy_managed` policy four times without a Recycle,
which the product refuses by design
([TROUBLESHOOTING #62](TROUBLESHOOTING.md#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)).
The rerun recycled before each change and replaces that run's records, which
are kept apart and not scored. In phase 5, every INT8 one-GPU and cfg2 row read
N/A: under the waiver the KSampler node returns a stamped result that
`benchmark/run_matrix.py` could not unpack. Commit `3244f29` fixes the tool.
The five INT8 cells reran with the fixed tool on 2026-10-16: every one-GPU and
cfg2 row rendered under the waiver and records its waiver run ids. uly2 drops
the pins and runs the exact kernel, so against the pinned one-GPU render it
read one-step NRMS 0.020 to 0.022 on the one-block prompt and 0.027 to 0.031
on the vendor-length prompt, under the floor, while no full 25-step render
matched. The two sides run different attention kernels, so that difference is
by construction. cfg2, which runs the pinned kernel on both ranks, was
bit-identical to one GPU on the vendor-length prompt and read 0.015 to 0.016 at
one step on the one-block prompt, the batch-fold difference BF16 shows
([FLUX.3 Klein Base](#evidence-flux-3-klein-base)). In phase 1, the Worker
refused the first case of the first base cfg2 cell, its one-GPU full
schedule, with `WorkerProcIdError`: it was the first render after a Worker
restart, the fault that
[issue #47](https://github.com/Deen-Media/dgx-monarch/issues/47) tracks and
commit `071c985` fixes. The cell's other cases ran, and the rerun rendered
that case.

**The scoped sweep.** Both FINDING rows were a label gap in the sweep matrix,
which pull request [#53](https://github.com/Deen-Media/dgx-monarch/pull/53)
closed: the matrix predicted an untyped refusal for `cfg2+fsdp` on the
step-distilled graphs at CFG 1, and the runtime refused with class P. The three
CHECK rows are step-distilled NVFP4 cells over the 0.10 floor at one step,
each against the local session's one-GPU render of the same graph: the edit
on uly2 and on auto, which picks uly2, read NRMS 0.117, and text to image on
ring2 read 0.183. The base model's NVFP4 edit read 0.019 on uly2 and auto,
within the floor. So step-distilled NVFP4 edits run on single, and ring2 on
the step-distilled model is measured, never granted: across the ladder and
the sweep its one-step readings ran from 0.026 (BF16 text to image) to 0.183,
so use uly2.

**Why uly2 edits differed, and the fused-norm parity change.** A probe on
2026-10-16, at the same release and ComfyUI, found the cause of the uly2 edit
readings above and of a gap the ladder never met, uly2 text to image at 704
square and below. FLUX.3's blocks normalize the whole stream in one call to
comfy-kitchen's fused AdaLN and then redo each later span in a call of its
own. comfy-kitchen 0.2.37 takes one CUDA kernel for calls of 1024 kernel rows
(batch times rows) or fewer and another above, and the two round a few
outputs one bf16 step apart. A uly2 rank normalized its piece of a span in a
call sized to the piece, while stock normalizes the same rows in a call sized
to the whole span. In a one-reference edit the first rank holds the first 40
reference rows, so its call took the small-call kernel where stock's took the
other, and attention spread that difference to every token. One rule
predicted all 11 probe geometries and every ladder leg: rows differ whenever
their uly2 call holds 1024 kernel rows or fewer while stock's call for them
holds more. An op-level probe at FLUX.3 Klein's widths found no other
row-count effect: the Linears, Torch Flash attention on half the heads and
the fused q and k norm with rotary embedding kept stock's bits at every rank
and stock row pair it tried.

| Geometry, BF16, one step, step-distilled unless named | Short call on uly2 | uly2 against one GPU, before the change |
|---|---|---|
| One square reference (the ladder's square edit) | 40 reference rows | NRMS 0.0545 |
| One 1152-square reference, 1024 target | 584 reference rows | NRMS 0.0538 |
| Two references, 1024 target | 680 reference rows | NRMS 0.0240 |
| Text to image, 512 square | 552 joint rows | NRMS 0.0336 |
| Text to image, 704 square | 968 image and 1008 joint rows | NRMS 0.0364 |
| Base, text to image, 512 square | 1024 kernel rows in the image phase | NRMS 0.0098 |
| Text to image at 736 square, base at 704; one 1408-square or 1 MP 4:3 reference; the ladder's two-reference edit | None | Identical |

The change runs every fused-norm call of the 33 sharded blocks at stock's row
count for its span: zero rows fill the piece's call to stock's size, and the
rank keeps its own rows. The op is row-local, so those rows carry stock's bits
wherever comfy-kitchen puts its branch. On the first Spark's GPU, padded
pieces of 40 and 1024 rows of a 4096-row span, and a batch-2 piece of 512
rows a sample, matched stock's call in every output, while the same pieces in
calls of their own differed in 4, 64 and 44 outputs
(`tests/test_flux3_adaln_parity.py`). A CPU test drives ComfyUI's own blocks
and checks that every fused-norm launch on every rank holds stock's rows.
After the change, BF16 and NVFP4 edits on uly2 and uly2 text to image at 704
square or below are bit-identical to one GPU, confirmed on 2026-10-16 on
release `dab81e3`: the five edit legs (E1, E1sq, E2, both checkpoints, and the
base E1 on dp2) read bit-identical to stock ComfyUI, and text to image at 512
and 704 square (step-distilled) and 512 square (base) read bit-identical to
one GPU where they had differed. The NVFP4 edit readings
are inferred to be the same switch, larger after 4-bit rounding; the probe
rendered BF16 only. The base model's cfg2 edit reading of 0.020 has another
cause: each of its calls stays on the same side of the branch as stock's, and
it reads like the batch-fold difference under
[FLUX.3 Klein Base](#evidence-flux-3-klein-base).

**CPU and file checks.** The CPU evidence builds ComfyUI's own FLUX.3 model
at toy size in float32 and runs the two-rank sharded forward against stock's:
every kernel call gets stock's q, k and v for its heads in stock order with no
pad row, and the output equals stock's. The cases cover an even grid, an odd 5
by 5 grid that pads both phases, one reference with the shard boundary inside
it, two references, batch 2, the step-distilled flow map with a real two-row
`sample_sigmas`, the fused q, k and v layout the INT8 files use, and cached
against uncached text states. Seam 33 holds the class-scoped census of modules
the base model runs outside its forward, seam 39 the attention pins and the
stock attention of the early text and reference blocks, and seam 41 the branch
that computes the base model's text and reference streams once per
conditioning, each against the installed ComfyUI.

On 2026-10-15 the pinned encoder and VAE were read against the vendor's
copies. The Comfy-Org `qwen3vl_4b_bf16.safetensors` (SHA-256 `36f3ff44`) and
the vendor's file of that name (`f0b24fc5`) hold the same 713 tensors with the
same names, dtypes, shapes and bytes, and differ only in 24 bytes of header
metadata. The pinned `flux2-vae.safetensors` (`d64f3a68`) and the vendor's
`vae/flux2-vae.safetensors` (`868fe7b3`) hold the same 250 inference weights
in two key layouts; each also carries a batch-norm step counter, with
different values, which inference never reads. One latent decoded through
both VAE files in one stock process gave equal pixels.

Records: the private evidence directory, record `70fb8997e3ea` of
2026-10-15 and 2026-10-16: the ladder state, the comparison summary and pairs,
the auto-table summary, the operator notes, the stock job records, the
sweep summaries and the edit probe of 2026-10-16 with its op-level results.

<a id="evidence-flux-3-klein-base"></a>

### FLUX.3 Klein Base

The [FLUX.3 Klein](#evidence-flux-3-klein) record covers this checkpoint's
uly2, ring, edit, FSDP, LoRA and quantization results. Its own cfg2 and
auto-table readings follow.

cfg2 splits the conditional and unconditional calls across the two GPUs. When
the prompt encodes past one 80-row block, stock ComfyUI on one GPU also runs
them as two batch-1 calls, and cfg2 matched it exactly: the shipped base
prompt (160 rows) and the vendor-length caption (480 rows) were identical to
stock at one step and to one GPU over the full schedule. When the prompt fits
one block, it has as many rows as the empty negative, and one GPU runs both in
one batch-2 call. cfg2 still runs two batch-1 calls, so it differs: at 1024
square the one-step latent read NRMS 0.0094 against one GPU and stock, and
the full 25-step render failed the same-math bar. A second reading points at
batch shape. On the step-distilled model at CFG 1, which has no uncond call,
dgx-monarch's dp2 render runs batch 1 on each rank: its row 0 equaled stock's
batch-1 render, and its row 1 read NRMS 0.0083 after the full schedule against
row 1 of stock's batch-2 render. Both readings suggest that ComfyUI's batch-2
math differs from its batch-1 math on GB10. That is an inference: no run
scored a stock batch-1 render against a stock batch-2 render. Chroma's
equal-length prompts read the same way ([Chroma](#evidence-chroma)).

The base model's auto-table cells ran 1024, 1152, 1216, 1280 and 1536 square,
BF16 and NVFP4, each with the 80-row and the 480-row prompt, head capped at
2100 MHz. uly2 was identical to one GPU in every cell. cfg2 was 1.88x to 1.98x
faster than one GPU and 1.23x to 1.41x faster than uly2. It was identical on
the 480-row prompt at every size; on the 80-row prompt the one-step NRMS read
0.009 to 0.010 (BF16) and 0.023 to 0.031 (NVFP4), and no full render matched.
Campaign K's bar asks a cfg2 row to be identical on both prompts at each size
([cfg2 crossover rows](#campaign-k-cfg2)), so no size qualified and FLUX.3 has
no cfg2 row; auto takes uly2. INT8 was recorded only, never a candidate. Both
checkpoints share one family, so a base-model cfg2 row would also reach the
step-distilled model at CFG other than 1. Two cells measured that case, the
step-distilled model at CFG 3.5: cfg2 read one-step NRMS 0.061 at 1024 and
0.059 at 1536 on the 80-row prompt.
[Benchmarks](BENCHMARKS.md#flux3-klein-one-gpu-uly2-and-cfg2-2026-10-15-to-2026-10-16-head-capped)
lists every cell's walls.

At CFG 1 the step-distilled model gives cfg2 nothing to split, and an explicit
cfg2 refused class P.

<a id="evidence-moss-image-sft"></a>

### MOSS-Image SFT

**2026-10-21, dgx-monarch `ff8dab8300aa`, ComfyUI `d52a7a5ef` on both
Sparks; re-measured on 2026-10-21 and 2026-10-22 at `9caa6aae660b`.** The
hardware ladder ran the release that added the family, with comfy-kitchen
0.2.39, torch 2.12.0+cu132, NCCL 2.32.3 and torchmonarch 0.7.0, on two NVIDIA
GB10 GPUs with TORCH_FLASH. The first Spark was capped at 2100 MHz, with a
highest sampled SM clock of 2093 MHz; the second was uncapped. ComfyUI
`d52a7a5ef` is a master commit that carries the control LoRA support of
e3bee7ffc. The 11 manifest files in the table below matched their SHA-256 on
both Sparks. Release `9caa6aae660b` adds the INT8 dispatch fix and the #129
refusal; the re-measurement ran on it with the same ComfyUI and the same cap:
the INT8 set, BF16 spot checks, the scheduler record, the #129 legs, a stock
LoRA probe and the two INT8 Campaign K cells. This record covers all three
checkpoints; [MOSS-Image Base](#evidence-moss-image-base) and
[Kromoss](#evidence-kromoss) add their own readings. The Studio lifecycle
pass ran on the first release as well
([Studio acceptance](STUDIO_ACCEPTANCE.md)).

The ladder compared each candidate with stock ComfyUI on one GPU, which ran
on the first Spark in fresh processes of one or more graphs each, and with
batch row 0 of a same-build dp2 render of batch 2, which carries one-GPU math.
Every dp2 reference scored against stock equaled its stock render, so the two
references agree. Legs used the shipped recipe unless a row says otherwise:
50 Euler steps on ComfyUI's `simple` scheduler, CFG 3, layer 22 skipped on
the negative pass over the whole schedule, the vendor's tag negative, the
BF16 Qwen3.5-9B encoder and the FLUX.2 small decoder. A salt on the prompt
set the parity of the joint stream, the image rows plus the text rows: at
1024 square the template's prompt gives 4,233 rows (odd) and the salted one
4,234 (even). One-step comparisons use the 0.10 NRMS floor. Multi-step
comparisons across topologies are recorded, not gated.

| Phase | What ran | Verdict |
|---|---|---|
| 0, preparation | The release slot on both Sparks, the SHA-256 of 14 files on both, pin-stripped copies of the three INT8 files, the CPU suite and the canaries | Every check matched; the import canary, which needs a visible GPU, has no GPU-visible run at this release |
| 1, regression null and first numbers | The five regression cells from the FLUX.3 ladder (Anima, Kandinsky 5, Wan 2.1, Kroma `uly2+fsdp` and FLUX.3 Klein uly2), run on the previous dev release and again on this one, and 7 candidate cells: 12 cells at this release | 10 of 10 null latents equal across the two releases; the uly2, skip and cfg2 readings below |
| 2, stock references | 21 stock jobs, 71 graphs, first Spark only | All finite and non-black; two stock processes gave identical output |
| 3, run_matrix candidates | 20 cells | Every gated comparison passed or refused as expected; INT8 uly2 read CHECK, below |
| 4, API legs | 68 legs in three driver sessions, then a per-key bake probe | 66 as expected; two failed twice, below; the probe found 0 of 378 keys different from stock's |
| 5, auto-table measurement | 15 Campaign K cells | 26 of 26 BF16 rows identical, one step and full schedule; INT8 recorded only |
| 6, scoped sweep | The 14 MOSS-Image templates: 28 local cells and 224 cluster cells | 252 of 252 PASS, no finding |
| 7, re-measurement at `9caa6aae660b` | 22 cells (INT8 identity, FLUX.3 Klein INT8, BF16 spot checks, the scheduler record and two INT8 Campaign K cells), 6 API legs, the 36 sweep cells of two of the four LoRA templates and a stock probe on one GPU | INT8 uly2 bit-identical; every BF16 latent unchanged; #129 refused every INT8 + LoRA cell that ran, as labelled |

The final scoring at `ff8dab8300aa` read 507 comparisons: 393 PASS, 39
harness controls equal to stock, 34 RECORDED, 35 CHECK, 3 MISSING, 1 second
batch row identical to its reference and 2 FAIL. Of the CHECK rows, 25 are
INT8 uly2 readings, the kind the dispatch fix closed, and 3 the BF16 LoRA
renders against stock, below. Six are LoRA legs whose journals explain the
reading: a first-use load before the hot swaps, or a residency change before
a reload.
The last is a batch-2 second row. The two FAILs are below. The three MISSING
rows are pinned-kernel record rows at 1536 that no stock job rendered. The
rescore at `9caa6aae660b` read 240 comparisons: 191 PASS, 28 harness
controls, 18 RECORDED and 3 MISSING, with no CHECK and no FAIL. The 24 INT8
uly2 comparisons it shares with the first release moved from CHECK to
identical, and no shared comparison moved the other way.

| Comparison | Checkpoints | Result |
|---|---|---|
| uly2, one step, 1024 square, both joint parities, and without the skip, BF16 | All three | Identical to one GPU and to stock, 18 of 18 |
| uly2, one step, 512, 640, 704 and 768 square, both parities at 512 and 768, and a long prompt of 6,347 joint rows, BF16 | SFT | Identical to one GPU and to stock, 16 of 16 |
| uly2, one step, odd 1040 square grid and 1456 square, BF16 | SFT and Kromoss at 1040 with both parities, Base at 1040, SFT at 1456 with both parities | Identical, 14 of 14 |
| uly2, one step, 1536 square, and batch 2 at 1024, BF16 | All three; SFT batch 2 | Identical, 7 of 7, both batch rows against stock's batch-2 render |
| uly2, one step and full schedule, 2048 square (16,521 joint rows), BF16 | SFT | Identical |
| uly2, full schedule, Campaign K | SFT at 1024, 1152, 1216, 1280, 1408, 1536 and 2048 square, also with an empty negative at 1024 and 1536; Base and Kromoss at 1024 and 1536 | Identical to one GPU at every size, one step and full schedule, 26 of 26 |
| uly2, full schedule, with the skip and without it | SFT | Both identical to one GPU and to stock; skip on against skip off read NRMS 0.558 (max abs 5.64), as two different recipes must |
| auto, 1024 and 1536 with and without the skip, CFG 1, and 2048 | SFT | Resolved uly2 (row 99) in all six cases, identical |
| The shipped graphs through the graph rewrite path, auto, full schedule | All three at 1024; SFT at 1040, at 1536 and with the skip off | Identical to stock |
| The layout graph | SFT | uly2 and single identical to stock at one step (4,238 joint rows); auto identical to stock over the full schedule |
| `mode=local` on one GPU | SFT one step and 50 steps; Kromoss one step | Identical to stock |
| The INT8 Qwen3.5-9B encoder on the cluster | SFT | Identical to stock with that encoder; stock with the INT8 encoder against the BF16 one read NRMS 0.0276, recorded |
| dp2 batch 2, full schedule | SFT | Row 0 identical to stock's batch-1 render; row 1 NRMS 0.129 against row 1 of stock's batch-2 render (CHECK, batch shape) |
| Determinism | SFT | Identical across a Recycle, a second driver process and a respawn, and between two stock processes; renders after each `comfy_managed` policy change identical |
| `uly2+fsdp`, prefetch depth 1 | SFT BF16: one step and full schedule with the skip, one step without it; Kromoss BF16: one step and full schedule; SFT INT8: one step | Identical to resident uly2 |
| `uly2+fsdp`, prefetch depth 2, full schedule | SFT BF16 | Identical to resident uly2 |
| `cfg2+fsdp` dispatch pair; `dp2+fsdp`, both rows | SFT BF16 | Identical to resident cfg2; identical to resident dp2 |
| `uly2+fsdp` against its resident twin in the sweep, full schedule | All three | BF16 14 of 14 and INT8 10 of 10 identical |
| cfg2 on the shipped graphs, which carry the skip | All three | Refused class P (#126) at the driver preflight, before dispatch; the next render ran with no Recycle |
| cfg2 with the skip off, a prompt and negative of different lengths (the dispatch pair) | SFT | Identical to one GPU and to uly2, one step and full schedule |
| cfg2 with the skip off, a pair of equal length (the fold pair) | SFT | 1024: one step NRMS 0.0229 (max abs 0.125), full schedule 0.0775 (2.31); 512: one step 0.0258 (0.141), full schedule 0.0307 (0.551). Measured, not identical |
| cfg2 at CFG 1, the 8-step DMD graph | Kromoss | Refused class P before the render, inside the first-use Gate |
| ring2, odd joint stream | SFT and Kromoss | Refused class K with the ring-pad card |
| ring2, one step, even joint stream, against stock | SFT; Kromoss | NRMS 0.0264; 0.0233. Under the floor, not identical |
| ring2 in the sweep, one step, BF16 | All three | Without a LoRA, NRMS 0.0025 to 0.031 against one GPU on ten padded graphs rendered under the ring-pad waiver. With a LoRA, 0.0034 to 0.0041 on the three unpadded 8-step graphs, rendered after their first-use Gate ceremonies failed, and 0.045 on the padded SFT LoRA graph |
| uly2, one step, 1024 and 1536 square, INT8 ConvRot on the pin-stripped copy | All three | At `ff8dab8300aa` NRMS 0.0268 to 0.0363 against the pin-stripped one-GPU render and stock (CHECK); at `9caa6aae660b` identical to both, 12 of 12 |
| uly2, full schedule, 1024 square, INT8 ConvRot on the pin-stripped copy | All three | At `9caa6aae660b` identical to one GPU, 3 of 3 |
| uly2 on the pinned INT8 file, one step | All three | Dropped the 42 pins and equaled the pin-stripped uly2 render at both releases; at `9caa6aae660b` also the pin-stripped one-GPU and stock renders, 18 of 18 |
| The pinned INT8 file on dp2, cfg2 and single | SFT | Each showed the Sol-Attn card (class K); under the grant, dp2 row 0 equaled stock's pinned render |
| INT8 `cfg2+fsdp` and `dp2+fsdp` | SFT | Refused class K (`FsdpGateProofError`) |
| BF16 with a DMD LoRA at 0.8 and 1.0, uly2 against single | Kromoss; Base and SFT in the sweep | Identical on uly2 and auto in every sweep template |
| The same, against stock with the same LoRA | Kromoss | One step NRMS 0.0365 (DMD2 at 0.8), 0.0489 (DMD at 1.0) and 0.0635 (DMD2 at 1.0); 8 steps 0.36 to 0.44. Recorded: stock applies the LoRA at cast time |
| INT8 ConvRot with the DMD2 LoRA, uly2, one step | Kromoss | At `ff8dab8300aa` NRMS 0.985 against stock, twice (FAIL); at `9caa6aae660b` refused class P (#129) |
| The color and depth control LoRA | SFT | Both ranks refused class P (#127) at load; the next render ran with no Recycle |
| `MOSSImageControl` with the color map and no control LoRA | SFT | No refusal: stock ComfyUI drops the map without the LoRA, and the Workers matched stock |
| The scheduler record: `normal` against `simple`, full schedule, 1024 square | SFT | uly2 identical to one GPU and to stock on each scheduler; `normal` against `simple` at the same seed read latent NRMS 0.140 (max abs 4.33) and pixel NRMS 0.116 (max abs 235), PSNR 31.3 dB |

**The INT8 dispatch fix.** At `ff8dab8300aa` INT8 ConvRot on uly2 read
one-step NRMS 0.0268 to 0.0363 against the pin-stripped one-GPU render at
1024 and 1536 square, under the floor and not identical. A CPU reproduction
found the cause: since the fused-norm parity change, the rows
`at_stock_rows` kept carried the filled call's leading stride at batch 1, and
outside inference mode, where a Worker samples, ComfyUI sent the next linear
on an INT8 weight, `qkv` and `ffn_up_gate`, down a route that dequantizes the
weight and skips activation quantization. Release `9caa6aae660b` gives the
kept rows a fresh tensor's strides. On 2026-10-21 at that release INT8 uly2
was bit-identical to the pin-stripped one-GPU render and to pin-stripped
stock at one step, at 1024 and 1536 square for all three checkpoints, and to
one GPU over the full 50 steps at 1024. Every BF16 latent rerun at that
release, 42 of them, equaled its latent from the first release, and so did
every INT8 one-GPU latent. INT8 uly2 also got faster: 0.78 to 0.79 s a step
at 1024 against 1.01 to 1.03 s before
([Benchmarks](BENCHMARKS.md#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped)).

**Quantized files.** Each INT8 ConvRot file pins comfy-kitchen's INT8
attention on its 42 blocks with a `layers.N.comfy_attention.config` tensor.
Removing those 42 tensors left the other 715 byte-identical, 5,413,693,208
bytes, and that copy is the exact reference for uly2, which drops the pins.
These copies are test inputs, not manifest rows. Stock ComfyUI on the pinned
file against stock on the copy, one step at 1024, read NRMS 0.0404 (SFT),
0.0401 (Base) and 0.0357 (Kromoss): the pinned kernel's own difference.
Against stock on the pinned file, uly2 read 0.0402, 0.0400 and 0.0356 at
`9caa6aae660b`, that same difference. INT8 renders on a topology without
sequence parallelism only under the Sol-Attn waiver, so INT8 walls and
comparisons there are recorded, never granted.

**LoRA.** ComfyUI maps the separate q, k and v keys of the Kromoss DMD LoRAs
onto slices of the fused `qkv` weight, so each rank bakes 168 keys: `qkv`,
`o`, `ffn_up_gate` and `ffn_down` on 42 blocks. A per-key probe on one GPU
read the store's bake against stock ComfyUI's merge at four stages, the base
weights, DMD2, the hot swap to DMD and the un-bake: 0 of 378 keys differed at
any stage, and 0 of the 42 `qkv` keys. The FSDP bake and the hot swap each
gave the resident bake's render, and the un-bake gave the render without the
LoRA. A hot swap took 1.1 to 1.8 s on one GPU, 1.6 to 1.8 s resident on the
pair and 3.2 to 3.4 s under FSDP, and the un-bake captured 168 keys as
checkpoint references instead of 8.86 GiB of backup. Inside dgx-monarch, uly2
with a LoRA equals one GPU. Against stock with the same LoRA it read one-step
NRMS 0.0365 to 0.0635 at strengths 0.8 and 1.0, because under DynamicVRAM
stock applies a LoRA at cast time and dgx-monarch bakes it, as on FLUX.3
Klein. The slab rung with a LoRA was not exercised: the first-use Gate kept
that leg on stock residency. ring2 with a LoRA is not supported. In the
sweep's cluster session on 2026-10-21, every first-use Gate ceremony of ring2
with a LoRA that reached a verdict failed, six in all, three of them on BF16
files with a largest latent difference of 1.02 to 2.35, and each quarantined
the low-RSS LoRA residency for its pair; the renders then ran without it. The
SFT LoRA graph's ceremony met the ring-pad guard first and reached no verdict.
The cause has not been investigated. Every other LoRA ceremony read 0.0, on
uly2, auto, `uly2+fsdp` and single.

**INT8 with a LoRA (#129).** At `ff8dab8300aa` the Kromoss INT8 file with the
DMD2 LoRA at 1.0 on uly2 read one-step NRMS 0.985 against stock ComfyUI with
the same file and LoRA, twice, and the sweep's SFT INT8 graph with DMD2 at 0.8
read 0.277 on uly2 against dgx-monarch's own one-GPU render on the pinned
kernel. Release `9caa6aae660b` refuses a LoRA on any `moss_image` INT8 file,
class P. On 2026-10-21 the API leg refused at the driver preflight with the
#129 text, before dispatch, so no lease existed to read; the next renders ran
on the same driver and fleet with no Recycle. The sweep ran two of the four
LoRA templates again, the SFT LoRA graph and the Kromoss DMD graph, and all 18
of their INT8 + LoRA cells refused: 9 with #129 (single in `mode=local`, auto,
uly2, ring2, and cfg2 on the CFG 1 graph), 7 with an older class P guard that
answers first (dp2 at batch 1, cfg2 on the skip graph, `uly2+fsdp` and
`cfg2+fsdp`) and 2 with the class K guard of `ring2+fsdp`. Their 10 BF16 +
LoRA cells rendered with the readings of the first release. The other two LoRA
templates, the shipped Kromoss Turbo graph and the Base DMD2 graph, carry the
new label and have not run again. The Workers' own #129 check in the model
store was not reached: every shipped path runs the driver's check first.

A probe of stock ComfyUI alone on one GPU found the cause upstream. With its
default DynamicVRAM loader, stock applies the LoRA at cast time. With
DynamicVRAM off it loads the INT8 file in full and bakes the LoRA into the
INT8 weights, and that render read NRMS 0.9977 at one step and 0.9894 at 8
steps against the DynamicVRAM render of the same graph. Without the LoRA the
two loaders gave identical output, and on the BF16 file with the LoRA they
read 0.066 apart. The cluster render belongs with stock's classic bake: both
moved the latent about 2.8 NRMS from the render without the LoRA (2.80 and
2.81), where every other LoRA path moved it 1.52 to 1.59, and they sat 0.28
apart. On the GPU every one of the 168 keys baked to within 0.0061 of the
exact sum, in the store and in stock's own bake, the bake changed no other
weight, no cast-time LoRA remained after it, and DMD2 at 2.0 sat 0.91 from
the classic bake at 1.0, so it is not a doubled LoRA. The DMD2 delta is 13 to
29 times smaller than the requantization noise each INT8 bake adds, and the
mechanism inside the classic INT8 render is still open. dgx-monarch's store
bakes classically on every topology, `mode=local` included, so #129 refuses
on every topology.

**cfg2.** cfg2 refuses on the shipped graphs: inside the skip's window ComfyUI
runs the two passes as separate calls, and cfg2 can split only a call that
carries both. The refusal came from the driver preflight before dispatch,
from `benchmark/run_matrix.py` and from the API, and the next render needed
no Recycle. With the skip off, a pair of different lengths splits one call
per rank with stock's math, and a pair of equal length folds into one
batch-2 call on one GPU, which cfg2's two batch-1 calls do not reproduce, as
on FLUX.3 Klein Base and Chroma. At 1024 square the dispatch pair ran 47.0 s
against 93.5 s on one GPU and 62.4 s on uly2. The fold pair misses Campaign
K's bar, so MOSS-Image has no cfg2 row
([DESIGN 5.4](DESIGN.md#54-topology--auto-mode)). At CFG 1 the 8-step graph
leaves cfg2 nothing to split, and it refused class P.

**Refusals on hardware.** Observed: #126 at the driver, #127 on both ranks
while the model loaded (the lease read CONSUMED), the ring-pad card on the
Workers (CONSUMED), the Sol-Attn card on dp2, cfg2 and single (CONSUMED)
and #129 at the driver. Not observed: the Workers' own #126 check, which a
Fleet KSampler with the skip on under cfg2 would reach, and the Workers' own
#129 check. With the shipped nodes, no graph reaches a Worker-side refusal of
a non-identity block replacement. `MOSSImageControl` with its color map and
no control LoRA does not refuse: stock ComfyUI adds the control channels
only when a control adapter has widened `video_embedder`, so without one it
drops the map, the Workers match stock, and with one #127 refuses first.

**The scheduler record.** The four CFG 3 public graphs run ComfyUI's
`normal` scheduler. On 2026-10-21 at `9caa6aae660b` the shipped SFT graph at
1024 square, 50 steps, CFG 3, with the skip and seed 42, rendered on both
schedulers in one process. On each, uly2 was identical to one GPU and to
stock. `normal` against `simple` at the same seed read latent NRMS 0.140
(max abs 4.33) and pixel NRMS 0.116 (max abs 235, mean abs 2.70), PSNR
31.3 dB: a different render of the same prompt. ComfyUI's own
`calculate_sigmas` on MOSS-Image's sampling, shift 5 and multiplier 1, puts
`normal` within 0.0005 of the vendor pipeline's schedule at 4, 8, 25 and 50
steps, while `simple` sits 0.0686 off at 50 steps, 0.148 at 25, 0.393 at 8
and 0.601 at 4. The Base and Kromoss graphs were not rendered on `normal`.
Every other reading in this record ran on `simple`. The 8-step Kromoss graph
keeps `simple`, the Kromoss card's ComfyUI recipe.

**Incidents that changed the evidence.** No power cut, wedge, abort or
fault-driven Recycle occurred at either release. Two API legs failed twice,
and their reruns were bit-identical to the first attempts. The Kromoss INT8 +
LoRA render (0.985) led to #129 and the stock probe above. The control node
leg expected a refusal that stock's behavior makes unreachable, as above. The
INT8 uly2 CHECK rows led to the dispatch fix and the re-measurement. The
first release had no scheduler record; the re-measurement added it.

**CPU and file checks.** Support follows ComfyUI e087d8f6c ("feat: Add
MOSS-image support", 2026-10-16), first released in ComfyUI 0.42.0 together
with the fused feed-forward activation of a639f187a. On 2026-10-20 the RoPE
scaling options (b9b290e43, 2026-10-19) and the color and depth control
(e3bee7ffc, 2026-10-20) were on ComfyUI master only, in no release. ComfyUI
0.42.0 requires comfy-kitchen 0.2.39. On 2026-10-20 the forward, LoRA and
guidance tests of `tests/test_moss_image_comfy_exactness.py`, the tests in
`tests/test_skip_layer_guidance_comfy.py`, `tests/test_lora_shape_guard.py`
and `tests/test_moss_image_skip_end_to_end.py`, and the custom-node
entrypoint canary passed on a `git archive` export of ComfyUI v0.42.0
(355cd6b14) with comfy-kitchen 0.2.39. Four cases skip there: the
`rope_options` case, because that release's `rope` reads no options; the two
widening "set" patch cases of the shape guard, because that release's
`calculate_shape` counts no set patch, so that release's own merge fails on
such a patch instead; and the on-disk
SD3 graph case, because the export holds no model files.

The CPU evidence, `tests/test_moss_image_comfy_exactness.py`, builds
ComfyUI's own MOSS-Image model at toy size in float32 and runs the two-rank
sharded forward against stock's. Only the all-to-all, the gather and the
attention kernel are stubbed, and the stub kernel runs ComfyUI's own
attention on the full gathered rows. Every kernel call gets stock's q, k and
v for its heads in stock order with no pad row, and the output equals
stock's. The cases cover an even stream; odd streams whose rank boundary
falls inside the image rows or inside the text rows; an odd stream whose pad
row, if kept, moves the output; four ranks with one rank of pads only;
batch 2; the fused and the separate q, k and v layouts; `rope_options`
reaching stock's table, and the four-argument `rope` of the 0.42 releases
called without them; ComfyUI's own Skip Layer Guidance skip; a marked
row-local block patch, honored, and the same patch unmarked, refused class
P; every fused-norm launch on every rank against stock's launch; one sharded
attention per block, with the text embedder on ComfyUI's default backend;
stock's prefetch order with no malloc graph; a padded ring, refused class K,
and a waived ring, which attends the pad; and each guard refusing class P
before the first exchange. The file's LoRA half shows ComfyUI's key map
putting separate q, k and v keys on slices of the fused `qkv` weight, and the
store's bake, un-bake and FSDP bake equal to ComfyUI's merge on every patched
key. Its guidance half shows the prompt and the negative published as
`CONDRegular`, which folds only at equal length, and ComfyUI's own
`SkipLayerGuidanceDiTSimple` making two calls inside its window, the second
with the layer-22 identity, which the sharded forward admits and honors with
stock's bits. `tests/test_moss_image_skip_end_to_end.py` carries the skip
from the Skip Layer Guidance (DGX Monarch) node through the request to a
two-rank sharded skip on a tiny real model and matches stock; the same
request on cfg2 refuses class P before any model call.

On 2026-10-20, with comfy-kitchen 0.2.39 and torch 2.12.0 on one GB10 (the
first Spark, under its 2100 MHz cap), kernel probes and
`tests/test_moss_image_parity.py` read every rank's window at 1047, 1031,
4119 and 16420 joint rows, batch 1 and 2, two and four ranks. The fused
`rms_adaln` norm filled to stock's rows, with the window at the front,
returned stock's bits in every window. Called unfilled, it differed in 15 to
27 elements, only where a piece fell below 1024 kernel rows at batch 1, so
the adapter fills both of a block's calls. The fused q and k norm and RoPE
and the BF16 GEMMs of the four block shapes (q, k and v 3072 to 9216, the
output 3072 to 3072, the gated feed-forward input 3072 to 16384 and its
output 8192 to 3072) returned stock's bits on every window with no fill, and
so did the INT8 ConvRot linears, the SiTU activation of the feed-forward
output included. These kernel results made uly2 exact by construction; the
hardware legs above confirm it on renders.

Canary seams 42 to 46 (`moss_image conds`, `moss_image block surface`,
`skip layer guidance simple`, `lora shape-changing set` and `dynamic prompt
asymmetry`) hold, against the installed ComfyUI, the conditioning (the empty
negative encodes to 10 tokens and the vendor tags to 23), the block and
forward surface the sharded forward reruns, the Skip Layer Guidance call
pattern, the LoRA shape predicate, and the schema fact the layout graph's
JSON prompt rests on: CLIP Text Encode's `text` declares dynamic prompts and
Text (Multiline)'s `value` does not.
Seam 26 (`nvfp4 activation scale`) also pins that an INT8 ConvRot weight
takes comfy-kitchen's fused path, which never calls the module, while an
NVFP4 weight calls it once, and seam 39 (`attention keyword handoff`) that a
tiny real MOSS-Image's attention pins drop under sequence parallelism.

On 2026-10-20 the pinned files were read from their headers. The three BF16
DiT files hold the same 379 tensors, 378 BF16 and the F32 `rope.bands` that
ComfyUI drops at load, with no header metadata; the three INT8 ConvRot files
hold the same 757, with 168 INT8 Linears and an attention config tensor on
each of the 42 blocks. ComfyUI's own detection named all six `MOSSImage` with
ratio 16. Kromoss's `qwen3.5_9b_bf16.safetensors` (SHA-256 `7e6e9f08`) and
`flux2-vae_small.safetensors` (`d8d52ba0`) are the same files as Comfy-Org's,
and the VAE is the same file as `diffusion_pytorch_model.safetensors` in
`black-forest-labs/FLUX.2-small-decoder` at `a3efc24f`, by size and SHA-256.
No decode has compared the VAE files.

Tested files and SHA-256 identities, matched on both Sparks:

| Artifact | SHA-256 |
|---|---|
| `moss_image_sft_bf16.safetensors` | `f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f` |
| `moss_image_base_bf16.safetensors` | `cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f` |
| `kromoss_bf16.safetensors` | `54c863b9392f7b6a02f557d32a416b22765b4ffbee322ade9945e379f09adaba` |
| `moss_image_sft_int8_convrot.safetensors` | `b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f` |
| `moss_image_base_int8_convrot.safetensors` | `f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f` |
| `kromoss_int8_convrot.safetensors` | `cacff0dd25cd2cf531e1f5ddae7872c3d57a54dbf66f1a1279f2f118514e7905` |
| `qwen3.5_9b_bf16.safetensors` | `7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53` |
| `qwen3.5_9b_int8_convrot.safetensors` | `32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a` |
| `flux2-vae_small.safetensors` | `d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b` |
| `kromoss_8-steps-dmd2-lora.safetensors` | `7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2` |
| `kromoss_8-steps-dmd-lora.safetensors` | `f0e6cd07f222dfaea798afe3b796e2c2995cadfd44ad56e02ca616f0467aec28` |

The control LoRA leg used the MOSS-Image color and depth control LoRA, a
community file of 313,063,368 bytes, matched on both Sparks; no template or
manifest row names it.

Records: the private evidence directory, record `ff8dab8300aa` of 2026-10-21
with its re-measurement at `9caa6aae660b` of 2026-10-21 and 2026-10-22: the
ladder state, the comparison summaries and pairs of both releases, the
auto-table summaries, the API leg records, the matrix cell results, the stock
job records, the per-key bake probe, the sweep summaries, the operator notes
and the INT8 investigation with its stock probe.

<a id="moss-image-on-comfyui-a071826fd"></a>

**ComfyUI a071826fd and the edit reference cache, 2026-10-22.** ComfyUI
commit 60aade0bb added MOSS-Image edit LoRAs and moved the model's forward. A
verified update then put both Workers on dgx-monarch `de36a06a6d3d`, which
follows that forward and refuses reference images off one GPU, at Gate
protocol 14, and `dgxm doctor` reported 44 checks with 0 failures. Both Sparks
ran ComfyUI 0.42.0 at `a071826fd` with comfy-kitchen 0.2.40, torch
2.12.0+cu132 and torchmonarch 0.7.0, with TORCH_FLASH. The first Spark was
capped at 2100 MHz in every leg, with a highest sampled SM clock of 2093 MHz,
and the second was uncapped. Stock references were rendered again on
`a071826fd`; no candidate was scored against a render from another ComfyUI
commit. Identity means equal float32 sampler latents and, where a leg decodes,
equal uint8 pixels. The SFT checkpoint ran every leg at 1024 square unless the
row says otherwise. A step's wall is the full render at seed 43 minus the
one-step render at seed 42, divided by 49.

| Leg | Compared with | Result | Seconds a step |
|---|---|---|---|
| BF16 uly2, one step, both joint parities, skip and plain | One GPU (dp2 row 0) and stock | Identical, 3 of 3 against each | One step only |
| BF16 uly2, 50 steps | One GPU | Identical over the full schedule | uly2 1.217, one GPU 1.849 (1.52x) |
| INT8 ConvRot pin-stripped uly2, one step, 1024 and 1536 square | One GPU and stock | Identical, 2 of 2 against each | One step only |
| INT8 ConvRot pinned on uly2, pins dropped, one step, 1024 and 1536 square | The pin-stripped references | Identical, 6 of 6 | One step only |
| INT8 ConvRot pin-stripped uly2, 50 steps | One GPU | Identical over the full schedule | uly2 0.839, one GPU 1.058 (1.26x) |
| One-GPU edit: ai-toolkit format with and without the KV cache, and Moss-Image-Edit; one step and 50 steps | Stock | Identical latents and pixels | 3.136, stock 2.142 |
| One-GPU render with no reference, one step and 50 steps | Stock | Identical latents and pixels | 1.769, stock 1.784 |
| Edit graph on uly2 | None | Refused class P (#130) at the driver before dispatch | None |
| uly2 render with no reference, before and after that refusal on the same fleet, no Recycle | Stock, and each other | Identical latents and pixels | One step only |

The comparison made 80 comparisons: 71 identical, 5 dp2 rows equal to stock,
3 recorded without a bar and 1 missing, with no CHECK and no FAIL. The
recorded rows are uly2 against stock running the pinned kernel at 1024, NRMS
0.0402, the stock kernel delta, NRMS 0.0404, and the edit refusal's lease row,
which holds no lease because the refusal came before dispatch. The missing row
is the pinned-kernel reading at 1536, which no stock job renders, as at every
earlier release. The Gate wrote INCONCLUSIVE `no_material` rows and
quarantined nothing.

The one-GPU edit matched stock at 1.46 times stock's time a step, because
stock caches the reference images' keys and values only in inference mode,
which a Worker's sampling thread is not in. The reference cache change,
dgx-monarch `812d011547a7` on the same stack, ran the same nine one-GPU legs
from a local-mode driver whose Workers import the driver's source. Its
review round later added the refusal off one GPU, which a one-GPU leg does
not reach. Every latent and decoded image equaled stock's: 43 comparisons, 42
identical and the stock kernel delta recorded. The edit ran 2.119 s a step
against stock's 2.142 (0.99x), and the render with no reference 1.798 against
1.784 (1.01x).

Records: the private evidence directory, record `de36a06a6d3d` of 2026-10-22
with its update receipt, comparison summaries, leg records and operator
notes, the stock records on `a071826fd`, and the reference cache record with
its comparison log.

<a id="moss-image-bias-on-comfyui-426f5eea"></a>

**The text attention bias on ComfyUI 426f5eea, 2026-10-27.** ComfyUI commit
9fe7ef764, released in v0.43.0, added `image_strength` to the Text Encode
MOSS-Image Edit node. When the text encoder sees the images, a strength other
than 1.0 attaches a bias, the log of the strength, to the images' tokens in
the prompt
([TROUBLESHOOTING #135](TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)).
Both Workers ran dgx-monarch `3438874fbf6c`, which refuses that bias under
sequence parallelism, and `dgxm doctor` reported 44 checks with 0 failures
before and after the legs. Both Sparks ran ComfyUI 0.43.0 at `426f5eea52a7`
with comfy-kitchen 0.2.40, torch 2.12.0+cu132 and torchmonarch 0.7.0, with
TORCH_FLASH. The first Spark was capped at 2100 MHz in every leg, with a
highest sampled SM clock of 2093 MHz, and the second was uncapped. Stock
references were rendered again on `426f5eea`; no candidate was scored against
a render from another ComfyUI commit.

The edit legs ran the SFT BF16 checkpoint at 1024 square, 50 Euler steps on
`simple` at CFG 3 with no skip, so cfg2 can split the passes. The edit node
encoded both the prompt and the negative with ComfyUI's bundled example
image, the ai-toolkit format, `image_strength` 0.5 and no VAE. Without a VAE
the node attaches the bias and no reference images: a CPU probe of the node's
own code found 144 entries at log 0.5 on the image's tokens of each prompt
and no reference latents, and at strength 1.0 it found no bias. Identity
means equal float32 sampler latents and equal uint8 pixels. A step's wall is
the full render at seed 43 minus the one-step render at seed 42, divided by
49; on dp2 the full render ran at seed 42, right after the one-step render.
The 50-step text-to-image row divides the warm full render by 50.

| Leg | Compared with | Result | Seconds a step |
|---|---|---|---|
| Text-to-image BF16 uly2, one step, both joint parities, skip and plain | One GPU (dp2 row 0) and stock | Identical, 3 of 3 against each | One step only |
| Text-to-image BF16 uly2, 50 steps | One GPU | Identical over the full schedule | uly2 1.205, one GPU 1.843 (1.53x) |
| Edit with the bias on uly2, after a uly2 render with the same Init | None | Refused class P (#135) at the driver before dispatch; the Workers logged nothing | None |
| Text-to-image uly2 after that refusal, same fleet, no Recycle | Stock, and the render before the refusal | Identical latents and pixels | One step only |
| Edit with the bias on ring2 | None | Refused class P (#135) at the driver before dispatch, after the Workers set up the ring2 mesh and loaded the model | None |
| Edit with the bias on `single` (`mode=local`, `gpus_per_host=1`), one step and 50 steps | Stock | Identical latents and pixels; at one step it differs from the edit at strength 1.0, NRMS 0.166 | 1.941, stock 1.934 |
| Edit at `image_strength` 2.0 on `single`, one step and 50 steps | Stock | Identical latents and pixels; at one step it differs from strength 1.0, NRMS 0.170 | Not measured |
| Edit with the bias on cfg2, one step and 50 steps | Stock on one GPU | Identical latents and pixels; at one step it differs from strength 1.0 | 0.979, stock 1.934 (1.98x) |
| Edit with the bias on dp2, batch 2, one step and 50 steps | Stock on one GPU, for row 0 | Row 0 identical latents and pixels; row 1 against row 1 of stock's batch-2 render, NRMS 0.030, recorded | 1.939 for two images, one on each GPU |

The run made 83 comparisons with no CHECK and no FAIL: 24 identity
comparisons, every one identical; 3 dp2 rows equal to stock; 4 bias renders
that differ from strength 1.0, as they must; 49 status, finiteness, refusal
and Gate checks that passed; and 3 recorded without a bar. The recorded rows
are the two refusals' lease rows, which hold no lease because each refusal
came before dispatch, and dp2 row 1. Stock shows the same batch effect: its
own batch-2 row 0 read NRMS 0.029 against its batch-1 render. The Gate wrote
INCONCLUSIVE `no_material` rows for single, uly2, dp2 and cfg2 and
quarantined nothing. ring2 had no Gate grant on this ComfyUI, so its first-use
ceremony began, stopped on the same refusal and wrote no verdict; the cfg2
and dp2 renders after it ran on the same fleet with no Recycle.

The node cannot attach an all-zero bias: at strength 0 or 1.0 it attaches
none, and at any other strength the log is not 0. The sharded forward's
handling of an all-zero bias therefore rests on CPU tests only. No leg
reached the Workers' own refusal, which needs a checkpoint header the driver
cannot read.

Records: the private evidence directory, record `3438874fbf6c` of 2026-10-27
with its comparison summary, leg records, conditioning probe and operator
notes, and the stock records on `426f5eea`.

<a id="evidence-moss-image-base"></a>

### MOSS-Image Base

**2026-10-21, as [MOSS-Image SFT](#evidence-moss-image-sft).** That record
covers this checkpoint's skip, cfg2, INT8, LoRA and scheduler results. Its own
readings follow, from the same ladder, release and cap. uly2 was identical to
one GPU and to stock at one step at 1024 square with both joint parities and
without the skip, at 1040 square with the template's prompt and at 1536
square. In Campaign K uly2 was identical to one GPU over the full 50 steps at
1024 and 1536 square, 1.48x and 1.52x faster than one GPU, and the shipped
graph through the rewrite path equaled stock over the full schedule. FSDP
rests on the sweep only: BF16 and INT8 `uly2+fsdp` equaled their resident uly2
twins, and no run_matrix FSDP cell ran Base. The INT8 file at `9caa6aae660b`
was identical on uly2 to its pin-stripped one-GPU render and to pin-stripped
stock at 1024 and 1536 square; stock's pinned kernel against the copy read
0.0401. With the Kromoss DMD2 LoRA at 8 steps, Base rendered and read NRMS
0.449 against stock with the same LoRA, recorded, and its ring2 first-use
ceremonies with that LoRA failed in the sweep, BF16 with a largest latent
difference of 2.35. Support follows ComfyUI e087d8f6c, first released in
ComfyUI 0.42.0.

<a id="evidence-kromoss"></a>

### Kromoss

**2026-10-21, as [MOSS-Image SFT](#evidence-moss-image-sft).** That record
covers this checkpoint's skip, cfg2, INT8, LoRA and #129 results. The ladder
ran `lodestones/Kromoss` at revision `7f9bb7b06d7b` (update 11000,
2026-10-20); the vendor publishes new weights under the same file names. On
2026-10-20 both LoRAs read as 756 FP32 tensors in `lora_down`, `lora_up` and
`alpha` keys on every block's q, k, v, o and feed-forward projections, rank
128 and alpha 128, and hold the same bytes as `lora.safetensors` and
`lora_dmd2.safetensors` in `lodestones/MOSS-Image-Base-8-steps-dmd`.

uly2 was identical to one GPU and to stock at one step at 1024 square with
both joint parities and without the skip, at 1040 square with both parities
and at 1536 square. In Campaign K uly2 was identical to one GPU over the full
50 steps at 1024 and 1536 square, 1.48x and 1.52x faster than one GPU, and the
shipped graph through the rewrite path equaled stock over the full schedule.
`uly2+fsdp` at prefetch depth 1 equaled resident uly2 at one step and over the
full schedule. `mode=local` equaled stock at one step. ring2 at an even joint
stream read one-step NRMS 0.0233 against stock, under the floor, and an odd
stream refused class K. cfg2 at CFG 1 on the 8-step DMD graph refused class P.

With the DMD2 and DMD LoRAs the store baked 168 patched keys per rank, and a
per-key probe found each equal to stock ComfyUI's merge at the base, DMD2, hot
swap and un-bake stages. On uly2 the 8-step graph with DMD2 at 1.0 ran 0.60 s
a step at 1024 square against 0.91 s on one GPU, and with the LoRA uly2
equaled single inside dgx-monarch. Against stock with the same LoRA it read
one-step NRMS 0.0365 (DMD2 at 0.8), 0.0489 (DMD at 1.0) and 0.0635 (DMD2 at
1.0), and the 8-step graphs, shipped and testing, read 0.36 to 0.37 after 8
steps, recorded, because stock applies the LoRA at cast time. The ring2
first-use ceremonies with either LoRA failed in the sweep, BF16 with a largest
latent difference of 1.02 and 1.69. The INT8 file with DMD2 at 1.0 read 0.985
on uly2 against stock at the first release and refuses class P (#129) from
`9caa6aae660b`; stock ComfyUI with DynamicVRAM off renders that pair 0.9977
from its own default.

<a id="evidence-longcat-image"></a>

### LongCat-Image

BF16 Ulysses matched DP2 latents at 1024 and padded 1448,
and the template probe matched one-GPU pixels exactly. The unchanged cfg2
control read NRMS 0.005598. FSDP matched resident output exactly.
CFG2 uses asymmetric-prompt padding and masks. Text is fixed-width; odd
canvases exercise the padding path. Padded ring/hybrid remains refused.

<a id="evidence-hunyuanimage-2-1-refiner"></a>

### HunyuanImage 2.1 and refiner

FP8 Image Ulysses matched DP2 latents at 2048 and 2080,
with 52/169/172 text rows, empty negative and prompt-as-negative controls.
The ring2 control read 0.015144. This does not extend to the refiner.
BF16 Image FSDP had identical resident output with a 20-step 2048 render;
FP8 stored-byte FSDP also passed its recorded first-use comparison.

The campaign testing-only FP8 base/refiner graph has 12 Ulysses+FSDP
records with both checkpoints sharded (8.1 and 7.0 GiB per rank). Every
FSDP warm output matched its resident output. This is not an independent
native comparison. Twenty resident Flash/cuDNN records completed cold then
refused warm for capacity; four lacked reference frames. All 12 resident
SAGE_AUTO records passed. SR FSDP remains unmeasured.

Plain unequal prompts work through the cfg2 text-mask equalizer; the measured
12/42 pair aligned to 48. ByT5/extended text requires equal-length Qwen
prompts. No tested ComfyUI-loadable LoRA passed a hot-swap comparison; the checked
Diffusers/PEFT artifact mapped zero patches. Earlier Image/refiner/Video/SR
acceptance included seven local-to-DP2 equal references and 16 topology
probes below 0.10, maximum 0.0825655; later records do not generalize those
specific schedules.

<a id="evidence-qwen-image"></a>

### Qwen-Image 2512 and Edit-2511

Base-2512 FP8 Ulysses matched DP2 latents exactly at 1328
and 1024, with odd/even image and text rows and batch 2. CFG2 control was
exact; ring2 read 0.051838. The base FP8 FSDP comparison matched
resident output. CFG2+FSDP was identical; DP2+FSDP at batch 2 had identical
rank-0 output and rank-1 NRMS 0.000777. Batch 1 refuses DP divisibility.
Edit-2511 FP8 mixed matched DP2 exactly under cfg2 and completed 20 steps;
its default `index_timestep_zero` reference mode refuses sequence sharding.
These are separate from Qwen Image 2.1. See [automatic kernel choices](#qwen-image-auto-sage-off).

<a id="evidence-mage-flow-t2i-edit-quality-turbo"></a>

### Mage Flow

The [native precision comparisons](#mage-flow-t2i-precision-rows) cover
T2I, Edit, Turbo and Edit Turbo only at their stated settings.
BF16 T2I Ulysses matched DP2 latents at 1024/1040 with 27/28 text rows,
long/equal prompts, batch 2, plain FP8 casting and Ulysses+FSDP. Turbo INT8
ConvRot also matched. After full-row text projection, rebuilt MXFP8 native
views and FP8 mixed changed from NRMS 0.081165/0.028193 to 0.0; NVFP4 native
views and mixed NVFP4 remained 0.0. Those four rebuilt files were removed
after testing. CFG2 control was identical and ring2 read 0.065302.
The BF16 template probe matched one GPU; warm Ulysses/single times were
14.0/18.0 seconds. Edit/Turbo and quantized FSDP, LoRA-backed slab,
other reference inputs and ring have no passing reference comparison.

<a id="evidence-qwen-image-2-1"></a>

### Qwen Image 2.1

Twelve DiT formats plus a second NVFP4 seed matched independent native
references at 1024 square, batch 1, one Euler/simple step, CFG 1, cache off
and world-2 Ulysses, on full latents and decoded pixels.
Ten-reference controls at 512 and 1024 also matched.

Full 25-step T2I, masked editing and ten-reference controls matched native
latents and RGBA pixels. Three-step cache off/default/INT8/INT4 comparisons
were exact against the matching native cache policy, using BF16 DiT and the
classic driver with dynamic memory disabled. Each enabled policy recorded
one fill and two hits per worker. The ten-reference default/auto 25-step
case used normal dynamic memory and recorded one fill and 24 hits per worker.
INT8/INT4 caching is approximate relative to unquantized caching; same-policy
agreement does not cover every cache/DiT precision combination. FSDP and
unlisted workflows remain unmeasured.

<a id="evidence-ernie-image"></a>

### Ernie-Image

BF16 Ulysses matched DP2 latents at 1024/1040 for a
conditional call, the template pair and equal-length batch-2 conditioning,
and under Ulysses+FSDP at 1024. Ring2 control read 0.025483. Template pixels
also matched one GPU, with 32.0/48.0 second warm times. FSDP matched
resident output. Shape-equal cfg2 has a 0.000 comparison. Unequal cfg2
conditioning is dispatched separately; its specific hardware comparison
and timing after the dispatch repair remain unmeasured.

<a id="evidence-z-image-latent"></a>

### Z-Image latent

Ulysses measured one-step NRMS 0.000 and completed a fresh 1536 render. Equal-token cfg2 measured 0.028 with a 20-step render. Unequal
conditioning now dispatches separately, but that path has no family-specific
hardware comparison. Effective attention masks refuse under sequence
parallelism. BF16 and Turbo BF16 FSDP matched resident outputs in the recorded comparison.

<a id="evidence-z-image-dct-pixelspace"></a>

### Z-Image DCT PixelSpace

Tests measured Ulysses 0.000 and ring2 0.015 at 1536, with a
30-step render; equal-token cfg2 measured 0.013 at 1024. These historical results do not make the
current L2P artifact runnable. It is refused before allocation on every world:
current ComfyUI detection supplies no supported L2P contract and this release
has no L2P forward. The FP32 artifact also fails the FSDP precision contract.

<a id="evidence-lens"></a>

### Lens

Shipped BF16 Ulysses probe pixels matched one GPU exactly,
with 28.0/40.0 second warm times. The 1344 pad matrix could not complete:
its driver-side 20B text encoder left too little memory and workers were lost.
This is not a passing padded-matrix result.
Asymmetric Turbo BF16 cfg2 and auto probes measured NRMS 0.0086,
50 seconds cold and 20 warm. MXFP8 cfg2 probes read 0.0 across
the three requested kernels. Earlier Ulysses comparisons read BF16
0.060-0.061, MXFP8 0.050 and Turbo BF16 0.048. BF16 and Turbo BF16 FSDP
matched resident output in the recorded comparison. Padded ring/hybrid retains its refusal.
See [latent ratios](#auto-latent-ratio) for size and kernel selection.

<a id="evidence-omnigen2"></a>

### Omnigen2

Ring2 measured one-step NRMS 0.002 against certified DP2; regular
dual-CFG completed. Its 21 heads cannot split over Ulysses 2.
Reloading changed ring attention output in the FSDP test, so ring+FSDP
refuses but the September campaign has three FP16 cfg2+FSDP records that
matched their resident outputs exactly. This does not cover ring+FSDP.
Unequal-prompt cfg2 dispatch matched the one-GPU probe exactly,
with 14.0 seconds warm versus 26.0 single. Equal-prompt timing after the
split repair was not remeasured. Two managed-residency ring cases failed
the first-use gate (maximum latent differences 0.141 and 2.08); fallback
then hit the bootstrap-policy refusal before rendering. The cuDNN case
passed with difference 0.0. This does not cover managed residency generally.

<a id="evidence-anima"></a>

### Anima

Ulysses matched a true single-GPU reference exactly; DP2 with
repaired raw T5 conditioning read NRMS 0.024 against Ulysses. FSDP matched
resident output with 118 auxiliary `llm_adapter` parameters replicated.
Explicit BF16 cfg2 at 1024, with 102/36-token prompts, matched
one-GPU probes on Flash, cuDNN and SAGE_AUTO: 12.0 seconds warm versus 20.0
single. Three cfg2+FSDP and twelve Ulysses+FSDP records matched their
resident legs. Since 2026-10-14 auto takes cfg2 for BF16 below 1.2 MP
(row 47, [Campaign K](#campaign-k-cfg2)) and Ulysses elsewhere.

<a id="evidence-boogu"></a>

### Boogu

FP8 Ulysses matched DP2 latents in all seven 1024/1040
controls after the instruct text layers used the full row set. The shipped
probe also matched one GPU, with 36.0/50.0 second warm times. Stored-byte FSDP matched resident output.
Unequal cfg2 prompts dispatch whole conditions independently; no padding is
used. The cfg2 comparison read 0.000, with 26.0 seconds warm versus
50.1 single. Equal-prompt timing after that repair remains unmeasured.

<a id="evidence-pixeldit-pid"></a>

### PixelDiT and PiD

World-2 BF16 exact-gather attention matched DP2 for all 20 PixelDiT
1024 steps under Ulysses and ring2, and for PiD's four-step 1024-to-4096
schedule with LQ features enabled. Auto chooses Ulysses.
Quantized sequence splitting and CFG parallelism refuse after MXFP8 read
NRMS 0.629 and BF16 cfg2 read 0.206. Larger worlds, PiD 1.5 and non-Flux
base checkpoints remain outside the tests.
PixelDiT FSDP matched resident output in the recorded comparison. After scalar-parameter
replication, four campaign BF16 PiD FSDP records completed cold, warm and
resident legs with identical decoded pixels. This supports those resident
comparisons, not a new native reference, quantized PiD, cfg2 or LoRA.

<a id="evidence-kandinsky5-image"></a>

### Kandinsky5 Image

Trim-forward cfg2 matched DP2 exactly and completed 50 steps. The official sampling shift is 3; auto selected Ulysses at
1024. FSDP matched resident output. Earlier family testing certified
local references against DP2 and measured Ulysses, ring, asymmetric cfg,
odd-token and Video Pro FSDP at maximum one-step NRMS 0.064. Those tests
also exercised a Video Lite LoRA, whose warm bake matched a fresh session.

## Video and audio models

<a id="evidence-ltx-2-3-t2v"></a>

### LTX 2.3 T2V

FP8-scaled auto/Ulysses completed cold and warm 121-frame renders;
the decoded one-step probe matched its local reference exactly. Native RDMA
latent return for this LTX workload is separate and remains NOT RUN / HOLD.

<a id="evidence-ltx-2-3-i2v-av-packed"></a>

### LTX 2.3 packed I2V/AV

Packed I2V/AV testing passed a local reference, direct video/audio
pair, Ulysses/ring/auto one-step comparisons below NRMS 0.10, three finite
renders, BF16 non-FSDP low-RSS rank-384 LoRA first-use checks, warm-versus-fresh
equality, lifecycle and typed guards. This is raw-latent evidence only:
no codec, decode-quality, media-output or performance claim.
Packed data parallelism refuses. FP8 FSDP is admitted as stored bytes but
has no measured fidelity. Campaign decoded AV attempts remained CHECK.

<a id="evidence-ltx-2-5-t2v-av-packed"></a>

### LTX 2.5 packed AV

World-2 tests on 22B dev/distilled INT8 ConvRot and distilled
BF16 compared one denoise step with stock one GPU. Ulysses video/audio
NRMS was 0.063/0.018 at 960x544x121 and 0.044/0.024 at 1920x1088x121;
ring2 measured 0.080/0.019. A Ulysses repeat after cache release and recycle
was identical. CFG above 1 was tested on dev; distilled LoRA was tested on
INT8 through first-use checks and a strength swap.

INT8 image-to-video, frame-zero guide and first/last-frame guides measured
video 0.058-0.099 and audio 0.019-0.054 on Ulysses/ring2, decoding denoised
output. Attenuated guides use pure Ulysses: strengths 0.7 and 0.4 measured
video 0.087/0.097 and audio 0.033/0.024 against stock. The null was identical;
strength 1 reproduced the earlier control. A dev CFG-4 guide case measured
0.048 video and 0.068 audio. Spatial masks use the same implementation, but
the campaign mask, multishot and two-stage upscaler attempts stayed CHECK.
An offline audio comparison did not change those verdicts.

Packed DP, cfg-parallel and `single` at world 2 refuse. Attenuated ring/hybrid
guides refuse. BF16 and INT8 files can pass the FSDP load checks, but no
FSDP render is recorded. BF16 LoRA could not complete its reload comparison;
two BF16 image-conditioning attempts refused after partial loading.
Duration-head and other workflows have no passing reference comparison. Stock MODEL-only guidance nodes cannot
be connected to DGXM_MODEL. LTX 2.3 results do not cover LTX 2.5.

On 2026-10-14 the four template shapes were rendered again at release 659f2bf
on ComfyUI e8665ca8 with the vendor's current Gemma 4 int8 encoder (revision
49b093a3, which replaced the encoder the August runs used). One-step Ulysses
NRMS of the video and audio latents against a stock single-GPU reference of
the same graph: text to video 0.062/0.019, image to video 0.097/0.047,
frame-zero guide 0.095/0.021, first and last frame 0.069/0.028, the same
bands as the August readings. Decoded frames and waveform were recorded but
not gated. The null pair was bit identical. Stock against stock, the encoder
change alone moved the first step by 0.25 video and 0.10 audio NRMS, so
renders made with the two encoders are not comparable.

<a id="evidence-wan-2-1-t2v"></a>

### Wan 2.1 T2V

FP8-scaled auto/Ulysses completed cold/warm 81-frame renders.
Its decoded one-step probe matched the local reference exactly. Other
precisions, FSDP and unlisted workflows do not inherit this comparison.

<a id="evidence-wan-2-2-t2v-high-low"></a>

### Wan 2.2 T2V

BF16 Ulysses high-to-low handoff at 832x480 with 21 latent
frames completed two physical updates with identical cross-rank output and
confirmed cleanup. This is separate from the later local-reference comparison.

FP16 high/low auto/Ulysses completed cold/warm 81-frame renders
and matched the local decoded one-step reference exactly. The FP8 T2V
high/low pair has not been compared; this FP16 result does not cover it.

<a id="evidence-wan-2-2-i2v"></a>

### Wan 2.2 I2V

The lightx2v high/low FP8 testing covered the official four-step
0-to-2 / 2-to-4 split at 640 square: first-use difference 0.0, identical
rank latents, spatial one-step NRMS 0.000 against DP2 and identical 17-frame
LoRA warm-versus-fresh outputs. Six load/unbake legs and two fresh loads
survived one planned process restart. No general temporal-quality, performance
or FSDP claim follows. A separate no-LoRA FP8-scaled high/low
graph completed cold/warm renders and matched a local reference on all
81 decoded one-step frames. It does not extend the four-step LoRA settings.

<a id="evidence-wan-2-1-i2v"></a>

### Wan 2.1 I2V

FP8-scaled world-2 auto/Ulysses completed cold/warm 81-frame
renders and matched its local decoded one-step reference exactly. Other
precisions, FSDP and other workflows have no passing reference comparison.

<a id="evidence-wan-flowrvs"></a>

### Wan FlowRVS

BF16 synthetic-clip auto/Ulysses completed cold/warm renders
and matched its local reference on 17 decoded one-step frames. A separate
CFG-1, no-added-noise source-latent test remains raw-latent evidence only.
Real-media segmentation quality, FSDP and other recipes have no passing reference comparison.

<a id="evidence-wan-bernini-r"></a>

### Wan Bernini-R

FP8-scaled high/low auto/Ulysses completed cold/warm 81-frame
renders and matched the decoded one-step local reference exactly. Each
stage starts fresh solver history. Continuous history, LoRA, other
precisions and FSDP have no passing reference comparison.

<a id="evidence-wan-scail-preview"></a>

### Wan SCAIL Preview

A BF16 SCAIL Preview checkpoint executed in FP16 on both
native ComfyUI and two-Spark Ulysses. The matched test used a synthetic
reference image with CLIP vision conditioning, 256x256 pixels, five frames,
one UniPC/simple step, CFG 5, shift 3 and seed 20261008. The DGX path used
explicit `uly2`, Torch Flash, stock residency, and no pose input, LoRA,
compilation or FSDP. Checkpoint and conditioning artifact identities matched
across both Sparks.

The raw latent shape was 1x16x2x32x32. Both outputs were finite and equal,
with NRMS 0.0 against the predeclared one-step limit of 0.10. Cross-rank
latents were equal. All five decoded 256x256 RGB frames were also equal;
that pixel comparison is separate from the latent acceptance. Mesh recycle and clean postflight checks were confirmed on both Sparks.
An earlier matching-output run lacked confirmed cleanup and does not count
as lifecycle acceptance.

A separate four-step control used the same geometry, seed, conditioning and
sampler settings. Its raw latents, all five decoded RGB frames and
cross-rank latents were equal. Its mesh recycle also completed. This is a
measured four-step equality result, not an extension of the one-step NRMS
limit or a general decoded-video quality threshold.

This result covers only the small synthetic one-step and four-step FP16 tests.
It does not establish accuracy for BF16 execution, other precisions, real pose-driven animation, LoRA,
ring, FSDP, warm reuse or the default workflow's 40-step, 81-frame recipe.

Campaign BF16-checkpoint auto/Ulysses cold, warm and probe runs each saved
81 decoded frames. Their local reference refused for capacity and saved no
frames. That historical comparison remains CHECK; the smaller synthetic test
does not supply its missing reference.

<a id="evidence-wan-scail2"></a>

### Wan SCAIL2

FP8-scaled synthetic-conditioning auto/Ulysses completed
cold/warm renders and matched the local reference on 81 decoded one-step
frames. Separate base-mode raw-latent testing used Turbo off, DPO strength 1,
Euler/simple, 40 steps, shift 5 and CFG 5, with warm/fresh DPO-LoRA equality.
The broader acceptance remains HELD. Arbitrary media, Turbo, FSDP and unlisted
LoRA settings have no passing reference comparison.

<a id="wan-scail2-degree-2-probe-2026-10-28"></a>

**Degree-2 probe, 2026-10-28.** A four-Spark owner reported SCAIL2 renders on
ring4, uly2+cfg2 and `auto` (uly2+ring2) outside the 0.10 one-step NRMS limit
at world 4 with no refusal, while uly4 passed
([community results](BENCHMARKS.md#community-four-node-results-2026-10-27)),
and the published video table lists only `auto` for this family. The pair ran
the sweep harness on a one-step copy of the testing-only SCAIL2 graph
(fp8_scaled, 896x512, 81 frames, TORCH_FLASH, one Euler step at CFG 5) at
world 2, with and without the DPO LoRA, scored against single-GPU references
rendered in the same run. Both Workers ran dgx-monarch `e11889da5986`; both
Sparks ran ComfyUI 0.43.0 at `426f5eea52a7` with comfy-kitchen 0.2.40, torch
2.12.0+cu132, NCCL 2.32.3 and torchmonarch 0.7.0. The first Spark was capped
at 2100 MHz and the second uncapped. The cold and warm legs ran at one step
too, so their times are not render times and no row above changes.

| Layout | Without a LoRA | With the DPO LoRA |
|---|---|---|
| uly2 | identical, NRMS 0.0 | identical, NRMS 0.0 |
| ring2 | identical, NRMS 0.0 | not scored, see below |
| cfg2 | NRMS 0.116, over the floor | NRMS 0.085, under the floor, not identical |

cfg2 renders differently from one GPU in both readings, with a largest pixel
difference of 255, where every other family with a cfg2 record renders
bit-identical, so the family refuses cfg-parallel from this release
([TROUBLESHOOTING #136](TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)).
The ring2 cell with the LoRA rendered its cold and warm legs in both of two
sessions and refused its probe leg class C both times: the cross-rank
capacity agreement priced the one-step re-load while rank 0 on the driver host
still held the resident it would replace
([#93](TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-rank)),
so that cell carries no fidelity reading. On 2026-10-28 the four-Spark owner
published their world-4 probe frames and per-frame NRMS in
[issue #100](https://github.com/Deen-Media/dgx-monarch/issues/100): uly4
bit-identical to one GPU, ring4 at 0.117, uly2+ring2 at 0.153 and uly2+cfg2
at 0.116246, which equals the world-2 cfg2 reading without a LoRA above to six
digits; the ring difference images show a vertical seam and the cfg2 error
spreads over the frame. NOT RUN: SCAIL Preview on cfg2
(refused alike, since it shares the forward), every layout at world 4 (no
four-node rig here), and the 40-step recipe under cfg2.

<a id="evidence-wandancer"></a>

### WanDancer

The broader acceptance remains HELD. Global-to-local
world-2 FP8 two-pass `uly2` testing returned 298 healthy
frames using image/audio conditioning and the stock decode/keyframe bridge.
Recomputed segment counts covered all three sampler executions with identical
cross-rank latents. First-use checks were
INCONCLUSIVE/no-material, so residency remained unchanged. `reference_latent`
input refuses. FP8 FSDP admission is unmeasured. The separate campaign
testing-only sample rendered cold but refused warm/probe for capacity,
retaining FINDING. Neither test establishes other topologies or performance.

<a id="evidence-wan-animate-2"></a>

### Wan-Animate 2

See [native controls](#wan-animate-2-controls) and [warm iteration](#wan-animate-2-warm-iteration).
Base precision tests, Base/Distilled portrait tests and official INT8
ConvRot with LightX2V have separate comparisons. Ordinary-residency INT8
and BF16 three-queue workflows retained models through full native decode.
Cold outputs matched their earlier DGX results; changed seeds/prompts
produced distinct finite outputs with identical rank latents. These are
warm-execution observations, not new native-reference comparisons or a
long-duration memory soak.
Release-before-decode is an explicit alternative that reloads on the next
queue. Slab/low-RSS has one later cold first-use PASS, without warm-session testing.
FSDP, ring and caches have not been tested on hardware. Other inputs need their own comparisons.

<a id="evidence-hunyuanvideo-1-5-sr"></a>

### HunyuanVideo 1.5 and SR

FP16 Video Ulysses at 1280x720, 121 frames and 77 text
rows matched DP2 one-step latents exactly. The campaign decoded local-reference
probe read NRMS 0.049519. SR does not inherit either result. Earlier testing
included bounded 720p/1080p SR schedules, but SR FSDP remains unmeasured.
FP16 Video and stored-byte FP8 SR FSDP admission is not a measured FSDP
render. CFG2 is plain-text only; split-custom SR needs explicit topology.

<a id="evidence-cogvideox-1-5-t2v"></a>

### CogVideoX 1.5 T2V

An earlier Ulysses comparison measured one-step NRMS 0.082 against certified DP2;
an 832x480, five-latent-frame render completed. BF16
world-2 auto/Ulysses completed 48-frame renders and read decoded probe
NRMS 0.042457 against a local reference. Conditioning has 226 tokens.
The tested 1.0-layout LoRA skipped all 1,344 keys on 1.5, so it did not modify the model.

<a id="evidence-cogvideox-1-5-i2v"></a>

### CogVideoX 1.5 I2V

ComfyUI-layout BF16 auto/Ulysses completed cold/warm
45-frame renders. Its decoded one-step probe read NRMS 0.015932 against
the local reference. Diffusers-layout loading and inpaint are not covered.

<a id="evidence-cogvideox-1-5-inpaint"></a>

### CogVideoX 1.5 inpaint

Inpaint has no separate checkpoint/workflow accuracy comparison. I2V
evidence does not cover it; no inpaint template is supplied.

<a id="evidence-kandinsky5-video-lite"></a>

### Kandinsky5 Video Lite

The official sampling shift is 5. BF16 auto/Ulysses completed
cold/warm 121-frame renders and matched its local decoded one-step reference
exactly. Earlier Video Lite LoRA testing matched warm and fresh bakes.

<a id="evidence-kandinsky5-video-pro"></a>

### Kandinsky5 Video Pro

Testing of the 4096-dimensional model passed under explicit FSDP
with a full render and an exact one-step reference comparison. FSDP LoRA
is admitted under low-RSS but unmeasured. Campaign resident testing rendered
121 frames but stayed CHECK because the local reference had no frames;
that attempt does not extend the earlier FSDP result.

<a id="evidence-minimax-h3-t2va-fl2va-ref2va-av"></a>

### MiniMax H3

Carried-audio testing compared Ulysses t2va at 512x320x5:
even packed totals were identical and odd padded totals read NRMS 0.000
per stream. BF16, pruned INT8 ConvRot and FP8-scaled, both timestep
regimes, both parities, fl2va and ref2va had per-stream 0.000 comparisons.
Even-total ring2 read 0.023 video and 0.072 audio. T2I uses the same
five-frame grid floor; other shapes are not implied by that fact.

Pruned INT8 guide tests matched stock one GPU on both latent
streams: a still at frame 62, soundtrack at frame 24, both together, and a
five-frame 448x256 clip. A null pair was identical across processes and
CPU guide accounting matched 16 native cases. Even-total ring2 still-guide
NRMS was 0.028 video/0.055 audio. Odd totals refused. Chained guides,
clips beyond 39 frames, hybrid and guide decode fidelity have no passing reference comparison.

H3 packs text, references, audio and video into one sequence. CFG above 1
uses sequential conditional calls; cfg2 cannot split a batch and refuses.
Packed DP, including `single` at world 2, refuses. Auto chooses Ulysses;
a true one-GPU reference uses local mode with one GPU. Padded ring2
requires an explicit accuracy waiver; accepting it does not establish accurate output.

BF16 Ulysses+FSDP rendered in 141.8 seconds after the text preprocessing
modules stayed replicated: 31.6 GiB weights per rank, 19 auxiliary parameters
and 12 replicated FP32 islands. The render completed, but no reference
comparison or first-use check passed. Pruned INT8 FSDP
is admitted but unmeasured. Stacked/PDD output heads, Fun ControlNet model
patches and unlisted workflows have no pair comparison; Fun ControlNet
cannot be wired through the DGXM_MODEL interface.

Compatibility testing of xFuser 0.4.5 against
`0.7.0+dgxm.npuimport1` reproduced local and Ulysses Config-A outputs:
pruned FP8, 512x320x5, one step, CFG 1, TORCH_FLASH, no guides/LoRA/FSDP/
Sol-Attn. Both latent streams, five decoded frames and 32-kHz stereo PCM
(6400x2) matched. This is dependency compatibility, not a broader quality
result. Campaign H3 AV records remained CHECK even after offline audio
comparisons; see [campaign limitations](#campaign-summary).

A separate 1280x720x241, 20-step `res_multistep`/simple, seed-42, CFG-1
INT8 ConvRot recipe measured 42:03 in solo ComfyUI, 42:20 through one
Monarch worker and 23:07 on the exact-attention Ulysses pair. Single-worker
and pair output matched in that comparison. Sage took 17:21, with different
numerics. These timings belong to that recipe, not the small Config-A
comparison or the campaign AV CHECK records.

Sol-Attn stays off by default and requires an accuracy waiver at every named
tau (0.6, 0.7, 1.0). A pure-sparse diagnostic build measured one-step NRMS 0.223-0.290,
above the 0.10 limit. The shipped dense-first recipe keeps the entire one-step
probe on exact attention, so that instrument never reaches its sparse path.
`SOL_VALIDATED_TAUS` remains empty for both reasons; the diagnostic's failed
numbers are not measurements of the shipped one-step recipe. Recipe-faithful CuTe times at 1280x720x241
were 15:39-16:19, 2.58-2.69 times faster than solo and 1.42-1.47 times
faster than exact-attention pair timing on that one prompt/seed/shape.
A faster diagnostic used a different schedule and is excluded. Backend
identity matters: without CUTLASS DSL the optional package may use Triton.
See [H3 capacity](#h3-capacity) before choosing BF16 reference-to-video or guides.
## Mage Flow T2I precision rows

Independent native comparisons used 1024 square, batch 1,
seed 20261003, one Euler/simple step, CFG 5 and Ulysses world 2. The latent
limit was NRMS <=0.10; pixel measurements are reported separately. Formats
ran across the applicable fixes, not as one final-build matrix.

| Format | Latent NRMS | PNG NRMS |
|---|---:|---:|
| BF16 | 0.048502 | 0.041023 |
| FP16 | 0.048090 | 0.040142 |
| Plain FP8 E4M3 | 0.047023 | 0.026964 |
| Plain FP8 E5M2 | 0.041977 | 0.034699 |
| Scaled FP8 E4M3 | 0 | 0 |
| FP8 mixed | 0.028193 | 0.065324 |
| INT8 | 0 | 0 |
| Official INT8 ConvRot | 0 | 0 |
| Mixed INT8 ConvRot | 0.071626 | 0.056535 |
| MXFP8 native views | 0.081165 | 0.067526 |
| NVFP4 native views | 0 | 0 |
| Mixed NVFP4 | 0 | 0 |

Full-recipe 1024-square native pairs were descriptive comparisons:

| Workflow | Steps / CFG | Latent / RGB NRMS |
|---|---|---|
| RL quality T2I | 30 / 5 | 0.029297 / 0.013133 |
| Edit, one reference | 30 / 5 | 0.017874 / 0.009655 |
| Turbo | 4 / 1 | 0.029181 / 0.015917 |
| Edit Turbo | 4 / 1 | 0.006628 / 0.005640 |

No multi-step threshold is claimed. Base BF16 T2I without LoRA passed two
independent FSDP clean reloads and the final render at latent/PNG NRMS
0.048502/0.041023, without fallback. Edit/Turbo FSDP, quantized FSDP and
LoRA-backed slab remain unmeasured. Later pure-Ulysses results are listed
in the [model entry](#evidence-mage-flow-t2i-edit-quality-turbo).

## Wan Animate 2 controls

Base tests used 320 square, five frames, batch 1, one
Euler/simple step, CFG 1, seed 20261003 and world-2 Ulysses against
independent native references. Every compared output had readable H.264
video and AAC stereo audio.

| Base format | Latent / PNG result |
|---|---|
| BF16, FP16, plain FP8 E4M3/E5M2, scaled FP8 E4M3, FP8 mixed, INT8, official and mixed INT8 ConvRot, mixed NVFP4 | Identical latents and all five frames |
| MXFP8 native views | NRMS 0.027134 / 0.019240 |
| NVFP4 native views | NRMS 0.089014 / 0.054633 |

Portrait tests used 480x832x81, batch 1, seed 20261003, CFG 1, shift 5,
world-2 Ulysses and ordinary residency with automatic first-use checks off:

| Recipe | Native comparison |
|---|---|
| Base BF16, 20 Euler/simple steps | Identical final latent and all 81 decoded frames |
| Distilled BF16, 10 LCM/simple steps | Identical final latent and all 81 decoded frames |
| INT8 ConvRot + LightX2V strength 1, one step at 320x320x5 | Latent NRMS 0.014074, below 0.10 |
| INT8 ConvRot + LightX2V strength 1, six LCM/simple steps | Latent 0.143221 / PNG 0.093584, descriptive only |

The LoRA was active on both accelerated legs. A BF16 accelerated control
stopped at an external 24-GiB available-memory limit and cleaned up normally;
it did not prove a product memory-pricing fault. Slab/low-RSS initially fell
back to stock and remained RETESTING. These tests do not cover arbitrary
continuations, caches, FSDP, ring or unlisted shapes and LoRAs.

## Wan Animate 2 warm iteration

Three queues ran the 480x832x81 six-step LCM/simple recipe
with official INT8 ConvRot or Base BF16 plus LightX2V rank64, strength 1,
CFG 1 and shift 5. Ulysses used TORCH_FLASH. Ordinary weight/LoRA residency
and campaign `auto_gate=off` were explicit. Actors and model weights stayed
resident through native full decode. Actor import sources were verified;
an earlier release-before-decode test had matching checkouts but a stale
peer import, so its cleanup observations are not uniform-source proof.

| Precision | Cold / changed seed / changed prompt wall | Minimum available RAM, first / second Spark |
|---|---|---|
| INT8 | 292.33 / 250.32 / 266.30 s | 45.16 / 67.58 GiB across the three queues |
| BF16 | 288.30 / 238.23 / 252.27 s | 18.45 / 36.41 GiB across the three queues |

INT8 used 24-GiB driver and Init reserves. BF16 used 8 GiB and
`ClearVRAM(level=soft, include_driver=true)` before full VAEDecode, retaining
30.54 GiB of weights per actor. All queues returned 81 H.264 frames with
AAC audio, distinct finite latents of shape `[1,16,22,104,60]`, and identical
rank latents. Cold latents/pixels matched the preserved earlier DGX output.
Changed-input queues establish reuse, not a fresh independent fidelity test
or indefinite soak. The public BF16 template leaves Init reserve at zero;
the measured 8-GiB control does not count as a run of that template unchanged.

Tiled BF16 decoding at spatial 256/64 and temporal 128/8 completed, but
pixels differed from untiled decode (NRMS 0.0044613). The full decoder
worked with soft cache cleanup, so the tiled diagnostic is not a promoted
workflow. Release-before-decode remains a measured explicit alternative,
with fresh setup/load on the next generation.

The INT8 template ran unchanged with both residency levers off,
Init reserve 0.0 and driver `--reserve-vram 2`. A cold queue took 294.1 s;
a new session's changed-seed queue took 302.0 s and its warm changed-prompt
queue 258.6 s. The first attempt lost the peer mid-render, so these are not
three uninterrupted warm queues. Cold latent output matched the earlier
control. With no risky residency/FSDP setting, first-use mode ran no proof.

A separate cold queue with slab and low-RSS both set to auto passed its
first-use identity check: 458.0 s total, including 185.7 s for two proof
renders; minimum available memory was 62.7/81.4 GiB. It recovered about
16 GiB on the driver host, but three consecutive warm queues remain untested.
The shipped INT8 template keeps those levers off.

## Ulysses comparisons

Ulysses preserves ComfyUI's joint-token order and attention inputs. Text
projections use full text rows before sharding where the model requires it.
Tests used TORCH_FLASH with SAGE off and same-build DP2 sample-0 references.
All eleven completed baselines were unchanged between the two comparison
rounds. Boogu was tested with its mixed-precision instruction layers, and
Mage Flow mixed formats included unquantized text layers. The model entries
list the tested shapes and exceptions.

Template one-step decoded comparisons against same-build one-GPU graphs:

| Model / format | Pixel result | Warm Ulysses / one GPU |
|---|---|---|
| Ernie BF16 | Identical | 32.0 / 48.0 s |
| HunyuanImage 2.1 FP8 | Identical | 56.1 / 80.1 s |
| Ideogram4 BF16 / FP8 / INT8 dual model | Identical on each | 34.0 / 56.1; 26.0 / 36.0; 28.0 / 44.2 s |
| Krea2 BF16 / INT8 | Identical on each | 48.1 / 74.1; 48.1 / 80.1 s |
| Mage Flow BF16 | Identical | 14.0 / 18.0 s |
| Qwen-Image 2512 FP8 | Identical | 216.2 / 268.3 s |
| Boogu FP8 | Identical | 36.0 / 50.0 s |
| Lens BF16 | Identical | 28.0 / 40.0 s |

Warm full-render times do not extend one-step equality to every denoise
step. Ring controls retained nonzero differences because these changes
apply only to pure Ulysses. Krea2 FP8 SAGE, Hunyuan refiner/SR and unlisted
formats have no new result from this matrix.

<a id="padded-ulysses-index-cache"></a>

### Padded Ulysses index cache (2026-10-14)

**October 14, 2026:** a back-to-back A/B timed the change that builds the
padded Ulysses path's index tensors once per layout and reuses them from a
small cache ([ADAPTERS.md](ADAPTERS.md#what-an-adapter-provides)). Before it,
the path read the device once per dropped row in the pad-row remap and once
more in each keep index's `nonzero`: about six host waits per attention call,
roughly six thousand in a 20-step Chroma render. `index_select` and the
scatter receive the same integers after the change, so the bar was speed with
identical pixels: the padded cell had to gain at least 1 percent beyond
noise, and the unpadded cell had to stay within noise.

Leg A ran the installed dev head `e436b53`, and leg B the candidate `0bfc6f0`
deployed through the verified update. The legs ran back to back in fresh
driver sessions, with both Sparks on ComfyUI `e8665ca8` and torchmonarch
0.7.0 and the first Spark capped at 2100 MHz. Both legs rendered Chroma1-HD
FP8-mixed on uly2 with Torch Flash, 20 steps and CFG 3.5, at 1152 square,
where the Ulysses path pads no rows, and at 1176 square, where it pads and
drops rows. Each cell took two warm-up renders, then seeds 101 to 105, then
one repeat render as an identity control. The sampler wall is the mean of the
five seeds, and the spread is the largest minus the smallest. The
whole-prompt wall adds text encoding and VAE decode.

| Cell | Sampler wall, A (spread) | Sampler wall, B (spread) | B against A | Last step, A / B (s per step) | Whole prompt, A / B |
|---|--:|--:|--:|--:|--:|
| 1152 square, no pad rows | 23.16 s (0.10) | 23.26 s (0.10) | +0.43% | 1.158 / 1.163 | 24.73 / 24.81 s |
| 1176 square, padded | 25.00 s (0.20) | 24.44 s (0.10) | -2.24% | 1.250 / 1.222 | 26.54 / 25.98 s |

Leg A and leg B decoded identical PNGs for all ten seed pairs under numpy
equality, and each leg's repeat render equaled its first. Neither leg logged
a driver ERROR line. The padded cell gained 0.56 s against spreads of 0.20 s
and 0.10 s, which clears the bar, and the unpadded cell moved 0.10 s, inside
its spreads, so the change ships. An earlier run with the legs two hours
apart read -1.07% on the padded cell and +1.15% on the unpadded one; this
back-to-back run supersedes it. Other families that take the padded path,
and other kernels, have no timing.

<a id="auto-latent-ratio"></a>

## Latent ratios and automatic topology

Automatic topology uses ComfyUI's detected spatial downscale ratio instead
of a fixed 8-pixel estimate; megapixels mean each frame's
sampled spatial grid, not the compressed latent area.

| Ratio | Families checked in checkpoint headers |
|---:|---|
| 1 | Radiance, Z-Image DCT, PixelDiT |
| 8 | Chroma, Flux 1, LongCat, Qwen-Image, Krea2, Z-Image latent, Kandinsky5 Image, Wan 2.2 14B |
| 16 | Flux2, Lens, Ideogram4, Ernie, Mage Flow, Qwen Image 2.1, HunyuanVideo 1.5, H3, FLUX.3 Klein, MOSS-Image |
| 32 | HunyuanImage 2.1, LTX 2.3 |

Lens 1344 square is 1.81 MP; Flux2 1024 square is 1.05 MP. On 2026-10-15
ComfyUI's own detection named all six FLUX.3 Klein vendor files `Flux3` with
ratio 16 from their headers, and on 2026-10-20 all six MOSS-Image and Kromoss
DiT files `MOSSImage` with ratio 16.
These dimensions use the detected ratios; they do not imply new hardware tests.
Lens FP8 auto keeps SAGE off: at the tested size SAGE/Flash/cuDNN all passed
(NRMS 0.036869/0.026198/0.032733), with 20.3/20.0/20.1-second warm times,
so no SAGE speed benefit was measured. Explicit kernel choices still apply.
Lens and Mage Flow checkpoint detection is distinct despite shared modules.
This was analysis of existing records and headers, not a new hardware run.

<a id="qwen-image-auto-sage-off"></a>

## Qwen-Image automatic kernel choice

Base-2512 FP8 at 1328 square measured SAGE Ulysses NRMS 0.16219,
above 0.10, versus Flash 0.082111 and cuDNN 0.081765. Ring SAGE read 0.16633
and ring Flash 0.078691. CFG2 read 0.0 across all three requested kernels.
Auto therefore keeps the requested kernel on Ulysses instead of switching
to SAGE. The model entry also records the newer Flash comparisons.

The LoRA graph failed these comparisons: its probes read SAGE auto 0.311584,
Flash Ulysses 0.150059 and cuDNN Ulysses 0.146393, all CHECK. CFG2 read 0.0
on all kernels and ring2 refused for capacity. Explicit SAGE remains selectable.

<a id="campaign-k-cfg2"></a>

## cfg2 crossover rows (Campaign K, 2026-10-14)

Campaign K compared cfg2 and uly2 with one GPU on 34 cells. Nine ran at
template sizes: the Krea2 template on four checkpoints plus a CFG 4.5
control, and the Qwen-Image, Kandinsky5, Boogu and Anima templates.
Twenty-four ran at 1152, 1216 and 1280 square: Chroma BF16, FP8-mixed and
INT8 ConvRot, LongCat, Krea2 Turbo FP8, RAW BF16 and RAW INT8 ConvRot, and
Qwen-Image. One more ran Krea2 Turbo FP8 at 1408. The cells ran
dgx-monarch `d549f10d` (Anima: `65d14333`) and ComfyUI `e8665ca8`, with the
first Spark capped at 2100 MHz. The reference is batch
row 0 of a same-build dp2 render of batch 2, which carries one-GPU math.
[Benchmarks](BENCHMARKS.md#campaign-k-cfg2-against-uly2-2026-10-14-head-capped)
lists every cell's walls, speedups and fidelity.

- **Bit-identical.** cfg2's full-schedule latent and its one-step latent
  equaled the reference exactly for Krea2 RAW BF16 and RAW INT8 ConvRot at
  1152, 1216, 1280 and 1536, Turbo BF16 at 1536, Krea2 Turbo FP8-scaled at
  1152, 1216, 1280, 1408 and 1536 (CFG 1 under `euler_cfg_pp`) and at 1536
  under CFG 4.5 Euler, Chroma1-HD BF16, FP8-mixed and INT8 ConvRot at 1152 to
  1280, Qwen-Image 2512 FP8 at 1152 to 1328, and Kandinsky5 Lite image, Boogu
  FP8 and Anima BF16 at 1024. uly2 on Torch Flash was bit-identical in every
  cell.
- **Krea2 RAW INT8.** cfg2 was exact at every size but no faster than uly2
  below 1536: 0.97x at 1152, 0.98x at 1216, 1.03x at 1280 and 1.08x at 1536.
  Exact is not enough for a row, so row 15 grants FP8 and BF16 only and INT8
  keeps row 12.
- **LongCat.** cfg2 read 1-step NRMS 0.006 to 0.007 at 1152 to 1280, and its
  full-schedule latent differed from the reference (largest absolute
  difference 2.65 to 7.62). That is under the 0.10 one-step floor, but uly2
  read bit-identical on the same cells, so no LongCat cfg2 row covers 1.2 MP
  and up. Row 28 keeps its 2026-09-06 record below 1.2 MP.
- **SAGE on Ulysses.** The route auto took for Krea2 Turbo FP8 (row 11) and
  Chroma FP8-mixed (row 21) above 1.2 MP read 1-step NRMS 0.030 to 0.058 and
  0.057 to 0.078 against one GPU. Both are under the floor, and neither full
  schedule equaled the reference.
- **Rows.** The bit-identical bands that beat uly2 at every measured size
  became automatic rows: 15 for Krea2 FP8 and BF16 from 1.2 to 2.36 MP, 19 for
  Chroma BF16, FP8 and INT8 from 1.2 to 1.64 MP, 68 for Qwen-Image FP8 from
  1.2 to 1.77 MP, and 79 for Boogu FP8 and 47 for Anima BF16 below 1.2 MP.
  Each record grants only the precisions that read bit-identical and beat
  uly2 at every measured size in its band. Kandinsky5 Lite image met that bar
  at 1024, but its image and video checkpoints share one family, so it has no
  row.

Not measured: MXFP8 and NVFP4 Krea2 files, Chroma
FP8-scaled and Radiance above 1024, Qwen-Image Edit-2511, equal prompt pairs on Krea2, Boogu
and Anima, which fold onto the slice path, and world 4. Each can still reach a cfg2 row
through a shared family and precision class: MXFP8 and
NVFP4 Krea2 files reach row 15 (in September, RAW NVFP4 cfg2 read 1-step NRMS
0.0113), Chroma FP8-scaled and Radiance reach row 19, an Edit-2511 FP8 file
would reach row 68, and an equal pair takes the slice path on rows 15, 79 and 47.
Kroma BF16 was measured on 2026-10-16 and row 15 excludes it
([Kroma cfg2](#kroma-cfg2)).
Where an equal pair was measured, the slice path stayed under the floor but
did not match one GPU: Chroma BF16 read NRMS 0.034 ([Chroma](#evidence-chroma)),
FLUX.3 Klein Base 0.009 to 0.031
([FLUX.3 Klein Base](#evidence-flux-3-klein-base)), Kroma BF16 0.009 to 0.011 and
MOSS-Image SFT with its skip off 0.023 to 0.026
([MOSS-Image SFT](#evidence-moss-image-sft)).
At world 4 a batch-1 render on a cfg row refuses class K. An evidence record
has no CFG field, so the Krea2 record covers every CFG other than 1, and CFG 1
under a CFG++ sampler that a DGX Monarch KSampler node names, on the strength
of the two regimes measured. A Sampler Custom graph names no sampler, so at
CFG 1 it keeps Ulysses.

## INT8 GEMM kernel comparison (2026-10-28 and 29)

The INT8 ConvRot checkpoints run their linears through comfy-kitchen's
`cutlass_int8_dequant` kernel. kan11's
[gb10-comfyui-tuning](https://github.com/kanthai/gb10-comfyui-tuning/tree/d68916bf06c7ab01e0c557c67c64a1a06eb0ceef)
replaces that one call with a Triton GEMM tuned for GB10, with the row and
column scales and the bias in its epilogue (comfy-kitchen#280 carries the same
proposal upstream). This leg measured that kernel on the pair as the basis for
the v1.5 option in #93: exactness first, then speed.

**Kernel equality.** 156 synthetic cases over 13 shapes (MiniMax H3 at 29184
rows per rank, Krea2 and Qwen stills, odd rows such as 127, 129, 1095 and 2049),
three output dtypes, with and without a bias, per-channel and per-tensor weight
scales: every case `torch.equal` against the CUTLASS result. The kernel leaves
on CUTLASS the fused-residual calls, rows under 128 and every call with K not a
multiple of 16 or a strided input. Upstream found CUTLASS declining the last
two; on comfy-kitchen 0.2.40 it takes them itself (the 2026-10-29 check below),
so they keep the stock route either way.

**Renders.** The sweep harness ran the INT8 ConvRot checkpoints of fifteen
templates twice on the same build (v1.4.0, ComfyUI 0.43.0, comfy-kitchen
0.2.40, torch 2.12.0, NCCL 2.32.3): once stock, once with the kernel installed
in every process (driver and actors) through the venv and switched on by its
mode file; the actors took it, as their journals and the sibling's fresh Triton
cache showed. Layouts: one GPU, uly2 and, where the template has a CFG, cfg2.
Legs: cold, warm and the one-step probe. Every output of the two passes was
compared file by file, in counter order per file type (an audio file saved
first shifts ComfyUI's frame counter by one), decoded pixels for PNG frames and
decoded samples for audio:

- 1530 of 1530 frames pixel-identical across the 28 cells: every still and
  every video frame of MiniMax H3 (guide and image-to-video), Wan Animate 2 and
  LTX 2.5, on one GPU and on uly2. The two mp4 templates, whose save node the
  harness cannot rename, were paired by counter as well: 12 of 12 pairs frame
  for frame identical.
- Audio: 9 of 12 tracks sample-identical. The three that differ are MiniMax H3
  tracks, in legs whose frames are identical, by at most 1e-3 of full scale and
  in different legs each time. That is the audio decode, not the GEMM: a kernel
  difference would move the latents and the frames with them.
- Harness verdicts matched between the passes (PASS, or CHECK where a family's
  single-GPU INT8 reference is refused under the checkpoint pin and the uly2 cell
  has nothing to compare against: FLUX.3 Klein, MOSS-Image, Kromoss).

**Coverage.** 15 of the 17 INT8 ConvRot files on the rig. Not run: the unpruned
H3 fl2va file and the H3 ref2va file, which share the pruned H3 file's GEMM
shapes. The block-scaled files (MXFP8, NVFP4) are not INT8 and were not part
of the comparison.

**Speed.** The warm legs in docs/BENCHMARKS.md ("INT8 GEMM: CUTLASS against the
Triton kernel"): 16 to 25 percent faster on MiniMax H3 and Wan Animate 2, 5 to
9 percent on LTX 2.5, 4 to 14 percent on Ideogram4 and Chroma, no change on
Krea2, Qwen Image 2.1 and the small turbo stills. The harness times at 2 s granularity. A first stock pass had read
Krea2 on one GPU at 86 s against 94 s under the kernel; the stock re-render
read 94 s, so that was run-to-run variance, not the kernel. At the GEMM level
the two routes are within 8 percent of each other either way below 4096 rows
(the worst reading 0.92 on the Ideogram4 qkv linear at 1024 rows), the kernel
is at or above CUTLASS from 4096 rows up on every shape but one reading
(Ideogram4 adaln at 8192 rows, 0.96), and from 6144 rows up CUTLASS falls to 37
to 55 percent of its peak while the Triton kernel holds about 175 TOPS. The
small-M losses sit on calls that are a small share of a render, and no measured
render came out slower, so no M threshold or family gate is needed for speed. The head ran its 2100 MHz clock cap throughout; kan11 reports the same
18 percent H3 gain uncapped.

**What this does not cover.** The kernel ran as a process-wide patch, not yet
as a Monarch option; the option's wiring, its gate and its documentation are
#93 item 1. torch.compile paths were not exercised (compile_dit off). The
MXFP8 and NVFP4 single-GPU references rendered in the same stock pass as a load
test of their recast files and are not part of the kernel comparison.

**The option's GPU check (2026-10-29).** With the kernel vendored as
`src/dgx_monarch/kernels/int8_gemm_triton.py`,
`tests/test_int8_gemm_triton_equality.py` ran 240 cases over 20 shapes on the
head against its own CUTLASS kernel (comfy-kitchen 0.2.40, torch 2.12.0,
Triton 3.7.0, GPU idle, the 2100 MHz cap on): the 13 shapes above, M at
exactly 128, H3's odd 29221 rows per rank, the Ideogram4 adaln and qkv and
Chroma linear1 linears, and two shapes for the kernel's masked tiers, each in
three output dtypes, with and without a bias, per-channel and per-tensor
weight scales. All 240 were `torch.equal` called directly, and all 240 through
the proxy a render calls, which ran the Triton kernel on the 228 cases at 128
rows or more and left the 12 at 127 rows on CUTLASS. The same run's decline
check found comfy-kitchen 0.2.40's CUTLASS taking K = 1000 and a strided
activation itself; the proxy handed both calls to CUTLASS unchanged, and the
Triton kernel ran on neither. The option's wiring (the
knob, the install points in the driver and the actors, and the fallbacks) had
CPU tests only at that point; the smoke below ran it on hardware.

**The option's install points on hardware (2026-10-29).** With the merged
option (dev `7404677`) verified-installed on both Workers, one short render of
the H3 guide graph (pruned INT8 ConvRot DiT, NVFP4 AWQ text encoder, 29 frames,
4 steps, a fresh seed each time) ran per process kind, and the `INT8 GEMM:`
mode line was read from the driver log and both Worker journals:

- a local actor on the head, everything at the default: 102.1 s; driver and
  actor logged `Triton kernel on ... (int8_gemm=triton, from the default)`;
- uly2 on the pair at the default: 108.1 s; all three processes Triton on;
- uly2 with `DGXM_INT8_GEMM=cutlass` on the driver: 78.1 s; the driver logged
  `stock comfy-kitchen 0.2.40 CUTLASS kernel (int8_gemm=cutlass, from
  DGXM_INT8_GEMM)`, both actors Triton on;
- uly2 with `int8_gemm = "cutlass"` under `[worker_args]`: 138.1 s; the driver
  Triton on; on each Worker the root actor logged Triton on at service start
  and its GPUWorker then logged stock CUTLASS `from [worker_args]`, the
  in-place switch.

No fallback WARNING in any process. Every render ran on stock residency with
the identity gate INCONCLUSIVE (nothing to gate: no LoRA stack and slab
residency not engaged) and the artifact's accuracy waiver stamp, as the
kernel comparison's renders did. The cluster file was restored byte for byte
afterwards. One attempt is not counted: a local-layout render asked for 15 s
after the previous driver exited, while the head's MemAvailable was still
60.6 GiB, and the loader footprint preflight refused the DiT with its typed
class C message; the leg reran on uly2 after a memory wait. The walls are
whole renders of a 29-frame, 4-step cell and compare with nothing above.

## FP8 compute paths

**Status: NOT RUN.** No render in this record has run ComfyUI's `fp8_ops` path
on GB10, in the driver or in an actor, and no render has run with
`[worker_args] fp8_matrix_mult` set. The rows below are the leg that will
measure it; each reads NOT RUN until a dated run replaces it.

ComfyUI 0.43.0 picks a model's operations class when it builds the model
([MODELS](MODELS.md#fp8-compute-paths)): a checkpoint with comfy-kitchen
metadata takes `mixed_precision_ops` whatever the flags say, and a plain FP8
file takes `fp8_ops` (`torch._scaled_mm`) only under `--fast fp8_matrix_mult`
or the loader's `fp8_e4m3fn_fast`, and `manual_cast` (dequantized to BF16)
otherwise. The actors start ComfyUI without its argument list, so the flag
reaches only the driver's own loads, and `fp8_matrix_mult` is the actors'
switch. On 2026-10-31 `tools/fp8_paths.py` read every candidate below from its
header on the head and confirmed its path, and its rule check passed against
the ComfyUI 0.43.0 checkout (426f5eea5). Two of the brief's candidates are
not FP8 compute files at all: the Wan 2.1 T2V 14B fp8_scaled file marks all
406 of its layers `full_precision_matrix_mult`, so it computes in BF16 under
every setting, and the Krea2 fp8_scaled files mark 96 of their 256 layers so.
Qwen-Image 2512, HunyuanImage 2.1 and Wan 2.1 T2V 14B have no BF16 file on
the head; their cells compare against the same file without the feature, and
the Krea2 BF16 cells give the FP8 paths a BF16 reference on the same weights.

**Method.** One driver session per cell, started through the maintained
launcher with `--fast fp8_matrix_mult` where the cell names it and never with
`--bf16-unet`, which upcasts FP8 at load. The cell sets `fp8_matrix_mult` in
the cluster file for that session and restores the file afterwards. Each cell
renders its template once in full and once cut to one step, at a fixed seed
per family, and keeps the latent and the decoded pixels. Fidelity is the
one-step NRMS (rms of the difference over the reference's standard
deviation, the 0.10 floor of `benchmark/gates.py`) and the pixel NRMS of the
full render, against the BF16 control where one exists and against the same
file and layout at the baseline setting. Speed is the wall and sampling time of
the full render. 1 Hz samplers record MemAvailable and each GPU process on
both hosts. The `FP8 compute:` lines of the driver log and both Worker journals
name the class each build picked, so every row states which path it measured.
The head runs its 2100 MHz cap. `auto_gate` and `slab_weights` are off: the
leg measures compute paths on stock residency, not residency.

| Group | File | Loader weight_dtype | `--fast fp8_matrix_mult` | `fp8_matrix_mult` | Layouts | What it measures | Status |
|---|---|---|---|---|---|---|---|
| A | Qwen-Image 2512 FP8 | default | off | auto | single, uly2, cfg2 | the dequantizing path, the baseline | NOT RUN |
| A | Qwen-Image 2512 FP8 | default | off | on | single, uly2, cfg2 | `fp8_ops` in the actors | NOT RUN |
| A | Qwen-Image 2512 FP8 | fp8_e4m3fn_fast | off | auto | single | the loader route to `fp8_ops`, against `on` | NOT RUN |
| A | Qwen-Image 2512 FP8 | fp8_e4m3fn_fast | off | off | single | `off` turns the loader's request into the dequantizing path | NOT RUN |
| A | Qwen-Image 2512 FP8 | default | on | auto | uly2 | the driver flag does not reach the actors | NOT RUN |
| A | HunyuanImage 2.1 FP8 | default | off | auto, on | single, uly2, cfg2 | the same pair on a second plain FP8 family | NOT RUN |
| B | Krea2 Turbo BF16 | default, fp8_e4m3fn, fp8_e4m3fn_fast | off | auto | single, uly2, cfg2 | BF16, the FP8 cast dequantized and the FP8 cast on `fp8_ops`, on the same weights | NOT RUN |
| B | Krea2 Turbo BF16 | default | off | on | single | what `on` costs a BF16 model, whose linears keep the cast path | NOT RUN |
| C | Krea2 Turbo fp8_scaled | default | off | auto, on | single, uly2, cfg2 | a kitchen file: no change expected | NOT RUN |
| C | Chroma1-HD fp8mixed, Ideogram4 fp8_scaled pair, Wan 2.1 T2V 14B fp8_scaled | default | off | auto, on | single | the same on three more kitchen files | NOT RUN |
| C | Krea2 Turbo fp8_scaled, Chroma1-HD fp8mixed, Ideogram4 fp8_scaled pair, Wan 2.1 T2V 14B fp8_scaled | fp8_e4m3fn_fast | off | auto | single | what the loader's FP8 dtype does to the weights comfy-kitchen leaves unquantized | NOT RUN |
| C | Chroma1-HD BF16, Ideogram4 BF16 pair | default | off | auto | single | the BF16 references for Chroma and Ideogram4 | NOT RUN |
| D | Qwen-Image 2512 FP8, stock graph | default, fp8_e4m3fn_fast | off, on | none | the ComfyUI process, one GPU | the driver's own path | NOT RUN |
| D | Krea2 Turbo BF16, stock graph | default | off, on | none | the ComfyUI process, one GPU | what `fp8_ops` costs a BF16 model on the cast path | NOT RUN |

**What the outcome changes.** If `fp8_ops` renders a plain FP8 file
measurably faster than the dequantizing path, with one-step NRMS against it
inside the 0.10 floor on every layout, MODELS.md recommends
`fp8_matrix_mult = "on"` for plain FP8 files and `--fast fp8_matrix_mult` for
the driver's own loads, and the launch notes say so. If it is no faster, or
moves the output past the floor, the launch notes say the plain FP8 path is the
slow one and the knob stays at `auto`. The kitchen rows check that neither
switch changes those files; a change there is a finding against this record.

## Memory calibration

GB10 host and GPU allocations share one physical pool. A CUDA-allocated
weight count is not total memory pressure: CPU LoRA backups, load staging,
driver text encoders/VAEs, activation storage and retained allocator memory
also consume it. CPU offload alone does not release this pool.

| Measurement | Result and use |
|---|---|
| Stock full placement | FP8-mixed Flux2 33.02 GiB reached 2.08 times file size during placement. The price rounds up to 2.1 times file plus 5 GiB. Older 1.85 pricing did not cover the full placement. |
| Partial-load margin | BF16 Flux2 slab batch 1 completed at 12.5-13.2 GiB available; batch 2 at 11.0 GiB partially offloaded and refused. The 5-GiB base floor does not replace per-render activation/driver pricing. |
| Low-RSS LoRA | A near-full-coverage Krea2 BF16 stack held 23.88 GiB CPU backup beside 23.88 GiB GPU weights. Merge-and-free removed the backup; lazy restoration/swaps took 14-16 s versus 90-157 s for reloads, with equal fresh-bake output. |
| Slab weights | Krea2 24.5-GiB load readiness was 20.1 s versus 26.0 s classic; retained dirty arena was zero versus 18-23 GiB. INT8/FP8 LoRA first-use comparisons passed with maximum latent difference zero. |
| Managed pinned staging | A 39.13-GiB BF16 file cost 2.29 times file size with pinned staging. Removing the pinned budget reduced the measured cost to 1.27 times. The larger configuration killed a 60.02-GiB load; no safe capacity claim follows from that attempt. |
| Managed versus stock LTX 2.5 | World 1 BF16 960x544x121, one step: equal video/audio latents, decoded frames and waveform; available-memory minima 48.37 versus 17.31 GiB. BF16 activation casts explain equality despite 290 stored FP32 tables. World 2 and INT8 were not measured. |

Chroma FP8-mixed ran two one-step Euler renders at 128x128
on two ranks with Ulysses, 16 synthetic zero-conditioning tokens, a fixed
seed and no LoRA. Hard Clear VRAM separated the renders. Each residency's
fresh-load output matched its own first output exactly; this was not a
managed-versus-stock fidelity comparison.

Hard clear emptied both model stores and ComfyUI's loaded-model cache.
PyTorch CUDA allocations fell to about 0.03 GiB per actor. Process RSS
remained about 8.6 GiB under comfy-managed residency, from a 1.2-GiB
baseline; the stock control retained about 8.4 GiB after its first clear.
Separate diagnostics found no surviving tracked model, patcher, parameter
or aimdo buffer objects. The residual process memory was not specific to
comfy-managed residency, and these checks do not identify all its owners.
Resetting the attached mesh retires those actor processes when their retained
memory is needed. These small repeat/reload checks do not establish full
host-memory recovery, a sustained multi-prompt soak or other model scopes.

Capacity checks identified further limits. An injected Flux2 slab
cleanup failure kept its resource owned and refused a second load; a fresh
fault-off process loaded and unloaded normally. Flux2 BF16-to-FP8 casting
completed a load/unload with all live parameters slab-backed after annex
cache release: 171 FP8 and 128 BF16 parameters, 22.38-GiB sampled minimum
available memory and 90.69-GiB maximum RSS. This had no sampler or fidelity
comparison and is not an absolute peak or general capacity price.
A DP2 batch-2 FP8-mixed load completed without partial-load divergence in
55.76 s, with sampled minima 25.97/56.2 GiB; DP outputs are intentionally
different. A BF16 FSDP-to-auto transition rendered the predecessor then
refused the 131.0-GiB stock price before either new load, leaving no resident
model. It validates the guard, not the old crash or a stock-reference result.

Flux2 BF16 and FP8-mixed Ulysses+FSDP one-step controls
completed in 48.039/46.037 s, with prices 39.812/44.625 GiB and sampled
memory minima 24.859/64.409 and 41.96/81.90 GiB. The direct FP8 wrap retained
16.5 GiB per rank without host-row materialization; sparse traces cannot
refit the 1.20 full-copy bound. Separate campaign results include
61 passing no-LoRA INT8 FSDP cells on two ranks: 30 Krea2, 30 Chroma and
one Ideogram4. Worker journals confirm quantized parameters sharded as
stored bytes. The recorded decoded outputs matched their resident references.
This establishes execution for those configurations, not a measured peak
for the 1.20 full-file bound or acceptance at its floor.

Four direct production guard checks passed under real
anonymous-memory pressure. The tests used measured available memory, kept
at least 32 GiB available and loaded no model weights. Each refusal left
about 4.54 GiB of projected headroom, below the required 5 GiB.

| Guard | Available memory | Required memory | Result |
|---|---:|---:|---|
| Flux2 stock final wall | 73.884 GiB | 74.343 GiB | Initial resolution passed; the final wall refused after memory was allocated. |
| H3 BF16 managed, pinned staging disabled | 66.265 GiB | 66.729 GiB | Managed guard refused. |
| LTX driver load window | 94.799 GiB | 95.258 GiB | Driver guard refused. |
| H3 INT8 driver resident window | 46.755 GiB | 47.217 GiB | Driver guard refused. |

All pressure allocations were freed. The driver checks used a real owned
mesh, confirmed empty model slots before and after, and completed recycle.
These are hardware guard-integration results, not a full loader race or a
render under pressure. They do not fit a capacity coefficient or measure the
1.20 full-file bound. The live checkpoint-copy race and damaged real-ledger
capability-context tests remain CPU-only.

An earlier low-memory LoRA prototype let ComfyUI unload while its backup
dictionary contained sentinels; a sentinel reached a real weight and the
dtype/shape guard forced a clean reload. The implemented swap keeps ComfyUI
load management out of that operation. Rebuilding a baked quantized wrapper
from file metadata also failed identity (maximum latent difference 2.53);
the slab path preserves the live wrapper's layout and parameters instead.
These failures explain why those apparently simpler alternatives are absent.

A file-backed slab requires low-RSS mode and is disabled for FSDP. Real
compilation also disables it because compiled FP8 GEMM rejected host-backed
addresses. Known compile-no-op families retain their ordinary residency
policy. These measurements do not cover every model/LoRA/precision.

Slab LoRA baking temporarily allocates patched keys before reabsorption.
Header-based pricing matched Krea2's 271 targets at 23.88 GiB compute width.
Flux2 mapped 106 of 170 targets; unresolved fused names conservatively price
the whole file. Narrow quantized keys are charged at stored width. A same-base
stock resident adopted for a slab-auto hot swap receives the measured swap
credit; its controlled run completed with available memory never below 16 GiB.

MemAvailable already excludes shared-memory slabs on the tested kernel.
Across 544 retained samples:

```
MemAvailable - MemFree = 1.0054*(Cached - Shmem) + 0.0025*Shmem - 0.735 GiB
RMS residual 0.229 GiB; maximum 0.745 GiB; Shmem range 0.00-63.52 GiB
```

Subtracting Shmem again would count the same allocation twice. The
[retained corpus](../tests/fixtures/meminfo_credit_corpus.jsonl) and
[regression tests](../tests/test_capacity_memory.py) check the accounting.
A 33.10-GiB hidden pinned-staging delta is charged separately. Dirty/writeback
cost has no measured calibration yet; neither it nor a generic cache haircut
is inferred from these samples.

## FSDP calibration

Leave `NCCL_PROTO` unset. Forced LL failed FSDP on this platform. Sequence
parallelism and FSDP also need deterministic collective launch ordering:
`NCCL_LAUNCH_ORDER_IMPLICIT` is set during every worker setup because NCCL
reads it once per process. Explicit forward prefetch remains disabled.
Ring+FSDP refuses because reload changed the measured ring output; a
successful old render does not establish accuracy on the current path.


The BF16/FP16 streaming build commits each rank's share plus temporary work.
Its measured peak was 0.547 times a 60-GiB file at world 2: 0.5 share plus
0.047 transient. Worker pricing uses `1/world + 0.08` plus a 5-GiB floor.
Other types or unknown worlds retain the full-copy bound (1.2 times file
plus floor). Quantized wrapping is not covered by the floating-point
streaming measurement.

A copy test identified why a shard was held twice: transferring
1.6 GiB directly from private mmap rows left 1.61 GiB anonymous inside the
mapping; a pinned 256-MiB bounce buffer avoided that copy-on-write storage.
With the corrected copy path, the H3 render reported zero checkpoint copy-on-write
and at least 31.38 GiB available on the driver host. Its anonymous peak was
13.75 GiB, exceeding the planned 12-GiB criterion; the peer peak was 8.19.
Flux2 FP8-mixed rendered at 7.5 GiB anonymous versus the earlier 29.4-GiB
partial-load failure. These are memory/execution results, not independent
passing accuracy results. Other planned shard-build checks have not run.

Driver pricing for a single co-resident rank uses 1.05 times its share,
never less than half a file, plus a 256-MiB bounce buffer. This assumes the
copy-on-write repair. Multiple local ranks retain the whole-file price.
The 1.05 margin covers measured small FP32 islands, not the maximum permitted
5-percent island set plus allocator slack; the higher worker check remains
necessary. A near-cap family needs header-derived island pricing before
claiming that driver estimate as a measured bound.

Text preprocessing outside the diffusion forward must remain replicated:
H3 has 19 such parameters totaling 1.487 GiB (0.743 GiB above its world-2
share); the checked LTX 2.5 file has 258 connector tensors totaling 3.755 GiB
(1.878 GiB extra). Anima's auxiliary module is also replicated. H3's repaired
render proves execution only, as described in its model entry.

The no-LoRA first-use check must complete every clean reload and comparison. It prices
each load before it starts and again after the reference render, when the
driver stack is resident. Each price credits the shard pool the next unload
drops, and the check can finish INCONCLUSIVE if a load still cannot fit.
`auto` does not select FSDP. Explicitly disabling the gate does not establish
fidelity. A successful shard build or resident-memory ledger update cannot
substitute for comparison.

**Clean-reload memory probe, 2026-10-13.** Two first-use proofs ran at world 2
under `uly2+fsdp` on the tested pair, on code without the credit. The driver
host carried its 2100 MHz GPU clock cap.

- The proof on a 60.02 GiB BF16 Flux2 file refused at the initial price, with
  its 33.14 GiB text encoder resident on the driver host. That host read 39.1 GiB available,
  with its 30.0 GiB share resident, against a 39.8 GiB price. The worker-only
  host read 82.6 GiB. The ledger row was INCONCLUSIVE with
  `cross_mode: CAPACITY`. The Worker guard had admitted the first load at
  70.1 GiB available.
- The proof on a 24.48 GiB BF16 Krea2 Turbo file passed: two proof renders
  were bit-identical. One load was priced at 19.2 GiB. Both unloads returned the
  whole share to the host. On the driver host the baseline unload went from
  81.8 to 94.0 GiB available, 12.2 GiB back for 11.9 GiB of shards, and the
  cycle unload from 66.6 to 84.8 GiB, 18.2 GiB back with sampling reservations
  included. The worker-only host matched. The next build's first wrap read
  0.0 GiB of torch reserved, so the pool was not retained. The reload trough
  sat 17.6 GiB below the post-unload level, and render A's trough 18.8 GiB
  below it, both inside the price.

These figures support crediting the resident shard pool, capped at one share.
With the credit, the Flux2 preflight reads 39.1 plus 29.9 GiB against
39.8 GiB. The Flux2 proof has not been rerun on hardware with the credit.

## LTX calibration

A warm world-2 session with the 20.03-GiB INT8 ConvRot
transformer measured available memory every second on the worker-only host:

| State | Available memory |
|---|---:|
| Idle | 115.17 GiB |
| Stock-load minimum | 74.63 GiB |
| Settled load | 88.37 GiB |
| After 16,320 video rows per rank | 85.06 GiB |
| After 28,800 video rows per rank | 81.68 GiB |

The second shape loaded nothing. The 3.38-GiB difference over 12,480 rows
fits 290,797 bytes per row, representing the sharded term plus twice the
replicated term at world 2. Settled weights leave a 6.77-GiB per-worker
runtime floor. BF16 and INT8 use the same BF16 activation loop; the measured
sum does not independently identify each activation component.
Native 4K at 121 frames (130,560 video rows) completed; 3840x2176 at 361
frames (375,360 rows) is refused before load. These are memory observations,
not accuracy gates. LTX 2.5 needs ComfyUI `bd34f338` or newer and checkpoint
geometry metadata preserved through loading.

## H3 capacity

Activation pricing charges each rank a fixed floor of 7.5 GiB, 634,368 bytes
per packed row on that rank (59 x 5376 x 2) and 73,728 bytes per row of the
whole packed total, which no topology divides. Four hardware points set the
floor and the per-rank slope; the constants and their derivation are in
`src/dgx_monarch/h3_calibration.py`.

| point | shape and stack | outcome | the estimate must |
|---|---|---|---|
| 1 | 1344x768x124, 37,730 rows, one rank, BF16 (2026-08-05) | killed twice | refuse, resident and pending |
| 2 | the same on world-2 uly2, BF16 (2026-08-05) | completed in 727.6 s | pass, resident and pending |
| 3 | the same on one rank, pruned INT8 (2026-08-03) | completed at 50.74 s per step | pass under an explicit preset |
| 4 | 1344x768x362, 109,463 rows, uly2, pruned INT8 ConvRot (2026-10-29) | completed | pass at 51.5 and 56.3 GiB usable, and charge more than the 17.6 GiB measured |

The budgets carry whole-GiB doubt, so the fit maximises the smallest margin
in GiB. It holds point 1 to the larger budget the old 4 GiB reserve left and
every pass to the shipped 5 GiB reserve, so the verdicts hold at either. The
optimum, a 7.70 GiB floor and 631,756 bytes per row, ties point 1 resident,
point 2 pending and point 4's second render at 4.09 GiB. The rounded constants
clear each point as follows, in GiB per rank, with the 4 GiB reserve's figures
in brackets:

| point and posture | estimate | usable | margin |
|---|---:|---:|---|
| 1, resident | 32.38 | 27.4 [28.4] | refuses by 4.98 [3.98] |
| 1, pending, 61.73 GiB of weights | 94.11 | 87.2 [88.2] | refuses by 6.91 [5.91] |
| 2, resident | 21.24 | 27.4 [28.4] | passes by 6.16 [7.16] |
| 2, pending, 61.73 GiB of weights | 82.96 | 87.2 [88.2] | passes by 4.24 [5.24] |
| 3, explicit preset | 32.38 | 55.5 [56.5] | passes by 23.12 [24.12] |
| 4, second render | 47.35 | 51.5 | passes by 4.15 |
| 4, gate slab path | 47.35 | 56.3 | passes by 8.95 |

Both measured points are lower bounds on what their shapes need. The estimate
is 1.30 times the 24.9 GiB the point 1 kill left unattributed and 2.69 times
the 17.6 GiB each actor reserved at point 4. The killed run still carried the
legacy load path's retained arena, since removed, so its reading overstates
what that shape needs now. A stock small-shape BF16 reference went from
114.7 GiB idle to a 28.4 GiB floor, and points 1 and 2 take their budgets from
it. The two VAE files total 5,813,063,304 bytes (5.41 GiB), not the 5.8 GiB
the pending budgets subtract; with exact bytes point 2's pending pass gains
0.39 GiB and point 1's pending refusal loses as much.

The first fit, of 2026-08-05, had no floor. Its 1,032,192 sharded bytes per
row sat near the centre of the window points 1 to 3 left, which was
(734,495, 1,359,225] bytes at the 4 GiB reserve and (706,036, 1,302,308] at
5 GiB, and 1.63 times above the kill's reading, with every fixed cost folded
into the slope. At point 4 it charged 60.1 GiB per rank, refusing the
session's second render 8.7 GiB short and the gate's slab path 3.8 GiB short,
and on one Spark it charged 112.3 GiB for 109,062 rows. The 2026-10-29 first
renders in
[BENCHMARKS](BENCHMARKS.md#minimax-h3-speed-and-memory-by-clip-length-2026-10-29-to-2026-10-30-head-capped)
ran under that model.

On one Spark the loader node reads the 362-frame cell as 109,062 rows of video
and audio. With the pruned DiT pending (19.53 GiB), 92.8 GiB MemAvailable and
the 5 GiB reserve, the model charges 79.4 GiB of activations (7.5 floor,
64.4 sharded, 7.5 replicated) and refuses, 11.2 GiB short. The 243-frame
cell, 73,386 rows, fits at the same reading with 12.4 GiB to spare. Constants
that admit the 362-frame cell exist, but every pair that does leaves point 1's
refusal, point 2's pending pass or the cell itself with 2.5 GiB or less,
against the 4 GiB this fit keeps. A measured one-Spark run decides that cell,
not the constants.

The suite of 2026-10-29 and 30 ran both one-Spark layouts. On the head beside
the driver the 243-frame cell rendered, as this fit predicts, with the head at
52.6 GiB MemAvailable at its lowest. The 362-frame cell refused there at the
loader, 10.8 GiB short at 93.2 GiB MemAvailable, so that layout has no
measurement at 362 frames. One rank on the sibling alone, with the driver on
the head, rendered the 362-frame cell
([measured points](#activation-across-clip-lengths-measured-2026-10-29-and-30)).


A 61.73-GiB BF16 reference-to-video load left 60.9 GiB available before
the driver stack consumed it and the host froze. The stack's
60.9-GiB cost is only a lower bound; it cannot be split into encoder, vision
and VAE costs from that record. The load path was not captured, so its
1.8-times-file transient is not a second calibration of stock loading.
Settled weights in that single run cost 53.6 GiB, or 0.87 times the file;
one observation does not justify discounting other pending loads.

Pruned INT8/FP8 Ref2VA files near 19.5 GiB completed the measured
512x320 reference setup with 2.5/4.7-GiB minima. A 31.70-GiB INT8 file
completed with a 0.7-GiB minimum. These narrow margins are observations,
not recommended reserves. BF16 slab Ref2VA has no passing output comparison.

Guide encoding was measured with the 4.85-GiB video VAE
already resident, so the following increments exclude its weights:

| Canvas | Frames | Additional memory | Outcome |
|---|---:|---:|---|
| 1344x768 | 1 | 13.6 GiB | Completed |
| 448x256 | 5 | 16.2 GiB | Completed |
| 672x384 | 5 | 66.9 GiB | Completed |
| 1344x768 | 5 | More than 94 GiB | Did not complete; host lost |
| 448x256 | 22 | 33.4 GiB | Completed |
| 448x256 | 39 | 47.5 GiB | Completed |

The preflight rounds area up to a measured point, with clip charges of
17/67/94 GiB, a 14-GiB still charge, and 17/34/48 GiB for 5/22/39-frame
clips at 448x256 or smaller. The top point and saturation beyond it are
lower bounds, not measured completed encodes. Larger canvases and clips
beyond 39 frames remain unmeasured. The loader conservatively treats an undeclared
guide as a clip and refuses that high-resolution case. Declaring a still
through `DGXM_H3_GUIDE_FRAMES` was exercised through the driver API.
The estimator cannot promise capacity for unmeasured longer clips or canvases.

### The 15 second render, measured (2026-10-29)

The pair rendered 1344x768x362 on uly2 at 20 steps with the pruned INT8
ConvRot DiT and the Triton kernel, the head capped at 2100 MHz, while 1 Hz
samplers on both hosts logged MemAvailable, page cache, committed memory and
each GPU process's memory and RSS. The prompt completed in 54:05.

| host | idle MemAvailable | minimum during the render | actor GPU memory peak | actor RSS | driver |
|---|---:|---:|---:|---|---|
| the sibling, actor only | 114.6 GiB | 68.2 GiB, mid sampling | 37.1 GiB | 20.9 GiB during the load, 8.8 GiB after | none |
| the head, actor and driver | 110.6 GiB | 47.5 GiB, during the load | 37.1 GiB | 20.9 GiB during the load, 8.8 GiB after | 14.6 GiB text encoder on the GPU, 10.5 GiB RSS |

Each actor's 37.1 GiB is 19.53 GiB of weights and 17.6 GiB beyond them. The
sibling's actor held 20.4 GiB right after the load, 0.9 GiB beyond its
weights, and climbed to its peak once sampling began. The sibling's whole use
at its minimum, 46.4 GiB, counts the weights, about 6 GiB of page cache growth
and the actor's 8.8 GiB of RSS. Neither host swapped, logged memory pressure
or logged a kernel OOM or reclaim line. A 1 Hz sampler can miss a short spike,
so each peak is a lower bound. In the same session the second request read
56.5 GiB MemAvailable on the head after the driver unloaded its VAEs, and the
gate's slab path read 61.3 GiB after its proof render; the model before the
refit refused both.

### Activation across clip lengths, measured (2026-10-29 and 30)

The suite in
[BENCHMARKS](BENCHMARKS.md#minimax-h3-speed-and-memory-by-clip-length-2026-10-29-to-2026-10-30-head-capped)
rendered 1344x768 clips of 124, 243 and 362 frames at 20 steps with exact
attention and the pruned INT8 ConvRot DiT, on the pair (uly2) and as one rank
on the sibling alone, with 1 Hz samplers on both hosts. Each actor's GPU
memory peak less the 19.53 GiB of weights is the activation it held. Beside it
stands the refit's charge per rank: the 7.5 GiB floor, 634,368 bytes per row
on the rank and 73,728 bytes per row of the whole packed total. In GiB:

| layout | frames | rows per rank | actor GPU peak | beyond the weights | charge per rank | charge over measured |
|---|---:|---:|---:|---:|---:|---:|
| pair, uly2 | 124 | 18,865 | 26.8 | 7.3 | 21.24 | 2.91 |
| pair, uly2 | 243 | 36,895 | 32.0 | 12.5 | 34.36 | 2.75 |
| pair, uly2 | 362 | 54,732 | 37.1 | 17.6 | 47.35 | 2.69 |
| one rank, the sibling alone | 243 | 73,790 | 31.1 | 11.6 | 56.16 | 4.84 |
| one rank, the sibling alone | 362 | 109,466 | 36.5 | 17.0 | 79.69 | 4.69 |

The 362-frame pair row measures point 4 again, with the same 17.6 GiB. The
unpruned 31.70 GiB INT8 file held
17.7 GiB beyond its weights at that shape, so the file does not move the
activation. A 1 Hz sampler can miss a short spike, so each measured figure is
a lower bound.

One rank holding every row held no more than a pair rank holding half of
them: 11.6 against 12.5 GiB at 243 frames, and 17.0 against 17.6 at 362. The
charge follows a rank's own rows, so it runs 2.7 to 2.9 times the pair's
measurements and 4.7 to 4.8 times the one-rank ones.

On one rank the 362-frame clip completed and the sibling never fell below
75.3 GiB MemAvailable, while the charge for that rank is 79.69 GiB of
activations against the 17.0 GiB it held. On the head beside the driver the
same charge refuses the cell at the loader by 10.8 GiB, and the cell was not
run there. These points change no constant. The fit must still refuse point
1, the one-rank BF16 kill of 2026-08-05, and the suite could not retest that
shape, because the BF16 rescue below never rendered. Whether the measured
points should replace that constraint is the maintainer's decision;
[issue #93](https://github.com/Deen-Media/dgx-monarch/issues/93) carries it.

### BF16 capacity rescue (2026-10-30), a recorded limitation

The 61.7 GiB BF16 file rendered on no layout, so this record is a limitation,
not a pass. Stock residency cannot load the file on this hardware: on the
pair the loader charges 129.6 GiB on the driver host against about 88 GiB
usable and refuses, class C and waivable.

On the pair at 1344x768x124, with consent from `DGXM_ALLOW_SLAB_RESCUE=1` or
from `DGXM_AUTO_RESCUE=1`, both ranks loaded the file into slab residency. The
head fell to 15.1 GiB MemAvailable and the sibling to 36.2 GiB, above the
8 GiB watchdog. The identity gate ran its proof renders, and then its
cross-residency check needed a stock reference of the same seed, priced at
134.6 GiB, which cannot load. The gate returned INCONCLUSIVE, unproven, and
switched slab off for the combination. The render then refused under stock at
the cross-rank capacity check before any rank loaded, and a second request
refused the same way. Afterwards the loader priced the combination at stock
only and raised no consent card, so the panel offered no way back.
[DESIGN.md](DESIGN.md#the-residency-ladder) says that active rescue consent
keeps slab after this verdict; neither headless form reached that rule.

On one Spark no rescue began. With the rank on the sibling alone the actor's
load failed, and its own rescue offer never reached the driver's consent
list. On the head beside the driver the loader refused outright, 61.7 GiB to
place against 55.8 GiB usable, a refusal no consent can waive. With the
driver preflights off, the head actor's load failed as the sibling's had, and
its offer did not reach the driver either.
[Issue #93](https://github.com/Deen-Media/dgx-monarch/issues/93) carries the follow-up.

### Driver unload before a render-site refusal (2026-10-29)

Hardware check for the unload step (#105) on dev d34d9b0, both Workers on
that slot, doctor 44 of 44, head at its 2100 MHz cap. MiniMax H3 guide
template with the pruned INT8 ConvRot DiT, the 32B NVFP4 text encoder and both
VAEs left loaded in the driver after a first render, topology auto (uly2), 4
steps. A balloon of populated anonymous memory on the head set MemAvailable
where the check needed it.

| render | frames | MemAvailable at submit | outcome |
|---|---|---|---|
| A, baseline | 29 | 93.9 GiB | rendered in 105 s; the driver then held the text encoder and both VAEs (MemAvailable 56.0 GiB) |
| B, 45 GiB balloon | 29 | 11.3 GiB | the check refused on the first reading; the driver unloaded MiniMaxH3VideoVAE, MiniMaxH3AudioVAE and MiniMaxH3TEModel_, MemAvailable 11.3 to 30.3 GiB, a gain of 19.0 GiB, and the render went through in 54 s with no recycle; both VAEs reloaded at decode and the frames and audio saved |
| C, 67 GiB balloon | 124 | 3.3 GiB | refused after the unload: 21.4 GiB needed per rank against 3.7 GiB usable (8.7 GiB MemAvailable minus the 5.0 GiB reserve), closing with the sentence that names the two models unloaded and MemAvailable 8.3 GiB before and 8.7 GiB after |

Not run: a pipelined submission behind a render in flight.

## Dependency and setup checks

xFuser `0.7.0+dgxm.npuimport1` was compared with 0.4.5 under
yunchang 0.6.4, torch 2.12.0+cu132 and torchmonarch 0.6.0. Chroma 1024
one-step local/Ulysses with 44/44 and 100/28 prompts, plus unequal cfg2,
matched within and across dependencies. Krea2 512 ring2 cuDNN read NRMS
0.009443 under both versions with identical same-path outputs. Chroma
26-step local/Ulysses matched under the compatibility build, without a
26-step cross-version baseline. The H3 result is in its model entry.
The compatibility change delays only an optional NPU import; these NVIDIA
checks do not cover NPU, other backends or future releases.

### Operator checks

| Check | Recorded result and limit |
|---|---|
| Existing-install discovery | Setup dry run preserved managed services and refused takeover. Readiness distinguished a stopped driver from observed render readiness. |
| Configuration repair | A held lock prevented changes. After the lock cleared, repair restored strict permissions without changing contents. |
| Setup on existing ComfyUI | All eight setup steps, 37 Doctor checks and a two-rank smoke passed. Flux Dev BF16 1024-square, 20-step output completed in 18.50 s (56.49 s first queue); a 1536 split graph took 44.67 s. Rank latents agreed, without an independent fidelity comparison. Protected service inspection still reported UNKNOWN ownership. |
| Stopped Worker | Stopping the local Worker made status exit 1 with overall and Worker readiness blocked. The Doctor Worker-row adapter also blocked; a full Doctor run was not performed. The original Worker was restored without restarting its peer. |
| Verified update | Source-changing and same-ref runs each passed all 15 steps, including Doctor, two-rank smoke, release readback and finalization. Both Workers were healthy afterward with no actors or GPU work remaining. |
| Update recovery | A failed final release readback triggered rollback to the prior release on both Sparks. An unauthenticated origin fetch refused before update steps ran. |

Update tests used local Git transport and unchanged dependency pins. They do
not test GitHub authentication or dependency upgrades. These operator checks
do not establish model accuracy or native latent RDMA support. See
[update requirements](INSTALL.md#updating).

### Package-layout compatibility

A disposable Python 3.12 environment installed the official TorchMonarch 0.6.0
ARM64 and color-matcher 0.6.0 wheels with pip. The older ownership verifier
rejected the resulting layout. The corrected verifier matched all 393 pinned
TorchMonarch files to the official wheel while preserving both packages'
records and all installed fixture files. Pip generated the external test-data
and CLI bytecode records without manual changes.

This checks installed-file compatibility. It does not test dependency
completeness, runtime imports, renders or a completed update on an affected
pair. See [the recovery procedure](TROUBLESHOOTING.md#110-verified-update-refuses-color-matcher-test-files).

### Fresh agent installation

An independent coding-agent session received the exact README setup prompt,
repository access and approved test boundaries. It followed the setup skill to
install new Python environments, ComfyUI and DGX Monarch on both Sparks.
The final instructions were tested from empty installation paths after the
first trial identified missing setup details.

The test reused the operating system, NVIDIA driver and physical network.
The three Chroma model files were copied independently and checked against
published hashes on both hosts. Temporary, approved administration provided
exact-peer fabric filtering and the documented read-only process inspector.
Global caches were preserved. Some Worker scratch and native logs used
default locations outside the test directory. This was not a cold-cache or
clean-machine test.

Tested versions were Python 3.12.3, ComfyUI 0.37.0, PyTorch 2.12.0+cu132,
TorchMonarch 0.6.0 and xFuser 0.7.0+dgxm.npuimport1.
The fresh environments used pip 24.0, and their saved dependency checks passed.
Pip 24.2 and later detect the [cuSPARSELt platform-tag issue](TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform).
For current installations, run `tools/check_dependencies.py` with the chosen
ComfyUI interpreter as described in INSTALL and retain any verified exception
line. This does not change the recorded pip 24.0 results or establish new
render evidence.

| Check | Recorded result |
|---|---|
| Guided installation | All eight setup steps passed; Doctor reported 38 checks and no failures. |
| First renders | The documented 1024-square Chroma cfg2 workflow saved valid PNGs with seeds 0 and 1 on stock residency. Sampling logs and matching source/rank records established participation by both Sparks. |
| Setup rerun | Read-only validation preserved source, package versions, configuration, units, Worker identities and all three model hashes on both hosts. |

The original installation was restored and verified with Doctor and a
two-Spark cluster smoke test. The temporary fabric guards were removed, and
a final check confirmed the original Workers were healthy and idle. Test
outputs and records were retained privately.

These checks establish installation and distributed rendering for this
recipe. They do not compare image fidelity with native ComfyUI or add
coverage for other models, memory optimizations or native latent RDMA.
See the [first-render recipe](QUICKSTART.md#first-distributed-render) and
[setup skill](../skills/dgx-monarch/SKILL.md).

### Compile checks

Flux2's `compile_dit` check ran eagerly because it has no compilable
block list. Stock/slab first-use checks passed locally and on Ulysses;
the one-step local/Ulysses pixel NRMS was 0.008895. Explicit slab-on cold
loading made one bounded retry. No speed or LoRA claim follows. Krea2
compiled 28 blocks, stayed stock and produced NRMS 0.011784; its no-material
first-use proofs stayed INCONCLUSIVE. The older Krea2 FP8 cfg2 compile
measurement was 28.4 versus 36.0 s, approximately 21 percent faster, with
latent drift around 1e-5 and about 9 s one-time tuning. It is opt-in and
not a general family-wide compile result.

## Lifecycle

Clients own their attached meshes and actors, not persistent Worker services.
A successful render never authorizes a client to stop a Worker coordinator.
Readiness checks use telemetry and passive process/listener inspection, not
raw TCP connections to the Worker protocol.

### Monarch 0.7.0 no-model lifecycle

**October 13, 2026 UTC, source `ad199a9`:** the healthy production lifecycle
control passed attach, status and owned shutdown, then `ensure_live` reattached
with fresh actors and completed a second shutdown. No model, NCCL setup or
sampling ran, and native latent return was disabled.

Separate authorized single-Worker stop/start and direct-restart cases both
failed same-driver recovery. Loss observation raised `SupervisionError` and
left the old process stop unconfirmed. After the existing all-host process-gone
check released the stop latch, each case made one replacement attach; both
reached the 60-second configuration-push timeout and poisoned the driver
transport. The direct-restart loss-observation call took about 73 seconds;
its replacement attempt took about 61 seconds. These are observed call
intervals, not guaranteed failure-detection or recovery deadlines.

Following the stop/start case, separately authorized coordinated Worker
restoration and a fresh driver's attach/status/shutdown passed with verified
owned exit. Recovery acceptance here is confined to that stop/start case;
it does not establish recovery after the direct-restart failure. The no-model
scope also establishes no mid-sampling, NCCL, physical-fabric or native-RDMA
failure recovery. These checks came before per-start Worker ids: upgrading the
dependency alone did not fix same-driver replacement. Per-start ids close
[issue #2](https://github.com/Deen-Media/dgx-monarch/issues/2)
([TROUBLESHOOTING #2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)),
which records their two-Spark recheck of 2026-10-15; the recheck of the
re-attach within the first sampler render after a restart is pending.

Under torchmonarch 0.6.0, a 62-log two-host study supported retaining both
in-flight and grace-window teardown handling. Log counts were upper bounds
on actual events, and test traffic exceeded ordinary interactive use.

When worker death blocked a fleet with only abandoned leases, renewal tests
skipped 8/8 or 14/14 attempts; live sample leases continued renewing. A live
unreachable handle logged one full fault plus repeat counts. Render, recycle
during telemetry, a second no-op recycle and a fresh render completed without
ERROR lines. This does not suppress faults from an unconfirmed stop.

Recycling releases weights by exiting the actor process, avoiding an
unnecessary copy of the second resident Ideogram4 checkpoint to CPU. The
two-model FP8 test improved from 74.465 to 5.253 seconds; Krea2 slab/FSDP and Flux2 controls remained around 5.5-8.6
seconds. Memory returned within 1 GiB of pre-load availability on both hosts,
with no leftover checkpoint aliases. Full unload pins both model slots before
global ComfyUI unloading. Declared reap grace values (150 s parent loss,
300 s lease loss) are configured time limits; hardware tests have not established them.

Under torchmonarch 0.6.0, live-render status latency was
177 ms median and interruption stopped both GPUs at about 4.5 s, with the
job settling near 6 s. The next dispatch correctly refused until recycle.
Observed timing bounds were 3-s hyperactor cleanup, 10-s process stop,
30-s teardown grace, 60-s group teardown, about 75-s dead-peer detection,
120-s RDMA read plus 15-s wait margin, about 135-s peer self-clear,
240-s token authority and 900-s dispatch/collection. Some are declared
budgets and others are measured detection times, not interchangeable limits.

<a id="reply-channel-fault"></a>

A network-fault test dropped local Worker replies during a two-Spark Chroma
render. The 32-step, 256-square control completed in 3.535 s; collection with
the fault timed out at 10.001 s. Fourteen packets were dropped, twelve before
the deadline. The drop window was 30 s; test-rule removal was confirmed about
33 s after activation.

Same-process recovery **FAILED**: two reattach attempts reached the 60-s
configuration timeout after a 30-s acknowledgement-delivery timeout closed the
reply channel. Original actors were confirmed recycled despite a teardown RPC
warning. A fresh client then passed two attach/render/cleanup cycles, with
outputs identical to the controls and both Worker services unchanged.

After confirmed cleanup, restart ComfyUI or the headless driver for this
failure; see [reply-channel recovery](TROUBLESHOOTING.md#15-attaches-still-time-out-after-dgxm-restart-driver-side-wedge).
This establishes bounded collection failure and new-process recovery, not
same-process recovery, physical fabric outage handling, NCCL failure recovery,
native RDMA support or model fidelity.

The NCCL 2.32.3 comparison measured two-rail 1-GiB allreduce at
19.5 GiB/s versus 20.1 on 2.30.7; single-rail 10.6-10.9 versus 10.7-11.2;
and 16-MiB two-rail 11.4 versus 10.4. Fixed collective output and all 14
Flux-family matrix latents were identical. This is an NCCL comparison,
not a test of native latent-return RDMA.

## Default-off messaging return

**October 8, 2026, before the 0.7.0 migration:** six production zero-step
custom-sampling cases passed on the two-Spark pair with Krea2 Turbo FP8 scaled. Returns of 8,372,224,
8,388,608 and 8,404,992 bytes covered below, at and above the 8-MiB threshold;
all three sizes were repeated after a fresh attach. Native RDMA remained off.

The production path converted the four-dimensional input into a
five-dimensional output. Primary and denoised tensors matched exactly, with
matching digests across all ranks. No RDMA owners, acknowledgements or readers
were created. Jobs and leases were empty after completion, and owned meshes
were confirmed recycled. Messaging passed at all three threshold sizes and after reattachment. No denoiser forward ran, so it provides no denoising-fidelity,
model-accuracy or native-RDMA result.

**October 12, 2026, source `ad199a9`:** a separate Krea2 RAW BF16 message
return check covered 8,355,840, 8,388,608 and 8,421,376-byte zero-step outputs,
then repeated the at-threshold case in a fresh driver and actor cohort. All
eight saved primary/denoised tensors were finite FP32 zeros with exact bytes.
Queues, jobs and leases settled; typed recycle and owned close completed with
persistent Workers unchanged. This is not stock zero-step equality, denoising,
VAE fidelity or native ownership evidence.

## Native RDMA

**Historical trial before the 0.7.0 selected-rail qualification:** the native
return-path test **FAILED** at exactly 8 MiB: the two ends
selected disconnected rails and could not establish the queue pair.
Below-threshold actor messaging passed tensor integrity. Above-threshold
and reattach cases were NOT RUN. There was no native RDMA PASS, ACK,
ownership-settlement or throughput result. Earlier host-memory microbenchmarks
at 6.98-7.55 GiB/s do not cover the current production return path.

### Selected-rail qualification

**October 12, 2026, source `ad199a9`:** the canonical production-return
qualifier passed 12 messaging cases and 28 selected-NIC native cases with one
QP and one application chunk. Native coverage included both owners across two
fresh cycles, exact CPU tensor bytes, wrong-generation and duplicate ACK
assertions, a two-entry registry capacity check, overflow fallback and recovery,
and confirmed owned-process stops. Paired passive checks were clear afterward.
The audit bound source, runtime, selected policy and executed protocol assertions;
raw ACK reply arrays were not retained.

This qualifies the recorded CPU-staged production return path on that rail.
It did not initialize a GPU or establish GPU zero-copy, full sampling,
fidelity, cancellation, other rails/QPs/chunks or a performance gain.

Two subsequent Krea2 RAW BF16 zero-step native returns at exactly 8 MiB
matched the messaging reference and each other in saved FP32 primary and
denoised tensor bytes. Each used fresh driver and actor processes. The primary
return was observed as native; the denoised extra remained a separate result.
Two fresh status observations showed no live handoffs, receiver jobs or poisoned
owners. Typed recycle, owned driver exit and subsequent paired idle checks
completed with persistent Workers unchanged. Phase counters and released
tombstones are observations, not independently captured token-bound ACK receipts.
This is actual-sampler zero-step return evidence, not a denoiser-forward test.

**October 13, 2026 (UTC):** repeated CPU-staged return measurements ran
message→native and native→message in fresh driver processes. Each mode/order
used both owners and two fresh actor cycles, with first-use 8-MiB returns
separated from seven warmed repeats at 8, 32 and 64 MiB. The four reports
contain 352 timing calls, plus 80 canonical integrity/ownership cases; all
eight owned actor stops and the following paired idle checks completed.

Across the eight owner/cycle/order strata, native's warmed median savings
were 27.9-32.8 ms at 32 MiB (1.34-1.42× message/native ratio) and
65.5-78.4 ms at 64 MiB (1.44-1.53×). At 8 MiB, native was slower in seven
strata; the remaining 0.108-ms difference was about 0.4%, not a consistent
or material benefit. First native use cost 1.989-2.139 seconds.

Each timed total is pack RPC plus read/drop/ACK; seeded CPU generation is
included, while verification, evidence writes and status checks are excluded.
These are combined phase timings, not continuous wall time, network-only
bandwidth or a denoising speedup. The results support explicit selected-rail
opt-in for repeated large returns, not a global default-on change.

A private workspace tool outside this repository produced the October 13
timing figures above. For later campaigns, the qualifier's `--timing` mode in
`benchmark/rdma_return_bench.py` implements this protocol through the same
production return path, admission and ownership checks, and its
`--timing-summary` mode pairs the four reports into per-stratum warmed medians,
savings and ratios, and refuses a set that lacks either order. The mode has not
yet run on hardware, so it has not reproduced these figures; a campaign that
runs it gets its own dated record.

**First-use phase split: recorded, NOT RUN.** The 1.989-2.139 s first native
use above is one combined total. Each native return now records its phases. On
the Worker actor, the result row carries `return_phases`: the child-process
preflight (`rdma_preflight`: a fresh interpreter that applies the policy,
imports Torch and Monarch, then registers and drops a 4 KiB buffer; it runs on
the first native pack in each actor process only), the registration
(`rdma_register`), any rollback drops (`rdma_drop`) and the rest of the pack
as `unattributed`. On the driver, for each completed native return in a normal
render, the `native return:` log line adds the read (`rdma_read`), the drops
(`rdma_drop`), the ACK (`ack`) and the rest. The QP handshake runs inside the
first read, so `rdma_read` includes it. When a record reads complete, its
phases add up to its total within half a millisecond for each rounded value,
the total included. The qualifier's `--timing` mode stores both splits with
each native call, and its summary carries the first-use split per stratum. The
phases come from each process's shared phase record, so the split is exact
only while one native return runs at a time, which is how the qualifier runs;
a return that overlaps it in the same process can add its phases. The split
measurement itself is **NOT RUN**: the next hardware leg takes it. Until then,
the view that the child-process preflight rather than the fabric makes up most
of the first-use cost is a hypothesis. Native return stays opt-in.

**October 13, 2026 UTC:** a separate RAW BF16 production-return experiment
completed one positive denoising step at 2048×4096 without VAE decoding. It
used the same model, prompt, seed and sampler in stock, messaging, selected-rail
native and a fresh native repeat. All eight saved FP32 primary/denoised tensors
were finite, nonzero and exactly 8 MiB; all twelve pairwise comparisons were
byte-exact. Both native cells recorded one native primary descriptor, released
tombstone and successful read/drop/ACK phases, with settled leases, live
handoffs, receiver jobs and poisoned-owner counts. These counters and phases
are observations, not separately retained raw ACK receipts.

Each positive-step cell followed an excluded zero-step residency warmup. The
experimental admission retained normal capacity handling, checked full BF16
residency and the exact actors' UMA activation logs, and required fresh host
MemAvailable above 40 GiB on each participating host. This is an empirical
screen, not a measured peak-memory guarantee or a general resolution limit.
Both native cells used fresh driver/actor processes; typed recycle, owned driver
exit and paired passive idle checks preserved the persistent Workers. The
scope is one positive step at the production return threshold, not a complete
quality recipe, VAE fidelity, GPU zero-copy or a denoising speed benchmark.

Native return stays **disabled by default**. Repeated large CPU-staged returns
on an explicitly selected and qualified rail can benefit from native opt-in;
the measured first-use cost and lack of consistent benefit at 8 MiB do not
justify a global default-on setting. Native cancellation and fabric-failure
recovery were **NOT RUN** and remain separate limits, rather than an unresolved
default-policy decision. GB10 uses unified memory; raw CUDA free memory alone
is not a sufficient capacity decision. NCCL compute
traffic and actor-message returns are separate; nested AV always uses messaging.
A timed-out native read cannot free backing or ACK ownership until its
operation and exact-owner settlement are known. Hard unload does not override
that ownership. Qualification of this selected path does not authorize general
native use or transfer acceptance to untested workloads.
