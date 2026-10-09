# Validation

This page supports the model support and performance claims with measured
results and known limits. For installation, start with [Quickstart](QUICKSTART.md).
For a workflow, use [Model support](MODELS.md); for speed comparisons, use
[Benchmarks](BENCHMARKS.md).

All hardware results below used two DGX Sparks with GB10 GPUs. They cover the
stated models, settings and software, not every combination supported by the
code. The records span different builds and do not constitute a complete
retest of version 1.0.

- [Installation and recovery](#dependency-and-setup-checks): fresh environments,
  two distributed renders, setup reruns and restoration.
- [Image models](#image-models) and [video/audio models](#video-and-audio-models):
  measured accuracy and untested configurations.
- [Memory calibration](#memory-calibration): measurements behind capacity checks.
- [Lifecycle](#lifecycle): cleanup, interruption and failure recovery.
- [Native RDMA](#native-rdma): failed return-path test; remains disabled.

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
Native latent RDMA remains **NOT RUN / HOLD**.

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

## Image models

<a id="evidence-krea2-raw-turbo"></a>

### Krea2 RAW and Turbo

Pure-Ulysses tests on TORCH_FLASH matched DP2 latents exactly:
RAW BF16 at 1536 and 1448 with odd/even text and image rows, batch 2 and the
template; RAW INT8 ConvRot and Turbo FP8-scaled on the template. This does
not cover the SAGE route auto chooses for FP8 above 1.2 MP. The unchanged
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
quantized wrapping transient did not fit beside the driver.

The 60.02 GiB BF16 artifact needs a stock-load price of about 131.0 GiB
(2.1 times the file plus 5 GiB), beyond one Spark in this test pair. Use
supported slab residency, explicit FSDP, or a smaller checkpoint. CFG2 and
DP2 still place a full copy on each rank. FP8-mixed stock placement was
measured after the placement repair: 77.0 seconds cold, 34.7 warm and a
20.0 GiB available-memory minimum.

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
resident legs. Auto still chooses Ulysses.

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
latent return is separate and remains NOT RUN / HOLD.

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

<a id="auto-latent-ratio"></a>

## Latent ratios and automatic topology

Automatic topology uses ComfyUI's detected spatial downscale ratio instead
of a fixed 8-pixel estimate; megapixels mean each frame's
sampled spatial grid, not the compressed latent area.

| Ratio | Families checked in checkpoint headers |
|---:|---|
| 1 | Radiance, Z-Image DCT, PixelDiT |
| 8 | Chroma, Flux 1, LongCat, Qwen-Image, Krea2, Z-Image latent, Kandinsky5 Image, Wan 2.2 14B |
| 16 | Flux2, Lens, Ideogram4, Ernie, Mage Flow, Qwen Image 2.1, HunyuanVideo 1.5, H3 |
| 32 | HunyuanImage 2.1, LTX 2.3 |

Lens 1344 square is 1.81 MP; Flux2 1024 square is 1.05 MP.
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

The no-LoRA first-use check must complete every clean reload and comparison. It rechecks
capacity after the reference render, when the driver stack is resident, and
can finish INCONCLUSIVE if a second load cannot fit. `auto` does not select
FSDP. Explicitly disabling the gate does not establish fidelity. A successful
shard build or resident-memory ledger update cannot substitute for comparison.

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

Activation pricing was calibrated at 1344x768x124, 37,730
packed rows: one-rank BF16 failed twice, world-2 BF16 completed in 727.6 s,
and pruned INT8 completed on one rank at 50.74 seconds per step. The model
charges 1,032,192 sharded bytes per row per rank plus 73,728 replicated
bytes per total row. With the original 4-GiB reserve, the observed pass/fail
intersection for the sharded term was (734,495, 1,359,225] bytes; the later
5-GiB reserve gives (706,036, 1,302,308], still containing the chosen value.
This fit amortizes fixed workspace/allocator costs into a slope and is a
lower bound below roughly 3,900 rows. It is not a fitted intercept model.
A stock small-shape BF16 reference went from 114.7 GiB idle to a 28.4-GiB
floor. The two VAE files total 5,813,063,304 bytes (5.41 GiB), not 5.8 GiB;
using exact bytes leaves the selected activation coefficient inside the bounds.


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

Six production zero-step custom-sampling cases passed on
the two-Spark pair with Krea2 Turbo FP8 scaled. Returns of 8,372,224,
8,388,608 and 8,404,992 bytes covered below, at and above the 8-MiB threshold;
all three sizes were repeated after a fresh attach. Native RDMA remained off.

The production path converted the four-dimensional input into a
five-dimensional output. Primary and denoised tensors matched exactly, with
matching digests across all ranks. No RDMA owners, acknowledgements or readers
were created. Jobs and leases were empty after completion, and owned meshes
were confirmed recycled. Messaging passed at all three threshold sizes and after reattachment. No denoiser forward ran, so it provides no denoising-fidelity,
model-accuracy or native-RDMA result.

## Native RDMA

The native return-path test **FAILED** at exactly 8 MiB: the two ends
selected disconnected rails and could not establish the queue pair.
Below-threshold actor messaging passed tensor integrity. Above-threshold
and reattach cases were NOT RUN. There was no native RDMA PASS, ACK,
ownership-settlement or throughput result. Earlier host-memory microbenchmarks
at 6.98-7.55 GiB/s do not cover the current production return path.

Native latent return stays **NOT RUN / HOLD**, disabled by default. NCCL
compute traffic and actor-message returns remain separate. Nested AV always
uses messaging. A timed-out native read cannot free backing or ACK ownership
until its operation and exact-owner settlement are known. Hard unload does
not override that ownership. The selected rails still need a successful production-path test.
