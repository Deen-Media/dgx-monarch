# Per-model guide

Choose a workflow, check its model details, and keep the listed constraints.
`auto` selects a topology from the model family, precision, resolution and
CFG, then logs its choice. [VALIDATION.md](VALIDATION.md) records what was
tested; [CONCEPTS.md](CONCEPTS.md) explains the parallel modes.

Status: **HW** means the distributed path ran on hardware and passed the
stated comparison, against DP or stock where one exists; a basic render test
may instead check identical latents across ranks. **impl** means the
code exists but its required comparison is missing or failed; a render that
completes without a comparison does not count. The Notes and model details
identify the tested configurations; `HW` does not mean every variant or mode was tested.
Identity Gate verdicts authorize residency optimizations separately, and
`INCONCLUSIVE/no_material` is not a failed fidelity comparison.

## Choose a workflow

Start with the supplied graph for your model. It includes the conditioning,
sampling and memory settings that model needs. Open **Workflow > Browse
Templates > dgx-monarch** in ComfyUI, or use the links below.

| Model | Workflows | Before you start |
|---|---|---|
| [Krea2 RAW and Turbo](#model-krea2-raw-turbo) | Text-to-image | Ulysses is the automatic choice. Explicit cfg2 can split the two calls made by a CFG++ sampler, even at CFG 1. Reference images require `single`. |
| [Ideogram4](#model-ideogram4) | Dual-model text-to-image | Explicit cfg2 puts the conditional and unconditional checkpoints on separate ranks. Use the supplied guider and sampler wiring. |
| [Qwen Image 2.1](#model-qwen-image-2-1) | Text-to-image, RGBA and masked-reference editing, up to ten references in the shipped graph | Use Torch Flash. Reference masks condition generation; they do not lock output pixels. |
| [Mage Flow](#model-mage-flow-t2i-edit-quality-turbo) | Quality, Turbo, Edit and Edit Turbo | Each variant needs its own checkpoint and graph. |
| [MiniMax H3](#model-minimax-h3-t2va-fl2va-ref2va-av) | Video with audio, still and clip guides | Use batch one and the matching guide settings. Clip encoding needs additional memory. |
| [Wan-Animate 2](#model-wan-animate-2) | Driven animation with INT8, BF16, quality and distilled recipes | Start with the INT8 workflow for warm iteration. Supply a driving video, reference image and both prompts. |

See [BENCHMARKS.md](BENCHMARKS.md) for measured render times. A faster
render and a successful output comparison are separate results. In particular,
Qwen-Image 2512 timings do not describe Qwen Image 2.1.

## Support matrix

The matrix lists each family's parallel modes. `single` works unless the Notes
say otherwise; above world size 1 it becomes data parallelism, which
packed-latent families refuse. Under `auto`, the Init node picks among the
distributed modes.

Cell legend: **✓** supported · **★** selected by `auto` · **🔒** constrained
(see Notes) · **n/a** the mode runs but no `auto` row offers it and nothing
measured it · **✗** the runtime typed-refuses it for every artifact in this row
· **⚪** FSDP was not exercised because resident execution fits every validated
configuration; that is not a claim it would work. FSDP is a manual
capacity choice: `auto` never selects it.

**★** means `auto` can select that mode at the family's usual CFG and some
resolution; it does not mean every configuration passed a hardware test.
**★⏳** means the selected mode has no current passing comparison, either
because it has not been tested or because a previous result no longer applies.
The row's Notes explain which. The table is checked against runtime choices
and reviewed against the linked test results.

Chroma (Radiance included) and LongCat have cfg2 rows below 1.2 MP, taken at
any CFG other than 1 when the row supports that precision, and
Ulysses rows from 1.2 MP up. Mage Flow and Qwen Image 2.1 have no row, and
Flux, Flux 2, Krea 2, Lens, Qwen-Image and Z-Image have rows only from 1.2 MP
up; those cases take the generic uly2 fallback (row 99). Every other family
has one full-range Ulysses or ring row.

On more than one rank, the first `auto` render of each family in a process
logs a reminder that identifying a model from its file header does not verify
that workflow's output. Explicit presets log no reminder. A passing result
covers only its listed checkpoint, graph, LoRA, precision and topology.

Each row links to its model details and to a workflow in `example_workflows/`,
or says `no template yet`; a missing template does not mean
missing runtime support. In ComfyUI, open Workflow > Browse Templates >
dgx-monarch. Use each workflow's template settings, and choose artifacts
that match the tested configurations in its model details. See
[QUICKSTART.md](QUICKSTART.md) for first use.

### Image models

| Model | single | cfg2 | Ulysses | ring | FSDP | Status | Notes |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| Krea2 (RAW+Turbo) | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | Padded ring needs a waiver; reference-image graphs use single. [Details](#model-krea2-raw-turbo) · [Workflow](../example_workflows/dgx-monarch-krea2-t2i.json) |
| Chroma | ✓ | ★ | ★ | ✓ | ✓ | **HW** | NVFP4 is waiver-only; Ulysses rejects effective masks. [Details](#model-chroma) · [Workflow](../example_workflows/dgx-monarch-chroma-t2i.json) |
| Radiance | ✓ | ★ | ★ | ✓ | ✓ | **HW** | cfg2 requires equal-length prompts. [Details](#model-radiance) · [Workflow](../example_workflows/dgx-monarch-radiance-t2i.json) |
| Ideogram4 | ✓ | ✓ | ★ | 🔒 | ✓ | **HW** | Explicit cfg2 splits the two checkpoints; sharded attention uses Torch Flash. [Details](#model-ideogram4) · [Workflow](../example_workflows/dgx-monarch-ideogram4-t2i.json) |
| Flux 1 Dev | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Padded ring is guarded; distilled checkpoints use CFG 1. Use converted LoRAs. [Details](#model-flux-1-dev) · [Workflow](../example_workflows/dgx-monarch-flux1-t2i.json) |
| Flux 1 Schnell | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | CFG 1; Schnell-specific padded inputs remain unmeasured. [Details](#model-flux-1-schnell) · [Workflow](../example_workflows/dgx-monarch-flux1-t2i.json) |
| Flux2 | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | FP8-mixed Ulysses and BF16 `uly2+fsdp` were compared with references. LoRA/FSDP rendered, but its reference ran out of memory. [Details](#model-flux2) · [Workflow](../example_workflows/dgx-monarch-flux2-t2i.json) |
| LongCat-Image | ✓ | ★ | ★ | ✓ | ✓ | **HW** | Padded image streams use Ulysses; ring/hybrid padding is guarded. [Details](#model-longcat-image) · [Workflow](../example_workflows/dgx-monarch-longcat-t2i.json) |
| HunyuanImage 2.1 (+refiner) | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Extended-text cfg2 needs equal Qwen lengths; refiner FSDP measured at FP8 only; SR FSDP unmeasured. [Details](#model-hunyuanimage-2-1-refiner) · [Workflow](../example_workflows/dgx-monarch-hunyuan-image.json) |
| Qwen-Image | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Edit reference-method limits differ from base; sharded Edit can refuse. [Details](#model-qwen-image) · [Workflow](../example_workflows/dgx-monarch-qwen-image-t2i.json) |
| Mage-Flow (T2I/Edit, quality + Turbo) | ✓ | ✓ | ★ | 🔒 | ✓ | **HW** | T2I/Edit/Turbo Ulysses tested; FSDP tested only on Base BF16 without LoRA. [Details](#model-mage-flow-t2i-edit-quality-turbo) · [Workflow](../example_workflows/dgx-monarch-mage-flow-t2i.json) |
| Qwen Image 2.1 | ✓ | 🔒 | ★ | 🔒 | 🔒 | **HW** | Torch Flash only; Compressed caches were compared with the same cache settings in native ComfyUI, not with cache off. [Details](#model-qwen-image-2-1) · [Workflow](../example_workflows/dgx-monarch-qwen-image21-t2i.json) |
| Ernie-Image | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | cfg2 comparisons used equal-shaped prompts; unequal-prompt cfg2 has no reference comparison. [Details](#model-ernie-image) · [Workflow](../example_workflows/dgx-monarch-ernie-t2i.json) |
| Z-Image latent | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | cfg2 requires equal-token prompts; Ulysses degree must divide 30 heads. [Details](#model-z-image-latent) · [Workflow](../example_workflows/dgx-monarch-zimage-t2i.json) |
| Z-Image DCT PixelSpace | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | L2P is unsupported; DCT and latent checkpoints need different workflows. [Details](#model-z-image-dct-pixelspace) · [Workflow](../example_workflows/dgx-monarch-zimage-dct-t2i.json) |
| Lens | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Asymmetric cfg2 is measured; padded ring remains guarded. [Details](#model-lens) · [Workflow](../example_workflows/dgx-monarch-lens-t2i.json) |
| Omnigen2 | ✓ | 🔒 | ✗ | ★ | ✓ | **HW** | Ulysses is unsupported; ring+FSDP refuses. [Details](#model-omnigen2) · [Workflow](../example_workflows/dgx-monarch-omnigen2-t2i.json) |
| Anima | ✓ | ✓ | ★ | ✓ | 🔒 | **HW** | Use the supplied graph for cross-attention and masks. Explicit cfg2 was compared; auto selects Ulysses. [Details](#model-anima) · [Workflow](../example_workflows/dgx-monarch-anima-t2i.json) |
| Boogu | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Unequal prompts use per-conditioning cfg2 dispatch. [Details](#model-boogu) · [Workflow](../example_workflows/dgx-monarch-boogu-t2i.json) |
| PixelDiT / PiD | ✓ | n/a | ★ | ✓ | ✓ | **HW** | Only BF16 was compared; quantized sequence sharding and cfg2 refuse. [Details](#model-pixeldit-pid) · [Workflow](../example_workflows/dgx-monarch-pixeldit-t2i.json) |
| Kandinsky5-Image | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Use SD3 shift 3; preserve the native 16-channel latent. [Details](#model-kandinsky5-image) · [Workflow](../example_workflows/dgx-monarch-kandinsky5-image.json) |

### Video models

| Model | single | cfg2 | Ulysses | ring | FSDP | Status | Notes |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| LTX 2.3 (t2v) | ✓ | n/a | ★ | ✓ | ✓ | **HW** | Native RDMA remains NOT RUN / HOLD. [Details](#model-ltx-2-3-t2v) · [Workflow](../example_workflows/dgx-monarch-ltx-t2v.json) |
| LTX 2.3 (i2v/AV packed) | ✓ | ✗ | ★ | ✓ | ✓ | **HW** | Packed DP refuses; raw-latent acceptance excludes decoded quality. [Details](#model-ltx-2-3-i2v-av-packed) · [Workflow](../example_workflows/dgx-monarch-ltx-i2v-av.json) |
| LTX 2.5 (t2v/AV packed) | ✓ | ✗ | ★ | ✓ | 🔒 | **HW** | Packed DP/cfg2 refuse; FSDP and extended guide recipes lack passing output comparisons; see failed and incomplete tests below. [Details](#model-ltx-2-5-t2v-av-packed) · [Workflow](../example_workflows/dgx-monarch-ltx25-t2v.json) |
| Wan 2.1 (t2v) | ✓ | n/a | ★ | ✓ | ✓ | **HW** | FP8 Ulysses passed its output comparisons. [Details](#model-wan-2-1-t2v) · [Workflow](../example_workflows/dgx-monarch-wan-t2v.json) |
| Wan 2.2 (t2v, high/low) | ✓ | n/a | ★ | ✓ | ✓ | **HW** | BF16 and FP16 high/low were tested separately; FP8 T2V has not been compared. [Details](#model-wan-2-2-t2v-high-low) · [Workflow](../example_workflows/dgx-monarch-wan22-t2v.json) |
| Wan 2.2 (i2v) | ✓ | n/a | ★ | ✓ | ✓ | **HW** | The four-step LoRA test and no-LoRA FP8 test used different recipes. [Details](#model-wan-2-2-i2v) · [Workflow](../example_workflows/dgx-monarch-wan22-i2v.json) |
| Wan 2.1 (i2v) | ✓ | n/a | ★ | ✓ | ✓ | **HW** | FP8 Ulysses passed; the BF16/FSDP capacity alternative has no reference comparison. [Details](#model-wan-2-1-i2v) · [Workflow](../example_workflows/dgx-monarch-wan21-i2v.json) |
| Wan FlowRVS | ✓ | n/a | ★ | ✓ | ✓ | **HW** | Tested with synthetic clips. Use CFG 1 and no added noise for segmentation. [Details](#model-wan-flowrvs) · [Workflow](../example_workflows/dgx-monarch-wan-flowrvs.json) |
| Wan Bernini-R | ✓ | n/a | ★ | ✓ | ✓ | **HW** | FP8 high/low T2V only; fresh solver history per stage. [Details](#model-wan-bernini-r) · [Workflow](../example_workflows/dgx-monarch-wan-bernini.json) |
| Wan SCAIL Preview | ✓ | n/a | ★ | ✓ | ✓ | **HW** | World-2 Ulysses, FP16 one- and four-step synthetic 256x256x5 only; default workflow lacks a usable reference; ring and FSDP have not been tested on hardware. [Details](#model-wan-scail-preview) · [Workflow](../example_workflows/dgx-monarch-wan-scail.json) |
| Wan SCAIL2 | ✓ | n/a | ★ | ✓ | ✓ | **HW** | Tested with synthetic conditioning. Supply both masks; the workflow produces one chunk. [Details](#model-wan-scail2) · [Workflow](../example_workflows/dgx-monarch-wan-scail2.json) |
| WanDancer | ✓ | n/a | ★ | ✓ | ✓ | **HW** | Two-pass FP8 tested; reference_latent is unsupported. [Details](#model-wandancer) · [Workflow](../example_workflows/dgx-monarch-wandancer.json) |
| Wan-Animate 2 | ✓ | n/a | ★ | 🔒 | 🔒 | **HW** | Warm INT8/BF16 workflows tested; FSDP, ring and cache have not been tested on hardware. [Details](#model-wan-animate-2) · [Workflow](../example_workflows/dgx-monarch-wan-animate2.json) |
| HunyuanVideo 1.5 (+SR) | ✓ | 🔒 | ★ | ✓ | 🔒 | **HW** | cfg2 plain-text only; SR uses explicit topology; FSDP unmeasured. [Details](#model-hunyuanvideo-1-5-sr) · [Workflow](../example_workflows/dgx-monarch-hunyuan-video.json) |
| CogVideoX 1.5 (t2v) | ✓ | ✓ | ★ | ✓ | ✓ | **HW** | Fixed 226-token conditioning; the tested 1.0-layout LoRA did not apply. [Details](#model-cogvideox-1-5-t2v) · [Workflow](../example_workflows/dgx-monarch-cogvideox-t2v.json) |
| CogVideoX 1.5 (i2v) | ✓ | ✓ | ★ | ✓ | ✓ | **HW** | Use the ComfyUI BF16 I2V checkpoint; Diffusers loading and inpaint have not been compared. [Details](#model-cogvideox-1-5-i2v) · [Workflow](../example_workflows/dgx-monarch-cogvideox-i2v.json) |
| CogVideoX 1.5 (inpaint) | ✓ | ✓ | ★ | ✓ | ✓ | impl | No inpaint comparison; no template yet. [Details](#model-cogvideox-1-5-inpaint) |
| Kandinsky5-Video Lite | ✓ | 🔒 | ★ | ✓ | ⚪ | **HW** | Use SD3 shift 5; FSDP unexercised. [Details](#model-kandinsky5-video-lite) · [Workflow](../example_workflows/dgx-monarch-kandinsky5-video-lite.json) |
| Kandinsky5-Video Pro | ✓ | 🔒 | ★ | ✓ | 🔒 | **HW** | FSDP validated for the 4096-dimension model; LoRA/FSDP unmeasured. [Details](#model-kandinsky5-video-pro) · [Workflow](../example_workflows/dgx-monarch-kandinsky5-video-pro.json) |
| MiniMax H3 (t2va/fl2va/ref2va AV) | 🔒 | ✗ | ★ | 🔒 | 🔒 | **HW** | Packed batch one; padded ring and Sol-Attn require waivers. [Details](#model-minimax-h3-t2va-fl2va-ref2va-av) · [Workflow](../example_workflows/dgx-monarch-minimax-h3-t2va.json) |

## Image model details

<a id="model-krea2-raw-turbo"></a>

<a id="krea2-raw--turbo-wan-family-image-dit"></a>

### Krea2 (RAW+Turbo)

Detail status (`Krea2 (RAW+Turbo)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-krea2-t2i.json).

**Tested configurations.** RAW/Turbo resident Ulysses, cfg2, and named BF16/FP8/INT8 paths, including blank-negative cfg2 and the FSDP comparisons linked below. Pure-Ulysses one-step renders on Torch Flash produced identical tensors to one GPU: RAW BF16 at 1536 and 1448 with odd and even text and image streams, at batch 2 and on the template, and the template on RAW INT8 ConvRot and Turbo FP8-scaled; the SAGE route that auto picks for FP8 was not part of that reading.

**Known limitations.** Reference latents refuse cfg2 and sequence parallelism;
run reference-image renders on `single`. Padded ring needs a waiver and has no passing
output comparison. Use NVFP4 only with the checkpoint and settings listed in
the test results.
LoRA on FSDP refuses class P: it needs `lora_low_rss` on, quantized shards
refuse by contract, and the RAW BF16 file's fp32-stored weights refuse because
they run as BF16. Run Krea2
LoRAs on a resident topology.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-krea2-raw-turbo).

**Workflow guidance.**

Auto never selects cfg2 for Krea2. Below 1.2 MP no Krea2 row matches, so it
takes the generic uly2 fallback; from 1.2 MP up it selects Ulysses, with SAGE
on an FP8 checkpoint, which needs sageattention on each worker. At CFG 1 a
plain sampler drops the unconditional pass, so cfg2 has nothing to split; a
CFG++ sampler keeps that pass and can use explicit cfg2. On cfg2, asymmetric
prompts are padded automatically; combined Ulysses+cfg has no such padding and
refuses them. A zeroed negative is real conditioning and stays unmasked.

<a id="model-chroma"></a>

<a id="chroma-flux-family-dit"></a>

### Chroma

Detail status (`Chroma`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-chroma-t2i.json).

**Tested configurations.** BF16 pure Ulysses at 1024 and 1536 and named multi-step controls; cfg2 through the mask-trim path; named FP8 and INT8 controls. FSDP support is separate from these resident comparisons.

**Known limitations.** These comparisons do not cover ring, hybrid, effective masks, other precisions or NVFP4. NVFP4 needs its waiver even though one uneven cfg2 pair reads exact under it.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-chroma).

**Workflow guidance.**

Below 1.2 MP at any CFG other than 1, auto selects cfg2 for BF16, FP8 and
INT8 checkpoints; from 1.2 MP up it selects Ulysses, with SAGE on FP8, which
needs sageattention on each worker. The mask and NVFP4 guards still apply.

<a id="model-radiance"></a>

### Radiance

Detail status (`Radiance`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-radiance-t2i.json).

**Tested configurations.** FP32 at 1024x1024 on Ulysses, ring2 and cfg2, including a 30-step render; Ulysses with odd-length text; FSDP against ordinary residency with the BF16 core and two replicated FP32 parameters.

**Known limitations.** Unequal cfg2 prompts refuse before dispatch, because the pad mask has no latent grid in pixel space. Chroma's fixes and controls cover Radiance only on the paths tested on Radiance.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-radiance).

<a id="model-ideogram4"></a>

<a id="ideogram4-pixeldit-head_dim-256"></a>

### Ideogram4

Detail status (`Ideogram4`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ideogram4-t2i.json).

**Tested configurations.** Ulysses comparisons for the named FP8, BF16, INT8 ConvRot and MXFP8 pairs. Dual-model cfg2 has its own FP8 pair reference comparison and finite render, and explicit cfg2 was tested on the shipped graph for the BF16, INT8 ConvRot, MXFP8 and NVFP4-mixed pairs: one-step NRMS 0.0 against one GPU on Torch Flash, cuDNN and SAGE_AUTO. Other precisions do not inherit these cfg2 results. The NVFP4 limit below applies to the sequence-split presets, which read 0.10 to 0.13 in the same campaign. Pure-Ulysses one-step renders of the FP8-scaled conditional model at CFG 1 produced identical tensors to one GPU at 1024 and 1040 with 283 and 284 text rows, and with `uly2+fsdp` at 1024. A one-step uly2 probe of the shipped dual-model graph at CFG 7 produced identical tensors to one GPU on the BF16, FP8 and INT8 pairs. That graph also rendered on `uly2+fsdp` for the BF16, FP8 and INT8 pairs, one record each, identical to its resident twin.

**Known limitations.** NVFP4 is over the fidelity limit. Ring is not automatic. Under the default `auto_gate=first_use`, FSDP refuses a dual-model request, because the clean-reload Gate proves one model only; `auto_gate=off` runs it under FSDP with slab and low-RSS residency off. `cfg2+fsdp` refuses for this family on every setting. No claim extends to other artifacts or kernels.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ideogram4).

**Workflow guidance.**

For dm-cfg2, use the unconditional loader, dual-model guider and custom
sampler; leave the negative input unconnected for the text-free unconditional
pass. Each rank holds one checkpoint; single/DP instead keeps both on each
rank. Leave the resident-adoption input unconnected under dm-cfg2. The shipped
workflow uses `sync_ulysses=off`. On a Ulysses or ring topology the worker
runs TORCH_FLASH in place of a cuDNN or sage kernel named on the Init node,
because neither carries head dimension 256.

<a id="model-flux-1-dev"></a>

### Flux 1 Dev

Detail status (`Flux 1 Dev`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-flux1-t2i.json).

**Tested configurations.** BF16 Ulysses at 1024 and 1448, including odd image grids and 313/316 text rows; BF16 FSDP against ordinary residency; one two-LoRA FSDP stack using low-RSS baking.

**Known limitations.** XLabs-format LoRAs do not apply; use converted LoRAs. A padded ring or hybrid render refuses unless waived. Unlisted LoRA and artifact combinations are not covered.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-flux-1-dev).

<a id="model-flux-1-schnell"></a>

### Flux 1 Schnell

Detail status (`Flux 1 Schnell`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-flux1-t2i.json). Use the shared Flux graph with the Schnell checkpoint, CFG 1 and its four-step schedule.

**Tested configurations.** BF16 at 1024x1024 on Ulysses and ring2, with a completed four-step render; BF16 FSDP against ordinary residency.

**Known limitations.** The padded Schnell path is unmeasured. Keep CFG at 1.0: the checkpoint is guidance distilled.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-flux-1-schnell).

<a id="model-flux2"></a>

### Flux2

Detail status (`Flux2`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-flux2-t2i.json).

**Tested configurations.** FP8-mixed Ulysses at 1024 and padded 1040 square; BF16 Ulysses+FSDP against an independent stock reference. Three BF16 FSDP+LoRA cold/warm runs completed, but lack reference output.

**Known limitations.** The LoRA records lack a completed resident-reference
comparison, so output accuracy is unknown. Quantized-shard LoRA-on-FSDP and
fp32-island dtype mismatches refuse by contract. Padded ring and hybrid
renders are guarded. Large BF16 loads can also use the slab-resident route. In
the earlier campaign, on the capped head beside the driver's
text encoder, the resident fp8mixed Ulysses path that auto selects mostly
refused class C: all six explicit uly2 and ring2 records, and 5 of 6
lever-free auto records. The store now places stock weights at load time and checks memory against the
full placement peak. A hardware test of that path admitted the load and rendered.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-flux2).

<a id="model-longcat-image"></a>

### LongCat-Image

Detail status (`LongCat-Image`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-longcat-t2i.json).

**Tested configurations.** BF16 Ulysses at 1024 and padded 1448 square; cfg2 with unequal-prompt padding and masks; BF16 FSDP against ordinary residency.

**Known limitations.** Padded ring and hybrid renders are guarded; no claim covers unlisted artifacts, LoRAs or topologies. Below 1.2 MP auto selects cfg2 only for BF16 at any CFG other than 1, because only BF16 cfg2 has been compared; other quantizations take uly2 there.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-longcat-image).

<a id="model-hunyuanimage-2-1-refiner"></a>

### HunyuanImage 2.1 (+refiner)

Detail status (`HunyuanImage 2.1 (+refiner)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-hunyuan-image.json).

**Tested configurations.** Base Image BF16/FP8 with ordinary residency and FSDP, including a 20-step BF16 FSDP render. Refiner tests with ordinary residency and FSDP used separate comparisons. The FP8 base plus FP8 refiner testing graph ran `uly2+fsdp` identical to its resident legs in 12 records with both checkpoints sharded. Pure-Ulysses one-step renders of the FP8 Image 2.1 base produced identical tensors to one GPU at 2048 with 52, 169 and 172 text rows and at 2080 with 169 and 172, with the prompt as negative and with an empty one; the refiner had no gate.

**Known limitations.** SR FSDP is unmeasured, and refiner FSDP is measured
only on the FP8 pair. On that graph's tested resident
presets, Torch Flash and cuDNN warm legs refused class C on the capped head in
20 records, while all 12 SAGE_AUTO records passed. The tested public LoRA mapped no patches in
ComfyUI. No LoRA hot-swap comparison passed. Base Image tests do not cover the refiner or SR.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-hunyuanimage-2-1-refiner).

<a id="model-qwen-image"></a>

### Qwen-Image

Detail status (`Qwen-Image`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-qwen-image-t2i.json).

**Tested configurations.** Base 2512 FP8 Ulysses/cfg2 and the named FSDP presets; Edit-2511 FP8-mixed DP2/cfg2. The base-model FSDP tests do not cover Edit FSDP. Pure-Ulysses one-step renders of the base 2512 FP8 file produced identical tensors to one GPU at 1328 and 1024 with odd and even image and text streams, and at batch 2.

**Known limitations.** Default Edit `index_timestep_zero` refuses sequence sharding. Batch-one DP FSDP is refused; base-model results do not establish accuracy for Edit or other reference methods. On Ulysses, auto keeps the Init node's attention kernel and does not switch base FP8 to SAGE, because SAGE on Ulysses or ring measured over the fidelity limit; an explicit SAGE choice still runs as selected. The LoRA test graph also exceeded the limit on Ulysses with Torch Flash and cuDNN.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-qwen-image).

<a id="model-mage-flow-t2i-edit-quality-turbo"></a>

### Mage-Flow (T2I/Edit, quality + Turbo)

Detail status (`Mage-Flow (T2I/Edit, quality + Turbo)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-mage-flow-t2i.json).

**Tested configurations.** Twelve T2I precision rows, named quality/Edit/Turbo recipes, and Base T2I BF16 no-LoRA Ulysses+FSDP. Pure-Ulysses one-step T2I renders produced identical tensors to one GPU: the BF16 file at 1024 and 1040 with odd, even, equal and long prompts, at batch 2, under `uly2+fsdp` and cast to plain FP8; the Turbo INT8 ConvRot file at 1024; and, at 1024, the MXFP8 native-view, FP8 mixed, NVFP4 native-view and mixed NVFP4 files.

**Known limitations.** Ring, other reference inputs, Edit/Turbo or quantized FSDP, and LoRA-backed slab have no passing reference comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-mage-flow-t2i-edit-quality-turbo).

**Workflow guidance.**

Use the separate [Edit](../example_workflows/dgx-monarch-mage-flow-edit.json),
[Turbo](../example_workflows/dgx-monarch-mage-flow-t2i-turbo.json) or
[Edit Turbo](../example_workflows/dgx-monarch-mage-flow-edit-turbo.json)
workflow and its matching checkpoint; the Base artifact does not replace
those variants.

<a id="model-qwen-image-2-1"></a>

<a id="qwen-image-21-native-comfy-conditioning-and-workflow-contract"></a>

### Qwen Image 2.1

Detail status (`Qwen Image 2.1`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-qwen-image21-t2i.json).

**Tested configurations.** Twelve DiT formats, named text/masked/ten-reference recipes, and cache-off/default/INT8/INT4 tests against native ComfyUI with the same cache settings. The named 25-step T2I, masked-edit and ten-reference controls produced identical complete latents and RGBA pixels to their native references.

**Known limitations.** Ulysses admits only TORCH_FLASH. Compressed caches matched native ComfyUI using the same cache settings. INT8 and INT4 caching are approximate; these tests do not show equality with cache off. No claim covers a DiT and cache quantization cross-product, FSDP, unlisted workflows, locked pixels, the prompt enhancer or FunControlNet.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-qwen-image-2-1).

**Workflow guidance.**

The [RGBA edit](../example_workflows/dgx-monarch-qwen-image21-edit-rgba.json)
and [masked-reference edit](../example_workflows/dgx-monarch-qwen-image21-edit-mask-rgba.json)
workflows are separate from T2I. The native conditioning node takes ordered
references `image_1` through `image_16`, and the shipped edit graph exposes up
to ten; the node returns positive and negative conditioning plus a batch-one
empty latent. The edit workflows rebuild RGBA with `JoinImageWithAlpha`,
because Load Image returns RGB plus an inverse-alpha mask. This is
alpha-conditioned reference editing, with no locked-pixel or denoise-mask
guarantee. The template is batch one, 1024 square, 25 Euler/simple steps at
CFG 1.

Keep the worker cache off unless you choose `DGXMonarchQwenImage21Cache`; the
stock MODEL cache node cannot connect to DGXM_MODEL. Default cache storage
keeps native K/V values, while INT8 and INT4 compression is approximate.

<a id="model-ernie-image"></a>

### Ernie-Image

Detail status (`Ernie-Image`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ernie-t2i.json).

**Tested configurations.** BF16 Ulysses, cfg2 with equal-shaped prompts, and BF16 FSDP against ordinary residency. Pure-Ulysses one-step renders of the BF16 file produced identical tensors to one GPU at 1024 and 1040: the conditional call alone, the template pair and an equal-length pair in one batch-2 call, with `uly2+fsdp` at 1024.

**Known limitations.** Unequal-prompt per-condition cfg2 dispatch has no comparison of its own, so its output accuracy has not been established. Effective masks and unlisted graphs are not covered either.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ernie-image).

<a id="model-z-image-latent"></a>

<a id="z-image-latent-and-dct-pixelspace"></a>

### Z-Image latent

Detail status (`Z-Image latent`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-zimage-t2i.json).

**Tested configurations.** BF16/Turbo Ulysses; cfg2 with equal-token prompts; BF16 and Turbo BF16 FSDP against ordinary residency.

**Known limitations.** On pure cfg2 the runtime admits prompts of unequal token counts, but no Z-Image comparison covers them: the passing cfg2 test used equal-token prompts. L2P is a separate, unsupported model; its loader refusal is not base-latent evidence.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-z-image-latent).

<a id="model-z-image-dct-pixelspace"></a>

### Z-Image DCT PixelSpace

Detail status (`Z-Image DCT PixelSpace`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-zimage-dct-t2i.json).

**Tested configurations.** BF16 at 1536x1536 on Ulysses and ring2, cfg2 with equal-token prompts, and a completed 30-step render.

**Known limitations.** L2P refuses at load and on the worker until upstream model support and the adapter forward exist. DCT tests do not cover L2P.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-z-image-dct-pixelspace).

<a id="model-lens"></a>

### Lens

Detail status (`Lens`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-lens-t2i.json).

**Tested configurations.** Ulysses with BF16, MXFP8 and Turbo checkpoints; BF16 FSDP against ordinary residency; cfg2 with equal-shaped prompts; and the supplied Turbo BF16 cfg2/auto graph with unequal prompts. The shipped BF16 graph's pure-Ulysses one-step render produced identical tensors to one GPU. The campaign also tested the shipped graph's cfg2 with the MXFP8 artifact: one-step NRMS 0.0 against one GPU on Torch Flash, cuDNN and SAGE_AUTO.

**Known limitations.** The asymmetric comparison passed within its stated tolerance and does not produce identical tensors; it is separate from the shape-equal result. Padded ring and hybrid renders stay waiver-only.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-lens).

<a id="model-omnigen2"></a>

### Omnigen2

Detail status (`Omnigen2`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-omnigen2-t2i.json).

**Tested configurations.** FP16 ring2 against a DP2 reference; cfg2 with unequal prompts against one GPU; and FP16 cfg2+FSDP against ordinary residency.

**Known limitations.** Ring+FSDP refuses because a model reload changes ring output. Three FP16 `cfg2+fsdp` records matched their resident legs exactly. Equal-prompt cfg2 performance has not been remeasured after the dispatch change. Ulysses cannot split its 21 query and 7 KV heads. With `comfy_managed` on, the first-use Gate failed on Torch Flash and SAGE_AUTO ring2 in the campaign and the cell refused class P; cuDNN passed. See the linked results for each tested configuration.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-omnigen2).

<a id="model-anima"></a>

### Anima

Detail status (`Anima`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-anima-t2i.json).

**Tested configurations.** Ulysses against one GPU, DP2, and BF16 FSDP with the auxiliary adapter replicated. Explicit cfg2 on the shipped template's prompt pair read one-step NRMS 0.0 against one GPU in three records, and three `cfg2+fsdp` records matched their resident legs exactly (BF16, 1024 square).

**Known limitations.** Unlisted FSDP artifacts and LoRA behavior are not covered. Auto has no cfg2 row: comfy's 512-row pad lands inside the forward, so an unequal pair splits the model call at any length. The cfg2 comparison covers that one prompt pair.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-anima).

<a id="model-boogu"></a>

### Boogu

Detail status (`Boogu`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-boogu-t2i.json).

**Tested configurations.** FP8 Ulysses, FSDP without changing stored precision, and cfg2 with unequal prompts. Pure-Ulysses one-step renders of the fp8 file produced identical tensors to the dp2 reference at 1024 and 1040, odd and even streams.

**Known limitations.** Equal-prompt cfg2 performance has not been measured after the batching change. Do not pad a dispatching conditioning pair.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-boogu).

<a id="model-pixeldit-pid"></a>

### PixelDiT / PiD

Detail status (`PixelDiT / PiD`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-pixeldit-t2i.json).

**Tested configurations.** PixelDiT BF16 Ulysses/ring; PiD BF16 Ulysses; named PiD BF16 Ulysses+FSDP resident comparisons.

**Known limitations.** Quantized sequence sharding and cfg2 are outside the fidelity limit. PiD 1.5, other checkpoints, LoRA, larger world sizes and other shapes need separate comparisons.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-pixeldit-pid).

<a id="model-kandinsky5-image"></a>

### Kandinsky5-Image

Detail status (`Kandinsky5-Image`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-kandinsky5-image.json).

**Tested configurations.** Ulysses and cfg2 against DP2, BF16 FSDP against ordinary residency, and the 50-step image recipe.

**Known limitations.** Do not apply the image shift to video variants. LoRA and unlisted variants are not covered.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-kandinsky5-image).

**Workflow guidance.**

Keep `DGXMonarchModelSamplingSD3` at shift 3. The image latent has
16 channels; the supplied latent node has the required shape. Changing the
sampling shift is per-render math and does not require a checkpoint reload.

## Video model details

<a id="model-ltx-2-3-t2v"></a>

<a id="ltx-23-and-25-t2v--i2v--av-video-dit"></a>

### LTX 2.3 (t2v)

Detail status (`LTX 2.3 (t2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ltx-t2v.json).

**Tested configurations.** 22B distilled Ulysses and LoRA hot-swap; the named FP8-scaled decoded-video probe also passed its local comparison.

**Known limitations.** Native latent RDMA remains NOT RUN / HOLD. Use actor messaging; other artifacts and graph settings need separate comparisons.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ltx-2-3-t2v).

<a id="model-ltx-2-3-i2v-av-packed"></a>

### LTX 2.3 (i2v/AV packed)

Detail status (`LTX 2.3 (i2v/AV packed)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ltx-i2v-av.json).

**Tested configurations.** Packed video/audio raw-latent acceptance, including the named Ulysses/ring comparisons and BF16 low-RSS rank-384 LoRA workflow.

**Known limitations.** The tests compared sampler latents, not decoded video or audio quality. Packed data parallelism refuses. FP8 FSDP can pass the load checks but has no passing output comparison here.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ltx-2-3-i2v-av-packed).

**Workflow guidance.**

Use strength-1, unmasked guides for the accepted packed graph. Its guide
attention-bias restrictions differ from the measured LTX 2.5 Ulysses path;
do not assume settings tested on 2.5 also work on 2.3.

<a id="model-ltx-2-5-t2v-av-packed"></a>

### LTX 2.5 (t2v/AV packed)

Detail status (`LTX 2.5 (t2v/AV packed)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ltx25-t2v.json).

**Tested configurations.** Two-GPU text-to-video with audio on INT8 ConvRot and BF16 checkpoints. INT8 image guides were compared on Ulysses/ring2; attenuated guides were compared on pure Ulysses. See the linked results for each guide and LoRA recipe.

**Known limitations.** Packed DP, cfg2 and world-2 single refuse. FSDP tests refused before dispatch; BF16 plus LoRA could not complete the reload comparison, and BF16 image conditioning refused after partial loading. Multishot and two-stage tests remain CHECK; duration-head paths have no passing comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ltx-2-5-t2v-av-packed).

**Workflow guidance.**

Use ComfyUI `bd34f338` or newer and keep the checkpoint's `config` metadata intact; a forced family cannot
repair missing geometry. Strength-1 unmasked guides were compared on
Ulysses and ring; attenuated guides use pure Ulysses and remain refused
on ring/hybrid. Spatial masks, multishot and the two-stage graph have
CHECK results rather than complete passing comparisons.

Stock duration, per-modality CFG and guidance nodes take MODEL and cannot
be connected to DGXM_MODEL. Use the distributed workflows and their
serialized sampling settings. Use tiled decode for the diffusion video VAE.

<a id="model-wan-2-1-t2v"></a>

<a id="wan-21--22-t2v--i2v-video-dit-moe-14b5b"></a>

### Wan 2.1 (t2v)

Detail status (`Wan 2.1 (t2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-t2v.json).

**Tested configurations.** FP8 resident Ulysses: the short latent smoke and the separate decoded 81-frame one-step local-reference probe.

**Known limitations.** Other precision, topology, LoRA and FSDP combinations need separate comparisons.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-2-1-t2v).

<a id="model-wan-2-2-t2v-high-low"></a>

### Wan 2.2 (t2v, high/low)

Detail status (`Wan 2.2 (t2v, high/low)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan22-t2v.json).

**Tested configurations.** BF16 Ulysses high-to-low handoff with rank identity; a separate FP16 high/low graph passed its decoded 81-frame local-reference probe.

**Known limitations.** FP8 T2V high/low has no reference comparison. Each stage has its own sampler history and swaps the resident checkpoint; a new artifact or schedule needs its own comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-2-2-t2v-high-low).

<a id="model-wan-2-2-i2v"></a>

### Wan 2.2 (i2v)

Detail status (`Wan 2.2 (i2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan22-i2v.json).

**Tested configurations.** The named four-step FP8 high/low LightX2V workflow, LoRA hot-swap and 17-frame comparison; a separate no-LoRA FP8-scaled graph passed its decoded 81-frame local-reference probe.

**Known limitations.** The two recipes were tested separately. FSDP, general temporal consistency and subjective quality have not been established; a longer recipe does not inherit the four-step LoRA result.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-2-2-i2v).

<a id="model-wan-2-1-i2v"></a>

### Wan 2.1 (i2v)

Detail status (`Wan 2.1 (i2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan21-i2v.json). Use the template settings and validated artifacts.

**Tested configurations.** FP8-scaled auto/Ulysses: cold and warm renders completed, and the decoded 81-frame one-step probe matched the local reference exactly.

**Known limitations.** FSDP, including the template's BF16 `uly2+fsdp` option, other precisions and other workflows have no passing reference comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-2-1-i2v).

<a id="model-wan-flowrvs"></a>

### Wan FlowRVS

Detail status (`Wan FlowRVS`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-flowrvs.json). Use the template settings and validated artifacts.

**Tested configurations.** BF16 synthetic-clip Ulysses, including the decoded 17-frame one-step local-reference comparison and the earlier source-latent raw proof.

**Known limitations.** Real-media segmentation quality, other recipes and FSDP have no passing reference comparison. This is referring-video segmentation, not video generation: use CFG 1, source latents and no added noise.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-flowrvs).

<a id="model-wan-bernini-r"></a>

### Wan Bernini-R

Detail status (`Wan Bernini-R`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-bernini.json). Use the template settings and validated artifacts.

**Tested configurations.** FP8-scaled high/low text-to-video under Ulysses, with cold/warm completion and an exact decoded 81-frame one-step local-reference probe.

**Known limitations.** Start fresh solver history for each stage. In-context conditioning, other precisions, LoRA, FSDP and continuous solver history have no passing reference comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-bernini-r).

<a id="model-wan-scail-preview"></a>

### Wan SCAIL Preview

Detail status (`Wan SCAIL Preview`): **HW**.

**Workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-scail.json). Its 81-frame recipe has completed renders but lacks a matched reference comparison. The smaller test below does not qualify the default workflow.

**Tested configurations.** On two Sparks, explicit `uly2` with Torch Flash matched native ComfyUI on one-step and four-step synthetic reference-image tests at 256x256, five frames. The BF16 checkpoint ran in FP16 on both paths. In both tests, raw latents, all five decoded RGB frames and cross-rank latents were equal. The one-step test passed its NRMS limit of 0.10; four-step equality is a measured control, with no general video-quality threshold. The recipe used UniPC/simple, CFG 5 and shift 3, without pose input or LoRA.

**Known limitations.** The original 81-frame local reference refused for capacity and saved no frames, so that comparison remains CHECK. The passing tests do not cover BF16 execution, other precisions, pose-driven animation, LoRA, ring, FSDP, warm reuse or the default 40-step recipe. The activation-footprint preflight can refuse a job ([docs/TROUBLESHOOTING.md #47](TROUBLESHOOTING.md#47-wan-scailscail-2-render-refused-activation-footprint-preflight)).

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-scail-preview).

<a id="model-wan-scail2"></a>

### Wan SCAIL2

Detail status (`Wan SCAIL2`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-scail2.json). Use the template settings and validated artifacts.

**Tested configurations.** Official ComfyUI settings with the DPO LoRA, comparing sampler latents; and FP8-scaled Ulysses with synthetic conditioning, comparing 81 decoded one-step frames.

**Known limitations.** Arbitrary driving media, Turbo, FSDP and other LoRA settings have no passing reference comparison. The shipped graph renders one 81-frame chunk.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-scail2).

**Workflow guidance.**

Supply a driving clip, its per-character mask, the reference image and
its mask; both modes need all four. Replacement mode uses a white
driving-mask background and a black reference-mask background; animation
mode reverses them. Every `replacement_mode` widget in the graph must match.
The template is the official Comfy base recipe, with
Turbo off, DPO strength 1, Euler/simple, 40 steps, shift 5 and CFG 5.

<a id="model-wandancer"></a>

<a id="wandancer-two-pass-contract-and-scoped-acceptance-2026-08-06"></a>

### WanDancer

Detail status (`WanDancer`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wandancer.json). Use the template settings and validated artifacts.

**Tested configurations.** The named two-pass FP8 Ulysses graph with image/audio conditioning, decode/keyframe bridge and cross-rank identity.

**Known limitations.** No performance result or passing FSDP comparison is available. Other topologies and Wan variants are not covered; reference_latent is unsupported. A separate later sample capacity-refused its warm/probe legs and remains FINDING.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wandancer).

**Workflow guidance.**

Keep the required image and CLIP-Vision conditioning on both stages.
The global model consumes the image/audio and produces a latent; stock
VAEDecode and WanDancerPadKeyframesList then form the local segment inputs.
Comfy list execution iterates the local refinement graph, with fresh audio
conditioning, the keyframe mask and the same CLIP-Vision reference. Leave
`reference_latent` unwired: the adapter refuses it, because that prepend
would misalign the per-frame audio residual.

The API graph in `tests/fixtures/workflows/` is a structural fixture with
placeholder media. Its CPU tests check shape, not hardware acceptance; the
[two-pass test results](VALIDATION.md#evidence-wandancer) names what was measured.

<a id="model-wan-animate-2"></a>

<a id="wan-animate-2-contract-and-measured-scopes"></a>

### Wan-Animate 2

Detail status (`Wan-Animate 2`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-animate2.json), the warm INT8 accelerated default. The BF16 warm, memory-saver, Quality Base and Distilled recipes are described below.

**Tested configurations.** Base twelve-format Ulysses probes, named Base/Distilled portrait recipes, official INT8 ConvRot acceleration and INT8/BF16 warm full-decode sequences. Only each cold leg matched its preserved baseline; changed-seed/prompt outputs were distinct.

**Known limitations.** Sharded renders need TORCH_FLASH; any other kernel refuses. The warm tests do not cover slab/low-RSS, first-use LoRA, FSDP, ring, cache, background replacement or other shapes. Every Wan-Animate 2 template ships `slab_weights=off` and `lora_low_rss=off`. Keep ordinary residency for the warm recipes; other shapes or background applications may need larger memory reserves. Explicit `lora_low_rss=off` refuses FSDP with a LoRA on these templates (class P, `validate_fsdp_launch_loras`), and the INT8 checkpoint refuses it whatever `lora_low_rss` is, because its shards are quantized (`refuse_fsdp_lora_checkpoint_property`). Run a LoRA recipe on a resident topology. On the BF16 warm and memory-saver templates the resident LoRA backup is about one BF16 model per rank.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-animate-2).

**Workflow guidance.**

For ordinary iteration, choose the
[INT8 accelerated workflow](../example_workflows/dgx-monarch-wan-animate2.json) or the
[BF16 warm workflow](../example_workflows/dgx-monarch-wan-animate2-bfloat16-warm.json).
The BF16 workflow runs a soft Clear VRAM before full decode. This releases
unused allocator buffers while keeping actors, weights and LoRA backups.
Keep `slab_weights` and `lora_low_rss` off for the tested warm workflow.
Changing either setting selects a configuration outside those tests.

With both settings off and no FSDP, `auto_gate=first_use` skips the first-use
check. The INT8 template was tested unchanged: no queue logged a gate check,
and its cold latent matched the earlier cold output. A separate cold queue
with both settings on `auto` passed the first-use check in 185.7 s and left
about 16 GiB more free on the driver host. That check runs once per model and
LoRA combination. The other four Wan-Animate 2 templates were tested only
with `auto_gate=off`, not with their supplied gate setting.

The warm-reuse tests set both the driver launch flag `--reserve-vram` and
Init `uma_reserve_gb` to 8 for BF16 warm and to 24 for INT8. The
as-shipped INT8 run used `--reserve-vram 2` and the
template's `uma_reserve_gb` of 0.0. A template cannot set the launch flag.
Every template ships the Init `uma_reserve_gb` and `reserve_vram_gb` at 0.0;
with `reserve_vram_gb` at 0.0, workers on unified memory reserve 8 GiB unless
`cluster.toml` `[worker_args]` sets another value. Keep a higher reserve when
other applications need the memory. If BF16 admission refuses, choose INT8 or the
[memory-saver workflow](../example_workflows/dgx-monarch-wan-animate2-memory-saver.json),
which recycles actors before decode and reloads them on the next queue;
persistent Worker services keep running. Do not disable admission guards.

All variants require a driving video, reference image, generation prompt,
pose prompt and both CLIP-Vision encodes. The visible ImageScale sets the
canvas (portrait 480x832 by default); pose and first-frame encodes use that
same resized clip, and output keeps its FPS and audio. Each queue produces
one 81-frame chunk. For continuation, carry decoded frames into
`continue_motion` and the previous `video_frame_offset` into the next chunk.

[Quality Base](../example_workflows/dgx-monarch-wan-animate2-base-quality.json)
uses no LoRA, Euler/simple, 20 steps, CFG 1 and shift 5. Accelerated Base
uses official INT8 ConvRot plus LightX2V strength 1, LCM, six steps and CFG 1.
[Distilled](../example_workflows/dgx-monarch-wan-animate2-distilled.json)
uses its own checkpoint, no LoRA, LCM, ten steps, CFG 1 and shift 5. Comfy CFG
1 implements upstream's conditional-only path; CFG 0 would select the wrong
prediction. Stock MODEL cache/patch nodes cannot connect to DGXM_MODEL.

<a id="model-hunyuanvideo-1-5-sr"></a>

### HunyuanVideo 1.5 (+SR)

Detail status (`HunyuanVideo 1.5 (+SR)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-hunyuan-video.json). Use the template settings and validated artifacts.

**Tested configurations.** Video 1.5 and 720p/1080p SR rank-identity checks, reference comparisons and completed renders; a separate FP16 Video decoded 121-frame probe read NRMS 0.049519 against its local reference, inside the 0.10 limit. A pure-Ulysses one-step render of the FP16 720p Video 1.5 file at 1280x720, 121 frames and 77 text rows produced identical tensors to one GPU; SR was not part of that reading.

**Known limitations.** cfg2 is plain-text only; split-custom SR needs explicit topology. FP16 Video and FP8 SR FSDP are admitted but unmeasured. Video tests do not establish SR accuracy.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-hunyuanvideo-1-5-sr).

<a id="model-cogvideox-1-5-t2v"></a>

### CogVideoX 1.5 (t2v)

Detail status (`CogVideoX 1.5 (t2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-cogvideox-t2v.json). Use the template settings and validated artifacts.

**Tested configurations.** Ulysses against DP2, and a BF16 48-frame one-step decoded comparison against the local reference.

**Known limitations.** Conditioning is fixed at 226 tokens. The available 1.0-layout LoRA did not apply to 1.5. No other LoRA has a passing comparison here.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-cogvideox-1-5-t2v).

<a id="model-cogvideox-1-5-i2v"></a>

### CogVideoX 1.5 (i2v)

Detail status (`CogVideoX 1.5 (i2v)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-cogvideox-i2v.json). Use the template settings and validated artifacts.

**Tested configurations.** ComfyUI-layout BF16 I2V Ulysses: cold/warm completion and a decoded 45-frame one-step comparison within the stated local-reference limit.

**Known limitations.** Diffusers-layout loading and inpaint are not covered. Use a valid I2V first-frame input and the template's even latent-frame layout; cfg2 and LoRA do not inherit this Ulysses result.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-cogvideox-1-5-i2v).

<a id="model-cogvideox-1-5-inpaint"></a>

### CogVideoX 1.5 (inpaint)

Detail status (`CogVideoX 1.5 (inpaint)`): impl.

**Recommended workflow.** No template yet; the I2V workflow does not qualify inpaint.

**Tested configurations.** No separate checkpoint/workflow comparison.

**Known limitations.** Status stays impl. T2V and I2V results do not qualify inpaint.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-cogvideox-1-5-inpaint).

<a id="model-kandinsky5-video-lite"></a>

### Kandinsky5-Video Lite

Detail status (`Kandinsky5-Video Lite`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-kandinsky5-video-lite.json). Use the template settings and validated artifacts.

**Tested configurations.** The named Ulysses/ring/CFG controls, finite renders and public Lite LoRA hot-swap; the BF16 Ulysses decoded local-reference probe also passed.

**Known limitations.** Use Model Sampling SD3 (DGX Monarch) at shift 5. FSDP was not exercised because the validated resident configurations fit; other LoRA or topology combinations have no passing comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-kandinsky5-video-lite).

<a id="model-kandinsky5-video-pro"></a>

### Kandinsky5-Video Pro

Detail status (`Kandinsky5-Video Pro`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-kandinsky5-video-pro.json). Use the template settings and validated artifacts.

**Tested configurations.** FSDP validated for the 4096-dimension model, including its named finite render and reference comparison.

**Known limitations.** Use SD3 shift 5. LoRA on FSDP is unmeasured. The later resident sample has a missing-reference CHECK and provides no output-accuracy result.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-kandinsky5-video-pro).

<a id="model-minimax-h3-t2va-fl2va-ref2va-av"></a>

### MiniMax H3 (t2va/fl2va/ref2va AV)

Detail status (`MiniMax H3 (t2va/fl2va/ref2va AV)`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-minimax-h3-t2va.json). Use the template settings and validated artifacts.

**Tested configurations.** Ulysses/ring with audio and anchored guides at the sizes listed in the linked results. The FP8 xFuser compatibility test at 512x320x5 also reproduced both latent streams, five decoded frames and audio.

**Known limitations.** Batch one; packed DP/cfg2 and world-2 single refuse. Padded ring and Sol-Attn need explicit accuracy waivers. BF16 FSDP completed a render but has no reference comparison. Stacked PDD heads and other guide settings have no passing reference comparison.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-minimax-h3-t2va-fl2va-ref2va-av).

**Workflow guidance.**

Use Ulysses for the packed batch-one sequence; for a one-GPU reference, see
[How a family qualifies](#how-a-family-qualifies).
CFG above 1 works through two sequential conditioning calls, not cfg2.
For unpadded ring, use only the tested sizes listed in the results; accepting a padding waiver does not establish output
accuracy. Sol-Attn is off by default and always requires an accuracy waiver;
its actual CuTe/Triton backend matters to any performance comparison. Install
`sol-attn` and `apache-tvm-ffi` on every worker; CuTe additionally requires
`nvidia-cutlass-dsl`. Without it, the package can fall back to Triton, so
verify the backend reported in the worker log.

Use the pruned Ref2VA artifacts when full BF16 plus the driver encoders does
not fit. Declare still-versus-clip guide shape correctly and respect guide
preflight: clip encoding has a separate large memory peak. Unlisted/chained
guides, large clips and guide decode quality have no passing reference comparison. Fun ControlNet
takes a stock MODEL and cannot be wired into this distributed graph.

## Model-adoption quant artifact catalog

This catalog names checkpoint files an operator can load through the DGX
loader. It changes no support status: each claim covers only its own
checkpoint, workflow, topology and dated validation record. Generated files are
local conversions, not upstream downloads.

The catalog format vocabulary is BF16, FP16, uniform plain FP8 E4M3 and E5M2,
scaled FP8 E4M3, FP8 mixed, INT8, official INT8 ConvRot, mixed INT8 ConvRot,
MXFP8, NVFP4, and mixed NVFP4. Use the filename that matches the loaded DiT;
do not treat a quantized text encoder, VAE, LoRA, or one model variant as a
successful test of another component or variant.

Each DiT format below has a dated one-step Ulysses comparison for Mage Flow,
Qwen Image 2.1 and Wan Animate 2 Base in VALIDATION.md. Those comparisons do not cover other
encoders, VAEs, graphs, topologies, LoRAs, cache settings or FSDP
combinations.

### Pinned source artifacts

Use these upstream revisions and download checksums to identify the original
files used for the comparisons. They are not hashes of private test outputs.

<details>
<summary>Source revisions and SHA-256 checksums</summary>


| Family and canonical source prefix | Pinned upstream source | Variant distinction |
|---|---|---|
| Mage Flow RL (`mage_flow_*`) | `Comfy-Org/Mage-Flow@6ff68fbf3c667325e961d551691b246791d5eec0`; `mage_flow_bf16.safetensors`, SHA-256 `04431abbc3acd1a5b86a7f77269f8e77e05a6cac3b21f35e1aef7493f6ab4934` | Upstream publishes distinct `mage_flow`, `mage_flow_turbo`, `mage_flow_edit`, and `mage_flow_edit_turbo` BF16/ConvRot artifacts. A locally generated `mage_flow_*` file below is derived from the listed primary source only. |
| Qwen Image 2.1 (`qwen_image_2.1_*`) | `Comfy-Org/Qwen-Image-2.1@cb504a4090723e43f17ad01cec0359490e2de613`; `qwen_image_2.1_bf16.safetensors`, SHA-256 `89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc` | The DiT, Qwen3-VL encoder, and VAE are separate artifacts. This catalog concerns the DiT; its official DiT ConvRot file is `qwen_image_2.1_int8_convrot.safetensors`. |
| Wan Animate 2 Base (`wan_animate_2_*`) | `Comfy-Org/Wan-Animate-2@e7181fe1896b7f2ce34120d1a3663413548240d7`; `wan_animate_2_bf16.safetensors`, SHA-256 `642e906c7deaa27072a90ea433319b18f274da438ba75e350373467a0421f177` | Base and Distilled are separate sources. This catalog names Base artifacts only; do not substitute a Base conversion for `wan_animate_2_distill_*`. The official Base ConvRot file is `wan_animate_2_int8_convrot.safetensors`. |

</details>

### Local artifact names and native encodings

BF16 means the pinned source file. FP16 local baselines use the `_fp16`
suffix. "Uniform plain" is a raw floating-point conversion of every floating
tensor, including norms, biases, and heads; it is distinct from the older
core-only diagnostic files. Scaled and mixed FP8, INT8, ConvRot, MXFP8, and
NVFP4 artifacts use the stock-Comfy quantized-tensor encoding for their
format, with its descriptor and scale side tensors. The official
`*_int8_convrot.safetensors` files are upstream artifacts, not local
conversions.

| Format | Mage Flow RL local name | Qwen Image 2.1 local name | Wan Animate 2 Base local name |
|---|---|---|---|
| FP16 | `mage_flow_fp16.safetensors` | `qwen_image_2.1_fp16.safetensors` | `wan_animate_2_fp16.safetensors` |
| uniform plain FP8 E4M3 | `mage_flow_uniform_fp8_e4m3.safetensors` | `qwen_image_2.1_uniform_fp8_e4m3.safetensors` | `wan_animate_2_uniform_fp8_e4m3.safetensors` |
| uniform plain FP8 E5M2 | `mage_flow_uniform_fp8_e5m2.safetensors` | `qwen_image_2.1_uniform_fp8_e5m2.safetensors` | `wan_animate_2_uniform_fp8_e5m2.safetensors` |
| scaled FP8 E4M3 | `mage_flow_fp8_e4m3_scaled.safetensors` | `qwen_image_2.1_fp8_e4m3_scaled.safetensors` | `wan_animate_2_fp8_e4m3_scaled.safetensors` |
| FP8 mixed | `mage_flow_fp8mixed.safetensors` | `qwen_image_2.1_fp8mixed.safetensors` | `wan_animate_2_fp8mixed.safetensors` |
| INT8 | `mage_flow_int8.safetensors` | `qwen_image_2.1_int8.safetensors` | `wan_animate_2_int8.safetensors` |
| official INT8 ConvRot | `mage_flow_int8_convrot.safetensors` | `qwen_image_2.1_int8_convrot.safetensors` | `wan_animate_2_int8_convrot.safetensors` |
| mixed INT8 ConvRot | `mage_flow_int8_convrot_mixed.safetensors` | `qwen_image_2.1_int8_convrot_mixed.safetensors` | `wan_animate_2_int8_convrot_mixed.safetensors` |
| MXFP8 | `mage_flow_mxfp8_native_views.safetensors` | `qwen_image_2.1_mxfp8_native_views.safetensors` | `wan_animate_2_mxfp8_batch_general_native_views.safetensors` |
| NVFP4 | `mage_flow_nvfp4_native_views.safetensors` | `qwen_image_2.1_nvfp4_native_views.safetensors` | `wan_animate_2_nvfp4_batch_general_native_views.safetensors` |
| mixed NVFP4 | `mage_flow_nvfp4_mixed.safetensors` | `qwen_image_2.1_nvfp4_mixed.safetensors` | `wan_animate_2_nvfp4_mixed.safetensors` |

### Conversion recipe boundaries

The recorded local CTQ conversions use CTQ 1.3.4 with `--simple`, manual seed
`20261003`, `--low-memory`, `--comfy_quant`, and saved quantization metadata.
`--simple` disables learned rounding; it still uses CTQ's simulated calibration
and bias-correction paths. Reproduction also requires the same source file, converter build and
preserve/exclude list. These settings alone do not guarantee an identical
converted checkpoint.

The profile names describe the recorded conversion semantics: scaled FP8 is
E4M3 with native Comfy scale-bearing tensors; FP8 mixed preserves additional
first- and last-block sensitive weights in all three families; and INT8 row
quantizes eligible weights with per-row scales. The recorded mixed-NVFP4
recipes use E4M3 for attention projections, NVFP4 for feed-forward matrices,
and BF16 for profile-excluded residual weights. Wan's mixed-NVFP4 artifact uses
the original Base inventory: 400 E4M3, 80 NVFP4, and 296 BF16 retained tensors;
do not derive that profile from the v2 native-view inventory.

The native-view MXFP8/NVFP4 boundaries are family-specific:

* Mage Flow: 145 native-view weights are quantized; 107 tensors remain BF16,
  including `img_in` and all attention output views.
* Qwen Image 2.1: 195 native-view weights are quantized; `img_in.weight` is
  the additional preserved linear (there is no `img_in` bias exception).
* Wan Animate 2 Base: 321 native-view weights are quantized and 615 tensors
  remain BF16; the preserved set includes four cross-attention K/V weights and
  their biases in each of 40 blocks, plus `img_emb.proj.1` and
  `img_emb.proj.3` weights with their existing biases.

Mixed ConvRot quantizes the eligible groups selected by its family profile;
it does not inherit the mixed-FP8 first/last preservation rule. Mage records
169 ConvRot linears plus `img_in.weight` as a plain row-wise INT8 exception.
Qwen Image 2.1 records 195 ConvRot linears plus the K=64 `img_in.weight` plain
INT8 exception. Wan Animate 2 Base records 483 ConvRot linears and no plain
exceptions.

## Shared topology and memory guidance

### How a family qualifies

A supported checkpoint must load through stock ComfyUI and match the adapter
that owns its architecture. Ulysses degree must divide its attention heads;
ring has no head-count requirement but can have model-specific padding or
mask restrictions. CFG splitting depends on ComfyUI's conditioning batching:
matching token constants, masks and prompt shapes still matter. The mode
symbols describe implementation; HW refers only to the listed hardware tests.

Use `auto` for the supplied workflow's family/resolution/quantization policy.
Choose an explicit topology only within that model's constraints. A true
one-GPU reference uses `mode=local`, `gpus_per_host=1`; `single` at world size
two can derive data parallelism and is refused by packed-latent families.

### FSDP, precision and residency

FSDP is a capacity preset, not a speed setting, and `auto` never selects it.
BF16/FP16 denoising cores, supported small FP32 islands and stored-byte
per-tensor FP8/INT8 layouts have distinct admission rules. MXFP8, NVFP4,
unproven pre-cast storage and unsupported live casts refuse.
A checkpoint whose `custom_operations` the pack does not
recognize reads as unknown precision and refuses; its parameter dtypes never
pass it as BF16. When ComfyUI would load a proven all-BF16 file as FP16,
select the UNET loader's explicit `bf16` weight dtype. A base FSDP admission
does not qualify a LoRA: LoRA requires low-RSS mode and refuses Comfy-kitchen
quantized shards or live/file dtype mismatches. Ring+FSDP refuses. See
[ADAPTERS.md](ADAPTERS.md) for the full contract.

A no-LoRA first-use FSDP proof compares independently loaded fresh shards on
every rank. An incomplete reload never returns PASS; failed cleanup requires
mesh recycle. The extra proof load can refuse for capacity even when a render
would fit ([docs/TROUBLESHOOTING.md #88](TROUBLESHOOTING.md#88-the-first-render-of-an-fsdp-topology-refuses-with-a-capacity-figure-and-no-proof)). Disabling the gate does not verify output accuracy or permit
otherwise-blocked memory optimizations. Preserve the driver and
worker admission guards.

The default `slab_weights=auto` uses shared backing only after the first-use
check has passed for that family on unified memory. Until then, loads use stock
residency. Low-RSS
LoRA baking reclaims original-weight backups using verified checkpoint
references; otherwise a broad LoRA can add another model-sized host copy.
Use live memory telemetry rather than checkpoint size alone. A forced family
does not inherit those first-use results ([Running finetunes](#running-finetunes)).
Runtime Gate authorization and model fidelity are separate, as the status
legend explains.

The shared NVFP4 activation scale follows the conditioning calls stock
ComfyUI makes, and it does not lift a family's accuracy refusal
([docs/TROUBLESHOOTING.md #95](TROUBLESHOOTING.md#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales)). Set `shared_act_scale` the same on every
worker, because it decides whether a rank issues a collective;
`log_act_scale` only logs. The [quant catalog](#model-adoption-quant-artifact-catalog)
identifies the separately measured new-model encodings.

### Video workflow boundaries

Wan high/low stages use separate sampler histories and replace the checkpoint
in one resident slot. Only the listed variants bind: ordinary Wan 2.1/2.2,
Bernini-R, FlowRVS, SCAIL/SCAIL2, WanDancer and native Animate2. Other Wan
subclasses refuse; Animate2 needs its metadata discriminator and native
conditioning, not generic I2V wiring.

Kandinsky5 needs Model Sampling SD3 (DGX Monarch): shift 3 for Image, and
shift 5 for Video Lite and Pro, where 5 is not the checkpoint default. Keep
the workflows' native 16-channel image and video latent layouts.

The KSampler Pipeline can overlap control work for video seed sweeps when
`pipeline_depth` exceeds one. Image seed sweeps usually use Empty Latent batch
size instead. Native latent RDMA is off by default and remains
**NOT RUN / HOLD**; actor messaging is the supported return path.

### Architectures with no shard path

| Model | Why | Fallback |
|---|---|---|
| HiDream-O1 | Stateful causal-prefix attention and cross-step KV cache | Stock ComfyUI on one GPU; capacity options belong to that path |
| Wan CausalAR | Autoregressive frame blocks and growing causal KV caches | Stock ComfyUI on one GPU |

## Custom sampler ecosystems (RES4LYF and others)

Both sampler routes work with third-party samplers, provided the pack
imports on every worker host:

* **Name path** (`DGXMonarchKSampler` / `DGXMonarchKSamplerAdvanced`
  `sampler_name` dropdown): the pack registers its names with
  `comfy.samplers.KSampler.SAMPLERS` at import time, and each worker preloads
  the pack too. RES4LYF `res_2s` runs two forwards per step through this path
  (measured on the pair).
* **Object path** (the SAMPLER input of `DGXMonarchSamplerCustom`, such as
  RES4LYF ClownSampler): the SAMPLER object is pickled by reference to its
  class, so the pack must import on each worker to unpickle it.

Every worker host needs the pack under `custom_nodes/` and its Python
requirements installed in the worker environment; a pack that fails to import
there produces a typed error rather than a silent euler render
([docs/TROUBLESHOOTING.md #11](TROUBLESHOOTING.md#11-custom-sampler-renders-as-euler--sampler--is-not-registered-on-this-worker)).

SDE-noise custom samplers (RES4LYF eta > 0): keep an explicit noise seed
wired (stock RandomNoise or the sampler's own seed widget). Seed -1 derives
from ambient torch RNG state, which the cross-rank identity gate flags if the
ranks diverge.

## Running finetunes

ComfyUI may load a supported family's finetune while dgx-monarch refuses it
or selects the wrong topology. The driver identifies the family from the
checkpoint header; the worker identifies it from the loaded model.

Renamed tensor keys or unusual prefixes can prevent detection or select the
wrong family. The Init node's `family_adapter` override selects the family
for both header detection and the loaded-model adapter, including its topology
and capacity rules. Use the actual child family for a finetune: for example,
Chroma requires `chroma`, not its parent `flux`.

The adapter must match the module tree built by ComfyUI. A mismatch refuses
before a kernel runs and names both classes. The override has its own
first-use context, so a forced run cannot reuse an `auto` result for the
same file. Leave the override at `auto` for normal detection.

What it cannot do:

* It cannot make ComfyUI load a checkpoint ComfyUI cannot detect. That failure
  happens inside the loader, before this project sees a model.
* It cannot correct a misread quantization. The header sniff also reports
  precision, and a nonstandard repack still misreads it and still picks the
  wrong table row. Use an explicit topology preset for that.
* It cannot supply an LTX header config. Without it ComfyUI's defaults build
  a structurally different model that still accepts most of the weights (see
  the LTX section above), so a forced `ltx` family on a checkpoint whose
  header lost its `config` metadata is refused rather than run: the adapter
  would be right and the architecture wrong. Re-export with the metadata kept.
* It does not verify the new checkpoint's output accuracy. A forced render logs the family
  it was given beside the family the header read, and the status in the support
  matrix belongs to the artifacts that were measured, not to a claim.

A family-mismatch refusal releases its sample lease, so correcting the
family and re-queuing needs no mesh reset. Tests using the detected family as
an explicit override produced the same output as `auto`.

A family override does not inherit saved family-level first-use results.
Selecting a name does not run a comparison, so `slab_weights=auto` uses stock
residency under an override and saves no family result. Zero-copy residency stays
available: set `slab_weights=on`, and that render gates as its own
combination with the cross-residency leg that proves slab against stock.

## Out of scope for the cluster path

The cluster path does not support SD, SDXL, the HunyuanVideo back-catalogue
or GGUF checkpoints. Generic stale-math accelerators such as TeaCache are
outside the exact-math policy. The named Qwen Image 2.1 worker cache is a
separate model feature. Its tests compare the same cache settings on native
and distributed paths; they do not compare compressed caches with cache off.
Chroma control tensors shard when present, but no cluster ControlNet loader
is provided.


This table is maintained from the linked [test results](VALIDATION.md).
Passing results apply only to the configurations listed in each entry.
