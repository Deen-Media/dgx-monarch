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
| [Krea2 RAW and Turbo](#model-krea2-raw-turbo) | Text-to-image | Auto picks cfg2 for FP8 and BF16 from 1.2 to 2.36 MP, including the template's CFG++ sampler at CFG 1, and Ulysses elsewhere. Reference images require `single`. |
| [Kroma](#model-kroma) | BF16, INT8 and two-pass 2x upscale | Uses the Krea2 loader and native shift. Start with 8 steps, CFG 1, Euler/simple. BF16, local INT8 and BF16 refinement matched their own stock references. See the [tested configurations](VALIDATION.md#evidence-kroma). |
| [FLUX.3 Klein](#model-flux-3-klein) | Step-distilled and base text-to-image and reference editing | Needs ComfyUI 0.41.0 or newer. Start with the BF16 workflows: the INT8 files pin an approximate attention kernel. BF16 and NVFP4 Ulysses text to image up to 1536 square matched one GPU exactly. Since the fused-norm parity change, edits and text to image at 704 square or below on uly2 are bit-identical to one GPU too, confirmed on hardware on 2026-10-16. See the [tested configurations](VALIDATION.md#evidence-flux-3-klein). |
| [MOSS-Image SFT and Base](#model-moss-image-sft) | Text-to-image, and a JSON layout prompt on SFT | Needs ComfyUI 0.42.0 or newer. Start with the BF16 workflows: the INT8 files pin an approximate attention kernel. Keep the template's 50 Euler steps at CFG 3 on the `normal` scheduler and its Skip Layer Guidance (DGX Monarch) node, which skips layer 22 on the negative pass; cfg2 refuses while that skip is on. BF16 Ulysses matched one GPU exactly, up to 2048 square on SFT; see the [tested configurations](VALIDATION.md#evidence-moss-image-sft). |
| [Kromoss](#model-kromoss) | Text-to-image at 50 steps, and at 8 steps with a DMD LoRA | A MOSS-Image Base fine-tune on the same files and adapter. Pin the manifest's revision: the vendor replaces the files under the same names. Use the DMD LoRAs with the BF16 file: a LoRA on the INT8 file refuses. BF16 Ulysses at 1024 and 1536 square matched one GPU exactly; see the [tested configurations](VALIDATION.md#evidence-kromoss). |
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

Chroma (Radiance included) and LongCat have cfg2 rows below 1.2 MP, and
Chroma has a second one from 1.2 to 1.64 MP. Krea 2 has a cfg2 row from 1.2
to 2.36 MP and Qwen-Image one from 1.2 to 1.77 MP. Anima and Boogu have cfg2
rows below 1.2 MP. Each family's Ulysses rows cover the rest. Auto takes a
cfg2 row at any CFG other than 1, and at CFG 1 under a CFG++ sampler chosen
on a DGX Monarch KSampler, KSampler Advanced or KSampler Pipeline node, when
its evidence covers the checkpoint's precision; otherwise the Ulysses row
applies. A Sampler Custom graph hands over a SAMPLER object with no name, so
at CFG 1 auto keeps Ulysses there. Mage Flow, Qwen Image 2.1, FLUX.3 Klein and MOSS-Image have no row, and Flux, Flux 2, Krea 2,
Lens, Qwen-Image and Z-Image have no row below 1.2 MP; those cases take the
generic uly2 fallback (row 99). Every other family has one full-range Ulysses
or ring row.

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
| Krea2 (RAW+Turbo) | ✓ | ★ | ★ | 🔒 | ✓ | **HW** | Auto cfg2 covers FP8 and BF16 from 1.2 to 2.36 MP; padded ring needs a waiver; reference-image graphs use single. [Details](#model-krea2-raw-turbo) · [Workflow](../example_workflows/dgx-monarch-krea2-t2i.json) |
| Kroma | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | BF16 and local INT8, pure Ulysses; BF16 2x refinement. Manual `uly2+fsdp`, batch 1; `dp2+fsdp` batch 2 **CHECK**. LoRA untested. cfg2 is measured, not the same math: on the shipped zeroed negative it read one-step NRMS 0.009 to 0.011 against one GPU at 1152 and 1536 and full renders diverged (2026-10-16), so `auto` keeps Ulysses. Padded ring needs a waiver. [Details](#model-kroma) · [Workflow](../example_workflows/dgx-monarch-kroma-t2i.json) |
| Chroma | ✓ | ★ | ★ | ✓ | ✓ | **HW** | NVFP4 is waiver-only; Ulysses rejects effective masks. [Details](#model-chroma) · [Workflow](../example_workflows/dgx-monarch-chroma-t2i.json) |
| Radiance | ✓ | ★ | ★ | ✓ | ✓ | **HW** | cfg2 requires equal-length prompts. [Details](#model-radiance) · [Workflow](../example_workflows/dgx-monarch-radiance-t2i.json) |
| Ideogram4 | ✓ | ✓ | ★ | 🔒 | ✓ | **HW** | Explicit cfg2 splits the two checkpoints; sharded attention uses Torch Flash. [Details](#model-ideogram4) · [Workflow](../example_workflows/dgx-monarch-ideogram4-t2i.json) |
| Flux 1 Dev | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Padded ring is guarded; distilled checkpoints use CFG 1. Use converted LoRAs. [Details](#model-flux-1-dev) · [Workflow](../example_workflows/dgx-monarch-flux1-t2i.json) |
| Flux 1 Schnell | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | CFG 1; Schnell-specific padded inputs remain unmeasured. [Details](#model-flux-1-schnell) · [Workflow](../example_workflows/dgx-monarch-flux1-t2i.json) |
| Flux2 | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | FP8-mixed Ulysses and BF16 `uly2+fsdp` were compared with references. LoRA/FSDP rendered, but its reference ran out of memory. [Details](#model-flux2) · [Workflow](../example_workflows/dgx-monarch-flux2-t2i.json) |
| FLUX.3 Klein | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | Step-distilled; CFG 1, euler, 16 steps or fewer. BF16 and NVFP4 uly2 text to image up to 1536 square is bit-identical to one GPU, and at one step to stock. ring2 is measured, one-step NRMS 0.026 to 0.183 against one GPU, so use uly2; padded ring needs a waiver. INT8 files pin an approximate attention kernel: single, cfg2 and dp2 ask for the Sol-Attn waiver, and uly2 drops the pin. Plain Euler at CFG 1 leaves cfg2 nothing to split. NVFP4 refuses FSDP. Before the fused-norm parity change, uly2 edits and uly2 text to image at 704 square or below differed from one GPU, up to one-step NRMS 0.117 for the NVFP4 edit; after it they are bit-identical in BF16 and NVFP4, confirmed on hardware on 2026-10-16. Edits scale each reference to 1 MP. Public LoRA: none; mechanism tested with a synthetic LoRA. [Details](#model-flux-3-klein) · [Workflow](../example_workflows/dgx-monarch-flux3-klein-t2i.json) |
| FLUX.3 Klein Base | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | CFG 3.5, 25 steps, euler. cfg2 is measured, not the same math: bit-identical when the prompt encodes past one 80-row block; on a shorter prompt it stays within the one-step floor but full renders diverge, and auto never picks it. ring2 is measured, one-step NRMS 0.011 to 0.057, and padded ring needs a waiver. Edits use uly2 or single in BF16 and NVFP4: the NVFP4 edit read one-step NRMS 0.019 on uly2 before the fused-norm parity change, and uly2 edits are bit-identical to one GPU after it, confirmed on hardware on 2026-10-16. Same uly2, INT8, FSDP and LoRA notes as FLUX.3 Klein. [Details](#model-flux-3-klein-base) · [Workflow](../example_workflows/dgx-monarch-flux3-klein-base-t2i.json) |
| MOSS-Image SFT | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | CFG 3, 50 steps, euler on the `normal` scheduler, with Skip Layer Guidance (DGX Monarch) skipping layer 22 on the negative pass. BF16 uly2 is bit-identical to one GPU from 1024 to 2048 square, one step and full schedule, and at one step to stock from 512 to 2048 square. INT8 files pin an approximate attention kernel: single, cfg2 and dp2 ask for the Sol-Attn waiver, and uly2 drops the pin; since the INT8 dispatch fix, confirmed on hardware on 2026-10-21, INT8 uly2 is bit-identical to a pin-stripped one-GPU render, where it read one-step NRMS 0.027 to 0.036 earlier that day. cfg2 refuses while the skip is on; with it off, a prompt and a negative of different lengths split on cfg2 bit-identical to one GPU, and an equal-length pair is measured, one-step NRMS 0.023 to 0.026, so auto never picks cfg2. ring2 is measured, one-step NRMS 0.0025 to 0.031 without a LoRA, so use uly2; padded ring needs a waiver, and ring2 with a LoRA is not supported. `uly2+fsdp` is bit-identical to resident uly2. NVFP4 and an fp8 `weight_dtype` cast refuse on uly2, ring and cfg2. A LoRA on the INT8 files refuses class P on every topology ([TROUBLESHOOTING #129](TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)); BF16 with a LoRA renders. Reference-image edits for edit LoRAs run on one GPU only and refuse class P on every cluster topology ([TROUBLESHOOTING #130](TROUBLESHOOTING.md#130-a-moss-image-render-with-reference-images-refuses-on-every-topology-but-one-gpu)). An edit with `image_strength` other than 1.0 refuses class P under sequence parallelism, at the driver before dispatch, as a hardware run observed on uly2 and ring2 on 2026-10-27; on single, cfg2 and dp2 row 0 it was bit-identical to stock on one GPU ([TROUBLESHOOTING #135](TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)). The driver preflight does not price the 19.31 GB BF16 encoder. [Details](#model-moss-image-sft) · [Workflow](../example_workflows/dgx-monarch-moss-image-t2i.json) |
| MOSS-Image Base | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | CFG 3, 50 steps, euler on the `normal` scheduler, with the same layer-22 skip. BF16 uly2 is bit-identical to one GPU at 1024 and 1536 square, one step and full schedule; its FSDP reading comes from the scoped sweep. Same cfg2, INT8, quantization, ring, LoRA, edit and driver notes as MOSS-Image SFT. An edit with `image_strength` other than 1.0 refuses class P under sequence parallelism, at the driver before dispatch; on 2026-10-27 a hardware run observed that refusal on uly2 and ring2 with MOSS-Image SFT, whose edits on single, cfg2 and dp2 row 0 were bit-identical to stock on one GPU ([TROUBLESHOOTING #135](TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)). [Details](#model-moss-image-base) · [Workflow](../example_workflows/dgx-monarch-moss-image-base-t2i.json) |
| Kromoss | ✓ | 🔒 | ★ | 🔒 | ✓ | **HW** | MOSS-Image Base fine-tune; no header field tells it from Base or SFT. CFG 3, 50 steps, euler on the `normal` scheduler, with the layer-22 skip, or 8 Euler steps on `simple` at CFG 1 with a DMD LoRA, where cfg2 has nothing to split and refuses. BF16 uly2 is bit-identical to one GPU at 1024 and 1536 square, one step and full schedule. The DMD LoRAs bake to stock ComfyUI's merge on every key and render the same on uly2 and one GPU; against stock they read one-step NRMS 0.0365 to 0.0635 at strengths 0.8 and 1.0, because stock applies them at cast time. Same cfg2, INT8, quantization, ring, LoRA, edit and driver notes as MOSS-Image SFT. An edit with `image_strength` other than 1.0 refuses class P under sequence parallelism, at the driver before dispatch; on 2026-10-27 a hardware run observed that refusal on uly2 and ring2 with MOSS-Image SFT, whose edits on single, cfg2 and dp2 row 0 were bit-identical to stock on one GPU ([TROUBLESHOOTING #135](TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)). [Details](#model-kromoss) · [Workflow](../example_workflows/dgx-monarch-kromoss-t2i.json) |
| LongCat-Image | ✓ | ★ | ★ | ✓ | ✓ | **HW** | Padded image streams use Ulysses; ring/hybrid padding is guarded. [Details](#model-longcat-image) · [Workflow](../example_workflows/dgx-monarch-longcat-t2i.json) |
| HunyuanImage 2.1 (+refiner) | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Extended-text cfg2 needs equal Qwen lengths; refiner FSDP measured at FP8 only; SR FSDP unmeasured. [Details](#model-hunyuanimage-2-1-refiner) · [Workflow](../example_workflows/dgx-monarch-hunyuan-image.json) |
| Qwen-Image | ✓ | ★ | ★ | ✓ | ✓ | **HW** | Auto cfg2 covers base 2512 FP8 from 1.2 to 1.77 MP. Edit reference-method limits differ from base; sharded Edit can refuse. [Details](#model-qwen-image) · [Workflow](../example_workflows/dgx-monarch-qwen-image-t2i.json) |
| Mage-Flow (T2I/Edit, quality + Turbo) | ✓ | ✓ | ★ | 🔒 | ✓ | **HW** | T2I/Edit/Turbo Ulysses tested; FSDP tested only on Base BF16 without LoRA. [Details](#model-mage-flow-t2i-edit-quality-turbo) · [Workflow](../example_workflows/dgx-monarch-mage-flow-t2i.json) |
| Qwen Image 2.1 | ✓ | 🔒 | ★ | 🔒 | 🔒 | **HW** | Torch Flash only; Compressed caches were compared with the same cache settings in native ComfyUI, not with cache off. On ComfyUI 503d785f0 or later, cache device `auto` under dynamic VRAM refuses on Ulysses; use `gpu` or `cpu` ([TROUBLESHOOTING #134](TROUBLESHOOTING.md#134-a-qwen-image-21-ulysses-render-with-the-cache-device-auto-refuses-on-comfyui-503d785f0-or-later)). [Details](#model-qwen-image-2-1) · [Workflow](../example_workflows/dgx-monarch-qwen-image21-t2i.json) |
| Ernie-Image | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | cfg2 comparisons used equal-shaped prompts; unequal-prompt cfg2 has no reference comparison. [Details](#model-ernie-image) · [Workflow](../example_workflows/dgx-monarch-ernie-t2i.json) |
| Z-Image latent | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | cfg2 requires equal-token prompts; Ulysses degree must divide 30 heads. [Details](#model-z-image-latent) · [Workflow](../example_workflows/dgx-monarch-zimage-t2i.json) |
| Z-Image DCT PixelSpace | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | L2P is unsupported; DCT and latent checkpoints need different workflows. [Details](#model-z-image-dct-pixelspace) · [Workflow](../example_workflows/dgx-monarch-zimage-dct-t2i.json) |
| Lens | ✓ | 🔒 | ★ | ✓ | ✓ | **HW** | Asymmetric cfg2 is measured; padded ring remains guarded. [Details](#model-lens) · [Workflow](../example_workflows/dgx-monarch-lens-t2i.json) |
| Omnigen2 | ✓ | 🔒 | ✗ | ★ | ✓ | **HW** | Ulysses is unsupported; ring+FSDP refuses. [Details](#model-omnigen2) · [Workflow](../example_workflows/dgx-monarch-omnigen2-t2i.json) |
| Anima | ✓ | ★ | ★ | ✓ | 🔒 | **HW** | Use the supplied graph for cross-attention and masks. Auto cfg2 covers BF16 below 1.2 MP on the shipped unequal prompt pair. [Details](#model-anima) · [Workflow](../example_workflows/dgx-monarch-anima-t2i.json) |
| Boogu | ✓ | ★ | ★ | ✓ | ✓ | **HW** | Auto cfg2 covers FP8 below 1.2 MP. Unequal prompts use per-conditioning cfg2 dispatch. [Details](#model-boogu) · [Workflow](../example_workflows/dgx-monarch-boogu-t2i.json) |
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

**Tested configurations.** RAW/Turbo resident Ulysses, cfg2, and named BF16/FP8/INT8 paths, including blank-negative cfg2 and the FSDP comparisons linked below. Pure-Ulysses one-step renders on Torch Flash produced identical tensors to one GPU: RAW BF16 at 1536 and 1448 with odd and even text and image streams, at batch 2 and on the template, and the template on RAW INT8 ConvRot and Turbo FP8-scaled. Campaign K (2026-10-14) timed cfg2 on the template against one GPU and uly2: Turbo FP8-scaled at 1152, 1216, 1280, 1408 and 1536 square, plus a CFG 4.5 Euler control at 1536, RAW BF16 and RAW INT8 ConvRot at 1152, 1216, 1280 and 1536, and Turbo BF16 at 1536. Every cfg2 render read bit-identical to one GPU over the full schedule. The same campaign read the SAGE route at 1-step NRMS 0.030 to 0.058 against one GPU at 1152 to 1536. Auto now picks that route for FP8 above 2.36 MP, and from 1.2 MP at CFG 1 under a plain sampler.

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

From 1.2 to 2.36 MP auto selects cfg2 for an FP8 or BF16 checkpoint at any
CFG other than 1, and at CFG 1 under a CFG++ sampler such as the template's
`euler_cfg_pp`, which keeps the unconditional pass, when a DGX Monarch
KSampler node names it. Only FP8-scaled was measured among the FP8 files;
MXFP8 and NVFP4 share its precision class. INT8 renders keep Ulysses: their
cfg2 read bit-identical but no faster than Ulysses below 1536 (0.97x to
1.03x) and 1.08x faster at 1536, so choose cfg2 explicitly only where a size
favours it. Below 1.2 MP no Krea2 row matches, so auto takes the
generic uly2 fallback; above 2.36 MP it selects Ulysses, with SAGE on an FP8
checkpoint, which needs sageattention on each worker. At CFG 1 a plain sampler
drops the unconditional pass, so cfg2 has nothing to split, and auto selects
Ulysses, with SAGE on FP8 from 1.2 MP. A Sampler Custom graph names no
sampler, so auto treats it the same way at CFG 1. On cfg2,
asymmetric prompts are padded automatically; combined Ulysses+cfg has no such
padding and refuses them. A zeroed negative is real conditioning and stays
unmasked.

<a id="model-kroma"></a>

### Kroma

Detail status (`Kroma`): **HW**.

Kroma is a Krea2 fine-tune. The v0.3.1 Turbo OPD release is a complete
checkpoint: it needs no separate Krea2 diffusion checkpoint or Kroma LoRA.
It uses the existing `krea2` adapter, model detection and quantized loader.
The similarly named Chroma uses a different architecture.

**Workflows.** [BF16 text-to-image](../example_workflows/dgx-monarch-kroma-t2i.json),
[INT8 text-to-image](../example_workflows/dgx-monarch-kroma-t2i-int8.json), and
[two-pass 2x upscale](../example_workflows/dgx-monarch-kroma-t2i-upscale.json).

**Required files.** Install the same files under each worker's ComfyUI model
directories and on the driver. The driver reads the diffusion checkpoint's
header and artifact identity even when its weights load on workers.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `kroma-v0.3.1-turbo-opd.safetensors` | `models/diffusion_models` | Official unquantized checkpoint, 26.28 GB; BF16 weights with FP32 components |
| `kromaKrea2_opd.safetensors` | `models/diffusion_models` | Alternative INT8 workflow filename; select your own compatible Kroma conversion |
| `qwen3vl_4b_bf16.safetensors` | `models/text_encoders` | Qwen3-VL encoder; set CLIPLoader type to `krea2` |
| `qwen_image_vae.safetensors` | `models/vae` | Official sample VAE |

Download links, byte sizes and SHA-256 hashes are in
[the artifact manifest](../example_workflows/artifacts.toml). The INT8 row is a
placeholder for a conversion you supply, so it carries no link, size or hash;
[the test record](VALIDATION.md#evidence-kroma) lists the tested file's SHA-256.
Its source remains unverified: a filename does not establish its release or quantizer.
Preserve its embedded quantization metadata. INT8 ConvRot requires ComfyUI
and comfy-kitchen support for `int8_tensorwise` with rotation; changing the
UNET loader's weight dtype does not create that conversion. Leave weight dtype
at `default`. The shipped graph uses only core ComfyUI and DGX Monarch nodes.

**Starting settings.** The [pinned upstream model card](https://huggingface.co/lodestones/Kroma/blob/734ec29814972997fa29a1cd313863a2e5e9921c/README.md)
recommends 8-12 steps, CFG 1.0-1.5 and mu 1.15 for Turbo OPD. The templates
adapt the [official quantized sample graph](https://huggingface.co/lodestones/Kroma/blob/734ec29814972997fa29a1cd313863a2e5e9921c/samples/output_00001.png):
1024×1024, 8 steps, CFG 1, `euler`, `simple`, denoise 1, `EmptyLatentImage`
and `ConditioningZeroOut` for the negative input. Describe the subject, scene,
lighting and composition in the positive prompt. These are upstream starting
settings, not a measured optimum across all prompts or resolutions.

ComfyUI's Krea2 configuration already sets the Flux-style mu to 1.15.
Keep that native sampling configuration; the SD3 sampling-shift node changes
the sampling model and does not express the same parameter. The template
uses plain Euler, so CFG 1 omits the unconditional pass. Substituting a CFG++
sampler changes that behavior and requires reconsidering the negative input.

The upscale graph decodes the first result, scales pixels 2x with Lanczos,
re-encodes with the same VAE, and refines for 4 steps at denoise 0.2 with
Euler/simple and CFG 1. This reproduces the sample's processing stages;
it costs another sampling pass and can change details.

**Parallel modes and validation.** `auto` uses the Krea2 topology policy:
one local device without a cluster, or Ulysses on a configured pair for the
shipped graphs. BF16 and INT8 use Torch Flash. Plain Euler at CFG 1 has no
unconditional pass for cfg2 to split. At other CFG values, or under a CFG++
sampler, `auto` still keeps Ulysses: Krea2's cfg2 row 15 excludes Kroma. The
shipped negative zeroes the positive conditioning, so both have the same shape,
and one GPU evaluates them in one batch-2 call where cfg2 makes two batch-1
calls. On 2026-10-16 at CFG 1.5, 1152 and 1536 square, cfg2 read one-step NRMS
0.009 to 0.011 against one GPU and its full renders diverged. With an empty
text negative it matched one GPU exactly and ran 1.24x to 1.31x faster than
`uly2`, but the shipped graphs do not use that negative. `auto` recognizes
Kroma by the merge record in the BF16 release's header; a Kroma file saved
without that record reads as Krea2 and can take row 15, so select `uly2`
there. Explicit `cfg2` still runs. [The cfg2 comparison](VALIDATION.md#kroma-cfg2)
records the tested settings. Reference-image conditioning requires `single`; padded ring retains
the Krea2 waiver boundary. LoRA remains untested and retains the
[Krea2 restrictions](#model-krea2-raw-turbo). The full 8-step BF16 and local
INT8 graphs at 1024×1024 matched
their own stock one-GPU references exactly under pure Ulysses on two GPUs.
The BF16 upscale graph also matched through its 4-step 2048×2048 refinement
at denoise 0.2. Conditioning, latents and decoded pixels were identical;
[comparisons and limits](VALIDATION.md#evidence-kroma) record the tested
artifacts and settings. The INT8 comparison uses the same conversion on both
sides; it does not measure quantization error against BF16 or establish the
conversion's upstream version. The one-step 1040×1040 padding control and a
one-step Krea2 RAW control also matched stock exactly.

For manual FSDP, use `uly2+fsdp` with batch size 1 for BF16 or local INT8
text-to-image or the BF16 2x-refinement graph. All three full workflows matched
their own stock references exactly, including both stages of refinement.
This qualifies the tested manual FSDP mode; peak memory and speed benefits
were not measured. The shipped `auto` setting and resident policy are unchanged.
`dp2+fsdp` with batch size 2
is **CHECK**: both checkpoints passed the normal FSDP reload proof and returned
the expected batch, but their latents and decoded pixels differed from stock
batch 2. Cross-group rank identity is not applicable to those independent DP
images. See the [FSDP comparisons](VALIDATION.md#kroma-manual-fsdp) for scope,
metrics and batch-shape diagnostics.

The upscale graph is batch 1 only. On the tested ComfyUI revision, its native
`VAEEncode` treats an IMAGE batch as video frames for this VAE: two input images
become one latent batch item after temporal rounding. Both stock and candidate
therefore reach refinement with batch 1, and `dp2+fsdp` correctly refuses it.
Increasing the initial batch widget does not enable independent DP2 refinement.

**Upstream and license.** [Kroma repository](https://huggingface.co/lodestones/Kroma)
and [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE).
The repository metadata names that license; its older README footer says MIT.
The template does not relabel the model weights as MIT.

<a id="model-chroma"></a>

<a id="chroma-flux-family-dit"></a>

### Chroma

Detail status (`Chroma`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-chroma-t2i.json).

**Tested configurations.** BF16 pure Ulysses at 1024 and 1536 and named multi-step controls; cfg2 through the mask-trim path; named FP8 and INT8 controls. Campaign K (2026-10-14) timed cfg2 on BF16, FP8-mixed and INT8 ConvRot at 1152, 1216 and 1280 square, bit-identical to one GPU. FSDP support is separate from these resident comparisons.

**Known limitations.** These comparisons do not cover ring, hybrid, effective masks, other precisions or NVFP4. NVFP4 needs its waiver even though one uneven cfg2 pair reads exact under it.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-chroma).

**Workflow guidance.**

Below 1.64 MP auto selects cfg2 for BF16, FP8 and INT8 checkpoints, at any
CFG other than 1 and at CFG 1 under a CFG++ sampler that a DGX Monarch
KSampler node names. Above 1.64 MP it selects Ulysses, with SAGE on FP8,
which needs sageattention on each worker. The mask and NVFP4
guards still apply.

<a id="model-radiance"></a>

### Radiance

Detail status (`Radiance`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-radiance-t2i.json).

**Tested configurations.** FP32 at 1024x1024 on Ulysses, ring2 and cfg2, including a 30-step render; Ulysses with odd-length text; FSDP against ordinary residency with the BF16 core and two replicated FP32 parameters.

**Known limitations.** Unequal cfg2 prompts refuse before dispatch, because the pad mask has no latent grid in pixel space. Chroma's fixes and controls cover Radiance only on the paths tested on Radiance. Radiance shares Chroma's rows, so auto picks cfg2 for it below 1.64 MP; Radiance cfg2 was compared at 1024 only.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-radiance).

<a id="model-ideogram4"></a>

<a id="ideogram4-pixeldit-head_dim-256"></a>

### Ideogram4

Detail status (`Ideogram4`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-ideogram4-t2i.json).

**Tested configurations.** Ulysses comparisons for the named FP8, BF16, INT8 ConvRot and MXFP8 pairs. Dual-model cfg2 has its own FP8 pair reference comparison and finite render, and explicit cfg2 was tested on the shipped graph for the BF16, INT8 ConvRot, MXFP8 and NVFP4-mixed pairs: one-step NRMS 0.0 against one GPU on Torch Flash, cuDNN and SAGE_AUTO. Other precisions do not inherit these cfg2 results. The NVFP4 limit below applies to the sequence-split presets, which read 0.10 to 0.13 in the same campaign. Pure-Ulysses one-step renders of the FP8-scaled conditional model at CFG 1 produced identical tensors to one GPU at 1024 and 1040 with 283 and 284 text rows, and with `uly2+fsdp` at 1024. A one-step uly2 probe of the shipped dual-model graph at CFG 7 produced identical tensors to one GPU on the BF16, FP8 and INT8 pairs. That graph also rendered on `uly2+fsdp` for the BF16, FP8 and INT8 pairs, one record each, identical to its resident twin.

**Known limitations.** NVFP4 is over the fidelity limit. Ring is not automatic. Under the default `auto_gate=first_use`, FSDP refuses a dual-model request, because the clean-reload Gate proves one model only; `auto_gate=off` runs it under FSDP with slab and low-RSS residency off. `cfg2+fsdp` refuses for this family on every setting. A Ulysses degree that does not divide the 18 attention heads, such as `uly4`, refuses after the load and before any forward binds ([docs/TROUBLESHOOTING.md #137](TROUBLESHOOTING.md#137-an-ideogram4-render-on-a-ulysses-degree-that-does-not-divide-18-heads-refuses-after-the-load-and-before-any-forward-binds)). No claim extends to other artifacts or kernels.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-ideogram4).

**Workflow guidance.**

For dm-cfg2, use the unconditional loader, dual-model guider and custom
sampler; leave the negative input unconnected for the text-free unconditional
pass. Each rank holds one checkpoint; single/DP instead keeps both on each
rank. Leave the resident-adoption input unconnected under dm-cfg2. The shipped
workflow uses `sync_ulysses=off`, which changes nothing on the pinned xfuser
0.7.0: its Ulysses all-to-all never receives the setting
([Adapters](ADAPTERS.md#what-an-adapter-provides)). The gate still proves
each setting apart. On a Ulysses or ring topology the worker
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

<a id="model-flux-3-klein"></a>

### FLUX.3 Klein

Detail status (`FLUX.3 Klein`): **HW**.

FLUX.3 Image Klein is Black Forest Labs' 5.6B-parameter image model with the
Qwen3-VL-4B text encoder and the Flux 2 autoencoder. Two checkpoints share the
`flux3` adapter: this step-distilled one, whose file names carry `_sd_`, and
[FLUX.3 Klein Base](#model-flux-3-klein-base). ComfyUI supports both from
0.41.0; dgx-monarch needs ComfyUI 0.41.0 or newer on the driver and on every
Worker ([INSTALL](INSTALL.md#model-files-required-by-each-template)).

**Recommended workflow.** [Text to image](../example_workflows/dgx-monarch-flux3-klein-t2i.json)
and [image editing](../example_workflows/dgx-monarch-flux3-klein-edit.json):
16 steps, CFG 1, `euler`, `simple`, the settings of the model card and the
vendor's reference code, at 1024×1024 for text to image. The model was
distilled for 4 to 16 steps; keep to that range and to Euler, because it reads
its next sigma off the schedule. 16 divides the 10,000-row sigma table
ComfyUI builds, so the `simple` scheduler lands on the vendor's schedule; the
vendor's own ComfyUI template promotes 15, which does not. Plain Euler at
CFG 1 never evaluates the negative.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `flux3_image_klein_sd_bf16.safetensors` | `models/diffusion_models` | Step-distilled BF16 DiT, 11.10 GB |
| `qwen3vl_4b_bf16.safetensors` | `models/text_encoders` | Qwen3-VL-4B encoder; set CLIPLoader type to `flux3` |
| `flux2-vae.safetensors` | `models/vae` | Flux 2 VAE |

The [artifact manifest](../example_workflows/artifacts.toml) lists the
download links, byte sizes and SHA-256 hashes, including the INT8 ConvRot and
NVFP4 files the testing graphs name. The DiT files come from the gated
`black-forest-labs/flux-3-image-klein-comfy` repository. The edit graphs scale
each reference to about 1 MP with stock nodes before encoding it; the render
takes its size from the first reference. Cite references in the prompt as
`<ref_image_0>`, `<ref_image_1>` in the order they are chained.

**Tested configurations.** On two GPUs with Torch Flash, uly2 one-step
latents were bit-identical to stock ComfyUI on one GPU at 1024 and 1536 square
in BF16 and NVFP4 and on the odd 1040 and 1552 grids in BF16. Full 16-step
renders were bit-identical to one GPU from 1024 to 1536 square, and the
shipped text-to-image graph matched stock exactly. On uly2 the INT8 ConvRot
file equaled stock running a copy with its pins removed. ring2 read one-step
NRMS 0.052 against one GPU on the ladder, under the 0.10 floor but not
identical, and 0.026 to 0.183 in the scoped sweep, over the floor for NVFP4
text to image, so use uly2. In that sweep the NVFP4 edit on uly2 read
one-step NRMS 0.117 against one GPU, over the floor. Those edit readings came
from one kernel switch in comfy-kitchen, which the fused-norm parity change
closes (Known limitations below): BF16 and NVFP4 edits on uly2 and uly2 text
to image at 704 square or below are bit-identical to one GPU after it,
confirmed on hardware on 2026-10-16.
`uly2+fsdp` equaled resident uly2 in BF16 and INT8. A synthetic rank-4 LoRA
hot-swapped from 0.5 to 0.8 without a load and read NRMS 0.0499 at one step
against stock's LoRA loader. CPU tests compare the two-rank sharded forward
with stock ComfyUI's on tiny real models. The
[2026-10-15 and 2026-10-16 record](VALIDATION.md#evidence-flux-3-klein) lists
every comparison; [BENCHMARKS](BENCHMARKS.md#flux3-klein-one-gpu-uly2-and-cfg2-2026-10-15-to-2026-10-16-head-capped)
has the walls.

**Known limitations.** The INT8 ConvRot files pin comfy-kitchen's INT8
attention, an approximate kernel, on 33 blocks: single, cfg2 and dp2 ask for
the Sol-Attn waiver ([TROUBLESHOOTING #117](TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)),
and uly2 drops the pins and runs exact attention, so a uly2 INT8 render
differs from stock INT8 on one GPU by construction. NVFP4 refuses FSDP, and
it is a capacity choice: uly2 NVFP4 equals one-GPU NVFP4, which is not a
fidelity claim against BF16. The diffusers-layout `variants/fp8r` file loads
in stock ComfyUI with its scales dropped and is refused
([TROUBLESHOOTING #120](TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
No public LoRA existed at release; a synthetic rank-4 LoRA tested hot swap
and the Gate. Before the fused-norm parity change, BF16 one-reference edits
on uly2 passed the one-step floor but were not identical to stock, NVFP4 edits
read over it, and uly2 text to image at 704 square or below differed too. The
cause was comfy-kitchen's fused AdaLN, which picks its CUDA kernel by a call's
rows: a rank normalized its piece of a span in a smaller call than stock's
call for the whole span. Each such call now runs at stock's size, so these
renders are bit-identical to one GPU, confirmed on hardware on 2026-10-16 (the
five edit legs matched stock, and text to image at 512 and 704 square matched
one GPU). The change gives each rank stock's full normalization
work, about 2 to 4 percent of a uly2 step by an op-level estimate. The vendor's optional prompt upsampler
is a separate 9.7 GB model on the driver that no shipped graph loads; its
memory is outside the driver estimate. In the 2026-10-15 test the driver
Spark's lowest sampled available memory was 44.6 GiB with it, still
falling when sampling stopped, and 54.9 GiB without it. The step-distilled
model reruns the five reference blocks every step on every rank, so
references cost more per step than on the base model. A grid with an odd row
count pads: uly2 and single render it, and padded ring or hybrid refuses
unless waived. An unscaled reference sets the render size from
the source file with no limit, so keep the scale node. The step-distilled
model reads the sampler's whole sigma schedule each step. Every stock
sampler node hands it over. A custom sampling path that does not fails in
stock ComfyUI with a `KeyError`, as it does on single, cfg2 and dp2; under
sequence parallelism dgx-monarch refuses it before the first block instead.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-flux-3-klein).

**Upstream and license.** [Model card](https://huggingface.co/black-forest-labs/flux-3-image-klein)
and [ComfyUI files](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy),
both gated. The weights are under the FLUX Kommunity License v1.0, which the
repositories carry; read it before commercial use. The Qwen3-VL encoder is
Apache-2.0.

<a id="model-flux-3-klein-base"></a>

### FLUX.3 Klein Base

Detail status (`FLUX.3 Klein Base`): **HW**.

The undistilled checkpoint of [FLUX.3 Klein](#model-flux-3-klein), on the
same adapter, encoder and VAE.

**Recommended workflow.** [Text to image](../example_workflows/dgx-monarch-flux3-klein-base-t2i.json)
and [image editing](../example_workflows/dgx-monarch-flux3-klein-base-edit.json):
25 steps, CFG 3.5, `euler`, `simple`, the vendor's ComfyUI settings, at
1024×1024 for text to image; the model card's default is 50 steps, for more
detail. Both counts divide ComfyUI's sigma table. The negative is the encoding
of an empty prompt, as the vendor's reference code uses; in the edit graph it
carries the same references as the positive.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `flux3_image_klein_bf16.safetensors` | `models/diffusion_models` | Base BF16 DiT, 11.08 GB |
| `qwen3vl_4b_bf16.safetensors` | `models/text_encoders` | Qwen3-VL-4B encoder; set CLIPLoader type to `flux3` |
| `flux2-vae.safetensors` | `models/vae` | Flux 2 VAE |

**Tested configurations.** As [FLUX.3 Klein](#model-flux-3-klein), on the
same dates and kinds of test: uly2 one-step latents were bit-identical to
stock on one GPU, full 25-step renders to one GPU from 1024 to 1536 square on
a one-block prompt and a 355-word one, and the shipped text-to-image graph to
stock. ring2 read one-step NRMS 0.011 on the ladder and 0.015 to 0.057 in the
scoped sweep. The NVFP4 edit on uly2 read one-step NRMS 0.019 against one GPU
in that sweep, within the floor; after the fused-norm parity change, BF16 and
NVFP4 edits on uly2 are bit-identical to one GPU, confirmed on hardware on
2026-10-16. cfg2 is a measured topology: it matched one
GPU exactly on prompts that encode past one 80-row block. On prompts that fit
one it read one-step NRMS 0.009 to 0.031 and every full render diverged,
because one GPU batches the two calls there and cfg2 cannot. `uly2+fsdp`, `cfg2+fsdp` and `dp2+fsdp` equaled their
resident twins, and `mode=local` on one GPU equaled stock. On every Worker the
base model computes its text and reference streams once per conditioning, as
stock ComfyUI does ([VALIDATION](VALIDATION.md#evidence-flux-3-klein-base)).

**Known limitations.** As [FLUX.3 Klein](#model-flux-3-klein), except three
step-distilled points: the base model computes its reference streams once per
conditioning, it does not read the sampler's sigma schedule, and its NVFP4
edits stayed within the one-step floor on uly2 even before the fused-norm
parity change, so they can run on uly2 or single. A prompt that encodes past one 80-row block does not batch with the
empty negative, so ComfyUI runs the two as separate calls and cfg2 gives each
rank one. A shorter prompt batches with the negative on one GPU, so cfg2
differs from it: within the one-step floor, while full renders diverge. Auto
has no cfg2 row for FLUX.3; cfg2 is an explicit, measured choice.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-flux-3-klein-base).

**Upstream and license.** [Model card](https://huggingface.co/black-forest-labs/flux-3-image-klein-base),
gated, under the FLUX Kommunity License v1.0.

<a id="model-moss-image-sft"></a>

### MOSS-Image SFT

Detail status (`MOSS-Image SFT`): **HW**.

MOSS-Image is the OpenMOSS team's 5B-parameter single-stream text-to-image
model, with the Qwen3.5-9B text encoder and the FLUX.2 small decoder as its
VAE. Three checkpoints share the `moss_image` adapter: this supervised
fine-tune, [MOSS-Image Base](#model-moss-image-base) and lodestones'
[Kromoss](#model-kromoss). dgx-monarch needs ComfyUI 0.42.0 or newer on the
driver and on every Worker
([INSTALL](INSTALL.md#model-files-required-by-each-template)).

**Recommended workflow.** [Text to image](../example_workflows/dgx-monarch-moss-image-t2i.json)
and [layout text to image](../example_workflows/dgx-monarch-moss-image-layout-t2i.json):
`euler` and ComfyUI's `normal` scheduler at the vendor's 50 steps, CFG 3 and
shift 5, at 1024×1024. ComfyUI's `normal` schedule lands within 0.0005 of the
vendor pipeline's sigmas at 50 steps, where `simple` sits 0.069 off, and at
the same seed the two render different images of the same prompt. The
negative is the tag list the vendor's guide recommends; the model card's
default is an empty negative. A Skip Layer Guidance (DGX Monarch) node
skips layer 22 (zero-based) on the negative pass at every step, as the
vendor's pipeline does. The vendor reports that layers 24 and 26 can also
help, and that without the skip an image looks more realistic but can lose
fine detail; clear the node's `single_layers` to turn it off. ComfyUI's
MOSS-Image configuration sets the sampling shift to 5, the vendor's value, so
the graph has no shift node. Keep that native shift: Model Sampling SD3
(DGX Monarch) applies multiplier 1000, which renders noise on this model's
multiplier-1.0 configuration, so dgx-monarch refuses it
([TROUBLESHOOTING #128](TROUBLESHOOTING.md#128-model-sampling-sd3-dgx-monarch-refuses-on-a-family-whose-comfyui-sampling-multiplier-is-10)). The vendor recommends 1 to 2 MP with sides
that are multiples of 16, and reports that 0.25 to 4 MP works. English
prompts work best and Chinese is supported. Put text to render in double
quotes; a line break in the prompt sets a line break in the rendered text.
The layout graph's prompt is a JSON layout that reaches CLIP Text Encode
through a link from Primitive String Multiline: typed into CLIP Text Encode's
own box, the browser's dynamic prompt pass would strip its braces. The vendor
marks layout prompts experimental.

**Required files.** Install the same files under each Worker's ComfyUI model
directories and on the driver. The driver reads the diffusion checkpoint's
header and artifact identity even when its weights load on Workers.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `moss_image_sft_bf16.safetensors` | `models/diffusion_models` | SFT BF16 DiT, 10.16 GB |
| `qwen3.5_9b_bf16.safetensors` | `models/text_encoders` | Qwen3.5-9B encoder, 19.31 GB; set CLIPLoader type to `moss_image` |
| `flux2-vae_small.safetensors` | `models/vae` | FLUX.2 small decoder; not the `flux2-vae.safetensors` the Flux 2 and FLUX.3 templates name |

The [artifact manifest](../example_workflows/artifacts.toml) lists the
download links, byte sizes and SHA-256 hashes, including the INT8 ConvRot DiT
files and the INT8 ConvRot encoder `qwen3.5_9b_int8_convrot.safetensors` that
the testing graphs name. The files come from the ungated `Comfy-Org/MOSS-Image`
repository. The encoder runs on the driver and is the largest file the graph
loads. The driver footprint preflight has no row for this family, so it does
not price that encoder. Comfy-Org also publishes the encoder as INT8 ConvRot
(9.75 GB) and W4A8 GPTQ (7.34 GB); Comfy-Org's own ComfyUI templates use the
W4A8 file, which no graph here names and no test here has loaded.

**Tested configurations.** On two GPUs with Torch Flash on 2026-10-21, uly2
one-step latents were bit-identical to stock ComfyUI on one GPU from 512 to
2048 square, on both joint parities where both ran, on the odd 1040 grid, at
batch 2 and with a 6,347-row prompt. Full 50-step renders were bit-identical
to one GPU from 1024 to 2048 square with the skip on, and at 1024 with it off,
and the shipped graphs, the layout graph included, matched stock exactly over
the full schedule. Those renders ran on the `simple` scheduler; on `normal`
the SFT graph's uly2 render again matched one GPU and stock. `uly2+fsdp` at
prefetch depths 1 and 2, `cfg2+fsdp` and `dp2+fsdp` equaled their resident
twins, and `mode=local` equaled stock. On the INT8 ConvRot files uly2 drops
the pins and, after the INT8 dispatch fix, was bit-identical to a copy with
the pins removed, on one GPU and on stock at one step at 1024 and 1536 square,
and on one GPU over the full schedule at 1024; before the fix, on the same
day, it read one-step NRMS 0.027 to 0.036. cfg2 with the skip off split a
prompt and a negative of different lengths bit-identical to one GPU, and an
equal-length pair read one-step NRMS 0.023 to 0.026. ring2 without a LoRA read
one-step NRMS 0.0025 to 0.031 against one GPU, and 0.023 to 0.026 against
stock on an even joint stream; with a LoRA it read 0.0034 to 0.045. The scoped
sweep of the 14 MOSS-Image graphs passed all 252 cells. CPU tests compare the
two-rank sharded forward with stock ComfyUI's on tiny real models, and kernel
tests on one GB10 GPU check comfy-kitchen's fused norm and INT8 linears on
every rank's window. The
[2026-10-21 record](VALIDATION.md#evidence-moss-image-sft) lists every
comparison;
[BENCHMARKS](BENCHMARKS.md#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped)
has the walls.

**Known limitations.** The INT8 ConvRot files pin comfy-kitchen's INT8
attention, an approximate kernel, on all 42 blocks: single, cfg2 and dp2 ask
for the Sol-Attn waiver ([TROUBLESHOOTING #117](TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)), and uly2 drops the
pins and runs exact attention, so a uly2 INT8 render differs from stock INT8
on one GPU by construction and matches a one-GPU render of a copy with the
pins removed. On uly2, ring and cfg2 only BF16 and INT8 ConvRot
run: an NVFP4 file, and an fp8 `weight_dtype` cast on a BF16 file, refuse
class P there and run on single and dp2. The joint stream holds every image
row and every text row, so an odd total pads at world 2: uly2 and single
render it, and padded ring or hybrid refuses unless waived. ring2 on an
unpadded stream is measured, not the same math, so use uly2. ring2 with a
LoRA is not supported: on 2026-10-21 every first-use Gate ceremony of ring2
with a LoRA that reached a verdict failed, six in all, and each quarantined
the low-RSS LoRA residency for its pair until an explicit Gate retest of that
pair passes
([TROUBLESHOOTING #35](TROUBLESHOOTING.md#35-identity-gate-reports-errorinconclusive-after-a-prior-pass)).
The cause has not been investigated. Auto has no cfg2
row for MOSS-Image. The shipped graphs carry the layer-22 skip, and cfg2
refuses while it is on ([TROUBLESHOOTING #126](TROUBLESHOOTING.md#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider)): inside its window
ComfyUI runs the conditional and unconditional passes as two separate
calls, and cfg-parallel can split only a call that carries both. With the
skip off, MOSS-Image publishes its text as a regular condition, so ComfyUI
runs a prompt and a negative of different lengths as two calls and cfg2
gives each rank one, bit-identical to one GPU on hardware; a pair of equal
length batches into one call on one GPU, and cfg2 read one-step NRMS 0.023
to 0.026 against it, so cfg2 earns no automatic row
([DESIGN 5.4](DESIGN.md#54-topology--auto-mode)). On the Kromoss BF16 file,
each Kromoss DMD LoRA baked to stock ComfyUI's merge on every key, and
against stock with the same LoRA a render read one-step NRMS 0.0365 to 0.0635
at strengths 0.8 and 1.0, because stock's default DynamicVRAM loader applies
the LoRA at cast time; SFT and Base with a LoRA rendered the same on uly2 and
one GPU. An INT8 file with a LoRA refuses class P on every topology, before
any weight moves: stock ComfyUI's own classic loader, with DynamicVRAM off,
bakes the LoRA into the INT8 weights and rendered the Kromoss INT8 file with
its DMD2 LoRA about 1.0 NRMS from its default, and dgx-monarch bakes the
same way
([TROUBLESHOOTING #129](TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)).
The slab rung with a LoRA has not run on hardware.
On 2026-10-20 ComfyUI's RoPE scaling options for MOSS-Image
(commit b9b290e43) and its color and depth control (commit e3bee7ffc, the
`MOSSImageControl` node) were on ComfyUI master only, in no release, and no
shipped graph uses either. The control LoRA widens the model's input, so
dgx-monarch refuses it before any bake ([TROUBLESHOOTING #127](TROUBLESHOOTING.md#127-a-lora-that-changes-a-weights-shape-refuses-before-any-bake)).
On 2026-10-22 ComfyUI master added edit LoRAs (commit 60aade0bb): the Text
Encode MOSS-Image Edit node attaches reference images to the conditioning,
and every image and text row attends to rows that come from them. No
hardware test has compared a cluster edit with stock, so a render with
reference images refuses class P on every topology but one GPU; run edits
with `mode=local`, `gpus_per_host=1` and topology `single`, which runs
ComfyUI's own forward ([TROUBLESHOOTING #130](TROUBLESHOOTING.md#130-a-moss-image-render-with-reference-images-refuses-on-every-topology-but-one-gpu)).
On 2026-10-23 ComfyUI added the node's `image_strength` (commit 9fe7ef764,
in v0.43.0): when the text encoder sees the images, a strength other than
1.0 adds a bias to their tokens' attention that the sharded kernel cannot
apply, so such an edit refuses class P under sequence parallelism
([TROUBLESHOOTING #135](TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)). No shipped graph uses the edit node.
On 2026-10-22, on ComfyUI a071826fd, a one-GPU edit in each of the three
edit formats was bit-identical to stock, and with the reference cache it ran
2.119 s a step against stock's 2.142; no cluster edit with reference images
has run on hardware.
The same day the text-to-image sharded forward on that commit was
bit-identical on uly2, in BF16 and pin-stripped INT8, to one GPU at one step
and over 50 steps and to stock at one step
([VALIDATION](VALIDATION.md#moss-image-on-comfyui-a071826fd)).
On 2026-10-27, on ComfyUI 426f5eea, an SFT edit at strength 0.5 with no VAE,
which carries the bias and no reference images, refused class P at the
driver before dispatch on uly2 and ring2. On single and cfg2, and in row 0
of a dp2 batch of two, it was bit-identical to stock on one GPU at one step
and over 50 steps, and at one step it differed from the same edit at
strength 1.0, so the bias reached the forward. Text-to-image uly2 on that
commit stayed bit-identical to one GPU at one step and over 50 steps, and to
stock at one step
([VALIDATION](VALIDATION.md#moss-image-bias-on-comfyui-426f5eea)).
The same seed does not reproduce the vendor's diffusers pipeline image,
because that pipeline draws its own noise and rounds its sigmas to the 1000
training sigmas. Comfy-Org's
ComfyUI templates can rewrite the prompt with the encoder model itself; no
shipped graph does, and that generation's driver memory is outside the
driver estimate. Community reports describe faint vertical color stripes
across smooth areas in SFT output that Base does not show; ComfyUI has no
fix for them, and no test here has measured them.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-moss-image-sft).

**Upstream and license.** [Model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT),
[ComfyUI files](https://huggingface.co/Comfy-Org/MOSS-Image) and
[source](https://github.com/OpenMOSS/MOSS-Image). The weights and the vendor's
code are under the Apache License 2.0, as the repositories' metadata and model
cards state. The Qwen3.5-9B encoder is Apache-2.0. The VAE file holds the same
bytes as Black Forest Labs' [FLUX.2 Small Decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder),
which is Apache-2.0.

<a id="model-moss-image-base"></a>

### MOSS-Image Base

Detail status (`MOSS-Image Base`): **HW**.

The base checkpoint of [MOSS-Image SFT](#model-moss-image-sft), on the same
adapter, encoder and VAE. The vendor says it had no post-training or
distillation of inference settings such as CFG.

**Recommended workflow.** [Text to image](../example_workflows/dgx-monarch-moss-image-base-t2i.json):
`euler` and ComfyUI's `normal` scheduler at the vendor's 50 steps and CFG 3,
at 1024×1024, with the vendor's tag negative and the same layer-22 skip. The vendor finds
CFG 3 enough for long prompts and suggests raising it, up to 6, for short
ones.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `moss_image_base_bf16.safetensors` | `models/diffusion_models` | Base BF16 DiT, 10.16 GB |
| `qwen3.5_9b_bf16.safetensors` | `models/text_encoders` | Qwen3.5-9B encoder; set CLIPLoader type to `moss_image` |
| `flux2-vae_small.safetensors` | `models/vae` | FLUX.2 small decoder |

**Tested configurations.** As [MOSS-Image SFT](#model-moss-image-sft), on the
same date and kinds of test: uly2 one-step latents were bit-identical to
stock on one GPU at 1024, 1040 and 1536 square, full 50-step renders to one
GPU at 1024 and 1536 square, and the shipped graph to stock over the full
schedule. FSDP rests on the scoped sweep, where BF16 and INT8 `uly2+fsdp`
equaled resident uly2; no other FSDP test ran Base. INT8 uly2 was
bit-identical to the pin-stripped copy after the INT8 dispatch fix. With the
Kromoss DMD2 LoRA at 8 steps Base rendered, and against stock with the same
LoRA it read NRMS 0.449, recorded. No test rendered Base on the `normal`
scheduler; the CPU and kernel tests listed for
[MOSS-Image SFT](#model-moss-image-sft) cover this checkpoint too.

**Known limitations.** As [MOSS-Image SFT](#model-moss-image-sft), apart from
the color stripes, which the reports do not find in Base.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-moss-image-base).

**Upstream and license.** [Model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base),
under the Apache License 2.0.

<a id="model-kromoss"></a>

### Kromoss

Detail status (`Kromoss`): **HW**.

Kromoss is lodestones' full fine-tune of
[MOSS-Image Base](#model-moss-image-base) at 1024 px. It keeps the
architecture, encoder and VAE, so it runs on the `moss_image` adapter and the
same files. The similarly named Kroma is a Krea2 fine-tune on a different
architecture ([Kroma](#model-kroma)).

**Workflows.** [Text to image at 50 steps](../example_workflows/dgx-monarch-kromoss-t2i.json)
and [8 steps with the DMD2 LoRA](../example_workflows/dgx-monarch-kromoss-t2i-turbo.json).
The full model takes the MOSS-Image settings: 50 steps, CFG 3, `euler`,
`normal`, the vendor's tag negative and the layer-22 skip. With either of the
repository's LoRAs, `kromoss_8-steps-dmd2-lora.safetensors` (the vendor's
recommendation) or `kromoss_8-steps-dmd-lora.safetensors`, use 8 steps,
CFG 1, `euler`, `simple`, strength 1 and one LoRA at a time; the vendor
reports that 4 steps also works. The repository names the two files
`8-steps-dmd2-lora.safetensors` and `8-steps-dmd-lora.safetensors`; save them
under the manifest's names. Plain Euler at CFG 1 never evaluates the
negative, so the 8-step graph zeroes it, carries no skip node, and leaves
cfg2 nothing to split.

| File | ComfyUI directory | Purpose |
|---|---|---|
| `kromoss_bf16.safetensors` | `models/diffusion_models` | Kromoss BF16 DiT, update 11000, 10.16 GB |
| `kromoss_8-steps-dmd2-lora.safetensors` | `models/loras` | 8-step DMD2 LoRA, 1.19 GB, for the 8-step workflow |
| `qwen3.5_9b_bf16.safetensors` | `models/text_encoders` | Qwen3.5-9B encoder; set CLIPLoader type to `moss_image` |
| `flux2-vae_small.safetensors` | `models/vae` | FLUX.2 small decoder |

The manifest pins `lodestones/Kromoss` at revision `7f9bb7b06d7b`, update
11000 of the fine-tune, published on 2026-10-20. The vendor has published new
weights under the same file names several times since 2026-10-18, so a file
from another revision has a different SHA-256, `check_artifacts.py --hash`
reports it, and no test record covers it. The repository's encoder and VAE
hold the same bytes as Comfy-Org's.

**Identity.** dgx-monarch reads Kromoss, MOSS-Image Base and MOSS-Image SFT as
one family, `moss_image`: their BF16 headers hold the same tensor names,
dtypes and shapes and no metadata, and their INT8 headers likewise, so no
header field tells them apart. The manifest and each test record name a file
by its SHA-256, and the first-use Gate binds a PASS to the file's name and a
sampled fingerprint of its bytes ([TRUST](TRUST.md)), so each checkpoint earns
its own PASS. A dated result for one checkpoint does not
cover another. Unlike Kroma, which auto recognizes by the merge record in its
header, Kromoss can take no checkpoint-specific auto rule.

**LoRA.** Both LoRAs were distilled from MOSS-Image Base and hold the same
bytes as lodestones' `MOSS-Image-Base-8-steps-dmd` release. Each is rank 128
in FP32 on every block's q, k, v, o and feed-forward projections, with
separate q, k and v keys that ComfyUI maps onto the fused `qkv` weight. The
vendor's diffusers recipe applies a LoRA unmerged and warns that merging it
into BF16 weights loses part of the delta to rounding. ComfyUI merges it into
the weights, and dgx-monarch bakes it with ComfyUI's own math, so a render on
the pair is compared with stock ComfyUI, not with the unmerged pipeline. On
2026-10-21 dgx-monarch's bake of either LoRA on the BF16 file equaled stock
ComfyUI's merge on every patched key, the FSDP bake and the hot swap between
them each gave the resident bake's render, and the un-bake gave the render
without the LoRA. The vendor's card says the INT8 file works with them, and
stock ComfyUI on one GPU runs that pair with its default DynamicVRAM loader.
With DynamicVRAM off, stock's classic bake rendered the INT8 file with DMD2
about 1.0 NRMS from that default, and dgx-monarch's bake is the classic one,
so dgx-monarch refuses a LoRA on the INT8 file on every topology
([TROUBLESHOOTING #129](TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)).
LoRA on FSDP keeps the [shared FSDP rules](#fsdp-precision-and-residency).

**Tested configurations.** As [MOSS-Image SFT](#model-moss-image-sft), on
2026-10-21: uly2 one-step latents were bit-identical to stock on one GPU at
1024, 1040 and 1536 square, full 50-step renders to one GPU at 1024 and 1536
square, and the shipped 50-step graph to stock over the full schedule.
`uly2+fsdp` equaled resident uly2, and ring2 on an even joint stream read
one-step NRMS 0.023 against stock. With the DMD2 and DMD LoRAs at 8 steps,
uly2 equaled one GPU inside dgx-monarch and ran 1.51x faster; against stock
with the same LoRA it read one-step NRMS 0.0365 to 0.0635 at strengths 0.8 and
1.0, and the 8-step graphs, shipped and testing, read 0.36 to 0.37 after 8
steps, recorded. cfg2 on the 8-step graph refused class P. No test rendered
the 50-step graph on the `normal` scheduler. The CPU and kernel tests listed
for [MOSS-Image SFT](#model-moss-image-sft) cover the architecture
([VALIDATION](VALIDATION.md#evidence-kromoss)).

**Known limitations.** As [MOSS-Image Base](#model-moss-image-base), ring2 with
a LoRA included. The INT8
ConvRot file `kromoss_int8_convrot.safetensors` pins comfy-kitchen's INT8
attention on all 42 blocks, as the Comfy-Org INT8 files do.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-kromoss).

**Upstream and license.** [Kromoss repository](https://huggingface.co/lodestones/Kromoss).
The weights, both LoRAs and the bundled pipeline code are under the Apache
License 2.0, which the repository's metadata, model card and LICENSE file
carry.

<a id="model-longcat-image"></a>

### LongCat-Image

Detail status (`LongCat-Image`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-longcat-t2i.json).

**Tested configurations.** BF16 Ulysses at 1024 and padded 1448 square; cfg2 with unequal-prompt padding and masks; BF16 FSDP against ordinary residency.

**Known limitations.** Padded ring and hybrid renders are guarded; no claim covers unlisted artifacts, LoRAs or topologies. Below 1.2 MP auto selects cfg2 only for BF16, at any CFG other than 1 and at CFG 1 under a CFG++ sampler, because only BF16 cfg2 has been compared; other quantizations take uly2 there. From 1.2 MP auto keeps Ulysses: in Campaign K (2026-10-14) BF16 cfg2 at 1152 to 1280 ran 1.26x to 1.27x faster than uly2 but read 1-step NRMS 0.006 to 0.007 against one GPU, where uly2 read bit-identical.

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

**Tested configurations.** Base 2512 FP8 Ulysses/cfg2 and the named FSDP presets; Edit-2511 FP8-mixed DP2/cfg2. The base-model FSDP tests do not cover Edit FSDP. Pure-Ulysses one-step renders of the base 2512 FP8 file produced identical tensors to one GPU at 1328 and 1024 with odd and even image and text streams, and at batch 2. Campaign K (2026-10-14) timed base 2512 FP8 cfg2 at 1152, 1216, 1280 and 1328 square, bit-identical to one GPU.

**Known limitations.** Default Edit `index_timestep_zero` refuses sequence sharding. Batch-one DP FSDP is refused; base-model results do not establish accuracy for Edit or other reference methods. On Ulysses, auto keeps the Init node's attention kernel and does not switch base FP8 to SAGE, because SAGE on Ulysses or ring measured over the fidelity limit; an explicit SAGE choice still runs as selected. The LoRA test graph also exceeded the limit on Ulysses with Torch Flash and cuDNN. From 1.2 to 1.77 MP auto selects cfg2 for an FP8 checkpoint; only base 2512 was timed, and an Edit-2511 FP8 file with the same header would take that row too.

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

**Known limitations.** Ulysses admits only TORCH_FLASH. A checkpoint that pins an attention kernel on its attention modules, such as comfy-kitchen Sol, still renders with the TORCH_FLASH path there, ComfyUI's default exact backend (torch scaled dot product attention when the worker cannot import xformers); the adapter logs the pin once at debug level and ignores it ([ADAPTERS.md](ADAPTERS.md)). Without sequence parallelism the pin runs as in stock ComfyUI, the render record names it, and a pinned approximate kernel asks for the Sol-Attn waiver first; under FSDP a bf16 or fp16 checkpoint that pins refuses class P at the launch check instead ([TROUBLESHOOTING #117](TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)). Compressed caches matched native ComfyUI using the same cache settings. INT8 and INT4 caching are approximate; these tests do not show equality with cache off. No claim covers a DiT and cache quantization cross-product, FSDP, unlisted workflows, locked pixels, the prompt enhancer or FunControlNet. ComfyUI 503d785f0 (2026-10-22) gave the forward a writable prefix K/V cache, which stock uses for cache device `auto` under dynamic VRAM in place of the cache these tests measured. On such a ComfyUI a Ulysses render with `auto` and a dynamic patcher refuses class P on every rank before any exchange; `gpu`, `cpu` and `off` take stock's earlier branch and are not affected ([TROUBLESHOOTING #134](TROUBLESHOOTING.md#134-a-qwen-image-21-ulysses-render-with-the-cache-device-auto-refuses-on-comfyui-503d785f0-or-later)). In CPU tests on the rig the sharded forward stays bit-identical to stock on that commit and on a071826fd, with the cache off and on; it has not been re-run on hardware there.

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

**Tested configurations.** Ulysses against one GPU, DP2, and BF16 FSDP with the auxiliary adapter replicated. Explicit cfg2 on the shipped template's prompt pair read one-step NRMS 0.0 against one GPU in three records, and three `cfg2+fsdp` records matched their resident legs exactly (BF16, 1024 square). Campaign K (2026-10-14) timed cfg2 on that pair at 1024 square, bit-identical to one GPU over the full schedule.

**Known limitations.** Unlisted FSDP artifacts and LoRA behavior are not covered. Below 1.2 MP auto selects cfg2 for BF16. comfy's 512-row pad lands inside the forward, so an unequal pair splits the model call at any length and cfg2 dispatches one conditioning per rank; an equal pair folds onto the slice path, whose speed is unmeasured. The cfg2 comparisons cover the shipped prompt pair.

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-anima).

<a id="model-boogu"></a>

### Boogu

Detail status (`Boogu`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-boogu-t2i.json).

**Tested configurations.** FP8 Ulysses, FSDP without changing stored precision, and cfg2 with unequal prompts. Pure-Ulysses one-step renders of the fp8 file produced identical tensors to the dp2 reference at 1024 and 1040, odd and even streams. Campaign K (2026-10-14) timed cfg2 on the shipped unequal pair at 1024 square, bit-identical to one GPU.

**Known limitations.** Below 1.2 MP auto selects cfg2 for FP8. Equal-prompt cfg2 performance has not been measured after the batching change. Do not pad a dispatching conditioning pair.

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

**Tested configurations.** Ulysses and cfg2 against DP2, BF16 FSDP against ordinary residency, and the 50-step image recipe. Campaign K (2026-10-14) timed explicit cfg2 at 1024 square, bit-identical to one GPU.

**Known limitations.** Do not apply the image shift to video variants. LoRA and unlisted variants are not covered. Auto keeps Ulysses: the image and video checkpoints share one family and precision, so a cfg2 row would move video renders too. Choose cfg2 explicitly for images.

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

**Known limitations.** The original 81-frame local reference refused for capacity and saved no frames, so that comparison remains CHECK. The passing tests do not cover BF16 execution, other precisions, pose-driven animation, LoRA, ring, FSDP, warm reuse or the default 40-step recipe. The activation-footprint preflight can refuse a job ([docs/TROUBLESHOOTING.md #47](TROUBLESHOOTING.md#47-wan-scailscail-2-render-refused-activation-footprint-preflight)). cfg2 and every cfg composite refuse class P for the family, since SCAIL-2 measured over the fidelity floor on cfg2 and SCAIL Preview shares the forward with no cfg2 record of its own; ring refuses beyond world 2 ([docs/TROUBLESHOOTING.md #136](TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)).

**Test results.** [Comparisons and limits](VALIDATION.md#evidence-wan-scail-preview).

<a id="model-wan-scail2"></a>

### Wan SCAIL2

Detail status (`Wan SCAIL2`): **HW**.

**Recommended workflow.** [Open workflow](../example_workflows/dgx-monarch-wan-scail2.json). Use the template settings and validated artifacts.

**Tested configurations.** Official ComfyUI settings with the DPO LoRA, comparing sampler latents; and FP8-scaled Ulysses with synthetic conditioning, comparing 81 decoded one-step frames.

**Known limitations.** Arbitrary driving media, Turbo, FSDP and other LoRA settings have no passing reference comparison. The shipped graph renders one 81-frame chunk. cfg2 and every cfg composite refuse class P: on 2026-10-28 one-step cfg2 probes of the fp8_scaled testing graph read NRMS 0.116 without a LoRA and 0.085 with the DPO LoRA against one GPU, never identical, while uly2 and ring2 were byte-identical ([VALIDATION](VALIDATION.md#wan-scail2-degree-2-probe-2026-10-28)). Ring refuses beyond world 2: a four-Spark owner reported ring4 and `auto`'s uly2+ring2 fold outside the 0.10 limit and uly4 correct, not reproduced here ([docs/TROUBLESHOOTING.md #136](TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)).

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

**Known limitations.** Sharded renders need TORCH_FLASH; any other kernel refuses. On sharded renders, a checkpoint that pins a kernel on its attention modules, such as comfy-kitchen Sol, renders its self-attention with the TORCH_FLASH path, ComfyUI's default exact backend (torch scaled dot product attention when the worker cannot import xformers), and the adapter logs the pin once at debug level and ignores it. The stock cross-attention ignores the pin too and runs that default backend ([ADAPTERS.md](ADAPTERS.md)). Without sequence parallelism the pin runs as in stock ComfyUI, the render record names it, and a pinned approximate kernel asks for the Sol-Attn waiver first; under FSDP a bf16 or fp16 checkpoint that pins refuses class P at the launch check instead ([TROUBLESHOOTING #117](TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)). The warm tests do not cover slab/low-RSS, first-use LoRA, FSDP, ring, cache, background replacement or other shapes. Every Wan-Animate 2 template ships `slab_weights=off` and `lora_low_rss=off`. Keep ordinary residency for the warm recipes; other shapes or background applications may need larger memory reserves. Explicit `lora_low_rss=off` refuses FSDP with a LoRA on these templates (class P, `validate_fsdp_launch_loras`), and the INT8 checkpoint refuses it whatever `lora_low_rss` is, because its shards are quantized (`refuse_fsdp_lora_checkpoint_property`). Run a LoRA recipe on a resident topology. On the BF16 warm and memory-saver templates the resident LoRA backup is about one BF16 model per rank.

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

For a combined playable file, use [MP4 with sound](../example_workflows/dgx-monarch-minimax-h3-mp4.json).
It preserves the generation settings and joins the decoded video and audio
with native ComfyUI nodes at 24 fps. The original workflow retains separate
PNG/FLAC outputs. Export-format coverage does not extend the model's fidelity
evidence.

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

<a id="int8-gemm-kernel"></a>

## INT8 GEMM kernel

INT8 ConvRot checkpoints run their linears through comfy-kitchen's INT8 GEMM,
which comfy-kitchen sends to one CUTLASS kernel on sm_80 and newer. DGX Monarch
runs that call as a Triton kernel instead: kan11's kernel from
[gb10-comfyui-tuning](https://github.com/kanthai/gb10-comfyui-tuning/tree/d68916bf06c7ab01e0c557c67c64a1a06eb0ceef),
vendored under the MIT license in `src/dgx_monarch/kernels/`. It applies the
row scale, the column scale and the bias in the same order as CUTLASS, so the
output is identical: 156 of 156 synthetic kernel cases and 1530 of 1530
rendered frames over 28 cells matched, and the option's own GPU check matched
240 of 240 cases over 20 shapes
([VALIDATION](VALIDATION.md#int8-gemm-kernel-comparison-2026-10-28-and-29)).
From 6144 rows up it holds about 175 TOPS on the wide linears, where CUTLASS
loses more than half its throughput, so MiniMax H3 and Wan Animate 2 renders
ran 16 to 25 percent faster, LTX 2.5 5 to 9 percent faster, and the small
stills did not change
([BENCHMARKS](BENCHMARKS.md#int8-gemm-cutlass-against-the-triton-kernel)).

The kernel is on by default in the ComfyUI process and in every actor. It
leaves on CUTLASS the fused-residual calls, rows under 128 and calls with K not
a multiple of 16 or a strided tensor. Each process logs one line, starting
`INT8 GEMM:`, that names the kernel it runs: the driver log for ComfyUI, the
Worker journal for an actor.

**Turn it off.** The actors read `int8_gemm` from `[worker_args]` in
cluster.toml:

```toml
[worker_args]
int8_gemm = "cutlass"
```

The ComfyUI process reads `DGXM_INT8_GEMM` from its environment when it loads
the node pack, so start ComfyUI with `DGXM_INT8_GEMM=cutlass` to turn the
kernel off there. Both take `"triton"` or `"cutlass"`, and both default to
`"triton"`. Neither reads the other, so set both to turn the kernel off
everywhere. To keep the default, leave the `[worker_args]` key out: writing it,
with either value, changes the identity-gate context of every render, as any
`[worker_args]` key does ([CLUSTER](CLUSTER.md#worker_args-optional)).

**Limits.**

- The first INT8 calls in each process compile the kernel, which takes
  seconds. Triton keeps the compiled kernel in its cache, so later processes
  load it.
- The validated comfy-kitchen releases are 0.2.37 and 0.2.40. On any other
  release the kernel stays off, and the process runs stock CUTLASS and logs
  one WARNING. A release joins the set only after the GPU check in
  `tests/test_int8_gemm_triton_equality.py` and a render comparison like the
  recorded one pass on it. `tests/test_int8_gemm_kitchen_version.py` fails
  when ComfyUI's requirements or the installed package move past the set.
- Triton builds a small C launcher against `Python.h`, so each process needs
  the Python development headers for its interpreter
  ([INSTALL](INSTALL.md#install-the-node-pack)). Without them, or without
  Triton, the process runs stock CUTLASS and logs one WARNING
  ([TROUBLESHOOTING #138](TROUBLESHOOTING.md#138-the-log-shows-an-int8-gemm-warning-and-int8-renders-run-stock-cutlass)).
- `compile_dit` was off in every comparison, so compiled DiTs have not run
  with the kernel.
- Two MiniMax H3 INT8 ConvRot files were not part of the comparison: the
  unpruned fl2va file and the ref2va file. They share the pruned file's GEMM
  shapes.
- The tile configuration was tuned and measured on GB10 only.

<a id="worker-assisted-vae-decode"></a>

## Worker-assisted VAE decode

A video decode runs in the ComfyUI driver and takes tens of seconds to a
minute on one GB10 while the Workers sit idle. With
`[worker_args] worker_vae_decode = "auto"` in cluster.toml, the stock VAE
Decode and VAE Decode (Tiled) nodes split a decode's units over the Workers:
the driver decodes its own share and each Worker rank on another GPU decodes
the rest. The output is byte-identical to ComfyUI's one-host tiled decode for
that VAE, because ComfyUI's own loop runs once on the driver with only the
per-unit decode swapped (DESIGN.md section 5.6). It is off by default until the
hardware byte compare passes ([VALIDATION](VALIDATION.md#worker-assisted-vae-decode)).

| family | VAE files the templates load | splits under | unit | hardware byte compare |
|---|---|---|---|---|
| MiniMax H3 video | `minimax_h3_video_vae_fp16.safetensors` | VAE Decode and VAE Decode (Tiled), which run the same decode on this VAE | one temporal chunk, which the VAE tiles inside | NOT RUN |
| Wan 2.1 (also Wan 2.2 14B, FlowRVS, SCAIL, WanDancer, Wan-Animate 2) | `wan_2.1_vae.safetensors`, `Wan2_1_VAE_bf16.safetensors`, `wan21_flow_rvs_mask_vae_bf16.safetensors` | VAE Decode (Tiled) | one tile of the tiled decode | NOT RUN |
| Wan 2.2 TI2V | `Wan2_2_VAE_bf16.safetensors` | VAE Decode (Tiled) | one tile | NOT RUN |
| HunyuanVideo 1.0 (also Kandinsky5 Video) | `hunyuan_video_vae_bf16.safetensors` | VAE Decode (Tiled) | one tile | NOT RUN |
| HunyuanVideo 1.5 | `hunyuanvideo15_vae_fp16.safetensors` | VAE Decode (Tiled) | one tile | NOT RUN |
| LTX 2.3 and LTX 2.5 | `LTX23_video_vae_bf16.safetensors`, `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-video-vae-conv-bf16.safetensors` | VAE Decode (Tiled) | one tile | NOT RUN |

**Which node.** On every family but MiniMax H3, VAE Decode runs the whole
latent through the model at once, a different result from the tiled decode,
so it stays on the driver; to split, use VAE Decode (Tiled), as the LTX
templates already do. The Wan, HunyuanVideo and Kandinsky5 templates use VAE
Decode and so keep their one-host decode. The MiniMax H3 templates split as
shipped.

**What stays on the driver.** CogVideoX, whose tiles are not independent; an
LTX VAE whose decode draws unseeded noise (timestep conditioning or injected
noise), which has no reproducible reference; image VAEs and every other class;
a still at or under 1.2 MP, which decodes in about a second; a MiniMax H3
still, which is one chunk; a VAE that came from a checkpoint rather than a VAE
loader file, since no Worker can name its bytes; and every encode. Each decode
logs one line saying whether it split and why not
([TROUBLESHOOTING #140](TROUBLESHOOTING.md#140-a-vae-decode-did-not-split-over-the-workers)).

**Limits.**

- Each Worker needs the driver's VAE file under the same name in its own
  `models/vae` folder with the same artifact signature, and the same ComfyUI
  commit, torch and CUDA build and comfy-kitchen release. A Worker adopts the
  driver's VAE dtype, attention functions and torch backend flags for the
  decode, so a driver started with `--use-sage-attention` needs the sage
  package on every Worker host. Any difference makes that rank decline, and
  the driver decodes its units.
- A Worker that took a unit keeps the decode VAE resident beside its DiT until
  `clear_vram`, `unload` or a teardown that releases models drops it: about
  5 GB for the MiniMax H3 video VAE in fp16 or bf16 (twice that in fp32) and
  0.25 to 2.5 GB for the others. The driver's free-before-refusing step drops
  it before a render-site capacity refusal.
- Decoded units a Worker finished wait on the driver until the decode reaches
  them, beside the decode's own output, within a 2 GiB budget. A Worker claims
  only units the budget can hold, near the driver's position, and pauses while
  it is full; the decode's line counts the pauses. The budget holds about
  twelve MiniMax H3 chunks at 1344x768 in fp16 or bf16, and a 362-frame clip
  leaves about ten on the Worker's side.
- MiniMax H3's decoder batches up to four spatial tiles per call by the free
  memory it reads, and Comfy's tiled loop casts and blends in the order of its
  own code. Both are the same on a GB10 with memory to spare, which the
  hardware compare confirms.
- No speed has been measured. Two ranks should roughly halve a long decode, less
  the transfer of each unit's pixels (about 173 MB per MiniMax H3 chunk at
  1344x768).

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

### FLUX.3 Image Klein vendor files

Black Forest Labs publishes FLUX.3 Image Klein in ComfyUI layout at
`black-forest-labs/flux-3-image-klein-comfy@3700a1012ec75ad50a54e92199f85ba7cba93c1e`,
a gated repository. These are vendor files, not local conversions. The
[2026-10-15 and 2026-10-16 tests](VALIDATION.md#evidence-flux-3-klein) cover
the BF16, INT8 ConvRot and NVFP4 files; the support matrix states what each
topology asks of each kind.

<details>
<summary>FLUX.3 Image Klein files, layouts and checksums</summary>


| File | Bytes | SHA-256 (first 12) | Header kind | Layout | Pinned attention |
|---|--:|---|---|---|---|
| `flux3_image_klein_sd_bf16.safetensors` | 11,097,404,512 | `c3e9813cdd04` | bf16 | separate q, k and v; `ref_time_in` (step-distilled) | none |
| `flux3_image_klein_bf16.safetensors` | 11,076,957,072 | `5c921759dc03` | bf16 | separate q, k and v | none |
| `flux3_image_klein_sd_int8_convrot.safetensors` | 5,827,874,779 | `847d8933548a` | int8 | fused `qkv`; 172 tensorwise INT8 Linears with ConvRot | comfy-kitchen INT8 on 33 blocks |
| `flux3_image_klein_int8_convrot.safetensors` | 5,807,427,339 | `8949ecfc1bf3` | int8 | as above | as above |
| `flux3_image_klein_sd_nvfp4.safetensors` | 5,277,685,632 | `15d421e25584` | fp8 | separate; 198 NVFP4 layers, each with a static input scale; BF16 on the text blocks, the first image block and the last four joint blocks | none |
| `flux3_image_klein_nvfp4.safetensors` | 5,257,238,192 | `f99714a57ada` | fp8 | as above | none |

</details>

The diffusers repositories' root `model.safetensors` files hold the same
bytes as the two BF16 files above. Their `variants/fp8r/model.safetensors`
files carry per-row FP32 weight scales with no ComfyUI quantization record:
stock ComfyUI loads them without an error, drops the scales and computes
unscaled, so dgx-monarch refuses them and ships no template or manifest row
for them ([TROUBLESHOOTING #120](TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
Their `variants/nvfp4/model.safetensors` files hold 557 of 953
(step-distilled) and 555 of 951 (base) tensors byte-identical to the
ComfyUI-layout NVFP4 files above and carry no ComfyUI quantization record.
In the 2026-10-15 test stock ComfyUI dropped their scales and failed at the
first step. dgx-monarch's header gate refuses them by the same rule, so no
template or manifest row names them.

### MOSS-Image vendor files

Comfy-Org publishes MOSS-Image in ComfyUI layout at
`Comfy-Org/MOSS-Image@3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb`, and lodestones
publishes Kromoss at `lodestones/Kromoss@7f9bb7b06d7b6c27d294b0bab5a2b3056c379279`.
Neither repository is gated. These are vendor files, not local conversions.
The [2026-10-21 tests](VALIDATION.md#evidence-moss-image-sft) cover the BF16
and INT8 ConvRot files; the support matrix states what each topology asks of
each kind.

<details>
<summary>MOSS-Image files, layouts and checksums</summary>


| File | Bytes | SHA-256 (first 12) | Header kind | Layout | Pinned attention |
|---|--:|---|---|---|---|
| `moss_image_sft_bf16.safetensors` | 10,164,655,808 | `f96738d8116d` | bf16 | fused `qkv`; 378 BF16 tensors and one F32 `rope.bands` that ComfyUI drops at load; no header metadata | none |
| `moss_image_base_bf16.safetensors` | 10,164,655,808 | `cacd158728a1` | bf16 | as above | none |
| `kromoss_bf16.safetensors` | 10,164,655,808 | `54c863b9392f` | bf16 | as above | none |
| `moss_image_sft_int8_convrot.safetensors` | 5,413,698,958 | `b4d1475bc7ee` | int8 | fused `qkv`; 168 INT8 ConvRot Linears, the q, k, v, o and feed-forward projections of 42 blocks | comfy-kitchen INT8 on 42 blocks |
| `moss_image_base_int8_convrot.safetensors` | 5,413,698,958 | `f601c1e11619` | int8 | as above | as above |
| `kromoss_int8_convrot.safetensors` | 5,413,698,958 | `cacff0dd25cd` | int8 | as above | as above |

</details>

The three BF16 files share one header layout, and so do the three INT8 files,
so nothing in a header tells the checkpoints apart. The encoder
`qwen3.5_9b_bf16.safetensors` (`7e6e9f08d598`) and the VAE
`flux2-vae_small.safetensors` (`d8d52ba03647`) are the same bytes in both
repositories, and the VAE is the same bytes as
`diffusion_pytorch_model.safetensors` in `black-forest-labs/FLUX.2-small-decoder`
at `a3efc24f613e`. Comfy-Org's `vae/flux2-vae.safetensors` (`868fe7b343cc`) is
the full Flux 2 VAE in Black Forest Labs' layout, not the file the manifest
pins for the Flux 2 and FLUX.3 templates, and no MOSS-Image graph names it.

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

A four-Spark owner has published a full four-node matrix, reported and not
reproduced here
([community results](BENCHMARKS.md#community-four-node-results-2026-10-27)).
In their runs uly2+cfg2 ran fastest for models that use CFG and uly4 for
MiniMax H3 and LTX, small or fast models gained nothing, and on Wan SCAIL2
every layout but uly4 failed the one-step check. At world 4, `auto` stays
conservative by design: a batch-1 render on a cfg row refuses class K rather
than build a cfg-plus-sequence layout no record has measured, since every cfg
row was measured at world 2 ([DESIGN.md](DESIGN.md#54-topology--auto-mode)).
The explicit presets that fill world 4, `uly4`, `ring4` and `uly2+cfg2`, are
the operator's choice there, within each model's constraints. This project
has verified none of them at world 4.

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
size instead. Native latent RDMA is off by default, and actor messaging is the
supported return path. [Selected-rail checks](VALIDATION.md#native-rdma) have
limited recorded coverage; native cancellation and fabric-failure recovery are
untested.

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
