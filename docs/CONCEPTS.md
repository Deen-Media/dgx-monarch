# Concepts, in plain language

This page defines project terminology for ComfyUI users with one or more GPUs.
Measurements and their dates live in [VALIDATION.md](VALIDATION.md).

## The machines and the processes

**Driver.** The foreground ComfyUI process that serves the browser UI and
loads the custom nodes. Its exit does not imply that cluster Worker services
stop.

**Worker service.** The small persistent process dgx-monarch runs on each
cluster host. It listens for a driver and coordinates creation of one
mesh-owned **actor** process per GPU after a mesh attaches. The service can be
up while no actors or models exist. Low-level logs may call it a worker loop.
The UI and CLI use *Worker service* to distinguish a service restart from
resetting actors.

**Attached mesh.** The driver's current client-owned set of GPU actors. It
loads models, renders, and reports status. Resetting the Attached mesh retires
those actors and their models; it does not stop the persistent Worker
services. When a render is split across GPUs, the mesh actors do the work and
the driver orchestrates them.

**The head and the sibling.** On a pair of boxes, the head is the box the
driver runs on and the sibling is the other one. The capacity refusals, the Old
model in the way card, the capacity agreement's log line and the Fleet
timeline name a box this way, never by rank number or by its address in
cluster.toml. With more than two boxes, each other box is "the sibling at" its
host name; a fleet on the driver's box alone is "this box". A Worker's own
refusal says "here", and the driver opens it with the box: "On the sibling,
stock residency cannot load X here". Operator logs and machine fields may still
name a host.

**Render session.** One logical unit of active render work holding the
Attached mesh, including pending results and identity-gate proof renders. A
Render session can block a reset or update even between visible denoising
steps.

**Unified memory (UMA).** On a DGX Spark (and Macs, Jetson, Strix Halo), the
CPU and GPU share one physical memory pool, 128 GB on a Spark, with no
separate VRAM. Models, LoRAs, the browser, and a local LLM compete for the
same pool.

## Operator words

**Readiness.** The live dashboard and browser panel separate **Health** from
**Activity** and the advisory **Maintenance** message. Healthy active rendering
shows `Health READY`, `Activity RENDERING`, and a message to wait for the render
to finish. Other active mesh work shows `BUSY`. Activity alone is not a health
degradation; missing evidence and actual blockers remain `UNKNOWN` or `BLOCKED`
even when rendering is observed. No display state grants permission to reset,
restart, or replace a mesh; lifecycle safety checks still apply.

The Worker service, Attached mesh, and Render session remain separate layers.
For compatibility, readiness JSON retains its version-1 `overall` vocabulary
(`ready`, `degraded`, `blocked`, `unknown`) and its existing active-work mapping
to `degraded`. Telemetry-derived reports add `presentation`, with
`schema_version: 1`, `health`, `activity` (`idle`, `rendering`, `busy`, or
`unknown`), and an inert `maintenance` enum (`none`, `wait`, `inspect`, or
`inspect_and_wait`). New displays validate this projection against the original
layers and aggregate. An absent projection keeps the legacy **Readiness** label
and explains that legacy `DEGRADED` includes active work; a malformed projection
reads unknown. Doctor retains its own warning-based `degraded` verdict and does
not publish this telemetry-only projection.

Live telemetry uses this precedence: a confirmed blocker, missing or
inconclusive evidence, known degradation, then ready. Missing, stale,
malformed, or failed reads cannot report ready. Status does not observe Render
sessions, so that layer stays `unknown` and the status overall is `unknown` at
best; use `dgxm top` or live telemetry to observe render activity. Doctor keeps
its own checks, counts, aggregate and exit codes. Its
render-session layer is `unknown` because doctor never observes one, and that
alone never downgrades a ready doctor result; missing driver or mesh evidence
still shows the Attached mesh as `unknown`. Readiness reports health and
suggests actions without executing them. Each action identifies its safety
class and the operator command, if one exists. An Attached mesh being created
reads as active, not idle; a private fleet left unpublished by an interrupted
creation reads as blocked for manual recovery, never as a safe empty cache.

The live Worker service row uses separate, fresh observations of each
configured Worker's process and listening socket. The driver refreshes these
passively in the background and limits repeat checks; dashboard requests do
not wait for them. A first observation still pending, an expired result, or a
failed or incomplete read shows `unknown`. A still-fresh result may remain
visible while the next check runs. Changed configuration or mesh identity
requires new observations.

A ready Worker service row does not prove an attached actor is healthy or a
render succeeded. Actor and render reports do not substitute for the service
check. Investigate an unknown row with the service diagnostics before changing
anything; the label alone is not a reason to restart a Worker or run a
privileged helper.

**Setup profile.** A choice made only during `dgxm setup`: `safe`, `balanced`,
or `advanced`. It compiles into existing strict `cluster.toml` keys plus
separate graph recommendations, then disappears: there is no runtime `profile`
key, and no profile label is written to the config. The explicit keys preserve
the chosen policy across future profile changes. The exact mapping lives in
[CLUSTER.md](CLUSTER.md#guided-setup-profiles).

**Operator receipt.** A bounded, versioned JSON record of a setup, safe repair,
or verified update. It is audit evidence, not a full log and not a
cryptographic signature of the machine that produced it. Publication,
retention, and disclosure details live in
[INSTALL.md](INSTALL.md#operator-receipts); the security boundary lives in
[SECURITY.md](../SECURITY.md#operator-workflow-boundary).
Receipt schema v2 records an indeterminate step as `unknown`, distinct from a
settled `failed` step and from the transaction's top-level `partial` outcome.

## Rendering across GPUs (topologies)

A **topology** defines how one render is split across GPUs. The Init node's
`auto` mode applies the repository's explicit conservative rule table and logs
the chosen row and reason. Some rules have not been validated on hardware, so
an automatic choice does
not guarantee correctness or improved performance. Rule sources and family
test results are listed in
[MODELS.md](MODELS.md).

**cfg2 (CFG-parallel).** Most renders compute prompt and negative-prompt
versions of every step. cfg2 places each on a separate GPU and targets small
to medium images.

**Ulysses and ring (sequence parallelism, "USP").** The image is split across
GPUs. Each GPU holds a token slice and exchanges attention data each step.
This mode targets large images and video.

**FSDP (weight sharding).** The *model weights* are split across GPUs, so each
holds its share, a half on a two-GPU pair. This is a capacity tool, not a
speed tool: it is about 2x slower and exists for models that do not fit (or to
leave headroom for a co-resident LLM).

**Fleet.** Each GPU renders a complete image from a separate prompt. This mode
targets throughput rather than single-render latency.

## Memory words

**Single-copy.** One weight allocation per GPU. The worker also needs memory
for LoRA backups and the retained arena described below.

**Driver models.** The text encoders, VAEs and prompt upsampler a graph loads
on the driver share the head with the DiT its Worker holds. Under `auto` the DiT loads at
the first render, after the driver has loaded them, so the UNET loader prices
the DiT's load beside every model the graph loads, and a model behind a
switch's unused branch is not counted. Models ComfyUI already holds are
credited up to their size, no more. When the sum does not fit, the loader
refuses before anything loads and names the smaller files on disk that would
and that this graph would run, and a pair with the Init node's topology set to
`uly2+fsdp`, where half the DiT would. On the pair, Ulysses and ring hold the
whole DiT on the head, so the pair alone does not help, and `auto` never picks
`uly2+fsdp`. ComfyUI does not run the loader again on a re-queue that changes
nothing upstream of it, so the sampler, or Identity Gate (DGX Monarch), runs the
same price for such a queue, crediting the models ComfyUI holds by then.

**The LoRA backup.** When ComfyUI applies ("bakes") a LoRA into a model, it
keeps a backup of every original weight it touched so it can undo the edit
later. A LoRA stack that touches most of the model therefore costs almost a
full extra copy of the model, about 24 GB for a 24 GB model. On a discrete
GPU, that backup uses system RAM; on unified memory it uses the shared pool.
A new strength, a new order or a LoRA removed reuses that backup, so the capacity
check counts it as held and charges only the LoRA files the change reads again:
on a BF16 Wan-Animate 2 model such a change took under 1 GB more, about its
LoRA file (2026-11-05).

**low_rss (lazy un-bake).** Enabled by default on unified memory. The original
weights remain in the checkpoint file. The worker bakes the LoRA, discards the
in-memory backup, and records each original weight's verified file location.
Changing a LoRA stack re-reads only the affected weights from disk and
re-bakes with identical results: about 10-23 seconds depending on the
quantization and residency, with a ~150 MB working footprint instead of ~24 GB
(measured 2026-07-07 and 2026-07-08).

**The retained arena (the "pool" gauge).** The classic model-loading path
retains a model-sized allocation (~20 GB for a 24 GB model) after model
unload. It returns to the operating system only when the actor process exits.
The "pool" segment in `dgxm top` shows this allocation. **Reset attached
mesh** in the panel (or the `recycle` level on the Clear VRAM node) reclaims
it (see Recycle below). Slab mode does not create the arena. What an actor
keeps varies by load: on 2026-11-05 a stock load with no LoRA left about
3 GiB of anonymous memory once the weights were placed, a load with a LoRA
kept 33.6 to 34.0 GiB, ComfyUI's backup of the weights the LoRA patches, and
after four templates on one Worker generation the head's actor had kept
6.0 GiB and the sibling's 29.9 GiB
([VALIDATION](VALIDATION.md#capacity-checks-2026-11-05)).

For a memory-heavy decode, connect the final sampled latent to Clear VRAM's
optional `samples` input and connect its `samples` output (slot 1) to the
decoder. Use `level=recycle` and `include_driver=false`: the actors retire
before the unchanged latent reaches the VAE, while the driver's encoder/VAE
models remain available. Persistent Worker services stay running. Busy or
unconfirmed recycle outcomes block the latent output. Place this after the
last distributed sampler; a later sampler recreates actors and reloads
weights. The status output stays slot 0.

**Slab (`slab_weights`).** A slab is a plain block of shared memory that holds
the model's weights exactly as stored in the checkpoint file. On Spark-class
hardware the GPU can compute on ordinary memory directly, so the weights do
not need a GPU-managed copy. Measured on the pair on 2026-07-08 with krea2
checkpoints:

* the model costs about its file size;
* no retained arena: measured 0.0 GiB against 20.7 GiB in the same worker;
* the 24.5 GiB pread itself is ~8.5 s; complete load readiness is ~20.1 s
  end to end versus ~26.0 s for the classic path;
* unloading returns the slab allocation to the OS without a recycle;
* output is identical on every quantization measured so far: bf16,
  fp8-scaled, and int8-convrot.

`auto` (the default) turns it on per model family on unified memory, for the
families with slab identity-gate evidence on this hardware class, a fixed list
in the code: Krea2 (2026-07-09) and Flux 2 (2026-08-06). Each dated campaign
included the family's recorded identity ceremony and a slab-vs-stock
comparison in its documented scope. A checkpoint's very first load stays on
the classic path while its family is learned; every later load of the same
file uses the slab. Other families stay classic until validated. Slab requires
low_rss and is disabled under FSDP, and under `compile_dit` for a family it
compiles ([TROUBLESHOOTING #14](TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on)); the identity ceremony
also renders one stock-residency reference whenever slab is active, so a
divergence quarantines slab.

**Comfy-managed (`comfy_managed`).** This opt-in residency starts ComfyUI
DynamicVRAM inside each actor, as stock ComfyUI's `main.py` does. ComfyUI then
places and pages the weights. Enable it on the Init node or in
`cluster.toml`. It has no `auto` value, forces `lora_low_rss` and `slab_weights` off,
and refuses unsupported contexts: a LoRA stack, FSDP, a dual-model render, a
Fleet call without an exact `PASS`, and RDMA latent return.
Changing it needs an Attached mesh reset so fresh actor processes can
start with the new setting. See [docs/TROUBLESHOOTING.md #62](TROUBLESHOOTING.md#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses).

**Recycle / Reset attached mesh.** Retire the mesh-owned actor processes and
spawn fresh ones on the next render. `recycle` is the graph and internal term;
the sidebar says **Reset attached mesh** to distinguish it from restarting a
Worker service. A reset reclaims the retained arena; slab mode normally avoids
it. When you move from one large model to another on the same attached mesh,
the Worker drops the last one before it loads the next, and the price counts
that, but the retained arena and any model in the other slot stay. So a
refusal may still say "the old model (<name>) is still loaded" and offer
**Clear and retry**, which runs this reset; **Always clear and retry** lets
the driver run it for you. A new LoRA on a large model held with another one
can meet the same card when the host copy that LoRA keeps holds the memory:
"The model loaded earlier (<name>) holds about <N> GiB on the head that this
change needs. Clear the Sparks and run again?" A new strength or order of the
LoRAs it holds, or one of them removed, reuses that host copy and does not.

## Trust words

**Identity gate.** A render-level self-test. The Gate renders one unchanged
request before and after the relevant swap or reload cycle. Both outputs must
be identical. When slab residency is active, the gate also compares a fresh
stock-residency reference; automatic family detection can add another proof
render. A gate run costs two to four short proof renders.

**Auto-gating.** The first render of a new model, residency and LoRA
combination runs the gate automatically, through any DGX Monarch sampler. A
dual-model render proves both of its models in one gate run, and that verdict
covers only the pair.

**The ledger.** Where verdicts live (`output/dgxm_gate_ledger.jsonl`). Each
contextual `PASS`, `INCONCLUSIVE`, or `RETESTING` row is keyed to bounded
start/middle/end content fingerprints of the model and LoRA files, exact
ComfyUI version, gate-protocol/dgx-monarch version, the canonical dgx-monarch
package-source manifest, and effective worker/fleet/topology capability
context. That source identity is cached on first use for the life of each
process, not monitored live: restart after an editable-install or branch
change, and a fresh process seeing same-version source drift treats prior
contextual authority as stale. A `PASS` applies only to its tested residency.
It cannot be reused between any two of the three residencies (stock, slab,
comfy-managed), across package sources, or across incompatible cluster policy.
Sticky FAIL quarantine remains source-independent. The bounded read catches
same-name replacements and corruption in those windows without filling the
unified page cache with a multi-GiB checkpoint. It is a practical fingerprint,
not a full-file integrity proof.

**Quarantine.** A `FAIL` turns `lora_low_rss` and `slab_weights` off for that
model combination on every rank, records the verdict so later sessions keep
them off for it, and uses the stock path when fallback is permitted. Other
checkpoints on the same Init node keep the settings the graph asked for. A
`FAIL` does turn `comfy_managed` off for the whole Init node.

**Ambient verify.** After each LoRA swap the worker recomputes a sample of
weights (two keys by default) through stock ComfyUI math. A mismatch drops
that model slot and reloads it the stock way, and counts the failure in the
Cluster Status node.

## Model words

**Quantization (quant).** Storing weights in fewer bits to save memory and
often gain speed: `bf16` and `fp16` are the full-size 16-bit kinds, `fp8`,
`int8` and `mxfp8` store 8-bit weights, and `nvfp4` stores 4-bit weights. FP16
is named separately because BF16 validation does not establish FP16 support
(FSDP, for example, admits fp32 islands inside a bf16 core but only a uniform
fp16 core). Model loading uses stock ComfyUI; each memory feature is gated
separately for every quantization it supports.

**Weight dtype cast.** The UNET loader's `weight_dtype` widget can load a file
in another dtype than it is stored in, for example an fp16 file as bf16 or a
bf16 file as fp8. The loaded copy then takes the new dtype's width, so it can
be larger or smaller than the file. The capacity checks read the file's header
to size that copy and price the load with it in place of the file.

**INT8 GEMM kernel.** INT8 ConvRot checkpoints multiply 8-bit integers through
comfy-kitchen, which sends each linear to one CUTLASS kernel. DGX Monarch swaps
that one kernel for a Triton kernel tuned for GB10, which gives identical
output and runs large linears faster. It is on by default in the ComfyUI
process and in every actor; `[worker_args] int8_gemm = "cutlass"` turns it off
in the actors and `DGXM_INT8_GEMM=cutlass` in the ComfyUI process. A process
that cannot run it uses CUTLASS and logs one warning
([MODELS.md](MODELS.md#int8-gemm-kernel)).

**Bake / hot-swap.** Baking is merging LoRA math into the weights so renders
run at full speed. Stock ComfyUI's default loader applies a LoRA when a layer
casts its weight instead, and the Workers bake each weight to the one that cast
computes, so a LoRA renders as it does in stock ComfyUI, an INT8 file included
([DESIGN 5.5](DESIGN.md#55-model-store-reuse--lora-hot-swap)). A hot-swap
changes the LoRA stack of a loaded model without reloading it from scratch.
When the swap would first have to copy the whole model off the GPU, as adding
a LoRA to a bare base does with low-RSS off, the Worker loads base and LoRA
fresh instead, which is faster at the same price and bakes the same way.

**Checkpoint.** The model file on disk (`.safetensors`).

---
Last documentation review: 2026-10-16. Dated hardware-validation evidence is
in [VALIDATION.md](VALIDATION.md).
