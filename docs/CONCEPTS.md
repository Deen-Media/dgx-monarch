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

**Render session.** One logical unit of active render work holding the
Attached mesh, including pending results and identity-gate proof renders. A
Render session can block a reset or update even between visible denoising
steps.

**Unified memory (UMA).** On a DGX Spark (and Macs, Jetson, Strix Halo), the
CPU and GPU share one physical memory pool, 128 GB on a Spark, with no
separate VRAM. Models, LoRAs, the browser, and a local LLM compete for the
same pool.

## Operator words

**Readiness.** A health summary shared by the CLI, live dashboard, and browser
panel. It reports the Worker service, Attached mesh, and Render session
separately, then derives `ready`, `degraded`, `blocked`, or `unknown` overall.
Live telemetry uses this precedence: a confirmed blocker, missing or
inconclusive evidence, known degradation, then ready. Missing, stale,
malformed, or failed reads cannot report ready. Status does not observe Render
sessions, so that layer stays `unknown` and the status overall is `unknown` at
best. Doctor keeps its own checks, counts, aggregate and exit codes. Its
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

**The LoRA backup.** When ComfyUI applies ("bakes") a LoRA into a model, it
keeps a backup of every original weight it touched so it can undo the edit
later. A LoRA stack that touches most of the model therefore costs almost a
full extra copy of the model, about 24 GB for a 24 GB model. On a discrete
GPU, that backup uses system RAM; on unified memory it uses the shared pool.

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
it (see Recycle below). Slab mode does not create the arena.

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
low_rss and is disabled under FSDP or `compile_dit`; the identity ceremony
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
it.

## Trust words

**Identity gate.** A render-level self-test. The Gate renders one unchanged
request before and after the relevant swap or reload cycle. Both outputs must
be identical. When slab residency is active, the gate also compares a fresh
stock-residency reference; automatic family detection can add another proof
render. A gate run costs two to four short proof renders.

**Auto-gating.** The first render of a new model, residency and LoRA
combination runs the gate automatically.

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

**Quarantine.** A `FAIL` disables the affected optimization across the fleet,
records the verdict so later sessions keep it off, and uses the stock path
when fallback is permitted.

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

**Bake / hot-swap.** Baking is merging LoRA math into the weights so renders
run at full speed. A hot-swap changes the LoRA stack of a loaded model without
reloading it from scratch.

**Checkpoint.** The model file on disk (`.safetensors`).

---
Last documentation review: 2026-10-07. Dated hardware-validation evidence is
in [VALIDATION.md](VALIDATION.md).
