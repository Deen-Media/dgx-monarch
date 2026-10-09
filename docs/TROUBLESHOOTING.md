# Troubleshooting

See [CONCEPTS.md](CONCEPTS.md) for project terminology. Search this page for
the error text or symptom; report failures not covered here.

> **A cold combination on a normal KSampler path can wait roughly 2-4 minutes
> on the tested hardware before showing first pixels.** Model size, storage,
> and hardware change that timing. This is expected with the default
> `auto_gate=first_use`: DGX Monarch KSampler and KSampler Advanced prove a new
> model/residency/LoRA combination before dispatching the user's first render,
> rather than rendering once through an unverified optimized path. Do not restart
> a healthy job during this proof. **Every dgx-monarch release version
> invalidates earlier PASS results.** A same-version package-source change seen
> by a fresh process also invalidates earlier results for that configuration.
> Each combination is checked again on first use after an upgrade or source change.
> A change to the artifact bytes, the ComfyUI commit, the gate protocol, the
> canonical package-source manifest, or the effective worker, fleet or topology
> context also needs a fresh proof. Each process reads the source manifest once,
> on first use, and does not watch it: restart the driver and the workers after
> changing an editable checkout.
> The check uses two short test renders and compares slab against stock when
> slab is active. A new model may also need a render to identify its family
> before reloading into slab, for up to four renders total. Fleet skips
> the inline identity check: an unverified Fleet combination runs with
> slab and low-RSS residency forced off. Single-model custom samplers
> require the explicit Identity Gate node. Dual-model custom requests are
> always dispatched with both residency optimizations off; the single-model
> Gate cannot authorize their second model slot.

## 1. Attach fails with `MESH_ATTACH_CONFIG_TIMEOUT`

dgx-monarch sets torchmonarch's attach timeout to 60s before the first
monarch import (`dgx_monarch/__init__.py`; hyperactor snapshots the
environment at that import, so a later set never applies). Export
`HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT` before ComfyUI starts to override it.
The driver waits 70s for the attach to initialize, strictly above the
config-push budget, so monarch's typed per-host error wins over a bare
TimeoutError. The wait follows the budget in force, not the shipped one: an
export of 180s waits 190s. Concurrent Init calls for the same mesh key share
one creator, and a wedged creator raises `MeshAttachError` after 180 seconds
rather than parking every waiter; `DGXM_MESH_CREATION_TIMEOUT` may override
that wait with a finite positive value up to 3600 seconds. Creation is
single-flight across the process: a different mesh key is refused with
`MeshAttachError` while an attempt is active; retry it after that settles.
Changing the limit does not make an unconfirmed old fleet safe to replace.

If attaches still time out, check the causes below. After any upgrade, rule
out a mixed-pin mesh first: driver and Worker services on different
torchmonarch builds attach and spawn cleanly, then every endpoint call times
out with no version message in either direction. Run `dgxm doctor` and check
both hosts' pin rows before treating it as a wedge.

**Symptom:** Init node (or probe) times out attaching; Worker service logs show a
connection that never completes.

**Cause:** the Monarch client advertised a loopback address. The default
transport resolves the driver's own hostname, and `/etc/hosts` maps it to
`127.0.0.1` on many distros. The workers' replies then dial themselves.

**Fix:** set `cluster.client_bind = "tcp://<driver fabric IP>:0"` in
cluster.toml. dgx-monarch refuses a cluster attach without it, so mapping each
box's hostname to its fabric IP in `/etc/hosts` is not enough on its own. Then
see [#2](#2-every-attach-times-out-after-one-failed-attach): the failed attach has left the Worker services wedged.

`dgxm doctor`'s `hostname resolution` row reads that setting. When the
hostname resolves to a routable address, the row reports `[ ok ]` with that
address. When it resolves to loopback or not at all (normal on minimal server
images and in containers), a usable `client_bind` (a unicast IP that is not
loopback) still gives `[ ok ]` and the row names the IP it pins, because the
client transport advertises the bind and never resolves the hostname. Without
one the row warns.

### Self-heal (default on)

Before attaching, the driver checks the exact Worker service process and the
host's passive LISTEN table, then restarts dead services, so a cold fleet can
start from a queue press. It never opens a protocol-blind connection to a
Monarch worker endpoint (docs/VALIDATION.md). If an attach fails anyway, the
driver retries once in-process (the current pin evicts the stale attach
session, meta-pytorch/monarch#4067). When the retry also fails, it restarts the
Worker services so the next attempt starts clean, and that ComfyUI session
refuses further attaches: restart ComfyUI (the fleet is already healthy). One
case skips the in-process retry: see
[#2](#2-every-attach-times-out-after-one-failed-attach). Set `auto_heal = false` in cluster.toml to opt out.

## 2. Every attach times out after one failed attach

**Symptom:** Worker services look alive (`dgxm status`: process running, listener open)
but attaches keep timing out.

**Cause:** the Worker service coordinator died while its TCP listener survived.
Two known triggers: a reply-timeout attach (see [#1](#1-attach-fails-with-mesh_attach_config_timeout)), and anything calling
`hosts.shutdown()` on an attached Worker service. dgx-monarch's own detach
never calls it; it stops its procs and leaves the service to `dgxm`. Normal
client exits and crashes did not wedge Worker services in the two-Spark
checks: repeated attach, NCCL bring-up and detach cycles ran without service
restarts ([VALIDATION.md](VALIDATION.md)).

**Fix:** auto-heal (`auto_heal` in cluster.toml, [#1](#1-attach-fails-with-mesh_attach_config_timeout)) heals dead Worker service
listeners before the first attach. A failed attach retries once in-process; a
second failure restarts the Worker services so the next attempt starts clean.
`dgxm restart` remains the manual fallback. One residual case needs more: [#15](#15-attaches-still-time-out-after-dgxm-restart-driver-side-wedge).

One case fails fast instead of retrying. When the driver has just retired a
dead fleet's replacement latch on positive proof that its worker processes are
gone, and the attach to the replacement then fails, that driver process cannot
attach again in the tested configuration with the pinned Monarch runtime
(see the hardware finding below). That attach therefore makes exactly one
attempt, and its error says no
in-process retry ran, names the recovery that works, and sends the operator to
restart ComfyUI
([#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out)). Every
other attach keeps two attempts, and its second failure poisons the session.

**What the hardware tests showed.** With the pinned Monarch runtime, changing
the fleet under a live driver left that driver unable to attach again. The
tests covered four failure cases and a successful recovery:

| What was tried | Result |
|---|---|
| A worker actor died while its loop was stopped, then the driver tried to attach (reproduced 4 times) | Every attach from that driver failed |
| One worker loop alone restarted under a live session, then the driver tried to attach | Failed, and stayed unattachable for every later driver until both loops restarted together |
| Both worker loops restarted together under a live session, the full 190 s pace waited | Still failed from that driver |
| `HostMesh.stop()` on the retired fleet | Timed out |
| Both loops restarted together (`restart_after_failed_attach`, auto-heal on), then a fresh driver process (ComfyUI restart) | Attached in 0.75 s |

The rule: **restart worker loops together (`dgxm restart`), never one loop
alone while a ComfyUI session is attached.** Any restart with no driver alive
is fine. Recovery is both loops restarting together, then a fresh driver
process.

## 3. Next render hangs at NCCL rendezvous after a crash

**Symptom:** after a driver crash or an interrupted render, the next run's NCCL
init hangs on the master port.

**Cause:** stranded model-holding procs from the crashed client keep the NCCL
master port. dgx-monarch shuts procs down on client death and in try/finally
paths, but a SIGKILLed driver cannot.

A headless validation run exposed the same cleanup failure: its results had
been collected, but Monarch's transport panicked during interpreter shutdown,
and an actor stayed alive. Treat that as teardown residue, not as evidence for
or against the completed render's distributed math. Headless runners must
recycle their client-owned mesh before exit and confirm that no actor remains;
they must never shut down the persistent Worker services.

**Fix:** `dgxm restart`. The `nccl master port` row of `dgxm doctor` warns when
the port accepts connections outside a render.

## 4. `SupervisionError: … process exited with non-zero code`

A worker actor died. The error identifies its endpoint and host. Check the
Worker service log on that host: `~/.local/state/dgx-monarch/worker-loop.log`
for nohup-started Worker services (`dgxm up`), `journalctl --user -u
dgxm-worker` for systemd-managed Worker services. On UMA boxes the usual cause
is OOM; see [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc). The driver
evicts the poisoned mesh automatically, and the next Init or render respawns
the actors. That needs no ComfyUI restart, with two exceptions: the Worker
service died, or the stop after the death timed out and the one attach to the
replacement fleet then fails. Either way that driver process cannot attach
again ([#2](#2-every-attach-times-out-after-one-failed-attach)):
run `dgxm restart` so both Worker services restart together, unless the error
says they already did, then restart ComfyUI.

## 5. Loads fail / OOM on UMA boxes (DGX Spark etc.)

**Symptom:** ComfyUI refuses a load that appears to fit, or the host swaps
heavily during loading. On UMA, the page cache and GPU allocations share memory.

**What to do:** before a large load, use
`sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` and set `reserve_vram_gb`
on Init. `dgxm doctor` warns when page cache crowds available memory. For a
capacity refusal, follow the alternatives in the message: slab residency,
an `*+fsdp` preset, a pruned or quantized checkpoint, or more free host memory.

**`stock residency cannot load...`** is a class C refusal on integrated/UMA
devices. The estimate is 2.1 times the checkpoint size for weights plus their
host placement copy, with a 5 GiB floor to avoid ComfyUI partial loading.
A 60.02 GiB BF16 checkpoint therefore requires about 131.0 GiB. The driver
uses the same placement charge. The previous 1.3x estimate admitted a load
that killed a host; measurements and calibration are in
[VALIDATION.md](VALIDATION.md).

**If `capacity rescue available` appears:** open the DGX Monarch sidebar,
click "Load with slab residency", then queue again. Nothing large has loaded
at the refusal. [#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)
covers headless consent, automatic rescue, and revocation.

A slab-only load has no stock reference on this hardware. Its identity result
stays `INCONCLUSIVE` with `cross_mode=CAPACITY`, even when consent permits the
render. Each load verifies checkpoint bytes
([#54](#54-slab-byte-verify-certificate-mismatch)), and the ledger records the
bypass and capability context. Neither consent nor a byte certificate grants
PASS. You do not need `auto_gate=off` for capacity rescue
([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)).

**If no card appears:** the refusal names the blocker: a non-safetensors
checkpoint, FSDP, `compile_dit`, `lora_low_rss` off, `slab_weights=off`, Gate
quarantine, or insufficient memory even for slab. Comfy-managed residency
forces slab off and cannot offer this rescue
([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)).
A loader `weight_dtype` cast disables this preflight because file size no
longer predicts resident size. Quarantine overrides consent; only a fresh
Identity Gate retest can clear it
([#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass)).

**If quarantine state is unknown:** consent cannot override it. The message
identifies the missing evidence:

| Reported condition | Next step |
|---|---|
| Incomplete artifact identity | Resolve the artifact identity before retrying. |
| Unreadable ledger | Check the ComfyUI output directory on that host. |
| Damage above the newest matching row, or damage not attributable to a matching capability context | Rerun Identity Gate for the combination. An unscoped read cannot order damage against a match; a scoped read may still be clear. |
| Matching row from another protocol, package version, or source build | Rerun Identity Gate on this build. |
| Unrecognized verdict | Rerun Identity Gate for the combination. |

Unknown state is nonwaivable class C, not class K. Only a persisted Gate FAIL
records an observed mismatch.

## 6. cfg2 render errors with "cfg-parallel got a model call of batch=…"

**Symptom:** typed error at render start under a cfg-parallel topology.

**Cause:** ComfyUI runs cond and uncond as separate calls when CFG == 1.0 or
the conditioning shapes are not concatenable, so there is nothing to split.
CFG 1.0 has one exception: a cfg++ sampler, whose name carries `cfg_pp`, sets
`disable_cfg1_optimization` and keeps the unconditional pass, so comfy folds
cond and uncond into one call of batch 2 and both cfg ranks work. That is why
the shipped krea2 template renders under `cfg2` at CFG 1.0 while the flux1
template, on the same preset and the same CFG but asking for `euler`, refuses.

**Fix:** for a family that supports cfg-parallel, raise CFG above 1.0, or keep
CFG 1.0 and pick a `cfg_pp` sampler, and use conditioning that can share one
batched call. On a pure cfg topology,
dgx-monarch applies the family-declared asymmetric-prompt padding contract;
combined USP+cfg presets can still refuse a real mask that the USP kernel
cannot honor. If the family has no batched cond/uncond call, use `auto`, a
documented family- and world-compatible topology whose physical cfg degree is
1, or a true one-GPU run. `uly2` or `ring2` is not a universal remedy.

## 7. Warning: "attention mask … IGNORED" under USP

yunchang/flash USP kernels support no arbitrary attention masks. Padded text
batches and regional prompts may degrade under ulysses/ring. Mask-critical
workflows need a family path that preserves their mask. For Chroma,
use `cfg2` when the graph makes a real batched cond/uncond call, or a true
one-GPU run with `mode=local` and `gpus_per_host=1`. Z-Image's stock Lumina
forward passes `cap_mask` as a hard `None`, so it has no documented
arbitrary-mask-preserving topology. See [#77](#77-a-chroma-omnigen2-or-z-image-usp-render-refuses-a-mask-or-sequence-length) for the typed guards and
family-specific remediation.

## 8. Version skew between hosts

**Symptom:** doctor row `torchmonarch=…; version skew vs driver pin`, a
failing `torch version match: skew across hosts` row, or pickling errors
mid-attach.

**Fix:** For a torch-version mismatch, install the same torch/CUDA build in
the driver and every worker environment; doctor treats the skew as a failure
because serialized objects and distributed kernels cross that boundary.
`dgxm update` uses `--no-deps`, so it does not replace torch.
For torchmonarch skew, run `dgxm update` (synchronize the repo's exact pin with
`--no-deps`, install/sync dgx-monarch with `--no-deps`, restart, then doctor).
This does not resolve or replace torch, NCCL, xfuser, or optional extras.
torchmonarch is pinned; bump it on all hosts at once.

## 9. Worker host death mid-render

In-flight calls fail with `SupervisionError` after ~69 s ("no status received
from process" in the fault-injection test, [VALIDATION.md](VALIDATION.md)); the
healthy hosts stay responsive, and respawning procs on the same mesh works
immediately once the host returns. Restart the render; a dead host also makes
full shutdown block (bounded by internal timeouts).

## 10. First RDMA latent return is slow

This applies only to the future native return path, not the current required
release gate; see [#75](#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails). After native RDMA is requalified, a successful first
`RDMABuffer` read is expected to pay connection/registration (~0.3 s at 64 MiB),
while steady-state reads are milliseconds. That first-touch benchmark guidance
does not authorize retrying a failed or ambiguous native operation.

## 11. Custom sampler renders as euler / "sampler … is not registered on this worker"

**Symptom (0.1.0 and earlier):** a custom sampler picked in the KSampler
dropdown (RES4LYF `res_2s`, `res_3s`, …) renders at euler cost and quality;
multi-stage samplers silently under-compute. **Symptom (current):**
the render fails typed with "sampler … is not registered with ComfyUI on
worker host …", or the SamplerCustom path fails deserializing a SAMPLER
object.

**Cause:** the pack providing the sampler did not import inside the worker
procs, so its names never reached the worker's `KSampler.SAMPLERS`, and
stock comfy substitutes euler for unknown names (dgx-monarch raises
instead). The two known import blockers, both fixed or diagnosable here:
packs that register server routes at import time hit a missing
`PromptServer.instance` in headless actors (workers install an inert stub),
and packs whose python requirements are absent from the worker env
(RES4LYF: `pywavelets`, `opencv-python-headless`, `matplotlib`).

**Fix:** check the Worker service log for `custom node pack … failed to import`.
The message names the missing dependency. Install the pack's requirements
into the worker python (`python` in cluster.toml) on every host (e.g.
`~/monarch-env/bin/pip install pywavelets opencv-python-headless matplotlib`),
`dgxm restart`, re-render. The log line
`preloaded N custom node pack(s)` counts successful imports.

## 12. bf16 / large model load hangs at "0 MB usable" (UMA)

**Symptom:** a large or bf16 model never finishes loading; the Worker service
log shows comfy sizing it at ~0 MB and offloading, and it recurs on every
render rather than being a one-off OOM.

**Cause:** comfy's `get_free_memory` reads `torch.cuda.mem_get_info`, which
under-reports free memory on GB10 unified memory (measured: about
61 GiB reported vs 118 GiB actually free at idle, worse mid-render). The
estimate loader sizes the model at ~0 MB usable and offloads it on every
`prepare_sampling` re-management, so a one-shot force-full-load cannot fix it.

**Fix:** automatic. The worker patches `get_free_memory` at comfy bootstrap
to `psutil.virtual_memory().available` on integrated CUDA devices (host RAM
and VRAM are one physical pool), divided by the number of GPUWorker actors on
the host. bf16 krea2 then renders single-copy. Confirm the log line
`patched get_free_memory for GB10 unified memory`. Discrete GPUs are
untouched and fp8 is unchanged (it always fit the estimate). Still tight?
Drop the page cache too ([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)) so host-available is not crowded.

## 13. uly2+fsdp / ring2+fsdp render deadlocks (hangs, no error)

**Symptom:** an FSDP topology combined with sequence parallelism (uly2+fsdp;
ring2+fsdp is refused) shards the model correctly, then the render hangs
forever with no error raised.

**Cause:** FSDP all-gather runs over the world group while USP's Ulysses
all-to-all runs over the ulysses group: two NCCL communicators on the same
2 ranks. On world=2 their collectives co-schedule and deadlock. This is not
NCCL_PROTO=LL (a separate FSDP-killer; never set it).

**Fix:** automatic. Every worker setup sets NCCL_LAUNCH_ORDER_IMPLICIT=1 before
any communicator exists, whatever the topology (needs NCCL >= 2.26). NCCL reads
the variable once, at a process's first launch, and keeps it, so setting it
only for fsdp with sequence or CFG parallelism leaves a worker first set up
plain unguarded for later fsdp+SP renders; this caused stalls in testing. The env var is the whole guard. The fsdp adapter's empty FSDP2
forward-prefetch list pins a torch default and is no part of the fix
(worker_args `fsdp_prefetch_depth` above 1 fills it: a measured speed
setting, not a guard). The same guard covers `cfg2+fsdp` and `dp2+fsdp`. The
render is then byte-identical to the resident result, sharded to ~1/world
weights per rank, about 2x slower (the measured all-gather cost;
[VALIDATION.md](VALIDATION.md)). Still deadlocking? Confirm NCCL >= 2.26 on every
host and that no fabric profile sets NCCL_PROTO.

## 14. First render pauses ~9 s with `compile_dit` on

**Symptom:** with the Init `compile_dit` widget enabled, the first render
pauses (looks hung) before the first step; later renders are fast.

**Cause:** `max-autotune-no-cudagraphs` compiles each DiT block on its first
forward, causing a one-time inductor autotune warmup (about +9 s measured for
the fp8 Triton GEMM search; [VALIDATION.md](VALIDATION.md)). Expected, not a
hang.

**Fix:** let it finish once; the compiled blocks persist for the actor's life
(later renders and LoRA hot-swaps reuse them). It is opt-in and per-model:
the worker fails open to eager if a block cannot compile (log
`DiT block compile skipped (...); running eager`), and a successful pass logs
`DiT: N blocks compiled`. Turn the widget off to skip warmup entirely (the
compile flag is cleared on the next setup).

**Which models it compiles:** only a diffusion model with a `blocks` list.
krea2 is the one measured; other `blocks` families (wan, minimax) compile
without a fidelity record. Flux, Flux2, Chroma, and LongCat name their lists
otherwise, so they compile nothing and the worker logs
`compile_dit: ... has no blocks list, so nothing was compiled; running eager`.
For a checkpoint whose live-model family has already been detected, those
known no-op families leave requested slab residency available. Other families
with differently named lists stay stock until their compile surface is
covered. The first load of unfamiliar bytes also stays stock: the
worker never uses a header guess to combine slab residency with a potentially
compilable model.

## 15. Attaches still time out after `dgxm restart` (driver-side wedge)

**Symptom:** Worker services freshly restarted, `dgxm doctor` fully green, yet every
attach from ComfyUI still hits `MESH_ATTACH_CONFIG_TIMEOUT`, and a worker may
die hard (supervision: "no status received") on the first load after restart.

**Cause:** the long-lived driver ComfyUI process is holding poisoned Monarch
client state from an earlier crashed mesh. The worker-side self-heal
(defunct-mesh eviction) covers actor deaths, but a client whose transport state
was corrupted by successive crashes can keep emitting attaches that cannot
succeed, which then wedge the fresh Worker services again ([#2](#2-every-attach-times-out-after-one-failed-attach)).

**Fix:** restart the driver ComfyUI process as well as the Worker services:
`dgxm restart`, then kill and relaunch the comfy process (by PID or
`fuser -k <port>/tcp`; never an inline `pkill -f "main.py …"`, which
self-matches the shell issuing it). A fresh driver and Worker services attach cleanly.

**After a reply-channel timeout:** restored connectivity may leave the driver
unusable even when both Worker services remain healthy. In the reply-channel
fault test, Monarch closed the client reply channel after its 30-s
acknowledgement-delivery timeout. Collection timed out at its 10-s deadline,
but both reattach attempts in the same process timed out. After confirmed
owned-actor cleanup, a fresh driver completed two render and cleanup cycles
without restarting either Worker. Restart ComfyUI or the headless driver for
this case; do not treat traffic restoration as proof that the existing client
can be reused. The [test scope and failed same-process
recovery](VALIDATION.md#reply-channel-fault) are recorded separately from
successful recovery in the new process.

## 16. `single` preset refuses because the latent batch does not divide dp

**Symptom:** topology `single` on a multi-rank cluster rejects a batch-1
render, saying the preset leaves dp2 but latent batch 1 is not divisible by 2.

**Cause:** explicit presets tile the whole world: `single` at world=2 derives
dp=2 (each rank renders a batch slice), which needs `batch % 2 == 0`. Only
`auto` folds spare ranks into ring for batch-1 renders; explicit presets are
always interpreted literally.

**Fix:** use batch size 2 (two images, one per rank), or Init `mode=local` +
`gpus_per_host=1` for a true one-GPU render, or leave topology on `auto`.

The same refusal also appears at the loader node, before anything loads,
whenever the graph states a literal batch on an empty-latent node ComfyUI
ships, including the Flux 2, Chroma Radiance, Hunyuan Image and HunyuanVideo
1.5 latents. It is class P at both places, with one message: no consent and no
extra memory changes what a batch of 1 can be split into, so no card appears,
and a graph that is both over budget and batch indivisible is told about the
batch rather than offered a rescue it cannot use. A batch wired from another
node, or one the loader cannot read as a literal, gets its answer at render
submit from the real tensor. The MiniMax H3 audio+video latent states no batch
at all; the packed-latent rule refuses its packed shape above dp1 instead.

## 17. FSDP load refused: LoRA or non-bf16 checkpoint

**Symptom:** an FSDP preset (`uly2+fsdp`, `cfg2+fsdp`, `dp2+fsdp`) refuses
checkpoint precision or LoRA loading. A changing checkpoint may instead raise
`ArtifactBindingError` after the driver header preflight.

**What to do:** use a stable bf16, fp16, fp8, or int8 safetensors checkpoint.
For LoRAs, enable `lora_low_rss` and use bf16, fp16, or plain-dtype fp8 without
a comfy-kitchen wrapper. Otherwise use a resident topology (`uly2`, `cfg2`,
or `ring2`). These are class P refusals, with no waiver.

| Refusal | Meaning and remedy |
|---|---|
| `LoRA on FSDP-wrapped models needs lora_low_rss on` | Enable `lora_low_rss`. The shard-aware bake reads pristine checkpoint weights and retains each rank's dim-0 chunk; ComfyUI's load/backup path must not touch those shards. |
| `LoRA on an FSDP-wrapped model is not admitted for this checkpoint` | Comfy-kitchen scaled fp8, fp8mixed, and int8 (with or without convrot) are unsupported for this bake. Neither `lora_low_rss` nor `auto_gate` can admit them. A plain `float8_e4m3fn` file, such as Qwen-Image, is different and can be admitted. |
| `FSDP capacity mode supports bf16, fp16, fp8 and int8 checkpoints at launch (got quant=…)` | Use an admitted source precision. mxfp8/nvfp4 lack registered shard layouts; `weight_dtype=fp8_*` casts lack file-backed shard bytes. |
| `ArtifactBindingError` | Finish downloads, syncs, renames, and symlink changes on every host. Confirm the logical model name resolves to the intended stable file, then reselect or rerun the loader. |

If ComfyUI's default turns a BF16 source into FP16, select explicit `bf16` in
the DGX Monarch UNET loader. This changes live dtype only; it cannot convert an
FP16/FP32 source into valid BF16 evidence or bypass the header check.

**Precision limits:** uniform bf16/fp16 safetensors are admitted. A bf16 core
may include fp32 islands totaling at most 5% of file bytes; these are
replicated, with `quant=bf16` and
`live_dtype_profile=bf16_core_fp32_islands_v1`, plus island count and bytes.
Wan's exact small FP32 input convolution has its separate
`wan_fp32_patch_embedding_v1` profile, audited only with a bf16 core. The fp16
Wan pair with fp32 ingress layers must use resident topology or a bf16 source.
Above the 5% ceiling, mixed fp16/fp32 and all-fp32 files report `quant=fp32`
and refuse. Unrecognized precision or custom operation/quant wrappers report
`quant=unknown`. Legacy `.ckpt`/`.bin` cannot prove source precision; obtain or
convert a readable safetensors file. An explicit bf16 cast cannot admit them.

**LoRA checks can also fail during baking.** Quantized-key detection uses
`scaled_fp8`, I8 tensors, `*.comfy_quant`, `_quantization_metadata` in
`__metadata__`, or a `*.weight_scale` sidecar beside F8/I8 weights. Missed
markers are checked again by `capture_record`/`bake_key` in
`actor/fsdp_lora.py`.

FP32 islands need live/file dtype agreement too, which the header and coarse
profile cannot establish. Krea2 RAW bf16 casts its island to bf16 and fails
`_file_tensor`; ChromaRadiance's two islands remained fp32 in the measured
base FSDP census ([VALIDATION.md](VALIDATION.md#evidence-radiance)). The per-key
check raises a rank-symmetric, class P `UnbakeError` when agreement fails.

**Checkpoint and cleanup ownership:** the driver binds the descriptor to
device, inode, size, mtime, ctime, and the `diffusion_models` resolver result,
then re-resolves before any cache reuse, mesh lifecycle, setup, or load RPC.
Replacement, retargeting, disappearance, metadata drift, or remapping refuses.
Each worker independently checks the same identity before and after ComfyUI
loads it.

An aborted FSDP Gate reload, incomplete rank evidence, failed A/B comparison,
or rejected provisional PASS unloads all ranks exactly once. Cleanup requires
exactly one positive `unloaded` response per expected rank. Otherwise the mesh
becomes DIRTY; failure to publish DIRTY makes the handle defunct. Recycle the
client-owned mesh before retrying.

## 18. First LoRA change on a quant model is one-time slow (HISTORICAL)

**Symptom (older builds):** the first int8/fp8 LoRA change was 2-4x slower,
with a large `gpu_load` phase such as
`sample 117.44s (hot-swap gpu_load 67.4s)`. Later swaps returned to about 10-20 s.
This was quant-kernel warmup on the reload, not the swap itself, a leak, or
model eviction (`disable_smart_memory` defaults on).

**What to do:** update dgx-monarch. Retesting with slab weights, lazy un-bake,
and ComfyUI nightly found flat int8 swap times (10.9/9.7/9.7 s), with only the
initial bake still slow. If the first-swap slowdown returns, enable Init's
`load_profile`: it profiles each `load_models_gpu` call lasting at least 2 s
into a private, exclusively created
`/tmp/dgxm-load-profile.<pid>.<t>.<random>.txt`.

A large `gpu_load` on *every* render indicates eviction instead. Check
`uma_reserve_gb` pressure and co-resident workloads.

## 19. Blank ComfyUI canvas (grid renders, no menu/sidebar) after an update

**Symptom:** ComfyUI serves, the litegraph grid and FPS counter render, but
the Vue UI never mounts: no menu, no sidebar, stacked error toasts, a
full-screen block overlay. The browser console shows vue-i18n/vite errors
inside the frontend bundles. dgx-monarch's routes and extensions all answer
200, so the only doctor row that sees the fault is `frontend/backend skew`.

**Cause:** frontend/backend version skew. The served ComfyUI frontend is
newer than what the backend declares in `required_frontend_version`. The
usual trigger is launching with `--front-end-version
Comfy-Org/ComfyUI_frontend@latest`, which re-downloads the newest published
frontend on every launch: it works until the newest frontend is newer than the
version the backend declares, then the canvas goes blank with no change on
your side.

**Fix:** run `dgxm doctor` with the driver up; the `frontend/backend skew`
row names both versions and the fix: pin the launcher flag to the version
the backend requires (`--front-end-version Comfy-Org/ComfyUI_frontend@X.Y.Z`)
or update the ComfyUI checkout so the backend targets the newer frontend.
Pin the frontend version in persistent launchers; doctor warns on `@latest`
even while the versions match.

## 20. "value... is not available" on a DGX Monarch node after an update

**Symptom:** a workflow saved before an update fails to queue with errors
like `The value "first_use" for DGX Monarch Init's slab_weights is not
available`, with values that belong to a different widget on the same node.

**Cause:** ComfyUI stores a node's widget values by position, and a node
update inserted a widget mid-list, shifting every saved value after the
insertion point. Combo widgets reject the shifted value with an error;
numeric and boolean widgets accept it silently, so the whole node can be
misconfigured, not just the flagged rows.

**Fix:** update dgx-monarch and reload the workflow. Every DGX Monarch node
also saves a name-keyed value map, and on load the map wins, so a later
reorder cannot shift those saves. For an Init node saved before the map
existed, the frontend extension realigns the values from the Init widget-order
history; a toast reports the migration, and re-saving the workflow keeps it.
If a workflow still trips it (another node, or a layout that history does not
list), recreate the node: add a fresh one and rewire it, set your values, and
re-save. A right-click `Fix node (recreate)` item does the same and keeps the
links, but an optional extension adds it; ComfyUI's own frontend package has
none.

<a id="21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid"></a>

## 21. "sequence length needs divisibility padding... ulysses-only" on ring/hybrid

A render's token count does not divide the sequence-parallel degree, so
`shard_seq` must pad, and the exact pad exclusion exists only on pure-ulysses
topologies. Ring attention would attend the synthetic pad rows (measured
left-edge corruption), so the render refuses instead. The count is the joint
image and text count for Krea2 or Lens, the packed audio, video and text total
for MiniMax H3 (see [#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)), and either of the two streams for LTX. Mage Flow and
the flux family refuse here too. [ADAPTERS.md](ADAPTERS.md) names the
families that pass their pad rows as `drop_rows` on every topology, and so
refuse here, and those that pass them under pure Ulysses only and attend them
on ring without refusing; other families keep their own measured contracts.
The text side counts too: the Krea2 TE strips its chat template, and Lens's
GPT-OSS TE also produces small, prompt-dependent context lengths, so either
family's text stream can need padding on its own; cfg_pp samplers evaluate the
negative prompt even at cfg 1.0 (an empty negative is 5 tokens).
`auto` picks from the rule table, whose rows carry different evidence
levels; [MODELS.md](MODELS.md) states which are hardware-tested.
At world 4 with batch 1, `auto` folds a uly2 row's spare ranks into ring
(uly2+ring2) until uly4 passes a 4-rank acceptance gate; the explicit `uly4`
preset is available as an operator choice there.
Fixes, any one of: use a `uly*` or `single` topology at this resolution; pick
a resolution whose token grid divides the SP degree; adjust prompt/negative
lengths to divisible counts. On MiniMax H3 the `single` preset is not one of
the fixes: above world 1 it derives dp, and H3's packed latent refuses dp
([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)). Use `uly2` there, or `mode=local` with `gpus_per_host=1` for a true
single-GPU reference. H3 says so in its own wording, because the adapter reads
the ring degree itself before the first block runs. LTX reads it the same way,
and either of its streams can trip this alone: video rows go odd when the
latent frame count and both latent canvas edges are odd, and audio rows are
odd for about half of all clip length and frame rate pairs, so the fix there is
usually a frame count or a rate rather than a canvas.

**An expert waiver exists, and it is off by default.** This refusal is class K,
incorrect output: the synthetic pad row produced left-edge artifacts on Krea2,
and the same mechanism reaches every family that hits this guard. If you
intentionally need that math (a comparison leg, a bug repro, an image you
accept), the DGX Monarch panel raises a card in the accuracy style whose one
button reads "Render under waiver (output stamped)"; headless, set
`DGXM_WAIVE_KNOWN_WRONG=ring_pad` on the driver process. The card states the
measured error before you authorize the bypass. Every render that then goes
through is stamped rendered-under-waiver and writes a permanent WAIVER record
to the gate ledger, and no accuracy waiver is ever granted automatically. Full
flow, stamp and revoke in
[#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means).

## 22. "a previous model load could not be cleaned up" / model reuse is blocked

**Symptom:** after any model load fails, every later load or reuse on that worker
refuses with `a previous model load could not be cleaned up`, or an explicit
unload reports `failed-load cleanup remains blocked`. Worker status shows
`failed_load_cleanup_pending: true`. `retained_failed_load_slabs` is nonzero
only when the failed load owns a slab. `retained_failed_load_resources`
separately counts pins and other non-slab owners; both counts can remain zero
when cleanup is poisoned without a concrete retained resource.

**Cause:** global model unload, cache cleanup, collection, or closing an owned
load resource was interrupted or failed. A slab-backed load can leave tensors
whose Parameters point directly into its mapping; a stock/FSDP load can leave
uncertain Comfy model state or a checkpoint pin that Comfy may still use. The
worker retains any concrete resource and latches cleanup poison even when no
resource exists, then refuses all further model work. This prevents a use after
close, or reuse of process state whose cleanup was never confirmed. The original
load error remains primary; the driver log records the cleanup failure
separately.

The ownership boundary begins before acquisition, not only after a load throws.
One process-rooted fresh-load transaction owns each pin or slab through the
interruptible Comfy load and records the exact model-store slot candidate before
publication. It releases those children only after that exact object is
confirmed in that exact slot, so recovery cannot close a resource under an
already-published resident. A slab load also prepublishes restoration of both
temporary process-global Comfy hooks. If either hook cannot be restored to its
exact original object, the hook guard and slab remain rooted and arena close is
blocked until a later cleanup confirms both restorations.

A low-RSS LoRA lazy swap is one form of the same ownership problem. It restores
and re-bakes checkpoint weights in place, so a cancellation, native panic, or
other non-`Exception` interruption can leave a resident only partly changed.
The worker discards that slot before it propagates the interruption, and the
next request does a canonical load. If the discard is also interrupted, the
slot reports `a previous unload failed` and remains blocked; old request
metadata never makes that resident reusable.

There are two retry classes. Failure during process-wide unload/cache cleanup,
FSDP alias unlink, or `mmap.close()` with a live exported view occurs before
numeric descriptor release; the worker still owns the original FD or mapping,
so a later explicit global unload may retry it. Once an arena, checkpoint pin,
or temporary slab checkpoint-source owner enters `close(2)`, any exception is
ambiguous: the kernel may already have released and reused that numeric FD. The
worker marks the owner uncertain before the syscall, never retries that number,
and keeps cleanup poisoned. A slab attempts both its primary and optional annex
arenas, but either arena becoming uncertain makes the slab unusable and
retained.

**Fix:** if the worker log already says a checkpoint, slab-source, or slab-arena
descriptor close outcome is uncertain, skip in-process cleanup retries and use
**Reset attached mesh** at once. Otherwise, first run an explicit all-model
unload with a DGX Monarch ClearVRAM node at level `hard`. A successful
process-wide unload/cache/collection retry closes retry-safe retained resources
and clears the poison. If it still reports
`failed-load cleanup remains blocked`, use the sidebar **Reset attached mesh**
button: it stops the mesh-owned actor processes without unloading their models
first, because process exit frees them, and the next render attaches fresh
actors. Do not keep retrying renders against the blocked actor: only an
actor-process reset can safely discard an uncertain descriptor table. If the
reset itself reports incomplete, leave the mesh blocked and inspect the driver
and Worker service logs before using `dgxm restart` as the deployment-level
recovery.

## 23. Status or dashboard polling seemed to destabilize Worker services

**Symptom:** after running an older `dgxm status`, `dgxm doctor`, auto-heal, or
`dgxm top`, a later attach times out even though the worker port still appears
to listen.

**Cause:** older builds used protocol-blind TCP connect/disconnect readiness
probes against Monarch worker endpoints. Monarch owns connection lifecycle, so
a bare TCP handshake is not a valid health request. The failure is
state/order-dependent rather than a guaranteed result of every connection.

**Fix:** update dgx-monarch, then run `dgxm restart` once if the old probe
already left a Worker service unusable. Lifecycle checks verify the exact
process and read the host's passive LISTEN table through the local/SSH runner.
`dgxm top` uses driver telemetry and shows `?` when Worker service state is
unavailable; it never connects to worker ports or SSHes to every host on its
polling interval.

## 24. An older worker log contains inherited environment values

**Symptom:** a Monarch launch diagnostic in `worker-loop.log` or its runtime
scratch log includes values inherited from the shell or service that started
the loop.

**Cause:** older dgx-monarch launches passed the worker loop's ambient
environment into the pinned runtime, whose native process-launch diagnostic
can render child launch options. Older nohup logs also followed the caller's
umask.

**Fix:** treat any credential present in an old log as exposed and rotate it;
permissions or deletion do not revoke it. Update dgx-monarch and run
`dgxm restart`. Worker services replace ambient inheritance with a small exact
allowlist before Monarch imports, take fabric settings only from validated
`cluster.toml`, route runtime scratch into an owned `0700` directory, and open
nohup/systemd logs under `0077`. Do not add broad `PassEnvironment=` entries to
the generated service.

## 25. CI reports `leak-check[...]` for a tracked file

**Symptom:** the leak-assertion CI step fails with a repository-relative
file, line, column, and rule, but does not print the suspected value.

**Cause:** a Git-tracked working-tree file or filename matches a private
identifier rule (such as a home path, a rig-style hostname or a private network
address) or a credential-shaped value. The diagnostic omits the value so CI
logs do not turn a local mistake into a second disclosure.

**Fix:** run `python tools/leak_check.py` locally and inspect the named location.
Remove or replace the private value; rotate it if it was a real credential.
Do not add a credential or rig identifier to the allowlist: the code accepts
allowances only for the fictional `GPU-deadbeef` / `examplebox-*` fixture
namespaces, each pinned to an exact file and match count. Write example
addresses from `192.0.2.0/24`, which the address rule does not match.

## 26. Comfy canary reports `Comfy touchpoint missing` or `incompatible`

**Symptom:** the scheduled/PR ComfyUI-master job, or a local run of
`tests/canary/comfy_entrypoint_canary.py`, exits with a dotted Comfy path and
possibly a missing signature parameter.

**Cause:** the Comfy checkout no longer exposes an internal API that
dgx-monarch imports, patches, or calls. The manifest checks these before a
model load, so upstream drift fails at the named contract instead of later as
a worker crash or a silent sampler substitution.
For rebound forwards, the surface assertion imports every declared target and
compares its exact parameter names, kinds, required/optional status, and
variadics; drift is incompatible even if the replacement has `**kwargs` and
Python would accept it silently.

**Fix:** first reproduce against the same Comfy commit with
`COMFYUI_DIR=/path/to/ComfyUI python tests/canary/comfy_entrypoint_canary.py`.
If the two hosts run different Comfy revisions, align them and retry. If master
changed, update the affected dgx-monarch call site and its
`src/dgx_monarch/comfy_surface.py` or rebound-contract declaration together,
run the full CPU and Comfy canary battery, then leave hardware render/identity
validation to the rig owner. Do not delete or relax the assertion without
removing the corresponding production dependency; the repository AST inventory
test also refuses an undeclared `Adapter.bind`/`self.bind` site or a declaration
with no matching bind.

## 27. torchmonarch canary reports compatibility or semantic drift

**Symptom:** the weekly/manual `torchmonarch-canary` workflow fails its final
gate. Its `torchmonarch-surface-*` artifact contains exact-pin and latest-stable
JSON snapshots, install logs, comparison JSON, and Markdown reports. A failure
names a missing/incompatible dotted Monarch surface or an event-loop semantic
change.

**Cause:** either the installed exact pin no longer matches the checked pin
snapshot, latest stable changed a production call shape, or raw `PythonTask` /
public `Future.get()` active-asyncio behavior changed. The workflow treats
semantic drift as review-required even when it might be an improvement,
because dgx-monarch's off-loop and bounded-wait paths are built on the old
behavior.

**Fix:** inspect `checked-pin-report.md` first. If it fails, reproduce the exact
pin before trusting the latest comparison. Then inspect
`latest-vs-pin-report.md` and the two snapshots; annotation-only signature
changes are informational, while missing call shapes and semantic changes are
blockers. Adapt the production caller and manifest together, run the full CPU
suite plus real Monarch/hardware gates appropriate to the affected surface,
and only then bump every exact-pin mirror by hand, with a changelog entry.
The canary never auto-bumps or writes to the repository.

## 28. Operator-armed setup-kill handoff does not reach PASS

Setup-failure tests use reviewed private scripts under the manual validation
contract in [THREAT_MODEL.md](THREAT_MODEL.md). CI never runs them.

**Symptom:** an authorized setup-failure test does not complete. It may report an unwitnessed TCPStore
listener, an untyped worker death, an exceeded shared bound,
`replacement_blocked`, a reused PID, failed world-1 setup, or an unconfirmed
stop of the recovered ProcMesh.

**Cause:** the probe kills its one client-owned worker during a real
incomplete NCCL setup. Every later assertion is a separate part of the
lifecycle contract; a healthy respawn does not excuse unconfirmed ownership or
cleanup, and a generic timeout is not typed supervision evidence.

**Fix:** establish the outcome of the previous run and confirm cleanup of its
owned processes before starting another test. Verify the exact reviewed source
on every host and obtain authorization for the destructive test. Match the failing
phase to its action: verify the test environment's Comfy/CUDA/xFuser install for a
missing setup witness, inspect production supervision and teardown for
typed-failure/retirement failures, and restart the driver only after process
ownership is known when `replacement_blocked` appears. Never
relax the PID ownership, typed exception, fresh-PID, timeout, or recovered-stop
assertions. A passing result requires every phase and confirmed cleanup of the
test's recovered ProcMesh. It does not authorize stopping a persistent Worker
service or qualify model fidelity.

## 29. Large RDMA latent return fails after workers finish rendering

**Symptom:** a large-latent prompt fails on the driver after workers have
finished and may already report matching latent signatures. The driver log
mentions a Monarch PythonTask or `Future.get()` being used while an asyncio
event loop is active; an attempted RDMA registration release may report the
same refusal. Small latents can appear unaffected because they stay below the
RDMA threshold and return through actor messaging.

**Cause:** older driver-side materialization called a blocking Monarch read (and
then blocking release cleanup) from ComfyUI's active event loop. Current
dgx-monarch runs the complete local read/drop and descriptor ACK transaction in
one bounded scratch job. Its join envelope contains a 120 s native-read budget,
a 15 s margin on the future get, one 120 s drop budget per descriptor part, up
to 2 ACK broadcast attempts with a 120 s budget per attempt, and a final 30 s
outer join margin. The generation lease is published before the thread handoff,
so a delayed target or caller timeout cannot authorize recycle over work that
may still start.

On the worker, each production descriptor and its CPU backing are owned under
the exact setup generation and a random 128-bit token. The registry is capped at
pipeline depth; saturation uses actor messaging and never evicts an unACKed
entry. A successful driver read does not by itself free that backing. After a
completed read, or an off-loop scratch proof that the native operation never
started, the driver must confirm every local native drop and read-authority exit
before broadcasting an ACK. The response set must contain exactly the configured
worker count, exactly one
`released`/`already_released` owner, and `unknown` everywhere else; rank numbers
are not used as ownership identity. Only that validated ACK retires worker
registry ownership in-process.

**Fix:** update dgx-monarch consistently, restart the ComfyUI driver so it
imports the updated source, and verify the configured hosts with `dgxm doctor`
before retrying. If an update cannot be deployed immediately, set
`rdma_latent_return = false` in the cluster configuration and restart the
driver to use the slower actor-message return path. Do not infer a model or
distributed-math failure from this transport error; verify the per-rank latent
signatures separately.

**If a started native read is incomplete.** dgx-monarch performs zero native
drops and sends zero ACK; it retains the destination and every part because the
operation may still be live. If a read completed, or an off-loop contender
proved that it never started, but a failed or ambiguous drop occurs, that handle
is not retried and the driver again sends zero ACK. `RDMA registration ownership
POISONED` identifies this local process-lifetime ownership. Constructor
ambiguity on the worker is stricter: production RDMA is refused with
`RDMA registration ownership is ambiguous; Attached mesh reset required` rather
than treating the generation as reusable; the operator action is
**Reset attached mesh**.

**If both scratch threads fail to start.** When both prebuilt scratch threads
fail to start before either claims the job while ComfyUI's event loop is
active, dgx-monarch performs zero native reads, drops, or ACKs on that caller
thread. The bounded `PENDING` job, generation lease token, descriptor parts,
and worker backing remain rooted; lifecycle and recycle refuse the handle. In
that state, restart the ComfyUI driver process. A ProcMesh recycle is not
enough, because the driver-local scratch-job authority is what could not be
transferred.

**If an ACK response is lost.** An ACK response can be lost after the worker
applied it, so retries repeat only the idempotent generation/token ACK. A
partial or malformed response set, zero or multiple visible owners, or two
unconfirmed ACK attempts fails collection and abandons/blocks the generation;
do not infer whether remote backing survived an ambiguous response. Multi-DP
collection still drains every unique leader token before it raises the
strongest failure. Operational cancellation outranks ordinary errors; within
the same class, the first error remains primary and later failures are
diagnostic notes.

Preserve that selected registration/read/ACK error as the diagnosis and leave
`rdma_latent_return = false` while recovering. Production does not recycle
automatically. After pending work is retired, explicitly recycle the
client-owned ProcMesh with the DGX Monarch panel's **Reset attached mesh**
action or the authenticated `/dgxm/recycle` endpoint. Restart the driver only
for driver-local process poison or when the
process-stop outcome cannot be confirmed. Do not force-clear a retained owner
or retry a numeric native handle in place.

## 30. Sampler Custom fails before model loading with `ModuleNotFoundError`

**Symptom:** a workflow using stock `RandomNoise` + `KSamplerSelect` into
`DGXMonarchSamplerCustom` fails before any checkpoint load. The exception names
`comfy_extras/nodes_custom_sampler` as a module that cannot be imported on the
worker.

**Cause:** current ComfyUI loads built-in extra-node files with a source-path
module identity. Cloudpickle can serialize an object while that path key is
present in the driver, but a separate worker cannot import the path as a Python
module.

**Fix:** update dgx-monarch on every host and restart the driver/workers. Stock
RandomNoise/DisableNoise objects are then rebuilt under Comfy's canonical
importable module before the existing by-reference transport. Samplers and
third-party NOISE objects keep their prior behavior; use [#11](#11-custom-sampler-renders-as-euler--sampler--is-not-registered-on-this-worker) when a custom pack
itself fails to deserialize on workers.

## 31. Kandinsky5 output does not match the official workflow

**Symptom:** a Kandinsky5 render completes, but its denoising trajectory or
output differs from the official Image/Video workflow even though the seed,
steps, scheduler, conditioning, and checkpoint match. Video is the common case:
the current checkpoint default reports SD3 sampling shift 10 while the official
workflow patches it to shift 5.

**Cause:** the SD3 sampling shift is render-time model math, not a checkpoint
loader option. A plain DGX Monarch model handle therefore keeps Comfy's
checkpoint default unless the graph applies the sampling node. The shift does
not belong in loader options, where it would change model residency, and
patching the shared resident in place could leak it into later warm renders.

**Fix:** place `DGXMonarchModelSamplingSD3` after the DGX Monarch model/LoRA
chain and before the sampler. [MODELS.md](MODELS.md) carries the per-variant
shift values and what the patch does and does not touch. The Kandinsky5
validation runs checked the shift lifecycle on a reused resident model,
certified Ulysses/ring/cfg2/FSDP raw-latent fidelity, and ran the full official
Image/Lite/Pro settings. If a render still differs, first verify that the
sampling node is present and connected after every model/LoRA branch.

## 32. Hunyuan Image 2.1 refiner is auto-detected as LongCat

**Symptom:** Init with topology `auto` selects a LongCat policy row for a
Hunyuan Image 2.1 refiner checkpoint, even though the matching base checkpoint
selects Hunyuan.

**Cause:** the public refiner export wraps tensor names in `model.model.` and
identifies the variant with TokenRefiner plus `time_r_in` keys. Older
dgx-monarch releases lacked the exact nested-wrapper normalization and knew
only the Hunyuan Image base and Video 1.5 markers. Without the refiner marker,
the shared Flux-style block names fell through to the broader LongCat
signature.

**Fix:** update dgx-monarch on every host and restart the driver/workers. Auto
then normalizes the exact nested ComfyUI wrapper and recognizes the refiner
signature before Flux/LongCat. Until the update is deployed, choose an
explicit supported topology such as `uly2` for this refiner instead of
`auto`; this workaround does not certify hardware fidelity by itself.

## 33. A live topology or residency-policy change times out while retiring a large resident

**Symptom:** a warm render completes, then changing distributed topology or
switching the same topology to a different residency policy eventually raises
`TopologyTransitionError`. A topology replacement logs `topology change:
tearing down old NCCL groups on live actors`; a policy-only change retires the
resident without rebuilding groups. The error reports the setup generation and
a cleanup outcome such as `timeout_unknown`; later renders or setup attempts on
that fleet are refused until it is recycled.

**Cause:** injected models and attention dispatchers belong to one process-group
generation, so a topology change must unload them before destroying and
rebuilding the groups. A policy-only change must also unload models built under
the prior residency knobs. Older releases put either retirement behind an
opaque 120-second wait. Large residents could take longer to unload than that timeout. The driver
could time out before group destruction began while the worker operation
continued in the background. The old explicit teardown path also kept
its local setup key published until every destructive stage succeeded.

**Fix:** update dgx-monarch on every host. Current releases consume the next
setup generation and retire setup identity before cleanup, allow a bounded 600
seconds for resident retirement on both policy-only and topology-replacement
paths, and require every rank to acknowledge an `UNSETUP` state before
replacement setup starts. Outstanding sample results also keep their
generation through raw-latent materialization, every local RDMA drop, and the
descriptor-owner ACK ([#29](#29-large-rdma-latent-return-fails-after-workers-finish-rendering)); an incomplete read, ambiguous drop, or unresolved
ACK abandons and blocks that generation until an explicit ProcMesh recycle.
dgx-monarch treats a collection timeout as terminal: it requests cancellation
and abandons the unread result, so it cannot be collected later. A raw read
already running keeps the active lease until its scratch thread exits, and
that abandoned work needs a recycle before a topology change.
If a typed DIRTY result still occurs, recycle the client-owned mesh with the
DGX Monarch panel's **Reset attached mesh** action or the authenticated
`/dgxm/recycle` endpoint and confirm cleanup. A confirmed recycle permits a
retry in the same driver. Use a fresh driver process only when transport is
poisoned ([#2](#2-every-attach-times-out-after-one-failed-attach)) or the process-stop outcome is unconfirmed ([#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)).
Do not queue more renders on the DIRTY handle, and never restart Worker services
underneath a live driver mesh.

## 34. A render is refused because another render session owns the mesh

**Symptom:** a direct render, Pipeline, Fleet, Identity Gate, loader, scheduler,
or ClearVRAM operation raises `ConcurrentRenderSessionError` saying another
render session did not drain within the bounded timeout. If an earlier
residency RPC timed out or its dispatch result was lost, the refusal may instead
be a `TopologyTransitionError` reporting a DIRTY `... RPC completion` phase.

**Cause:** one mesh has one setup and model-residency state, and each live
sample owns setup-generation, descriptor-read, and ACK authority. Independent
logical sessions cannot safely change topology or worker policy, or unload
backing while another result/RDMA transaction or Identity Gate check is still
live. `DGXM_RENDER_SESSION_TIMEOUT` bounds both handle-lock acquisition and
owner draining; increasing it only changes how long a contender waits. A
timed-out or return-lost mutating RPC has unknown remote completion, so the
handle latches DIRTY rather than let a new session overlap it.

**Fix:** let the current queue, Gate, Fleet wave, or PendingRender finish; do not
bypass the owner or run two independent queues against the same mesh. If the
error identifies a DIRTY RPC-completion phase, use the authenticated DGX
Monarch **Reset attached mesh** action and confirm the actor processes stopped
before retrying.
Only classified supervision reconciliation can bypass session ownership; the
public lifecycle API cannot force it. A confirmed recycle permits a fresh setup
in the same driver, while an unconfirmed process-stop outcome still requires
operator verification as described in [#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

The panel's **Reset attached mesh** action and its internal `/dgxm/recycle`
route preserve that rule: a reset requested while a session or sample lease is
active stops nothing and returns a retryable busy result, so wait for the owner
to drain, then retry. See [#78](#78-the-dashboard-records-unknown-or-mesh-reset-says-busy-or-outcome-unknown) for a reset that reports busy or an unknown
outcome, and [#69](#69-worker-bring-up-is-interrupted-during-spawn-rollback-or-client-lease-startup) for when a failed stop allows another and why a timed-out one
never does.

## 35. Identity Gate reports `ERROR`/`INCONCLUSIVE` after a prior PASS

**Symptom:** an ordinary combination that previously passed is forced to stock
residency, an FSDP request raises a typed clean-reload refusal, or the ledger
contains a `RETESTING` row and no newer final verdict. Logs may also say that
the Gate ledger is unreadable, damaged, or not writable.

**Cause:** positive Gate authority fails closed. Before a retest starts, the
Gate writes one durable `RETESTING` transaction, a revocation that survives a
crash. An ordinary residency check revokes its normal, Fleet, and eligible
explicit-slab contexts; a no-LoRA FSDP check revokes only its separately
scoped normal-render context. A killed check or failed final append leaves
that denial in force. Likewise, an invalid-schema or unreadable row newer than
an older PASS may have been a revocation, so the older PASS is not resurrected.
A valid exact row written after earlier damage heals only that context;
parseable FAIL quarantine remains an availability block until a complete
explicit retest from the current source records a newer `PASS` in that same
exact tested configuration. A PASS in another context cannot clear it. A
later `INCONCLUSIVE` or `RETESTING` row does not supersede that compatible
FAIL, and a foreign-source terminal row cannot mask it. Without a governing
FAIL, either status still denies. Contextual `PASS`, `INCONCLUSIVE`, and
`RETESTING` rows, and the process, normal-render, and Fleet tokens, bind the
package-source manifest cached on first use in that process. A missing or
different source identity is stale and authorizes nothing; a sticky FAIL
quarantine does not depend on the source.

**Fix:** correct output-directory ownership/storage errors first; do not delete
the ledger to regain optimized residency. Run the explicit Identity Gate again
for the exact model/LoRA/config/topology context and let it finish. A
current-source PASS clears the older FAIL boundary only for that exact context.
If the checkpoint or LoRA bytes, ComfyUI commit, package version or source
manifest, protocol, setup, or physical configuration changed, a fresh proof is
required. Restart every process from the same source before re-gating. Ordinary
failures remain stock; FSDP failures must complete all-rank unload (or recycle
a DIRTY/defunct handle) before another FSDP attempt.

## 36. Packed audio/video sampling fails after denoising

**Symptom:** an LTX audio/video workflow finishes denoising and then raises on
`.detach()`, `.cpu()`, `torch.equal`, `.abs()`, or latent-shape access. A
custom sampler may instead return a flat or missing second output. Older Gate
builds can appear to inspect the video-shaped first modality without proving
the audio modality.

**Cause:** ComfyUI represents packed audio/video samples as `NestedTensor`.
Its public shape describes the first modality, while Tensor-only operations do
not validate the complete wrapper. The custom sampler's second output is
rebuilt from the callback x0, which arrives either as one flat pack or as the
nested view of it, and either way must come back as the original ordered
modality shapes.

**Fix:** update dgx-monarch on every host, restart the driver and Worker
services from the same source, and rerun the exact check. The Gate compares
every finite modality in both proof legs and includes every modality in the
immutable transaction snapshot. Sampler leaders preserve a detached, contiguous
CPU wrapper; packed returns use actor messaging, never RDMA. SamplerCustom
takes either x0 shape and returns a distinct shape- and dtype-matched denoised
wrapper; a flat x0 must carry the exact packed width. Pipeline and Fleet
aggregate modalities independently. Mixed, malformed, non-finite, aliased, or
structurally drifting outputs fail closed. Packed DP is a typed refusal, so
choose a topology whose model-parallel degree consumes the full world. Every
contextual PASS row from an older protocol is stale and re-proves once under
the current one.

## 37. Strict provenance refuses setup or an operation snapshot

**Symptom:** a sealed validation run fails before rendering with
`strict provenance requires custom nodes disabled at bootstrap`, reports that
the custom-node bootstrap policy changed, lacks its READY setup snapshot, or
rejects a Git query, source, or loaded-module origin as dirty, inexact, or
outside the attested tree.

**Cause:** strict evidence is bound to one worker process lifetime and one
source-only bootstrap policy. Changing `disable_custom_nodes` after ComfyUI was
imported, relying on the legacy `DGXM_NO_CUSTOM_NODES` preload switch, running
different or dirty source on either host, retaining stale managed-source
siblings/bytecode, or loading an owned module from another path invalidates the
claim. These refusals guard evidence; they are not render fallbacks. Only the
top-level ComfyUI `custom_nodes` directory is exempt from Git status,
hidden-index-flag, and source-inventory checks, and only when the fixed
`custom_nodes_disabled` bootstrap policy kept it from loading. Every other
checkout path stays strict, as do hidden index flags on included paths, tracked
sources, and loaded origins.

**Fix:** stop the live driver and recycle its owned mesh before touching Worker
services. For a sealed campaign, set `[worker_args] disable_custom_nodes = true`,
restore clean exact dgx-monarch and ComfyUI checkouts, then run `dgxm restart`
so managed source is exact-synced and both worker processes start under the
source-only bytecode policy. Confirm `dgxm doctor` and source parity before
retrying from a fresh driver. Do not substitute `DGXM_NO_CUSTOM_NODES`.
Interactive workflows that need third-party nodes leave the strict policy false
and cannot claim strict campaign provenance. Git installed under a prefix works:
the worker looks it up once on its bootstrap `PATH`, pins the resolved absolute
path, and runs it with a sanitized environment. Put it on that `PATH` before the
restart and do not replace it while the process lives; a changed executable
identity fails closed.

## 38. FSDP exact checkpoint pinning requires Linux procfs

**Symptom:** an FSDP load refuses before Comfy opens the checkpoint with
`FSDP exact checkpoint pinning requires Linux with procfs mounted at
/proc/self/fd`.

**Cause:** the exact-inode load contract uses Linux's `/proc/<pid>/fd/<fd>`
path to give Comfy a suffix-preserving alias for the already-proven open file.
Non-Linux workers do not provide that contract, and a Linux container can hide
or omit procfs. Continuing would reopen the user-visible path and bring back
the checkpoint-replacement race that the FSDP proof closes.

Teardown removes the alias before it closes the checkpoint descriptor. A failed
alias removal can be retried, because the worker still owns the descriptor.
Once the descriptor close begins, any reported error counts as an ambiguous
release that the worker never retries; use **Reset attached mesh** as [#22](#22-a-previous-model-load-could-not-be-cleaned-up--model-reuse-is-blocked)
describes.

**Fix:** run FSDP workers on Linux with procfs mounted so `/proc/self/fd`
exists, then retry from a fresh load; otherwise use a resident non-FSDP
topology. The refusal comes before the checkpoint descriptor or the temporary
alias directory exists.

## 39. Plain Wan Ring2 refuses the TORCH_FLASH q/k/v dtype

**Symptom:** a plain Wan `ring2` render raises
`Wan ring TORCH_FLASH is validated for matching FP16/BF16 q/k/v only` before
sampling enters ring communication.

**Cause:** the bounded pure-Ring2 path first proves that sequence-parallel rank
order matches the bound ring-group slots and their global-rank mapping. For
each local batch it then places both K/V shards directly into that verified
order and runs one native ATen Flash Attention call with local Q and
full-context K/V. It does not use yunchang's low-precision block outputs or
block-LSE merge. On the validated kernel contract, matching FP16 or BF16 q/k/v
preserve their input dtype; FP32 activations or mixed q/k/v dtypes are outside
that contract. The refusal comes before ring P2P; the path never casts
activations or picks an unverified fallback.

**Fix:** leave stock Comfy Wan inference dtype selection in place: default
quantized loads normally compute attention in FP16, while BF16 loads remain
supported. Remove any custom operation that forces FP32 or mixed attention
projections. A uniformly FP32 workflow must use `single` so stock Comfy can
select an FP32-capable attention backend; mixed projections must be corrected.
Do not cast FP32 to a lower dtype to get past this guard. Full-context K/V
uses more peak memory than the dependency's blockwise ring algorithm, but the
temporary allocation is bounded to one local batch. If that allocation does
not fit, use `uly2` or `single`; the pure-Ring2 path raises the failure and
never falls back to blockwise merging. This is the runtime contract, not a
hardware test result: rerun the sealed Ring2 matrix before promoting a
new build.

## 40. Cross-rank identity refuses malformed or topology-misgrouped evidence

**Symptom:** a distributed render raises `cross-rank latent identity FAILED`
and reports either malformed latent signatures or a global-rank/DP-rank
mismatch. No leader output is returned.

**Cause:** the driver validates each worker's complete canonical latent
signature before grouping or comparison, even when a DP group has only one
rank. It also derives the only valid DP coordinate from the global rank and the
pinned model-parallel width. Missing fields, inconsistent shape/count data,
non-finite statistics, malformed digests or projections, and balanced but
swapped DP groups are damaged evidence rather than a render result.

**Fix:** keep the exact same topology snapshot on the driver and every worker,
then recycle any worker that retained stale setup state and retry. If the
failure persists on a clean mesh, preserve the generic error and worker logs
for diagnosis; do not copy latent-signature payloads into reports or relax the
check. A custom worker or adapter must emit the canonical signature the stock
worker path produces, never partial evidence of its own.

## 41. LoRA reloads after `unbake capture skipped`

**Symptom:** `unbake capture skipped (reason)` appears during a quantized LoRA
load. Rendering may succeed, but subsequent LoRA changes reload the model;
status lacks `low_rss` and authoritative unbake keys.

**What to do:** update every host and restart driver/workers from the same
source. Do not edit checkpoint markers or bypass the low-RSS status gate.
Unsupported markers or mismatched live conversion must fall back to reload.

Two valid formats were rejected by older builds:

* **Wan FP8-scaled:** a zero-length root `scaled_fp8` tensor. Current builds
  accept zero-, one-, or two-element legacy markers; only two enables
  `full_precision_matrix_mult`. Dtype, namespace, scales, live quantization
  state, and checkpoint bytes are still checked.
* **MiniMax H3 int8-convrot:** per-layer `<layer>.comfy_quant` markers rather
  than `_quantization_metadata`, with recipe
  `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
  Current builds accept the direct recipe flat or under `params`, retaining
  byte, live-contract, and owning-module checks. The defect was marker format,
  not pruning; embedded Krea2 metadata already worked.

An unsupported recipe key still forces reload and is named in the log, for
example `the marker names 'per_row', which the int8_tensorwise restore does
not replay`.

## 42. A resident-adoption proof render is refused

**Symptom:** an explicitly instrumented proof render fails with a
`resident-adoption` context, artifact, setup, request, rank, or cross-rank
evidence error. A second render in the same context is refused even when the
first attempt failed. Pipeline samplers reject the context before submission.
Ordinary renders that do not opt into this private proof scope are unaffected.

**Cause:** the proof contract authorizes exactly one render submission and
requires every rank to report the same request-bound resident model and
full-content artifact identities. A reused or malformed context, missing rank,
changed artifact, stale setup generation, request drift, or disagreeing worker
evidence is damaged proof rather than permission to continue. Multi-render
pipeline surfaces cannot preserve this one-shot authority.

**Fix:** preserve the exact error and per-rank worker logs, contain the failed
attempt, and audit the independently authorized prompt, artifact hashes,
runtime setup, and context digest. Start a separately authorized fresh proof
context only after correcting the mismatch; do not automatically retry, copy
evidence between attempts, relax all-rank validation, or route the proof
through a pipeline sampler.

## 43. A render dies with `takes from N to M positional arguments but M+1 were given`

**Symptom:** after a ComfyUI update, a rewritten forward raises an argument
count `TypeError`, such as
`Krea2Adapter.inject_usp.<locals>.usp_forward() takes from 4 to 6 positional arguments but 7 were given`.
Gate may then abort and stock fallback may report
`LifecycleBusyError: 1 abandoned sample(s) may still be running`. Diagnose the
original `TypeError`; the lease refusal is a consequence.

**What to do:** update dgx-monarch to match ComfyUI. Doctor checks
infrastructure and may remain green. Until a compatible release is available,
topology `single` uses no rewritten forward. On a newer ComfyUI commit, run
`python -c "from dgx_monarch.comfy_surface import assert_comfy_surface; assert_comfy_surface()"`
against that installation.

On older builds without Gate-abort cleanup, use `dgxm restart`, then restart
ComfyUI to clear the blocked fleet. Current builds leave the mesh clean when
the fallback push was refused before dispatch; the remaining abandoned lease
can recover through the one-respawn path in
[#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).
The underlying `TypeError` still abandons its lease and needs a compatible
adapter.

**Why:** rewritten forwards require exact stock signatures and must accept
every declared call shape. Inserted or reordered arguments, swallowed new
keywords, and newly populated positional slots can break this contract even
when imports succeed. `comfy_forward_contracts.py`,
`comfy_rebound_signatures.py`, and `comfy_rebound_sites.py` define and check
these bindings; stock outer callers have separate prefix-compatible checks.

**Krea2 reference latents:** concatenated reference tokens with a resolved
`reference_latents_method` refuse on the driver before dispatch, without an
abandoned sample or reset. Older builds refused mid-forward and needed a
Recycle. The worker guard remains a backstop. Requests without a resolved
method, and `single`/`dp2`, are unaffected. This preflight cannot inspect
`post_input`/`attn1_patch`: no shipped node transports those `model_options`
patches into the resident spec, so those rely on the mid-forward guard.

<a id="44-lens-uly2-1-step-fidelity-nrms-013-016-pre-issue-132-historical"></a>

## 44. Historical Lens uly2 one-step fidelity NRMS 0.13 to 0.16

**Symptom (older builds):** Lens `uly2`/`ring2` had one-step NRMS 0.13-0.16
against certified single-GPU/dp2 output, while `cfg2` and `dp2` matched exactly.
The adapter omitted divisibility-pad exclusion, exposing synthetic attention
keys when image or GPT-OSS text tokens did not divide the SP degree.

**What to do:** update dgx-monarch. Lens now uses
`padded_row_indices`/`drop_rows` in image-first joint order. The retest measured
NRMS 0.061 at 1328x1328 and 0.060 across two 1344x1344 prompts, below 0.10,
with cleanup confirmed. Current padding restrictions remain in
[#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid);
see CHANGELOG.md for the fix.

## 45. Anima `dp2` (and `single` at world 2 with an even batch) rendered a different image

**Symptom (older builds):** Anima `dp2`, or `single` at world 2 with an even
batch, produced unrelated images. `uly2` and `ring2` were unaffected.

**What to do:** update dgx-monarch. Until then, use `uly2` (the auto choice)
or `ring2`; avoid both affected presets. The hardware retest measured DP2
against certified Ulysses2 at NRMS 0.024, below 0.10, with cleanup confirmed.

**Why:** the generic DP slicer mistook Anima's unbatched T5 token IDs and
weights for per-image data, sometimes leaving a rank one token. Anima now
marks `t5xxl_ids` and `t5xxl_weights` in `Adapter.dp_cond_exempt_keys`, so every
rank receives the full arrays for `llm_adapter`, as in a single-GPU batch.

## 46. An older build refuses PixelDiT/PiD sequence parallelism

Older builds refuse `uly2` and `ring2` because tests measured NRMS 0.265 and
0.338 against a verified `dp2`/single reference. Update dgx-monarch. The fix keeps
projections and MLPs sequence-sharded but runs attention in gathered stock
stream and head order. World-2 PixelDiT 1024 matched `dp2` exactly on both
`uly2` and `ring2` across 20 steps. PiD 1024-to-4096 matched `dp2` exactly on
`uly2` across its full 4-step schedule with the optional pixel-stage LQ
feature active. `auto` selects `uly2` for this family.

The `sp_unvalidated:pixeldit_comfy` accuracy waiver is retired. Ledger rows
written in its protocol-v9 vocabulary stay readable, but the adapter raises no
PixelDiT waiver card and ignores `DGXM_WAIVE_KNOWN_WRONG=sp_unvalidated`. World
sizes other than two remain untested on hardware.

## 47. Wan SCAIL/SCAIL-2 render refused: activation footprint preflight

**Symptom:** a `WanSCAILToVideo` (or SCAIL-2) render is refused before any
GPU work starts, with a `StockLoadCapacityError` naming "activation footprint
preflight refuses...". A Wan Animate 2 render can meet the same refusal.

**Cause:** SCAIL/SCAIL-2 concatenate reference-image tokens onto the video
latent's temporal axis and append pose tokens onto the flattened token axis
before the DiT blocks. Sequence parallelism shards only the token axis, while
the 14B weight stays fully resident on every rank whatever the topology. On
UMA, `stock_load_preflight` (see [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)) checks only the checkpoint file size at
load time and knows nothing of resolution, frame count or these extra token
streams, so on its own it would let an oversized SCAIL config reach an untyped
kernel or NVRM out-of-memory kill. `activation_footprint_preflight`
(`mesh_safety.py`) estimates weight plus activation bytes from the
checkpoint's file size and the video, reference and pose latent shapes, and
raises the same `StockLoadCapacityError` as [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) when the estimate exceeds
`MemAvailable`. It runs on the driver before any worker RPC, only on UMA, and
only for the families in `mesh_safety.REF_POSE_TOKEN_FAMILIES` (`wan_scail`
and `wan_animate2`); no other family meets it. `minimax_h3` uses the same
guard id from its own module, because a packed audio+video+text sequence needs
its own row algebra and its own calibrated constants: see [#56](#56-minimax-h3-render-refused-activation-footprint-preflight).

**Reading the error:** the message splits the estimate into weights and
activations, gives the token count of each stream (video+ref and pose), and
names a `tokens/rank <= N` bound. The driver prices the render as one rank, so
the bound applies to the render's whole token count.

**Fix:** shrink resolution, frame count or reference-image count until the
token count is under that bound, or free memory (`sync && echo 3 | sudo tee
/proc/sys/vm/drop_caches`, stop co-resident LLM servers, and `dgxm reap`). If
this preflight refuses a config you know fits, set
`DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver process before launch to
bypass it. The same variable also turns off the H3 activation preflight
([#56](#56-minimax-h3-render-refused-activation-footprint-preflight)) and the
LTX one, together with the render-activation charge each adds to the
loader-site capacity check, and the render memory preflight
([#79](#79-a-cross-spark-render-refuses-because-the-ranks-disagree-about-weight-residency)).
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)'s preflight, the driver
footprint preflight
([#52](#52-driver-side-footprint-preflight-refused-a-render)) and every gate
stay on. This estimator's activation-bytes-per-token constant is an
uncalibrated placeholder awaiting hardware measurement, so a refusal at the
margin may be conservative, not a hard ceiling.

**Warm re-renders:** a checkpoint that already rendered in this driver
session, or that a pipeline sampler has submitted, is credited: its weight
bytes already sit inside the lower `MemAvailable` reading, the message says
"weights resident, credited", and only the activation term is estimated.
Every successful Recycle drops that credit, and so does a render that fails
while its result is collected and evicts the fleet. Other evictions do not,
for example a fleet that dies during submit or in a Fleet wave. If the
workers lose the model another way, for example when a different slot
replaced it, the credit is stale and the estimate omits the weight term. That
case can still end in an allocator OOM, as it would with no preflight, but it
never blocks a render that fits.

**Minimal-footprint gate runbook (first SCAIL attempt on new hardware):**
build a `WanSCAILToVideo` graph with a small, 32-divisible `width`/`height`
(for example 384x384), a small `length` (for example 17, giving 5 latent
frames), `batch_size=1`, exactly one `reference_image`, and no `pose_video` or
`pose_video_mask`. On the Init node set `auto_gate=off` for this first
plain-render probe, which skips the multi-leg check, with world=2 and
topology=auto (resolves uly2). There are three possible outcomes. The render
completes: capacity at this minimal config is proven, not the family in
general; re-run with `auto_gate=first_use` next to attempt the real check.
The typed `StockLoadCapacityError` above fires before any GPU work: confirm in
`journalctl`/`dmesg` that no `nv_err_no_memory` or kernel OOM-killer event
occurred; this is the catchable outcome the preflight is for. Or a kernel or
NVRM OOM occurs although the preflight passed: report it as a regression,
because the estimator under-refused.

**No panel card.** The activation estimate is class C like the residency
capacity check in [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), but the
consent subsystem
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)) raises a
card only where a rescue exists, and no residency change makes an over-budget
activation footprint fit: residency changes when weight bytes appear, never
how many activation bytes a shape needs. The escape stays the environment
variable above. The MiniMax H3 refusal in
[#56](#56-minimax-h3-render-refused-activation-footprint-preflight) carries
the same guard id, `activation_footprint_preflight`, also with no card, for
the same reason.

## 48. MiniMax H3 refuses `cfg2`, `dp2`, and `single` at world 2

**Symptom:** an H3 render under the `cfg2` preset is refused with a
`PackedCfgParallelError`. `dp2`, an `auto` fold that derives dp, and the
`single` preset at world size 2 (which derives dp2 without saying so) are
refused with a `PackedDataParallelError`. `auto` never selects any of them for
this family.

**Cause:** stock ComfyUI runs this DiT at batch 1 and raises
`ValueError("MiniMax H3 supports batch size 1")` on anything larger. It also
rebuilds the packed layout object per conditioning entry, and comfy decides
whether cond and uncond may share one batched call by comparing those payloads
by value, which two freshly built layout objects never satisfy. So every H3
step is two separate batch-1 model calls at any CFG value, and cfg-parallel has
no batched call to split: the split would fail on the leading batch dimension
even if the dispatch reached the worker. The driver refuses both on the latent
itself: H3's latent is a `NestedTensor` pair (video plus audio), and the
generic packed-latent guard will not split it across ranks for either cfg or
data parallelism.

**Reading the error:** both refusals fire on the driver before the first-use
gate and before any dispatch; each took 0.0 s in testing (docs/VALIDATION.md,
MiniMax H3 anchored guides). If a cfg or dp topology ever gets past that guard,
a backstop in the submit path raises an `UnsupportedModelError` naming the
family.

**Fix:** use `uly2`, which is what `auto` picks for this family at every
resolution, or `ring2` on an unpadded packed total ([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). For a true
single-GPU reference use `mode=local` with `gpus_per_host=1`; the `single`
preset at world 2 is not that reference. None of this constrains
classifier-free guidance itself: CFG above 1.0 renders normally on every
supported topology, at two sequential model calls per step instead of one
batched call.

## 49. MiniMax H3 `ring2` refused on a padded packed sequence

**Symptom:** `ring2`, or any hybrid preset carrying ring degree, raises a
divisibility-padding refusal (the H3-worded one, same cause as [#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid)) on a graph
that runs fine on `uly2`. Editing one word of the prompt can flip the same
graph between refusing and running.

**Cause:** H3 denoises one packed sequence, and its length is

```
S = text_rows + cond_rows + ref_rows + 2 * audio_t + latent_t * (height // 32) * (width // 32)
```

where `text_rows` is the raw Qwen presentation length of the prompt. The audio
term is always even, but the text term is user-typed, so the parity of `S` can
change with every prompt edit. When `S` does not divide the sequence-parallel
degree the shard pads, and H3's attention is maskless and fully bidirectional,
so a zero pad row is a real key for every real token: the first block's
modulation multiplies the row by its scale, then adds a learned shift, and the
row stops being zero. The adapter therefore passes the pad rows as explicit
drop coordinates, and that exclusion exists on the pure-Ulysses path only, so
ring refuses instead of attending to invented rows.

Observed in hardware testing: on the first use of a model/context
combination, this refusal fires inside the auto-gate check (the check's
own proof render hits it before your render would), the check aborts, and
the fleet then refuses further queues until a Recycle (the sidebar's **Reset
attached mesh**, or a Worker service restart when no driver is up). One Recycle was enough in the test. At world 2
`auto` never routes H3 onto ring, so only an explicit ring preset on an odd
total reaches it there. At an odd world `auto` falls back to pure ring, and at
an even world above 2 it folds the spare data-parallel ranks into ring; both
refuse a padded sequence the same way.

**Fix:** use `uly2`. If you need ring for a capacity or comparison reason,
make the packed total divide the degree: pick a canvas where `(width // 32) *
(height // 32)` is even (1344x768 gives 42 * 24 = 1008), which makes the
video, audio and keyframe terms all even, and then only the prompt's token
count and any reference-image blocks decide the parity. The negative prompt
has its own token count whenever the sampler evaluates the unconditional pass:
comfy skips it at cfg 1.0 only for samplers that leave the cfg-1 optimization
on, and every `_cfg_pp` sampler turns it off, so it runs even at cfg 1.0 (an
empty negative is still a few tokens). H3 builds a separate packed layout per
conditioning, so cond and uncond each pad or do not pad on their own. For a
ring leg, either pin a non-cfg_pp sampler (both committed H3 example graphs
pin `res_multistep`) or land both prompts on an even total. There is no
supported way to read the packed total from the UI, so treat ring on H3 as an
expert path and `uly2` as the default. Do not force ring past the refusal: it
keeps a padded ring render from producing wrong math that nothing flags.

The refusal opens with `[dgxm:K guard=ring_pad:minimax_h3 waivable=1]`
(docs/DESIGN.md section 5.9): output known to be incorrect with an expert waiver behind
it, off by default. The generic backstop in the adapter base carries the
unscoped id `ring_pad`; the two stay separate so the permanent row names which
family's guard was waived. The card states the measured error it asks you
to accept: Krea2 artifacts caused by attention to padding rows. H3 has no
padded-ring measurement of its own and needs none: the mechanism that
corrupted Krea2 is present here in full, so the guard follows from that
mechanism rather than an H3 threshold. A granted waiver stamps the output
rendered-under-waiver and writes a permanent ledger row
([#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means)). When
the refusal fires during the check, recover in the order above: allow it, then
Recycle, then queue again.

## 50. MiniMax H3 capacity: planning a clip length before you start one

**Symptom:** you want to know how long an H3 clip fits before you start one.
The typed refusal and everything it charges are in [#56](#56-minimax-h3-render-refused-activation-footprint-preflight); the driver-side half is
in [#52](#52-driver-side-footprint-preflight-refused-a-render).

**Capacity estimate.** The row count in
[#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence) grows linearly
in frames and in canvas area: at 1344x768 the video term alone is 1008 rows per
latent frame, and a 124-frame clip carries 37 latent frames. Start from the
smallest measured shape, 512x320 at 5 frames and 1 step (the small-shape H3
retest in docs/VALIDATION.md), then grow canvas and length one step at a time.
The loader-site capacity check prices the text encoder and the two VAEs from
the graph a queued render passes; a headless call passes no graph, so budget
them by hand there
([#85](#85-a-script-or-harness-fills-the-box-although-the-loader-wall-exists)).

## 51. A worker actor outlives its client and holds tens of GiB

**Symptom:** an idle host keeps losing free memory while a Python actor holds
tens of GiB. `dgxm status` may be green and doctor may attribute pressure to
page cache. Older deployments and hosts outside cluster.toml may retain the
actor even after `dgxm restart`.

**What to do:** inspect `dgxm doctor`'s `orphaned actor procs` row, then run
`dgxm reap --dry-run` to review the sweep. `dgxm reap` sweeps every configured
host and the local host, even without cluster.toml. `dgxm reap --grace 300`
requires that much orphan dwell time, or process age if dwell was not recorded.
The sweep is safe during rendering: it leaves live owners, children of live
Worker services, and uncertain identities alone.

Only proven strays are signalled, with `(pid, start time)` rechecked in procfs
immediately before SIGTERM, then SIGKILL if needed. Attribution accounts for
reparenting to init/subreapers, dead owners, and parent services born after the
candidate. The driver embeds its identity in spawned-worker environments;
processes cannot expose later environment writes through `/proc/<pid>/environ`.
An inspection error, remote timeout, incomplete sweep, or survivor makes the
host UNKNOWN/failed and the command nonzero. No signal or success line is
proof of cleanup unless the sweep settled; repair the host/transport and retry.

**Automatic cleanup:** the driver renews actor leases. After 300 seconds
without renewal by default, the actor logs `dgxm-reap:` and exits. Inspect
`journalctl --user -u dgxm-worker` for that line.
`dgxm down` sweeps after stopping the service, including its children;
`dgxm up` sweeps only orphans. `dgxm restart` uses both. A host is not reported
successful before its sweep finishes.

Doctor's orphan row covers only its local host. It names PID, size, dwell, and
remedy for marked actors holding at least 4 GiB with no live owner able to stop
them. It does not identify a dead-client actor while its Worker service lives:
an established service socket says nothing about the driver. The actor lease
and the next `down` sweep cover that case. Liveness recovery after a timed-out
stop is covered in
[#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

**Lease settings, on the driver:** `DGXM_REAP_GRACE_S` accepts 300-3600
seconds. Other values are replaced by 300, not clamped, with a warning. The
lease minus a 30 s renewal interval must exceed the teardown token's 240 s
lifetime. Use `dgxm reap --dry-run` to inspect cleanup rather than shortening
the grace.

`DGXM_DISABLE_ACTOR_REAPER=1` sends a zero lease and disables both actor
triggers; only the sweep and doctor remain. Both settings arrive with the
first renewal cast. Until then, parent-loss detection remains armed; it covers
the gap before Monarch installs its parent-death SIGKILL and stands down when
a live lease exists.

**Local or unmarked actors:** local (`this_host`) actors are ComfyUI children
and have no configured hosts for `up`/`down` to sweep. Use `dgxm reap` or
restart ComfyUI. Recycle requires a live mesh and working in-process stop; it
helps only while that fleet is attached and rendering has finished.

An older actor without ownership markers is reported but not signalled. On its
host, review `python -m dgx_monarch.cli.actor_reaper --sweep --no-marker`
with `--dry-run` first. That option accepts the launcher marker alone; `dgxm`
never passes it. Alternatively, before signalling an exact PID yourself,
confirm all three: `/proc/<pid>/cmdline` contains
`monarch._src.actor.bootstrap_main`, its UID is yours, and it has no
ESTABLISHED socket. Send SIGTERM first, then SIGKILL only if it does not exit.
Never use `pkill -f`: it can match the issuing shell or unrelated Monarch
applications. A process in uninterruptible kernel sleep that survives SIGKILL
requires a reboot.

**Why this happens:** client death, runtime teardown panic, or a lost stop
outcome can defeat cleanup inside the actor. Older service-only scans missed
these children. Memory returns only when the actor exits; an uncertain stop
must not be retried merely because the memory remains allocated.

## 52. Driver-side footprint preflight refused a render

**Symptom:** `DriverFootprintCapacityError` (a `StockLoadCapacityError`)
refuses before worker RPCs. Render-time messages start
`driver-side footprint preflight:`; loader messages start
`driver-side footprint preflight at the loader node:`. The latter carries
`[dgxm:K]` if a Gate FAIL blocks rescue; ordinary capacity refusals are class C.

**What to do:** follow the reported shortfall and fitting options. For H3,
try a smaller text encoder, a pruned/quantized DiT, fewer or smaller references,
a smaller canvas/clip, or more free host memory. Pruned int8/fp8 reference-video
files are 19.5 GiB versus bf16's 61.7 GiB. For LTX, start with a smaller canvas
or clip and a world-2 `uly*` or `ring*` preset, then the same artifact and memory
options. To free memory: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`,
stop co-resident LLM servers, or use `dgxm reap`.

H3 truncates reference video to the generation length before encoding. A
48-frame reference in a 5-frame generation costs five encoded frames;
shortening only the submitted reference may change nothing. The error reports
encoded geometry, including block count, frames, and pixels.

**Why it refuses:** on UMA, the driver and a colocated worker share memory.
The driver also holds text/vision encoders, VAEs, and reference encodes. A
worker's checkpoint-only estimate cannot protect that combined footprint.
The H3 reference-video failure exhausted memory during driver work after an
admitted DiT load; see [VALIDATION.md](VALIDATION.md).

**Read the two loader windows:**

| Window | Charge and remedy |
|---|---|
| `load` | Stock placement peaks near 2.1x checkpoint size, before the driver stack exists. Slab avoids the host copy and may offer rescue. |
| `resident` | Weights plus the graph's future text encoder, VAEs, reference/guide encodes, and declared render activations. If this does not fit, use smaller artifacts or inputs; changing residency cannot remove the stack. |

Rescue is offered only if slab fits both windows
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)). With
`slab_weights` auto/unset, `capacity_fit.SLAB_VOUCHED_FAMILIES` determines
whether the family has slab identity evidence: verified families omit the host
copy; other families include it. H3 activations are described in
[#56](#56-minimax-h3-render-refused-activation-footprint-preflight).

The loader checks before liveness, topology, or rank effects. It prices the
pending stack from the prompt graph. An unreadable graph leaves that stack
unpriced and warns once per family; other estimator failures also admit the
load. A cleared charge is cached per checkpoint and world for the session.
Direct headless calls need their own budget
([#85](#85-a-script-or-harness-fills-the-box-although-the-loader-wall-exists)).

**Render-time estimate:** both `run_render` and pipelined depth>1 submissions
check before dispatch or mesh effects. Only pending rank-0 weights on the
driver host are charged. `auto` defers loading to this point; explicit presets
load eagerly, so their weights are already in `MemAvailable`. The reserve is
5 GiB or a larger configured `uma_reserve_gb`.

* Pending weights cost the checkpoint's file size, a conservative upper bound.
* A host copy adds 1.1x only when `slab_weights` is explicitly off. Auto,
  unset, and comfy-managed residency omit it because placement is unknown.
* A persisted FAIL can write `slab_weights=False` into graph worker args. This
  estimate runs before the ledger read, so that extra charge affects later
  pending loads, not the first render that reads the FAIL.
* Existing text encoders, VAEs, and encoded references are reported, not added
  again: their memory is already absent from `MemAvailable`.

Explicit eager loading can therefore pass before reference work exhausts the
host. The loader's graph estimate covers that gap; the render check cannot
protect work that ran before the sampler. Predictions that `auto` would have
refused the historical H3 failure are derived from its memory curve, not a
recorded rerun.

**FSDP charges differ.** The loader charges `1.05 * file/world + 256 MiB` and
omits the host copy only when all five conditions hold: an explicit FSDP preset
(not `auto`), world >= 2, one rank per host with rank 0 on the driver, a
bf16/fp16 header, and comfy-managed residency off. Otherwise it charges the
whole file. FP8/int8 direct wrapping still moves full blocks. `slab_weights`
does not affect a sharded rank.

The worker uses `(1/world + 0.08) * file + 5 GiB` for known-world bf16/fp16;
fp8, int8, unreadable precision, or unknown world uses `1.20 * file + 5 GiB`.
Hardware confirmation of the floor and direct-wrap bound is **NOT RUN**.
FP8 direct-wrap and Krea2/Chroma/Ideogram4 INT8 tests proved stored-byte
execution, not those bounds (VALIDATION.md, memory calibration).

The shard estimate assumes the current bounce-buffer build; older workers
can use about twice the shard. It excludes ComfyUI's sample-time reservation,
roughly another shard, so an admitted load can still meet the partial-load
refusal in [#79](#79-a-cross-spark-render-refuses-because-the-ranks-disagree-about-weight-residency).

**Scope and limits:** this is an admission check, not a peak-memory model.
It covers only UMA `minimax_h3` and `ltx`
(`DRIVER_STACK_FAMILIES` in `driver_footprint.py`). It reads safetensors
headers, sizes, `/proc/meminfo`, cached host resolution, and a local-address
route probe; it issues no RPC. Except for `localhost`, `127.0.0.1`, and `::1`,
cluster host resolution uses `getfqdn`, plus `getaddrinfo` for a hostname.
Those calls have no timeout and cache once per configured name per driver.

Only a complete over-budget estimate refuses. Unresolved files, unreadable
headers, non-Linux/discrete hardware, unsupported families, loader dtype casts,
previously cleared loads, or estimator exceptions proceed without this check.
A cast changes resident size, so charging file size could falsely refuse it.
Cleared-load caches avoid charging weights again after a sampling failure;
stale entries can undercount, but cannot create a false refusal.

The error lists projected/usable totals, reserve, artifact, charges and their
provenance, omitted terms, fitting artifact size, and shortfall. If nothing
fits, it says so. Budget figures lead to survive the Gate's 200-character
truncation. Calibration separates measured failed and successful H3 loads;
near-boundary refusals may be conservative. See VALIDATION.md for the
measurements rather than treating these figures as peak-memory predictions.

**Unstable artifact or ledger:** a checkpoint being copied or rewritten has
no stable byte identity. Wait for it to finish, then queue again. This is a
nonwaivable class C refusal: even standing consent cannot authorize unidentified
bytes. Rescue first reads durable ledger denials without a capability context;
only `unknown` is retried within this load's context. Click resolution uses the
same rule. A torn line still refuses because it could contain a relevant
verdict; rerun Identity Gate. FAIL and other definite denials are not bypassed.

**Disabling a known false-positive check:** set
`DGXM_DISABLE_DRIVER_PREFLIGHT=1` before starting the driver. It disables both
loader and render driver-footprint checks and logs one WARNING per process.
It is separate from `DGXM_DISABLE_ACTIVATION_PREFLIGHT`
([#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight));
other capacity checks, including
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), and all gates remain active.

## 53. A render failed and the panel is asking for one click

**Symptom:** Queue fails before a large load. The error names a panel button
and headless environment variable; the sidebar shows a badge, toast, and card
with the model, measurements, risk, and available protection. ComfyUI cannot
pause execution for consent, so authorization happens after the refusal.

**What to do:** read the card, then click its primary button if you accept.
The driver must write the permanent ledger row before saving consent. A failed
write grants nothing. Once the card says "Allowed", use "Queue this graph
again" or queue the intended workflow yourself: the button queues the tab
currently open, which may differ from the failed graph. A capacity-rescued
load uses slab with a byte-verify certificate. Dismiss records no consent;
the card returns if the next queue reaches the same refusal.

Capacity cards offer slab when stock cannot fit. Accuracy cards concern
`ring_pad`, `sol_attn`, or `shard_quant_scale`; read
[#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means) before
accepting. Tags distinguish capacity (`C`), unproven correctness (`U`), known
incorrect output (`K`), and physical constraints with no bypass (`P`);
see DESIGN.md section 5.9.

**No button means no panel remedy.** Class C messages can link here without
offering consent. Follow their "What would fit" clause; all cases below are
`waivable=0`, and `DGXM_ALLOW_SLAB_RESCUE=1` grants nothing:

| Guard/path | Why no rescue is offered |
|---|---|
| `stock_load_preflight` | Slab is off or quarantined; the file is not safetensors; FSDP/`compile_dit` is active; `lora_low_rss` is off; or slab also cannot fit. The message names the blocker. |
| `slab_load_preflight` | Slab itself does not fit. The message includes the larger stock estimate, so disabling slab cannot help. |
| Comfy-managed `stock_load_preflight` | That residency cannot fit and has forced slab off. Turn off `comfy_managed` and follow its reset requirements in [#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses) before using slab. |

The slab and comfy-managed checks retain a 5 GiB floor to prevent partial
loading during sampling. A larger `uma_reserve_gb` wins where applicable,
capped by the arena stock loading would retain. Consent cannot add memory.

**Headless consent:** set the error's variable on the driver before ComfyUI
starts. Capacity rescue uses `DGXM_ALLOW_SLAB_RESCUE=1`. It writes the same
ledger authorization but no persistent memo; authority lasts for the process.

**Scope and revocation:** panel memos bind checkpoint device, inode, size,
mtime, ctime, consent kind, and a narrow context. Replacing or rewriting the
file invalidates the memo and greys its panel entry. Revoke in "Consents and
waivers" takes effect on the next render. Memo schema 2 also retains the legacy
artifact-digest alias. Other schemas are ignored, dropping stored consents and
Auto-allow capacity rescue with a log message; grant them again if intended.

**Automatic capacity rescue:** the panel toggle skips individual capacity
prompts. Changing it requires a standing ledger row; a failed write leaves it
unchanged. Each rescued combination writes a `WAIVER` row on first use and
after each fleet recycle, not on every load. Unlike accepting a new card,
failure to write this per-combination row does not stop an already-authorized
load; the driver logs the missing audit record.

`DGXM_AUTO_RESCUE=1` grants the same process-wide standing authority without a
card or memo. It covers every auto-eligible kind and every checkpoint the
process loads. Slab is currently the only eligible kind, so
`DGXM_ALLOW_SLAB_RESCUE=1` has the same scope. The panel shows an environment-
enabled toggle checked and disabled; unset the variable and restart the driver
to change it. Automatic rescue never grants an accuracy waiver.

**Quarantine always takes precedence.** FAIL applies before residency
selection, and an explicit stock request remains stock regardless of consent.
An effective FAIL consumes its pending capacity card and revokes the class C
memo it disproved. Accuracy waivers survive because they affect math guards,
not residency settings.

Both list and accept recheck the ledger. List hides cards/consents for FAIL,
RETESTING, unknown, or corrupt state without altering stored state; accept
returns a conflict. Without a governing FAIL, INCONCLUSIVE and RETESTING deny
acceptance without consuming the card or revoking the memo. Later INCONCLUSIVE
or RETESTING cannot override FAIL. Only a newer current-source PASS in the
same canonical capability context clears that FAIL and permits new consent.
Consent never turns INCONCLUSIVE into PASS
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

**Hardware scope:** the 61.7 GiB H3 bf16 reference-to-video and first-last-to-
video files cannot stock-load on the measured host, so they require slab
consent. Earlier stock-residency fidelity results do not validate their slab
runs. Pruned and unpruned int8/fp8 artifacts still load stock.

**If the card is missing:** reload a stale browser tab and confirm the sidebar
is installed. Its poller alone fetches cards; the consent route uses driver-
local state and remains available even if the mesh is blocked. Headless
variables are the alternative when the panel is unreachable. If acceptance
reports "the waiver could not be recorded", repair ComfyUI output-directory
write access, then retry the click; no consent was granted.

**Error metadata:** `[dgxm:measured {...}]` contains ledger measurements and
`<<<DGXM-CONSENT-V1 ... DGXM-CONSENT-END>>>` contains the card descriptor.
These survive text-only worker exception transport. Follow the prose before
them; the panel reads the metadata.

## 54. Slab byte-verify certificate mismatch

**Symptom:** a slab load aborts with a typed refusal naming a tensor, a file
offset, a slab offset and the index of the first differing byte inside a
window. Nothing is adopted, no model becomes resident, and the message says
that nothing was quarantined and that this is not a model-correctness finding.

**Cause:** every slab load verifies the bytes it just placed against the
checkpoint they came from, and this one did not match. The check runs per
tensor, immediately after that tensor's read and before anything can cast,
bake or reabsorb it, with the file's inode identity pinned on both sides of the
read. So a mismatch means one of three things: the file changed under the
loader, the storage returned different bytes than it claimed, or the artifact
was staged badly (a truncated copy, an interrupted rsync, a bad download).

**Fix:** treat it as an artifact or storage fault, not a model problem.
Checksum the file against its source, check `dmesg` for IO errors on that
device, and re-stage the artifact. On a cluster, check every host's copy: each
host verifies its own local copy, and only one of them may be bad.

**Why there is no bypass.** This is class P: no consent, no waiver, no
environment variable. A byte mismatch means the slab does not hold the model
you asked for, so proceeding would publish output from incorrect weights. That
is why every slab load is verified, not only loads that skipped a gate.

**The other arm: no certificate at all.** A consented pre-gate slab load whose
worker returned no certificate row refuses the same way, class P, for a
different reason: nothing was measured wrong; the driver asked for a row shape
the worker does not write. That is version skew, so the fix is `dgxm restart`
on the Worker service and one more queue. On the way out the driver asks the
workers to unload all their models, not only that slot, so nothing uncertified
stays resident; if that call fails, the driver log says so and asks for a fleet
recycle before the next render. It is class P because this is missing load
evidence, not a measured image mismatch that could be accepted under an
accuracy waiver.

**What it does not do.** It writes no gate verdict, sets no quarantine lever,
and revokes no consent. A storage fault does not evaluate model correctness;
quarantining it would also block a later run from healthy bytes. The
certificate is evidence about bytes; the identity gate is evidence about
pixels; neither substitutes for the other, and a certificate is never a PASS
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

**Reading the ledger side.** A slab load that completes on a box where stock
provably could not fit records a `CAPACITY_CERTIFIED` row carrying the
certificate summary next to the INCONCLUSIVE verdict, never in place of it.
Those rows and the `WAIVER` rows live in a separate key namespace, so they can
never displace a real verdict. The two audit reads are
`grep '"verdict":"CAPACITY_CERTIFIED"'` and `grep '"verdict":"WAIVER"'` over
`dgxm_gate_ledger.jsonl` in the ComfyUI output directory.

## 55. Rendering under an accuracy waiver, and what the stamp means

**What you see:** a red "accuracy waiver" card names the model, known error,
and measurement. Three class-K guards offer waivers: `ring_pad`
([#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid),
[#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)), `sol_attn`
([#84](#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call)),
and `shard_quant_scale`
([#95](#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales)).
They refuse known incorrect math by default. Waivers are never automatic.

**What to do:** accept only if you intend to run with the stated accuracy loss,
for example a comparison or reproduction. Click "Render under waiver (output
stamped)", then queue again. The permanent ledger row must be written before
the memo; failure to record it grants nothing
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)). If the
refusal occurred during first-use proof, Recycle before requeueing
([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). Proof renders
never use waivers: the check may refuse again, record its result, and yield
without raising another card for the accepted waiver.

**Headless:** set `DGXM_WAIVE_KNOWN_WRONG` before starting the driver. Valid
values name guards (`ring_pad`, `sol_attn`, `shard_quant_scale`) or consent kinds
(`waive-known-wrong:ring-pad`, `waive-known-wrong:sol-attn`,
`waive-known-wrong:shard-quant`). Commas combine names, for example
`ring_pad,sol_attn`. A bare `1`, scoped ledger IDs (`ring_pad:minimax_h3`,
`sol_attn:minimax_h3`, `shard_quant_scale:chroma`), and retired PixelDiT names
(`sp_unvalidated`, `waive-known-wrong:pixeldit-sp`) are invalid. Unknown names
warn once per distinct set and list accepted spellings; valid names do not
warn merely because another guard has no use for them.

Environment grants write ledger rows but no memo. They apply to every render
in that driver process, across checkpoints, topology, and world size, and
survive checkpoint replacement. They have no revocable panel row: unset the
variable and restart the driver. Prefer the panel for narrowly scoped grants.

**Panel scope:** a memo covers the checkpoint, its options and LoRA names,
resolved topology, and world size, bound to file identity. A different degree
asks again. An explicit pad waiver also covers `auto` if it resolves to that
same topology: world 3 can resolve `ring3`; a world-4 batch-1 `uly2` row can
resolve `uly2+ring2`. The panel and ledger name the resolved topology.

The driver resolves consent at every dispatch and puts it on that request;
resident models retain no authorization. Revoke in "Consents and waivers"
writes a revocation row and affects the next render without reloading.
Checkpoint replacement/rewrite invalidates a panel memo too. A Gate FAIL does
not revoke a math waiver, but its residency quarantine still wins.

**What is recorded:** every waived render carries
`rendered-under-waiver: <kind> (<measured wrongness>)` in its mandatory class-K
ledger `stamp`, a permanent `WAIVER` row with `"action":"use"`, its returned
latent, and the driver log. The PNG is not stamped: the stock save node owns
that path. Retain the workflow embedded in the PNG and the matching ledger
for provenance.

Chaining that latent preserves prior entries under `"inherited": true`.
A later leg that triggers no guard writes no new use row and does not claim a
waiver of its own, even though its output descends from waived math. Any waiver
it independently uses gets a separate entry.

Pipeline/Fleet joins concatenate `batch_index` in render order, omitting it
only if every constituent omitted it. Waiver entries retain first-seen order,
deduplicated by `(run_id, guard)`; Fleet records every actual use before the
join. Inherited entries write no new use row. Before gating, mesh/session work,
or submission, inherited metadata must be an exact list/tuple of mappings with
nonempty string `run_id` and `guard`. Mixed or malformed `batch_index` and
malformed provenance refuse
([#71](#71-pipelinefleet-refuses-batch_index-or-a-chained-sampler-refuses-inherited-waiver-provenance)).
Older aggregates may contain only the first constituent's metadata; update
and rerun the batch rather than editing it by hand.

**Proof and memory limits:** a waiver grants no PASS, converts no INCONCLUSIVE,
changes no MODELS.md support row, and never overrides quarantine. Auto-allow
capacity rescue excludes class K. A first-use proof that encounters the waived
guard ends without comparison evidence and quarantines no residency setting
([#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)). With no PASS,
the following ordinary render uses stock residency and may need more memory
or time. If only slab fits, accept its separate capacity card too
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

FSDP cannot obtain its required clean-reload PASS through a waiver. In
particular, `uly2+fsdp` without LoRAs cannot proceed if a class-K guard such as
`sol_attn` refuses the proof. Use a non-FSDP topology or an input/kernel the
guard accepts. Ring padding never reaches that proof because ring+FSDP is
already refused at topology validation.

**`audit window is full`:** the driver allows at most 512 outstanding
waiver-audit joins and never evicts one that might return waived output. New
requests omit the waiver until work settles, so their guards may refuse.
Let outstanding renders settle or close, then queue again. The memo is still
valid; the limit protects mandatory audit records.

**Audit:** run `grep '"waiver_class":"K"' dgxm_gate_ledger.jsonl` in the
ComfyUI output directory. Rows identify grant/use/revoke, checkpoint, guard,
channel (`panel`/`env`), the card's exact sentence, and stamp. Waiver rows use
a separate audit namespace and cannot replace a trust verdict.

## 56. MiniMax H3 render refused: activation footprint preflight

**Symptom:** an H3 render is refused before any GPU work with a
`StockLoadCapacityError` tagged `[dgxm:C guard=activation_footprint_preflight
waivable=0]`, whose text after the tag starts `minimax_h3 activation
preflight:`. No kernel out-of-memory kill appears in `dmesg`: this refusal
fires in its place.

**Cause:** H3 runs one packed row sequence through its blocks, and every stream
in it costs memory, but the nested AV latent's `shape` reports only the video
tensor. The guard counts every stream, with the row formula [#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence) states:

```
S = text + keyframe + reference + 2*audio_t + latent_t*(height//32)*(width//32)
```

Each positive conditioning entry is its own model call that builds its own
packed sequence, and ComfyUI copies the same keyframe and reference lists onto
every entry of a scheduled prompt. So the guard takes the largest value of each
stream across the entries, not their sum: the peak is one call.

In two tests of a 1344x768x124 bf16 render on one rank, the kernel recorded
103.3 GiB of anonymous RSS for the ComfyUI process (61.7 GiB slab weights, 16.7
GiB resident text encoder, and roughly 25 GiB of activations and runtime) and
killed it. The capacity checks then in place had allowed both runs.

**Reading the error.** After the tag comes the budget: bytes needed per rank,
bytes usable, the MemAvailable reading, and the reserve subtracted from it,
named as the driver floor (5 GiB) or your larger `uma_reserve_gb`. Then the
charged terms: the DiT weights, with a note that says whether they were
charged, credited as resident, or stood down for a `weight_dtype` cast; and the
packed activations, with the total row count and its five-stream split. Then
rows per rank over the sequence-parallel degree, and the split between the part
that shards with the sequence and the replicated staging that no topology
divides. Then what would fit: the largest packed total this topology and budget
admit, beside the render's own total, and the free memory the render is short
by. The offered row bound is reachable: shrinking to exactly that number
passes, and one row more refuses. When the weight charge alone fills the budget
the message offers no row bound and says so. No canvas change fixes that one,
so take a smaller artifact or free memory.

**Fix,** in the order the message prints them:

1. Shrink the canvas or the clip length. Rows grow linearly in latent frames
   and in canvas area, and at 1344x768 every latent frame is 1008 rows.
2. Run it on world 2 with a `uly*` preset. Sequence parallel is the only degree
   that shards the packed row axis for this family, and `cfg` and `dp` are
   refused for it outright ([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)). The 1344x768x124 bf16 shape completed this
   way in 727.6 s.
3. Cut the number, resolution or length of reference blocks and keyframes.
   Every reference row stays in the packed sequence for every sampling step.
4. Use a pruned or quantized DiT artifact. The pruned int8 and fp8
   reference-to-video files are 19.5 GiB each against bf16's 61.7.
5. Free host memory: drop caches, stop co-resident LLM servers, `dgxm reap`.

**Calibration and limits:** the estimate uses 1,032,192 bytes per sharded
row and 73,728 bytes per replicated row. Constants and derivation are in
`src/dgx_monarch/h3_calibration.py` (`H3_SHARDED_ROW_BYTES` and
`H3_REPLICATED_ROW_BYTES`); measured cases are in VALIDATION.md. Calibration
rejects the killed single-rank 1344x768x124 bf16 case while admitting its
world-2 Ulysses and single-rank int8-convrot controls. The current 5 GiB floor
preserves those outcomes. The observed kill slope is only a lower bound and
would admit the killed shape if used directly.

For 1344x768x124 bf16 at world 2 under `auto`, the pending estimate is
82.5 GiB, requiring about 87.5 GiB MemAvailable with the default floor, or
more with a larger `uma_reserve_gb`. The 92.2 GiB pending-memory example is
derived from idle memory minus encoder/VAEs, not directly measured. An explicit
`uly2` preset uses the already-resident calculation; select it or free memory
if the pending estimate refuses. A four-step pair run completed in 245 s with
61.8 GiB MemAvailable during dispatch-time loading. These examples do not
establish a universal peak-memory limit.

**Per-rank division.** The divisor is the mesh world, and for this family that
is exact: the sharded runtime requires world equal to dp x ulysses x ring x cfg,
and H3 refuses `cfg > 1` and `dp > 1` before dispatch ([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)), so every H3 render
that dispatches has ulysses x ring equal to world. Weights are never divided,
because Ulysses replicates the checkpoint on every rank, and the replicated
staging is charged for the full packed total on every rank at any degree.

**Warm re-renders and residency.** The weight term is credited, not charged,
when an explicit topology preset means the loader node already performed the
load, or when one of three session memos names this checkpoint: the last
completed render, the last pipelined submission, and the last pending load the
driver footprint check cleared. The memos are sampled before that check runs,
or a render would credit itself for its own charge. A `weight_dtype` cast also
stands the weight term down, because after a cast the file size does not price
the resident weights; the activations are still charged. A successful recycle,
and an eviction that retires the fleet, clear all three memos. A memo that
still goes stale can only drop the weight term: it can admit a render that
should refuse, never refuse one that fits.

**The loader-node half.** The loader node's settled window also charges the
render the graph declares ([#52](#52-driver-side-footprint-preflight-refused-a-render)), so a 61.7 GiB eager load is refused before it
happens when the render behind it cannot then run. The graph shows only video
and audio rows: text, keyframe and reference rows do not exist until their
nodes run. That under-count errs toward admitting, and the render site catches
what the loader site misses one node later.

**Escape hatch.** `DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver. It is the
same variable [#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight) documents, because a kill switch belongs to its guard and this
site raises under the same guard id: a separate variable would leave an
operator who set the documented one still meeting an activation preflight. The
driver logs a WARNING once per process when it is set; grep for that line after
an incident.

| site | `DGXM_DISABLE_ACTIVATION_PREFLIGHT` | `DGXM_DISABLE_DRIVER_PREFLIGHT` |
|---|---|---|
| render-site H3 activation guard | stands down, WARNING logged once | no effect |
| loader-site activation fold | term drops to 0 | term drops to 0 |
| render-site driver footprint ([#52](#52-driver-side-footprint-preflight-refused-a-render)) | no effect | stands down |
| SCAIL activation preflight ([#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight)) | stands down | no effect |

The loader row is the one asymmetry. There the activation bytes are a term
inside the driver footprint estimate, so that site's own switch must remove
them, and this guard's switch must too, or an operator who turned the guard off
would still meet its charge one node earlier.

**No panel card.** No consent can change this verdict: residency decides when
weight bytes appear, not how many activation bytes a shape needs.

**One case is not modelled:** a `_cfg_pp` sampler evaluates two independent
packed totals in sequence ([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). The peak is still one packed total, so the
estimate stands, but the two calls can carry different row counts and only the
positive one is measured.

## 57. Console repeats "Future.get() called from within an active event loop"

**Symptom:** the ComfyUI console carries `[WARNING] [actor=<root>] Future.get()
called from within an active event loop` through cluster bring-up, topology
selection, the identity gate and the render itself, roughly ten times per
render cycle. Renders complete, latents match across ranks, the gate passes and
`dgxm doctor` is clean. The line does not appear on torchmonarch 0.5.0.

**Cause:** torchmonarch 0.6.0 forwards that WARNING to its tracing chain on
every `Future.get` made while an asyncio loop is running on the calling thread,
and ComfyUI executes nodes on such a loop. The dgx-monarch releases that print
it make that blocking call on the loop at four driver-side sites: the attach
initialization wait, the partial bring-up rollback stop, the setup rollback
group teardown, and the one dispatch function that every load, sigma
computation, gate leg and sample collection passes through.

**Fix:** update dgx-monarch on every host, and restart the ComfyUI driver as
well as the Worker services so it imports the updated source. Current
dgx-monarch runs all four calls through the bounded scratch-thread helper it
also uses for proc stops and RDMA latent reads. A complete render cycle should
then show the line zero times.

**If it persists after an update.** The message is noise, never a failure:
never infer a distributed-math or transport problem from it. All four sites
are in the driver, so confirm the driver restarted, not only the Worker
services: the driver holds the imported module. Do not silence the message with
a warnings filter or a logging level change. It is a true statement about the
calling thread, and hiding it would also hide a new on-loop caller.

## 58. `worker status poll timed out while a render is in flight`

**What it is.** An INFO note, not a fault. The DGX Monarch sidebar polls
`GET /dgxm/telemetry` every 2.5 seconds, and the worker status call inside that
poll has a 10-second bound. A worker answers `status` on the same actor that
runs your render, so while both ranks are saturated the ping can expire. The
panel keeps its last good numbers and the next poll recovers. The render is
unaffected: `status` is not a session-scoped endpoint, so a timed-out ping
cannot dirty the setup generation, and a plain timeout is not a supervision
failure, so it cannot evict the fleet.

**What to do.** No action is needed. The note appears at most once per render,
and only when the panel has a cached snapshot.

**Warnings that need attention.** Three cases log `worker status refresh failed:
...` at WARNING instead: the same timeout with the fleet idle, any non-timeout
failure, and a timeout with no cached snapshot to fall back on. The last is
what a fleet looks like before its first successful poll, and there the panel
shows no workers at all. Check that the Worker services are up (`dgxm status`),
then run `dgxm doctor`.

One WARNING case is expected: a long model load. The note keys on the render
progress tracker, which opens at dispatch, so a ping lost while the workers
still read a checkpoint from disk logs the WARNING for the same reason a busy
render logs the note. Let the load finish before you investigate.

## 59. `board hotspots` says the box has no temperature sensors, or names a hot one

**Symptom A.** `dgxm doctor` prints

```
[ ok ] board hotspots: no temperature sensors on this box: nothing readable
under /sys/class/hwmon and `sensors` added none.
```

This row is information, not a fault: the box publishes no hwmon temperature,
so the thermal tripwire is inert here and cannot warn you. It is `ok` rather
than `WARN` because no action clears it and nothing points to a problem. A
stock Spark publishes its board sensors through sysfs with no package
installed. So if you see this row on a Spark, `/sys` is not readable here:
check that `/sys` is mounted and that the process can read it (a container
started with a restricted `/sys` is the usual cause).

**Symptom B.** `dgxm doctor` prints

```
[WARN] board hotspots: 1 of 16 sensors at or above 90 C: hwmon0/acpitz/temp6
91.4 C. The Spark fan curve tracks CPU load and can miss a board sensor
```

This warning requires attention. The Spark fan curve ramps on CPU load, and at least
one board sensor sits outside it: the community-identified `temp6` can reach
90 C+ with the fans idle. Check case clearance and airflow first, then whether
anything sits on or against the unit. A hot board sensor is one plausible cause
of a silent hard power-off (the Spark failure catalog below: PD-stuck state and
its hard power-off signature), so do not leave it running hot while you
investigate.

**Symptom C.** The row ends with a count it could not read:

```
[ ok ] board hotspots: 15 sensors via sysfs hwmon (1 unreadable), hottest ...
```

One `temp*_input` errored or held a non-number, or a whole chip directory would
not list. The count makes a sensor that goes dark visible instead of only
shrinking the total. It is `ok`, not `WARN`, because the usual causes never
clear: an mlx5 module bay with no transceiver in it errors on every read, and
so does the `mt7925_phy0` wifi sensor on a Spark with the radio down. Find the
culprit by reading each file yourself and watching which one fails:

```bash
for f in /sys/class/hwmon/hwmon*/temp*_input; do echo "$f $(cat $f)"; done
```

Act on it only if the sensor that went dark is one you rely on. A previously
working sensor that becomes unreadable warrants investigation.

**Reading it yourself.** Each name in the row maps to a sysfs file:
`hwmon0/acpitz/temp6` is `/sys/class/hwmon/hwmon0/temp6_input`, in
millidegrees:

```bash
cat /sys/class/hwmon/hwmon0/temp6_input      # 91400 -> 91.4 C
```

A sensor with a `temp*_label` file shows its label instead of the file stem, so
`hwmon1/nvme/Composite` is whichever `tempN_input` in `hwmon1` has
`tempN_label` = `Composite`.

**No package is needed.** The check reads sysfs directly. `lm-sensors` is
optional: doctor runs `sensors -u`, its machine format, only when sysfs
publishes no temperature at all.

## 60. Which teardown absorption still fires: the `dgxm-absorb-fire` soak

**What you see:** INFO lines tagged `dgxm-absorb-fire` record teardown
classification. They do not themselves suppress faults.

**How to use them:** count sites and actual absorptions separately with the
commands below. torchmonarch 0.5.0 produced transport faults on every stop;
0.6.0 did not in a clean loopback test. That did not justify removal: the
two-host soak still exercised the correlated and grace-window paths.
[VALIDATION.md](VALIDATION.md#lifecycle) records the evidence required before
removing a path.

The fault hook and the supervision path share one list of stop reasons, so a
deliberate recycle, a client detach, a failed-setup rollback and a partial
bring-up rollback each write the same `reason-marked` line wherever the fault is
handled. A typed `SupervisionError` that carries a stop reason writes it too:
the reason check runs after the type check, not inside the string fallback.

**The soak grep** counts lines per site (`$DRIVER_LOG` is wherever the driver's
console output is captured):

```bash
grep -o "dgxm-absorb-fire site=[a-z-]*" "$DRIVER_LOG" | sort | uniq -c | sort -rn
```

With the marker or branch that fired, not only the site:

```bash
grep -o "dgxm-absorb-fire site=[a-z-]* detail='[^']*'" "$DRIVER_LOG" | sort | uniq -c | sort -rn
```

| site | what fired | how to read it |
|---|---|---|
| `reason-marked` | a fault carried an explicit stop reason, one line per marker; `detail` is the marker | the unconditional path. The first two-box soak never saw it: no fault carried a stop reason, so an empty count does not mean the counters are dead |
| `fault-marker` | reasonless fault text matched a transport marker, one line per marker | a marker with zero lines across a whole soak is dead vocabulary in this wheel's taxonomy |
| `in-flight` | the classifier matched the fault to a deliberate stop still in flight | |
| `grace-window` | the classifier matched the fault to the post-stop grace window | a straggler arrived after the stop was confirmed |
| `token-authority` | a teardown token was checked against its time bound; `detail='granted'` or `detail='expired'` | `expired` means the bound refused a stale token and the fault stayed visible: the guard working, not failing |
| `string-fallback` | the supervision string net matched where the `SupervisionError` type check did not; `detail` is the string | usually a fault shape the type check misses, which is a reason to widen the typed check, not the strings. Rule out the other cause first: the typed check also yields when the wheel stops exporting `SupervisionError` from the audited import path, and then every line here is a moved import, not a taxonomy gap |

**No line here is an absorption.** Each site reports what a classifier saw, and two
later rules can still discard that answer. Visibility wins when another mesh is live or
a creation is in progress, so a `fault-marker` line fires and the fault still surfaces
as ERROR. `in-flight` and `grace-window` fire inside `is_deliberate_teardown_fault`, and
the fault hook re-reads the teardown state version after classification
(`src/dgx_monarch/mesh_helpers.py`, the condition that requires `absorb` and an
unchanged `teardown_state_version()`): a teardown transition in between discards the
verdict after the line is written. Site counts are an upper bound. Count real
absorptions from the surfaces that perform them:

```bash
grep -c "teardown notice (deliberate)" "$DRIVER_LOG"    # the fault hook
grep -c "stray in-flight call absorbed" "$DRIVER_LOG"   # the supervision path
```

Sites also count lines, not faults. `reason-marked` and `fault-marker` write one line
per matching marker, so one fault that carries two markers writes two.
`token-authority` writes one line each time a present token is checked: the fault hook
checks every mesh that holds a token for each reasonless fault, and checks one again
while its state changes mid-read; the supervision path can check its own mesh several
times for one failure, or not at all. Divide before you compare a site count with a
count of faults.

**An empty log** is a result, not a failure: under this pin, on this fabric, with this
traffic, the correlated path never fired. Record the soak window and the workload
beside the empty count; a later prune rests on that evidence.

## 61. The first render took minutes and the next one took seconds

**Symptom:** a render that should take about 45 s took about 250 s, and the
progress bar gave no reason. The identical render right after was about 5x
faster.

**Cause:** the first render of a new model+stack combination pays three
one-time stages before your render samples: NCCL bring-up, a cold NVMe weight
load, and the identity gate's 2-4 short test renders. The callout at the top of
this file says what makes a combination new. Every release invalidates earlier
PASS results, so the first render after an upgrade repeats the gate.

**What you should see.** The driver reports these stages in the ComfyUI window,
not only in the console. Toasts appear beside the canvas for a deferred NCCL
bring-up, a deferred cold load, the gate start and the gate verdict, and the
gate's toast stays up for the whole check. Every phase, the test renders
included, reaches the DGX Monarch sidebar's event tail as a `notice` row,
`dgxm top`'s event ticker (its graphs mark gate and load events `G` and `L`)
and the driver log. The closing line names the verdict, the seconds the gate
cost and the number of test renders it took:

    auto-gate: identity gate PASS in 61.4s over 3 proof renders. Your render
    starts now and is the only cost left; the next render of this combination
    skips the gate and the cold load

**A verdict other than PASS denies optimized residency.** FAIL, INCONCLUSIVE or
ERROR means the gate could not verify the optimized residency paths (slab
weights, low-RSS LoRAs), so the driver either runs your render on stock
residency or, under FSDP, refuses it. That toast is styled as a warning and
names the verdict; the driver log above it carries the reason. Entry 35 covers
an `ERROR`/`INCONCLUSIVE` verdict after a prior PASS.

**It denies that combination, not your session.** The ERROR line names the
model file and the combination key whose optimizations were disabled. The next
combination you render through the same graph starts from the settings the
graph asked for and pays for its own check, so a sweep whose first cell ends
unverified does not measure stock for every cell after it. Three exceptions.
The combination that failed keeps its residency settings off until its own next
check. A measured FAIL is written to the ledger and carried into every later
session for those bytes
([#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass)), and
that carry is written to the graph, not to one combination, so a persisted FAIL
for one model costs the next model on the same graph its optimized residency
until you reload the graph. And a check that a measured class-K guard stopped
disables no residency setting;
[#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora) has that line and
what to do about it.

**If you see no toasts:** the notices use the same websocket channel as
cluster-fault toasts, so a browser that never shows either has not loaded
`web/js/dgx_monarch.js` (hard-refresh the tab, and check `WEB_DIRECTORY` is
served). The driver log and the `events` array of `/dgxm/telemetry` carry every
notice whatever the browser does, so a headless driver loses only the toast.

**Do not restart a healthy job during the test renders.** Killing the queue
during the check makes the next attempt pay the same cost again.

## 62. Comfy-managed residency: what the `comfy_managed` widget does, and everything it refuses

`comfy_managed` starts ComfyUI DynamicVRAM (comfy-aimdo) in every actor to
place and page DiT weights. It is opt-in, off by default, has no `auto`, and
is not enabled by `slab_weights`.

It is experimental, LoRA-less, and lacks a sustained multi-prompt soak. Use
slab for LoRAs or LLM co-residency. Hard Clear VRAM emptied model stores and
ComfyUI's cache in two small Chroma FP8-mixed tests, and reloading reproduced
the first output exactly, but actor RSS stayed about 8.6 GiB versus a 1.2 GiB
baseline (stock retained about 8.4 GiB too). Use **Reset attached mesh** to
retire the actors and release retained process memory; persistent Worker
services stay available.
See [Memory calibration](VALIDATION.md#memory-calibration) for scope.

### The two things it forces, and the one it leaves alone

| lever | forced to | why |
|---|---|---|
| `lora_low_rss` | off | The low-RSS lazy un-bake replays comfy's full-load bake protocol, and comfy's `ModelPatcherDynamic` overrides the patch, unpatch, load and backup methods that protocol depends on. dgx-monarch keeps comfy's memory manager out of every LoRA swap path, because on this hardware it once wrote a sentinel over a real weight mid-load (docs/VALIDATION.md, LoRA original-weight backup test), so under this residency the swap path does not run at all. |
| `slab_weights` | off | Slab requires `lora_low_rss`, and a slab tensor is a raw host mapping aimdo's own allocator knows nothing about. |
| `disable_pinned_memory` | nothing, the host's configuration governs | This residency owns placement, not pinning. On unified memory the worker's configuration turns pinning off, and that is what makes DynamicVRAM zero-copy: with no pinned budget ComfyUI builds its staging buffer with no capacity and weights page straight out of the checkpoint mapping. A pinned budget sizes that buffer at twice the model and fills it with a second copy of every weight in the same pool the device computes from, measured at 2.3x the checkpoint. An operator who sets `disable_pinned_memory = false` still gets it, and the capacity check charges the copy. |

The Init node refuses `comfy_managed=on` beside an explicit `slab_weights=on`
or `lora_low_rss=on` rather than silently overruling you, and `cluster.toml`
gets the same two rules from the schema.

### The seven render-time refusals, by first line

Match the error against this list. All seven are class P: they carry
the bare `[dgxm:P]` tag, no guard, no card, and no consent clears them, because
inside a running process there is no bypass. Each names the working
alternative, and none of them loads, bakes or quarantines anything.

1. **`comfy-managed residency is on for this worker and this render carries N LoRA(s)`**
   Turn the widget off for graphs that use LoRAs and **Reset attached mesh**, or
   render the graph without a stack. Nothing was loaded and nothing was
   quarantined: the refusal fires before the load.
2. **`comfy-managed residency is on for this worker and FSDP is active for this render`**
   FSDP reshards weights into DTensors after the load while DynamicVRAM pages
   the same weights independently on every rank. The combination is untested. Turn the widget off for FSDP topologies, or pick a topology
   without FSDP. The driver also raises this
   sentence at the loader node, above the footprint card, whenever the preset
   carries `fsdp` and the widget is on. The worker's copy needs a `load_model`
   call, so without the driver's copy a large artifact would meet a capacity
   card offering slab residency, which this residency cannot use. Both sites raise
   the same string.
3. **`comfy-managed residency and the RDMA latent return cannot run in the same worker`**
   Both register host memory with the CUDA driver, and that collision has never
   been measured. `rdma_latent_return` is off by default (docs/DESIGN.md
   section 5.6), so you only see this if you turned it on.
4. **`Fleet cannot run the identity ceremony itself`**
   A Fleet call has no way to prove a new combination and no way to turn a
   bootstrap policy off for one call, so it refuses. Render the combination
   once through a normal DGX Monarch KSampler so the check can prove it,
   then Fleet takes the exact PASS.
5. **`comfy_managed is a bootstrap policy and this worker process already started with it ...`**
   You changed the widget on a live actor. Use the panel's **Reset attached
   mesh** button and queue the render again. Restart the persistent Worker
   service only if the mesh reset cannot confirm actor teardown.
6. **`comfy-managed residency is on for this worker and the driver could not authorize this render's residency`**
   Two graphs land here. One is a combination with no exact PASS for the
   comfy-managed capability context. The other is a **dual-model** render: a graph
   that dispatches a second, unconditional model is always stamped stock,
   because a check proves exactly one model. Dual-model families are
   therefore out of scope for this residency in this release; run them with the
   widget off.
7. **`comfy-managed residency is on for this worker and the first-use identity ceremony for this combination cannot prove it`**
   The check's own proof render meets a class-K guard, and a check
   renders without accuracy waivers deliberately, so the waiver that clears that
   guard on an ordinary render does not clear it inside the check and the
   check can never reach PASS here. The message names the guard. What
   works: grant that guard's waiver from the panel, or headless from the
   variable the message names, because the card the check raised expires
   after half an hour and this class-P refusal carries no panel sentence of its
   own. Then set the Init node's `auto_gate` widget to off for this graph, and
   every render is stamped rendered-under-waiver; [#67](#67-what-auto_gateoff-costs-and-what-it-does-not) says what
   `auto_gate=off` costs. Or turn `comfy_managed` off and use **Reset attached
   mesh**. If reset cannot confirm actor cleanup, follow its recovery message
   before retrying.

   The first render of such a combination in a ComfyUI session meets the
   class-K refusal itself, with its one-click card, rather than either class-P
   sentence: the check runs, its proof render refuses, and that refusal
   answers the render instead of a forced-stock fallback. The known shape is a
   chroma nvfp4 checkpoint on a sharded topology, whose
   `shard_quant_scale:chroma` guard is [#95](#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales).

   A render that dispatches a second, unconditional model reads refusal 6
   instead, even when an earlier check for the same checkpoint stopped on a
   class-K guard: that render is stamped stock because of the second model, and
   `auto_gate` off does not change that, so refusal 6 names the cause and the
   one setting that works.

There is also one capacity refusal on this path, class C under the usual
`stock_load_preflight` guard and
[#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click):
**`comfy-managed residency cannot load <model> on <host>`**. See "The capacity
wall it keeps" below.

### The five bring-up refusals, by first line

These fire while the worker is starting, before anything is loaded, so you
meet them on a first bring-up rather than on a render. They are class P too.
The first four share this remedy: *turn the Init node's `comfy_managed` widget
off (or remove `comfy_managed` from `cluster.toml`) and reset the Attached
mesh, which gives this box its ordinary stock or slab residency back.*

1. **`comfy-managed residency was requested but comfy-aimdo is not installed`**
   The worker's Python environment has no `comfy_aimdo` package. Install it
   into the same environment the workers run from, on every host.
2. **`comfy-managed residency was requested but comfy_aimdo.control.init()`**
   ` could not load aimdo.so on this host.` The package is importable but its
   native library is not loadable here (wrong architecture, missing build).
3. **`comfy-managed residency was requested but this host does not meet`**
   ` ComfyUI's own DynamicVRAM requirements (NVIDIA, not WSL, torch 2.8 or newer).`
   This is stock ComfyUI's own gate, restated, not an extra rule.
4. **`comfy-managed residency was requested but`**
   ` comfy_aimdo.control.init_devices() reported no working install for device N.`
   aimdo loaded but could not install its patches for this worker's device.
5. **`comfy-managed residency was requested and this worker process already tried`**
   ` to bring ComfyUI's DynamicVRAM up and failed; ...` One of the four above
   already fired in this process. It cannot be retried in place, because comfy
   is imported by then and its patcher was never rebound, so every later render
   and every policy push into this actor refuses until you reset the Attached
   mesh. Look further back in the same worker's log for the first refusal: that
   one says what went wrong.

### Why a change needs an Attached mesh reset

Changing this policy requires **Reset attached mesh**. It patches the loaded
CUDA driver and rebinds `CoreModelPatcher` process-wide; neither can be undone
in a live actor. `disable_custom_nodes` has the same bootstrap restriction.
A persisted FAIL that turns off comfy-managed residency on the graph therefore
hits refusal 5 on an existing actor. Reset before retrying.

### What the first render costs

Enabling the widget adds `comfy_managed: true` to the capability context and
requires a new identity check. Turning it off returns to the original context
and its PASS without re-proving it.

In `cluster.toml`, **omit the key when off; never write
`comfy_managed = false`**. The merged worker-args dictionary is part of every
capability context, so either explicit value changes it and requires fresh
proof. Init writes the key only when on; quarantine changes it only on a graph
that already carried it.

### What the ceremony does and does not prove

The two test renders check consistency across ranks within comfy-managed
residency. They do not compare against classic residency: switching managers
requires fresh actors, so a same-seed, same-graph comparison must cross a reset.
VALIDATION.md records byte-equal output for one LTX 2.5 export at world 1;
world 2 is unmeasured.

No slab or cross-residency leg runs. A class-K guard cannot be waived inside
proof, so such a combination cannot pass; see refusal 7 and
[#84](#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call).

### The capacity wall it keeps

The estimate is `file size + 5 GiB`, or `2.3 * file size + 5 GiB` with a
pinned budget. It includes no legacy load arena and never charges less than
the file. The measured 61.7 GiB artifact used an 86.3 GiB working set with
28.4 GiB MemAvailable remaining; lazy placement did not reduce it below file
size. Slab is forced off, so this path cannot offer slab rescue.

On UMA, DynamicVRAM sees host available memory divided by the number of GPU
workers on that host, rather than stock `torch.cuda.mem_get_info`. Placement
can therefore differ from stock. Watch for lowvram, partial-load, or
models-unloaded lines on any rank.

The worker reserves only the 5 GiB floor. A larger `uma_reserve_gb` is enforced
by driver-side checks only for MiniMax H3/LTX on the driver host. Other
families/hosts have just the floor. Watch `dgxm top` and MemAvailable on the
first large load: UMA exhaustion can stall the host without a catchable CUDA
OOM ([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), "The Spark failure
catalog" below).

### How to check it took

`dgxm status` shows, per worker:

    comfy_managed: true
    comfy_aimdo: 0.4.13
    aimdo_enabled: true
    weight_residency: comfy_managed
    patcher_class: ModelPatcherDynamic

and the worker journal carries one line per rank:

    comfy-managed residency ACTIVE: comfy-aimdo <version>, device 0,...

Do not grep for stock ComfyUI's own "DynamicVRAM support detected and enabled"
line: that string belongs to `main.py`, which a worker never runs.

The model store's load line ends in a bracketed group that opens with
`residency=comfy_managed rung=comfy_managed`, then `priced=` or `unpriced`,
then the residency's `reason=`. The signed resident-adoption evidence
reports `cudaMalloc` for this residency deliberately: that field records the
storage backing, and this residency changes which manager places the weights, not
what backs them. Read the `weight_residency` field above for the residency,
not the evidence row.

### Related

* [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) covers a load that fails or takes the box down on unified memory, and the
  card that offers slab residency instead. Under this residency no card appears,
  because slab is forced off.
* [#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click) covers class C capacity refusals, with and without a one-click card;
  this residency's capacity refusal carries no card.
* [#52](#52-driver-side-footprint-preflight-refused-a-render) covers the driver-side footprint preflight and `uma_reserve_gb`.

## 63. A toast at queue time says the workflow is missing DGX Monarch nodes

**Symptom:** pressing Queue pops a toast beside the canvas that names node ids
and the DGX Monarch nodes that replace them. The render starts anyway.

**Cause:** the graph is valid and runs on the driver while the cluster stays
idle. At queue time, the driver lists the changes needed for distributed
rendering. The advice neither blocks the queue nor edits the graph.

**The three advisory cases.**

* **No DGX Monarch nodes at all.** An info toast: this render is stock and
  single-box. It lists the swap for each stock node that has a cluster
  counterpart, as `#<node id> <stock class> -> <DGX Monarch node>` (eight at
  most, then a count of the rest), and it names the Init node you have to add.
  A graph with no such node gets no toast. Three nodes have no one-for-one
  replacement and are described instead: `CheckpointLoaderSimple` becomes the
  DGX Monarch diffusion-model loader plus the stock CLIP and VAE loaders;
  `LoraLoader` becomes the DGX Monarch LoRA node, which patches the model only
  and leaves the CLIP half with no cluster path; `SamplerCustom` becomes a DGX
  Monarch guider feeding the DGX Monarch custom sampler.
* **DGX Monarch nodes with no Init node.** A warning toast naming them. Init
  attaches the cluster and hands out the mesh every other node in the pack
  takes, directly or through a loader, so a graph without it is incomplete.
* **Init present, stock nodes left over.** A warning toast listing them. Those
  nodes run in the ComfyUI process while the rest of the render runs on the
  cluster, which is usually a half-finished conversion.

**What it never flags.** The CLIP and VAE loaders, text encode, latent, image
and save nodes. Those run on the driver on both paths by design, so there is
nothing to swap them for. The advisor works from a table of known
replacements, so any node not in that table stays silent.

**One swap carries a condition.** A stock `BasicScheduler` swaps for Basic
Scheduler (DGX Monarch), and that node needs an explicit topology on the Init
node: ring2, uly2 or cfg2, not the `auto` default. It computes the sigmas
against the resident model, and `auto` picks the topology at the first render,
after the sigmas are already needed, so the node refuses with a RuntimeError
instead. Every toast that names that node says so; set the topology while you
make the swap and the first Queue works.

**One toast per graph.** Each advisory appears once per driver process.
Changing the relevant nodes produces new advice; requeueing unchanged nodes
does not. The driver log and sidebar event tail retain advisories even without
a browser.

**Turning it off:** set `DGXM_GRAPH_ADVISOR` to `0`, `false`, `off` or `no`
(case and surrounding spaces do not matter). It is read per queue, so it takes
effect on the next Queue with no restart. Anything else, including an unset
variable, leaves it on.

## 64. A packed audio/video custom render dies on the last step

**Symptom:** a MiniMax H3 or LTX audio/video graph built on `RandomNoise` +
`KSamplerSelect` + Basic Scheduler (DGX Monarch) + Basic Guider (DGX Monarch) +
Sampler Custom (DGX Monarch) samples every step and then raises `custom sampler
x0 must be a Tensor, got NestedTensor`. The sample is abandoned rather than
retired, so the next render meets `LifecycleBusyError` until `dgxm restart`.
The same graph on KSampler (DGX Monarch) renders.

**Cause:** an older dgx-monarch build. ComfyUI runs the sampler on one flat
pack, and since Comfy-Org/ComfyUI#15196 it hands every callback the nested view
of `x0` back whenever the latent carries more than one modality. Older
dgx-monarch builds rebuilt the custom sampler's second output from the flat
pack alone, so the only x0 shape a packed render still gets was the one shape
they refused. The denoised output is rebuilt whether or not a node consumes it,
so leaving that slot unconnected never helped.

**Fix:** update dgx-monarch on every host and restart the driver and Worker
services from the same source. Both x0 shapes rebuild the same shape- and
dtype-matched denoised wrapper, per modality, as stock SamplerCustomAdvanced
does. Nothing about the graph or the family changes.
Related: [#36](#36-packed-audiovideo-sampling-fails-after-denoising) for the packed output contract, and [#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2) for H3's topology
refusals.

## 65. `dgxm status` and `dgxm doctor` are green and every attach still fails

**Symptom:** both diagnostics report a healthy cluster,

    10.0.0.1: worker_service=running listener=open (tcp://10.0.0.1:26600, nohup)
    10.0.0.2: worker_service=running listener=open (tcp://10.0.0.2:26600, nohup)

while every queued render raises `MeshAttachError`, or a `LifecycleBusyError`
counting abandoned samples. The driver log usually shows the identity gate
aborting on that error first and falling back to stock residency.

**Cause:** the Worker service rows are true, but they answer the wrong
question. A running process and an open listener are host facts; what refuses
the render is driver-side state on the cached mesh handle: a sample abandoned
by a render that died mid-sample, or a teardown whose outcome was lost.
Neither touches the Worker services, so no per-host probe can see them.

**Read the mesh row.** Both commands ask the driver over its telemetry route
and print what it says:

    mesh: ok (cached, 0 active leases)
    mesh: DIRTY (1 abandoned sample may still be running); reset the attached mesh...
    mesh: POISONED (the driver's Monarch transport cannot be reused...); restart ComfyUI...
    mesh: idle (no attached mesh; the next render session creates one)
    mesh: busy (load_model RPC completion in flight); wait for it, nothing to fix
    mesh: no driver reachable (no mesh view)

`doctor` prints the same verdict in its `mesh health` row, and a DIRTY mesh is
a failure there, not a warning: the driver has already decided that the next
attach raises. Active leases are not a fault; they are renders in flight.

**`busy` is not a fault either.** A cold first load runs for minutes and a
gate cycle longer, and the driver holds the same lifecycle latch throughout.
The row names the operation and passes `doctor`, because the reset a DIRTY
row prescribes would kill that load. Wait for it. If the named operation is
still there long past its budget, treat it as the dirty case below.

**Fix:** follow the mesh row's remedy. An abandoned sample clears with **Reset attached
mesh** in the sidebar panel; restart the Worker service only if setup still
fails after that. An unresolved teardown outcome asks for the same reset first;
restart ComfyUI if the reset cannot confirm the stop.
Never restart the Worker service underneath a live driver mesh.

**Two rows the reset button cannot clear.** `POISONED` reports that this
driver's Monarch transport cannot be reused after a partial bring-up or a
creation failure. It is process state, not handle state, so it reads the same
with an empty cache and no reset touches it: restart ComfyUI. A `DIRTY` row
naming a fleet that was created and never published is the other one. The reset
route reads the published cache only, so it answers `no_live_mesh` there and
changes nothing. Start the next render instead: it drops an interrupted fleet
that is already gone and attaches a fresh one. While that fleet is still live,
or while the driver cannot tell whether it is, the render refuses with a typed
error naming the restart, because a second fleet on
one transport is worse than a stopped render.

**Related:** [#1](#1-attach-fails-with-mesh_attach_config_timeout) and [#2](#2-every-attach-times-out-after-one-failed-attach) cover attach timeouts and a wedged Worker service, a
different failure that looks the same at first. Start there when the mesh row
is `ok` and attaches still fail. [#3](#3-next-render-hangs-at-nccl-rendezvous-after-a-crash) covers stranded procs on the NCCL
master port. `no driver reachable` means the CLI found no ComfyUI on this box,
so there is no mesh to report on yet; both commands find your driver on
whatever port it was launched with, not only the default one.

## 66. The gate says INCONCLUSIVE on a graph with no LoRA

**Symptom:** the first render of a LoRA-free graph reaches a verdict of
INCONCLUSIVE whose reason is that there was nothing to compare. It reads as a
failure, and on older builds it was treated as one: the line

    ERROR auto-gate could not complete; lora_low_rss, slab_weights disabled
    for this session

followed, and every later render in that ComfyUI process ran on stock
residency whatever model it loaded, for the rest of the session.

**What you see now.** A check that ran end to end on a graph with no LoRA
stack, with slab residency not engaged, had nothing to compare. That is the
expected outcome for such a graph, not a failure, and it logs one line at INFO:

    INFO identity gate: nothing to gate on this graph (auto_first_use,
    <model>): no LoRA stack and slab residency is not engaged, so there is
    nothing to compare; INCONCLUSIVE recorded, residency levers unchanged

The ledger and the gate report row still record INCONCLUSIVE for that
combination, and both rows carry `"inconclusive_kind": "no_material"` beside
the verdict. Your render still runs on stock residency, because an unproven
combination never gets the optimized paths. The canvas toast still names the
verdict and says optimized residency is off, which is true of this render.
What no longer happens is the session-wide disable: every other model and
every other graph in the process keeps the residency it was configured with.

**A no-material verdict skips later checks.** First use of the same
combination, in this driver or a later one, reuses the ledger verdict. The
render runs on stock residency and logs at INFO:

    INFO normal render: the identity gate found nothing to gate for this
    combination, so it keeps that verdict; this dispatch runs on stock
    residency and the session's levers stay as configured

**Scope and invalidation.** The skip binds the same identity as a PASS: model
bytes, LoRA set, ComfyUI commit, residency/topology policy, gate protocol,
package release, and the canonical source manifest cached on first use. Any
change requires one fresh check, including a restart into changed same-version
source. The cache does not monitor live source edits. Source-independent FAIL
quarantine still takes precedence for the same bytes. Older rows without
`inconclusive_kind` require one new check.

**What still disables both residency settings, and for which combination.**
Every other INCONCLUSIVE, at ERROR. A check that could not finish the
comparison it set out to make leaves the fleet in an unmeasured state, so both
residency settings are disabled. That record names the combination it came from: the model file, its options and its
LoRA set. The next combination you render through the same graph starts from
the settings the graph requested and runs its own check, because a check
that never loaded those bytes proved nothing about them. The combination that
aborted keeps its residency settings off, and its own next check re-decides
them. The one exemption to the write itself is a live capacity-rescue consent
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)).
`NO_QUARANTINE_VERDICTS` in `src/dgx_monarch/nodes/gate_inconclusive.py` lists
the outcomes that disable no residency setting, and
`src/dgx_monarch/nodes/gate_quarantine_scope.py` decides which combination the
write lands on.

**A check that a class-K guard stopped disables no residency setting.** A check
renders without accuracy waivers deliberately, so a measured guard such as
`shard_quant_scale:chroma` or `sol_attn` refuses inside it whether or not you
hold the waiver. That refusal is an accuracy answer about that combination and
says nothing about slab residency or the low-RSS swap path, so nothing is
quarantined and no worker policy is pushed. The line names the guard:

    ERROR auto-gate could not complete for <model> (combination <key>): the
    class K guard shard_quant_scale:chroma refused inside the ceremony. That
    guard is measured, so it answers this combination's accuracy and takes no
    residency lever from the graph

Render that combination again in the same session and it answers from the
cache, with no second check and no quarantine, and still on stock: the guard's
answer stands until you clear it. Grant the guard's card and set `auto_gate` to
off for that graph ([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)),
or render it on a shape the guard does not refuse.

**The no-material outcome covers the residency you ran under, and nothing
else.** Turn `slab_weights` on afterwards and you are asking for a combination
this check never exercised, so that render runs its own check. The grant is
scoped the same way. The check writes a row for the Fleet scope and for the
explicit-on slab siblings too, and none of those rows carries the
classification, so none of them grants a skip.

**Telling them apart:** INFO with `nothing to gate on this graph` took nothing
from you. ERROR with `auto-gate could not complete` and `disabled for that
combination` took both residency settings from the combination it names; the class K line
above took none. Grep `dgxm_gate_reports.jsonl` for `inconclusive_kind` to get
the same answer per run, or `dgxm_gate_ledger.jsonl` to see which combinations
hold the grant.

Another ERROR-path INCONCLUSIVE says the two repeated optimized-path renders
diverged even though the fresh stock comparison matched one of them. That is
repeat instability, not proof: it publishes no process PASS, persists no PASS
for normal, Fleet, or slab sibling contexts, and disables the risky residency
for that combination. The gate protocol version that added this check
invalidates PASS rows written before it. Preserve the report and investigate
nondeterminism; do not relabel the result as a LoRA failure when the repeat
itself was unstable.

**Not this entry:** [#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass) covers an `ERROR`/`INCONCLUSIVE` verdict on a
combination that previously passed, which is a revoked proof rather than an
absent one.

## 67. What `auto_gate=off` costs, and what it does not

Turning the Init node's `auto_gate` widget off skips identity proof. It keeps
LoRA rendering and the requested optimized residency available, subject to the
independent guards below.

**What off does.** It skips the proof, and nothing else. A single-model render
is stamped `operator_off` and dispatched on the worker policy your graph asked
for, untouched. It does not use the forced-stock pair an unverified combination
gets under `first_use`. So, with `auto_gate=off` on a single-model render:

| | with `auto_gate=off` |
|---|---|
| a LoRA stack | renders, on whichever residency the settings below resolve to |
| `lora_low_rss` | stays as configured (`auto` by default, which is on for unified memory): the lazy un-bake, not baked mode |
| `slab_weights=auto` | still resolves per family on the worker, and a family verified for slab residency still slab-loads |
| `slab_weights=on` | still honored |
| the capacity rescue ([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)) | still offered, still consented, still certified |
| render accuracy | unproven; checking it is up to you |

**Four things that stay true with off, and that off does not cause.**

1. **Fleet keeps forcing both residency settings off for an unverified combination.** A Fleet
   call cannot run the check inline, so it reads the ledger and, finding no
   exact PASS, renders that call with `slab_weights` and `lora_low_rss` off.
   `off` records no PASS, ever, so it never fills the ledger in either. The
   Fleet job still runs and still applies your LoRAs; it bakes them, at the
   residency the `lora_low_rss` tooltip describes for its OFF setting. Prove
   the combination once with the Identity Gate node or `dgxm gate` and Fleet
   takes that exact PASS.
2. **A FAILED check still quarantines.** `off` skips a missing proof, never
   a measured one. A persisted FAIL for this exact combination and capability
   context turns both residency settings off before setup whatever the widget
   says. [#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass) is the way back.
3. **`comfy_managed=on` refuses LoRA stacks.** That widget sits in the same
   Init advanced block, so its refusal is easy to blame on `auto_gate`.
   Different lever, different reason ([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)), and `auto_gate` neither causes
   that refusal nor clears it.
4. **A dual-model custom render dispatches on stock residency.** When such a
   render dispatches at all, both residency settings are off for it whatever the widget
   says, for the reason the note at the top of this file gives. `off` is not
   what turned them off, and `first_use` would not have turned them back on.

**Not the capacity route.** For a checkpoint only slab residency can fit, use
the capacity rescue: a card, a consent, and a ledger row that records the
combination as INCONCLUSIVE rather than unexamined.
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) and
[#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click) carry that
flow.

**When to use it.** A combination you have already proved and do not want
re-proved once per release; a graph whose identity test render triggers a
refusal your own render would have tripped anyway, so the check aborts and
costs you a Recycle on the way
([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)); a bisect
where the extra test renders confuse the measurement. In each case `off` saves
test renders; it adds no feature.

**Effect on later renders.** A check that aborts on a measured class-K guard,
the shape the second case above describes, takes no residency setting from the
graph at all, and any other aborted check takes them only from the combination
that aborted ([#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)).
So `off` here saves you the test renders and the Recycle, not the residency of
later renders.

## 68. MiniMax H3 fails with `cannot import name time_shift_slope` after a ComfyUI update

**Symptom:** the first distributed MiniMax H3 render fails inside the rebound
forward with an import error for `time_shift_slope`, or the Comfy canary names
the `minimax h3 audio carry` seam. A newer dgx-monarch beside an older, partial
or mixed ComfyUI tree instead refuses because `minimax_payload` lacks a valid
`audio_scale` marker, or because the bound stock `MiniMaxH3Model.forward` does
not carry the matching audio conversion. Infrastructure and ordinary actor
checks can stay green.

**Cause:** current ComfyUI removed that helper and moved the dual-schedule
conversion around the inner `_forward`: its stock outer
`MiniMaxH3Model.forward` restores the audio latent before invoking wrappers and
converts the returned raw velocity afterward. An older dgx-monarch adapter
still imports and applies the former inner derivative, so it crashes on the
removed name, and would transform audio twice if the helper were copied back.
The current distributed inner contract under an older outer forward would omit
the conversion. The payload marker comes from a different Comfy module, so it
can sit beside the legacy outer method in a partly updated tree. The adapter
therefore requires both a finite positive `audio_scale` and the fingerprinted
current outer forward before any distributed math runs.

**Fix:** update dgx-monarch and the complete ComfyUI checkout on every host,
check that driver and workers resolve the same source revision, and restart
them so no old payload producer, outer model class or adapter module stays
imported. Do not restore the removed helper or patch the stock outer forward.
Until the versions match, use a true stock single-GPU run with
`mode=local, gpus_per_host=1`; the world-2 `single` preset is no substitute,
because MiniMax H3 refuses it for the data-parallel reason in [#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2).

**Lease behavior after the refusal:** the three payload arms (absent payload,
missing `audio_scale`, malformed `audio_scale`) are decided from the request,
fire identically on every rank before any collective, and carry the class P
tag, so their sample lease retires consumed: fix the cause and queue again on
the same fleet, with no recycle. The bound-forward arm ("does not carry
ComfyUI's matching audio unscale") is decided per host: on a mixed fleet the
sibling rank may already sit inside a collective, so that arm is untagged and
retires abandoned. After it, the next dispatch refuses with the
abandoned-sample guard until one Attached mesh reset (hardware test: one reset,
then the next render dispatches normally).

Version alignment proves only that the current outer and inner contract is
present; it does not prove fidelity. The support row in docs/MODELS.md is the
status authority. The carried-audio retest passed (docs/VALIDATION.md), so that
row is marked HW.

## 69. Worker bring-up is interrupted during spawn, rollback, or client-lease startup

**Symptom:** Init reports `worker actor bring-up failed`, says spawned
processes were `rolled back` or `not safely stopped`, or reports a
client-lease renewer startup failure. When rollback is incomplete, the error
also says the Monarch transport or prior fleet is not reusable. A request for
a different mesh configuration can instead report that another mesh creation
is still in progress; retry it after the active attempt settles.

**Cause:** actor creation touched a client-owned ProcMesh and then raised, a
completed actor spawn was interrupted at its return handoff, partial-ProcMesh
rollback raised as well, or the process-wide daemon that renews actor lifetime
leases could not start after the handle was built. These boundaries can raise
native `BaseException` subclasses, not only `Exception`, and cleanup treats
both as failures. The handle becomes ready only after the lease renewer starts,
so a waiting caller cannot reuse a fleet with no renewer. If native ProcMesh
construction aborts before the pinned callback, active registry or return value
exposes its identity, the driver cannot prove stop authority and poisons the
session instead of guessing. Mesh creation runs one at a time per process,
even across cache keys, because the transport, callback registry and
ownership handoff are process-global.

**Fix:** read the original spawn, handoff or lease failure first; rollback
evidence adds to it and never replaces it. Confirmed cleanup after an
already-built handle or client-lease startup failure may evict that handle and
retry the same transport mode. A `WorkerSpawnError` after transport
initialization leaves the process poisoned even when its partial ProcMesh
rollback was confirmed; restart ComfyUI or the driver before retrying. Only a
local failure proven to occur before transport import or initialization is
retryable in-session.
If a completed stop call reports a definite non-timeout ordinary failure,
replacement stays blocked, but a later explicit **Reset attached mesh**,
shutdown or supervision cleanup gets a fresh bounded stop attempt. A second
definite failure stays blocked, and only another explicit cleanup retries it;
the failed action never retries itself. If the stop timed out, was cancelled or
interrupted, its outer caller timed out while the lifecycle thread was still
settling, or an attempt token survives without a published outcome, do not
issue another stop: the first call may already have succeeded. Wait for
settlement, or confirm the client-owned actor processes are gone and restart
ComfyUI. In every case, do not queue a replacement until cleanup is confirmed.
Never call `hosts.shutdown()` or stop the attached Worker services beneath a
live driver; they are owned separately.

## 70. `dgxm update`, `gate`, or `top` misses a ComfyUI driver on a custom port

**Symptom:** ComfyUI was started by `scripts/comfy-driver.sh` on a port other
than 8188 or 8191, but an old CLI falls back to 8188, or `dgxm update` does not
refuse while that driver is live.

**Cause:** the shipped launcher changes into the ComfyUI checkout before it
runs `python main.py --port N`, so process inspection sees the bare relative
script name, not a path containing `ComfyUI`. Older discovery discarded that
process before probing its listening port.

**Fix:** update dgx-monarch. Discovery admits the bare `main.py ... --port N`
shape as a candidate only. It confirms the pack through ComfyUI's stock
`/object_info/DGXMonarchInit` endpoint and requires the exact Init node name,
category and output type, so an unrelated `main.py` listener or any HTTP 200
or 503 is not enough. `gate` and `top` then use `/dgxm/telemetry` only to rank
confirmed instances by live workers; a cold telemetry timeout or a valid 503
does not hide a confirmed driver, and `update` does not wait on that slower
endpoint. If process inspection is unavailable or restricted, pass the address:
`dgxm update --host 127.0.0.1:N`, `dgxm gate --host 127.0.0.1:N`, or
`dgxm top --host 127.0.0.1:N`. Stop ComfyUI before rerunning an update that
refuses the live driver.

## 71. Pipeline/Fleet refuses `batch_index`, or a chained sampler refuses inherited waiver provenance

**Symptom:** a multi-render node raises `inconsistently returned batch_index
metadata`, says the value must be a list, says its entries must be nonnegative
integers, or reports that its length does not match the constituent sample
batch. A chained sampler can instead say `inherited waiver provenance must be
a list or tuple`, that an entry must be a mapping, or that it must carry
nonempty string `run_id` and `guard`.

**Cause:** a Pipeline or Fleet result joins several separately rendered latent
batches. An explicit index list belongs to each constituent batch and must
concatenate in the same order as `samples`; copying only the first list
creates `N x B` samples with `B` indices. The node refuses mixed presence,
non-list values, entries other than exact non-boolean nonnegative integers,
and wrong lengths instead of publishing a misaligned latent. Separately,
`_dgxm_rendered_under_waiver` is trust-bearing input metadata: when present it
must be an exact list or tuple of mappings keyed by nonempty string
`(run_id, guard)`. The Sequential, Pipeline and Fleet entry paths validate it
before any gate, mesh, session or submission side effect. Valid duplicates
keep the first entry; inherited entries never claim or record a second waiver
use. [#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means) describes the audit contract.

**Fix:** update and restart the driver and Worker services from one source
revision, then rerun the complete Pipeline or Fleet batch. If the refusal
persists, inspect the upstream or custom sampler that produced the named
constituent: every result must either omit `batch_index`, or give a list whose
length equals that result's sample batch and whose entries are exact
nonnegative integers. For an inherited-provenance refusal, do not delete or
hand-edit the stamp: find the custom node that published the malformed
container or identity, update it, and rerun from the last valid latent.
Padding, truncating or cleaning the metadata by hand can change seed and index
meaning, or erase the record of an earlier render under an accuracy waiver.

## 72. A distributed render refuses a Comfy attention patch or NAG-style hook

**Symptom:** a sequence-parallel Flux-family, Chroma, Hunyuan or Wan-Animate 2
render raises `UnsupportedModelError` naming `attn1_patch`,
`attn1_output_patch`, or (for Wan-Animate 2) `attn2_patch`; a Qwen Image 2.1
render refuses a Comfy block or attention hook; or a Qwen Image 2.1,
Wan-Animate 2 or LTX sequence-parallel render refuses a foreign
`optimized_attention_override`. A CFG-parallel render can also refuse an output
hook that needs both condition branches.

**Cause:** each USP rank owns only a token shard, and each CFG rank owns only
one condition branch. Passing the original callable through unchanged does not
keep its tensor domain: a full-stream image slice can address the wrong local
rows under USP, and an NAG-style hook that looks for both positive and negative
branches silently returns unchanged under CFG. Wan-Animate 2's cross-attention
hook also receives only spatially local rows. Qwen Image 2.1's block-causal
prefix must stay replicated and unchanged. A foreign attention override writes
the key the distributed adapter owns, so one of the two writers would be
dropped, and on the native-attention paths (Qwen Image 2.1, Wan-Animate 2) it
would replace the kernel the selector gate checked. Exact math requires the
hook to declare and implement the matching shard-local or condition-local
contract, so unknown hooks fail before distributed attention instead of
producing finite but wrong guidance.

Stock Comfy patch nodes accept `MODEL` and DGX loaders output `DGXM_MODEL`, so
a stock NAG node cannot attach to the shipped DGX graph. This refusal guards
out-of-tree worker-side patch carriers and any later patch transport.

**Fix:** remove the model-option patch or attention-backend override, or use
topology `single` for stock hook behavior. Do not rename or wrap the hook to
get past the guard. Qwen Image 2.1 admits none of these hooks under USP. A
hook opts in through its `_dgxm_attention_patch_capabilities` attribute:
`sequence_shard_local` on the Flux-family, Chroma, Hunyuan and Wan-Animate 2
USP paths, `condition_shard_local` for a CFG output hook. Declare
`sequence_shard_local` only when the hook maps its image slice to local rows in
every affected USP block. Declare `condition_shard_local` only when the hook is
correct on one CFG branch per rank; a hook that needs both branches must
exchange across conditions to be correct. Declare either one only after
hardware fidelity validation has passed.

## 73. Wan SCAIL/SCAIL-2 refuses a multi-row timestep embedding

**Symptom:** a distributed SCAIL or SCAIL-2 forward raises
`UnsupportedModelError` because its timestep embedding has more than one row.
A normal sampler timestep shaped `[batch, 1]` is accepted.

**Cause:** the SCAIL sequence-parallel contract replicates one scalar timestep
row on every rank. Replicating a multi-row `e0` while sharding the combined
reference, video and pose token stream repeats the rows in each local shard
instead of aligning them once over the global sequence. Plain Wan and
WanDancer have different expansion rules; borrowing them without proving
SCAIL's complete combined-stream layout would be another silent math change.

**Fix:** use the stock scalar-timestep SCAIL path or topology `single`. A
custom sampler that emits multiple timestep rows needs a specified
full-sequence expansion and an identity and fidelity campaign before the
distributed guard can be relaxed. This is separate from [#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight)'s activation
footprint refusal.

## 74. `cluster.toml` is rejected for an unsupported key

**Symptom:** `dgxm` or the Init node raises `ClusterConfigError` naming the top
level, `[cluster]`, an indexed `[[hosts]]` entry or `[worker_args]`, and lists
one or more unsupported keys. A `[fabric.<name>]` profile error instead names
the one key it rejects, such as a name outside the allowed `NCCL_`, `GLOO_` and
`UCX_` tuning variables, and fires even when that profile is not selected.

**Cause:** configuration tables are closed schemas. An unknown root, cluster or
host key is never ignored, because a misspelling such as `client_bnd` or a
misplaced `transport_security` would otherwise select a default in silence and
fail later at attach or security enforcement. Every declared fabric profile is
validated when the file is read, so a typo in an unselected profile cannot
become active at a later profile switch.

**Fix:** move the setting into the table documented in docs/CLUSTER.md and use
the exact listed key. Remove obsolete private metadata instead of renaming it to
look like a supported field. A valid unselected fabric profile stays
unapplied; only its schema is checked.

## 75. Native RDMA fails at QP RTR because the two ends selected different rails

**Symptom:** after an operator enables `rdma_latent_return`, a small latent
returns by actor messaging, but the first threshold-sized or larger native
return fails while bringing an RDMA queue pair to RTR. The two ends report
different selected rails even though normal dual-Spark NCCL renders,
`dgxm doctor`, and passive fabric health can stay green.

**Cause:** without an explicit RDMABuffer rail selector, each transfer end
chooses on its own among NICs with the same best CPU-path score. A
point-to-point dual-rail fabric has no cross-rail path, so a deterministic
address and size choice can put the two QP endpoints on different links. A QP
split of one limits the descriptor to one part; it does not force both hosts to
choose the same rail. The operator-authorized test measured this failure at
exactly 8 MiB; the full gate failed, and the later above-threshold and reattach
cases were not run.

**Fix:** leave `rdma_latent_return=false`, its shipped default. This does not
disable rendering across two Sparks: distributed compute continues over NCCL,
and latent results return through actor messaging. Keep the first native
failure and confirm the run's cleanup and postflight; do not retry that native
registration or read, increase the QP split, or hide, rename, reorder or remove
devices to force a rail. Device hiding is not the supported selector and can
silently make the RDMA, NCCL and diagnostic views disagree. If the failed
attempt reports ownership poison or unconfirmed settlement, follow [#29](#29-large-rdma-latent-return-fails-after-workers-finish-rendering) and do
not reuse that process or numeric handle.

Native RDMA remains **NOT RUN / HOLD**. Retesting requires a stable
TorchMonarch release with the `rdma_ibverbs_target` rail selector. That release
must then pass the selected-rail integrity, ownership, recycle and reattach
checks before native latent return can be enabled. Ordinary dependency updates
do not clear this hold.

## 76. An operator workflow refuses, reports `unknown`, or leaves Worker services stopped

**Symptom:** the panel, TUI or JSON readiness is not green, `dgxm setup` prints
a plan but changes nothing, `dgxm doctor --repair` declines a diagnosis, or
`dgxm update --verify` exits with a receipt and, by design, does not restart
the Worker services.

**Readiness:** inspect the named Worker service, Attached mesh or Render
session layer, and rerun the read after telemetry is back. Do not treat
`unknown` as lifecycle authority. Status and doctor observe different subsets
by design; the meanings and aggregation rules are in
[CONCEPTS.md](CONCEPTS.md#operator-words).

**Guided setup:** fix the named host or prerequisite and rerun the dry run; do
not add `--apply` to make a blocker go away. For an ownership conflict, stop
and inspect the existing unit, enablement and source deployment, then use
verified update where it applies, or retire or move the legacy managed source
by hand, before retrying. `config_lock_unavailable` means another
cooperating config operation is active or the private lock state is unsafe;
let the operation finish or inspect that state, and do not run competing
config writers. `service_activity_unknown` means process enumeration was
incomplete, and `service_ownership_unknown` means setup could not prove the
exact source or unit it would change; neither authorizes config publication.
`verification_unknown` means doctor or smoke transport, evidence or cleanup was
incomplete after apply. In that state setup leaves the config and Worker
services in place, attempts no optimistic compensation, and records a partial
top-level result with `unknown` receipt steps. Inspect the named evidence
before deciding whether to stop, repair or retry. `verification_failed`, by
contrast, is exact negative evidence with settled smoke ownership, and may
compensate the owned services and roll back the config.
After an interruption, keep the private output and inspect the config, unit,
source and receipt state. The complete plan, confirmation, recheck, ownership,
verification, compensation and settlement rules are in
[INSTALL.md](INSTALL.md#guided-multi-node-setup); the profile mapping and Gate
digest effect are in [CLUSTER.md](CLUSTER.md#guided-setup-profiles).

**Safe doctor repair:** if a safety recheck refuses, apply the named non-file
remedy by hand. If receipt publication fails, inspect the config mode first:
permission tightening may already have happened even though doctor did not
rerun and the command returned nonzero. Keep the failure output, repair the
private receipt destination, then run doctor again. The narrow contract is in
[INSTALL.md](INSTALL.md#safe-doctor-repair).

**Verified update:** use the receipt's last stable step to find the uncertain
boundary. After an ambiguous stop, switch, compensation, start or driver
commit, do not start a possibly mixed cohort by hand; inspect per-host release
links, release identities, pin payloads, and all-rank source and status
evidence first. An interrupt may leave no receipt, so keep the private terminal
log. Smoke cleanup owns only its ProcMesh and never calls `hosts.shutdown()`;
attach auto-heal can restart an unhealthy Worker service within the lifecycle
authority already granted. Recognized legacy units are migrated
under an installation journal; an edited unit or override is not adopted.
If ownership or installation readback is unknown, inspect the captured unit,
source link and journal before retrying. Keep the retained controller snapshot,
prior releases and transaction evidence. A definite `failed` step stays
eligible for stop plus guarded compensation and restart; schema-v2 `unknown`
produces top-level `partial` instead and ends mutation. An unknown stop,
activation, target start, compensation or prior restart runs nothing more; an
unknown doctor, cluster smoke, exact readback or driver commit after target
start runs one `fail_closed_stop` and nothing more. Inspect each host before
you start anything. Preconditions and settlement are canonical in
[INSTALL.md](INSTALL.md#updating).

**Receipts:** a disclosure-reduced receipt is not automatically safe to
publish. Review it before sharing, and keep full private logs for omitted
details. The default and custom path rules, how the `setup_compensation`
counts tell an unchanged config from a rolled-back one, and every no-receipt
and partial-publication case are in [INSTALL.md](INSTALL.md#operator-receipts).

The [operator checks](VALIDATION.md#operator-checks) cover readiness, repair
(including refusal during a conflicting operation), guided setup and its
two-rank smoke test, and verified updates on the colocated two-Spark layout. A
stopped Worker correctly blocked status and the Doctor Worker-row adapter, and
the same service was then restored; this was not a full Doctor run. The record
includes update successes, refusals before changes began, and rollback results.
These results do not change
[#75](#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails):
native latent RDMA stays off by default and on HOLD, while ordinary dual-Spark
NCCL rendering and actor-message latent return stay available.

## 77. A Chroma, Omnigen2, or Z-Image USP render refuses a mask or sequence length

**Symptom:** a sequence-parallel render raises `UnsupportedModelError` saying
that an effective attention mask cannot be applied, or that a sequence length
is not divisible by the sequence-parallel degree and the adapter requires
exact divisibility. The refusal occurs before distributed denoising.

**Cause:** the USP kernel takes no arbitrary attention bias, so Chroma and
Z-Image cannot keep a real caption or joint attention mask on a sharded
forward. Separately, an adapter with no named evidence for excluding a
synthetic divisibility-pad row from attention refuses a stream that does not
divide the degree: padding and trimming only at gather would leave the pad row
as a real key in every block, so the runtime refuses instead of returning a
finite output whose accuracy has not been checked. Only Omnigen2 refuses this
way, on its one joined ring stream. Chroma, Flux, Flux2 and LongCat each
declare a pad-exclusion check: they pad the stream and remove those rows before
every kernel call (docs/ADAPTERS.md). The Flux, Flux2 and LongCat tests passed
every check for the tested padded shapes; Flux 1 Schnell has no leg of its own
and stays unmeasured on a padded stream.

**Fix:** for a Chroma effective mask, use `cfg2` where the graph has a real
batched cond/uncond call, or a true one-GPU run with `mode=local` and
`gpus_per_host=1`; stock Chroma attention honors that mask.
Z-Image has no supported topology for an arbitrary mask, because its stock
Lumina forward clears `cap_mask`. Use mask-free conditioning; standard
asymmetric prompts under pure cfg2 take the exact trim path, but do not
treat `single` as proof that an arbitrary mask was kept. For an unmasked USP
render, choose a topology whose SP degree divides every applicable stream. The
error names the family and the affected stream with its length, dimension and
degree: Omnigen2 checks one joined-token stream and is ring-only. Do not drop
a real mask or add prompt tokens to silence the guard. A padded shape needs an
exact pad-row exclusion contract plus a named fidelity campaign, which Chroma
and the three other flux-family members carry. That contract has one limit:
the exclusion applies at a full-sequence point only Ulysses has, so a padded
stream on ring or hybrid raises the waivable class K `ring_pad` card
(entry [#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid)) instead of rendering. It fires before any collective, so the
render stops instead of hanging. These divisibility refusals are not
accuracy-waiver cards.

## 78. The dashboard records `UNKNOWN`, or mesh reset says busy or outcome unknown

**Symptom:** after pausing or scrubbing `dgxm top`, a slow live response does not
appear. A malformed poll creates an `UNKNOWN` tick, or only the run block says
`run story unavailable (Type)`. The retained-pool number can hold its previous
value while the driver is down. In the browser, **Reset attached mesh** can say
`attached mesh busy; wait`, `reset already active; wait`, or `mesh outcome
unknown; inspect` instead of quickly returning to its idle label.

**Cause:** each of these fails closed: none publishes what it cannot prove. A
poll started before timeline navigation is stale even if the user returns to
`LIVE` while it is in flight. Normalization and run-narration failures are
isolated, so stale READY data or one broken panel cannot replace current
uncertainty. The local retained-pool scan has a 0.5-second budget and keeps its
last complete value instead of publishing a partial `/proc/*/smaps` sum. Mesh
reset runs one at a time and is rate-limited; a timed-out HTTP request does not
prove that its background lifecycle operation stopped, so the gate and visible
outcome stay until that operation settles.

**Fix:** for the TUI, return to `LIVE` and wait for the next fresh poll. Treat
`UNKNOWN` as missing evidence; inspect `dgxm status --json` and the driver log
instead of assuming the previous green state still applies. A narrator-only
error does not invalidate the other panels. For mesh reset, wait while a render,
lease, or already accepted reset is active. A reset refused for live work stops
nothing, and the `/dgxm/recycle` route answers it with HTTP 409 and a retryable
result naming that work. Do not send repeated resets after a
timeout or unknown outcome. Inspect lifecycle status and retry only after the
accepted operation has settled or a definite non-timeout ordinary failure
grants a later fresh cleanup attempt. A reset skips the workers' model unload,
and the process stop frees the weights instead, so a stop that
times out leaves those processes holding their weights until `dgxm reap` or a
worker service restart removes them. An unexpected reset failure is sanitized
in the browser; use the driver log for private diagnostic detail.

## 79. A cross-Spark render refuses because the ranks disagree about weight residency

**Symptom:** a world-2 render stops with a class C refusal saying the ranks
disagree about weight residency: at least one rank holds the model only
partially. Every rank raises the same message. Just before it, each box's
worker journal carries one line, `partial-load guard: local full_load=...
world=...`, and a box that reported `False` is short. ComfyUI's own line above
it says `loaded partially` on a short box and `loaded completely` on a whole
one.

**Cause:** ComfyUI decides how much of a model it can hold when sampling
starts, not when the file loads, and it decides per box against the unified
memory free at that moment. Two Sparks rarely have the same amount free, so one
can drop into the partial-load regime while the other does not. A partially
held rank streams and casts its offloaded blocks per step, so the ranks compute
different weights for the same step. The render does not crash: its ranks drift
apart. Measured on an LTX 2.5 bf16 transformer with a baked LoRA stack under
forced-stock residency: the cross-rank latent canary read 2.27e-03 relative
drift. The usual source of the pressure is a bf16 checkpoint whose LoRA stack
is baked with ComfyUI's full weight backup still held, which costs about twice
the checkpoint on every box. Under an FSDP preset, an older shard build itself
caused the pressure: it copied this rank's rows to the device straight from the
mmap'd checkpoint, the driver broke the mapping's copy-on-write under its pin,
and a family with any file-backed leftover (a replicated fp32 island, a scalar,
a matching buffer) held its whole shard again as anonymous host pages until the
model was dropped. The current build sends rows through a pinned bounce buffer
and the mapping is released before ComfyUI's load. The worker journal's `FSDP load: checkpoint mapping released` line names the copy-on-write bytes, and a
non-zero figure there means that defect is back, not a capacity fact.

**A stock resident is placed at load time.** ComfyUI sizes a stock model's
device copy at sample time, while the loaded file still sits on the host, so a
load the stock estimate allowed could partially load there. The store now runs ComfyUI's own full load when the stock load
finishes, and the stock memory estimate charges that placement's whole peak (2.1 times
the file plus the floor), so a box without the room refuses at
`cross_rank_capacity` before anything loads. A stock resident with a LoRA stack
baked through ComfyUI's weight backup (hot-swap residency, `lora_low_rss` off)
places at sample time instead and can meet this guard.

**Fix:** reduce memory pressure, then rerun. Load an int8 or fp8 checkpoint
in place of bf16; turn `lora_low_rss` on so the stack bakes and the weight
backup is freed; or run the shape on a single box, where a partial load is
ComfyUI's ordinary correct behavior and nothing can disagree with it. The
refusal is not waivable and has no card: going ahead would authorize ranks that
compute different weights. The mesh survives it. The lease is consumed, not
abandoned, so the next render dispatches on the same fleet with no recycle.

ComfyUI's own sample-time estimate can also exceed the box before any rank
loads, and then ComfyUI offloads every weight on that box (entry 91 shows one
such checkpoint). The render-memory preflight refuses that case before
dispatch, and its refusal points here. It refuses at world 1 as well, because
the estimate is per box, so the single-box remedy above does not answer it:
reduce the resolution or use a smaller or lower-precision checkpoint.
`DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver is its only bypass.

**When the short rank holds a slab or an FSDP shard,** read the load lines
first. Both builds put every weight on the compute device without ComfyUI's own
load running, so ComfyUI's weight ledger would read zero and it would ask for a
whole model of free memory at sample time for bytes the box already holds.
The fresh load declares the residency to ComfyUI, and the
worker journal carries two lines for it at load time. ComfyUI writes the first
from inside the load the declaration asks it to run: `loaded completely; N MB
loaded, full load: True`. The pack writes the second: `resident ledger:
declared N GiB of slab weights to comfy...` (`fsdp shard weights` for a
shard). Where nothing was declared there is no ComfyUI load line, and the
pack's line reads `resident ledger: ... left undeclared (reason)`, naming why.

A declared rank prints no load line when sampling starts, because ComfyUI takes
its early return for an already-loaded model and that return logs nothing. The
sample-time evidence is the guard's own line, `partial-load guard: local
full_load=True world=2`, on each box.

Three shapes tell the cases apart:

* `loaded partially` at sample time with no declared line above it, and a
  loaded figure well under the model size: the zero-ledger case. The `usable`
  figure on that line is what ComfyUI had left to spend, not the size it asked
  for. Check why the declaration was skipped.
* `Unloaded partially: X MB freed, Y MB remains loaded` at sample time after a
  declared line. ComfyUI has the weights but not the working set, so it gave
  weights back. The fixes above apply: a smaller checkpoint, a lower
  resolution, or more free memory on the short box.
* a declared line, no load line at sample time, and `local full_load=True` on
  both boxes. Nothing is wrong; this is what a declared render looks like.

## Power-user escape hatch

`monarch-tui` (ships with torchmonarch) attaches to a live mesh to inspect
actor state when the node surface is not enough.

## Never disable DynamicVRAM in stock ComfyUI on a Spark

Measured on a single GB10: krea2-class 24.5 GiB bf16 DiT plus 7 LoRAs, 20 steps
at 1024, identical flags, page cache prewarmed.

| | dynamic (default) | --disable-dynamic-vram |
|---|---|---|
| warm render | 76 s | 238 s (3.1x slower) |
| GPU-attributed after load | 40.3 GiB | 8.6 GiB |
| retained allocator pool | 0 | 19.7 GiB |
| anonymous memory | 5.3 GiB | 41.8 GiB |

Without comfy-aimdo's DynamicVRAM, ComfyUI falls back to its legacy VRAM
estimator, which misjudges unified memory and drops to offload streaming
without saying so: only about 8.6 GiB stays resident, and the rest of the model
moves through the offload path on every step. On a Spark the disable flag does
not mean "classic full load"; it means accidental lowvram mode, at 3x the
render time, plus the retained one-model-sized allocator arena that the legacy
load path leaks (the "pool" segment in `dgxm top`). The arena returns to the OS
only when its process exits, which in stock ComfyUI means a restart. For
dgx-monarch workers the ClearVRAM node's `recycle` level and the panel's
**Reset attached mesh** button do that: they stop the actor processes, which
frees the model and the arena together, and the next render attaches a fresh
fleet.

DynamicVRAM can run inside Monarch workers. Its background threads were tested
starting, running beside concurrent endpoint calls and joining cleanly. The
integration depends on funchook patches to the loaded libcuda library. Enable
it with the Init `comfy_managed` widget; it is opt-in, supports no LoRAs, and
has not had a sustained multi-prompt test
([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)).

The default worker-side answer is the Init `slab_weights` widget (zero-copy
shared-memory weight residency, docs/VALIDATION.md): with it on, a worker shows
pool 0 like stock DynamicVRAM (0.0 vs 20.7 GiB dirty in the same process),
loads in about 20 s, renders with identical output, and returns the model's
memory to the OS as soon as it unloads.

### Single-box LoRA swaps under stock DynamicVRAM are fast and exact

Measured on the same setup: a strength change re-renders with about 10-14 s of
overhead, and the swapped image is pixel-identical to a fresh load at the same
strengths. Compare pixel arrays, not PNG file hashes: ComfyUI embeds per-run
metadata in the PNG. Single-box users do not need this stack for memory or swap
speed; the README's [What we measured](../README.md#what-we-measured) says what
the pair gains.

## 80. An LTX attenuated guide refuses on ring or hybrid

**Symptom:** an LTX 2.5 image to video or first-and-last-frame render with a
guide strength other than 1.0, or with a connected `attention_mask`, stops with
a class P refusal before any pixels. It names a pure-ulysses topology and
`single` as the alternatives. Pure ulysses renders the same graph.

**Cause:** LTXV Add Guide records one entry per guide. When a guide asks to be
weakened or strengthened, comfy turns those entries into an additive
self-attention bias so the noisy tokens and the guide tokens attend each other
at a chosen weight. Two settings ask for it: a strength other than 1.0, and a
connected `attention_mask`.

Comfy never builds a dense mask for that. It splits the query axis into the
noisy tokens, the tracked guide tokens, and anything after them, then attends
each group against the whole sequence under its own weights. Ulysses
re-expresses that exactly: after the head scatter every rank holds the whole
token axis for a subset of heads, and the weights do not vary by head, so each
rank applies the same groups to its own heads. Ring cannot. Its ranks see the
keys one block at a time, and the kernel that walks those blocks takes no bias
argument, so there is nothing to hand the weights to and nothing a waiver could
grant.

**Fix:** run a pure-ulysses topology. On a two-rank fleet that is `uly2`, which
is also what `auto` picks for LTX at every canvas size. For a stock one-GPU
run, use topology `single` (`mode=local` with `gpus_per_host=1`), where comfy
applies the bias itself; `cfg` topologies also run stock attention per rank, so
they take an attenuated guide unchanged.

Setting strength to 1.0 on every LTXV Add Guide and leaving `attention_mask`
unconnected works everywhere, including ring: comfy then builds no bias.

The concat route has no such limit: LTXV Image To Video writes the frame into
the latent and sets no conditioning keys, so its own `strength` widget is a
noise-mask value, not an attention weight, and every value shards.

The refusal is rank-symmetric and decided from the request, so it retires the
sample lease consumed: fix the graph and queue again on the same fleet, with no
Recycle. This entry is a launch constraint, not a defect, and carries no
accuracy-waiver card.

## 81. Explicit slab residency still reads quarantined after a LoRA-only gate FAIL

**Symptom:** `slab_weights=on` refuses for one model and LoRA stack, and the
log names a persisted quarantine, but no check ever ran slab residency for that
combination. Rescue consent for the same combination refuses too.

**Cause:** older builds wrote an identity-gate FAIL verdict onto the
explicit-slab sibling contexts whether or not the check had exercised slab
residency. A failure that came from the lazy-swap leg alone therefore recorded
a FAIL against a capability nothing had tested. Those rows are sticky for the
same artifact bytes and terminal for the automatic path, so the quarantine
outlives the check that wrote it.

**Fix:** the gate writes a FAIL onto the sibling contexts only when the
cross-residency leg itself diverged, the only leg a check runs under slab
residency. Otherwise the preflight `RETESTING` rows stand: the sibling stays
denied and re-gates on next use.

**Recovery for a row already on disk:** capability-scoped ledger rows bind to
the running gate protocol and package version, so an upgrade that changes
either one retires the stale row. The rescue consent path reads the same
binding, so a retired row stops blocking the capacity ladder when it stops
governing the gate.

Consent ignores the rows this entry is about whatever version wrote them: a
stamped sibling row whose check never diverged on the cross-residency leg
records nothing about slab residency. Older releases read them, which trapped
the operator: the rescue consent is what lets the checkpoint load
slab-resident, so refusing it also blocked the one explicit check that would
overwrite the row.

On a version-pinned install, a stale row the gate still reads clears with that
check: add the Identity Gate node to the graph for that exact model, LoRA
stack, and residency policy, and run it once. An explicit check re-tests and
overwrites the row, where the automatic path treats it as terminal and never
retries.

## 82. A MiniMax H3 guide refuses at the loader node for its encode

**Symptom:** a graph with Add Guide for MiniMax H3 refuses before the model
loads, and the refusal charges a guide encode of 94 GiB at 1344x768, 67 GiB at
672x384, or, at 448x256, between 17 and 48 GiB depending on how many frames the
wall could prove. The image on that input is a single still.

**Cause:** Add Guide resizes its frames to the target canvas and encodes them
with the video VAE before the sampler runs. That encode grows faster than the
canvas area does. Measured on one DGX Spark, with the VAE already resident: one
frame at 1344x768 cost 13.6 GiB, five frames at 448x256 cost 16.2 GiB, five at
672x384 cost 66.9 GiB, and five at 1344x768 never completed, past 94 GiB.
Length was swept at the one canvas where a long clip completes, 448x256: 16.2
GiB at 5 frames, 33.4 at 22 and 47.5 at 39. Roughly a gigabyte a frame, so
length matters as much as a resolution step. The frames arrive on a link, so
the graph does not say how many there are.

**Known guide length.** A Load Image node names its file, so the capacity
check opens that file before the render and counts its frames. One frame takes
the measured still charge and the render proceeds. The check uses file
contents: an animated PNG, WebP or GIF reports its real count and takes the
clip charge, and a video, a missing file or a damaged one does not open at
all, which also takes the clip charge. Put a scaler or any other node between
the loader and the guide and the file is no longer named, so nothing is proved
and the clip charge stands.

**Fix when nothing can be proved:** start ComfyUI with
`DGXM_H3_GUIDE_FRAMES=1`. The capacity check then prices the measured still,
14 GiB at 1344x768 or smaller, and charges every other term as before. Any
value under 5 prices a still, because a batch that short anchors its first
image alone. A value of 5 or more prices a clip. Above 448x256 a clip
declaration buys nothing: the canvas price stands. At or below 448x256 the
declaration also picks a length bucket (5, 22 or 39 frames), so a short
declaration reduces the estimate from the 39-frame worst case used for an
unreadable link. A frame count proved from the file outranks the declaration
either way.

That declaration is a statement about the whole driver process, so a driver
started with `=1` believes it for every guide whose length nothing proves. Set
it for graphs whose sources hide their files, and leave it unset otherwise.

**Fix for a clip:** shrink the canvas *and* let the capacity check see the length. Five
frames at 448x256 fit beside the DiT and the text encoder, but only when a Load
Image node names the file or `DGXM_H3_GUIDE_FRAMES` declares the count. An
unread link at that canvas is charged the longest clip a run measured, 39
frames, at 48 GiB, and can still refuse. At 1344x768 no clip fits on this
hardware, which entry 83 covers.

`DGXM_DISABLE_DRIVER_PREFLIGHT=1` also clears the refusal, but it turns off the
weight, text encoder, VAE, reference and activation charges with it.

## 83. A multi-frame guide clip at 1344x768 takes the whole host down

**Symptom:** a guide clip at the full H3 canvas does not raise, does not
degrade, and does not finish. Load average climbs past 200, MemAvailable sits
near zero for tens of minutes, and the kernel eventually kills the process
with the highest `oom_score`, which is usually not the one that filled the
memory.

**Cause:** two things compound. The encode itself needs more than 94 GiB at
that geometry, which is more than a 121.6 GiB box has beside a DiT and a text
encoder. And ComfyUI's tiled-VAE fallback, built to rescue an encode this size,
never fires: it triggers on a caught CUDA out-of-memory error, and on GB10
unified memory there is no clean CUDA OOM to catch. The allocation succeeds
against host memory, the host exhausts, and the box thrashes instead of
degrading.

**Fix:** respect this measured hardware limit. Anchor a
still at this canvas, or move a real clip to 448x256. Entry 82 covers the
charge that refuses the clip before it starts, on a render queued from the
browser. A script that imports the loader node and calls it directly hands the
wall no graph, so that charge never runs and the host thrashes as above.
Entry 85 covers that caller.

## The Spark failure catalog

This catalog lists recurring GB10 failure modes from the community forums and
this project's own history. `dgxm doctor` checks the safe subset. It runs no
load test. The one GPU kernel it runs is a small Sage probe, which it runs only
when no other process holds the GPU (entry [#89](#89-doctors-sageattention-row-fails-its-kernel-probe-or-counts-fallbacks)). The other checks belong in a
maintenance window the operator controls.

### PD-stuck state

One USB-C power-delivery bug shows three ways: silent hard power-offs under GPU
load, renders suddenly ~3x slower after an OTA update, and "GPU reads 11 W at
44 C while a workload is running." A hard power-off leaves nothing in any log:
the absence of a kernel panic, Xid or OOM record across an unclean boot is the
signature. After a crash the GPU can stay pinned at ~650 MHz and 5-9 W across
normal reboots. `dgxm doctor` warns when the last 30 boot records hold more
than one reboot with no clean shutdown before it (row `unexpected shutdowns`).
It discounts one, because the oldest reboot in the window often lacks its
pair. Doctor reports its `GPU clock under load` row as **NOT RUN** and gives no
PD-state verdict. During a workload you already authorized, compare
its clock and power readings with a healthy host; do not start a diagnostic
load beside a render. The community-confirmed remedy is a full cold
power-cycle: unplug the power brick at both ends for at least 30 s, then boot.
A plain `reboot` does not clear it. Some users also cap clocks
(`nvidia-smi -lgc`) as prevention. A cap costs speed: the head box of this
project's pair runs under a 2100 MHz stability cap, which cost up to about 4
percent of render time when measured
([BENCHMARKS.md](BENCHMARKS.md)).

### ConnectX-7 / cross-Spark fabric

* Dead rail after boot (link trains DOWN, phys LinkUp): reboot the affected
  box. There is no software fix; do not spend time on driver restarts.
* NIC vanished entirely (lspci shows the bridge but no Mellanox device;
  dmesg: "retraining non-functional downstream link"): full power cycle,
  not a reboot.
* Bandwidth halved with both rails UP (nccl-tests busbw ~15-17 GB/s
  instead of ~22-24): a known regression when the SoC firmware moves from
  the 10500 family to 10600. Check `fwupdmgr get-releases`; the community
  remedy is `fwupdmgr downgrade` to the prior SoC firmware.

### Everything else

* CUDA "operation not permitted" after a container idles for hours:
  restart the container; no host reboot needed.
* Sage or Triton silently ~20x slower with correct launch flags: the kernel
  fails on every call and comfy falls back to torch attention, logging
  `Error running sage attention` each time. Two common causes are missing
  Python development headers, which break Triton's JIT (install the matching
  `python3.x-dev`; doctor row `python dev headers`), and a stale-ABI Sage build
  after a torch bump. Doctor runs the kernel and counts the real fallbacks in
  `~/comfyui-monarch.log` (entry [#89](#89-doctors-sageattention-row-fails-its-kernel-probe-or-counts-fallbacks)). For a ComfyUI that logs elsewhere,
  count the lines that carry `Error running sage attention` and not
  `Unsupported head_dim`: a growing count means Sage is not running, whatever
  the flags say.
* Memory exhaustion can hard-lock the whole OS rather than OOM-kill the
  offender (still possible after firmware improvements). Prevention on this
  stack: `lora_low_rss` (on by default on a Spark) replaces comfy's full
  weight backup with references into the checkpoint file, and
  `uma_reserve_gb` warns when a model load or swap leaves less than that many
  GiB available and also refuses a driver-side weight load
  that would eat into it ([#52](#52-driver-side-footprint-preflight-refused-a-render)). For Docker co-tenants, set explicit --memory
  limits.
* One board sensor ("temp6") sits outside the fan curve and can reach 90 C+
  quietly; `dgxm doctor` warns on it, reading `/sys/class/hwmon` directly, so
  the check needs no package and runs on a stock Spark (entry #59).

## 84. A sol-attn render refuses with a waiver card, or stalls on its first call

The `SOL_ATTN_TAU*` attention kernels (tau 0.6, 0.7 and 1.0 ship as named
values; lower tau keeps more key blocks and runs closer to exact attention, and
the three elapsed times sit within about 4 percent of each other) are
approximate by construction and not identity preserving. A persisted panel
consent covers its combination, not one tau: switching the widget between
shipped taus on the same model and topology re-uses the standing grant, and the
ledger's use row records which tau ran. Measured on a two-Spark pair, the same
seed and graph give a coherent, sharp video of a different composition from
exact attention. So it refuses by default; that refusal is intended, not a
fault.

Accept it the way any class-K waiver is accepted. In the browser the sidebar
offers a card; headless, submit once, let it refuse, then read the pending row
and accept it:

```
curl -s localhost:8191/dgxm/consents            # read `pending`
curl -s -X POST localhost:8191/dgxm/consent \
  -H 'X-DGXM-Action: consent' -H 'Content-Type: application/json' \
  -H 'Origin: http://localhost:8191' \
  -d '{"action":"accept","key":"<key>","id":"<id>"}'
```

Then re-queue. Both ranks log KNOWN-WRONG MATH PROCEEDING UNDER WAIVER, the
driver logs RENDERED UNDER WAIVER, the output carries
`_dgxm_rendered_under_waiver`, and the gate ledger gets one `use` row per
guard. `DGXM_WAIVE_KNOWN_WRONG=sol_attn` on the driver is the headless
equivalent of the click.

**The first call of each shape is slow, and that is not a hang.** The kernel
compiles per shape and per configuration on first use, seconds rather than
milliseconds, worst at long sequences. A first render pays it once per shape,
and a resolution or duration change pays it again. Wait it out before
diagnosing a stall.

**Other refusals you may see, and what each means.** A render at
sequence-parallel degree 1 refuses because the model keeps ComfyUI's own
attention there, so the kernel would never run under a name that says it did.
A ring or hybrid topology refuses: the kernel returns no log-sum-exp for ring
ranks to merge. A family other than MiniMax H3 refuses with its own reason. A
box without the package refuses naming that box: the driver checks every rank
before it dispatches, because a rank raising inside a collective abandons the
render and strands its peers.

**The kernel is optional and is not a dependency of this project.** Install it
on every box in the mesh:

```
pip install 'sol-attn @ git+https://github.com/NVlabs/Sana@71350fae#subdirectory=techniques/sparse_backends' apache-tvm-ffi
```

`apache-tvm-ffi` is required: the fast kernel needs it at run time, and the
package's own requirements do not pull it in. Without it upstream dispatch
selects the fast backend and then fails at call time, so a box that lacks it
refuses as if the package were missing.

**On a DGX Spark the fast kernel needs one dispatch entry.** Upstream matches
compute capability exactly and ships no `(12, 1)` key, so a Spark silently
runs the portable reference; every published Spark figure this project found,
including NVIDIA's own GB10 configuration, ran on that slow path. This project
adds the key when the device really is `(12, 1)` and both the cutlass DSL and
`tvm_ffi` import, and logs which backend it got. The DSL is the CuTe compiler
and no render refuses without it (the `dgxm doctor` row `sol-attn install`
warns), so a worker log reading `backend=triton` on a Spark usually means it
is missing: `pip install nvidia-cutlass-dsl==4.7.0` (the version measured
here) on every box, then read the backend again. Do not "fix" any of this by
forcing the reported capability to `(12, 0)`: the compiler then targets
`sm_120a` and every call dies with "no kernel image is available for execution
on the device". The gap is easy to miss: the two backends differ by about 20
percent end to end on the same graph, and two earlier runs recorded the fast
path while running the portable one. [VALIDATION.md](VALIDATION.md) carries the
timings for the shipped recipe and the earlier backend misidentification. Use
the logged backend to identify which implementation ran.

**A first-use identity check does not run under this kernel, and that has a
cost.** The check compares residency, not kernels: every leg reuses one frozen
request, so both legs would run sol and the comparison could only measure the
kernel against itself, where a PASS would record known-wrong math as proven. So
the gate logs that it did not run and writes nothing, and no residency setting
is quarantined, because a kernel choice does not implicate slab or lazy-swap
residency. The stated limit that follows: with no PASS for that context, a sol
render on a risky combination (a LoRA stack, or slab residency requested) runs
under stock residency, which means baked LoRA and no slab. Renders with no LoRA
and no slab request are unaffected. That limit stands until a cross-kernel
measurement verifies a tau. That instrument is separate from the check, and
this project does not ship one.

## 85. A script or harness fills the box although the loader wall exists

**Symptom:** a leg driven by a Python script, a benchmark or a test loads a
model that the same graph would have refused in the browser. MemAvailable falls
to near zero during the driver's load and encode stage, the box thrashes, and
the kernel eventually kills some other process. The driver log carries one line:
`loader footprint preflight: no readable graph, headless call`.

**Cause:** the loader-site capacity check prices the text encoder, the VAEs, reference and
guide encodes and the unspent render activations from the graph ComfyUI passes
with a queued render. A caller that imports the node classes and calls `load()`
passes no graph, so the capacity check charges the checkpoint weights and nothing else. On
a family whose driver stack is tens of gigabytes before any media, the term that
would have refused is the missing one. Nothing has failed when the line appears:
it describes the caller, not the checkpoint, and it is printed once per model
family per driver process. The capacity check prices only the families it holds a
driver-stack profile for, so a headless load of any other family prints nothing:
the capacity check never armed for it and had no stack to leave unpriced.

**Fix: gate capacity in the caller.** A direct node call cannot see the graph, so
the harness owns the budget. Run one model-stage configuration per process, wait
for the previous leg's process to be gone and MemAvailable back near idle before
launching the next, and record the reading each leg launched against.

**Or submit the graph.** A harness that has an API-format graph should POST it to
the driver's `/prompt` endpoint instead of importing the node classes. That path
carries the graph, so every term is priced and every refusal is typed and
readable.

Entries 82 and 83 cover the guide encode and the hardware exclusion behind it.

## 86. A render error says the fleet stalled, or a worker refuses with a poisoned CUDA context

**What happened.** A rank died inside the render: a sticky device fault (a
misaligned address or illegal access class CUDA error), an out-of-memory kill,
or any crash that never sends its reply. Its peer then blocks inside the next
collective, so no rank answers and the driver's wait would otherwise sit
silent until the hard timeout while the prompt queue holds.

**The stall error.** The driver compares the fleet's progress stream against a
stall budget (default 600 s; `DGXM_SAMPLE_STALL_TIMEOUT_S` overrides it, and 0
disables the guard). A fleet silent past the budget is declared stalled with a
typed error, the mesh is marked defunct, and the next render respawns workers.
The eviction completes its own teardown, so the session heals in place and no
ComfyUI restart is needed. The respawned workers hold nothing, so the next
render loads the checkpoint again, and the driver stops crediting them with
weights they no longer hold. Where a capacity estimate reads that credit, that
render is charged for the load it now has to pay for and can refuse where the
one before it did not. Raise the budget for graphs whose single step or cold
load takes longer than the budget without stalling. Each worker journal
(`journalctl --user -u dgxm-worker.service`, per box) names what its rank was
doing when the stream stopped.

**The poisoned-context refusal.** A sticky fault outlives its exception: every
later kernel in that worker process fails or corrupts. After any endpoint
failure the worker probes its own context once, and a dead context latches a
refusal on every later GPU endpoint in that process. The refusal is untagged
because the decision is host-local and a mixed fleet can leave the healthy rank
mid-collective, so clearing it costs one Recycle (sidebar button, or
`POST /dgxm/recycle`), which respawns clean workers.

**What to look at next.** The first failure in the poisoned worker's journal is
the real defect; the stall and refusal report its consequences.

## 87. A log line says the abandoned-lease wedge healed and the render paid a respawn

**What you see:** after a worker-side crash without a typed refusal, a later
render logs that it healed an abandoned lease and pays for fresh actors plus a
cold model load.

**What to do:** diagnose the original crash in the failed worker's journal.
The automatic reset addresses abandoned state, not the underlying defect. If
the retry also fails, follow the reported recovery; it does not loop.

**Automatic reset is limited:** it runs at most once per submit, only when
abandonment is the sole obstruction, with no active sample lease or pending
RDMA read. The ProcMesh stop must be observed before retry. Active work is left
alone. An unconfirmed earlier stop keeps replacement blocked; supervision text,
a timeout, or a stalled channel is not proof that processes exited. In a
recovery test, a reported-dead worker remained alive for eleven more minutes.

**After worker death plus a timed-out stop:** the driver requests read-only
actor liveness on every configured host, over SSH remotely and locally on the
driver host. It checks actors, zombies, live ledger rows, and unidentified
processes. Unidentified includes empty command line plus environment, and a
process holding >=4 GiB with unreadable environment and no Python module in
its command line. A named actor module is counted as an actor. Zombies count
as gone; stopped processes remain alive.

All hosts must report gone before the fleet is considered gone. Any live
process keeps it alive; incomplete evidence stays unknown. Either blocks
replacement. The check runs at Init attach, render liveness, and Reset:

* A `gone` result younger than 30 seconds can be reused; retirement rechecks
  its age.
* Each handle is probed at most once every two seconds. Alive/unknown results
  are not cached beyond that interval.
* Without configured hosts, one shared thread reads local procfs. A stuck
  read leaves later probes unknown instead of creating more threads.

A fleet-wide gone result retires the cached handle and permits one fresh fleet;
it never issues a second stop. If replacement setup fails, the new handle is
blocked and the error identifies its liveness-proven predecessor. Hardware
found that the pinned Monarch driver could not reattach after its fleet
changed; that replacement attach therefore tries once and names the restart
recovery in [#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out).

Logs show every result as `liveness probe host=... answer=...` and
`liveness fold answer=...`. Release adds
`replacement latch released on liveness proof`, host names, and abandoned count.

**A timeout without worker death is different.** An operator Reset, clean
detach, or stall-guard eviction whose stop times out cannot be released by this
proof and gets no second stop. Confirm the workers are gone, then restart
ComfyUI. The render, panel, and Reset error distinguish this state from the
worker-death recovery above.

## 88. The first render of an FSDP topology refuses with a capacity figure and no proof

**What you see:** first-use FSDP proof refuses with a capacity estimate before
it can establish PASS. Clean-reload proof unloads/reloads every rank and
compares two renders; on UMA, retained allocations can prevent that reload.
This says nothing about topology or weight correctness.

**What to do:** free memory outside this render's shards, such as another
model or the driver's text encoder (Clear VRAM, `include_driver=True`). A
smaller bf16/fp16 checkpoint may help; fp8/int8 still pays the full-file bound
below. Reset alone does not help because the estimate runs after your next
initial load. To render without proof, set Init's `auto_gate=off` and accept
its ordinary ungated status.

**Read where the refusal occurred:**

| Stage | Resident state and retry cost |
|---|---|
| Initial preflight | No shards unloaded or proof render run. The ledger records INCONCLUSIVE with `cross_mode: CAPACITY` and measured bytes; no verdict is cached. Free memory and requeue to re-estimate. |
| Reprice after baseline, or capacity refusal during proof | Shards have already been unloaded and a baseline may have run. Requeue repeats proof and its render cost. Only initial preflight preserves the starting residency. |

The least-free rank governs. The driver uses the worker launch estimate:
known-world bf16/fp16 needs `(1/world + 0.08) * file + 5 GiB`; other precisions
or unknown world need `1.2 * file + 5 GiB` (DESIGN.md 5.4). The streaming build
commits shards plus one block; its world-2 0.58 factor exceeds the measured
0.547x high-water on a 60 GiB file. A 60 GiB bf16 Flux2 reload fit beside the
initial shards but failed the post-baseline reprice after the driver's encoder
became resident ([VALIDATION.md](VALIDATION.md#evidence-kandinsky5-image)).

Proof does not replace the entire fleet to reclaim its arena: that would
unload existing residents and break the initial-preflight preservation promise.

**A worker refusal inside proof** can mean the estimate was unavailable
(discrete GPU, loader dtype cast, unmeasurable checkpoint, or missing rank
memory), or that an initially fitting load was followed by another load while
sampling reservations remained. The driver logs every applied estimate, even
when it fits. Such ledger rows retain `cross_mode: CAPACITY` and classify the
refusal instead of recording driver-measured bytes. The estimate covers one
load, not proof's peak memory.

See [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) for ordinary stock-load
capacity and [#86](#86-a-render-error-says-the-fleet-stalled-or-a-worker-refuses-with-a-poisoned-cuda-context)
for a stall rather than a refusal.

## 89. Doctor's sageattention row fails its kernel probe, or counts fallbacks

**What happened.** `dgxm doctor` runs a real SageAttention
kernel on a small synthetic tensor (head_dim 128, then 64) in a bounded
subprocess of the same Python, whenever a CUDA device is present and no other
process holds the GPU. A FAIL on that row means the installed `sageattention`
package cannot run on this torch. The usual causes are a build from before a
torch bump (a stale ABI) and a PyPI wheel that shadowed the local build when a
custom node's requirements pulled it in. ComfyUI does not stop on either: it
logs `Error running sage attention` and samples on torch attention, slower.
That is what the second row counts from the driver log
(`~/comfyui-monarch.log`); lines that carry `Unsupported head_dim` are the
normal per-shape fallback and are not counted.

**What to do.** Rebuild sageattention against the installed torch in the same
venv (`"$COMFY_PYTHON" -m pip install --no-build-isolation --no-deps` on the
sageattention checkout, with `TORCH_CUDA_ARCH_LIST` naming this GPU's
architecture), then run doctor again: the OK row reads `kernel ran`.
Missing Python headers surface as a build failure and have their own row
(`python dev headers`). A `skipped: GPU busy` row is not a verdict: doctor
found a live process on the GPU and left it alone; run it again on an idle
box. A WARN with a fallback count names the log it read, and the count resets
when that log rotates.

## 90. `install-service` or `uninstall` refuses an existing Worker unit

**Symptom:** the command refuses and names the refusal it hit.
`REFUSED_FOREIGN_UNIT` reports that `dgxm-worker.service` is not this
lifecycle's canonical unit. `REFUSED_SETUP_UNIT` reports that guided setup wrote
the unit and points at `dgxm uninstall`. `REFUSED_FOREIGN_DROPIN` names the
files it found in a drop-in directory systemd reads for
`dgxm-worker.service`, each name written from your home directory down, or in
full for one systemd read from the runtime directory. None
of the three makes a source, generation, service, or unit change. Uninstall
runs that ownership proof, widened to accept the setup token, before it stops
a worker or clears a generation record, so a refused uninstall leaves the host
as it found it.

**The drop-in refusal.** systemd reads `dgxm-worker.service.d/*.conf` after
the unit. An override can replace `ExecStart=` while the unit body remains
canonical. `install-service` therefore refuses every file in those directories,
including editor backups, whether or not the unit exists. Remove the files and
rerun. Uninstall instead removes the owned unit and reports the overrides as
`left behind:`; a later install will refuse them.

**Six directories, not one.** The check reads `dgxm-worker.service.d` and
its dash-truncated equivalent, `dgxm-.service.d`, under each of
`~/.config/systemd/user`, `~/.local/share/systemd/user`, and
`$XDG_RUNTIME_DIR/systemd/user` (used by `systemctl --user edit --runtime`).
It excludes general `service.d` overrides, which affect every service, and
root-owned `/etc/systemd/user` and `/usr/lib/systemd/user`, which this user
lifecycle cannot remove. If the effective command differs from the unit body,
inspect `systemctl --user cat dgxm-worker.service` for all applied overrides.

**Cause:** standalone lifecycle commands mutate only an absent unit, a unit
carrying the current dgx-monarch managed marker, or the pre-marker canonical
unit for the current configuration. Two byte shapes count as pre-marker: the
current render, and that render as the previous release's installer wrote it,
one trailing newline longer. A marked unit counts as owned whatever its body,
so a config change can rewrite it. An unmarked manual edit, a different address
or interpreter, a symlink, a non-regular file, or an unrelated unit at that
name is foreign state. So is a unit whose body predates the current canonical
render, such as a hand-hardened copy. Only bytes this lifecycle can reproduce
prove ownership; a running process or listener cannot prove who owns the unit
bytes.

`install-service` also refuses a unit written by `dgxm setup`
(`REFUSED_SETUP_UNIT`): its first line carries the setup token that remote
source sync reads before it writes, and publishing the canonical body would
drop that token. `dgxm uninstall` does remove that unit, because uninstall is
the only teardown guided setup has. Guided setup will not take over a
pre-existing unit either, so removal comes first whichever installer you rerun.
The `~/.local/share/dgx-monarch/src` link guided setup published into its
release slot survives that removal. Source sync refuses a link without its
matching managed unit. Remove the link by hand on every host that
carried the setup unit, in the same pass.

**Fix:** inspect the unit privately with `systemctl --user cat
dgxm-worker.service` and compare it with the intended configuration. Back up
any manual unit. To retire it, stop and remove it yourself, then rerun
`dgxm install-service`; otherwise restore the managed unit or the configuration
that wrote it. Verified update can migrate recognized generated legacy and
hardened units after exact ownership checks. Other hand-edited units still
need manual retirement. Do not force the command past the
ownership refusal.

## 91. A Z-Image render refuses with an upstream gate, or takes the host down

**Symptom:** a Z-Image render stops before any load with a class P refusal
saying the checkpoint is the L2P pixel-space surface and this release carries
no L2P forward. It names no guard and offers no card and no waiver. On an older
build the same artifact does not refuse: the render loads on one box, the
worker journal reads `loaded partially; 0.00 MB usable`, the host runs out of
memory with no OOM kill, comfy's interrupt cannot reach the render, and the
worker has to be killed to keep the box alive.

**Cause:** the checkpoint is the Z-Image L2P pixel-space surface. Its header
carries `local_decoder` and no `dec_net` head. Current ComfyUI has no L2P model
contract, so it reads the header as latent Z-Image and gives that model a
`memory_usage_factor` of 2.8, which prices about 129 GiB of sample-time memory
on a box that holds 121. ComfyUI then offloads every weight, reports zero usable
bytes, and streams and casts every block on every step. Past that point this
release would bind the latent forward onto pixel-space weights, so the image
would be wrong even if the memory held.

**Fix:** use the latent Z-Image checkpoint or the DCT PixelSpace checkpoint;
both are hardware validated and neither is gated. There is no waiver and no env
var: no consent can supply a forward this release does not have. The refusal
retires when ComfyUI carries the L2P model contract (upstream draft
Comfy-Org/ComfyUI#14055) and this release carries the L2P forward. The driver
log carries one INFO line per file, opening `upstream gate:`, that names the
matched key paths, so a record can confirm which spelling the gate read.

## 92. A render refuses with `FsdpGateProofError`, and the retry says the same thing

**What happened.** `auto_gate=first_use` on an fsdp topology runs a clean-reload
proof before your render. Something stopped that proof before it reached a
verdict. What you see depends on what stopped it:

- `FSDP clean-reload proof aborted before a terminal verdict: <type>: <text>`
  is an abort. Nothing answered. The text after the colon, cut at 200
  characters, names what stopped the proof; the driver log holds the whole
  exception.
- `FSDP clean-reload proof is incomplete: <reasons>` means the reload cycle ran
  but the ranks did not return the evidence a PASS needs; the reasons name the
  missing rank evidence.
- A message that opens with a class tag (`[dgxm:P`, `[dgxm:C` or `[dgxm:K`) is
  not an abort. A guard inside the proof answered your combination, and that
  answer stands: read its class and its remedy, not this entry.
- `[dgxm:K] the class K guard <id> refused inside the FSDP clean-reload proof`
  restates a waivable class K refusal. A check renders without accuracy
  waivers deliberately, so a waiver you already hold cannot be spent inside the
  proof, and granting one would return the same card forever. That refusal
  cannot be waived on this topology, for this shape. It keeps the guard's own
  remedy and entries, so a padding guard still sends you to token counts and
  topology and a sol-attn guard still sends you to an exact kernel.

**What the driver records.** Before any side effect, the check writes a durable
`RETESTING` row. Each ordinary stop closes it with an `INCONCLUSIVE` row
identifying the class and guard for a typed refusal, or the abort. The row
includes neither refusal prose nor remote tracebacks and authorizes nothing.
Both statuses deny execution; closing the transaction records completion
without changing that denial.

**What the retry does.** When a typed guard answered, the process-local
prejudgement the check published is withdrawn, so your next queue meets that
same guard and reads the same class. On an older build that
retry reads `automatic FSDP execution denied by the process-local clean-reload
Gate verdict INCONCLUSIVE` instead. The withdrawal holds for attempts queued one
after another. A second render already waiting on this combination when the
first one refused wakes to an untyped denial instead; queue it again and it
reads the guard.

When nothing answered, the denial stands: an aborted proof proved nothing, and
fail-closed means the combination stays denied until a proof completes. The
retry reads `automatic FSDP execution denied by the process-local clean-reload
Gate verdict INCONCLUSIVE`. Restart ComfyUI to clear it, or run the combination
with `auto_gate=off`.

A capacity stop taken during the proof, and the reprice the check runs between
the baseline render and the reload cycle, both happen after the check unloaded
your shards. The reprice, and any stop after the baseline render's own load,
also come after that render was paid. Re-queueing runs the proof again and pays
that cost again. Only the preflight stop (entry 88) leaves your residency
untouched.

**A check that was killed leaves its row open.** A ComfyUI interrupt or a
process kill writes no terminal row and keeps its denial, by design (entry 35):
the check settles only an ordinary exception, and a process kill leaves it no
chance to write. `dgxm gate --repair` finds those rows, and `--apply` closes
them:

```
dgxm gate --repair                 # list them; exits 1 when it finds any
dgxm gate --repair --apply         # close them; exits 0
```

The dry run prints one line per open row with its time, model, protocol and
package version, then a total, then a count of inert rows on its own line. An
inert row belongs to another gate protocol or another package build; it is
skipped, and a run that finds only inert rows exits 0, so read that line rather
than the exit code alone. `--apply` appends one terminal `INCONCLUSIVE` row per
open capability context and never rewrites, deletes or truncates a line. It
changes nothing about what is refused: an open `RETESTING` row already denies.
Run it at a driver boundary, with no render queued: it reads the open rows
first and appends after, so a check that passes in between would be
superseded by a repair row landing later and would have to prove itself again.

## 93. A multi-rank render refuses before any rank loads, and names a rank

**Symptom:** a world-2 render stops with a class C refusal that names one rank
and its numbers, or says a rank did not answer within 30 s. No worker journal
shows a load starting, nothing is quarantined, the fleet is still up, and the
next queue dispatches on the same fleet with no recycle.

**Cause:** every rank prices the load before any rank allocates. The driver
asks each rank one bounded question on a read-only endpoint that takes no GPU
lock and mutates nothing, collects the rows, and decides once. The leanest
answer governs, because the load runs everywhere: a rank that cannot hold the
model refuses the render for the fleet rather than letting its peers load first
and meet the refusal sixty seconds in. Four shapes reach the operator:

* **A rank cannot fit.** The refusal names every refusing rank with its
  numbers, worst headroom first, and the fitting ranks with the residency they selected.
  A rank that answered on slab residency is named as such. When the hosts selected
  different residencies, one cause is one host having verified the family for slab residency
  while the other has not.
* **A rank did not answer.** A timeout at 30 s, a row count short of the world,
  two rows carrying one rank id, or ranks disagreeing about the checkpoint's
  own bytes all refuse the same way. This is not a capacity verdict: a rank
  that cannot answer a memory question in thirty seconds is starved or gone,
  and it has not said it cannot load anything. Nothing is torn down and no rank
  is evicted.
* **A rescue is on offer.** One rank needs slab residency and no rank refuses,
  so the ordinary consent card is raised before any rank loads, carrying the
  leanest rescuing rank's numbers.
* **World 1 is a no op.** The one box prices its own load.

Read the numbers with two limits in mind. Each rank prices against what it
holds at that moment, so a card can quote a figure taken with the previous
model still resident, and it says so. The quote does not price shared memory
either: a rank holding a weight slab in the other slot, or a slab retained by a
load that failed, reads as having more available than it has. The rank's row
records both (`shmem_bytes`, each slot's resident and the retained-slab count)
and draws no conclusion from them.

**Fix:** when the message names what would fit, start there. For a shortfall,
run the checkpoint slab-resident on every rank, use a pruned or quantized
artifact, or free unified memory on the short box. For a silent rank, recycle
the fleet from the DGX Monarch panel, or check that box, then queue again. For
disagreeing bytes the two boxes are looking at different files: confirm the
checkpoint is the same artifact on both, then queue again.

**Estimate limits.** The agreement exposes every rank's answer before
allocation; it cannot establish that the estimates are correct. A load that
every rank prices as fitting is admitted, even when the price is too generous
and the box dies anyway. It prevents loading when a rank has already reported
insufficient capacity. The post-load checks still run: the partial-load guard,
the readiness exchange (entry 94) and the byte-verify certificate.

## 94. A multi-rank render refuses saying a peer rank could not load

**Symptom:** a world-2 render stops with a class P refusal saying a peer rank
could not load or verify the model, that this rank loaded cleanly, and that the
other box's journal names the cause. This box's journal shows its own load
finishing normally and then the line `rank readiness: this rank loaded`. The
other box's journal carries `load failed on this rank` and the real cause under
it. Nothing was sampled. The refusal arrives in seconds, not minutes, and the
fleet is still up.

**Cause:** every rank prices the load before any rank allocates (entry 93), but
memory moves between that quote and the load. A capacity check can still refuse on one box
afterwards: the allocation seam refuses, ComfyUI's own load fails, a file is
readable on one box and not the other, or the resident identity check does not
match. On every topology that puts the whole model on every rank, uly2, ring2,
dp2 and cfg2, the rank that failed and the rank that loaded cleanly cross one
readiness flag after the load and before the render's first collective. The
message on the healthy box is short on detail: the exchange carries agreement,
not identity, so this box does not know which peer failed or why. The class is
P because no consent on this box could change the answer here.

On an older build only dm-cfg2 crosses that flag; on uly2 and dp2 the healthy
rank walks into the partial-load guard, all-reduces alone, and waits for the
refused peer until the process group's timeout, ten minutes by default.

**Fix:** read the other box's worker journal. It prints the real cause with its
own refusal class, and that is the one to act on: a capacity refusal there names
what would fit. Then run the same graph again, or run it on a single box, where
there is no peer to disagree with. A rank that died rather than refused joins no
exchange, so a render that still ends in a group timeout is the dead-worker
case, not this one: see entries 9 and 87. One other case still ends that way.
The flag itself has to allocate, on the box that has just run out of memory, so
it can fail to cross; that box then prints `readiness flag not crossed` above
its own cause and this box waits out the group timeout. The cause on that box is
still the one to act on.

**Related:** entry 93 (the price every rank answers before allocation), entry 79
(the ranks disagree about weight residency, which is the check one step later,
after ComfyUI has placed the weights).

## 95. A sharded nvfp4 render refuses with a waiver card about activation scales

**Symptom:** a chroma nvfp4 checkpoint on uly2, ring2 or cfg2 refuses before the
first step with a class K message quoting a 1-step NRMS against a bf16 baseline
of 0.032 and a floor of 0.10. The message offers a waiver card and names
`DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`. The same checkpoint renders on one
GPU or on dp2 without complaint. Two texts arrive here. One says the render
already quantizes against one shared scale and still does not match a one-GPU
render. The other counts nvfp4 layers that would quantize against a scale each
rank takes from its own shard.

**Cause:** ComfyUI quantizes each activation with the module's `input_scale`,
and this checkpoint ships none. With no scale, nvfp4 alone falls back to a
statistic over the whole activation tensor, so the answer depends on which rows
the rank holds. dgx-monarch hands every rank one shared scale, which makes each
rank's rows identical to a one-GPU render's rows. That hook is on by
default and it is exact.

**It is not enough on chroma.** With the hook on, both ranks agreeing and 256
of 256 nvfp4 linears wrapped, a chroma nvfp4 uly2 render read 0.120 and cfg2
0.129 against 0.131 and 0.134 with the hook off. The rest of that gap has no
explanation yet, and it is not the kernel: bf16, fp8 and nvfp4 kernels return
identical rows whether a row is computed inside a full call or a half call. So
chroma nvfp4 refuses on every sharded topology, hook on or hook off, and only
the message changes. Krea2 read 0.011 with the hook on, under the floor, and is
not refused.

The second text means the hook declined a layer it would otherwise wrap. Two
ways in:

* `shared_act_scale = false` is set in worker_args, or `DGXM_SHARED_ACT_SCALE=0`
  on the workers, which is the off switch;
* the checkpoint carries `pre_quant_scale` on a layer with no `input_scale`, or
  it carries an expert bank. ComfyUI applies the AWQ smoothing before it takes
  the amax, so on a smoothed layer the statistic belongs to a product the hook
  never sees, and an expert bank quantizes with no scale at all, so one written
  for it would be dropped. The hook declines both.

A third gap prints neither text: a new comfy quantization layout whose
activation scale is a whole-tensor statistic, with no rule in the pack yet. The
hook neither wraps nor counts such a layer. The daily comfy canary turns red on
an unclassified layout before a render meets it.

**Fix, in order:**

1. Render on one GPU (`mode=local` with `gpus_per_host=1`) or a dp topology,
   where each rank sees the whole activation tensor and no scale is shard
   dependent.
2. Use an fp8, mxfp8 or int8 artifact of the same model. fp8 falls back to a
   constant, mxfp8 scales per 32-element block inside a row, and int8 does not
   quantize activations at all, so none of the three depends on the shard.
3. Render under the waiver, knowing the output is stamped. On chroma a waived
   render keeps the gap measured above. The grant is re-read on every render
   and its stamp rides each request, never the fleet's worker args, but the
   memo behind it stays live until you revoke it in the panel, so later renders
   of the same shape take it too (below). The headless spelling is
   `DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`, and it grants on every render
   until it is unset. The guard is family scoped, so a grant for one family
   never silences another.
4. If the message tells you to turn the shared scale back on, set
   `shared_act_scale = true` in worker_args, or remove the key, and reload the
   model. The log line `shared activation scale: N/M nvfp4 linears wrapped`
   says whether the hook ran. On chroma this changes the message, not the
   outcome, but the hook removes a real term. A bare `DGXM_SHARED_ACT_SCALE=0`
   exported on a worker host is not removed by dropping the key; unset it on
   that host and restart the mesh.

**Setting the off switch on one box only.** `shared_act_scale` is a worker arg
so that one setting reaches every worker; the bare `DGXM_SHARED_ACT_SCALE` is a
single-box aid and an absent worker arg leaves it alone. The switch decides
whether a rank issues the shared-scale collective, so set it on every worker or
on none. With the variable set on one box of a pair, a chroma render with no
waiver refuses on both boxes, each with its own class K text: that box counts
the layers it declined and its peer names the shared scale. Read the journal on
both boxes. Under a waiver, or on a family the bar does not refuse, nothing
refuses here and the two boxes issue different collectives.

**Measuring it.** Set `log_act_scale = true` (or `DGXM_LOG_ACT_SCALE=1`) and read
one step: each rank prints its own local amax, the reduced amax and the scale per
wrapped layer. With the hook on, both ranks print the same reduced amax. With the
hook off the line still prints, with `reducers=none` and the rank's own amax in
both fields, which is how the before and after legs are compared. That trace showed the two ranks disagreeing in 338 of 548 sp-reduced layers, by
up to 6.2x, which is the term the shared scale removes. On chroma both legs need
`DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale` to reach the render at all.

**A later render of the same model may not refuse at all.** A granted card is a
memo scoped to the combination, the topology label and the world, so once one
render at cfg2 has been waived, a later render of that checkpoint at cfg2 on
that world takes the same grant, renders, and is stamped. Read the ledger rather
than the absence of a refusal: a render that wrote a `use` row for
`shard_quant_scale:chroma` met the bar and was waived. A sweep run of four
chroma cfg2 cells saw this: one refusal and three carried grants
(docs/VALIDATION.md).

**Every cfg path meets the bar.** The check is read on the sample path before
the first collective, from the record the install wrote, so a render whose
conditionings run one per rank rather than as one batched call (entry 97) meets
it as a batched call does. The shared scale drops its cfg reducer on that
dispatched path, and on a padded pair whose original lengths one GPU would not
have batched: each rank then runs a whole conditioning of its own, and a max
across the cfg group would merge two amax a one-GPU render never merges. The
trace line names the reducers the call ran, so `reducers=sp` on a uly2+cfg2
render is one of those two paths, not a missing hook.

**Related:** entry 55 (the ring padding waiver, the same class and card shape),
entry 94 (the readiness exchange this bar is read inside).

## 96. A cfg2 render is no faster than the same render on one GPU

**Symptom:** a `cfg2` render completes, the image is right, and the warm
elapsed time matches a single-GPU render of the same graph. Both boxes show
full GPU load throughout. Nothing refuses and no gate complains. The sweep
harness scores such a cell CHECK with the line "cfg2 did not split".

**Cause:** the split was installed on a wrapper seam that family never reaches.
ComfyUI applies a `DIFFUSION_MODEL` wrapper nowhere central; each family's own
forward builds the executor over it, and a family that builds none (boogu,
ernie and omnigen2) leaves the wrapper installed and never called, so every
rank runs the whole cond and uncond batch. The output is correct, which is why
no identity gate and no fidelity floor can see it.

**Fix:** update dgx-monarch. It declares the seam per
family and puts those three families on `APPLY_MODEL`, the seam comfy applies
for every family. To confirm on your own graph, time the same render at `cfg2`
and as a one-GPU render (`mode=local` with `gpus_per_host=1`). In the
sweep on two DGX Sparks (docs/VALIDATION.md), for every cfg2
family whose split landed, the one-GPU reference took 1.7 to 2.4 times as long.
If you have added a family adapter of your own, run
`python tests/canary/comfy_seam_contracts.py`; the "cfg split" seam names any
family whose declaration and whose ComfyUI forward disagree.

**Related:** entry 97 (the same topology refusing instead of rendering, when
the model call reaches the split as a single row).

## 97. A cfg2 render refuses saying the model call does not split

**Symptom:** a `cfg2` render stops with a class P refusal saying cfg-parallel
got a model call of batch 1 which does not split into 2 equal slices. The card
describes the batched call ComfyUI builds and the sampler that decides it
exists.

**Cause:** ComfyUI folds the conditional and unconditional passes into one
model call only when it runs both passes. It runs both above CFG 1.0, and at
CFG 1.0 only under a sampler whose name carries `cfg_pp`, which keeps the
unconditional pass. Every other sampler drops that pass at CFG 1.0 and hands
the split a single row. That sampler case is most of what reaches this
refusal: a pair of conditionings that does not concatenate goes to the per-cond
dispatch one seam further out instead.

**Fix:** the card names it. Raise CFG above 1.0, or pick a `cfg_pp` sampler.
Where neither fits the workflow, use a ulysses or ring topology, as the card
says. Topology `auto` also avoids it at CFG 1.0, because it skips cfg-parallel
rows there.

**Unequal prompts render through the dispatch.** The per-cond
dispatch at the sampler's `CALC_COND_BATCH` seam installs for every family on a
cfg topology and decides per render by ComfyUI's own concat rule.
Each rank runs one whole conditioning, so no batched call is needed. Two prompts
of very different length on a family that pads nothing are the common case. A
family that publishes the real caption length as a `CONDConstant` takes the
same path: comfy compares a constant by value, so such a pair never folds, and
padding would add rows the constant does not count. Older builds without per-condition dispatch carried a
second card for that case, telling you to give the two prompts the same real
token count; that card is gone.

What still reaches this card besides the sampler case is a conditioning the
rule declines to read: an area, a mask, a gligen patch, a timestep window, a
hook group, or two conditionings on different control objects. There, use a
ulysses or ring topology, or write two prompts close enough in length that
ComfyUI batches them on its own.

**Related:** entry 96 (the same families before the seam fix, where the render
completed and only elapsed time showed that the topology did nothing).

## 98. A sharded render logs an attention kernel you did not select

**Symptom:** an Ideogram4 render on `auto`, `uly2` or `ring2` runs, and the
worker journal carries a line opening `USP attention kernel: TORCH_FLASH` that
goes on `TORCH_FLASH substituted for` the kernel the Init node's `attention`
widget names, `TORCH_CUDNN` or `SAGE_AUTO`. The render completes normally. On
`cfg2` the same graph logs no substitution.

**Cause:** a sharded topology routes attention through xFuser's ring layer into
yunchang's `pytorch_attn_forward`, and that function restricts torch to one
backend per kernel name. `TORCH_CUDNN` enables cuDNN alone, and cuDNN carries a
head dimension of at most 128. Ideogram4's transformer declares 256. A sage
kernel refuses the same head with `Unsupported head_dim`. So the runtime builds
the sharded attention on `TORCH_FLASH`, which carries up to 256, and says so in
the log.

It substitutes rather than refusing because nothing else runs those kernels on
this family either. ComfyUI orders its own SDPA backends by priority with all of
them enabled, so a cuDNN decline off the sharded path runs flash, and comfy's
sage wrapper catches the kernel's error and calls its pytorch attention. A
`cfg2` or single-box render of this family already computes flash whichever of
the three you select, and the sharded path matches it.

On an older build the render crashes instead: the model loads, the fleet comes
up, and the first attention call raises `RuntimeError: No available kernel. Aborting execution.` about forty seconds in, untyped.

**Fix:** no action is needed. If you want the log line to match the widget, select
`TORCH_FLASH`. If you are comparing kernels, note that this family has no cuDNN
or sage measurement to compare on any topology, sharded or not.

**When it refuses instead.** A head dimension no kernel this build ships can
carry, and a kernel that declines the live probe at load with nothing left to
substitute, both refuse class P at model load, on every rank, before the first
collective. There is no waiver and no consent: the kernel must support the
requested head width.

**Related:** entry 84 (the sol-attn kernel, which refuses its own geometry with
the same class and never substitutes, because no other kernel computes what it
computes).

## 99. A render refuses saying an acceptance fault was injected on that rank

**Symptom:** one box's worker journal carries a class P refusal reading
`acceptance fault injected on this rank`, quoted in the line `load failed on
this rank; refusing on every rank`. The other box refuses with the peer-load
message of entry 94. Nothing was sampled, the fleet is still up, and every
fresh load fails the same way until something changes. Higher up in that same
journal, from when the worker set up, sits a line reading `DEBUG LOAD FAULT
ARMED`.

**Cause:** a debug knob is set on that box's worker.
`DGXM_FAULT_LOAD_RANK=<rank>` makes the rank it names refuse inside the model
load, after every rank has priced the load and before any weight is read.
`DGXM_ACCEPTANCE=1` has to be set on the same worker or the knob is refused and
says in the journal that it stayed off. Both are read once per worker setup and
logged at WARNING then, so the arming line always sits above the refusal it
explains.

The knob tests the readiness exchange between pre-load agreement (entry 93)
and the first render collective (entry 94). External faults such as missing or
truncated files, held pages, or permission changes are normally caught by the
agreement before any checkpoint opens. Timing a permission change against the
load races a roughly one-second window and is not repeatable. This switch
injects the failure inside that window.

**Fix:** clear both variables from that worker's environment, wherever they were
set, and restart that worker. Nothing else on the box is wrong. The refusal
prices nothing, writes no gate verdict and quarantines no residency setting, and the box
renders as it did before once the variables are gone. If you did not set them,
read it as an acceptance run that leaked into a unit and check that unit.

**Related:** entry 94 (what the healthy box refuses with, and the exchange this
knob exercises), entry 93 (the agreement that catches every other injectable
fault first).

## 100. An FSDP render dies with "got mixed torch.Tensor and DTensor"

**Symptom:** the shard build finishes, both boxes report the model loaded
completely, and then every rank fails the render with

```
RuntimeError: aten.addmm.default got mixed torch.Tensor and DTensor, need to
convert all torch.Tensor to DTensor before calling distributed operators!
```

The last frames are `process_conds`, `encode_model_conds`, `extra_conds`, then
a model method with `text_embeds` or `prefix` in its name. Nothing is sampled.
The same render on one box, or under any resident topology, is fine.

**Cause:** ComfyUI runs `extra_conds` once per sampling, before the denoise
loop and outside the diffusion model's forward. Some families reach into the
diffusion model from there to refine the text embeddings, which means those
modules run before any FSDP2 pre-forward hook has gathered anything. If the
wrap sharded them their weights are DTensors while the text states are plain
tensors, and the first matrix multiply stops the render. FSDP replicates them
instead: they ride `ignored_params`, stay plain tensors on every rank, and
never enter a collective. Three families reach one, Anima, MiniMax H3 and
LTXAV, and the pack names all seven modules.

The cost is bytes, not speed. A rank holds those modules whole where a shard
would have been 1/world of them, 0.74 GiB a rank on the 61.7 GiB H3 file at
world 2 and 1.88 GiB on the LTX 2.5 bf16 transformer. The FSDP capacity line
in the worker journal states the count and the GiB as `auxiliary-module
parameters ignored`, so you can read what a rank replicated.

**Fix:** update dgx-monarch. There is no setting or workaround in the graph:
the decision is made once, inside the wrap, before the model is published. On
an older pack the render fails on every attempt of that family under FSDP, and
the way past it is a resident topology (drop `fsdp` from the preset) until the
pack is updated.

**Related:** entry 88 (the FSDP capacity refusal that stops a launch before
the shard build), entry 92 (an FSDP gate proof that refuses).

## 101. Reading the attach trace after a cluster attach times out

**Symptom:** a render refuses with `attach to workers [...] failed
(TimeoutError()); the in-process retry already ran`, and every later render in
that ComfyUI process refuses at once with `this ComfyUI session's Monarch
transport is not reusable after a partial bring-up failure`. Both worker loops
are up, `dgxm doctor` reads clean, and a fresh driver attaches on its first
try. Known triggers include a manual restart
of one loop, `dgxm restart` followed immediately by a render, and
dead-rank recovery releasing its latch. In the third case the attach
stops after its first attempt, and its refusal says that no in-process retry
ran; its trace reads differently (below).

**What the driver writes.** Every cluster attach writes one line per phase
into the driver log, all sharing a sequence number. Grep the ComfyUI log:

```
grep attach_trace comfy.log
```

Every line reads `attach_trace event=<name> seq=<n> key=value ...`. The `seq`
counts attaches for the life of the driver process, so `grep 'seq=3 '` gives
one attach. Values never contain a space, so `cut`, `awk` and a second `grep`
work on them.

Every attach ends in one `close` line carrying an `outcome` and, unless it
attached, the `stage` it stopped at. Grep `event=close` rather than reading
the tail of the sequence, because one line can land after it: when auto-heal
is on and the attach failed for good (after its retry, or after a released
fleet's one attempt), the best-effort Worker restart for the next session
writes `post_fail_restart` after the close. A sequence with no `close`
is an attach still running, or a driver process that died during one.

**The events.**

| event | says |
|---|---|
| `open` | the loop addresses and their count, `client_bind`, the `attach_config_timeout` in force, whether this process had already bound the transport, whether a liveness proof released the fleet just before (`released_predecessor`), whether auto-heal is on, and `init_wait_s`, the bound on the mesh wait |
| `heal_probe` | which loops the pre-attach probe read as down, and which ones it never observed |
| `heal_restart` | the loops read as down, and the loops the restart touched (every configured loop) |
| `heal_settled` | whether worker health settled after that restart, and how long it took |
| `loop_generation` | one line per loop before every attach attempt: the generation `gen`, the process that holds the loop's LISTEN socket and how long that process has been up, the unit restart count, and `changed=true` when that process is not the one this driver read last time |
| `phase` | one timed step. `phase=transport_enable`, `phase=auto_heal`, or `phase=attach attempt=1..2`, each with `wall_s` and an `outcome` of `ok` or the exception class |
| `attach_call` | the two halves of one monarch attach: `half=attach_to_workers` is the worker config push, `half=initialized_get` is the wait on the mesh, bounded by its `bound_s` (70 s under the shipped 60 s config-push budget) |
| `close` | the terminal line, one per attach, with the total duration. `outcome=attached` is the good end. `outcome=poisoned` names the stage that poisoned this session: `transport_enable`; `attach_1_released`, a released fleet's one attempt failing; `attach_2_failed`, any other attach failing both attempts (older builds named it `listener_wait`); or `attach_1` or `attach_2`, the first or second attempt cut short by an exception that does not subclass `Exception`, such as `KeyboardInterrupt`, after which no retry runs. `outcome=refused` means no attach ran: `stage=transport_poisoned reason=earlier_attempt` for a session an earlier attach already poisoned, and `stage=transport_bind reason=client_bind_mismatch` for a `client_bind` this process cannot switch to. `outcome=failed stage=auto_heal` is a pre-attach heal that raised, so no attach ran; `stage=unexpected` is any other escape, carrying the exception class as `evidence` |
| `post_fail_restart` | the loops the post-failure Worker restart touched and the failure class that prompted it. It is the one line that lands after the `close` |

**Read the attach timings first.** Each `phase=attach` line follows the `attach_call`
lines of its attempt, which split its duration between the config push and the
mesh wait. A `half=attach_to_workers` that took the entire duration is
a config push that never landed on a loop. A `half=initialized_get` that ran
to its `bound_s` and raised is a push that landed and a mesh that never came
up. Which one you see decides whether the next question is about the loop or
about the driver's own transport.

**Then read the generations.** Each reading is compared with the last one
this driver process took of that loop, in this attach or an earlier one.
`changed=true` before an attach that fails says the driver and the loop
disagree about which listener is live, the suspected cause of the
hand restart case. `changed=unknown` means there is nothing to compare: this
is the driver's first reading of that loop, or this reading did not come back.
`gen=none` means nothing holds the loop's LISTEN socket; that is a reading, so
a loop that goes from a generation to `none` reads `changed=true`.
`gen=unknown` with an `error=` field means the probe did not answer. That loop
is unread, not healthy and not moved: the line reads `changed=unknown`, and
`previous` keeps the last generation this driver did read, so the next reading
that comes back is still compared against something.

**What each of the three cases should look like.**

*After a hand restart of one loop.* The failing attach is the first one after
the restart, so read its first `loop_generation` line per loop. The restarted
loop should read `changed=true previous=<the generation the last attach talked
to>`. If the untouched loop also reads `changed=true`, something moved a loop
without a recorded request. The `heal_probe` line names the loops the probe found
down, and the `heal_restart` line names every configured loop, because the
restart is fleet-wide even when the probe found one loop down.

*After `dgxm restart` with a render posted straight away.* Compare `age_s` on
the `loop_generation` lines with the `wall_s` on the failing `attach_call`. A
loop a few seconds old under an attach that spends its window in
`half=attach_to_workers` says the handshake needs the loop to settle, and the
driver's attach window is absorbing a wait the start path should make.

*After the dead-rank recovery released its latch.* The `open` line carries
`released_predecessor=true`, there is only one `phase=attach attempt=1` line,
and the `close` line names `stage=attach_1_released`. Hardware tests with the
pinned Monarch runtime found that the existing driver could not attach to the
replacement fleet. Neither a second attempt nor a 70 s wait changed the
outcome, so neither runs.
The recovery is the restart path: both worker loops restarting together, then
a fresh driver process, which attached in 0.75 s in the recovery test
([docs/TROUBLESHOOTING.md #2](#2-every-attach-times-out-after-one-failed-attach)).

**The doctor row.** `dgxm doctor` prints one `listener generation` row per
loop, among that host's rows after its Worker service row:

```
  [ ok ] <host> listener generation: gen=<generation> pid=4242 age_s=612 restarts=0 started=<timestamp> marker=match
```

`gen` is a short digest of the boot id, the process holding the LISTEN socket,
and that process start time. It changes exactly when that process changes and
carries nothing about the box it came from, so it is safe to paste into an
issue. `age_s` is how long that process has been up, `restarts` is the user
unit's `NRestarts`, and `marker` compares the process with the generation
marker that a start by `dgxm update` or `dgxm setup` writes. `marker=absent` is
the normal reading after a plain `dgxm up` or a hand `systemctl --user
restart`: neither writes the marker, and the unit clears it before every start.
The row is advisory and never moves doctor's exit code.

A host doctor could not read still gets its row:

```
  [WARN] <host> listener generation: unobserved: ssh failed: OSError; the live listener generation is unknown (docs/TROUBLESHOOTING.md #101)
  [WARN] <host> listener generation: unobserved: remote check exited 255; the live listener generation is unknown (docs/TROUBLESHOOTING.md #101)
```

The first is an ssh that raised, the second a remote check that exited
nonzero; the `<host> env` row just above each carries the fault itself.
Neither moves the exit code.

**Trace limits.** These lines record driver actions and external listener
state, not the worker's protocol health. A wedged loop can retain both its
process and LISTEN socket.

**Related:** entry 1 (the attach timeout ladder and `client_bind`), entry 2 (a
wedged Worker service), entry 87 (the dead-rank recovery whose latch release
leads to case 3).

## 102. A CogVideoX checkpoint reads family `unknown`, or ComfyUI will not load it

**Symptom:** on older builds, a CogVideoX 1.5 file in the
diffusers layout logs `no signature matched this checkpoint header; the
closest is cogvideo (1 of 2 key paths present, missing
time_embedding_linear_1.)`, auto topology gets no table row and no family
preflight, and a sweep marks every cell derived from the template that names
that file `skip:unknown-family`. A CogVideoX template pointed at a file in the
ComfyUI layout runs as usual. On current builds the same file reads
`cogvideo`, and the load can still fail inside ComfyUI with `Could not detect
model type`.

**Cause:** CogVideoX ships under two key spellings. The ComfyUI layout
flattens the embeddings and names its blocks `blocks.`, so a 1.5 image-to-video
file holds `time_embedding_linear_1.`, `ofs_embedding_linear_1.` and
`blocks.0.norm1.linear.weight`. A file straight out of diffusers keeps
`time_embedding.linear_1.`, `ofs_embedding.linear_1.` and
`transformer_blocks.`. Older builds matched only the ComfyUI
spelling. Both spellings now read `cogvideo`, but for the diffusers one that
is a name only: ComfyUI detects CogVideoX from
`blocks.0.norm1.linear.weight` and carries no diffusers conversion for this
family, so no loader can build that file.

**Fix:** read the header keys of the file the template names. Dotted embedding
names mean the diffusers layout: convert it with a rename pass over the state
dict, or fetch an export already in the ComfyUI layout, then point the template
at that file. A converted file usually carries `converted_from` in its header
metadata, which is a quick way to tell two staged copies apart when both sit
under nearly the same name. A missing OFS pair is not a fault: it says the file
is text-to-video, since only the image-to-video and inpaint surfaces carry the
OFS embedding.

**Related:** entry 91 covers another case where a header identifies a family
but the release cannot render the file.

## 103. A slab-resident LoRA load refuses right before the bake, naming an exact GiB figure

**What happened.** Under `lora_low_rss`, a fresh slab load bakes the stack
through comfy's own full load before the slab reabsorbs the patched keys, and
every key comfy's patcher touches in between is a fresh allocation outside
the slab. `slab_load_fit` prices that transient from the LoRA and checkpoint
headers before either file loads. A target it cannot map onto a single base
key (a diffusers-named LoRA over a checkpoint that fuses q/k/v into one
weight, for instance) makes it charge the checkpoint's whole file size for
that stack, which errs high but is not exact (docs/VALIDATION.md).
This refusal is the exact check behind that estimate: once `ModelStore` has
built the real patch set (`active.patches`), it re-checks MemAvailable against
every patched key's own live byte count immediately before the bake runs, and
refuses typed instead of letting the kernel OOM-kill the worker.

**What the message says.** It names the exact duplicate size, the host floor
charged beside it, MemAvailable and the shortfall, under guard
`slab_lora_bake_preflight`. The base weights were loaded, but the bake did not
run and nothing was quarantined: the refusal lands before `_merge_and_free`
allocates anything.

**What helps.** Free unified memory that is not this render's own model
(another resident model, the driver's own text encoder, `dgxm reap`), use a
smaller or narrower-rank LoRA stack, or set `slab_weights=off`, which bakes the
same stack through the stock residency's load arena instead: larger, but
priced by the stock preflight.

**What it does not mean.** It is not evidence that the LoRA price is wrong in
general. The estimate errs high where it cannot map a target, and this backstop
is a second, exact check, not the primary capacity check.

**Related:** entry 5 covers the stock preflight. Unlike entry
53's rescue offers, this refusal is not waivable and has no panel action: slab
is already the residency, and no consent adds memory.

## 104. xFuser 0.7 CUDA setup imports a missing NPU helper

**Symptoms:** worker setup fails before model loading with an import error for
`yunchang.ring.utils.update_npu_out`. A plain `import xfuser` can still succeed.

**Cause:** the official xFuser 0.7.0 wheel imports an optional NPU ring
backend while it constructs CUDA attention, and released yunchang 0.6.4 lacks
the helper that backend imports.

**Action:** follow [INSTALL.md](INSTALL.md) in the existing ComfyUI environment
on every host. Its builder verifies the official wheel and produces the
temporary `0.7.0+dgxm.npuimport1` compatibility wheel, which makes the NPU
import lazy and leaves the NVIDIA attention math unchanged. A bare pip install
of this local version cannot resolve it from PyPI; pass the builder's output
with `--find-links` as INSTALL.md shows. Keep the selected CUDA Torch build,
and compare dependency versions across hosts before restarting while no render
runs. The temporary wheel stays until an official xFuser
and yunchang pair constructs CUDA attention without the local fix. It adds no
NPU support. The GPU import canary exercises the ring import that exposed this
failure; the two-host constructor and fidelity controls are recorded in
[VALIDATION.md](VALIDATION.md#dependency-and-setup-checks).

## 105. Nodes 2.0 shows nodes and links but no input widgets

**Symptoms:** the Fleet, Identity Gate or dual-Spark split template opens, but
input controls and Note text disappear while nodes and links remain. Opening
a new browser page can appear to help temporarily.

**Cause:** older copies of these templates hold a colon in their top-level
workflow `id`. ComfyUI frontend 1.53.6 keys widgets with colon-separated
fields, so that extra separator stops Nodes 2.0 registering stock and custom
widgets. The failure reproduces with every custom node disabled and one stock
node, without running a model (docs/VALIDATION.md).

**Action:** update dgx-monarch, restart ComfyUI as described in INSTALL, and
reopen the updated template from Browse Templates; the generator drops the
colon. To keep edits in an older saved workflow, remove colons only from its
top-level `id` string and reopen that file. Leave numeric node IDs, links and
widget values unchanged. The repair touches UI metadata only and changes no
sampling or model-residency behavior.

## 106. Sampling finishes but VAE decode runs out of memory

A finished distributed sampler can leave its DiT weights, LoRA backups and
allocator memory resident in the actors while ComfyUI loads the driver's VAE.
On unified memory those allocations compete with the decoder for the same RAM.

For the measured BF16 accelerated Wan Animate 2 recipe, use the BF16 warm
template: it passes the trimmed latent through Clear VRAM at `soft` before
native VAE decode. That frees unused allocator buffers without unloading the
DiT or LoRA backups, so the next prompt or seed reuses them. Its acceptance
runs used 8 GiB driver and Init reserves. A larger reserve can still refuse a
warm render; if you need one, keep it and choose a smaller quantization.

For more capacity at the cost of reloading weights, wire the final sampler or
latent trim into Clear VRAM's optional `samples` input, select `recycle`, set
`include_driver=false`, and wire its `samples` output to VAE Decode. The Wan
Animate 2 memory-saving template uses this order; the standard INT8 template
keeps its model loaded. The node returns the latent unchanged only after the
client-owned actors have retired; persistent Worker services and driver models
stay available. The next distributed render pays a fresh actor setup and model
load.

If the node reports that the recycle did not complete, it blocks decode. Let
active render and result leases finish, or inspect the reported lifecycle
failure before retrying; a busy or unknown outcome does not guarantee that any
memory was released. A Clear VRAM output node with nothing connected sets no
order between sampling and decode. `hard` alone does not guarantee that an
actor's retained allocator memory returns to the OS.

## 107. The same cluster fault repeats on a timer

**Symptoms:** the driver log shows one `cluster fault` ERROR whose body starts
`undeliverable message to cast.`, then lines such as `cluster fault repeated 8
times`. Nobody is rendering. Older builds printed the whole fault
and a browser toast every 30 seconds for as long as ComfyUI stayed up.

**Cause:** the driver renews a client lease against its worker fleet every 30
seconds ([#51](#51-a-worker-actor-outlives-its-client-and-holds-tens-of-gib)). The renewal is a broadcast, so a fleet that cannot receive it
answers later, as an unhandled fault, and the renewer never sees a failure it
could back off from. Two states do it. In one, the fleet stopped receiving
(it died or was restarted, for example) while the driver sat idle, and the
driver still holds its handle as live. In the other, the fleet latched after a
worker death while a failed render's abandoned lease remains; its renewals
fault once the worker services no longer know the fleet, for example after
they restart.

**Action:** the first report indicates an unreachable fleet.
Queue again or reset the Attached mesh: the driver then finds out what happened
to the fleet and either replaces it or prints what blocks it. The repeat lines
count one fault at 2, 4, 8 and every later power of two; they are not new
faults, and a different fault is always printed whole. For a fleet latched
after a worker death the driver stops renewing by itself and says so once:
`client lease renewals stopped for a fleet latched after a worker death`. An
actor process that survived such a latch then exits at the lease grace, which
lets the next attach prove the fleet gone; with actor reaping disabled it
stays until its Worker service restarts. The measured record is in
[VALIDATION.md](VALIDATION.md#lifecycle).

## 108. cuDNN Ring refuses its native normalization check

**Symptoms:** a Ring render with `TORCH_CUDNN` selected refuses before
sampling with a native LSE, output-contract or processor-binding error. A peer
may report that another rank could not load or verify the model (entry 94).

**Cause:** combining Ring key/value blocks needs each block's real
log-sum-exp (LSE) normalization; a normalized attention output with
placeholder zero LSE values cannot reproduce full attention. The cuDNN Ring
processor calls the private operator `aten._scaled_dot_product_cudnn_attention`
with `compute_log_sumexp` set, and checks a small native output and LSE example
before the rank-readiness exchange. The check fails when the installed Torch
lacks that operator or returns another result layout, when the installed
xFuser or yunchang does not keep the custom processor, or when the native
result is wrong on one worker. It runs after the final model and head binding,
including a warm kernel switch. A request the head-dimension rule already
moved to Flash does not run it. On a supported Ring topology a failing check
enters the same readiness exchange as a load failure, so a healthy peer
refuses before its first Ring communication.

**Action:** keep the failing worker's exact error and compare the installed
Torch, CUDA and dependency versions on both hosts. Make any update through the
normal verified workflow while no render runs. Do not replace LSE with zeros,
loosen the comparison tolerance, or treat a passing CPU stub test as native
cuDNN validation. Switching to another supported attention kernel is a
separate configuration and needs its own reference comparison. Ring with FSDP
still refuses.


## 109. pip check reports cuSPARSELt is not supported on this platform

**Symptom:** on Linux ARM64, pip 24.2 and later report:

```text
nvidia-cusparselt-cu13 0.8.1 is not supported on this platform
```

**Cause:** the official wheel's filename says `manylinux2014_aarch64`, but
its internal `WHEEL` tag is `py3-none-manylinux2014_sbsa`. The internal tag is
invalid. Installation selects the wheel by its filename; pip 24.2 and later
check the recorded tag. Pip 24.0 does not perform that check.

The defect affects the cu13 ARM64 wheels for 0.8.0 and 0.8.1. Versions 0.9.0,
0.9.1 and 0.10.0 have the correct tag, and the x86_64 wheels are unaffected.
PyPI files cannot be replaced, so 0.8.1 cannot be corrected in place. PyTorch
still pins 0.8.1 and links against its library. Installing 0.9.x would create
a dependency conflict. The tag error does not affect imports or renders.

**Action:** run the repository's dependency checker using the ComfyUI Python
and repository path selected in [INSTALL](INSTALL.md#install-from-source):

```bash
"$COMFY_PYTHON" "$REPO_DIR/tools/check_dependencies.py"
```

The checker runs `python -m pip check` with that same interpreter. A successful
check with no errors passes silently. It accepts the line above only when it
is the sole error, the host is Linux aarch64, and the installed
`nvidia-cusparselt-cu13` version is 0.8.1 with the invalid internal tag.
It downloads the [official wheel](https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl),
checks SHA-256
`4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`,
and compares every installed wheel member except `RECORD`, which pip rewrites.
This requires network access. No package or metadata is changed.

After verification, the tool prints one line identifying a known upstream
platform-tag failure, its verified wheel hash and [issue #5](https://github.com/Deen-Media/dgx-monarch/issues/5).
Save that line with the installation record. A download failure, mismatched
file or hash, different platform, or any additional dependency error fails
the check and preserves the full pip output. Resolve the failure before
continuing; do not downgrade pip or change the working Torch/NVIDIA packages
to bypass it. Continue with the required imports, Doctor and saved distributed
render after the dependency check passes.

The remaining upstream blocker is PyTorch's dependency pin. Watch the
cuSPARSELt entry in PyTorch's
[`generate_binary_build_matrix.py`](https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py).
Remove the allowlist entry only after the pinned Torch build depends on
`nvidia-cusparselt-cu13>=0.9.0` and a fresh ARM64 environment passes dependency
checks, imports and the ARM64 CI job. A corrected NVIDIA wheel alone does not
meet that condition.
