# dgx-monarch: Design and Architecture

**Status:** current architecture contract for the two-Spark release. The
message-return boundary and reattach checks passed on October 8;
native latent RDMA remains **NOT RUN / HOLD**. Verified update now
supports a Worker
on the driver's host; [Validation](VALIDATION.md#operator-checks) records the
update results and limits. The benchmark
campaign covers a documented subset, not the full matrix. The package declares
version 1.0.0; the release tag waits for launch. Section 9 tracks acceptance.
The runtime also includes low-RSS LoRA residency, zero-copy slab residency,
the identity gate and ledger, `dgxm top`, Fleet sampling, an RDMA latent-return
path that is off by default and **NOT RUN / HOLD** (§5.6), and the per-family
adapter registry (§5.3). Hardware-validation scope is recorded per family and
path in [MODELS.md](MODELS.md) and [VALIDATION.md](VALIDATION.md).

This document defines the architecture and safety requirements. See
[CONCEPTS.md](CONCEPTS.md) for terminology, [TRUST.md](TRUST.md) for the
identity-gate trust model, and CHANGELOG.md for release history.

dgx-monarch distributes ComfyUI workloads across GPU nodes using Monarch
actors, typed failures, and benchmark-seeded topology selection.

[Validation](VALIDATION.md) records tested configurations and their limits.
The canaries under `tests/canary/` catch ComfyUI and torchmonarch API changes.

---

## 1. Vision and positioning

dgx-monarch is a **ComfyUI custom-node pack and cluster runtime** that runs
diffusion inference across multiple GPUs/nodes using:

- **PyTorch Monarch** (BSD-3-Clause) for orchestration: persistent per-GPU actors,
  supervision trees, typed remote errors, one persistent
  Worker service per host;
- **xDiT/xfuser** (Apache-2.0) for the sequence/CFG-parallel algorithms,
  through a temporary local compatibility wheel (§3.1);
- **NCCL / torch.distributed** for tensor communication.

The runtime provides:

| Dimension | dgx-monarch behavior |
|---|---|
| Standing infrastructure | one `worker_loop` process per host; ComfyUI is the client |
| Failure surfacing | typed `SupervisionError`, fail-fast handling for known-dead actors, and `MeshFailure` telemetry |
| Recovery granularity | per-actor respawn on the live mesh |
| Big-tensor transport | actor messaging plus size-gated, default-off one-sided RDMA (§5.6) |
| Cluster bring-up | `dgxm` CLI, config file, and preflight doctor |
| Topology selection | benchmark-seeded auto mode (§5.4) |
| Repository license | Apache-2.0; dependency boundaries are documented in LICENSE-NOTES.md |

The control plane improves operation, failure handling, and user experience;
render speed depends on the unchanged data plane.

---

## 2. Scope

### 2.1 Model architecture targets

These models set distinct architecture and capacity requirements.

| Model | Family / attention shape |
|---|---|
| **Krea2** (RAW + Turbo) | wan-family DiT |
| **Chroma** | flux-family DiT |
| **Ideogram4** | PixelDiT (head_dim 256), dual-model asymmetric CFG |
| **LTX** (2.3 and 2.5) | LTX video DiT, packed video+audio streams |
| **Wan 2.2** (t2v/i2v) | wan video DiT (14B MoE: high-noise and low-noise checkpoints) |
| **MiniMax H3** | packed text+audio+video single stream, NestedTensor AV latent |

[MODELS.md](MODELS.md) holds the complete support and evidence matrix and is
the reference for each family's recommended topology, quantization support, and
hardware-validation status.

Quantization follows the **stock ComfyUI loader** surface: bf16, scaled fp8,
and the comfy_kitchen formats (int8-convrot, mxfp8, nvfp4). GGUF remains
deferred. The project's exact-math policy excludes caching and staleness
accelerators such as TeaCache, EasyCache, and FBCache.

### 2.2 Hardware targets

- **Tier 1 (validated):** 2x NVIDIA DGX Spark (GB10, aarch64, UMA) over ConnectX 200G RoCE.
- **Tier 2 (designed-for, needs community validation):** any 2-8 node Linux
  cluster with NCCL-capable fabric; single-node multi-GPU boxes.
- The code must never hardcode Spark specifics; they live in a fabric profile
  (§6.1, [CLUSTER.md](CLUSTER.md#fabric-profiles)).

---

## 3. Licensing and dependency boundaries

The repository source is Apache-2.0 licensed, copyright Deen Media, LLC.
Dependencies and runtime hosts remain separately installed.
[LICENSE-NOTES.md](../LICENSE-NOTES.md) is the canonical dependency and
distribution-boundary inventory.

### 3.1 The rules

1. Repository implementation is authored for this project from its
   specifications, current public APIs, and documented runtime behavior.
   Contributions must identify any external source that informed an
   implementation and must not copy third-party implementation code.
2. Measurements and architecture requirements belong in dated project records.
   Implementation should depend on the recorded contract, not an undocumented
   external implementation detail.
3. Apache-2.0 dependencies remain external dependencies. The temporary xFuser
   compatibility wheel `0.7.0+dgxm.npuimport1` is rebuilt by INSTALL.md from
   the SHA-verified official 0.7.0 wheel, is not vendored, and changes only a
   lazy optional-NPU import and its version metadata, so NVIDIA attention math
   and APIs are unchanged. Remove it when official xFuser and yunchang releases
   build CUDA attention without it. See VALIDATION.md, xFuser 0.7 compatibility
   build.
4. LICENSE-NOTES.md records dependency licenses and distribution boundaries.
   Contributors update it when those boundaries change.
5. The registry identity is `dgx-monarch`. Node class names and categories use
   `DGXMonarch*` and "DGX Monarch".
6. `LICENSE` is Apache-2.0. External contributions use DCO sign-off; there is
   no CLA.

### 3.2 Engineering implications

The USP attention bridge from Comfy model forwards to xfuser attention is a
maintained integration surface (§5.3). Loaders and samplers stay stock where
possible. The benchmark harness is authored in this repository and preserves
the documented matrix, quality-gate, and N/A contracts.

---

## 4. Capability decisions

| Capability | Decision | Rationale |
|---|---|---|
| Monarch mesh and actor lifecycle | **retain** | per-actor respawn and typed supervision are validated (§5.1) |
| GPU worker surface (load, sample, options) | **retain** | typed endpoint contract in §5.2 |
| Request-key reuse and LoRA hot-swap | **retain** | avoids unnecessary checkpoint reloads (§5.5) |
| Keyed idempotent patches | **retain** | repeated graph execution must not stack wrappers |
| USP/ring attention injection | **maintain per family** | adapter registry owns family-specific forward contracts (§5.3) |
| CFG-parallel and mask-safe padding | **retain** | required by validated model families |
| Dual-model asymmetric CFG | **retain** | required by Ideogram4 |
| FSDP capacity mode | **retain** | enables checkpoints that do not fit resident (§5.4) |
| Leader-only result transfer | **retain** | optional RDMA return is specified in §5.6 |
| `torch.compile` and cuDNN-SDPA controls | **retain** | the Init `compile_dit` widget block-compiles a DiT that has a `blocks` list; see ADAPTERS.md and [TROUBLESHOOTING.md #14](TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on) |
| Sage attention selection | **retain** | auto-gated to the validated quantization and topology combinations |
| GGUF loader path | **defer** | outside the launch model scope |
| PipeFusion config + code | **drop** | unsupported on the validated fabric |
| easycache / caching accel nodes | **drop, policy** | exact-math only |
| LTX expansion nodes (STG, Adain, guiders) | **evaluate per node** | add only behavior missing from current ComfyUI |
| Mirrored Comfy node trees | **do not implement** | §5.7 wraps stock objects instead |
| Benchmark harness | **retain** | matrix and quality gates are project-owned scripts |
| NCCL, UMA, and fabric measurements | **retain as dated evidence** | canonical in VALIDATION.md and fabric profiles |
| Cluster lifecycle | **use the `dgxm` CLI** | centralized preflight and idempotent service control (§6) |

---

## 5. Architecture specification

### 5.1 Process topology

```
ComfyUI (the driver; user's normal comfy process, any frontend)
  └─ dgx-monarch nodes hold a HostMesh handle (attach_to_workers)
       ├─ host A: Worker service (systemd) ──spawns── GPUWorker actor (per GPU)
       └─ host B: Worker service (systemd) ──spawns── GPUWorker actor (per GPU)
                    actors form NCCL/xfuser groups among themselves (data plane)
```

- Worker services bind explicit `tcp://<fabric-ip>:<port>` addresses from
  config, never hostname-derived ones. Preflight rejects loopback advertisement.
- The ComfyUI process is the single Monarch client: when it dies its procs die,
  so no orphaned process holds a model. A supervision-class failure on a
  setup, load or render dispatch evicts the fleet, so the next Init or render
  respawns it (`MeshHandle._await_or_evict`), and a mutation RPC that times out
  or is interrupted after its send latches the mesh DIRTY until a recycle
  (`mesh_rpc.call_all_bound`). `dgxm up` restarts wedged Worker services
  idempotently, and the driver restarts dead services before attaching
  (`auto_heal`).
- torchmonarch's in-process reaping (keepalive orphan timeout, parent death
  signal) stays the fast path. Under it, the driver renews a client lease
  against every live fleet, so an actor whose client is gone exits itself
  after a bounded grace (300 s by default), and an out-of-process record of
  loop-spawned actor processes lets `dgxm up`, `down` and `restart` sweep
  survivors by exact identity while `dgxm doctor` names any that remain. The
  grace sits above the 240 s teardown token authority, so an in-flight
  deliberate teardown always resolves first and no absorption or idempotency
  rule changes. After a proc stop times out, the driver uses the same record to check
  liveness. Only confirmation from every configured host that its processes
  are gone releases the replacement latch. Supervision text, an unanswered
  probe, and every other result leave the latch set. See [TROUBLESHOOTING.md #51](TROUBLESHOOTING.md#51-a-worker-actor-outlives-its-client-and-holds-tens-of-gib) and [#87](TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).
- `unhandled_fault_hook` is installed at node-pack import: faults surface as
  ComfyUI toasts and log lines with the actor/host named, never as silent hangs.
### 5.2 GPUWorker actor (endpoint surface)

One typed actor runs per GPU; Monarch preserves endpoint exceptions. The core
endpoints:

- `setup(env)`: one env dict carrying generation, rank, world, master, port,
  topology and the fabric environment. NCCL init and xfuser model-parallel
  bring-up. Idempotent only for the exact generation and topology key;
  re-called after an authorized replacement.
- `load_model(unet_name, options, lora_stack)` / `load_uncond_model(...)` /
  `unload()`: §5.5 state machine. A LoRA stack is part of the load request, not
  a separate endpoint.
- `sample(request, progress_port)`: the denoise loop; streams progress (§5.8);
  leader returns latents, peers return None.
- `status()`: VRAM/UMA snapshot, resident model key, topology, NCCL initialized.
- `provenance_baseline(setup_generation, artifact_manifest=None)`: Gate's
  setup-token-bound rank/world/package-source queue barrier for the exact READY
  generation. With a manifest it also returns strict saved pre-READY setup and
  fresh post-operation runtime snapshots, Comfy-resolved artifact metadata
  identities, and an event watermark, serialized behind prior GPU mutations.
- `clear_vram(level)`: soft/hard cleanup with typed failure reporting.

Same-handle topology replacement is a generation transaction. Driver and worker
retirement publish DIRTY before clearing the prior READY setup identity or
rank/topology metadata, then require an explicit `UNSETUP` acknowledgement from
every rank. Recycle and shutdown stop the worker processes next, so their group
teardown skips the model unload: the process exit frees the weights, where
ComfyUI's unload would first copy them to host RAM. A worker that kept its
models stays DIRTY and refuses a new setup. Setup publication starts under a
DIRTY latch, writes every READY-qualified field, and clears DIRTY only as its
final authorization edge, so any unknown, failed, or interrupted outcome stays
typed DIRTY. Every setup-bound node RPC is authorized by a token over the exact
generation, process-group key, and worker-policy key; authorization and enqueue
share the handle lock, so a stale request cannot cross into a replacement
generation. The generic RPC surface cannot call setup, teardown,
policy-application, or raw sample lifecycle endpoints. A submitted sample holds
a driver-side generation lease until every leader latent is materialized, every
local RDMA registration drop is confirmed, and every generation/token owner ACK
is settled; replacement, recycle, and shutdown refuse a live transaction.
Torchmonarch's pinned future records a collection timeout as terminal: the
driver requests cancellation and abandons the unread result rather than
offering a retry. Abandon retires the active lease once any already-started
raw-read scratch thread exits, makes later reads terminally invalid, and leaves
abandoned work that requires an Attached mesh reset before changing topology.
Actor teardown and failed setup retain unACKed backing in the generation/token
registry; a later pack never evicts it, and it leaves only after exact ACK or
process reset. A successful replacement can publish a new return only after
the old generation has no live or unresolved descriptor authority. Destructive
lifecycle calls use bounded lock acquisition, so a timed-out caller cannot stay
queued and stop a later READY generation.

A handle-scoped render session is the capability for setup, residency/policy,
ordinary liveness, sample, and setup-bound RPC mutation. Direct submission
claims that capability before quarantine, topology resolution, or setup;
Pipeline, Fleet, and nested calls activate the same owner. Long mutation RPCs
hold it through terminal completion rather than releasing it after enqueue,
and only explicit forced supervision reconciliation bypasses the ordinary
liveness check. The Identity Gate holds one session across every reference,
swap, restore, and verdict leg, making the whole ceremony one atomic residency
transaction.

Risky single-model normal-render residency is separately authorized for every
immutable dispatch snapshot. A Gate PASS is only the source of authority: the
request grant binds the primary model and detailed checkpoint/LoRA artifact
set, the complete effective worker policy, physical config identity, resolved
topology and attention, and the selected READY setup generation/key. A Gate
ceremony proves exactly one model, so dual-model requests are forced onto
complete stock residency while driver and worker artifact preflight still bind
both model slots. Driver preflight, worker preflight, and post-adoption checks
all re-derive the applicable snapshot. Each normal request is explicitly
stamped `required`, `stock`, `operator_off`, or `gate_internal`, so an absent
or stale grant cannot inherit risky defaults.
Before an explicit or automatic ceremony can retest old authority, one fsynced
multi-context `RETESTING` ledger row revokes the normal, Fleet, and eligible
explicit-slab siblings. Final verdict rows supersede it only after the proof
finishes, so a crash or failed final write leaves authority denied instead of restoring
an old PASS. See [TRUST.md](TRUST.md) for the complete current protocol.

The ledger also holds `WAIVER` rows for granted,
used, or revoked consent and `CAPACITY_CERTIFIED` rows for a slab load whose
stock alternative provably could not fit. Both are audit-only rows in a
capability namespace no trust context can emit, with verdicts the
authorization lookup does not accept, so audit evidence cannot become gate
authority. A v8 PASS must re-prove once per combination because it predates
the pre-gate certificate requirement. The current version is
`GATE_PROTOCOL_VERSION` in `gate_ledger.py`, 13; every version change
requires a new proof for each combination on first use, and a burned number is never
reused.

`WAIVER` is written when the consent endpoint accepts a request and when the
driver applies either headless consent setting: the standing auto-rescue toggle or
the per-kind environment variable. `CAPACITY_CERTIFIED` is written for an
eager consented load under an explicit topology because `load_model` returns
the model-store snapshot that carries the certificate. Under `auto`, loading
occurs in `sample`, whose result has no snapshot; the worker still certifies
the load and refuses a mismatch, but no driver row is written. The identity
gate writes its ceremony capacity row separately.

**Sampler and loader failures trigger cleanup.** Every sampler and loader
body runs under `cleanup_on_failure` (`actor/failure.py`), which cleans up the
active patcher, empties the allocator cache, and runs gc on failure.
Every patch on a Comfy ModelPatcher is keyed, and a keyed wrapper is added only
when its key is absent, so repeated execution cannot stack patches.

**Concurrency.** Every endpoint is `async def`, because torchmonarch refuses
to spawn an actor that mixes sync and async endpoints. The blocking render
runs on one GPU thread (`_on_gpu`, with contextvars copied so the RDMA return
binds to this actor's instance) under an `asyncio.Lock`, and the lock keeps
compute serial: no two renders, or a model mutation and a render, overlap on
one GPU, model, or topology. Under torchmonarch >= 0.6.0 queue dispatch, every
endpoint that takes the lock must be `@concurrent_endpoint` (the comment at
the top of `actor/worker.py` gives the rule). This lets status and supervision
respond while rendering holds the GPU lock; model mutations remain serialized.

### 5.3 Attention adapters

A small registry is keyed by Comfy model class:

```
adapters/
  base.py              # protocol: matches(base_model) -> bool;
                       #   inject_usp(diffusion_model, ctx);
                       #   pad rules and the drop_rows exclusion primitive
  detect.py            # ordered header signatures -> family string
  __init__.py          # the ordered ADAPTERS registry; most specific first
  <family>.py          # one file per family, plus shared helpers
                       #   (flux_shared, usp_pad_exclusion, wan_ring_attention,
                       #    cfg_parallel, fsdp, sol_attention)
  cudnn_ring_attention.py # native cuDNN output/LSE contract and instance binding
```

`adapters/__init__.py` is the registry of record. Its order matters: several
families subclass others and the first match wins, so the most specific
adapter comes first. [ADAPTERS.md](ADAPTERS.md) is the authoring guide;
[MODELS.md](MODELS.md) carries the per-family evidence.

Resolved cuDNN Ring attention uses an owned per-instance processor through
xFuser's custom-processor interface and keeps xFuser's ordinary Ring
transport. The private ATen cuDNN operator supplies the FP32 log-sum-exp values
needed to merge key/value blocks; normalized output and placeholder zeros
cannot provide those weights. The dispatcher prepares this binding at the end
of the protected load phase, after final family/head resolution and before rank
readiness, so a failed native capability probe joins the existing not-ready
exchange. The binding adds no collective, and `Topology.validate` already
refuses Ring with FSDP. Warm kernel changes and setup-generation changes
invalidate the prepared implementation. CPU interface checks are not native
kernel or model/reference fidelity evidence.

Each adapter, working only from the comfy model source (the GPL runtime host,
normal node-pack practice) and xfuser public APIs:

1. locates the attention modules for its family,
2. wraps forward to route q/k/v through xfuser's Ulysses/ring primitives
   (`xfuser.core.long_ctx_attention` and its siblings, imported and never
   vendored),
3. declares sequence-padding rules, including divisibility by SP degree and
   attention-mask preservation,
4. is bound by the worker onto each freshly loaded base, once per load
   (`GPUWorker._inject_for_topology`); `Adapter.bind` replaces a method rather
   than wrapping it, so repeated graph executions never stack.

The default for non-divisible sharded axes is aligned zero-pad plus gather-time
trim. The Krea2 padding measurement found that an attended pad key at
RoPE (0,0,0) produces left-edge artifacts. `adapters/usp_pad_exclusion.py`
removes those rows: a family that declares its padded row indices has them
dropped inside the ulysses path exactly (same all-to-all, same inner kernel,
zero rows re-inserted), and a padded sequence on ring or hybrid is a typed
class-K refusal (waivable per render) rather than known-wrong math. Families
that have not been wired and probed keep the measured-tradeoff contract;
[MODELS.md](MODELS.md) records which is which per family. Adapters may
require exact divisibility only as an explicit, independently validated
exception.

Adapter acceptance requires identical cross-rank latent statistics and
fidelity against a single-GPU reference within the measured noise floor.

### 5.4 Topology + auto mode

`Topology` dataclass: `ulysses, ring, cfg, dp, fsdp, world` with validation
(divisibility, per-model constraints declared by adapters).

**Auto mode** uses a static decision table seeded from the dated two-Spark
measurements in [VALIDATION.md](VALIDATION.md) and keyed on
(model family, quant, megapixels, world size):

- families with explicit resolution-crossover rows use cfg-parallel at the low
  end and USP at the high end, with per-model thresholds stored as data; other
  families have explicit full-range ring or Ulysses rows. Mage Flow and Qwen
  Image 2.1 have no row, and Flux, Flux 2, Krea 2, Lens, Qwen-Image and Z-Image
  have rows only from 1.2 MP up; those cases take the generic uly2 fallback
  (row 99). Data parallelism is never a table row: `with_derived_dp` derives it
  from the ranks a row leaves spare, and a batch that cannot split across them
  folds them into sequence parallelism, or on a cfg row refuses (class K);
- a row may take cfg-parallel only where its exact family, quantization, and
  megapixel band have a dated run that measured it faster than the one-GPU elapsed time
  and no worse than the step-nrms floor.
  `topology_cfg_evidence.CFG2_EVIDENCE` is keyed by row number and records the
  admitted quantizations. Missing, out-of-band, or inconclusive evidence does
  not preserve a legacy cfg choice: auto selects its Ulysses fallback and logs
  why. Tests hold every predicate, including the fallback, so a new cfg row
  cannot inherit another family's result;
- ideogram4: auto takes Ulysses2, never cfg2; explicit Ring2 stays for
  diagnosis, and explicit cfg2 runs the dual-model split (MODELS.md);
- sage auto-on only for fp8 + USP at high res;
- FSDP only when the model does not fit resident (capacity mode), never for speed;
- every auto decision logs its row and reason, for example
  `auto: uly2; krea2 fp8 at 1.0 MP, world 2, table row 99 (...)`. Krea 2 lost
  its cfg2 row below 1.2 MP (crossover table row 10).

Users can always override; auto is the default widget value.

**FSDP capacity mode (tested at world=2).** `uly2+fsdp` renders
byte-identical to the resident (unsharded) result with each rank holding about
1/world of the weights, at about twice the elapsed time: the all-gather cost
keeps FSDP a
capacity tool, not a speed one. It runs on the 2-Spark pair with no 4-GPU or
2D-mesh requirement. `ring2+fsdp` raises a typed refusal: a model reload changes ring attention output
(resident ring2 render-unload-render NRMS 0.004436 while the identical uly2
sequence is exact), so the clean-reload gate can never certify the
combination and every first-use ceremony on it fails. At world=2,
FSDP's all-gather (world group) and USP's Ulysses/ring all-to-all
(sequence-parallel group) are two NCCL communicators over the same two ranks,
and they co-schedule into a deadlock. This differs from the NCCL_PROTO=LL rule
(Appendix A), which still holds. The fix is `NCCL_LAUNCH_ORDER_IMPLICIT=1`
(NCCL >= 2.26, relying on the single max_workers=1 GPU thread's deterministic
submission order), which every worker setup sets whatever the topology,
because NCCL reads it once per process at the first launch. At the
default `fsdp_prefetch_depth` of 1 the FSDP adapter also pins FSDP2's explicit
forward-prefetch list empty (`set_modules_to_forward_prefetch([])`). That is a
defensive pin on a torch default, not a second half of the fix: the list
already starts empty, and the default forward path overlaps each group's
all-gather one ahead on its own stream either way. A `worker_args` depth of 2 to 8 names that many blocks ahead through the
explicit list instead, trading reserved memory for speed; a changed depth is a
new capability context and requires a new Gate proof. `cfg2+fsdp` and `dp2+fsdp` presets join
`uly2+fsdp`: the launch-order guard already counts cfg as model parallelism,
and a dual-model family (Ideogram4) typed-refuses cfg-parallel under fsdp
because slotting one checkpoint per rank would have the world-group
all-gather exchange halves of two models.

Constraints, enforced as typed errors: a bf16, fp16, fp8 or int8 checkpoint
with a matching denoising core. Quantized files shard as stored bytes
(`adapters/fsdp_quant.py`): a `QuantizedTensor`
subclass adds the structural ops FSDP2 runs on a parameter, chunks any per-row
scale with its rows, and rebuilds the full wrapper after each all-gather,
which keeps comfy's registered kernels on the compute path. Block-scaled
layouts (mxfp8, nvfp4) and transposed storage refuse; a live cast to a
quantized kind refuses because it has no file-backed bytes. A LoRA stack on a
sharded model needs `lora_low_rss` on (`actor/fsdp_lora.py`): every rank reads the pristine full weight from the checkpoint file, runs
comfy's own merge and key-seeded rounding on it, and copies its dim-0 chunk
into the local shard, with no collective and no comfy backup, so the bake is
the resident bake exactly; a key leaving the stack is restored with one
contiguous pread of the rank's rows. With `lora_low_rss` off a sharded model
refuses the stack, typed. The checkpoint itself must also be admitted for the
bake, before any load, on every `lora_low_rss` or `auto_gate` setting
(`adapters/fsdp_lora_admission.py`). Admitted: bf16, fp16, and a
plain fp8-dtype file with no comfy-kitchen wrapper (a literal `float8_e4m3fn`
weight, as in Qwen-Image). Refused regardless of the lever setting: a
comfy-kitchen quantized checkpoint, whose live parameter is a QuantizedTensor
wrapper that `actor/fsdp_lora.py` refuses outright; the header markers that
name one are listed in `adapters/detect.py`. The lever's own card fires only
when the effective `lora_low_rss` is off. The launch-time bake admission check
does not cover fp32 islands: no header or coarse live signal distinguishes an
island that Comfy's load keeps live at fp32 (safe to bake) from one it casts
to the core's compute dtype (a live/file mismatch), so `actor/fsdp_lora.py`'s
in-bake `_file_tensor` check, per LoRA-patched key, is the backstop for that
shape. That check and the quantized-key check in `capture_record`/`bake_key`
(a backstop for a checkpoint the header markers miss) raise the same class P
tag as the launch-time refusal, not a bare `UnbakeError`: both are
deterministic per checkpoint, so every rank raises the identical refusal,
which is what makes tagging them rank symmetric and safe. Header and
loaded-model detection both classify any full-precision mix containing F16 as
`fp16`, while fp8/int8 storage markers take precedence; an fp16 core is
admitted uniform only. fp32 parameters inside a bf16 core are islands
(`adapters/fsdp_islands.py`): when they hold at most 5%
of the checkpoint bytes the header proof still reads `bf16`, the live scan
reports `live_dtype_profile=bf16_core_fp32_islands_v1` with the island count
and bytes, and the wrap passes every island to FSDP2 as `ignored_params`, so
they stay replicated plain tensors and the forward math is untouched. The same
`ignored_params` set carries every module a family runs outside the denoise
path, which is what keeps them out of every collective. The rule: ComfyUI runs
`BaseModel.extra_conds` once per sampling, ahead of the denoise loop, so a
diffusion-model module reached from there runs before any FSDP2 pre-forward
hook has gathered a weight. If sharded, its DTensor weight meets a plain text
tensor and every rank fails on the first `addmm` (the
first H3 FSDP render to reach a forward). Three supported families reach one:
Anima's `llm_adapter`, H3's `condition_proj` and `token_refiner`, and LTXAV's
two caption projections and two embedding connectors.
`adapters/fsdp_islands.py` declares them per family, under the comfy class
whose `extra_conds` makes the call, and derives the one flat tuple the wrap
reads by name off the live model. A name is not unique to the family that
declares it, so the wrap reads the family off the comfy model it already holds
(`type(patcher.model)`) and subtracts a name another family owns on its own
forward path: LTXV builds `caption_projection` in the LTX base it shares with
LTXAV and calls it inside its forward, where a sharded weight is gathered and
a replicated one would only cost a rank the whole module. A rank then holds
the replicated bytes whole where a shard would have been 1/world of them:
1.487 GiB on the 61.7 GiB H3 file, 3.755 GiB on the LTX 2.5 bf16 transformer.
The shard-build price reads a file size and a world, not a module census, so
`capacity_fit.py` records why that replication fits inside the flat transient
fraction.

Stock Comfy Wan's tiny FP32 `patch_embedding` convolution keeps its own
profile, `wan_fp32_patch_embedding_v1`, reported only on an exact match of
parameter pair, upstream class, convolution structure, byte ceiling and
precision share; a near-miss lands on the generic islands profile, and an fp32
share over the ceiling remains `fp32` and is refused. Cheap quant
and LoRA launch refusals run before adapter/compile preparation; the worker
then performs one authoritative live scan after that preparation and
immediately before the first FSDP parameter mutation, so a same-patcher model
replacement or dtype drift refuses before sharding, and direct sharding calls
take the identical live-validation path. Exact-inode Comfy loading is supported
only on Linux workers with `/proc/self/fd` available, because the proven open
descriptor is exposed through a suffix-preserving procfs alias.

**Proving a clean reload is a capacity question, not a liveness one.** An
in-process unload on unified memory leaves the allocator arena committed to
the worker. The FSDP Gate must therefore fit a second load beside the memory
retained from the first. Before any proof side effect, the driver estimates
that load using the worker's admission helper. An estimated non-fit records
INCONCLUSIVE without unloading or rendering.

Recycling the fleet would return the arena, but would also violate that
no-side-effect refusal guarantee. Instead, `actor/fsdp_streaming.py` assigns
mmap-backed checkpoint tensors and shards each block before allocating the
next. A rank needs its shards plus one block rather than a full second copy.
`fsdp_required_bytes` adds the shared 5 GiB host floor for bf16/fp16 streaming.
FP8, INT8, and an unreadable checkpoint kind use the conservative 1.20
full-file bound plus that floor.

In the Flux2 capacity measurement, the head had 42 GiB free when
the 60 GiB BF16 reload was estimated (`capacity_fit.py`). With the driver's
text encoder resident, however, a 60 GiB Flux2 render under
`auto_gate=first_use` returns INCONCLUSIVE: the Gate rechecks reload capacity
after the baseline render (VALIDATION.md). The BF16 streaming and
FP8-mixed direct-wrap FSDP runs kept every sampled host above the floor, but
did not recalibrate the 1.20 bound. Additional runs exercised INT8
stored-byte FSDP for Krea2, Chroma, and Ideogram4 without calibrating that
bound or its floor (VALIDATION.md, memory calibration). If an in-process
second load cannot fit, `auto_gate=off` permits execution without that proof,
subject to the ordinary admission guards.

**A rank holds its shard once.** Never hand shard rows to the device straight
from the mmap'd checkpoint with `.to(device)`: for a pageable source the
driver pins the pages with write intent, and on a private file mapping that
breaks copy-on-write, so every row the copy touches becomes an anonymous page
of the worker. Those pages live as long as the mapping, and one replicated
fp32 island, scalar or matching buffer still file-backed keeps the mapping
alive for the model's life. Measured with that direct copy: a 61.7 GiB bf16 family with twelve islands held its 30.9 GiB
shard on the device and the same 31 GiB again as anonymous host pages, while
a family with none paid the shard alone, and ComfyUI then sized the load by
what was left. The build copies every row through one pinned bounce buffer
(`adapters/fsdp_shard_build.py`, `DeviceCopier`), so the mapping holds clean
page cache and nothing else; the direct wrap that quantized files take stages
each block onto the device the same way right before `fully_shard`, which
would otherwise move the block itself. After the build the leftover
file-backed tensors are cloned into anonymous memory and the mapping's pages
are given back before ComfyUI's own load (`actor/fsdp_streaming.py`,
`actor/mapped_pages.py`), and ComfyUI's per-module byte sum is made shard
aware (`actor/comfy_shard_size.py`): it reads a DTensor at its global size,
which charged a 30.9 GiB shard 61.7 GiB across the block loop. Nothing here
admits a load: the stock preflight still prices the launch, and the
partial-load divergence guard still refuses a rank that could not hold its
shard.

### 5.5 Model store: reuse + LoRA hot-swap

Each actor owns the state machine below. The `lora_low_rss` merge-and-free mode
and `slab_weights` zero-copy residency are documented in VALIDATION.md; neither
changes this key and transition contract.

Every fresh load begins with a process-rooted `FreshLoadOwnership` transaction
before the first FSDP pin or slab acquisition. Slab/arena ownership is rooted
while it is built and then handed into that transaction. A provisional FSDP pin
first publishes into its dedicated process root; exact identity then transfers
it into the transaction before downstream load or slot publication. Temporary
checkpoint-source descriptors instead use finalizable local ownership and a
separate fail-closed retention registry if close is ambiguous. Before model-slot
publication the transaction also records the exact `ModelStore`, slot attribute,
and `StoredModel` candidate. It disarms only after identity confirms that exact
object in that exact slot. If an interruption lands after publication, recovery
adopts the already-published slot instead of closing resources beneath the
resident model; otherwise the same transaction owns cleanup.

Failed-load teardown then publishes the complete slab/non-slab batch at module
scope before process-wide model unload, cache cleanup, collection, or any
resource close. Publication is removed only after that exact resource's
`close()` returns normally. Temporary Comfy slab hooks have their own
prepublished restore guard: both process-global hooks must be restored to their
exact original identities before the guard releases the slab's close barrier.
If either restore is interrupted or cannot be confirmed, the guard and slab
stay rooted; a later cleanup retries restoration before arena close. If any
ownership publication itself is interrupted, process poison remains latched
and no unproved destructive close begins, so recovery requires recycle rather
than reuse of unproven process state.

The retry boundary sits before numeric descriptor release: a
failed process-wide unload, cache collection, FSDP alias-unlink, or arena
`mmap.close()` leaves the mapping under known ownership and may be retried,
while anything after entry to `os.close(fd)` is ambiguous because the kernel may
already have reused that number. Such a close is never retried in the worker; it
is cleanup poison and requires process recycle. The exact ordering and the
uncertain-arena rules live in `actor/slab_arena.py`,
`actor/store_load_ownership.py` and `actor/stored_model.py`.

- `request_key = (unet_name, options, lora_signature, quant_kind)`: the store
  reuses an exact match on the same artifact identity, with no fresh load and
  no swap; a slot that a global unload evicted reloads its GPU weights on that
  reuse.
- `base_key = (unet_name, options)`: a match with a different LoRA stack is a
  **hot-swap**. The store clones the pristine base patcher (its weights stay on
  the GPU) and applies the new stack with ComfyUI's own merge math (fp32
  intermediates and stochastic rounding), so the result matches stock ComfyUI.
  Under `lora_low_rss` the swap instead restores the LoRA-covered weights from
  the checkpoint in place and bakes the new stack (lazy un-bake); a swap that
  fails recoverably drops the slot for a fresh load.
- Quantized models (every detected kind except bf16, fp16 and fp32): a
  hot-swap outside `lora_low_rss` (the clone path) also unloads every model, so
  the patched GPU weights rebuild from the CPU state dict and the other slot
  reloads on its next use. This is the measured fp8/int8 conversion-spike
  workaround (`store_detect._is_quantized_kind`).
- A miss is a full load. Every transition logs its class (`reuse`, `hot-swap`
  or `load`), so the log line names the source of a slowdown.

With automatic slab residency, a first stock load that identifies a family validated for slab permits exactly one in-memory retry, even if the best-effort family memo is unavailable. Slab-capable `.safetensors` and `.sft`
suffixes are case-insensitive. Unsupported paths and dtype fallback remain
stock, and an already-consumed retry cannot be re-armed into a reload loop. A
memo key includes device, inode, size, mtime, and ctime, so rewriting the same
inode and restoring its old size/mtime still invalidates the family decision.
A metadata sweep that touches only ctime also invalidates every row on the box
at once, so a side index on the safetensors header digest
sits beside the memo. A lookup consults it only after the memo key misses
outright, confirms a hit by reading this box's own file a second time, and
writes nothing. Index rows come from the same loads that write memo rows, and
no row is ever adopted from a peer. A load's own memo write
also asks the index on a key miss: when the digest names the family this load
detected, the write drops the rows the sweep left behind for the same bytes;
when it names another family, every older row stays and the load warns. The
index confirms a family already established by loading those bytes on this
host. It cannot infer a family from an unloaded header.

**A resident build declares itself to ComfyUI.** ComfyUI decides how much of a
model it can hold at sample time, and it reads the answer off two fields of
the model object: `model_loaded_weight_memory` and `model.device`. A slab
build and an FSDP shard build fill neither, because the weights reach the
compute device without ComfyUI's own load ever running, so ComfyUI asks for a
whole model of free memory for bytes the box already holds.
`actor/resident_ledger.py` closes that gap by running ComfyUI's own full load
once at the end of the fresh load, through `partially_load(load_device,
1e32)`, the call ComfyUI's manager makes under `force_full_load`. A stock
resident (neither slab nor FSDP, and not `comfy_managed`) takes the same
build-time load for the opposite reason: left to sample time, ComfyUI sizes
the move while the loaded file still sits on the host, can place only part of
the model, and the rank meets the partial-load guard. Placing it at load time,
where the stock price was taken, frees the host copy before any sampler sizes
the model (docs/VALIDATION.md, stock placement peak). Running
ComfyUI's load rather than writing its fields by hand preserves the
`.to(device)` walk, per-module `comfy_patched_weights` marks, weight-function
setup, and hooks. A patcher that still holds weight patches or wrapper patches
is left undeclared, because baking weight patches costs a model-sized backup
that belongs at sample time, not in the build's peak. Slab and FSDP reach the
declaration with an empty patch dict: slab requires `lora_low_rss`, FSDP LoRA
requires it too, and both bakes clear the patches. A stock resident whose
stack is still pending (baked hot-swap residency) keeps ComfyUI's sample-time
load. The declaration is per fresh load. A later lazy swap on a slab or FSDP
resident builds a patcher with a new patch uuid and takes ComfyUI's ordinary
reload path; a lazy swap on a stock resident names its new
uuid on the model instead, so ComfyUI does not unpatch and reload it
(`store_bake._adopt_patch_uuid`). The comfy canary's resident ledger seam
checks the early return and the fields the load writes.

Render-time model patches are outside both keys. `ModelSpec.model_sampling`
carries either `None` or a validated canonical wire spec such as
`{kind: "sd3", shift: 5.0}`. The worker applies that spec to a clone of the
resident ComfyUI patcher immediately before sampling (and before sigma
calculation where applicable), so changing a sampling shift neither mutates the
stored patcher nor causes a checkpoint reload. The coarser gate combination key
also excludes this render field, while Identity Gate and Fleet authorization
bind the complete primary and optional unconditional requests, so mid-flight
sampling-spec drift fails closed.

### 5.6 Transfers

- Conds and latents from the driver to the actors ride Monarch actor
  messaging: about 440 MiB/s, so a 1536² latent (about 18 MB) takes about
  40 ms (measured on the tested Spark pair).
- Leader latents from the actors to the driver: **RDMABuffer one-sided read,
  size-gated, default OFF** (see docs/VALIDATION.md). When it is
  on, latents below `rdma_min_bytes` (8 MiB) return by messaging, so image
  workloads keep the message path used in their benchmarks,
  and latents of 8 MiB or more (video) take RDMA (7.55 GiB/s in an earlier measurement). It falls back to messaging when ibverbs is unavailable. The
  release gate for the default-off posture requires the below, exactly-at and
  above-threshold cases plus a fresh reattach all to use messaging with intact
  tensors, no RDMA descriptor ownership or ACK, and zero relevant leases or
  jobs. That gate covers the dual-Spark NCCL-plus-message render path, not
  native RDMA. It passed with six production zero-step custom
  sampling cases: three sizes around 8 MiB, repeated after fresh attach,
  exact primary/denoised tensors and confirmed owned recycle. No denoiser
  forward ran; this is messaging acceptance, not model fidelity
  ([recorded scope](VALIDATION.md#default-off-messaging-return)). The
  native-enabled probe arm is retained outside this repository for future
  requalification.
  Native RDMA is **NOT RUN / HOLD**. Requalification requires a stable
  torchmonarch release with `rdma_ibverbs_target` and a passing selected-rail
  matrix: fresh integrity, ownership, recycle and reattach evidence on the
  selected rail. [Validation](VALIDATION.md#native-rdma) records the failed
  production-path check and the limits of earlier microbenchmarks.

  Current limit (measured on the tested Spark pair): the torchmonarch pin hashes each
  host-memory registration across every NIC tying for the best CPU path, per
  transfer end independently, so on a multi-rail point-to-point fabric it can
  pair a transfer across rails with no network path (observed at 8 MiB). The hash is deterministic per (address, size), so one passing
  transfer of a shape does not prove the next. `DGXM_RDMA_QP_SPLIT` above 1 is
  expected to scatter across every tying rail rather than open several queue
  pairs on one, so it is treated as carrying the same hazard and stays at 1. A
  rail selector is necessary but does not by itself turn the default back on:
  the selected-rail matrix above must pass first. Dated measurements are in
  docs/VALIDATION.md. RDMABuffer requires a CPU tensor (host-staged); GB10 has
  GDR off, so no GPU-memory path is lost. ComfyUI packed `NestedTensor`
  results never take RDMA. Every direct modality is detached, made contiguous,
  moved to CPU, and validated; the wrapper then rides actor messaging and
  never owns an RDMA keepalive. SamplerCustom validates the exact packed x0
  width before rebuilding the original ordered modality shapes, and the driver
  rejects missing, non-finite, aliased, or structurally changed denoised
  output. Pipeline and Fleet concatenate packed batches one modality at a
  time. A finite one-value sigma vector is the only zero-step custom schedule;
  malformed schedules fail before sampling. Packed latents remain non-DP.
  Production descriptor ownership is a fail-closed token transaction. Before
  publication, the actor's process registry owns every registration and its CPU
  backing under `(setup_generation, random 128-bit token)`. Live entries are
  capped at pipeline depth; a full registry uses actor messaging, never eviction
  of an unACKed owner. Construction intent and every later handoff are
  prepublished. A constructor outcome that cannot be proved poisons production
  RDMA in that actor and requires an Attached mesh reset rather than a silent
  message retry.

  The driver publishes a generation read lease before handing an on-loop job to
  its bounded scratch thread. If process-root publication is interrupted before
  any token, preparation, or runner start, the driver proves those absences and
  removes that exact root. An incomplete or failed native read performs zero
  native drops and retains the destination and every part because the operation
  may still be live. After a completed read, cleanup attempts each local drop
  exactly once. The same cleanup is safe when an off-loop scratch contender has
  proved that the native operation never started. A failed or ambiguous drop is
  never retried and sends zero ACK; only its local handle and any memory it may
  still touch stay process-owned. A confirmed local drop releases the driver's
  registration, not the worker's registry backing.

  If both prebuilt scratch threads fail to start before either claim is applied
  while the caller is on an active event loop, there is no safe off-loop
  contender. The caller performs zero native reads, drops, or ACKs; the bounded
  `PENDING` job, generation lease token, descriptor parts, and worker backing
  remain process-rooted. Lifecycle and recycle refuse that handle, and the
  driver process must restart.

  Only after all local drops and read-authority exit are confirmed does the
  driver broadcast the generation/token ACK. It requires exactly the configured
  world-size responses, exactly one `released` or `already_released` owner, and
  `unknown` from every other worker; rank values are not identity. Response loss
  retries only this idempotent ACK, never a native drop. A validated ACK is the
  only in-process event that retires the worker registry backing. Zero owners,
  multiple owners, malformed/partial responses, or an unconfirmed ACK outcome
  fail collection and abandon/block the setup generation. Multi-DP collection
  still drains every unique leader token before raising the strongest failure:
  an operational cancellation outranks ordinary failures, with the first error
  retained within the same class. Production does not recycle automatically:
  after work retirement the client/operator must explicitly recycle the
  ProcMesh.
- **Cross-render pipelining**: the `KSampler Pipeline` node
  and Init `pipeline_depth` overlap render N+1's control plane (conds transfer,
  dispatch and latent return) with render N's compute, through an eager-submit
  and deferred-collect split (`submit_sample`/`collect_sample`). It is opt-in;
  depth 1 is off, and the RDMA owner registry cap above follows the depth. The
  gain is queue-only and shared-model-only and scales with the control-plane
  fraction: negligible for compute-bound image renders, material for video,
  large latents or many fast renders. The actor lock keeps compute serial, so
  every latent stays byte-identical.
- The VAE and text encoder stay on the driver as stock ComfyUI nodes, so
  CONDITIONING and LATENT objects flow through the graph as usual and the node
  ecosystem keeps working. Worker-side TE/VAE is a later headless-pipeline
  option, not the node path.

### 5.7 ComfyUI node surface

Category **"DGX Monarch"**. The node surface is small by design:

- `DGXMonarchInit`: attaches the cluster (its fabric profile comes from
  cluster.toml, §6.1) and picks the topology (`auto` by default). Outputs
  MESH. It also carries the `pipeline_depth` widget for the KSampler Pipeline
  node (§5.6) and `mmap_fallback` (BOOLEAN, default off: on unified memory the
  worker reads safetensors through the lean pread backend, which avoids the
  mmap page-cache spike at load, while FSDP loads keep mmap; turning it on
  falls back to the stock mmap loader for an A/B).
- `DGXMonarchUNETLoader`: checkpoint name plus a weight-dtype option; returns
  a mesh-model handle. `default` keeps stock ComfyUI selection, and `bf16`
  maps to `torch.bfloat16` for capacity-sensitive loads. Unknown or malformed
  values fail independently at the driver and worker boundaries, and the
  choice is part of model-store and Identity Gate request identity.
- `DGXMonarchLoraLoader`: stackable; feeds the LoRA signature (hot-swap aware).
- `DGXMonarchModelSamplingSD3`: wraps a DGXM model with a validated per-render
  SD3 shift, applied to a clone on the worker and kept outside model residency
  identity (§5.5). ComfyUI's own default for the Kandinsky5 video checkpoints
  is shift 10; docs/MODELS.md gives the shift each Kandinsky5 model needs.
- `DGXMonarchQwenImage21Cache`: sets Qwen Image 2.1's replicated prefix KV
  cache on every worker, off by default. `default` storage is lossless; `int8`
  and `int4` use ComfyUI's compressed K/V cache and are approximate. The policy
  rides in the loader options, so it is part of `base_key` (§5.5).
- `DGXMonarchKSampler` + `DGXMonarchKSamplerAdvanced`: stock sampler semantics,
  with the same widget names in the same order as the stock nodes.
- `DGXMonarchKSamplerPipeline`: a seed sweep that takes a seeds string (one
  image per seed) and returns one batched LATENT, each seed byte-identical to
  `DGXMonarchKSampler`. Init `pipeline_depth` above 1 overlaps the renders
  (§5.6); depth 1, the default, runs them in sequence. For an ordinary image
  seed sweep, an Empty Latent `batch_size` is simpler.
- `DGXMonarchFleetKSampler`: independent world-1 jobs dispatched in waves.
  Private topology-only latent metadata such as `_dgxm_latent_downscale` is
  removed before every Fleet worker request and from the returned LATENT, as
  on the ordinary sampler path. A dedicated driver thread collects each
  submitted future, and the coordinator reads a stdlib completion queue, so a
  slow older job cannot hide a younger failure. The first failure broadcasts
  fire-and-forget cancellation to unfinished peers before their mandatory
  lease, audit and progress settlement. Driver-side latent materialization and
  result validation happen at that completion boundary; output publication and
  ownership settlement stay in prompt order. Each dispatched job opens a row in
  the `fleet` sub-block of the telemetry render block, and its answer closes it
  (`src/dgx_monarch/telemetry_fleet.py`). The row carries the job index, the
  prompt digest, the answering host, the rank the driver submitted to, that
  rank's short box label, the worker's measured duration, and `started_s` and
  `ended_s`, both counted from the moment the wave opened. The block carries
  that moment and is retained the way `render.last` is. The sidebar timeline
  draws one row per job in dispatch order, tagged with its box label and
  answering host, beside the per-box memory cards. A job that never came back
  keeps the row its dispatch opened and reads as no result, so a stopped wave
  still names the box the lost job went to. A row carries
  `host_reported`, and the sidebar says it answered from another box, when the
  answering host is one this driver's config gives to a different box. The
  reply's own rank is never compared: every Fleet worker is rank 0 of its own
  world-1 mesh, so its rank names no box, and comparing it with the driver's
  pair-wide rank would mark every healthy job off the first box.
- `DGXMonarchDualModelGuider` + `DGXMonarchUncondUNETLoader`: the Ideogram4
  asymmetric-CFG feature implemented for distributed execution.
- `DGXMonarchSamplerCustom`: wraps the stock SAMPLER/SIGMAS/GUIDER objects the
  graph supplies (see the anti-pattern ban below) for cluster dispatch; it
  validates the packed x0 width before rebuilding the ordered modality shapes
  (§5.6).
- `DGXMonarchBasicScheduler`: sigma computation on the worker's resident model,
  so a schedule never forces a driver-side load. It needs an explicit Init
  topology: `auto` resolves at the first render, after the sigmas are needed.
- `DGXMonarchBasicGuider` / `DGXMonarchCFGGuider`: the stock guider shapes, as
  specs the workers build against their resident models.
- `DGXMonarchIdentityGate`: runs one ceremony on demand and writes its verdict
  row; `auto_gate=first_use` runs the same ceremony before a new combination's
  first render, and `dgxm gate` drives it from the CLI (§6.1).
- `DGXMonarchStatus` / `DGXMonarchClearVRAM`: ops nodes. ClearVRAM returns its
  STRING status at slot 0 and, when a LATENT is connected, passes it through
  slot 1 only after cleanup. With `level=recycle`, the latent is withheld
  unless the client-owned ProcMesh stop is confirmed. A downstream VAE decode
  therefore waits until worker model and allocator memory has returned to the
  OS, without stopping the persistent Worker services.

**Anti-pattern ban:** do not mirror ComfyUI's node tree into this repository.
Custom sampler and guider ecosystems are supported by *wrapping* the stock
objects the graph supplies (SAMPLER/SIGMAS/GUIDER inputs), not by forking their
node code. When a stock node's behavior must change on the cluster path, wrap
it at dispatch time with a keyed wrapper.

Node contract stability is workflow stability: **NODE_CLASS_MAPPINGS keys,
input names, and defaults are semver-governed API.** New inputs are optional or
hidden and go at the end, because ComfyUI stores widget values by position.
`tests/test_node_contract.py` snapshots the keys, ordered input names, types
and defaults (`tests/node_input_contract.json`). Ten of the 18 nodes declare a
hidden `UNIQUE_ID`; no node reads its value, but it is part of that contract.

### 5.8 Progress, logging, observability

- The leader actor streams per-step progress to the driver over a Monarch
  channel port, so a cluster render's ComfyUI progress bar tracks the whole
  world rather than one rank.
- One-time first-render costs announce themselves in the UI, not only in the
  console: NCCL bring-up, cold load and the identity ceremony each publish one
  notice through `first_render.notice()`, which writes the driver log, the
  telemetry event ring (`kind="notice"`, so the sidebar tail and `dgxm top`
  carry it with no new route), and a `dgx-monarch.notice` websocket broadcast
  the browser turns into a toast. It uses the same channel as the fault hook
  and, like it, stays silent when there is no ComfyUI process. The two deferral
  announcements deduplicate per driver process, because NCCL bring-up and a
  cold load are each paid once there; the gate phases announce once per
  ceremony, because the gate runs per model+stack combination.
- On a cluster, worker logs stay on their host in the Worker service log
  ([docs/TROUBLESHOOTING.md #4](TROUBLESHOOTING.md#4-supervisionerror--process-exited-with-non-zero-code)), and each dgx-monarch log line names its host.
  They do not reach the ComfyUI log: the attached HostMesh has `stream_logs`
  off, and nothing calls `ProcMesh.logging_option` to forward them.
- `DGXMonarchStatus` reads actor `status()` payloads; `dgxm top` and the browser
  sidebar show the live telemetry view. `dgxm status` is the cheaper lifecycle
  check: the host runner verifies the exact Worker service process
  and its exact passive `/proc/net/tcp{,6}` LISTEN entry. It never connects to
  the Monarch protocol socket as a readiness probe. Its one driver-side row is
  mesh health, read over the telemetry route: lease and teardown state decides
  whether a render can attach and no host probe can see it, but the CLI must
  not become a second Monarch client to ask ([docs/TROUBLESHOOTING.md #65](TROUBLESHOOTING.md#65-dgxm-status-and-dgxm-doctor-are-green-and-every-attach-still-fails)).
- Strict acceptance provenance uses a setup-token-bound worker baseline rather
  than diagnostic status alone. The setup snapshot is captured before READY;
  it is retained with that generation and cleared on teardown or failed setup.
  The post-operation snapshot and Comfy resolver artifact metadata identities
  are captured behind prior GPU mutations. Both snapshots bind exact
  rank/world/topology, source-only import policy, path-redacted machine and
  process-lifetime identities, package versions, Gate protocol, clean Git or
  exact managed-copy source manifests, the exact ComfyUI checkout, structured
  loaded-module origins, and the fixed custom-node bootstrap policy. Worker
  artifact identities hash a stable file-stat tuple, not the file bytes;
  docs/THREAT_MODEL.md states that the external acceptance preflight
  full-hashes the selected artifact bytes on both hosts.
  A managed source copy may have no local `dgx-monarch` distribution metadata
  (or stale metadata from an older install), so that package version is always
  read from the loaded source `__version__`; Torch, Monarch, xfuser, and
  yunchang remain bound to installed distribution metadata on every host.
  Loaded-origin maps may only grow within one process lifetime; removal,
  rebinding, immutable provenance drift, or cohort mismatch rejects the phase.
- Acceptance establishes READY before an operation's PRE snapshot, then keeps
  eager worker loads and sampling inside its PRE/POST pair. Two-stage pipelines
  add contained, non-overlapping stage pairs in exact order. A strict,
  explicitly attested Identity Gate first durably marks every capability
  RETESTING and process-local INCONCLUSIVE, then runs PRE, the GPU proof, and
  POST before any PASS can authorize work. It publishes process-local PASS
  before the durable PASS record. Automatic first-use ceremonies expose no
  external evidence observer, but still run the mandatory internal all-rank
  source-cohort PRE/POST bracket; they retain the process-local denial while
  recording the result, revalidate the current artifact/context token, and only
  then publish process-local PASS. A typed refusal still closes its outer
  operation's POST edge. A typed refusal met inside an FSDP clean-reload proof
  settles that combination rather than aborting it: the ceremony closes its
  RETESTING transaction with one terminal INCONCLUSIVE row, keeps the guard's
  own class and remedy on the wire, and retracts the process-local
  prejudgement, so the next queue meets the same guard instead of a sticky
  verdict. An untyped abort keeps its denial and says what stopped the proof.
  Diagnostic status publishes its event watermark and tail from one locked
  snapshot, so the two values always describe one coherent event history.
- For actor state the node surface does not show, `monarch-tui` (shipped with
  torchmonarch) attaches to a live mesh; docs/TROUBLESHOOTING.md ("Power-user
  escape hatch") covers it.

### 5.9 Refusal taxonomy: what a typed refusal is allowed to do

**Rules gate accuracy, never availability.** Missing correctness evidence
may require explicit consent, but it must not be treated as measured failure.
Unreadable evidence is different: it fails closed until the evidence can be
read, and consent cannot bypass it. Capacity refusals permit rescue only when
a priced residency fits every required rank and the identity-gate ledger
reads clear for those artifact bytes. If no residency fits, the capacity
refusal stands. Each typed refusal declares its class, which determines
whether a bypass is allowed.

| class | what it means | bypass policy |
|---|---|---|
| **P** physics | the model, ComfyUI, or the machine cannot do it: batch caps a family's DiT at 1, no kernel carries the mask, the checkpoint bytes on disk do not match themselves. | **None.** No consent, no waiver, no env var. The message must name the working alternative, because that is all the user can act on. |
| **C** capacity-protective | a preflight: this will not fit here. | **Never a bare refusal.** Offer the fitting strategy first (slab residency through one click), refuse outright only once the ladder is spent, and then say what would fit. Consent-clearable where a rescue exists and the identity-gate ledger reads clear for those bytes. |
| **U** unproven correctness | family validation, first-load-stock, Gate results, and untested fold degrees. Correctness is unproven; no error has been measured. | **Always consent-bypassable.** Log the bypass, write a permanent waiver row, then proceed. Neither U guard has a card this release (`waivable_now` is False for both) and no site raises class U: a rescue grant covers first-load-stock for its load, and an explicit slab request loads a family without slab validation. |
| **K** known-wrong math | measured wrongness: attended pad rows on ring, an approximate attention kernel the operator selected, and a sharded nvfp4 render of a family measured past the fidelity floor. | **Refuse by default.** The ring divisibility-pad guard carries an expert waiver: one card in the accuracy style carrying the measured wrongness in its own text, `DGXM_WAIVE_KNOWN_WRONG=ring_pad` headless, never auto-eligible, and every render under it stamped. The Sol-Attn kernel carries the same shape under `DGXM_WAIVE_KNOWN_WRONG=sol_attn`, and a sharded nvfp4 render of a measured family carries it under `DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`. The card and the headless value use the unscoped name. Sol-Attn and shard-quant raise family-scoped at every site; ring pad raises scoped for MiniMax H3 and unscoped at the adapter-base backstop. Historical PixelDiT waiver rows remain readable, but that guard was retired after the exact-gather fix passed hardware parity. |

Two related boundaries:

* A **byte-verify certificate mismatch** is class P. The slab does not hold the
  checkpoint's bytes, so the load aborts and nothing is adopted. It is a
  storage or artifact fault, not a model-correctness finding: it writes no gate
  verdict, quarantines no lever, and revokes no consent. Classifying it K would
  imply a waiver could exist, and none ever can.
* **`auto` never substitutes known-wrong math.** A waiver is an expert choice
  about a topology the operator typed, so the auto-resolution refusals carry
  class K with no guard and say in one sentence why no card will appear.

#### The tag rides in the message

A worker's refusal reaches the driver as Monarch-wrapped text; the exception
object does not survive. So the class is carried in the message by one helper,
`refusal.py`:

```
[dgxm:C guard=stock_load_preflight waivable=1] stock residency cannot load ...
```

One helper appends the panel action and headless fallback to every tagged
message. Waivable refusals must offer the action; non-waivable refusals must
not. `refusal()` raises on an inconsistent declaration.

`refusal.GUARDS` is the frozen guard vocabulary shared by the refusal sites,
the consent subsystem and the WAIVER row's `target_guard` field. A guard
carries `waivable_now`, the current truth: a guard whose waiver is designed but
not wired stays False and no site may declare it waivable, so no message can
name a card that does not exist.

`tests/test_refusal_classes.py` walks every module, holds the classified sites
in a frozen ledger in both directions, and pins the per-module count of sites
that are not classified yet, so that backlog can only shrink and a new typed
refusal that nobody classified fails the build. Section 7 carries the dated
ceiling.

#### Consent, waivers, and what they can never buy

Consent is explicit this release: a render that reaches a consent point fails
immediately with its typed refusal, the driver registers a pending consent, and
the DGX Monarch sidebar shows one card with one button. The standing
auto-rescue toggle exists and defaults off. Env vars are the headless fallback,
never the primary route.

Four invariants that no consent can change:

1. **Every bypass is audited.** A grant writes a permanent ledger waiver row
   naming what was waived, under which capability context, from which channel,
   before the memo that authorizes it is written. A memo without a row is an
   unaudited bypass and must not exist; a row without a memo is harmless.
2. **A quarantine outranks every consent.** A measured identity-gate FAIL
   quarantines its levers for those artifact bytes in that capability context.
   The quarantine is applied before residency is resolved, the consent memo is
   never consulted for a quarantined combination, and the worker treats an
   explicit `slab_weights=off` as terminal regardless of any consent, so a
   stale or buggy driver push cannot rescue into a quarantined lever. A gate
   FAIL also revokes the class C memos it disproved, so the panel never shows a
   live residency consent for a path the gate has proven unsafe. A class K
   accuracy waiver clears a math guard rather than a lever, so a FAIL neither
   disproves it nor revokes it; it stays live and stays stamped.
3. **A certificate is never a PASS.** The byte-verify certificate attests that
   the bytes that entered the slab are the checkpoint's own bytes. It says
   nothing about pixels. When the stock reference cannot load on this hardware,
   the ceremony stays INCONCLUSIVE and the certificate is recorded beside that
   verdict, never in place of it. Certificates and consents never convert an
   INCONCLUSIVE into a PASS.
4. **A consented rescue permits one load; it does not validate the family.**
   It bypasses the
   consequence of the first-load-stock rule for one load. It does not write the
   family memo early, does not add the family to the validated slab set, and does not
   earn the sibling-context stamp, so a later `auto` render of the same file
   uses a different capability context and requires a separate proof.

**What an aborted ceremony may take, and from whom.** If a Gate run ends
without a verdict, first-use handling normally disables both residency
optimizations before dispatching the requested render.
`nodes/gate_quarantine_scope.py` limits that change in two ways:

- A waivable class K refusal concerns accuracy, not residency. It disables no
  residency setting and sends no worker-policy update. The refusal tag is
  checked against `refusal.GUARDS`; this applies to Sol-Attn and every other
  waivable class K guard. The cached denial retains the class for later
  requests. `authorize_normal_render` still requires a PASS and stamps the
  dispatch stock when none exists.
- An ordinary abort affects only the combination tested. The next combination
  through the graph starts with the graph's requested settings. The aborted
  combination remains denied until its next Gate run resolves it.

An abort leaves the earlier `RETESTING` row in place. These limits do not
clear a measured FAIL: invariant 2 preserves that quarantine across sessions
until the Gate protocol or package version changes. An unscoped FAIL remains.

For a LoRA stack under an explicitly declared FSDP topology, an abort leaves
`lora_low_rss` unchanged and disables `slab_weights`. FSDP LoRA baking requires
`lora_low_rss` (`actor/fsdp_lora.py`); disabling it would force an unsupported
fallback and make the next render refuse.

**Class K in practice.** The known-wrong guards reuse that whole subsystem and
add one rule of their own: an output produced under a class-K waiver is
stamped. The waiver row carries a mandatory `stamp` field, the per-render
`use` row ties it to the run, and the render result carries the same stamp
back to the driver, so the driver can tell every waived render from a clean
one. The stamp stops at the driver: dgx-monarch returns a latent and ComfyUI's
own save node writes the file, so the PNG is not stamped. Three additional
requirements apply. The card must describe the measured
error so the operator can assess it. The headless variable must name its
guard; a generic truthy flag cannot waive an accuracy check. No class-K kind
is auto-eligible, and there is no accuracy equivalent of the standing
auto-rescue toggle.

#### The residency ladder

Under `auto`, residency is resolved in this order: explicit request, slab for
a validated family, stock if it fits, then an offer to use slab with consent
and a mandatory byte-verify certificate. Stock is the default otherwise.

Comfy-managed residency is a separate, opt-in process setting on the Init node.
It starts ComfyUI's DynamicVRAM and delegates weight placement to ComfyUI.
It has no `auto` value and takes precedence over the request-level choices.
It forces both residency optimizations off and refuses LoRA stacks, FSDP,
dual-model renders, Fleet calls without an exact PASS, and RDMA latent return.
It writes no capacity certificate or capacity audit row because it performs
no byte verification.

Its Gate run tests repeatability within comfy-managed residency. It cannot
compare against classic residency in the same process: DynamicVRAM installs
process-wide instruction patches that cannot be undone. Cross-residency
comparison requires the separate hardware acceptance test in docs/VALIDATION.md.

**Where the yields live, and why it is not in the ladder.** Comfy-managed
residency forces `lora_low_rss` off, selecting the baked hot-swap path. That
path returns before `actor/store_residency.resolve` because the base key
excludes the LoRA stack. A yield inside the ladder would therefore admit
ComfyUI's memory manager into the swap path, where pool pressure has caused
it to overwrite a real weight with a sentinel (docs/VALIDATION.md, rejected
sentinel-backup design). Both yields instead sit in `actor/store_fsdp.ensure`,
which every load path shares, and refuse before `ModelStore.ensure`.

**The bootstrap-policy consequence.** Changing the widget on a live worker
requires an Attached mesh reset: aimdo's inline patches and the patcher rebind
cannot be undone in-process. Failed bring-up also latches refusal. ComfyUI
marks bootstrap complete before bring-up finishes, so a later call would
otherwise return early and run classic residency under a comfy-managed policy.
That fallback could bind a durable PASS to a policy the worker never honored.
A persisted quarantine or a Fleet call without an exact PASS therefore also
refuses; neither can downgrade the running process.

**A class-K guard that blocks the ceremony answers first.** Gate proof renders
ignore accuracy waivers. A class-K refusal therefore blocks every proof
attempt for that combination. Stock-capable residency paths can continue to
the ordinary render, where an explicit waiver may apply. Comfy-managed
residency cannot fall back in-process, so it reports the class-K refusal and
card directly instead of asking the user to repeat the same failing proof.

The driver remembers the blocking guard and includes it, its waiver, and the
working settings in subsequent residency refusals. Accepting the waiver clears
only the ordinary render's accuracy guard; it does not authorize residency.
Running that combination requires `auto_gate` off or comfy-managed residency
off. Sol-Attn follows the same separation when it skips its Gate run
([docs/TROUBLESHOOTING.md #84](TROUBLESHOOTING.md#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call)).

**How a grant reaches the load.** Only the driver reads the consent store.
`nodes/consent_projection.project` copies a valid consent into `worker_args`,
which is published atomically to all ranks and included in the Gate capability
context. Consented and unconsented requests therefore have distinct contexts.
Projection runs before the loader's eager load and in
`nodes/render_preflight.bind_packed_render_model` before topology resolution.
Persisted-quarantine enforcement runs after both and overrides consent.

Consent is accepted only when the ledger contains no applicable FAIL for the
exact artifacts. `nodes/consent_quarantine` uses the Gate's validity rules:
a context-scoped FAIL expires when its recorded protocol or package version
changes; a legacy unscoped FAIL remains.

**What the loader asks the ledger, and in what order.** The loader checks the
ledger before resolving consent and again when applying it. Both checks first
query without a capability context. A FAIL or other durable denial is final
at that stage, because the loader runs before `ensure_live` and cannot build
the Gate's full context.

Only `unknown` triggers a second query using the load's own context. This lets
a clear result for the load override an unknown result from another context.
`live_grant` uses the same fallback only for consents with no context of their
own; otherwise a granted card could be rejected on every subsequent queue.

Gate rows use contexts that this loader cannot construct, and load certificate
rows use a separate audit namespace. The contextual query therefore clears
unknown results except ledger damage. A torn row still blocks consent because
it could have contained a verdict for any context.

**An over-budget load with no stable artifact identity refuses.** Consent
requires a stable fingerprint of the artifact bytes. A checkpoint still being
copied may have none. If the loader estimates that the load will not fit, it
raises a non-waivable class C refusal naming the artifact before resolving
consent. This explicit refusal must bypass the estimator's general error
handler, which otherwise allows execution when estimation fails.

**Where the ceremony cannot help.** If stock residency cannot load the
checkpoint, the Gate has no stock reference and returns INCONCLUSIVE. This is
unproven correctness, class U, rather than measured error, class K.

With active capacity-rescue consent and no LoRA stack, this outcome keeps
`slab_weights` and its required `lora_low_rss` enabled. Returning to stock
would retry the load already found not to fit. A LoRA stack introduces a
separate unproven path, so both settings are disabled, except that FSDP keeps
`lora_low_rss` as described above. `CAPACITY_CERTIFIED` records the load's
byte verification and availability; it is never an accuracy verdict.

**Two windows, not one sum.** The loader checks two peaks separately. The
transient window covers loading against MemAvailable minus the reserve: a
stock load uses about 2.1x the artifact size while host and device copies
coexist. The settled window covers weights at 1.0x plus the driver models and
other memory the graph still needs to allocate. Slab removes the host copy
and its transient peak; it does not reduce the settled graph requirement.

**A declared shard build is priced as a shard.** Both windows charge the whole
file, which is what a stock load places, and under a `*+fsdp` preset no rank
ever places it: each holds `file/world`, and the streaming build assigns
file-backed rows into shards rather than copying the file, so it retains no
legacy arena either. The loader node charges that placement one rank's share
times `driver_footprint.FSDP_SHARD_BUILD_FACTOR`, plus the pinned bounce
buffer the build allocates, and drops the arena term; the text encoder, the
VAEs, the reference encodes and the activations are unchanged. Five facts
decide it, all on the driver before any RPC: the preset declares fsdp, the
world resolves to 2 or more, the driver host holds one of the shards and not
several (`gpus_per_host` is 1; a host running two ranks holds two shards, and
a `this_host()` mesh runs the whole world in children of the driver process,
so both keep the whole-file price), the header reads a full-precision kind (a
quantized file takes FSDP2's direct wrap, which still moves each block at full
size, so it keeps the whole-file price), and comfy-managed residency is not
requested. Anything else keeps the whole-file price, because a share applied
to a placement that is not a shard build admits what the box cannot hold. An
`auto` preset keeps it too, even where the auto table would resolve to fsdp:
the price reads the declared preset and never the resolved topology, so the
unresolved reading is the one that charges more. Slab policy does not enter
it: `worker_env.slab_mode_effective` is False under FSDP, so a sharded rank
takes no slab whatever it asked for. This moves the price
and not the decision rule: the refusal is the same class C, at the same window,
with the same rescue, quarantine and consent arms behind it, and the card names
the share and the world it priced. The factor assumes the shard build's bounce
copy is deployed on every box that loads; without it a rank pays about twice
its shard and this charge is too low. Under `auto`, residency is resolved against the validated slab set before either window is estimated. A family outside that set loads through stock and is charged for the host copy. Both windows also charge the absolute 5 GiB host floor, as the
operator's reserve when they set no larger `uma_reserve_gb`.

**Slab residency is priced.** `capacity_fit.slab_load_fit` charges the file
plus a host-memory floor, capped by the arena a stock load would retain. The
slab estimate therefore never exceeds the stock estimate. Every slab selection checks this estimate before returning; `preload_capacity_check` checks it
again immediately before entering ComfyUI.

**One absolute host floor.** The stock price, the slab price, the
comfy-managed capacity check and the driver's own two windows all charge
`capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES`, 5 GiB, sized from ComfyUI's own
rule for keeping a model whole rather than loading part of it (the derivation
is in `capacity_floor`). It does not scale with the artifact, because what it
pays for does not. With two floors, one residency admits on a box what another
refuses, and the driver admits a placement its rank would refuse. The
floor is a lower bound on the room kept, never a cap: a larger
`uma_reserve_gb` wins in the driver's two windows and in the slab price, where
the arena a stock load would have retained caps it. No leg that ran clean is
refused at this floor: `tests/test_driver_footprint_preflight.py` re-prices the
driver's recorded legs against the shipped constant rather than a literal.
FSDP uses the same floor in its shared worker, agreement, and clean-reload
arithmetic, and its final worker capacity check re-takes that kind-aware price. The
stock path re-takes `stock_load_fit`, including the host copy and the floor,
rather than a bare-file check. The driver's own loader window remains a
separate, conservative shard-plus-bounce estimate against the same reserve.
The pair rendered through the bf16 streaming and fp8mixed
direct-wrap FSDP capacity checks, and a stock load on one box refused class C before any
load (docs/VALIDATION.md, capacity-admission acceptance). Direct production guard checks under real memory pressure
passed for the final stock capacity check after an initially successful resolution,
the unpinned managed capacity check, and both driver windows at about 4.54 GiB of
projected remaining headroom. No weights were loaded. These checks establish
guard integration, not a full natural loader race, render under pressure or
capacity-coefficient measurement ([scope](VALIDATION.md#memory-calibration)).
Campaign runs also exercised INT8 stored-byte FSDP for Krea2,
Chroma and Ideogram4. Those runs confirm execution, not the 1.20 full-file
bound or its floor.

**Memory estimates by residency.** Each internal residency choice must have a memory estimate and a row in this table.

| rung | what it charges | file and function |
|---|---|---|
| `comfy_managed` | one resident copy, or 2.3 times the file where ComfyUI stages through a pinned host buffer (measured on the tested Spark pair), plus the floor | `actor/capacity_quote._managed`, then `capacity_fit.managed_required_bytes` at `actor/store_residency.preload_capacity_check` |
| `explicit`, `vouched_auto` | the file plus a capped floor, plus the LoRA bake's stray transient where `lora_low_rss` and a stack are both set; a target the header estimate cannot map prices the whole file | `actor/capacity_quote._slab_rung` over `capacity_fit.slab_load_fit`, itself over `capacity_lora_bake.lora_bake_bytes`, taken again by `actor/store_slab_admit.admit`, then by the exact backstop `actor/store_slab_admit.admit_bake` once the real patch set is built |
| `consented_rescue` | the same, after both probes ran | the consent branch of `actor/capacity_quote.price`, over the same `capacity_fit.slab_load_fit`, taken again by `actor/store_slab_admit.admit` |
| `explicit_stock`, `stock_fits` | the file plus the host copy it is placed from (2.1 times the file in all), plus the floor; the store places every weight at load time, so this is the whole peak | `capacity_fit.stock_load_fit` over `stock_required_bytes`, re-taken by `actor/store_residency.preload_capacity_check` |
| an FSDP launch | bf16/fp16: this rank's shards plus one block; fp8/int8 or an unreadable kind: the conservative full-file bound; every case includes the floor | `actor/capacity_quote_fsdp.fsdp_rung`, `adapters/fsdp.fsdp_load_capacity_check`, and `actor/store_residency.preload_capacity_check`, all over `capacity_fit.fsdp_required_bytes` |
| the gate ceremony's clean reload | the same kind-aware FSDP figure | `fsdp_reload_price.price_fsdp_clean_reload` |
| the driver host's own stack | the transient and settled windows above, one rank's share where a shard build is declared | `nodes/loader_preflight.preflight_loader_footprint` and `nodes/render_preflight`, both over `driver_footprint.estimate_driver_footprint`, which prices a declared shard build through `driver_footprint.fsdp_shard_build_bytes` |
| every rank at once | the leanest rank's row, before any rank allocates | `capacity_agreement.collect` and `capacity_agreement.decide` |

CPU tests require `actor/capacity_quote.UNPRICED_RUNGS` to stay empty. They also compare this table's internal residency names and function names with the source, rejecting missing rows and stale function references. ComfyUI's sample-time memory accounting is separate from these residency choices. A slab or an FSDP shard runs ComfyUI's full load once its weights are
resident (`actor/resident_ledger.declare`), and a
stock load does the same (the stock row above), so the sample-time load does
not ask again for memory the box already holds. A stack still pending under
the baked hot-swap residency keeps ComfyUI's sample-time load, and the
`partial_load_divergence` guard still refuses a render whose ranks load
partially.

**Slab capacity refusals offer no consent card.** Slab is the final residency
option; consent cannot resolve its memory shortfall.

**Every rank prices before any rank allocates.** Before eager loading and again
before collective sampling, the driver asks every rank for a memory estimate
through a read-only endpoint outside the GPU lock. The lowest-headroom rank
determines whether the fleet may load.

Each request has a deadline. This avoids leaving ranks waiting inside an
all-reduce, allows the refusal to identify the affected rank, and handles a
future that never returns. Silence beyond the 30 s fleet budget, missing or
duplicate rank responses, and inconsistent checkpoint byte counts all refuse.
These are protocol or liveness failures, not capacity verdicts.

If at least one rank needs rescue and none refuses, the driver offers the
stock-load rescue before any allocation. Ranks must agree on the descriptor's
identity and other rank-invariant fields; their memory measurements may differ.
The offer uses the ordinary consent-registration path. World 1 needs no
agreement.

**The agreement does not correct a price.** It enforces the lowest-headroom
rank's estimate before allocation; an underestimated load can still exhaust
memory. The partial-load guard, post-load readiness exchange
([TROUBLESHOOTING #94](TROUBLESHOOTING.md#94-a-multi-rank-render-refuses-saying-a-peer-rank-could-not-load)),
and byte-verify certificate still run, and a capacity-certified row grants no
authority. The estimate also excludes shared memory: a slab in an untouched
slot or retained after a failed load can make reported availability too high.
The quote records these limitations.

### 5.10 Upstream forward-contract breaks

ComfyUI is tracked at nightly (§6.3), and adapters depend on its forward
contracts. Import canaries detect moved names and signatures. Behavioral
canaries detect semantic changes behind an unchanged signature. Each incident
keeps its detail in its own record; operator diagnoses are in
[TROUBLESHOOTING.md #36](TROUBLESHOOTING.md#36-packed-audiovideo-sampling-fails-after-denoising) and [#64](TROUBLESHOOTING.md#64-a-packed-audiovideo-custom-render-dies-on-the-last-step).

**Triage ladder.** Follow these steps in order.

1. **Which side moved?** Run the behavioral suite against the previous comfy
   commit. A contract test that already failed indicates a dgx-monarch bug; one
   that starts failing after the ComfyUI update indicates an upstream change.
   Establish that difference before filing against ComfyUI.
2. **Typed refuse, or adapt?** Adapt where stock's new behavior is expressible
   without re-deriving family math. Refuse, typed and named, where it is not:
   use §5.9 to select the refusal class instead of allowing an unclassified
   mid-render crash.
3. **Canary first.** The transaction that proves the break goes in before the
   fix, and must fail on the unfixed tree.

**The fix-extends-canary rule.** A fix for an upstream contract break does not
merge until it extends `tests/canary/comfy_seam_contracts.py` with a transaction
that fails against the broken behavior and passes against the fixed one. The
rule prevents the same break from becoming silent again.
`tests/test_comfy_seam_contracts.py` holds the suite's shape in both directions,
so a table row without a transaction, an unreferenced transaction, or a
transaction that stops asserting fails the unit tier without Comfy installed.

**The inventory.** `SEAMS` in `tests/canary/comfy_seam_contracts.py` lists the
behavioral contracts. `tests/test_comfy_seam_contracts.py` checks that its
module docstring lists the same contracts in run order. Consult that canary
file for the current inventory. The behavioral tier runs on the comfy-canary
schedule against comfy master; the signature tier is
`comfy_surface.TOUCHPOINTS`, aggregated and re-exported by
`comfy_forward_contracts.py`. Exact rebound contracts live in
`comfy_rebound_signatures.py`, the static `self.bind`/`Adapter.bind` inventory
in `comfy_rebound_sites.py`, and the runtime comparator in
`comfy_rebound_validation.py`. Repository tests require the AST inventory and
declarations to be bijective and prove each replacement accepts the declared
stock call shapes. The canary imports every declared target and rejects added,
removed, reordered, kind-changed, or requiredness-changed parameters even when
a replacement's `**kwargs` could otherwise accept the changed signature.

Seven seams are held outside the behavioral suite:

| seam | where the pack rides it | held by |
|---|---|---|
| LoRA bake | `actor/model_store` per-key bake | `bake_equivalence_canary.py` |
| legacy quant conversion | `actor/unbake` | `slab_hook_canary.py` |
| node-pack entrypoint, routes | `__init__.py`, PromptServer registration | `comfy_entrypoint_canary.py` |
| object-patch model sampling | `nodes/model_sampling.py` | `comfy_entrypoint_canary.py` |
| packed transaction and message | gate identity, latent return | `comfy_entrypoint_canary.py` |
| family forward overrides | every adapter `bind` | exact rebound signature inventory and per-family tests |
| node widget schemas | committed templates, sweep converter | `template_widget_canary.py` |

---

## 6. Operations and installation

### 6.1 Install paths

1. **Source install:** follow [INSTALL.md](INSTALL.md) to place the checkout in
   ComfyUI's custom-node path and install its Python package. A first graph run
   with a local-only mesh (single host, N local GPUs) needs no cluster config or
   persistent Worker service.
2. **Multi-node:** `dgxm` CLI (installed with the pack, also `python -m dgx_monarch`):
   - `dgxm setup`: explicit-host, plan-first bring-up. The parser, bounded
     probes, pure compile-time profiles, config transaction, owned source/unit
     installation, verification, and receipts are separate leaves. Operational
     behavior is canonical in [INSTALL.md](INSTALL.md#guided-multi-node-setup);
     profile mappings are canonical in
     [CLUSTER.md](CLUSTER.md#guided-setup-profiles).
   - `dgxm doctor`: preflight with a check that the hostname does not resolve
     to loopback (cleared by a configured `client_bind`), fabric IP reachability,
     NCCL env sanity, torch/monarch version match across hosts, UMA/page-cache
     headroom warnings (drop_caches guidance), rail health. `--repair` is
     limited to a confirmed, inode/content-rechecked `chmod 0600` of the exact
     config; network/SSH/package/service/process/mesh/power remedies remain manual.
   - `dgxm init`: legacy interactive `cluster.toml` writer; config publication
     shares the destination lock and exact transaction used by the newer paths.
   - `dgxm up / down / status / restart`: Worker service lifecycle (idempotent;
     systemd-backed when its user unit is installed, with a nohup fallback).
     `up` and `down` also sweep worker actor processes that outlived their
     client, by exact procfs identity, never by pattern. Mutating lifecycle
     legs cooperate on the target user's per-host lock; its exact scope and
     limitations are canonical in [INSTALL.md](INSTALL.md#guided-multi-node-setup).
   - `dgxm reap`: the same sweep on demand, including on a local-mode box
     with no `cluster.toml`; `--dry-run` prints the plan and signals nothing.
   - `dgxm update`: compatibility git-pull path that synchronizes the
     repository's exact torchmonarch pin with `--no-deps`; install/sync dgx-monarch with
     `--no-deps`; restart and doctor. It never resolves torch/NCCL/xfuser.
     `--verify` selects the certainty-aware remote-cluster transaction split
     across inspection, remote release-slot, staged driver-pin, smoke, receipt,
     and state-machine leaves; its operational contract is canonical in
     [INSTALL.md](INSTALL.md#updating).
   - `dgxm top`: live TUI; `dgxm gate`: identity-gate the last prompt;
     `dgxm install-service`: refuses a unit it does not own, syncs the package,
     then installs and starts each host's Worker service as a systemd user
     unit, and fails a host whose user lingering is off.
   - `dgxm uninstall`: under the shared config lock, stops Worker services,
     removes units, and offers a checked unlink of the reread config path.
     The ComfyUI node-pack checkout, Python environments, and models stay in place.
   - Operator lifecycle/readiness APIs share **Worker service**, **Attached
     mesh**, and **Render session** vocabulary. Unavailable live telemetry reports unknown;
     doctor's adapter preserves its legacy aggregate and exit semantics while
     projecting missing mesh evidence as unknown. The canonical meanings are in
     [CONCEPTS.md](CONCEPTS.md#operator-words).
   - Setup, safe repair, and verified update share one bounded receipt adapter;
     publication and disclosure behavior is canonical in
     [INSTALL.md](INSTALL.md#operator-receipts).
3. **cluster.toml**: single config file for hosts, bind addresses, transport,
   NCCL env as named **fabric profiles** (`[fabric.dgx-spark-pair]` ships
   pre-tuned: SOCKET_IFNAME, IB_HCA both rails, GID_INDEX 3, GDR_LEVEL SYS,
   BUFFSIZE; `[fabric.generic-roce]`, `[fabric.generic-ib]`, `[fabric.single-node]`
   templates included).

Readiness, profile compilation, guided setup, repair, receipts, verified update,
and cluster smoke have CPU unit tests and two-Spark hardware checks.
Fresh installation, setup reruns, repair, and stopped-Worker readiness results
are recorded in [Validation](VALIDATION.md#operator-checks), with their scope
and limits. These checks do not establish model fidelity or fabric performance.
Verified update runs from a frozen controller snapshot and journals
migration of recognized units and source layouts to immutable releases on
both Sparks. It retains prior installations for guarded compensation.
[Validation](VALIDATION.md#operator-checks) records the normal CLI update
results and rollback scope; [INSTALL](INSTALL.md#updating) gives requirements.
Native latent RDMA remains under the default-off
HOLD and requalification gate in §5.6.

### 6.2 Operator and contributor skills (`skills/`)

- **`dgx-monarch`** (user skill): install, update, uninstall, cluster bring-up,
  doctor-driven troubleshooting, and topology selection.
- **`dgx-monarch-dev`** (contributor skill): adapter-authoring guide (how to
  bring a new model family up, with the cross-rank consistency and fidelity
  acceptance gates), release checklist, benchmark-harness usage, the licensing
  rules from §3 as a hard checklist.

Skills call `dgxm` rather than reimplementing operations.

### 6.3 Versioning and compatibility policy

- semver; `dgxm doctor` enforces matching Monarch and Torch versions across
  hosts.
- torchmonarch is pinned and bumped with a changelog entry per bump; the
  developer skill documents the introspection-first upgrade process.
- xFuser is pinned to the local `0.7.0+dgxm.npuimport1` compatibility wheel
  until an official release replaces it (section 3.1, rule 3).
- ComfyUI: track nightly (project policy), with a CI canary job that runs the
  node-pack import and a CPU-only graph-build against comfy master daily.

---

## 7. Repository layout & maintainability

This is a selected map of the files that own a contract, not a full file
inventory. Add a file here when it takes ownership of a lifecycle,
compatibility, or release boundary that a maintainer must be able to find.

```
dgx-monarch/
├── LICENSE                    # Apache-2.0
├── NOTICE                     # copyright line
├── README.md                  # §8
├── pyproject.toml             # package metadata + pins, dgxm entrypoint
├── src/dgx_monarch/
│   ├── nodes/                 # thin nodes + setup binding
│   │   ├── action_security.py # same-origin checks for browser driver actions
│   │   ├── fleet_cleanup.py   # cancellation-safe Fleet resource retirement
│   │   ├── gate_ceremony.py   # one ceremony's render legs, frozen as evidence
│   │   ├── gate_process_state.py # process-lifetime first-use Gate state
│   │   ├── gate_provenance.py # mandatory all-rank Gate source bracket
│   │   ├── gate_verdict.py    # verdict derivation, lever quarantine, publication
│   │   ├── metrics_route.py   # bounded finite Prometheus projection
│   │   ├── pending.py         # terminal sample-result ownership and cleanup
│   │   ├── pipeline.py        # bounded FIFO submission/abort orchestration
│   │   ├── recycle_guard.py   # reset single-flight/rate ownership past HTTP timeout
│   │   ├── render_session.py  # node-side names for the mesh_session.py API
│   │   ├── sampler_wire.py    # sampler output merge and stock noise rebuild
│   │   ├── strict_json.py     # non-finite-safe public JSON normalization
│   │   └── setup_binding.py   # follow-up RPC topology/token reassertion
│   ├── actor/                 # GPUWorker, model store, failure decorators
│   │   ├── comfy_custom_nodes.py # headless custom-node discovery/cleanup
│   │   ├── pread_backend.py   # lazy safetensors backend shim
│   │   ├── capacity_quote.py  # one rank's price for one slot, as data: the
│   │   │                      #   ladder's arithmetic and the wire row it fills
│   │   ├── rescue_offer.py    # capacity rescue offer parts, shared by the
│   │   │                      #   ladder that raises one and the quote row
│   │   ├── rank_readiness.py  # the post-load readiness flag every whole-model
│   │   │                      #   topology crosses before the first collective
│   │   ├── sample_protocol.py # the GPUWorker sample body `_sample_impl` runs
│   │   ├── slab_wrap.py       # minimal DLPack view of slab-owned byte ranges
│   │   ├── resident_ledger.py # tells comfy what a slab or a shard already
│   │   │                      #   holds, so its sample-time load stops asking
│   │   ├── store_load.py      # fresh-load path of ensure: ladder, load, adoption
│   │   ├── store_load_ownership.py # exact fresh-load resource/slot transaction
│   │   ├── store_slab_admit.py # the slab capacity check, its card text, and
│   │   │                      #   the sentence printed when neither fits
│   │   ├── stored_model.py    # resident-slot resource identity and close order
│   │   ├── worker_capacity.py # the capacity_quote endpoint body: this rank's
│   │   │                      #   slots, one snapshot, one running budget
│   │   ├── worker_compile.py  # opt-in DiT block compilation boundary
│   │   └── rdma_handoff.py    # durable worker descriptor ownership + ACK
│   ├── adapters/              # §5.3 registry (one file per family)
│   │   └── flux_shared.py     # control-span/shard helpers the flux forwards
│   │                          #   share with hunyuan, lens, and qwen_image
│   ├── comfy_surface.py       # runtime Comfy touchpoint/signature assertion
│   ├── comfy_forward_contracts.py # aggregate/re-export forward compatibility map
│   ├── comfy_rebound_signatures.py # exact stock rebound-method contracts
│   ├── comfy_rebound_sites.py # declared static self.bind/Adapter.bind inventory
│   ├── comfy_rebound_validation.py # exact signature/requiredness comparator
│   ├── topology.py            # dataclasses + auto-mode table (data-driven)
│   ├── error_utils.py         # total diagnostics + cancellation-first composition
│   ├── retry_policy.py        # bounded retry outcomes without stale diagnostics
│   ├── actor_lifetime.py      # actor-side reaper (stdlib leaf; watcher installed
│   │                          #   at bootstrap, lease trigger armed by the client)
│   ├── client_lease.py        # driver-side lease renewer for live fleets
│   ├── mesh.py                # attach/spawn/shutdown, registry lock + callback
│   ├── mesh_capacity_consent.py # capacity-rescue consent constants + grant check
│   ├── mesh_creation.py       # creation publication, claims, and recovery
│   ├── mesh_lease.py          # setup-generation and raw-RDMA-read leases
│   ├── mesh_lease_retirement.py # lease retirement, finalization, owner handoff
│   ├── mesh_liveness.py       # out-of-process proof that a dead fleet's worker
│   │                          #   processes are gone, and the latch release
│   ├── mesh_recycle.py        # typed deliberate-stop outcomes and transaction
│   ├── mesh_session.py        # driver-side exclusion for logical render sessions
│   ├── mesh_setup.py          # setup generations, budgets, acks + dispatch tokens
│   ├── mesh_setup_state.py    # typed same-handle process-group transition state
│   ├── mesh_safety.py         # lifecycle classification + pure safety policy
│   ├── mesh_teardown.py       # stop authority and supervision reconciliation
│   ├── monarch_semantics.py   # stdlib-only torchmonarch behavior probes
│   ├── monarch_touchpoint_spec.py # touchpoint record + signature comparison
│   │                          #   (stdlib leaf; keeps the manifest importable)
│   ├── absorb_fire.py         # one greppable INFO line per teardown-absorption
│   │                          #   site (stdlib leaf; used by soak diagnostics)
│   ├── fault_repeats.py       # report an unhandled cluster fault once, then
│   │                          #   count its exact repeats (stdlib leaf)
│   ├── attach_trace.py        # one greppable line per cluster-attach phase:
│   │                          #   the vocabulary, the timer, and the parser
│   │                          #   the TROUBLESHOOTING #101 grep depends on
│   ├── capacity_agreement.py  # driver-mediated pre-load fan out: every rank
│   │                          #   prices before any rank allocates
│   ├── capacity_memory.py     # one /proc/meminfo parser and what it may claim
│   ├── consent_observe.py     # worker refusal descriptors as consent-panel cards
│   ├── driver_footprint.py    # driver-side footprint policy (no torch or
│   │                          #   monarch import; render-path pending-load refusal)
│   ├── fsdp_proof_scope.py    # FSDP clean-reload scope + the topology test
│   │                          #   that selects it, below either caller
│   ├── operator_actions.py    # inert, safety-classed readiness remediation catalog
│   ├── operator_profiles.py   # compile-time safe/balanced/advanced authoring policy
│   ├── operator_readiness.py  # canonical three-layer readiness normalizer
│   ├── telemetry_events.py    # bounded strict-JSON event-tail projection
│   ├── telemetry_fleet.py     # the last Fleet wave's per-job rows: rank, host,
│   │                          #   box, duration, prompt digest, driver-clock start
│   │                          #   and end, and a marker when another box's host answers
│   ├── transfer.py            # messaging + RDMA return paths
│   ├── transfer_utils.py      # pure tensor-tree/split/read helpers
│   ├── rdma_poison.py         # process-lifetime poison records + one-time report
│   ├── rdma_ownership.py      # process-lifetime handoff registry + tombstones
│   ├── rdma_job_registry.py   # bounded process roots for unresolved reads
│   ├── rdma_read_attempt.py   # durable per-contender operation state
│   ├── rdma_read_driver.py    # root/start/join/recovery envelope
│   ├── rdma_read_job.py       # exact-once scratch-read state machine
│   ├── rdma_read_token.py     # read-token publication, proof, and retirement
│   ├── rdma_receiver.py       # fail-closed read/drop/ACK transaction
│   ├── rdma_settlement.py     # durable local-drop publication helpers
│   ├── tui/readiness_view.py  # pure canonical readiness renderer for the TUI
│   ├── tui/view_helpers.py    # dashboard rendering/recording helpers
│   └── cli/                   # dgxm command adapters and lifecycle leaves
│       ├── arguments.py        # public parser; command effects stay in leaves
│       ├── cluster_smoke.py    # owned DP/NCCL/source-cohort smoke + typed cleanup
│       ├── config_removal.py   # checked lock-bound durable config unlink
│       ├── doctor_nccl.py      # loaded NCCL library probe and the cross-host row
│       ├── doctor_permissions.py # config-mode diagnostic row
│       ├── doctor_repair.py    # exact-file chmod-only repair transaction
│       ├── legacy_init.py      # compatibility interactive config writer
│       ├── legacy_update.py    # compatibility pull/sync/restart update
│       ├── lifecycle.py        # Worker service lifecycle and package/pin effects
│       ├── lifecycle_generation.py # Worker generation marker/fence capabilities
│       ├── lifecycle_host.py   # host certainty, locality, and listener proofs
│       ├── lifecycle_lock.py   # shared per-user service-mutation serialization
│       ├── lifecycle_scripts.py # lock-held Worker start/stop payload builders
│       ├── lifecycle_systemd.py # canonical legacy unit + ownership proof
│       ├── listener_generation.py # which process holds a loop's LISTEN socket,
│       │                       #   how long it has held it, and the doctor row
│       ├── operator_receipt.py # bounded disclosure-safe receipt schema/writer
│       ├── receipt_sanitize.py # conservative receipt-note redaction leaf
│       ├── setup_apply.py      # confirmed lock-held setup mutation/settlement
│       ├── setup_cli.py        # strict public setup parsing/presentation boundary
│       ├── setup_command.py    # plan/confirmation/config-lock coordinator
│       ├── setup_config_fd.py  # cancellation-safe bounded fd primitives
│       ├── setup_config_io.py  # strict rendering/snapshot/atomic config I/O
│       ├── setup_config_lock.py # shared per-destination config-mutation lock
│       ├── setup_config_parent.py # durable no-follow parent creation
│       ├── setup_config_transaction.py # prepublished config recovery authority
│       ├── setup_config_types.py # config snapshot/mutation records
│       ├── setup_models.py     # immutable setup requests, plans, and outcomes
│       ├── setup_probe.py      # explicit-host bounded discovery and comparison
│       ├── setup_probe_script.py # bounded remote probe script payload
│       ├── setup_receipt_timing.py # bounded receipt timing/outcome helpers
│       ├── setup_recheck.py    # confirmation-bound config/host identity recheck
│       ├── process_inspector.py # temporary root-authenticated marker-only inventory
│       ├── setup_process_inspection.py # source/UID/boot/freshness-checked inspection client
│       ├── setup_services.py   # certainty-aware owned service/source transaction
│       ├── setup_services_compensation_script.py # intent-latched owned cleanup
│       ├── setup_services_generation.py # embedded generation marker/fence helpers
│       ├── setup_services_platform.py # concrete SSH/systemd transaction adapter
│       ├── setup_services_activation_scripts.py # exact systemd effect scripts
│       ├── setup_services_manifest_script.py # trusted remote manifest/JSON reader
│       ├── setup_services_scripts.py # bounded remote ownership/effect scripts
│       ├── setup_services_rsync_lock.py # token/inode-bound staging lock
│       ├── setup_services_storage_scripts.py # private release reservation/cleanup
│       ├── setup_source.py     # imported source identity/manifest binding
│       ├── setup_verification.py # doctor-first cluster-smoke evidence adapter
│       ├── update_cancellation.py # cancellation-precedence helper
│       ├── update_command.py   # concrete verified-update platform adapter
│       ├── update_driver_pin.py # driver-pin change: stage both wheels, switch or restore offline
│       ├── update_entrypoint.py # serialized execution and receipt publication
│       ├── update_inspect.py   # exact target dependency/source inventory checks
│       ├── update_lock.py      # per-checkout flock: one driver-side update at a time
│       ├── update_receipt.py   # bounded update step-to-receipt adapter
│       ├── update_release.py   # private remote release-slot switch/compensation
│       ├── update_release_finalize.py # all-host verified finalization scripts
│       ├── update_release_switch.py # journaled remote live-site switch scripts
│       ├── update_release_types.py # slot value types, runner seams, metadata checks
│       ├── update_stage_cleanup.py # total pre-activation cleanup settlement
│       ├── update_stage_failure.py # cleanup-aware staging failures
│       ├── update_support.py   # private staging dir and process-enumeration guards
│       ├── update_transaction.py # certainty-aware verified-update state machine
│       ├── update_types.py     # certainty-bearing update operation contract
│       ├── update_worker_lifecycle.py # generation-bound Worker update settlement
│       ├── update_worktree.py  # owned detached-worktree cleanup
│       └── update_worker_attestation.py # exact remote source/pin-payload proof
├── web/js/dgx_monarch_readiness_model.js # pure fail-closed browser normalizer
├── tests/canary/              # Comfy and torchmonarch surface canaries
├── tools/check_artifacts.py   # offline manifest check, and the only writer of
│                              #   example_workflows/artifacts.toml
├── example_workflows/artifacts.toml # the model files each template needs
│                              #   staged: models subfolder, source, size, sha256
├── benchmark/                 # matrix and sweep harnesses + quality gates
├── docs/                      # DESIGN.md (this), INSTALL, CLUSTER, MODELS,
│                              #   TROUBLESHOOTING, VALIDATION, ADAPTERS
├── skills/                    # §6.2
└── .github/workflows/         # lint+unit (x86 CPU), daily comfy-master and
                               #   weekly torchmonarch canaries
```

Maintainability rules: type public and internal boundaries; run ruff, mypy and
pytest in CI; keep each module at 500 lines or fewer unless the ledger below
lists it; keep per-model conditionals in `adapters/`; represent the auto-mode
table as data; cite dated evidence for empirical constants; record release
changes in CHANGELOG.md.

Keep concurrency order, failure direction, lifetime, math, and empirical
provenance close to the code. Record release history in CHANGELOG.md and public results in docs/VALIDATION.md.
Keep private operational logs outside the repository.

Single-home rule: each fact has one canonical home, and every other surface
links to it instead of restating it. Surfaces an operator acts on with no docs
open may keep a one-sentence summary beside the link: the `dgxm doctor` rows,
the `dgxm init` prompt, and rendered `cluster.toml` comments. Version-anchored
prose belongs in dated records (CHANGELOG.md, docs/VALIDATION.md, the ledgers
below); evergreen prose states version-neutral facts.
`tests/test_doctrine_hygiene.py` enforces the version rule on every docs/ page
except VALIDATION.md, on README, SECURITY, CONTRIBUTING, cluster.example.toml
and the skills, and on src and benchmark code. It also enforces the single
home of five facts by one fragment each: fabric trust in SECURITY.md, RDMA
latent-return posture and its HOLD in §5.6 F3, attach timeout and wedge
recovery in [docs/TROUBLESHOOTING.md #1](TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout) and [#2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach), and the attached-mesh self-heal
in [docs/TROUBLESHOOTING.md #87](TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

Line-rule exception ledger: `CEILINGS` in `tests/line_limit_helpers.py` lists
every module above the 500-line default cap with its ceiling.
`test_module_line_limit_exception_ledger`
(`tests/test_surface_remediation.py`) fails when the set of modules over 500
lines differs from that list or a module passes its ceiling. A ceiling
includes no allowance for future additions. Changing the set or a ceiling
requires a reviewed ledger edit; new behavior should prefer an extracted
module. An exception is not permission to fold a focused lifecycle, policy,
validation, or compatibility module back into the listed file. The table below
mirrors the ledger by hand; no test reads it, and its reasons are context, not
an enforced requirement. The current exceptions are:

| module | ceiling | cohesive reason |
|---|---:|---|
| `mesh.py` | 898 | driver mesh lifecycle, actor dispatch, teardown, and recovery orchestration; deliberate-recycle outcomes live in `mesh_recycle.py`, creation publication in `mesh_creation.py`, behavior probes in `monarch_semantics.py`, and diagnostic composition in `error_utils.py` |
| `cli/doctor.py` | 774 | user-facing preflight plus the hardware failure catalogue |
| `cli/actor_reaper.py` | 603 | procfs actor identity, orphan provenance, the termination sequence, the doctor report, and the read-only liveness answer the driver asks after a stop times out |
| `adoption_evidence.py` | 732 | opt-in, request-bound resident-adoption authority, full-file identity/cache, and strict worker/result validation; the render path retains only narrow integration hooks |
| `actor/unbake.py` | 692 | byte-verified checkpoint capture/restore engine |
| `config.py` | 658 | typed cluster schema, fabric profiles, validation, and TOML rendering; strict table-key vocabulary lives in `config_schema.py` |
| `actor/sampling.py` | 620 | worker sampler protocol and distributed execution paths; per-adapter data-parallel exemptions keep raw token/weight sequences intact across ranks |
| `actor/comfy_bridge.py` | 543 | Comfy worker bootstrap and narrowly-scoped runtime patching; DynamicVRAM state, latches, and policy live in `actor/comfy_dynamic.py`, and the pread shim lives in `pread_backend.py` |
| `actor/capacity_quote.py` | 539 | residency memory estimates and the response fields parsed by the driver, including slab LoRA bake memory and default LoRA-path selection |
| `capacity_agreement.py` | 567 | the driver's collection, its one pure decision function and the fleet card it composes |
| `actor/worker.py` | 509 | ordered checks and adapter installation for each freshly loaded base in `_inject_for_topology` |
| `gate_ledger.py` | 527 | the ledger's protocol number and its burn history live together, so the record of why a version was retired cannot drift from the number itself |
| `comfy_surface.py` | 526 | one executable inventory of every stock ComfyUI touchpoint, deliberately flat so a failing canary identifies the changed API |
| `adapters/fsdp.py` | 589 | the FSDP launch contract in one place: admitted kinds and profiles, the live precision rescan, the wrap with ignored islands and quantized shards, the streaming shard build hand-off, and the shard-build price |
| `nodes/gate_identity.py` | 583 | canonical normal/Fleet capability construction, proof-scope binding, authorization, and typed denial |
| `nodes/auto_gate.py` | 506 | driver-side Gate coordination, exclusive ownership, process verdict publication, and abort handling |
| `nodes/loader_preflight.py` | 593 | the loader-site capacity answer in one place: the two windows, the residency ladder they are priced against and the slab rescue offer |
| `runtime_provenance.py` | 596 | stdlib-only exact Git/source inventory, loaded-origin validation, and runtime snapshot boundary |
| `adapters/flux_family.py` | 560 | the flux-family adapters and the sharded forwards they install |

Refusal-class ledger (enforced by `tests/test_refusal_classes.py`, doctrine in
section 5.9): a typed refusal declares its class through `refusal.refusal()`.
`tests/refusal_class_ledger.json` records every classified site and, per
module, the count of sites not yet classified. That grandfathered count has a
ceiling of **198**, and may only fall: a new or edited typed
refusal must declare its class rather than join the unclassified sites. The
ledger's `new_sites_awaiting_a_class` is a branch-local work list and must be
empty on every merged head. The walker reads `raise` statements, so it sees a
class only where the refusal is raised. A helper that builds its exception and
returns it for a caller to raise, as
`nodes/consent_rescue._uncertified_slab_load` does, declares a class the
ledger never records: the site is absent from the classified sites and from
the untagged counts alike, and only that site's own test holds the letter.
Prefer raising where the class is declared. Where a helper must build the
exception, pin its class in a test, because the ledger cannot see it and its
ceiling does not cover that site.

---

## 8. Documentation contract

- **README** gives the general reader a short description, headline measured
  speedups, a two-Spark quickstart, a "when not to use this"
  section (one small GPU, a model that already fits and meets its render time,
  quick low-resolution images where the orchestration cost rivals the gain), a
  pointer to the docs/MODELS.md support matrix, which is its only home,
  dependency credits, and the Apache-2.0 badge.
- **docs/INSTALL.md** covers source installation; **docs/CLUSTER.md** the
  cluster.toml reference and fabric profiles; **docs/MODELS.md** the support
  matrix, which marks what `auto` selects for each family, and one section per
  family with its validated scope and known limits;
  **docs/BENCHMARKS.md** the measured timings and peak memory;
  **docs/TROUBLESHOOTING.md** the symptom-first diagnosis catalogue;
  **docs/VALIDATION.md** the tested scope and limitations; **docs/ADAPTERS.md** the §5.3
  authoring guide.
- Public claims name the tested model, settings, dependency versions and scope.
  A documentation edit does not count as hardware validation. Retain dates and
  source identities in run records where they help reproduce a result, without
  making readers reconstruct the development timeline. `benchmark/run_matrix.py`
  records settings, hardware and artifact identities. `tools/campaign_stats.py`
  generates campaign pages from the retained sweep exports.

---

## 9. Release acceptance gates

The release targets two DGX Sparks. The records below support only their
named configurations. Other hardware has not been validated. The full benchmark
matrix remains incomplete, and published claims must stay within the measured subset.
Hardware checks run manually on an approved pair. Hosted CI does not access the
Sparks, and passing CPU checks does not establish hardware acceptance.

| Area | Status | Evidence and remaining work |
|---|---|---|
| Packaging, license and CPU checks | Recorded | Apache-2.0 source distribution, wheel/sdist installation checks, CPU suite and ComfyUI canary checks have passed. Required CI must also pass for the release commit. |
| Image rendering and parallel modes | Recorded for named configurations | [Validation](VALIDATION.md) records Krea2, Chroma, Ideogram4 and other image comparisons. Ideogram4 dual-model Ulysses matched DP2; Ring2 missed the fidelity limit. These results do not qualify every precision or topology. |
| Residency, LoRA reuse and FSDP | Recorded for named configurations | [Memory measurements](VALIDATION.md#memory-calibration) cover retained weights, LoRA swaps and capacity limits. Each model's supported scope remains in [MODELS.md](MODELS.md). |
| Video rendering | Recorded for named configurations | LTX, Wan, HunyuanVideo, H3 and other video results are recorded in [Validation](VALIDATION.md). Completion, cross-rank agreement and reference fidelity remain separate results. |
| Operator setup, repair and update | Passed for the recorded two-Spark scope | Setup, repair, stopped-Worker readiness and verified-update results are recorded in [operator checks](VALIDATION.md#operator-checks), with their limits and rollback evidence. |
| Default-off messaging return | Passed for the boundary check | Zero-step production cases passed below, at and above 8 MiB and after fresh attach, with intact primary/denoised tensors and clean ownership. Section 5.6 records the scope; no denoising or native RDMA qualification follows. |
| Native latent RDMA | **NOT RUN / HOLD** | Disabled by default. Requires the upstream release and selected-rail requalification described in section 5.6. Messaging or NCCL results do not qualify it. |
| Fresh installation on two Sparks | Passed within the documented scope | A fresh agent installed Python, ComfyUI and Monarch, saved two distributed renders, and checked a safe setup rerun. Existing OS, drivers and model files were retained; see [fresh installation](VALIDATION.md#fresh-agent-installation). |
| Full launch benchmark matrix | Incomplete | The campaign selected 2,632 of 7,192 cases; 35 selected cases lacked a usable reference and did not run. Published results cover the recorded subset. See [campaign limits](VALIDATION.md#campaign-summary). |

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Monarch 0.x API churn | hard pin; deliberate bumps by the developer skill's introspection-first procedure (surface canary snapshot, dispatch probe, hardware gate); a weekly canary against the latest stable torchmonarch |
| Third-party implementation copied into project code | §3 contributor rules, source/provenance prompts in the PR template, and dependency-boundary review |
| A dependency exception hides an unrelated failure | [Dependency-check guards and tests](TRUST.md#installation-dependency-checks); [the maintained exception procedure](TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform) |
| Per-family attention complexity | each adapter lands behind its own cross-rank identity and fidelity gates |
| Wan 2.2 MoE and capacity constraints | FSDP capacity mode and an independent family acceptance gate |
| ComfyUI nightly breakage | daily canary against Comfy master; bind to model classes; wrap stock nodes instead of copying them (§5.7) |
| Bare-metal Monarch behavior lacks upstream detail | runnable probes, symptom-first diagnoses, and focused upstream repros |
| Single-maintainer continuity | contributor skills, canonical docs, and a CPU test suite and canaries that run on hosted CI |
| GPL-3 runtime-host boundary | distribution boundary and dependency inventory in LICENSE-NOTES.md |

---

## 11. Release quality bar

Every milestone must satisfy this checklist:

1. **Failure reports are actionable.** Dispatch and shutdown use bounded
   timeouts and typed errors wherever possible.
2. **LoRA changes avoid base reloads.** Re-rendering a changed LoRA stack does
   not reload the base checkpoint.
3. **Recovery is scoped.** One dead actor does not discard the fleet's warm
   state.
4. **The default topology is benchmark-seeded.** Auto mode logs the row it took
   and why. A row seeded from a sibling family, a scope the table records as
   unmeasured, and the unknown-family fallback each say so in that reason.
5. **Bring-up is one command per stage**: install, `dgxm setup` (or `dgxm up`
   for a hand-written or `dgxm init` config), `dgxm doctor`. Every stage is
   idempotent, except that `dgxm setup` refuses a managed source or unit that
   already exists instead of adopting it (docs/INSTALL.md). `dgxm doctor` may
   check the static environment rows before the Worker services start; only
   the full preflight, once they are up, is authoritative.
6. **Every observed failure mode has a documented diagnosis.**
7. **A newcomer completes the documented first distributed render on two Sparks.**
   The setup skill covers discovery, configuration, validation and safe recovery.
8. **Licensing is explicit.** Project code is Apache-2.0 and dependency boundaries are
   documented.

---

## Appendix A: empirical design inputs

- cfg-parallel won at or below about 1 MP and USP above about 1.5 MP (Krea2
  and Chroma); the `topology.py` auto table rows that split
  by resolution switch at 1.2 MP.
- Ideogram4 head_dim 256, FP8 pair at 1024x1024: Ulysses2 matched
  DP2 at NRMS 0.000; Ring2 measured 0.102 against the 0.100 limit.
- Sage attention helps only fp8 with USP at high resolution (about 5-8% at
  1536) and loses on bf16 at 1024.
- FSDP is a capacity tool, not a speed tool; NCCL_PROTO=LL can deadlock FSDP,
  so never set it.
- GB10 has GPUDirect RDMA off (`cuMemGdrSupport 0`), so NCCL stages through
  host memory: about 17 GiB/s busbw for a 2-rank 256 MiB all-reduce inside
  Monarch actors.
- On unified memory the page cache competes with GPU allocations and the
  ComfyUI planner reads free memory, so doctor warns and suggests drop_caches.
- Monarch bring-up failures map to preflight checks and troubleshooting entries.
- `scripts/setup_env.sh` documents the torchaudio constraint on aarch64.

Measurement scope and limitations are recorded in [VALIDATION.md](VALIDATION.md).
