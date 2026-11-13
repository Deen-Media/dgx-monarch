# dgx-monarch: Design and Architecture

**Status:** current architecture contract for the two-Spark release. The
message-return boundary and reattach checks passed on October 8, before the
0.7.0 migration. Native return stays off by default; later selected-rail checks
have the limited scope recorded in [Validation](VALIDATION.md#native-rdma). Verified update now
supports a Worker
on the driver's host; [Validation](VALIDATION.md#operator-checks) records the
update results and limits. The benchmark
campaign covers a documented subset, not the full matrix. The package declares
version 1.4.0 and keeps the Monarch dependency that 1.2.0 moved to. That
migration has local CPU, native-load and [exact-source paired hardware evidence](VALIDATION.md#gpu-checks-for-the-070-migration).
Clean cancellation remains unproven despite successful explicit recovery;
native cancellation and fabric-failure recovery remain untested.
[Release history](../CHANGELOG.md) records what changed since 1.2.0 and how to
update from it; section 9 tracks acceptance.
The runtime also includes low-RSS LoRA residency, zero-copy slab residency,
the identity gate and ledger, `dgxm top`, Fleet sampling, an RDMA latent-return
path that is off by default with scoped qualification (§5.6), and the per-family
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
| Recovery granularity | a supervision failure retires the whole fleet, and the next Init or render brings up a fresh one once the old processes are confirmed gone (§5.1); an endpoint error keeps the live fleet |
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
   implementation and must not copy third-party implementation code. Two
   shipped exceptions exist. The first is the compiled Studio bundle under
   `src/dgx_monarch/studio/static/assets/`: `npm run build` produces it from
   the locked frontend packages, so it carries the MIT-licensed Svelte runtime
   and small Vite, clsx and esm-env helpers, whose notices live in
   `THIRD-PARTY-NOTICES.txt`. A lockfile change that moves one of those
   packages updates its notice in the same change. The second is the INT8 GEMM
   kernel in `src/dgx_monarch/kernels/int8_gemm_triton.py`, vendored under the
   MIT license from kan11's gb10-comfyui-tuning at commit `d68916b`. Section 4
   admits it as a kernel replacement, and its notice sits in the file header,
   `THIRD-PARTY-NOTICES.txt` and `NOTICE`.
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
| Monarch mesh and actor lifecycle | **retain** | typed supervision is validated, and a supervision failure retires the whole fleet for a fresh one (§5.1) |
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

**Kernel replacements.** DGX Monarch replaces a dependency's kernel inside its
own processes only when a dated record proves the replacement gives identical
output on the shapes, dtypes and releases it admits; every other call stays on
the stock kernel. A replacement ships on by default with an off switch. Where
its prerequisites are missing it steps aside for the stock kernel with one
warning and never refuses a render: §5.9's rule that rules gate accuracy, never
availability, holds for it too. The one replacement is the INT8 ConvRot GEMM
(`kernels/int8_gemm.py`, with the vendored kernel in
`kernels/int8_gemm_triton.py`). Its record is
[VALIDATION.md, INT8 GEMM kernel comparison](VALIDATION.md#int8-gemm-kernel-comparison-2026-10-28-and-29),
and it runs only on the comfy-kitchen releases that record covers.

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
- Worker identity is per start. Each Worker loop start serves
  `<proc-id>@tcp://<fabric-ip>:<port>` with a new Monarch proc id
  (`worker_proc_id.py`) and records the id in its actor ledger loop record
  before it listens. The driver dials only an id it read from that host's
  listener-generation probe for the current attempt, and keeps every
  bookkeeping key (the released check, the attach trace, errors) on the bare
  address. When the first reading of an attach lacks a loop's id, the attach
  refuses before anything dials, with its own type, `WorkerProcIdError`, and
  the session stays usable. The first dial to a loop whose id changed since
  this driver last dialed it is primed with one throwaway message. The driver
  still enables one process transport, once, so it re-attaches to a restarted
  loop only within Monarch's message delivery timeout; beyond it the recovery
  is a fresh driver. After every loop restarted, a sampler render that owns
  its session and keeps the last render's setup, or an explicit loader's
  eager load, attaches again within that render: the capacity price wait
  ends on the first delivery failure the fault hook records for a fleet host
  and the mesh it prices on (`delivery_faults.py`), and the render compares
  each loop's id with the one it dialed, retires the stale mesh through the
  deliberate teardown with a short stop bound and the liveness release below,
  attaches by the new ids and dispatches once more (`restart_reattach.py`).
  Some loops moved and others not, no loop moved, an unread id, or old
  processes not proven gone each refuse at once, class C, with the remedy.
  Pipelined renders, Gate legs, the Fleet node, a one-host fleet and a render
  whose setup changed meet the old actors as before and take the supervision
  path. Mechanism and evidence:
  [TROUBLESHOOTING.md #2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach).
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
- `unhandled_fault_hook` is installed when the driver first loads Monarch,
  after policy admission and before any transport call: faults surface as
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
  It runs inline on the actor's event loop and never waits on the GPU thread:
  it reads torch's device figures only while the GPU lock is free, and while a
  load or a sample holds the lock it answers with the figures it last read,
  marked `stale` with their age (`age_s`). A stale mark means a busy Worker,
  not a dead one.
- `provenance_baseline(setup_generation, artifact_manifest=None)`: Gate's
  setup-token-bound rank/world/package-source queue barrier for the exact READY
  generation. With a manifest it also returns strict saved pre-READY setup and
  fresh post-operation runtime snapshots, Comfy-resolved artifact metadata
  identities, and an event watermark, serialized behind prior GPU mutations.
- `clear_vram(level)`: soft/hard cleanup with typed failure reporting. Either
  level also drops a resident decode VAE (§5.6).
- `vae_decode(request)`: one unit of a Worker-assisted VAE decode (§5.6), on
  the GPU thread under the lock. It never raises: a unit the rank cannot
  decode exactly comes back declined, with its reason.

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
A collection timeout ends the wait, not the work. An endpoint Future survives
a timed-out `.get(timeout)` and can be observed again, which the stall guard
relies on when it waits in slices (`mesh_lease.collect_with_liveness`); a
`returns_future` body, such as `ProcMesh.stop`, is cancelled by the same
timeout. After the hard timeout the driver still requests cancellation and
abandons the unread result rather than offering a retry, because the sample may
still be running and may still own a result or RDMA descriptor of its setup
generation; a later read cannot settle that ownership, and only a reset can.
Abandon retires the active lease once any already-started
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
set, the complete effective worker policy, physical config identity, the
driver's torch and CUDA build, resolved topology and attention, and the
selected READY setup generation/key. The
physical config identity hashes only the `cluster.toml` settings that can
change math, residency or transport, leaving out `auto_heal`, `ssh_key`,
`monarch.startup_timeout_s`, `[monarch.diagnostics]` and the driver-only
`[worker_args]` keys (`config_schema.DRIVER_ONLY_KEYS`)
(`monarch_policy_io.trust_config_fingerprint`). Every capability context also
names the driver's torch and CUDA build (`gate_runtime.runtime_build_identity`),
because a PASS proven under one torch says nothing about another. Each rank
reports its own build in its setup reply and in each Gate provenance row, and
Gate authority needs every rank on the driver's build: the driver refuses a
PASS lookup, a ceremony or a grant on a pair whose reports differ or lack the
build, naming each box's torch and CUDA, and each Worker refuses a normal or
Fleet grant whose build is not its own (`worker_runtime.py`). The context still
names one build, so a pair that matches keeps its PASS rows. A dual-model
render is proven as one unit. Its ceremony renders the dual request itself,
runs a swap cycle on each slot that carries a LoRA stack, and writes its
verdict under a combination key and an artifact digest that name both slots (`gate_dual_identity.py`). A single-model
PASS therefore never grants a dual render, and a dual PASS never grants a render
of either model alone; the dual grant binds both artifact sets, and the Worker
recomputes both before it accepts one. At dispatch a dual render reads two FAIL
keys, the pair's own and its conditional model's single key, and either FAIL
turns its levers off: a FAIL on that model alone is the same model on the same
residency path (`nodes/render_quarantine.py`). Where that proof cannot run, a dual render
keeps complete stock residency (`nodes/gate_dual.py`). On a cfg-parallel
topology each rank holds one of the two models and cannot cycle the other. Under
FSDP the clean-reload proof reloads one slot, so automatic FSDP refuses a dual
request. Comfy-managed residency has no two-model proof. With `auto_gate=off` a
dual render also stays stock: it has no residency without its proof. Driver and
worker artifact preflight bind both model slots in every case. Driver
preflight, worker preflight, and post-adoption checks all re-derive the
applicable snapshot. Each normal request is explicitly
stamped `required`, `stock`, `operator_off`, or `gate_internal`, so an absent
or stale grant cannot inherit risky defaults.
Before an explicit or automatic ceremony can retest old authority, one fsynced
multi-context `RETESTING` ledger row revokes the normal, Fleet, and eligible
explicit-slab siblings. Final verdict rows supersede it only after the proof
finishes, so a crash or failed final write leaves authority denied instead of restoring
an old PASS. An ordinary residency ceremony first runs the render site's
capacity checks for the whole request; a request that cannot fit the path it
would prove writes no `RETESTING` row (see "Where the ceremony cannot help"
below). See [TRUST.md](TRUST.md) for the complete current protocol.

The ledger also holds `WAIVER` rows for granted,
used, or revoked consent and `CAPACITY_CERTIFIED` rows for a slab load whose
stock alternative provably could not fit. Both are audit-only rows in a
capability namespace no trust context can emit, with verdicts the
authorization lookup does not accept, so audit evidence cannot become gate
authority. A v8 PASS must re-prove once per combination because it predates
the pre-gate certificate requirement. The current version is
`GATE_PROTOCOL_VERSION` in `gate_ledger.py`, 14; every version change
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

**Concurrency.** Every endpoint is `async def` so that out-of-band calls
(status, sample cancellation and client-lease renewal) run beside a render. A
synchronous actor handles one message at a time, so it would queue them behind
the render, and a render longer than the actor reaper's lease grace would let
the reaper stop a healthy actor. Monarch also refuses to spawn an actor that
mixes sync and async endpoints, so no endpoint can opt out. The blocking render
runs on one GPU thread (`_on_gpu`, with contextvars copied so the RDMA return
binds to this actor's instance) under an `asyncio.Lock`, and the lock keeps
compute serial: no two renders, or a model mutation and a render, overlap on
one GPU, model, or topology. Under Monarch's queue dispatch, every
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
                       #    cfg_parallel, fsdp, sol_attention, stock_attention,
                       #    attention_inputs, index_cache)
  cudnn_ring_attention.py # native cuDNN output/LSE contract and instance binding
```

`adapters/__init__.py` is the registry of record. Its order matters: several
families subclass others and the first match wins, so the most specific
adapter comes first. [ADAPTERS.md](ADAPTERS.md) is the authoring guide;
[MODELS.md](MODELS.md) carries the per-family evidence.

The Init node's `family_adapter` override names a family where detection
reads a header wrong or not at all. It selects that family's policy and
adapter; it does not widen what the adapter takes. A forced family binds a
model exactly when its adapter's own `matches()` takes the model's nearest
ComfyUI class, the first class in the model's ancestry that
`comfy.model_base` defines. A derivative class that ComfyUI does not define
therefore binds through the vetted class it extends, and a subclass that
ComfyUI defines and the adapter refuses under `auto` refuses under the
override too, class P
([TROUBLESHOOTING #154](TROUBLESHOOTING.md#154-a-forced-family_adapter-refuses-a-model-its-adapter-does-not-take-under-auto)).

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
   than wrapping it, so repeated graph executions never stack. A family bind
   may also target a `comfy.model_base` method on the base model, as FLUX.3's
   `extra_conds` bind does; the same rebound inventory and bijection hold it
   as they hold the `comfy.ldm` targets.

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

A row-local op is not always row-count-invariant. Measured on 2026-10-16,
comfy-kitchen 0.2.37's fused AdaLN took one CUDA kernel for calls of 1024
kernel rows (batch times rows) or fewer and another above, and the two
rounded a few outputs one bf16 step apart. FLUX.3 normalizes each modulation span in a call of its own, so a
rank's piece of a span, normalized in a call sized to the piece, could take
the other kernel than stock's call for the whole span: on 2026-10-16 every
uly2 one-reference edit and every uly2 text to image at 704 square or below
read one-step NRMS 0.01 to 0.055 against one GPU for that reason alone. The
parity rule: where a sharded forward runs a row-count-sensitive stock op on a
rank's piece, it launches the op at stock's call size for those rows, filling
the call with zero rows and keeping the piece's rows, rather than copying a
kernel threshold that a later comfy-kitchen may move. The kept rows go on
with a fresh tensor's strides, as stock's output does, because outside
inference mode ComfyUI picks a quantized linear's route by its input's
strides (`adapters/int8_linear_layout.py`). FLUX.3 applies it to
every fused-norm call of its 33 sharded blocks
(`adapters/flux3_adaln_parity.py`), at the cost of each rank doing stock's
full normalization work. MOSS-Image applies it, through the same
`at_stock_rows`, to both `rms_adaln` calls of each of its 42 sharded blocks
(`adapters/moss_image_seams.py`), with the rank's window at the front of the
filled call; its head norm runs after the gather, on stock's rows. Its fused
q and k norm and RoPE runs unfilled, because on 2026-10-20 comfy-kitchen
0.2.39 returned stock's bits for that op on every rank's window. Flux 1,
Flux 2, Chroma and LongCat, through the Flux-family forward, and Qwen Image
2.1 apply it to every `adaln` call of their sharded blocks
(`adapters/adaln_parity_blocks.py`): on 2026-11-07 uly2 at 512 square read
one-step NRMS 0.006 to 0.062 against one GPU on Flux 1, Flux 2, Chroma and
Qwen Image 2.1, while their shipped 1024 square read bit-identical. LongCat,
and Radiance through Chroma's forward, take the route unmeasured. A block
whose stock forward stops reading the norm global runs unfilled, and the
Worker log names it. That path read bit-identical at the shipped sizes and
off one GPU at 512 square, as every render did before the route, so it does
not refuse as FLUX.3 does, whose unfilled path missed at shipped sizes. Wan
2.1 read bit-identical at every size measured, so it runs unfilled. LTX
applies it, through the same `at_stock_rows`, to every `rms_adaln` call of
its sharded blocks, the text cross attention's included, each filled to its
stream's stock rows, and to the per-head gate projection, a GEMM one column
a head wide (`adapters/ltx_parity.py`). On 2026-11-07 LTX 2.3 on uly2 read
one-step latent NRMS 0.012 to 0.014 against one GPU at every geometry tried,
the shipped one included, where every call holds far more than 1024 rows,
and an op probe found the cause: the gate projection rounded tens of
thousands of a rank's outputs differently from stock's call at every LTX
geometry, on cuBLAS 13.4.0.1 and 13.4.1.3 alike, and the fused AdaLN did so
at 1024 rows a rank only. The fill was exact with the rank's rows at the
front, as `at_stock_rows` places them, as it was with the rows at their
stock offset. `at_stock_rows` keeps the op's own output width, so the gate's
one column a head comes back as stock's does. Like FLUX.3's, LTX's unfilled
path missed at the shipped size, so a block that stops reading the routed
globals refuses a sharded render. HunyuanVideo 1.5 runs the same Flux blocks
in its own sharded forward, unfilled and unchecked
([issue #54](https://github.com/Deen-Media/dgx-monarch/issues/54)).

Ranks must also agree on each model call's batch. ComfyUI's
`_calc_cond_batch` runs cond and uncond as one call only when its estimate of
that call, times 1.5, is under the free memory the model's patcher reports, and
on unified memory each rank reports its own box (`actor/comfy_bridge.py`). A
short head could split a pair the sibling batched, and every sharded collective
of the call then differs in size and count: on 2026-11-04 a Wan 2.2
text-to-video proof render stalled for 600 s with rank 1's collectives carrying
twice rank 0's elements. So wherever a model call is sharded (Ulysses times
ring above 1, cfg above 1, or FSDP), a `CALC_COND_BATCH` wrapper takes the
minimum of every rank's figure over the default process group once per call and
lets ComfyUI's rule run against it (`actor/cond_batch_agreement.py`), installed
beside the partial-load guard on every base either slot loads. Every rank takes
the leanest rank's batch, for one 8-byte all-reduce per model call. A pure
data-parallel call runs no collective, so its ranks need not agree, and there
the exchange would only leave a rank's peer waiting until the stall budget when
it stops early; it is not installed there. The pinned figure lasts through the
whole call, so code in the forward that reads the patcher's free memory, such
as Qwen Image 2.1's prefix-cache placement, reads the leanest rank's figure
from the start of the call.

The estimate behind ComfyUI's load decision must count the rows the forward
holds. ComfyUI prices the whole latent and every reference, and with UltraTex
token masks its Flux forward drops the background tokens before the blocks, so
the price ran to 2.2 to 3.1 times the rows of the largest call and a rank with room offloaded
its weights. On a Flux, Flux 2 or LongCat base a `PREPARE_SAMPLING` wrapper
counts the kept tokens with ComfyUI's own mask rule and hands ComfyUI's
estimate those counts for the length of its own `prepare_sampling` call
(`actor/held_rows_price.py`), at every world size. Each cond is priced as its
own call and the largest decides, so a negative with no masks counts the whole
latent and no reference, never one cond's latent with another's references.
A masked call counts its kept references once a sample, as the forward holds
them; where that passes stock's own count, which takes them once a call,
stock's figure stands. A ComfyUI that renames the mask rule gets a warning at
install and at each such render, and the canary names the rule.
It never prices above stock, keeps stock's figure on any doubt, leaves
`_calc_cond_batch` its own figure, and gives a rank no discount for its share
of the sequence: the measured H3 renders hold a pair rank at about what one
GPU holds for the whole stream, and LTX's fit puts it at 0.62, so the whole
stream is the safe count. The partial-load guard reads ComfyUI's decision as
before.

Adapter acceptance requires identical cross-rank latent statistics and
fidelity against a single-GPU reference within the measured noise floor.

### 5.4 Topology + auto mode

`Topology` dataclass: `ulysses, ring, cfg, dp, fsdp, world` with validation
(divisibility, per-model constraints declared by adapters).

**Auto mode** uses a static decision table seeded from the dated two-Spark
measurements in [VALIDATION.md](VALIDATION.md) and keyed on
(model family, quant, megapixels, world size):

- families with explicit resolution-crossover rows use cfg-parallel in a
  measured band and USP elsewhere: below 1.2 MP for Chroma, LongCat, Boogu and
  Anima, and from 1.2 MP up to 1.64 MP for Chroma, 1.77 MP for Qwen-Image and
  2.36 MP for Krea 2. Per-model thresholds are stored as data; other
  families have explicit full-range ring or Ulysses rows. Mage Flow, Qwen
  Image 2.1, FLUX.3 Klein and MOSS-Image have no row, and Flux, Flux 2, Krea 2, Lens, Qwen-Image and Z-Image
  have no row below 1.2 MP; those cases take the generic uly2 fallback
  (row 99). Data parallelism is never a table row: `with_derived_dp` derives it
  from the ranks a row leaves spare, and a batch that cannot split across them
  folds them into sequence parallelism, or on a cfg row refuses (class K);
- a row may take cfg-parallel only where its exact family, quantization, and
  megapixel band have a dated run that measured it faster than the one-GPU elapsed time
  and no worse than the step-nrms floor. Since Campaign K (2026-10-14) a new
  row also needs cfg2 faster than uly2 and bit-identical to one GPU, full
  schedule and one step, at every measured size in its band; rows granted
  before then keep their records.
  `topology_cfg_evidence.CFG2_EVIDENCE` is keyed by row number and records the
  admitted quantizations. Missing, out-of-band, or inconclusive evidence does
  not preserve a legacy cfg choice: auto selects its Ulysses fallback and logs
  why. Tests hold every predicate, including the fallback, so a new cfg row
  cannot inherit another family's result;
- a cfg row is skipped at CFG 1.0 unless the sampler keeps the uncond pass
  there (`topology.sampler_keeps_uncond_at_cfg1`: the cfg++ samplers). The
  KSampler, KSampler Advanced and KSampler Pipeline nodes and the Identity
  Gate tag the latent with that answer, so every site that resolves a
  render's topology, the Gate capability context included, reads the same
  one. Sampler Custom takes a SAMPLER object with no name, so its latent
  stays untagged and keeps the skip;
- a cfg row's `world_min` is 2, so it also matches at world 4. A batch-1
  render there leaves a spare data-parallel rank, and auto refuses it
  (class K) rather than fold the rank into a cfg-plus-sequence composite.
  Every cfg row's evidence was measured at world 2;
- spare data-parallel ranks a batch cannot split fold into a larger Ulysses
  degree only where `topology.VALIDATED_ULYSSES_FOLDS` grants that family and
  degree, and into ring otherwise. The one grant, MiniMax H3 at world 4
  (uly4, issue #188), bends the measured-rows rule: it rests on what
  community four-Spark users reported, not on this project's two Sparks. One
  report showed a graph whose latent on uly4 was bit-identical to their
  pair's uly2; another showed three rows on uly4 within the 0.10 one-step
  limit of one GPU, a closeness check rather than an identical latent. That
  falls short of the rule in CONTRIBUTING.md for reports from larger clusters,
  which asks for two independent reports of a latent identical to a validated
  layout's, so the row stands as an exception the maintainer allowed for this
  case. p-h3-image takes the same fold: it reads as `minimax_h3` and runs
  H3's DiT attention, and the table-wide test names it. It replaced
  uly2+ring2, a fold no record shows rendering, where the adapter refuses
  about three prompts in four after setup and the load; uly4 runs the same
  Ulysses code and pad exclusion as uly2, and 56, the one head count H3
  shards, splits four ways. The adapter checks the live head count at
  injection and refuses in words a degree that does not divide it. Where the
  fold runs, the logged reason adds where it stands; on a pair the line
  speaks only of the pair.
  Where H3 or MiniMax-H3-Image still lands on ring (a typed ring preset, an
  odd world, worlds 6 and 8), the driver counts each model call's packed
  total before the first-use check, setup and the load, and refuses a padded
  one with the Worker's own card (`nodes/h3_ring_pad.py`). It leaves to the
  Workers' guard what it cannot count: MiniMax H3 Context Windows, an entry it
  cannot read, and an entry ComfyUI may skip at some steps or call on part of
  the latent (a step window, a default entry, an area, or a mask set to its
  bounds). The auto decision
  itself never reads the conditioning, because several sites resolve a
  render's topology and must agree. The same check refuses at once Wan
  SCAIL's ring layouts above a sequence-parallel world of two, which its
  adapter refuses whatever the sequence: at four Sparks `auto` folds SCAIL's
  uly2 row into one of them, and SCAIL gets no uly4 grant, because no report
  shows its uly4 latent identical to a validated layout's;
- ideogram4: auto takes Ulysses2, never cfg2; explicit Ring2 stays for
  diagnosis, and explicit cfg2 runs the dual-model split (MODELS.md);
- flux3 earns no cfg2 row. Measured by Campaign K's method on 2026-10-15 and
  2026-10-16 with the first Spark capped at 2100 MHz, cfg2 ran 1.88x to 1.98x
  faster than one GPU and 1.23x to 1.41x faster than uly2 from 1024 to 1536
  square on BF16 and NVFP4. It was bit-identical when the prompt encoded past
  one 80-row text block, where ComfyUI already makes two batch-1 calls. A
  prompt that fits one block has as many rows as the empty negative, so one
  GPU folds the pair into one batch-2 call, which cfg2's two batch-1 calls did
  not reproduce on FLUX.3 or on Chroma: the one-step latent read NRMS 0.009
  to 0.031 and no full render matched. That meets the floor but not
  Campaign K's bar; LongCat above 1.2 MP read the same way and got no row
  either. Auto keeps the uly2 fallback, and cfg2 stays an explicit, measured
  choice (VALIDATION.md);
- moss_image earns no cfg2 row. Its shipped CFG 3 graphs carry Skip Layer
  Guidance on layer 22, and cfg2 refuses class P while that skip is on
  (TROUBLESHOOTING #126), so Campaign K measured no cfg2 cell on them. With
  the skip off, measured on 2026-10-21 with the first Spark capped at
  2100 MHz, cfg2 split a prompt and a negative of different lengths one call
  per rank, bit-identical to one GPU, in 47.0 s against 93.5 s on one GPU
  and 62.4 s on uly2 at 1024 square. A pair of equal length folds into one
  batch-2 call on one GPU, and cfg2 read one-step NRMS 0.023 to 0.026 against
  it, with no full render matching, so it misses Campaign K's bar as FLUX.3
  did. Auto keeps the uly2 fallback (VALIDATION.md);
- a cfg record can name checkpoints it measured and does not grant.
  Kroma reads family krea2, but its shipped negative zeroes the positive, so one
  GPU folds the pair into one batch-2 call, and on 2026-10-16 cfg2 read one-step
  NRMS 0.009 to 0.011 against it at 1152 and 1536 square. Row 15's record
  excludes Kroma, which `topology_cfg_evidence.checkpoint_variant` recognizes
  by the merge record in its header, so a Kroma file keeps row 12; one saved
  without that record reads as Krea2 (VALIDATION.md);
- a cfg record can be scoped to prompt pairs whose text rows differ. One GPU
  runs such a pair as two batch-1 calls, which cfg2's per-cond dispatch
  reproduces, and folds a pair of equal rows into one batch-2 call, which
  cfg2's two batch-1 calls do not. Boogu row 79 was measured on its
  template's pair, whose rows differ; on 2026-11-07, with the first Spark
  capped, two pairs of equal rows read off one GPU (one-step NRMS 0.014 to
  0.025, no full schedule matching), so the record grants only a pair the
  sampler node tagged. Anima row 47 had the same scope until 2026-11-10, when
  the Workers began to prepare Anima's text as stock does: stock pads it to
  512 rows before the batching decision, so one GPU folds every Anima pair,
  and the row was withdrawn (TROUBLESHOOTING #151). Chroma rows 20 and 19
  take the same scope: their pair was the shipped one, whose rows differ, and
  a BF16 pair of
  equal rows read NRMS 0.034 on cfg2. The tag (`cfg_pair.text_rows_differ`)
  needs one plain conditioning entry a side whose text tensors differ in rows.
  It rides on the latent beside the cfg++ tag, so every site that resolves
  the render's topology, the Gate capability context included, reads the
  same one, and the wire drops it. An equal pair, a combined, masked or
  timed conditioning, a Sampler Custom graph whose guider has no negative,
  and the dual-model guider with a second model wired stay untagged and take
  each family's Ulysses row (Chroma below 1.2 MP, which has none, the uly2
  fallback). The rule holds because both families' text is a
  `CONDRegular` (Chroma's through Flux's `extra_conds`), which never folds
  rows that differ, and canary seam 50 (`pair-scoped text conds`) holds that
  premise against the installed ComfyUI; a family whose text is a
  `CONDCrossAttn` folds some unequal lengths too
  (`cfg_dispatch.cross_attn_pair_folds`), so its records would need that rule
  before they could take the scope (VALIDATION.md);
- a Flux, Flux 2 or LongCat render whose conditioning carries UltraTex view
  inputs (`view_grid`, `token_masks`, `reference_views`, ComfyUI 0.45.0)
  skips every cfg row: stock's forward reads the batch as each sample's
  views, and a cfg split cuts them apart or runs batch-1 calls no test has
  compared with one GPU's. The sampler, Gate and Sampler Custom nodes tag the
  latent beside the pair tag, so every resolving site reads the same answer,
  and the reason names the skipped row. Where data parallelism or ring would
  remain, on a mesh wider than two GPUs, auto refuses (class P): only one GPU,
  uly2 and uly2+fsdp have been checked equal to stock for these inputs (on CPU
  so far), and an
  explicit preset outside them refuses alike (`adapters/flux_views.py`,
  TROUBLESHOOTING #148);
- sage auto-on only for fp8 + USP at high res;
- FSDP only when the model does not fit resident (capacity mode), never for speed;
- every auto decision logs its row and reason, for example
  `auto: uly2; krea2 fp8 at 1.05 MP, world 2, table row 99 (...)`. Krea 2 lost
  its cfg2 row below 1.2 MP (crossover table row 10) and gained one for FP8
  and BF16 from 1.2 to 2.36 MP (row 15, Campaign K, 2026-10-14).

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
contiguous pread of the rank's rows. A key on a module the wrap replicates
(the outside-forward census below) bakes as a world-1 shard: every rank writes
the whole merged tensor, which is the resident bake, and any other unsharded
parameter still raises. A key ComfyUI renames as it loads the file is read
under the file's name: a FLUX file stores a q/k norm weight as
`<...>_norm.scale`, and the model and a LoRA call it `<...>_norm.weight`. The
file's name comes from the model config's own rename, run over the header's
names (`actor/load_renames.py`), and the rank's rows are byte-verified against
it like any other key. A config that also builds weights as it renames
(PixelDiT strips `net.` and splits a modulation weight in two) keeps the map
for the keys it only moves; a weight built from part of a tensor has no file
name. A name only proposes a tensor: where a file holds one under the key's
own name and another under the name ComfyUI loaded it from, each candidate of
the live dtype and shape is byte-checked and the first that matches is bound.
That check is the last arbiter and it samples three 4 KiB windows, so a wrong
binding would take a config that loads two file tensors onto one key, a load
order other than the header's, and two tensors equal in those windows; no
registered config does the first. A patched key the file holds under no name refuses
class P in `capture_record`, before any shard is written. With `lora_low_rss` off a sharded model
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
island that Comfy's load keeps live at fp32 from one it casts
to the core's compute dtype (a live/file mismatch), so `actor/fsdp_lora.py`'s
in-bake `_file_tensor` check, per LoRA-patched key, is the backstop for that
shape. That check and the quantized-key check in `capture_record`/`bake_key`
(a backstop for a checkpoint the header markers miss) raise the same class P
tag as the launch-time refusal, not a bare `UnbakeError`: both are
deterministic per checkpoint, so every rank raises the identical refusal,
which is what makes tagging them rank symmetric and safe. A key on an island
kept at fp32 passes the `_file_tensor` check and still raises a bare
`UnbakeError`: islands stay whole, but outside the census the bake reads as
replicated. Header and
loaded-model detection both classify any full-precision mix containing F16 as
`fp16`, while fp8/int8 storage markers take precedence; an fp16 core is
admitted uniform only. One family names F16 tensors as islands instead:
MiniMax-H3-Image keeps its 51 adaLN linears on the curve basis in F16 beside a
bf16 core, so its header reads `bf16` and the header proof counts those bytes
as island bytes (`adapters/minimax_h3_image_header.py`); the loaded-model side
of that rule belongs to the family's adapter. The header proof does not count
a uint8 attention pin (`<block>.comfy_attention.config`), which is no weight,
so a bf16 or fp16 file that pins reaches the shard build, and the rule is:
class K guards a pinned kernel that runs; where the bf16 or fp16 FSDP shard
build drops the pins, nothing approximate runs, no card asks, and the record
says dropped (FSDP shard build), on every `+fsdp` preset and for every family.
fp32 parameters inside a bf16 core are islands
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
first H3 FSDP render to reach a forward). Four supported families reach one:
Anima's `llm_adapter`, H3's `condition_proj` and `token_refiner`, LTXAV's
two caption projections and two embedding connectors, and FLUX.3's early
streams (`txt_in`, `time_in`, `emb_in`, `early_stream_modulations`,
`txt_mode_blocks` and `content_mode_blocks.image_cond`), which only the base
model reaches from there. Flux and other families build `txt_in` and `time_in`
and run them inside their forwards, so FLUX.3's dotted paths sit in a table
keyed by its comfy class rather than in the flat name list.
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
fraction. FLUX.3's early streams do not fit: they are about 2.8 GB of its
11.1 GB BF16 file, so a world-2 rank holds more than its half plus the flat
transient fraction before any block is in flight. FSDP is a function test for
that family, not a capacity need, and its price stays flat until a hardware
run records the per-rank peak.

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

**Proving a clean reload is a capacity question, not a liveness one.** The
proof unloads every rank before each load it makes: the ceremony drops the
user's shards before the baseline load, and the cycle drops the baseline
before the reload, so two shard sets never coexist. The in-place shard drop
returns that storage to the host; on 2026-10-13 both unloads of a 24.5 GiB
bf16 proof gave back the whole share. Before any proof side effect, and again
after the baseline render, the driver estimates the next load with the
worker's admission helper. It compares that estimate with each rank's
MemAvailable plus the shard pool the next unload drops (torch reserved, capped
at one share), and the leanest rank after that credit decides. An estimated
non-fit at the first price records INCONCLUSIVE without unloading or
rendering. The Worker prices the load again after the real unload, and its
guard stays decisive.

Recycling the fleet would free other residents too, and would violate that
no-side-effect refusal guarantee. Instead, `actor/fsdp_streaming.py` assigns
mmap-backed checkpoint tensors and shards each block before allocating the
next. A rank needs its shards plus one block rather than a full second copy.
`fsdp_required_bytes` adds the shared 5 GiB host floor for bf16/fp16 streaming.
FP8, INT8, and an unreadable checkpoint kind use the conservative 1.20
full-file bound plus that floor.

In the Flux2 capacity measurement, the head had 42 GiB free when
the 60 GiB BF16 reload was estimated (`capacity_fit.py`). With the driver's
33 GiB text encoder resident, a 60 GiB Flux2 render under
`auto_gate=first_use` returned INCONCLUSIVE at the first price on 2026-10-13:
39.1 GiB available against 39.8 GiB, with the user's 30.0 GiB share resident.
The credit reads that state as about 69 GiB; the Flux2 proof has not been
rerun on hardware with it (VALIDATION.md, FSDP calibration). The BF16 streaming and
FP8-mixed direct-wrap FSDP runs kept every sampled host above the floor, but
did not recalibrate the 1.20 bound. Additional runs exercised INT8
stored-byte FSDP for Krea2, Chroma, and Ideogram4 without calibrating that
bound or its floor (VALIDATION.md, memory calibration). If a load still cannot
fit after that credit, `auto_gate=off` permits execution without that proof,
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
  reuse. On unified memory a stock slot's drop is not a global unload: it
  leaves the other slot loaded (`resident_ledger.unload_dropped`).
- `base_key = (unet_name, options)`: a match with a different LoRA stack is a
  **hot-swap**. The store clones the pristine base patcher (its weights stay on
  the GPU) and applies the new stack with stock's cast-time math (below), so the
  result matches stock ComfyUI.
  Under `lora_low_rss` the swap instead restores the LoRA-covered weights from
  the checkpoint in place and bakes the new stack (lazy un-bake); a swap that
  fails recoverably drops the slot for a fresh load. A patched key is bound to
  its checkpoint tensor by name and verified by bytes; a key ComfyUI renames as
  it loads the file (a FLUX q/k norm weight, `<...>_norm.scale` in the file and
  `<...>_norm.weight` in the model) is found under the file's name, taken from
  the model config's own rename (`actor/load_renames.py`).
- Outside `lora_low_rss`, a hot-swap on a stock base that ComfyUI has placed on
  the GPU is a fresh load instead when ComfyUI holds no host copy of a weight
  the new stack patches (`store_bake.copies_off_the_gpu`). ComfyUI re-applies a
  stack by unpatching the model to the host: the weights its backup holds go
  back as they are, and every other weight is copied off the GPU first. From a
  bare base that is the whole model; on 2026-11-05 the BF16 Wan-Animate 2 swap
  spent 230 s there, against 13.1 s for a fresh load of base and LoRA and 4.0 s
  for its patch. The drop takes no host copy, the fresh load costs what the
  swap's price already charges (section 5.9), and the result is the stock
  path's own.
  A strength change, a reorder or a removal of a held LoRA keeps the in-place
  swap, because the backup holds every weight it patches, and so does a store
  whose other slot holds a model. That gate dates from the drop's global
  unload, which pushed that model off the GPU the same slow way; on unified
  memory a stock drop no longer touches the other slot, and the gate stays until the
  route beside a held slot is measured on hardware. The test counts the weights the new
  stack patches, while ComfyUI's cost is every placed weight its backup lacks,
  so it matches that cost only for a LoRA that patches nearly every weight, as
  the Wan distill LoRA does. A strength change on a narrow LoRA stays in place
  and still copies the rest of the model off the GPU, as before; a new LoRA
  that patches a few weights beyond a full-coverage one reloads, and the drop
  first moves the held backup back onto the GPU. Both are speed, not memory, and
  a test by bytes waits for a measured change from one LoRA to another.
- Quantized models (every detected kind except bf16, fp16 and fp32): a
  hot-swap outside `lora_low_rss` (the clone path) also unloads every model, so
  the patched GPU weights rebuild from the CPU state dict and the other slot
  reloads on its next use. This is the measured fp8/int8 conversion-spike
  workaround (`store_detect._is_quantized_kind`).
- A miss is a full load. Every transition logs its class (`reuse`, `hot-swap`
  or `load`), so the log line names the source of a slowdown.

**A LoRA bakes to stock's cast-time weight.** Stock ComfyUI's default loader,
DynamicVRAM, never merges a LoRA at load: it applies the LoRA when a layer
casts its weight for a forward, in the forward's compute dtype, requantizes a
quantized weight with stochastic rounding seeded by the module's name, and
keeps that one weight from the first forward on. Every bake on a Worker, the
sample-time load, the merge of `lora_low_rss`, both hot swaps and the FSDP
shard bake, computes that same weight once (`actor/lora_cast_time.py`).
ComfyUI's classic bake, the one stock runs only with DynamicVRAM off, works in
the LoRA compute dtype (fp16 on GB10) with fp32 intermediates and seeds from
the key: on a BF16 file that is another rounding, and on an INT8 ConvRot file
another draw of requantization noise larger than many LoRAs' own change
([TROUBLESHOOTING #145](TROUBLESHOOTING.md#145-a-lora-render-on-a-worker-differs-from-stock-comfyui)).
A key stock bakes at load itself (a module of 16 KiB or less, a LoRA that
resizes the weight, a parameter other than a weight or bias) keeps the classic
bake, and so does a key whose cast does not settle on one weight: a weight
stored in another dtype than the forward, or a quantized layer on a
full-precision matmul, where stock's first forward uses another weight than
the later ones. A module ComfyUI leaves off the device, in a partial load or a
partial unload under memory pressure, computes with the full load's weights
too. ComfyUI's classic patcher gives every such patched module a cast-time
patch that runs on the weight in the forward's dtype, dequantized, and that is
the full load's weight only for a plain weight above 16 KiB stored in the
forward's dtype. After each `load` and `partially_unload`, the patch on every
other key is swapped for one that runs the full load's own
`patch_weight_to_device` on the module's stored weight when the layer casts:
ComfyUI's classic bake for a module of 16 KiB or less (on GB10 the two recipes
round a BF16 norm weight with a diff apart on about 7 percent of its values)
and for a weight stored in another dtype than the forward, and stock's
requantized weight for a quantized Linear (ComfyUI's classic cast never
requantizes; an INT8 ConvRot layer's output lands about 0.01 NRMS from the
full load's). A quantized weight outside a Linear keeps ComfyUI's own patch.
On a format that also quantizes the layer's input (FP8, NVFP4, MXFP8) the
weight at cast is the full load's and the output is not, because ComfyUI's
classic forward quantizes the input only when the layer carries no cast-time
patch, and the bake on every cast costs time. So on one GPU the Worker does not
leave such a load partial where it can help it: after ComfyUI's
`prepare_sampling` load and before the first forward, `actor/lora_partial_load.py`
takes for that load the route `lora_low_rss` takes (a bake under a full load,
then ComfyUI's backup and the patches freed) and marks the slot baked, so a
later stack change reloads from disk. It takes the route where the load device
shares the host's memory, as a Spark's does, and the host has room above the
5 GiB floor for the backups of the patched weights still off the device, which
ComfyUI keeps while it bakes; the model then holds less memory than the
partial load did (under the mmap loader the host copies are file pages, so
the room must also cover every layer the load moves). The rest of the model
is loaded through the model's own patcher and not through ComfyUI's model
manager, which would free every other model the sampling loaded, a dual-model
guider's other checkpoint among them, and unpatch its LoRA; and the check runs
on every model `prepare_sampling` loaded. Nothing is cleared before the full
load has succeeded. A full load that fails is another matter: ComfyUI undoes a
load that raises and leaves the model unpatched and unloaded, so any failure
of the route refuses the render, class C, whatever the format, and after a
route that reports success the check reads the model itself (patched, whole,
counted as loaded, still held by ComfyUI) and refuses if it is not what a
full load leaves. Where
the route cannot be taken (no room, or a GPU with memory of its own, which was
loaded partly because the model does not fit it) the cast-time weights stand,
exact for every weight-only format, and a render with a patched
input-quantizing layer off the device is refused, class C, before its first
forward (`partial_load_divergence`, raised at one rank by
`lora_partial_load.check`), saying what was tried. `lora_low_rss` on, the Spark
default, bakes at load time and never reaches this. Comfy-managed residency
runs ComfyUI's own patcher and is not touched.

A whole load is weighed by the same check. ComfyUI's classic bake keeps the
host copy of every patched weight as its backup for the whole render (the
Worker held 17.7 GiB of anonymous memory beside a 15.5 GiB model on record,
most of it that backup), and its whole-load condition leaves only its reserve
for what the sampler takes beyond ComfyUI's own estimate, which grows with the
latent. So where a patched model is loaded whole on a device that shares the
host's memory, and the room above the 5 GiB floor is under 1.5 times ComfyUI's
full estimate for sampling this latent (`estimate_memory`'s first figure, the
doubled batch; 1.5 is the share of free memory from which ComfyUI itself runs
cond and uncond in one call),
`lora_partial_load.free_backup_for_sampling` takes the route's last step
alone before the first forward: the backup and the patches go, on every clone
between the store's patcher and the sampler's, no weight moves, the slot is
marked baked, and one line says so. With more room than that the backup stays
and a stack change swaps in place as before, and nothing is logged. Every
format holds that backup under the classic bake, the cast-time math included
(`actor/lora_cast_time.py` records each patched key's original weight as
stock's method does), so a silent step means room at or above the ask, not a
missing backup: on 2026-11-13 an LTX 2.3 scaled FP8 render with a 7.08 GiB
stack had about 5.5 GiB of room, against an ask that ComfyUI's factor for that
model (0.077) puts well under 1 GiB, and sampled with the box at 10.51 GiB.
It moves no price, and the
weights the sampler reads are the ones the classic bake left. What can differ
is ComfyUI's own choice, made from free memory at every step, to batch cond
with uncond: on a graph with two conds on one model (a cfg other than 1, one
checkpoint) and a large estimate, the memory given back can turn two calls a
step into one, as it would with the same memory free under `lora_low_rss` on,
and the float order of that render then differs from the same render with the
step off. No shipped template that reaches the step is such a graph (the three
below run at cfg 1.0), and no hardware cell reads one.
On hardware the free has run only as the route's last step: it gave back
15.3 GiB within 2 s after one load line into a fresh Worker, and 5.4 to 5.8 GiB
where the INT8 ConvRot Wan-Animate 2 file was loaded a second time in the same
process (2026-11-12). What stayed, 9.5 to 9.8 GiB, equals that file's tensors
under 32 MiB (9.78 GiB; its tensors of 32 MiB and more are 5.70). glibc raises
its threshold for giving a block its own mapping to the size of any mapped
block the process frees, up to 32 MiB; a block under it comes from an arena,
and the Worker's loading thread has one made of 64 MiB heaps, whose freed
pages stay with the process. A probe read it on 2026-11-13: after the free the
Worker held 9.95 GiB in 243 mappings of 50 to 55 MiB at 64 MiB-aligned
addresses; a soft and a hard clear returned none of it; and the next load of
the file went into it (a peak of 17.20 GiB, not 25). So the free is followed
by one `malloc_trim(0)` (`actor/host_trim.py`), which walks every arena: in
the one-GPU step, in the pair's act and in the route's tail, where the line
that reports the free gives MemAvailable before the free, after it and after
the trim; and after the store drops a slot, which the Worker's hard clear
does for every slot (`resident_ledger.flush_dropped`): a placement alone left
10.8 GiB in the arenas in the same probe, so without it a stack change's
fresh load after a freed render could be refused as that bare load was. One
shipped template, the memory-saver, drops a slot once a render by its design;
the drop's line times the trim, and if a proof block reads it over 0.3 s a
render there it moves to the store-wide drop alone. A load's placement and
a soft clear are not followed by a trim in this release. The trim's return
and its time have no hardware reading yet; neither has a whole load's own
free.

On a pair the ranks free together or not at all
(`actor/backup_agreement.py`). After the cross-rank guard has found every rank
whole, each rank adds five integers to the guard's own exchange: whether it is
short (it holds a backup and its room is under the same ask), whether it can
free, whether its switch is set, its margin with its rank in the low bits, and
whether it holds a slot this agreement freed before. The margin is its room
less 1.15 times ComfyUI's single-batch estimate plus 1 GiB (a rank added
6.4 GiB where the estimate was 6.2; one GPU added 20.0 at 241 frames where it
was 17.4), plus the shared memory the rank has yet to map. A Worker of a pair
maps about 5.5 GiB of shared memory, about 2.7 by its load's end and about 2.7
more in the first seconds of its first forward, after the vote, and keeps it
for the actor's life; ComfyUI's estimate does not count it. The need and the
ask that makes a rank short therefore add 5.6 GiB less what the Worker's own
RssShmem reads at the vote, never under zero: about 2.7 GiB on a fresh Worker
and about 0.1 GiB on a warm one. By its mappings (256 equal anonymous shared buffers
of about 21 MiB beside the RDMA devices) it is the transport's registered
buffers; no library names it, so that owner is read from the shape. One GPU
maps 0.05 GiB of it, so the term is the pair's alone (a rank on a CUDA device;
none off one). The refusal's figures name it. On a held box the kernel's
memory compaction after a free can slow the first forward for up to about a
minute, and its peer waits; the render is the same, and this release changes
no rule for it (TROUBLESHOOTING #12). Every decision is made from the
MIN that comes back, so no rank
decides alone. Any rank short and none unable: every rank frees, the roomy
one too, so the slots stay alike and the next stack change reloads on all of
them. A rank that cannot free (a patched model outside its store): every rank
keeps its backup and says what it left standing. A rank whose load device has
memory of its own (a GPU that does not share the host's) takes no part: its
sampler does not draw on the host, so it is never short, never the rank that
cannot free and never the tightest margin, it keeps its backup, and it sends
the same five integers and makes the same exchanges; a fleet of such ranks
never frees and never refuses here, and a mixed fleet goes by its ranks on
unified memory. A switch set on any rank
turns the rule off for the fleet. After a free the ranks make one more
exchange, of each rank's model as read and its margin as read: a model that
is not what a full load leaves, or a tightest margin under zero, refuses on
every rank before the first forward, class C under the guard
`pair_backup_room`, naming the rank and the figures. MIN carries agreement
and not identity, and the driver receives one rank's error for the fleet,
whichever arrives first, so where the fleet refuses the ranks make one gather
more of each rank's own facts (its model's fault in its own words, its room,
its need). Every rank's error then leads with the short or wrong rank's cause
and figures, the others' after, and carries that rank's mark, so the driver
names its box; that rank also logs its refusal at WARNING. Once the error
names that box, a healthy rank's text closes with that rank's own remedy, not
the words for a peer it cannot name. Each rank's facts cross in 1,024 bytes;
a cause too long for them once escaped keeps as many whole characters as fit
and ends with a mark that it was cut, so the line always parses and never
garbles, and that rank's journal has it whole. A render that
samples makes no such gather. The margin is also
checked where a short fleet kept its backups, and at every later render on
slots the agreement freed: a fleet that freed and was refused has no backup
left to be short of, and is refused again until the Sparks are cleared or
the room is back. The margin weighs the sampler's need alone, on every rank:
it carries NO term for the driver's decode, because the driver cannot price a
decode before sampling (the VAE is a later node). The decode beside the driver
is the decode relief's to cover, by the Workers' release of their idle device
pools (`vae_decode_relief`, in the section on the driver's decode). A render
that does not free makes the one collective it made before, with more
integers in it; both Workers must therefore run one build, and a fleet whose
Workers state two builds at setup, or of which one states none because it
predates the check, is refused before its first load or render
(`mesh_builds.py`; a Fleet's world-1 actors share no collective and are not
compared). The shared memory a rank maps for its process groups (5.5 GiB on
record) has no price term in this release. For a proof on real ranks,
`DGXM_FAULT_BACKUP_RANK` with `DGXM_ACCEPTANCE=1` makes the named rank vote
short and read its model as wrong after the free. `DGXM_DISABLE_BACKUP_FREE=1` in the
Worker's environment keeps the backup however little room is left, for
diagnosis and to read one build with and without the step; the Worker then
logs what the step would have weighed, and the worker loop's environment
filter lets the name through to the actors (`cli/worker_process_env.py`).
A Worker-side switch is set in the one file each actor reads as it starts,
`~/.config/dgx-monarch/worker.env` (`actor/worker_switches.py`): the unit a
verified update writes clears its environment, so a unit drop-in does not
reach the Worker there. The file admits six diagnostic names and no other,
must belong to the Worker's user alone, yields to a name already in the
environment, and is never silent: the actor logs every switch at WARNING as
it starts, the driver repeats it at each setup, and status lists it
([CLUSTER.md](CLUSTER.md#setting-a-worker-side-variable)).

Three shipped templates reach this as shipped: `dgx-monarch-wan-animate2` (the
INT8 ConvRot file), `dgx-monarch-wan-animate2-bfloat16-warm` and
`dgx-monarch-wan-animate2-memory-saver` (the BF16 file), each with its LoRA at
81 frames and with `lora_low_rss` and `slab_weights` off. The generator sets
them so on purpose, and the templates' own notes give the reason: the recipe's
reference comparison and its warm tests were run with ordinary weight and LoRA
residency, slab and low-RSS have no passing reference comparison for this
family, and with both off there is nothing for the first-use gate to prove, so
the first render skips it (`tools/gen_templates.py`; so since the first public
commit). That reason still stands with this step in. The step runs after
ComfyUI's own classic bake and moves no weight, so the weights are the ones
the reference comparison covered; what it gives up is the backup, which only a
later stack change in place would have read. The low-RSS bake at load time
stays unproven for this family, and the templates stay as they are.

On the BF16 file the same step matters at the shipped 81 frames. The stack
patches most of the model's weights, so the backup is most of a 30.5 GiB model
(the Worker held about 34 GiB of anonymous memory after that load). The BF16
price is the stock price, the line and the files (69.1 + 11.2 + 0.7 = 81.1 GiB
before the file is read), and it sits ABOVE the reading from which ComfyUI
would load the model whole (about 75 by its own condition; not read), so every
admitted BF16 load is a whole load and its tightest reading is the price
itself. One such load, on the second Spark of a pair, took 77.4 GiB, 1.35 more
than that price leaves above the floor (the recorded shortfall of section
5.9). On one GPU, with the backup freed before the first step, that shortfall
should go, and on a pair the ranks now free together (above). No
hardware reading of the BF16 templates at their edge exists yet, on one GPU or
on a pair, before or after.

The bake-equivalence canary compares the bake with comfy.ops' own
cast on INT8 ConvRot and BF16 layers, and the keys it bakes at cast time with
the keys stock's own `ModelPatcherDynamic.load` chooses on both sides of the
16 KiB cut-off, against ComfyUI master every day.

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
`{kind: "sd3", shift: 5.0}` or `{kind: "minimax_h3", shift_video: 1.0,
shift_audio: 1.0}`, each kind with an exact key set. The worker applies that spec to a clone of the
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
  native RDMA. The October 8 pre-migration check passed with six production zero-step custom
  sampling cases: three sizes around 8 MiB, repeated after fresh attach,
  exact primary/denoised tensors and confirmed owned recycle. No denoiser
  forward ran; this is messaging acceptance, not model fidelity
  ([recorded scope](VALIDATION.md#default-off-messaging-return)). The repository's
  `benchmark/rdma_return_bench.py` now exercises CPU staged returns through the
  production descriptor, ACK and lease machinery, in messaging or native mode.
  Its evidence is narrower than the custom-sampling gate above and establishes
  no render fidelity. The 0.7.0 `rdma_ibverbs_target` selector enabled the
  [recorded selected-rail qualification](VALIDATION.md#native-rdma), including
  fresh integrity, ownership, recycle and reattach checks. One-step at-threshold sampling passed; cancellation and fabric-failure
  recovery remain untested; native return stays off by default. [Validation](VALIDATION.md#native-rdma) retains the historical failed
  production-path check alongside the later qualification and scope limits.

  The earlier disconnected-rail failure remains historical evidence, not a
  qualification of the upgraded runtime. With native return on, current policy
  requires every participant, the driver and each configured host, to resolve
  to an explicit `nic:<device>` target whatever `max_nics_per_buffer` allows; the
  driver checks every host before any process starts, and each process validates
  native readback before transfers. NIC affinity pairs device names; it does not
  discover network reachability. Application chunks and native QPs are separate
  settings, both defaulting to one; the validated chunk count cannot be changed
  through an ambient environment override after admission. A rail selector is
  necessary but does not by itself turn the default back on:
  each newly selected path requires a passing selected-rail matrix before use. Dated measurements are in
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
  may still be live, and any failed native read latches later setups to messaging
  until the driver restarts ([TROUBLESHOOTING #115](TROUBLESHOOTING.md#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)). After a completed read, cleanup attempts each local drop
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
  retained within the same class. Production does not recycle automatically,
  unless the user has ticked Always clear and retry or set
  `DGXM_ALLOW_CLEAR_RETRY` and a model loaded earlier is in the way
  of a load (section 5.9, "A model in the way offers a clear"). Otherwise,
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
  ecosystem keeps working. Worker-side text encoding is a later
  headless-pipeline option, not the node path. Worker-assisted VAE decode,
  below, keeps the stock decode nodes on the driver too: only the units of
  their own decode loop travel.

**Worker-assisted VAE decode.** With `[worker_args] worker_vae_decode =
"auto"` (absent means off, the default), the stock VAE Decode and VAE Decode
(Tiled) nodes split a video decode over the Workers wherever the family
allows ([MODELS.md](MODELS.md#worker-assisted-vae-decode)). The Init node arms
it for the mesh it attaches (`vae_decode_route.py`). Six rules hold it:

- **The reference is ComfyUI's one-host tiled decode**, and the split output
  is byte-identical to it. Stock tiled and untiled decodes already differ, so
  the untiled VAE Decode of a family that tiles through `tiled_scale_multidim`
  never splits; MiniMax H3's VAE tiles itself, so both nodes split there.
- **ComfyUI's loop runs once, on the driver, with only its per-unit decode
  swapped.** The split reimplements no ComfyUI code. The plan of
  `tiled_scale_multidim` comes from running the stock function on meta
  tensors, and MiniMax H3's temporal chunks from the model's own chunk helper
  (`vae_decode_plan.py`). A Worker's unit is used only for the exact input the
  stock loop asks for, checked by view geometry or `torch.equal`, and only when
  it matches the first unit the driver decoded in dtype and channel count.
  Every other unit, and every unit after a mismatch, is decoded on the driver
  by the stock function.
- **The driver decodes its own share.** It takes units from the front of the
  loop as the loop reaches them; ranks on other GPUs claim units from the back
  (`vae_decode_schedule.py`). A rank that shares the driver's GPU takes none.
  A rank's reply waits on the driver until the loop reaches it, so a rank
  claims only within a window ahead of the loop and within a reply budget
  (`vae_decode_remote.REPLY_BUDGET_BYTES`, 2 GiB): each claim reserves the
  bytes of the reply the stock function would return, and the reservation ends
  when the loop takes the reply or the unit fails. While the next unit would
  not fit, the rank waits, and it resumes as the loop drains replies.
- **A Worker's decode must be the driver's decode.** The Worker loads the same
  file (same name, same artifact signature) with the driver's VAE dtype, casts
  to the driver's output dtype, and decodes under the driver's attention
  functions and torch backend flags, putting its own back after. A different
  ComfyUI commit, torch or CUDA build, comfy-kitchen release or VAE class is a
  decline, and so is an attention kernel the Worker cannot run
  (`vae_decode_env.py`). A driver VAE whose compute dtype ComfyUI's VAE
  Compute Dtype node set stays on the driver: that node keeps the weights the
  loader built, and a Worker builds its VAE in one dtype
  (`vae_decode_remote.vae_source`).
- **What an actor may hold.** One decode VAE beside its DiT, placed on the GPU
  directly rather than through ComfyUI's model manager, so loading it never
  evicts the DiT. Its load is priced against MemAvailable before any byte is
  read (the stock placement factor times the file, plus the absolute host
  floor or the operator's larger reserve), and each unit's working set before
  it runs; a price that does not fit is a decline. `status()` reports it with
  its bytes, and `clear_vram` at either level, `unload` and a teardown that
  releases models drop it, so the driver's free-before-refusing step (§5.9)
  reclaims it like any other reloadable cache. A DiT price reads MemAvailable, which counts a resident
  decode VAE as spent until it is dropped.
- **Fail-open, and the knob stays on the driver.** A rank that raises,
  declines, times out or answers with the wrong dtype or shape (every
  dimension is checked against what the stock function returns for that
  unit) is retired for that decode, and its units are decoded on the driver with one WARNING naming
  the cause: a failure costs time, never a wrong image. `worker_vae_decode`
  never reaches an actor, so it stays out of the published worker policy,
  every Gate capability context and the trust fingerprint, because a decode
  changes no DiT math.

Per unit, the wire carries the unit's latent as a CPU tensor in the VAE dtype,
the VAE's file identity, dtype and header metadata, the output dtype and the
driver's decode environment. The reply is the decoded unit in the dtype the
stock function returns, copied into CPU pages the Worker touched first
(`vae_decode_remote.py`). Every decode logs one INFO line on the driver saying
whether it split and over how many ranks, and `dgxm doctor` reads the last one.
The hardware byte compare per family is in
[VALIDATION.md](VALIDATION.md#worker-assisted-vae-decode).

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
  The driver refuses it, class P, on a checkpoint whose ComfyUI configuration
  samples at multiplier 1.0, where the node's multiplier 1000 renders noise
  ([TROUBLESHOOTING #128](TROUBLESHOOTING.md#128-model-sampling-sd3-dgx-monarch-refuses-on-a-family-whose-comfyui-sampling-multiplier-is-10)).
- `DGXMonarchModelSamplingMiniMaxH3`: stock `MiniMaxH3SigmaShift`'s two
  widgets (ModelSamplingMiniMaxH3) in stock's order with stock's defaults, on
  the same wire as the SD3 shift and, like it, outside model residency
  identity. Each Worker applies ComfyUI's own node to its render clone, which
  sets the paired video and audio sampling and the two shift keys the H3
  forward reads. p-h3-image needs it at 1.0 and 1.0. The driver refuses it,
  class P, on a checkpoint that is not MiniMax H3, and refuses a one-frame H3
  audio-video latent, the p-h3-image image latent, that no H3 shift reaches,
  for Fleet KSampler too; each Worker repeats both checks inside its load
  guard, from the loaded model, where it also refuses a ComfyUI without the
  stock node
  ([TROUBLESHOOTING #144](TROUBLESHOOTING.md#144-a-p-h3-image-render-or-modelsamplingminimaxh3-dgx-monarch-refuses)).
- `DGXMonarchSkipLayerGuidance`: stock `SkipLayerGuidanceDiTSimple`'s four
  widgets in stock's order with stock's defaults, kept on the model spec and,
  like the shift, outside model residency identity and outside the Gate
  combination, so a PASS covers a combination with and without the skip. Each
  Worker applies
  ComfyUI's own node to its render clone after the shift; empty layer lists
  change nothing. The driver refuses an active skip, class P, before dispatch
  on a cfg topology, with a two-checkpoint guider, and under sequence
  parallelism on a family whose adapter does not declare
  `sp_honours_block_replace` (MOSS-Image alone so far), and each Worker
  repeats the check from facts every rank holds
  ([TROUBLESHOOTING #126](TROUBLESHOOTING.md#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider)).
- `DGXMonarchMiniMaxH3ContextWindows`: stock `MiniMaxH3ContextWindows`'s five
  widgets in stock's order with stock's defaults, kept on the model spec and,
  like the skip, outside model residency identity and outside the Gate
  combination; the model request carries it only when the node is present.
  Each Worker applies ComfyUI's own node to its render clone after the skip.
  Windows are cut above the forward, so each is one model call on a shorter
  clip, whose packed sequence the sharded forward splits and whose layout it
  rebuilds. The driver refuses the node, class P, before dispatch on a family
  other than MiniMax H3 and on a two-checkpoint guider's negative model, and
  each Worker repeats both checks inside its load guard, where it also refuses
  a ComfyUI without the stock node. The sample deadline counts every window's
  rows; the activation preflight still prices the whole clip
  ([TROUBLESHOOTING #142](TROUBLESHOOTING.md#142-a-render-with-minimax-h3-context-windows-dgx-monarch-refuses)).
- `DGXMonarchQwenImage21Cache`: sets Qwen Image 2.1's replicated prefix KV
  cache on every worker, off by default. `default` storage is lossless; `int8`
  and `int4` use ComfyUI's compressed K/V cache and are approximate. The policy
  rides in the loader options, so it is part of `base_key` (§5.5). On ComfyUI
  503d785f0 or later, `auto` under dynamic VRAM refuses on Ulysses, because
  stock then uses its writable K/V cache instead
  ([TROUBLESHOOTING #134](TROUBLESHOOTING.md#134-a-qwen-image-21-ulysses-render-with-the-cache-device-auto-refuses-on-comfyui-503d785f0-or-later)).
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
  rank's box in the words a refusal uses ("the head", "the sibling", "this
  box", worked out once per wave by `box_names.py`), the worker's measured
  duration, and `started_s` and
  `ended_s`, both counted from the moment the wave opened. The block carries
  that moment and is retained the way `render.last` is. The sidebar timeline
  draws one row per job in dispatch order, tagged with its box and
  answering host, beside the per-box memory cards, so a row and a refusal
  name one box the same way. A job that never came back
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
  first render through any sampler node, and `dgxm gate` drives it from the CLI
  (§6.1). A Sampler Custom proof replays the request's own guider, SAMPLER and
  NOISE objects on the first two steps of its sigma schedule
  (`nodes/gate_proof_request.py`); a one-sigma schedule renders nothing to
  compare and runs no proof.
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
and defaults (`tests/node_input_contract.json`). Ten of the 21 nodes declare a
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
  verdict. An ordinary residency ceremony meets its request's capacity before
  that transaction opens: a refusal there writes one terminal INCONCLUSIVE row
  in the same capacity vocabulary and no RETESTING row, publishes no
  process-local verdict, and lets the render dispatch stock, so the next queue
  meets the same check. An untyped abort keeps its denial and says what
  stopped the proof.
  Diagnostic status publishes its event watermark and tail from one locked
  snapshot, so the two values always describe one coherent event history.
- For state the node surface does not show, use `dgxm top`, the sidebar panel,
  `/dgxm/metrics`, a `dgxm diagnostics` capture or, with
  `[monarch.diagnostics] metrics = true` and an `otlp_endpoint`, the ownership
  occupancy counters in your own collector (docs/CLUSTER.md). Monarch's own
  observability services (`enable_telemetry`, `Job.profile`, `monarch query`,
  the dashboard and the mesh admin agent that `monarch tui` reads) stay off,
  and there is no inspector mode for them: no form of them in the pinned 0.7.0
  wheel is both stoppable and confined to what the integration owns. The paths
  below are inside that wheel.
  - `enable_telemetry`, `Job.profile`, `monarch query` and the dashboard each
    need a Job (`monarch/_src/job/job.py:502-597`; `monarch query` loads the
    active Job at `monarch/tools/cli.py:601-606`). A Job over the persistent
    Workers (`StoreJob`, `monarch/_src/job/spmd.py:405-482`) attaches to them
    as a second client, and attach pushes that client's config to the
    Workers' host agents (`monarch/_src/actor/bootstrap.py:258-265`;
    `monarch/_rust_bindings/monarch_hyperactor/config.pyi:179-180`). The
    driver is the single Monarch client (section 5.1).
  - The mesh admin agent needs no Job. The private `_spawn_admin`
    (`monarch/_src/actor/host_mesh.py:482-517`) starts it on any HostMesh the
    caller holds, such as the driver's own, through the caller's actor
    instance (509-515). It can bind loopback: `_spawn_admin` and
    `MeshAdminConfig` (`monarch/_src/job/job_components.py:55-69`) take an
    `admin_addr`, and the `mesh_admin_addr` config (environment variable
    `HYPERACTOR_MESH_ADMIN_ADDR`) sets the fallback. That fallback defaults to
    `[::]:1729` in the native binary, which is every interface; the stub
    gives the same address only as an example (`config.pyi:177-178`). Its
    handle has no methods
    (`monarch/_rust_bindings/monarch_hyperactor/host_mesh.pyi:178-184`), and
    upstream's own reset only drops the Python reference
    (`job_components.py:317-319`). So, as far as the Python surface shows, it
    ends only with the process that spawned it: started in the driver, it
    would last as long as ComfyUI, and a separate inspector process that
    attaches to the Workers to get a HostMesh makes the second-client attach
    above.
  - Telemetry, `monarch query` and `Job.profile` run in the job sidecar, a
    daemon built to outlive its caller (`monarch/_src/job/once_daemon.py:82-84`)
    that a shutdown call can stop (`monarch/_src/job/job_sidecar.py:102-113`).
    Its query API binds 127.0.0.1
    (`monarch/monarch_dashboard/server/app.py:89-104`).
    `start_telemetry_servers` takes a `host` that defaults to `::`
    (`app.py:107-113`), but `TelemetryConfig` has no bind host
    (`monarch/_src/job/telemetry_config.py:98-101`) and the sidecar call
    passes none (327-331), so the dashboard, when enabled, binds `::`, every
    interface. The sidecar also spawns a telemetry proc mesh on every Worker
    host (`monarch/_src/job/telemetry_actor.py:148`) whose stop is best effort
    with a 5 s bound (`telemetry_actor.py:183-196`).
  - The periodic snapshot actor has no stop handle. Upstream says proc
    lifecycle owns it and that new config needs the job killed and applied
    again (`job_components.py:362-375` and the TODO at 529-531), so it runs
    until its procs stop. Only `snapshot_interval_secs = 0` keeps it off
    (`job_components.py:400-402`).
  - `monarch tui` needs `--addr` of a mesh admin server
    (`monarch/tools/cli.py:85-161`, registered at 719). The wheel also ships a
    separate `monarch-tui` binary; nobody has checked its arguments.

  An inspector mode becomes possible once upstream gives the admin handle a
  stop and periodic snapshots an abort handle, lets telemetry run without
  spawning procs on the hosts, and lets a client attach without pushing its
  config. A bind host in `TelemetryConfig` would also let it include the
  dashboard. docs/TROUBLESHOOTING.md ("Power-user escape hatch") says what to
  use until then.

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
| **K** known-wrong math | measured or by-construction wrongness: attended pad rows on ring, an approximate attention kernel the operator selected, an approximate kernel a checkpoint or LoRA pins on an attention module where nothing drops the pin (except a checkpoint's own header pins in a world of one GPU, which run as stock ComfyUI runs them and ask nothing; a `single` preset on a two-rank mesh resolves to dp2 on the Worker and keeps the guard), and a sharded nvfp4 render of a family measured past the fidelity floor. | **Refuse by default.** The ring divisibility-pad guard carries an expert waiver: one card in the accuracy style carrying the measured wrongness in its own text, `DGXM_WAIVE_KNOWN_WRONG=ring_pad` headless, never auto-eligible, and every render under it stamped. The Sol-Attn kernel carries the same shape under `DGXM_WAIVE_KNOWN_WRONG=sol_attn`, and a sharded nvfp4 render of a measured family carries it under `DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`. The card and the headless value use the unscoped name. Sol-Attn raises family-scoped at the kernel and under `sol_attn:checkpoint_pin` where a pinned approximate kernel would run outside a world of one GPU, or in one where a LoRA sets the pin or a live attention module differs from the checkpoint's (docs/ADAPTERS.md); shard-quant raises family-scoped at every site; ring pad raises scoped for MiniMax H3 and unscoped at the adapter-base backstop. Historical PixelDiT waiver rows remain readable, but that guard was retired after the exact-gather fix passed hardware parity. |

Two related boundaries:

* A **byte-verify certificate mismatch** is class P. The slab does not hold the
  checkpoint's bytes, so the load aborts and nothing is adopted. It is a
  storage or artifact fault, not a model-correctness finding: it writes no gate
  verdict, quarantines no lever, and revokes no consent. Classifying it K would
  imply a waiver could exist, and none ever can.
* **`auto` never substitutes known-wrong math.** A waiver is an expert choice
  about a topology the operator typed, so the auto-resolution refusals carry
  class K with no guard and say in one sentence why no card will appear.

**A client that sends the same graph again gets the same refusal.** A typed
refusal names what must change before the render can run, and every attempt
pays the bring-up again and, where the refusal comes after it, the model load.
So a script or API client reads the tag that opens the error
(`[dgxm:<class> ...]`) before it decides. Class P is never worth a retry
without the change the message names, to the graph, the topology or the
software. Class K is worth one only after the waiver its card names is granted
or the change it names is made. Class C is worth one after its card is
accepted, after memory on the box is freed, or with the residency it names; a
driver spend refusal (`DriverSpendCapacityError` at the node that would spend,
class C, no card) is worth one only after memory on the box is freed or the
spend is made smaller, since the same graph gets the same refusal until then. No
site raises class U this release. An error with no class tag (a transport
fault, a Worker restart, a timeout) is not a refusal, and a retry can succeed.

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

**The sentence leads the error.** Monarch prints its wrapper as "Actor call
... failed." and the remote traceback, with the tagged sentence at the end, and
ComfyUI shows that as the prompt error. The node methods that meet a Worker's
refusal (the samplers, Fleet KSampler, the UNET loaders, Identity Gate) raise the inner
refusal itself, chained from the wrapper, so the error starts with the tag, the
sentence and the card it names, and ComfyUI's log keeps the remote traceback
(`nodes/refusal_lead.py`). On 2026-11-05 a class K ring-pad refusal and a
Worker's stock wall at the sampler both reached the prompt error that way,
with no wrapper and no traceback; a refusal that offers a card ends with the
consent block the panel reads. Only a text that leads with a tag is lifted out, so
a crash keeps its wrapper, and every decision that reads the wrapper (the
sample lease, eviction, the card) runs below the node method. So the lifted
error drops the machine blocks after the sentence, the measured numbers and the
consent descriptor: the receive point has already registered the card and the
gate has recorded the numbers, the user reads the sentence and the card's plain
words, and ComfyUI's log keeps the blocks on the original error, which the
shortened one is raised from and never changes, wrapped by Monarch or raised by
the driver directly. A wrapper the
prompt error keeps names the box on its own first line ("On the sibling, actor
call dgxm_worker.load_model failed."), by the rule under "A Worker's own
refusal says here". When the driver
sends a render on stock although `slab_weights=on`, for want of an Identity
Gate PASS, after a failed authorization, for a second model, or from Fleet,
the Worker sees only the setting arrive off; the same seam replaces its
"slab_weights is off" clause and its "set slab_weights=on" option with that
cause, and with the Identity Gate run where its stock reference fits as the
remedy where a PASS would help.

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
`nodes/gate_quarantine_scope.py` limits that change in two ways, and
`nodes/gate_identity.py` limits the push in a third:

- A waivable class K refusal concerns accuracy, not residency. It disables no
  residency setting and sends no worker-policy update. The refusal tag is
  checked against `refusal.GUARDS`; this applies to Sol-Attn and every other
  waivable class K guard. The cached denial retains the class for later
  requests. `authorize_normal_render` still requires a PASS and stamps the
  dispatch stock when none exists.
- An ordinary abort affects only the combination tested. The next combination
  through the graph starts with the graph's requested settings. The aborted
  combination remains denied until its next Gate run resolves it.
- An abort on an undeliverable Worker actor (`WorkerDeliveryError`: a Worker
  loop restarted under the mesh, issue #47) disables the combination's levers
  like any ordinary abort but sends no worker-policy update. The actors that
  would take it are gone, and the stale mesh refuses every price until it is
  replaced.

A capacity stop before the proof render is no abort and takes nothing
("Where the ceremony cannot help" below). An abort leaves the earlier
`RETESTING` row in place. These limits do not clear a measured FAIL:
invariant 2 preserves that quarantine across sessions until the Gate protocol
or package version changes. A legacy unscoped FAIL remains.

**What a FAIL takes, and from whom.** A FAIL disables `lora_low_rss` and
`slab_weights` for that checkpoint combination in that capability context. It
never writes those two levers into the graph's policy dict, which every
checkpoint wired to the same Init node shares. Each dispatch of the failed
combination reads the FAIL back from the ledger, or from the process verdict
cache in the session that measured it, and renders a clone with both levers off
(`nodes/render_quarantine.py`). Another checkpoint on that Init node keeps the
settings the graph asked for and runs its own Gate. Two writes still reach the
shared dict. A FAIL turns `comfy_managed` off for the whole Init node, because
a live worker cannot change that bootstrap policy for one dispatch. A ledger
that cannot be read turns every guarded lever off, because nothing then says
which combination is safe. Because the levers are scoped per combination, two
checkpoints on one Init node can need different worker settings, and a live
sample lease refuses a settings change. A pipelined dispatch whose setup
differs from the live one first collects the renders still in flight
(`nodes/render_submit.py`), as the pipeline does before a ceremony.

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

**The loader's log level, and who may read the allocator.** Before 1.5 a
comfy-managed render could stop for good inside its first step when a status
read met it; the stall guard ended it after its budget and the next render
recovered by itself. The loader (comfy-aimdo) hands every log line to a
Python callback and its native level starts at "everything", so once its hooks
are in it calls Python at each device allocation and free, from inside the
call torch makes while it holds its allocator lock. A call into Python needs
the interpreter lock. The Worker's status read asked torch for its allocated
and reserved figures, which takes that allocator lock without giving up the
interpreter lock. The GPU thread then held the first and waited for the
second, and the status read the reverse. Two rules follow, and both ship:

- The bring-up sets the loader's native log level to INFO, as ComfyUI's own
  process does, after the library loads and before its hooks go in
  (`actor/comfy_dynamic.set_native_log_level`). The per-allocation lines are
  dropped inside the library; its warnings and errors still reach the log.
  The bring-up reads the level back from the library after the hooks are in
  and says it in its line (`... native log level INFO (read back as 4)`).
  Nothing in the library but its level setters writes that value, so the one
  read holds for the life of the process.
- No code of the pack reads or changes torch's device allocator on a Worker
  off the GPU thread while a job runs. Every load, sample, decode, unload,
  clear (the decode relief's pool level included), setup, teardown and gate
  cycle runs on the GPU thread under the GPU lock, the readiness exchange and
  the backup agreement inside the sample. The calls that stay off the lock
  (`capacity_quote`, `artifact_identity`, `renew_client_lease`,
  `cancel_sample`, `ack_latent_handoff`) read files and the kernel's figures
  and never the device. `status()` was the one exception, and now reads the
  device only while the lock is free (`actor/worker_status.device_figures`).

The root is the loader's: a Python callback run under torch's allocator lock.
With the level at INFO it can still do that in two places: an error line from
inside a hook, on a failure path, and its page-out callback when it evicts
pages, which ComfyUI hands it only for a model it marks writable (among the
shipped families, Qwen Image 2.1). Neither can meet a status read any more.

**What the second rule cannot promise.** It binds the pack's own code, not
Python's. A garbage collection or a finalizer can free a device tensor on the
actor's event-loop thread while that thread holds the interpreter lock (a dead
cycle that still held one, ComfyUI's own model finalizer), and the free takes
torch's allocator lock. If the GPU thread is at that moment inside one of the
two calls above, the same two-way wait results. It is possible by mechanism
and no record shows it: before the level was set every device allocation
opened that window, and now only an error line or Qwen Image 2.1's page-out
under a memory deficit does. The stall guard still ends such a render in
words and the next render recovers. Closing it for good is the loader's
change: no call into Python from under the allocator lock.

**A class-K guard that blocks the ceremony answers first.** Gate proof renders
ignore accuracy waivers. A class-K refusal therefore blocks every proof
attempt for that combination. A checkpoint's own attention pins in a world of
one GPU meet no guard, so a ceremony there runs every leg with the pins, on
every residency, as the render does. A bf16 or fp16 file under `+fsdp` meets
no guard either, by the shard-build rule in the precision paragraph above.
Stock-capable residency paths can continue to the ordinary render, where an
explicit waiver may apply. Comfy-managed
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

Before the leg counts as unable to load, it frees what the driver host can get
back, with the two steps the render-site checks take ("The driver frees what
ComfyUI can reload before it refuses", below): it unloads the models ComfyUI
holds there, then asks the actors to release their allocator caches, and runs
the leg once more after each (`nodes/gate_cross_mode.py`). That holds whether
the fleet agreement refused the leg or a Worker's own ladder did, as on one
Spark, where the refusal reaches the driver as remote text and the leg counts it
by `mesh_safety.is_stock_load_capacity_error`. The ceremony's
conditioning is already encoded, and ComfyUI reloads its models when the graph
next needs them. The driver's text encoder counts against the head's stock
price: on 2026-11-04 Flux 2 fp8mixed priced 74.3 GiB against 63.0 GiB there,
with its 33 GiB encoder loaded, on a fresh pair whose head read 97.5 GiB
before the graph ran. Only the last answer counts. Without a LoRA stack this
leg is the whole proof, so its refusal, not the swap cycle, is the reason the
verdict gives.

With active capacity-rescue consent and no LoRA stack, this outcome keeps
`slab_weights` and its required `lora_low_rss` enabled. The consent is the
panel click, the standing auto-rescue toggle or `DGXM_AUTO_RESCUE`, or
`DGXM_ALLOW_SLAB_RESCUE`; each produces the same consent record. Returning to
stock would retry the load already found not to fit. A LoRA stack introduces a
separate unproven path, so both settings are disabled, except that FSDP keeps
`lora_low_rss` as described above. `CAPACITY_CERTIFIED` records the load's
byte verification and availability; it is never an accuracy verdict.

Nor can the ceremony help a request that cannot fit the path it would prove.
Before its `RETESTING` row, an ordinary residency ceremony runs the render
site's capacity checks for the whole request under its own worker policy
(`gate_capacity_stop.ceremony_activation_preflight`). It credits the DiT only
when a finished render or a pipelined submission loaded it, or an explicit
preset loaded it at the loader node; a pending load that the render site's own
check cleared a moment earlier is charged, because the ceremony's load has not
happened. The checks free what they can first, as at the render site. If one
still refuses, no proof render could finish, so none runs
(`nodes/gate_capacity_stop.py`). The ceremony writes one terminal INCONCLUSIVE
row with the FSDP stop's fields (`cross_mode` `CAPACITY`, a bounded class of
what refused, no measurement) and no `RETESTING` row, publishes no
process-local verdict, and takes no lever. On the automatic path the render
then dispatches on stock residency, because no PASS authorizes more, and the
next request of the combination meets the same check. A resolved FSDP topology
has no stock to fall back to, so there the refusal answers the render, and an
explicit Gate node raises it. The no-LoRA FSDP proof keeps its own clean-reload
price.

**A placement the ceremony would drop is not made.** The ceremony's first leg
must come from a clean load from the file, so it unloads every rank first. With
an explicit topology the UNET loader would otherwise place the model at node
execution and the ceremony would drop it a moment later and load the file again
(issue #157, a 40.4 GiB file placed twice on 2026-11-06). The loader therefore
leaves the load to the ceremony's first proof render when a ceremony will run
before the graph's first render (`nodes/loader_gate_defer.py`). It reads the
prompt from the loader to the nodes that render its model, collects the LoRA
stack on the way, and asks what the first-use gate would do for each one; an
Identity Gate node always defers it. Any doubt keeps the eager load: a Fleet
consumer, a node the walk does not know, a linked LoRA widget, a projected
capacity-rescue consent, or an error. A BasicScheduler (DGX Monarch) also keeps
it: the scheduler loads the model on the Workers to compute its sigmas before
any ceremony, and the model sampling object it reads comes from a loaded
model, so a graph with one still loads the file twice. No trust claim changes: the ceremony
still unloads every rank and loads its first leg from the file, now once, and a
loader-time placement never stands in for that leg. The loader's own price and
the render site's credit for an explicit preset are unchanged; the deferred load comes where the ceremony's
reload came before, after the driver stack, and the Worker's load-time
capacity check prices it.

**Two windows, not one sum.** The loader checks two peaks separately. The
transient window covers loading against MemAvailable minus the reserve: a
stock load uses about 2.1x the artifact size while host and device copies
coexist. The settled window covers weights at 1.0x plus the driver models and
other memory the graph still needs to allocate. Slab removes the host copy
and its transient peak; it does not reduce the settled graph requirement.

**A still guide's encode and the render activations are alternatives where
the order is proved.** Add Guide encodes in the driver before the sampler it
feeds, and ComfyUI with DynamicVRAM empties its device pool after every node
(the executor's `finally` calls `reset_cast_buffers`, which ends in
`soft_empty_cache`), so the encode's memory is back before rank 0 samples. The
settled window then charges the larger of the two, not both, but only for
guides whose files prove them stills (fewer frames than a clip, as the still
charge itself is proved). The graph must also prove the order: ComfyUI in the
driver returns the pool after every node; no node that uses a model is
upstream of a guide; every node that may sample has every guide among the
nodes ComfyUI is certain to run before it; and no pipeline or fleet sampler is
in the graph. Anything else keeps the sum, and the budget clause and the
guide's line say which was charged and why (`nodes/guide_order.py`). The
admitting path prints no card, so an admitted load logs one line at INFO that
names the rule that ran: the two as alternatives, with what the sum would have
said at the same reading, or both charged together and why. A seam
contract holds ComfyUI to the per-node reset. Where the DiT is already resident
while a guide encodes and the guide charge is the larger term, no activation
charge pads the guide's, so the alternative rests on the guide charge alone.
A still's charge (14 GiB) is read against the real driver's encode by the
release's proof; a clip's (34 to 94 GiB) is not, so every clip, and every guide
whose length is declared or unread, keeps the sum. Filed for a later release
with the cell that would read it: a binding clip guide (448x256, 39 frames,
charged 48 GiB) encoded with the DiT resident, its peak read on the box against
the charge (issue #187).

**A deferred load meets the driver's models.** An explicit preset loads at the
loader node, before ComfyUI builds the text encoders, the VAEs and the encodes,
so its load window charges the DiT alone. Under `auto` the loader defers the
load to the first render, and by then the driver has loaded every model the
graph loads, so that load window charges those models too: the text encoders
at their resident ratio, the VAEs, and any checkpoint or stock UNET the driver
loads itself. Encodes and activations come after the load and stay in the
settled window. Until 2026-11-05 the deferred window charged the DiT alone,
and the p-h3-image template, a 37.7 GiB DiT beside an 18.0 GiB prompt
upsampler, passed the loader and refused at the sampler 2.0 GiB short after
the upsampler had run; this price refuses it before anything loads, about
7 GiB on the refusing side. A model only a lazy switch's other branch reaches
never loads, and neither does one no output node reaches (`nodes/graph_live.py`).
Held models credit what ComfyUI holds and no more: it cannot say which files
they came from, so a held encoder stands in for this graph's, never for its
upsampler too. The stack counts against MemAvailable even when no DiT is left
to place. The refusal leads with one plain sentence, the figures inside its
first 200 characters, and then names what would fit, each tried in both
windows: same-stem files on disk here in place of the DiT or a driver model, a
switch whose other setting skips a model, and a pair with the Init node's
topology set to `uly2+fsdp`, whose first render runs the gate's clean-reload
proof. **A remedy is never
one the graph would refuse.** A DiT file is named only when a graph here names
it (the artifact manifest lists it), its kind is one every topology here runs,
and none of the refusals the driver can read before a load applies: the LoRA
route, the FSDP launch, the upstream gate, a pinned kernel off sequence
parallelism. The pair is not offered beside a LoRA. What the driver cannot
prove, it does not name. On the pair `auto` resolves to Ulysses or ring, which
hold the whole DiT on the head, so the pair alone changes nothing there; only
a shard build halves it, and `auto` never picks one (`nodes/graph_load_price.py`).

**A re-queue the loader does not see is priced where it renders.** ComfyUI
keeps a node's output while nothing upstream of it changes, so a re-queue in
the same driver can reach the sampler without running the UNET loader, and the
price above never sees it: on 2026-11-05 a re-queue that switched the
p-h3-image upsampler on rendered unpriced, only because the box had room. The
first DGX Monarch node that renders such a queue, a sampler or Identity Gate
(DGX Monarch), which ComfyUI runs on every queue, runs the same price before
any session or Worker call and at every world, for a checkpoint the loader did
not price in this prompt (each price records the ComfyUI prompt id it ran for),
under `auto` only, and only for what is not yet held. The DiT is charged while
its deferred load is pending, as the loader memo reads it: Reset attached mesh,
Clear and retry and a Clear VRAM node set to `recycle` drain that memo, a
Worker restart drains it only when a Worker call fails, after this price, and a
Clear VRAM node at `soft` or `hard` does not, so after those two the DiT reads
as held and the render-site checks price it. The driver models are charged
beyond what ComfyUI holds. By then ComfyUI holds the encoders and the
upsampler, and MemAvailable counts them at their resident size, so held bytes
are credited as encoders at the 1.144 ratio measured on 2026-08-05, which the
charge rounds up to 1.15, and never above it. A VAE still charged is not held,
so it loads for the decode after the DiT places: it stays in the settled window
and leaves the load window. The reference and guide encodes have run by then
too, and the render activations belong to the render-site checks that run next,
which credit what the Workers retain and free what ComfyUI can reload, so this
price leaves all three out; charged here, they would refuse a re-queue of a
reference-to-video graph the box had just rendered. Its refusal is the
loader's, cards included, and says which node priced the graph and that the
Workers loaded nothing (`nodes/requeue_price.py`). The cross-rank agreement is
not that place: it does nothing at world 1, where that re-queue ran, and it
prices Worker rows, not a graph. This price comes after the re-queue's own
upstream nodes have run, so what they spend is priced where they spend it,
below.

**What a node spends on the driver before the sampler is priced where it
spends, where the loader did not price it.** A re-queue's upstream nodes run
before the sampler: a text encoder or upsampler a changed file or a switch newly
selects loads there, and a guide or reference video ComfyUI no longer keeps
encodes there. A still swapped for a clip at 1344x768 encodes for 94 GiB or
more, which takes the host down (TROUBLESHOOTING #83), and ComfyUI does not
promise to run the loader before such a node on a first queue either. So the
driver prices the spend at the node that makes it (`driver_spend_price.py`):
- **What it prices.** Add Guide (MiniMax H3)'s encode, on the canvas and frame
  count it holds, at the loader's guide charge, a clip with 6 GiB more where the
  graph's activation term resolves; Reference to Video (MiniMax H3) with a
  reference video, at the loader's 40 GiB; and the stock loader nodes whose
  inputs the loader charges (Load CLIP, the dual, triple and quadruple CLIP
  loaders, LTX's audio-video text encoder loader, the checkpoint loaders and
  Load Diffusion Model), at what the loader charges for those inputs: a text
  encoder's file at 1.15, a model's whole, several files summed. It wraps those
  classes' node methods on the class objects in ComfyUI's node mapping, so
  ComfyUI's own internal reloads, other packs' loaders and every Worker path
  are out by construction; a Worker, by its environment marker, installs
  nothing.
- **The mark, per term.** Where the loader admits a prompt at the loader site on
  a graph it read, it records what it charged for each spend: the guide at the
  room it held for the encode's window (the larger of the guide and the
  activations, or both), the reference encode, and each driver load node's
  files where no held-model credit took them. A wrap steps aside where its own
  spend was charged at a figure at least its own, and logs that once per node.
  A spend the loader charged at 0, such as a guide on a checkpoint whose family
  it could not read, another family's loader in a mixed graph, or a load the
  held-model credit took, is priced live. A mark is keyed by the running queue
  item's number as well as the prompt id: ComfyUI takes a client's own prompt
  id, checking only its form, so an id sent again finds no earlier marks.
- **When it prices nothing.** Without the running prompt's graph in hand: the
  executing context gives the prompt and node id, and the prompt queue's
  running item with that id carries the graph. And in a graph whose DGX Monarch
  UNET Loader is not of a family the loader prices a driver stack for.
- **The promise.** A marked spend is never refused. An unmarked spend is refused
  only where it would take the box under its floor
  (`driver_footprint.reserve_bytes`) after every rescue: the room is
  MemAvailable plus what this process's torch pool holds unallocated, and before
  it refuses it unloads what ComfyUI holds and trims the Workers' idle caches,
  as the render-site checks do. That is a new refusal for a queue that today
  renders only by dipping under the floor. The loader's held-model credit and
  its credits for a model a later load drops are not this room's, so a spend
  ComfyUI runs before the loader can be refused where the loader-first order
  would have admitted it, for example under an explicit preset with another
  model still in the Worker's slot; the refusal is still true of the box.
- **The refusal.** Class C under `driver_footprint_preflight`, the driver-host
  guard of the frozen vocabulary, with no card, as its own type, so no consent
  card matching the stock-load family offers a slab load for an encode. It
  names the spend, its charge and the room above the floor, says whether this
  prompt reached the loader (the loader priced it but did not charge this spend
  at its size, or it has not priced this prompt at all), and that nothing was
  spent. Its remedies are a shorter clip, a smaller canvas or file, and memory
  freed on the box; clearing the Sparks frees what the Workers hold there, and
  the next queue then runs the loader, whose refusal itemises the whole graph.
  A spend it admits logs one line with its charge and room.
  `DGXM_DISABLE_DRIVER_SPEND_PRICE=1` on the driver turns it off, for
  diagnosis. A class another pack registers over one of these names is not
  wrapped, and the first loader price in the process logs a WARNING naming it.
- **Not priced.** VAEs, 6 GiB at most here, under the floor's margin; image
  references and frame anchors, which the loader charges 0 on every queue by an
  earlier ruling, so a re-queue is no worse than a first queue there (filed for
  a loader charge with its own measurement); LTX's guide and image-to-video
  encodes, CLIP vision, ControlNet and IP-adapter loads, and the prompt
  expander's generation, none of them charged at the loader either; and a text
  encoder ComfyUI faults back into a CLIP it already built, which returns the
  box to the state the last render held. The render activations stay with the
  render-site checks.

The loader's reference encode survives its held-model credit, as the guide's
does: it is a transient, and a held model does not show that it has run. A
first queue of a video-reference graph with ComfyUI's models held is charged
its 40 GiB. LTX's audio-video text encoder loader (`LTXAVTextEncoderLoader`) is
charged for what it loads: its `text_encoder` file as a text encoder, and of
the LTX checkpoint it names only the text projection ComfyUI keeps, read from
the file's header (`nodes/driver_model_charge.py`, cached by path), since with
DynamicVRAM ComfyUI maps the checkpoint and reads nothing else of it (the
embeddings connectors are kept only in ComfyUI's compat mode, which the header
also shows). It was charged the whole checkpoint and no text encoder. Without
DynamicVRAM the safetensors library decides what each tensor holds, which
ComfyUI's code does not show, so there, and where the header cannot be read,
the whole checkpoint is still the charge.

**A prompt expander that cannot see the request refuses before the render.** The
MiniMax-H3-Image templates wire ComfyUI's expander chain, Replace Text into
Generate Text behind a lazy switch, read the caption by pattern, and ship it
off. The two shipped templates carry this project's own task prompt and
system prompt in the chain's text boxes, because the text ComfyUI's templates
carry there is MiniMax's and this project does not ship it; the testing graph
keeps both boxes empty. With the switch on and the task prompt box empty,
Replace Text hands on an empty text whatever the request says, and the
expander answers nothing the user asked. A task prompt without the marker
Replace Text fills loses the request the same way, and ComfyUI raises no
error for either.
Neither is a render the user asked for, on any world or budget: the first
spends the expander on nothing and renders from what it wrote or, through
the fallback below, from the request it was meant to improve, and the second
renders something else. So the UNET loader refuses
them class P beside the upstream gate, before the batch preflight, the policy
card, the price and `ensure_live`, and a sampler or Identity Gate refuses them
ahead of its re-queue price, since a re-queue that only flips the switch never
runs the loader. ComfyUI has run the expander by then, as it runs every node
upstream of the sampler, and the refusal still comes before the render. A
Fleet sampler receives no prompt graph, so the loader's call answers for it.
Only what the graph states is read: a Generate Text node ComfyUI is sure to
run for a DGX Monarch sampler, Fleet sampler or Identity Gate
(`nodes/graph_live.py`, its `sure` walk from those nodes, so an expander that
runs only for a Preview as Text beside the path refuses nothing), fed by a
Replace Text node whose text is typed or comes from a Text box. The live walk
keeps both branches of a switch it cannot decide, the safe side for a price;
here that side would refuse a render whose expander is off. So the sure walk
follows a switch's branch only when the switch states its choice, typed,
through a Boolean box or through Compare Text over texts the graph states, and
follows no input of a Branch node. It still follows
what feeds a switch's own `switch` input, which ComfyUI must run to choose.
That is what keeps the refusal standing in the shipped graphs. An answer with
no caption falls back to the request through an If/Else whose `switch` reads
Compare Text over the expander's answer, so with the outer switch on and a
request to expand, the expander runs whichever way it goes. An empty request
skips the expander through one more If/Else whose
`switch` is Compare Text over the trimmed request box. The graph states that
text, so the walk reads the choice as it reads a typed one: an empty request
runs no expander and refuses nothing. A graph wired another way and
one the read fails on refuse nothing (`nodes/expander_prompt.py`).

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

**A loader dtype cast is priced.** File size does not bound the copy a
`weight_dtype` cast builds (an fp8 file cast to bf16 doubles), and until
2026-11-05 the stock probe therefore made no claim on a cast, so every price
that read it admitted the load with no figure: a cast DiT was admitted on a
head with 12.2 GiB available against the 60.9 GiB this rule now charges it.
The header does bound it. `capacity_fit.cast_resident_bytes` charges every
16- and 8-bit float tensor at the cast's width, fp32 and fp64 at no less than
their stored size (ComfyUI keeps some fp32 islands live through a cast), and
every tensor of a quantized layer, one with a `weight_scale`, `scale_weight`
or `comfy_quant` beside it, at its stored size, since ComfyUI's quantized
operations keep that payload. The stock price charges that copy in place of
the file, beside the host copy at the file's own width and the floor, so a
cast that keeps the width charges what an uncast load charges. That host copy
is unmeasured for a widening cast: ComfyUI builds the model at the cast's dtype
before it copies the weights in, so a cast that widens may place more than
this charges. A header the
parser cannot read is charged the file times the target width, the most a cast
can build, since no castable element is narrower than a byte; the file's own
size would understate a bf16 cast of an fp8 file by half. A cast load that
does not fit offers no slab rescue: a slab load under a cast keeps the cast
copy in an annex beside the file, so it saves nothing, and for the same reason
an explicit `slab_weights=on` refuses a cast, which its price cannot measure,
and the explicit-stock refusal never names that lever for one. The driver's
own loader windows take the same figure where the load runs stock: the cast
copy in place of the file in both windows and the host copy at the file's width
in the load window, with no slab rescue, and no file twin or shard build named,
since each would price another copy. Before this rule a cast stood the
loader's check down whole, its driver stack included. A vouched slab,
comfy-managed or FSDP load under a cast still makes no claim, at
the Worker or at the loader, because the staging those paths do under a cast
is unmeasured.

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
| `explicit`, `vouched_auto` | the file plus a capped floor, plus the LoRA bake's stray transient where `lora_low_rss` and a stack are both set; a target the header estimate cannot map prices the whole file; `explicit` refuses a loader dtype cast, which this price cannot measure | `actor/capacity_quote._slab_rung` over `capacity_fit.slab_load_fit`, itself over `capacity_lora_bake.lora_bake_bytes`, taken again by `actor/store_slab_admit.admit`, then by the exact backstop `actor/store_slab_admit.admit_bake` once the real patch set is built |
| `consented_rescue` | the same, after both probes ran | the consent branch of `actor/capacity_quote.price`, over the same `capacity_fit.slab_load_fit`, taken again by `actor/store_slab_admit.admit` |
| `explicit_stock`, `stock_fits` | the file plus the host copy it is placed from (2.1 times the file in all), plus the floor; the store places every weight at load time, so this is the whole peak. Under a loader dtype cast the copy the cast builds replaces the file, and the host copy stays the file's. A swap over the slot's own stock resident is charged that price less the resident's GPU copy at the fleet agreement, because the store drops it before the load; a LoRA change on a stock copy of the same file is the same credit, 1.1 times the file plus the floor, whether the store re-applies the stack in place or, when that would first copy the model off the GPU, drops the copy and loads fresh, and a change to a stack of the held stack's own files is also credited ComfyUI's backup of its weights, up to that host copy and net of the new stack's own LoRA files; the Worker's own ladder prices after its drop, with no store snapshot, and never applies a credit | `capacity_fit.stock_load_fit` over `stock_required_bytes` and, for a cast, `capacity_fit.cast_resident_bytes`, re-taken by `actor/store_residency.preload_capacity_check`; the credits are `capacity_lora_bake.credit_stock_swap` over `actor/worker_capacity._resident_bytes` and `capacity_lora_bake.credit_hot_swap` over `actor/store_bake.swaps_in_place`, whose in-place or fresh choice is `actor/store_bake.copies_off_the_gpu`, and `capacity_lora_bake.held_backup_credit` over `actor/worker_capacity._backup_bytes`, net of `capacity_lora_bake.stack_file_bytes` |
| an FSDP launch | bf16/fp16: this rank's shards plus one block; fp8/int8 or an unreadable kind: the conservative full-file bound; every case includes the floor | `actor/capacity_quote_fsdp.fsdp_rung`, `adapters/fsdp.fsdp_load_capacity_check`, and `actor/store_residency.preload_capacity_check`, all over `capacity_fit.fsdp_required_bytes` |
| the gate ceremony's clean reload | the same kind-aware FSDP figure, against each rank's MemAvailable plus its resident shard pool (torch reserved, capped at one share) | `fsdp_reload_price.price_fsdp_clean_reload`, crediting `fsdp_reload_price._rank_credit_bytes` |
| the driver host's own stack | the transient and settled windows above, one rank's share where a shard build is declared; under `auto` the transient window carries the driver models the graph loads | `nodes/loader_preflight.preflight_loader_footprint` and `nodes/render_preflight`, both over `driver_footprint.estimate_driver_footprint`, which prices a declared shard build through `driver_footprint.fsdp_shard_build_bytes` |
| every rank at once | the leanest rank's row, before any rank allocates | `capacity_agreement.collect` and `capacity_agreement.decide` |
| a model in the way | nothing: it prices no load, it decides whether a refusal offers a clear, from what a clear would give back on each short box | `old_model.rank_gain`, `old_model.fleet_offer` and `actor/actor_footprint.arena_bytes` |

CPU tests require `actor/capacity_quote.UNPRICED_RUNGS` to stay empty. They also compare this table's internal residency names and function names with the source, rejecting missing rows and stale function references. ComfyUI's sample-time memory accounting is separate from these residency choices. A slab or an FSDP shard runs ComfyUI's full load once its weights are
resident (`actor/resident_ledger.declare`), and a
stock load does the same (the stock row above), so the sample-time load does
not ask again for memory the box already holds. A stack still pending under
the baked hot-swap residency keeps ComfyUI's sample-time load, and the
`partial_load_divergence` guard still refuses a render whose ranks load
partially.

**Slab capacity refusals offer no slab consent card.** Slab is the final
residency option; consent cannot resolve its memory shortfall. A model
loaded earlier and still in the way is another matter: when a clear would make
room, the slab refusal offers the Old model in the way card below, because the
fix is a reset, not a consent.

**The driver frees what ComfyUI can reload before it refuses.** The render-site
checks (`nodes/render_preflight`) read MemAvailable on the driver host, where
ComfyUI's own text encoder, VAEs and other loaded models count as spent
although ComfyUI reloads them on demand. Before one of those checks refuses,
`driver_unload` unloads every model ComfyUI holds in the driver process through
ComfyUI's own model management, reads MemAvailable again and runs the checks
again from the one that refused. An actor keeps the last render's activations
in its allocator cache for the next render, and on the driver host those count
as spent too. So if a check still refuses and the render's mesh is live and set
up, the driver asks every actor to release its allocator cache (the Worker
`clear_vram` endpoint at `soft`, which unloads no DiT and drops a resident
decode VAE, §5.6) and runs the checks
once more from the one that refused. Only the last answer refuses, and it says
what was unloaded and released and the readings around each step, which keeps
the class C rule of offering the fitting strategy first. Each step runs at most
once per render, and once more in the identity gate's stock reference leg
("Where the ceremony cannot help"), only after a check refused, never while a render, a result
lease or a Worker call is live in the driver, and never reaches a Worker
resident or ComfyUI's node output cache. "Live" is read on the fleet attached
now (`driver_unload.busy_reason`): ComfyUI caches the Init node's output, so
after a reset a graph still carries the mesh the reset stopped, whose handle
keeps the reset's own teardown mark. A mesh with a confirmed stop has no
process left and holds no work, so it is not read; the fleet attached in its
place is, and the actor trim goes to that fleet. A mesh whose stop was not
confirmed is still read, since its Workers may be alive. With two live meshes
on one config the check reads the one `recycle_drain.attached_fleet` finds
first, as the swap credit's question does. The loader windows take the first
step only, after the swap credit (below): before a loader refuses it unloads
ComfyUI's models, which ComfyUI reloads on demand for the text encoder and
VAEs the graph still needs, or, where ComfyUI holds no model, releases the
allocator cache it kept, and prices the load again. They do not ask the
actors to trim. The
agreement below credits a Worker resident that a load would replace only where
the drop was measured: the GPU copy of the slot's own stock resident, never
the arena a stock load retains, which an in-process unload returns only at
the allocator's purge (below), and never a slab resident. A slab resident
does hold MemAvailable: on 2026-11-05 the drop of a 33.02 GiB Flux 2 slab
freed its shared memory within 2 s, and MemAvailable rose from 49 to 88 GiB.
Two stock loads over that slab were admitted with no refusal that day, so no
credit for it has been needed yet
([VALIDATION](VALIDATION.md#capacity-checks-2026-11-05)).

**ComfyUI in the driver reads the box as it is.** The checks above are the
pack's own and read MemAvailable. ComfyUI has readers of its own in the driver
process, and on unified memory one of them is wrong. comfy-aimdo, which places
every weight of a DynamicVRAM model, reads MemAvailable from `/proc/meminfo` on
an integrated Linux GPU, against a floor of a sixteenth of the device clamped
to 2 and 8 GiB (7.6 GiB on a Spark). ComfyUI's Python
`model_management.get_free_memory` reads `torch.cuda.mem_get_info`, which on
GB10 follows MemFree: a few GiB once the page cache is full, with tens of GiB
available. With DynamicVRAM on that reading does not place weights. With it
off, ComfyUI's classic loader reads it to choose which models to unload and,
under `--disable-dynamic-vram`, how much of a model to load. It sets how many
images a VAE takes in a batch, whether SeedVR2's VAE decodes tiled, whether
stock sampling runs cond and uncond in one forward, when the allocator cache is
flushed and the free figure the UI shows, so those choices moved with the page
cache, and a VAE's result for an image depends on the batch it shared. At node
import the pack replaces the device-free term of that function with MemAvailable
less comfy-aimdo's floor (`driver_uma_memory.install_driver`), on Linux and
only where the driver's CUDA device reports itself integrated, the gate the
actors' correction uses (TROUBLESHOOTING #12). Every other device and the
allocator's cached-free bytes stay ComfyUI's own. The floor is the driver's
alone: an actor's loads are priced before a byte is read, and nothing prices a
VAE batch. `DGXM_DISABLE_DRIVER_UMA_MEMORY=1` leaves ComfyUI's reading in
place. What the reading alone does not settle: a batch whose estimate is most
of the box (six 2048 views in one VAE batch are priced at 82.8 GiB) would
still split by what the session has available. The next rule settles that for
encodes. `dgxm doctor`'s `driver memory reading` row says which reading the
running driver has, from the driver's own `/system_stats`.

**The driver encodes a batch of images one image a call.** Any rule that
divides memory by an estimate moves when memory moves, and a VAE's result for
an image depends on the batch it shared. Only one split never depends on
memory: one image. So on the same GPUs as the reading, and by default,
`vae_encode_item` runs an image VAE's encode of N images as N calls of
ComfyUI's own `VAE.encode` on one image each, joined in order. Each call's
batch count is 1 whatever memory reads, so a graph's conditioning is one
fixed tensor at every size and in every memory state, and image i's latent is
what stock gives for image i alone. Three things bound it:

- **Encode only.** `VAE.decode` is not wrapped, and neither is
  `VAE.encode_tiled`: measured with FLUX.2's VAE on a Spark, a decode gave
  the same result in any batch, and one image a call cost up to 10 percent.
  No other VAE's decode was measured, and its batches still follow the
  reading. A video
  VAE's frames are one item along time and stay one call. The test for an
  image VAE is stock's own: `VAE.encode` treats the first dimension as
  separate items exactly when `latent_dim` is 2, or 3 with `not_video` set;
  an audio VAE carries `extra_1d_channel`. A VAE that lacks one of the three
  attributes is left whole.
- **It steps aside, it does not break a render.** A call with arguments this
  version does not know, a batch it cannot slice or results it cannot join
  hand the call back to stock's whole batch, once and for the life of the
  process, with one WARNING. An error ComfyUI itself raises on an image (out
  of memory, a cancelled prompt) is passed on, and nothing runs again after
  it. The surface canary holds the call's shape and the canary runs the
  rule's test on ComfyUI's real `VAE`.
- **It is not what a box without this pack gives.** A stock ComfyUI with room
  for the whole batch encodes each image beside the others and gets other
  bits (one BF16 step on most values). A single image matches either way. A
  stock render and a DGX Monarch render in the same driver take the driver's
  conditioning, so they compare on the same bits.

Measured on a Spark with FLUX.2's VAE (2026-11-11, 4 and 6 images at 1024 and
2048): one image a call took 0.86 to 1.0 of the whole batch's time and peaked
at 2.28 GiB, where six 2048 images in one batch took 12.41 GiB. The same
measurement is why decode is left alone: its results were identical either
way and one image a call read up to 10 percent slower. Other image VAEs are
exact by construction and were not timed. `DGXM_DISABLE_VAE_ENCODE_ITEM=1` on
the driver turns the rule off; the reading's switch does not. The driver
publishes the rule's state in `/dgxm/telemetry` (`driver_rules`, read for
each answer and never cached with the rest of the payload), and doctor's
`driver memory reading` row repeats it: on, off by the switch, or stepped
aside.

**Before a decode priced above free memory, the driver frees what the decode
needs.** ComfyUI prices a VAE decode and asks its loader for that much room.
With DynamicVRAM on, every model in the driver is dynamic, the loader unloads
none of them for another dynamic model and logs `0 models unloaded.`, and the
decode runs whole anyway, because its batch is at least one item. On a
discrete GPU the allocator then fails at the card's limit and ComfyUI falls
back to tiles. On unified memory nothing fails until the kernel does. On
2026-11-12, in two one-GPU video renders on a DGX Spark under its clock cap
(max 2100 MHz), a decode priced at 16.5 GiB with ComfyUI's reserve started
with 6.7 GiB read as free, used 9.3 and took the box to 6.7 GiB available
while the driver held 32 GiB of text encoders no later node needed; and a
decode priced at 5.6 GiB started with 4.9 read as free and took the box from
12.5 to 7.8 GiB. No fault followed either. Both graphs set `lora_low_rss`
off, which keeps ComfyUI's weight backup on the Worker; rerun on a fresh
fleet, the first graph read a low of 11.63 GiB with nothing given back, 11.41
with the driver's memory reading switched off (so the reading is not what
took the box down), and 34.47 with the Spark's default `lora_low_rss`. A
driver that has run other graphs first holds
their text encoders too, which is ordinary use. So on the same GPUs as the
reading, with DynamicVRAM on, and by default, `vae_decode_relief` wraps
`VAE.decode` in the driver. When the decode's price plus ComfyUI's reserve
exceeds what ComfyUI reads as free, it calls ComfyUI's own `free_memory` for
that requirement, as ComfyUI's classic loader does there, keeping every model
loaded except the text encoders. ComfyUI frees just what the call needs from
the text encoders it holds, in its own order, and a DynamicVRAM model gives
up pages without being unloaded where that is enough. Then ComfyUI's decode
runs. Whenever the relief acts, whatever the text encoders' free returned,
the driver also asks the Workers for memory they hold idle. Until
2026-11-13 it asked only where the decode was still priced above ComfyUI's
reading after the free; on two held pair renders that day the free met the
price on that reading (7.3 GiB free against 7.2 needed, ComfyUI's reading
being MemAvailable less its own floor), the Workers were never asked, and
the first Spark read 10.3 GiB through the decode. The ask is one call of the
trim the driver unload already makes (`driver_unload.trim_actors`, every actor's
`clear_vram` at `pool`), which empties each Worker's device pool and collects, and drops
nothing. A Worker keeps the pool its sampler grew reserved after the last
step, 6.4 GiB on record beside a driver whose decode then took 2.4 to 4.4
(a pair, 2026-11-13, the first Spark under its clock cap, max 2100 MHz: the
box read 4.29 GiB at that decode). The call goes to the fleet attached NOW:
the driver keeps only a weak note of the fleet it last submitted to, and
reads the one attached in its place if a reset retired it
(`vae_decode_relief.serving_now`). It never runs under a sample: the relief
does nothing while one of the driver's samplers runs, the Worker takes the
call under its GPU lock, and the claim on the mesh's render session fails
within seconds where a pipelined render owns it, after which the decode runs
as before. The relief's line always ends its Worker clause with one of three
things: the Workers released (with MemAvailable on each side), they were not
asked and why, or no fleet was attached to ask. The ask has no hardware
reading yet. Eight things bound the relief:

- **Only text encoders are freed.** By an allow-list, not a deny-list: a text
  encoder is the one kind of model ComfyUI marks for certain (`is_clip` on its
  patcher, set in `comfy.sd.CLIP`), nothing samples with one, and every gain
  on record is one (24 and 7.7 GiB in the first record, 6.3 in the second). A
  diffusion model, the VAE in hand, another VAE, CLIP vision, a ControlNet, an
  adapter or a model patch is never touched, whoever calls, from whatever
  thread and whatever sampler. A text encoder on another device than the
  decode's is left too, since ComfyUI's call frees on one device.

- **The trigger is ComfyUI's own.** The price is `VAE.memory_used_decode` and
  the requirement is the one `load_models_gpu` computes from it
  (`minimum_inference_memory`, `extra_reserved_memory`), both called and not
  copied, against the reading ComfyUI takes a moment later. It is the state in
  which ComfyUI itself logs `0 models unloaded.` A decode that fits costs two
  calls and nothing else. The VAE's own weights are left out of the trigger:
  DynamicVRAM places them against its own budget.
- **No result changes.** The wrapper calls ComfyUI's decode with the caller's
  own arguments in every path and computes nothing that reaches the result. A
  decode of one item (every video, every single image) takes a batch of one
  before and after. A decode of several images takes `free / price` images a
  batch, as it always did, and that free figure already counted what
  DynamicVRAM could evict (`ModelPatcher.get_free_memory`), so the count
  barely moves. Freeing changes no weight: ComfyUI faults a page back, or
  loads a model again, with the same bytes when a node next asks for it.
- **Never while a sampler runs.** Preview nodes call `VAE.decode` at every
  step, and a stock sampler in the driver is nothing the pack's own render
  tracker sees. So `comfy.samplers.CFGGuider.sample` is wrapped with a
  process-wide depth count, the relief does nothing while it is above zero,
  whichever thread the decode comes from, and the decode wrapper is never
  installed without that guard. ComfyUI runs one prompt at a time, so a
  process-wide count loses nothing. A sampler that builds its own loop and
  never reaches `CFGGuider.sample` is not counted; for that one the allow-list
  is the whole protection, and it is enough.
- **Its tests of "in use" are its own.** The render-site unload asks
  `driver_unload.busy_reason` because it drops every model and calls the
  Workers. The relief does neither, and it does not ask that check: on
  2026-11-12 the check read "a call to the Workers is still in flight" after a
  render had finished, and a relief that leaned on it would go dead exactly
  where a decode follows a render. The sampler guard and the allow-list are
  the whole test. A render on the Workers uses no model in the driver's
  process.
- **It honours the unload switch and DynamicVRAM off.**
  `DGXM_DISABLE_DRIVER_UNLOAD=1` means the driver unloads none of ComfyUI's
  models on its own, and the relief then keeps everything. With DynamicVRAM
  off ComfyUI's classic loader already unloads for the decode, and unloading a
  classic model on unified memory copies it first, so the relief does nothing.
- **It adds room; it does not bound a decode.** It frees what ComfyUI prices,
  and ComfyUI's prices have run above the use in every record (16.1 GiB priced
  for the decode that used 9.3). With nothing idle the decode runs as ComfyUI
  runs it, and one line says so. The Workers are not asked for their cached
  memory here: the one Worker call there is reaches every box, and only the
  driver host's Workers share the decode's memory.
- **It steps aside, it does not break a decode.** If a call it makes raises,
  ComfyUI's own decode answers then and for the life of the process, with one
  WARNING. A VAE that cannot price its latent is that decode's matter alone.
  ComfyUI's interrupt passes through.

What it costs when it acts: a text encoder that gave up pages faults them
back, or loads again, when a prompt's text next changes; an unchanged
prompt's conditioning is cached. What a freed page returns has no hardware
record yet. The nearest is a whole unload, which the render-site checks make:
on 2026-11-11 and 2026-11-12, with the first Spark under its clock cap (max
2100 MHz), unloading the LTX text encoder and two VAEs whole returned 30.6
GiB. `DGXM_DISABLE_VAE_DECODE_RELIEF=1` on the driver turns the
relief off, for diagnosis; the reading's switch does not. The driver
publishes its state beside the encode rule's in `/dgxm/telemetry`
(`driver_rules`), and doctor's `driver memory reading` row adds a clause only
when it is off, stepped aside or not installed. `VAE.decode_tiled` is not
wrapped (it is already the bounded form) and neither is `VAE.encode`. **NOT
RUN** on hardware as merged: the two records above are of the build before it
(TROUBLESHOOTING #12).

**A Worker makes room for its CUDA context before it creates one.** The
rules above are the driver's. This one is the Worker's, because the context
is made there: a Worker's first CUDA call is ComfyUI's import of
`comfy.model_management` inside `comfy_bridge.ensure_comfy`, which every
setup reaches by one door, `worker_env.setup_impl` (an attached Worker, a
local mesh and a respawn alike). No endpoint that runs before setup touches
CUDA: status and `clear_vram` both say so and return early. On a GPU that
shares the host's memory the NVIDIA driver needs free memory for a new
context, and the page cache does not count: on 2026-11-12, on a DGX Spark,
three Worker starts failed in `cuDevicePrimaryCtxRetain`, with 1.52, 4.61
and 8.85 GiB free while 44.7, 47.4 and 50.8 GiB were available, and the
kernel log read `NV_ERR_NO_MEMORY` from the driver's allocation of the
context's main buffer. One start on record made its context with 8.87 GiB
free, and every other had 10.6 GiB free or more, so the line lies near 9
GiB. Each failure came while a program held tens of GiB (a test hold).
Reading files does not by itself bring the box there: 260 GiB of model files
read with nothing held left 26 to 28 GiB free, and the context that
followed was made with 10.6 to 11.5 GiB free. An ordinary GPU allocation
meets the same answer from the kernel and gets past it: the same day a
Worker that was already up began a model load with 2.4 and then 1.8 GiB
free (30 GiB available), the kernel logged the driver's `NV_ERR_NO_MEMORY`
once with no context line, and the load went on and the render finished. So
the step belongs before the first CUDA call and nowhere else: a load
recovers by itself, and the context's main buffer does not.
`actor/context_room.py` holds two rules for it.

- **Room first, then the context at once.** Setup calls the bootstrap through
  `context_room`, which reads `/proc/meminfo` before the bootstrap runs.
  With 16 GiB free or more it does nothing else. With less, and 21 GiB or
  more available (the 16 GiB mark and the 5 GiB floor), on a GPU that shares
  the host's memory, it writes to anonymous memory in steps of 1 GiB up to
  16 GiB, holds every
  step, and releases all of it together. The kernel gives page cache back to
  serve the writes, and the release leaves that memory free. Before each
  step it checks that the step leaves 5 GiB available, and stops if not. It
  locks nothing. The amount is the mark and not the mark less what is free:
  a process's first writes take the free pages, so writing 7 GiB on a box
  with 5 GiB free leaves about 7 free afterwards. Free memory does not stay
  free, since any file read on the box turns it into page cache again. So
  torch is in the process before the writes (a Worker has imported it by
  then; the step makes sure), and straight after the release the
  Worker makes the context itself by the call ComfyUI's import makes first
  (`torch.cuda.mem_get_info`); the bootstrap then finds it made. Under
  comfy-managed residency the Worker makes no context of its own, because
  stage A of that bring-up runs in the bootstrap and stays ahead of the
  context: the room is made, the bootstrap's import makes the context a
  moment later, and memory freed can be taken again in between. There the
  bootstrap reads free memory once more just ahead of that import and logs
  it, so the record has the figure the driver met.
- **The gate asks the driver library, not torch.** Whether the GPU shares the
  host's memory is the CUDA driver's device attribute
  (`cuDeviceGetAttribute`, `CU_DEVICE_ATTRIBUTE_INTEGRATED`), which answers
  from a device handle with no context. The pack's other gates read torch's
  device properties, which is right where a context exists; before one, no
  torch call is known to make none. The reading comes first, so a start with
  room makes no driver call either.
- **A refusal in words behind it.** If the context cannot be made for want
  of memory, at the Worker's own try or in the bootstrap before
  `comfy.model_management` has imported, the Worker raises a class C refusal
  (`ContextMemoryError`, guard `gpu_context_memory`, no card: no consent
  makes memory free) that says what
  failed, what is free and what is available on its box, what the make-room
  step did or why it did nothing, and what to do. It carries the Worker's
  rank, and the driver names the box of any error a Worker raises at setup,
  as it does at a load (`worker_box.name` in
  `mesh_setup_transition.ensure_setup`, from the rank whose answer it was
  awaiting, which covers a Fleet's actors too). Any other error passes on
  as it was raised. On a pair the refusal is not yet prompt, and only the
  head's reaches the prompt. The rank that cannot make its context raises
  before it joins the group (`worker_env.setup_impl` calls the bootstrap
  ahead of `dist.init_process_group`), but the driver awaits the answers in
  rank order. Where the head (rank 0) failed, the driver has its refusal at
  once. The sibling is by then waiting to connect to the store the head
  never opened; that wait holds the sibling's GPU lock (the connect has a
  600 s limit and then tries again), the rollback's `teardown_group` call
  queues behind it on that lock, and the driver raises the refusal, opening
  "On the head, this Worker could not start on its GPU", only when the
  rollback reaches its own limit (`rollback_cleanup_s`, 600 s, and a 10 s
  margin). On 2026-11-12 the raw error took 610 s to show that way, and the
  driver logged that the rollback had timed out. Where the sibling failed,
  the head has opened its store and waits for the sibling to join, and the
  driver is awaiting the head: what reaches the prompt about 600 s later is
  a timeout (the driver's own setup limit, which starts at dispatch, or the
  head's store's), and the sibling's refusal, returned at once, is never
  read. Ending the wait at the first rank's failure, and raising the refusal
  of the rank that failed, is a change to the
  setup protocol and is not in this rule. The remedy it names is a reset of the attached
  mesh after memory is freed: no record shows a second try in the same process
  making a context, and a setup that failed leaves the Worker marked as not
  torn down until a reset or a teardown settles it.
- **It fails open, and it has one switch.** A fault anywhere in the step is
  logged once and the bootstrap goes on as before. `DGXM_DISABLE_CONTEXT_ROOM=1`
  in the Worker's environment turns the step off, for diagnosis; the refusal
  stays. The worker loop's environment filter lets that name through to the
  actors (`cli/worker_process_env.py`). One line in the Worker's journal
  says what the step did whenever free memory was short, and the Worker's
  status carries the same (`context_room`: the GiB touched, the seconds,
  free memory before the step, after the release and at the context, and
  whether the Worker made the context at once or left it to the bootstrap).

The mark is 16 GiB because the record's line is fine: 8.85 failed and 8.87
worked. 16 keeps about 7 GiB over it, at no cost on a roomy start.

What it costs when it acts: a few seconds, and up to 16 GiB of page cache
that files must be read into again. The Worker's unit sets no memory limit,
so the reading is the box's. Not yet run on hardware: whether memory freed
this way serves the driver's request is what the proof cell reads, and if it
does not, the refusal stands alone.

**A LoRA stack's classic bake has a price, and where it does not fit the
Worker bakes in place if the identity gate allows.** With `lora_low_rss` off a
Worker bakes a stack ComfyUI's classic way: the slot is dropped, base and
stack load fresh from the host, and ComfyUI keeps a host backup of every
patched weight. The stock price (the file, the host copy it is placed from and
the floor) had no term for the stack. On one GPU of a DGX Spark under its
clock cap (max 2100 MHz), 2026-11-12, as the rise of the Worker's GPU figure
over the same model's load with no stack, and what stays after a soft clear:

| checkpoint and how its patched layers bake | stack on disk | patched weights | switch | rise | stays |
|---|---:|---:|---|---:|---:|
| LTX 2.3 scaled FP8, ComfyUI's classic bake | 7.08 GiB | 1660 (18.07 GiB) | off | 12.34 GiB | 10.65 GiB |
| the same, a smaller stack | 0.305 GiB | 480 (12.0 GiB) | off | 5.03 GiB | 3.40 GiB |
| MiniMax H3 INT8 ConvRot, cast-time math | 1.82 GiB | 208 | off | 1.21 GiB | 0.12 GiB |
| Qwen Image plain FP8, classic bake | 0.79 GiB | 720 | off | 0.84 GiB | not read |
| LTX 2.3, its own stack baked in place | 7.08 GiB | 1660 | on | 8.28 GiB | 0.47 GiB |
| LTX 2.3, the smaller stack baked in place | 0.305 GiB | 480 | on | 2.25 GiB | 0.62 GiB |

The latents are the same with the switch off and on, for both LTX stacks. The
rise follows how many weights the stack patches and not the stack's size
(the two LTX stacks are 23 times apart in size and 2.5 apart in rise): the
line through the two LTX points is 2.06 GiB plus 6.34 MiB a patched weight.
The first row's load was priced at 50.9 GiB and used 8.7 GiB more than that
price allows above its floor. No code on the classic path keeps a buffer for
a patched layer. The memory that stays is the allocator's own reserve and not
live tensors: after the classic load the Worker read 22.5 GiB allocated
against 34.2 reserved, and after a soft clear 21.8 against 32.5. The price
does not depend on it.

The rises above are what the term was drawn from. A second set of rows, read
the same day on the same hardware, is what a load with its stack and its
render took from the box after the Worker's price line, against what the
stock price (no term) promised it would take. It says which formats the term
is true for:

| checkpoint, patched keys, how it loaded | the price is read | reading | low | took | the stock price promised | short by | now charged |
|---|---|---:|---:|---:|---:|---:|---:|
| Wan-Animate 2 INT8 ConvRot, 1261, loaded partly and baked whole by the partial-load route | after the file | 44.6 | 10.97 | 33.6 | 32.6 | 1.0 | 2.05 (6.95 with a reserve under 8) |
| the same stack, loaded whole | after | 46.7 | 7.29 | 39.4 | 32.6 | 6.8 | 2.05 (6.95), see the INT8 rule below |
| the same whole load on a roomy box | after | 95.7 | 57.80 | 37.9 | 32.6 | 5.3 | 2.05 (6.95), the same rule |
| MiniMax H3 INT8 ConvRot, 208, whole | before the file | 50.9 | 7.73 | 43.2 | 41.0 | 2.2 | 4.7 |
| the same, the base held | after | 51.5 | 9.25 | 42.3 | 41.0 | 1.3 | 2.8 |
| Qwen Image plain FP8, 720, whole | before | 49.1 | 10.19 | 38.9 | 40.0 | none (1.1 to spare) | 2.8 |
| the same model, a second stack of 240, partly and the route | after | 50.0 | 10.74 | 39.3 | 40.0 | none | 2.0 |
| that stack loaded whole on a roomy box | after | 98.6 | 60.19 | 38.4 | 40.0 | none | 2.0 |
| LTX 2.3 scaled FP8, 1660, whole, the base held | after the file | 75.9 | 20.32 | 55.6 | 45.9 | 9.7 | 14.2 |
| the same stack into an empty Worker | before | 74.2 | 11.7 | 62.5 | 45.9 | 16.6 | 21.2 |
| the same 7.08 GiB file listed twice, empty Worker | before | 82.4 | 12.87 | 69.5 | 45.9 | 23.6 | 28.3 |
| the same file, 480, whole, the base held | after | 58.2 | 10.46 | 47.7 | 45.9 | 1.8 | 5.5 |
| Wan-Animate 2 BF16, 1261, whole, the second Spark's Worker of a pair | before | 113.6 | 36.21 | 77.4 | 64.1 | 13.3 | 11.9 |

(GiB.) **Which reading a price is held against matters as much as the
charge.** The Worker reads every LoRA file of a stack into host memory, one
read for each entry, and holds them through the bake. A price read before
those reads (a load into an empty slot, and every quote) must charge the
files; a price read after them must not. The second happens when the slot
holds the base: the store tries its hot swap, which reads the stack to count
what it would copy, declines, and the fresh load's walls then read a box
those bytes have already left. Both happen in ordinary graphs: the records
hold graphs that load the base bare and then change the stack in one
prompt, and graphs that load base and stack in one step; each run's own log
says which (one load line, or a bare load and the declined swap ahead of the
stack's price line). LTX's rows show both, 7 GiB apart, and the file listed
twice took 7 more.

The charge went through two versions that this table refutes. The first
charged the line and the files for every format on both readings, and
refused seven of these rows, among them the shipped Wan-Animate 2 template's
own pair (INT8 ConvRot with its LoRA, `lora_low_rss` off) at a reading of
44.6 against a price of 49.5, and the LTX load at 68.5 against 72.1. The
second took the per-weight share off INT8 and the files off scaled FP8 on
both readings, and came up short on three: the whole INT8 load by 4.1 GiB,
and the two LTX loads into an empty Worker by 2.4 and 4.7. A test holds
every row against the price (`tests/test_classic_bake_records.py`): it must
admit each reading on record and cover what the load took plus the floor,
and each earlier version fails it by name. Two rows, the same stack loaded
whole on a tight box and on a roomy one, do not assert that cover, and say
why: that whole INT8 load takes about 4 GiB more than its price promises at
the default reserve and still leaves the floor, because ComfyUI places that
model whole only from a reading of 45.1, where 39.4 taken leaves 5.7, and
places it partly under that, where the low-RSS route carries it. Those rows
assert that, and that with a reserve under the recorded one the price
covers the whole load by itself. Three rules follow.

- **The term** (`capacity_classic_bake.py`). The line through the two LTX
  rows of the first table is 2 GiB plus 7.5 MiB a patched weight (5.52 and
  14.16 GiB for the two stacks). The weights are counted from the
  checkpoint's and the LoRAs' headers, which give the Worker's own counts
  for both LTX stacks (1660 and 480); a patched bias counts as a weight, a
  stack that cannot be mapped charges every weight, and a checkpoint whose
  header cannot be read is charged as the largest stack measured (1660
  weights) and its files. What a classic price adds is the term for the
  format, and the stack's files where the price is read before them:
  - **The files.** Each file as often as the stack lists it, on a reading
    taken before the Worker reads them; none on a reading taken after
    (`stack_on_host`). "After" is a fact the store holds, not a matter of
    when freed memory returns to the box: the declined hot swap hands the
    store the stack it read (its patches, which reference the files' tensors
    and nothing of the model; `in_place_fallback.note_stack_read`), and the
    store keeps them alive through the drop of the old model and both walls
    of the fresh load that follows in the same call. Each wall asks at the
    moment it reads. The fresh load lets them go just before it reads the
    files again, so the two copies never stand together, and
    `ModelStore.ensure` lets them go on every way out, so a refused or
    failed load leaves the box as it found it. A stack let go before a wall
    reads is charged its files, and so is one held only in part: what is
    held must be the files' own bytes (at least 0.98 of them), which it is
    for every recorded stack and is not for a LoRA with tensors the model
    does not patch, such as text-encoder keys. A quote is always before. So
    the quote
    charges the files against its reading and the wall charges none against
    a reading the files have left, and the two give one verdict.
  - **Scaled FP8** (an FP8 weight with its scale or ComfyUI's quant record
    beside it): the line. It leaves 3.7 to 5.0 GiB above take plus floor on
    the four LTX rows and on three more runs of them, whole loads all, so
    this price too covers the whole load by itself at any reserve.
  - **INT8 ConvRot**: the fixed 2 GiB and 4 MiB a patched key, with one
    recorded exception. Loaded whole, ComfyUI's classic bake keeps the host
    copy of every patched weight as its backup (17.8 GiB of the Worker's
    memory against 1.5 after the bare load) beside the same bytes on the
    GPU and the sampler's working memory, and the larger stack took 39.4
    GiB, 6.8 over the stock price; placed partly and baked by the
    partial-load route, which frees that backup, 33.6. The share makes the
    price cover the whole load: 44.5 after the file and 45.2 before it for
    the shipped Wan-Animate 2 template. It is a fit through that one whole
    load (0.1 GiB to spare), not a mechanism.

    **The exception, and its record.** ComfyUI places a model whole only
    when its free reading is at least the model, the Worker's reserve and
    its own estimate for the latent (`load_models_gpu`: model + reserve +
    max(0.8 GiB, estimate)); under that it places the model partly, which
    on one GPU the route carries. A whole load therefore starts with the
    reserve still free, and what it then takes beyond ComfyUI's accounting
    comes out of the reserve. That was read once: on Wan-Animate 2's INT8
    ConvRot file at the default reserve of 8 GiB, ComfyUI placed the shipped
    template whole only from a reading of 45.1, the load took 39.4, and so
    it ended with the reserve less 2.3 GiB: 5.7, above the floor by 0.7.
    (The 2.3 is about 1.8 that the load holds beside the model, its backup
    and the sampler's memory, and about 0.5 by which the sampler's 6.7 GiB
    passed ComfyUI's estimate of 6.2 for that clip.) So for that model, at
    a Worker reserve of 8 GiB or more, the share is not charged
    (`reserve_keeps_floor`): the price is 39.6 after the file and 40.3
    before it, and on one GPU every reading from there up renders, by the
    route under 45.1 and whole above it. (On a pair a rank that loads
    partly is refused, typed, with nothing sampled: there the fleet admits
    from the price and a rank loads this template whole only from about
    49.5, as it did before this price existed.) The exception holds for any
    LoRA stack on that model, because of the 2.3 only the backup follows
    the stack and a backup cannot pass the model, which ComfyUI's condition
    already counts. The model is named by the pack's own header read (the
    family `wan_animate2`), not by a file name. It is one model's record
    and nothing more:
    - **Every other INT8 ConvRot model pays the share at every reserve**,
      until a whole load of its own is read against ComfyUI's condition
      (MiniMax H3's two recorded loads are admitted and covered at that
      price; the others are listed for the measured pass).
    - **A reserve under 8 pays the share.** The reserve is the Init node's
      `reserve_vram_gb` widget (or `[worker_args]` in `cluster.toml`; 0
      keeps the default 8). A lower one lets ComfyUI place the model whole
      at readings where a whole load does not leave the floor (at a reserve
      of 3, from about 40.1, where it would leave under 1 GiB). The
      arithmetic would allow 7.3; the edge is the reserve on record because
      one load's take varies by 0.5 to 1.0 GiB between runs, which is all
      of the 0.7 GiB left at 8.
    - **A Worker whose ComfyUI does not budget its loads pays the share.**
      ComfyUI weighs the free reading only in its NORMAL and LOW memory
      states; the price reads that state with the reserve
      (`in_place_fallback.worker_reserve`).
    - **A longer clip is not on record.** The sampler's use over ComfyUI's
      estimate was read at one clip (0.5 GiB). If it grows with the latent, a clip
      about three times as long could bring the lowest whole load under
      the floor at the default reserve. The last hardware block reads that
      whole load three times and once at a longer clip; a low under 5.0
      GiB in any of them removes the exception before the release.
    - The rule leans on ComfyUI's code, which is not pinned, so a canary
      seam (`whole load condition`, `tests/canary/comfy_seam_contracts.py`)
      turns the daily lane red if the condition, its gate, the estimate for
      this model or the Worker's own free reading changes. It did so on its
      first run: ComfyUI commit c5751c8 (2026-11-11, "Remove directml
      support") had deleted `lowvram_available`, a flag the gate and the
      price both read, and the seam caught it in CI against ComfyUI master
      before any box met it. The gate is otherwise unchanged, so the rule
      holds there; `worker_reserve` and the seam now read the gate with or
      without that flag, and both hold on ComfyUI 4d64a8bc and on master at
      c1eece6.
  - **Plain FP8** (an FP8 weight with no marker): the fixed 2 GiB, no share.
    Two stacks on one model took 2.7 to 5.0 GiB less than that allows. Its
    price covers its recorded whole load by itself, so it does not lean on
    ComfyUI's whole-load condition and the reserve does not change it.
  - **Small keys beside such weights.** In a stack whose patched bytes are
    mostly INT8 ConvRot or plain FP8 weights, a patched key under 1 MiB (a
    bias, a norm weight) takes that format's share and not the line's, and a
    larger weight of another format keeps the line's (the Wan-Animate 2 file
    has 480 INT8 weights among the 1261 keys its LoRA patches; 774 of the
    rest are at most 640 KiB, and the file's next tensor up is 1.4 MiB).
    Every other stack is counted as the line was fitted, its biases
    included; that difference is a fitting choice and not physics. A stack
    is read by the format that holds most of its patched bytes, so one stray
    key of another format does not change how it is charged.
  - **Full precision** (BF16, FP16, FP32): the line. A Klein
    BF16 stack of 121 patched weights rose 1.00 GiB against a line of 2.89
    and the BF16 Wan-Animate 2 stack of 1261 at most 7.4 against 11.24, and
    the table's last row took 1.3 GiB more than that price covers, so
    nothing is trimmed there. Whether that 1.3 is the bake or the render's
    own activations, which a load price does not claim, is for the measured
    pass to say.
  - **The bytes guard** (0.75 of a format's patched bytes, for a weight
    larger than any measured) has no record behind it and is off. Both LTX
    stacks' largest patched weight is 64.0 MiB; a scaled FP8 file with
    larger ones is past the records and unmeasured.

  Four constants hold those choices: `LIGHT_SHARE_BYTES` (a format's own
  share in place of the line's), `RECORDED_WHOLE_LOAD_FAMILY` and
  `RECORDED_WHOLE_LOAD_RESERVE_BYTES` (the model and the reserve of the INT8
  exception's one record) and `GUARDED_BY_BYTES`. The Worker's reserve
  reaches the one price function at the quote, at both walls and at the
  in-place choice (`in_place_fallback.worker_reserve`, read from ComfyUI's
  own `extra_reserved_memory`); a price that is not told it charges the
  share. A LoRA in a form the mapping does not know (LoHa, LoKr, GLoRA, OFT,
  BOFT) reads as unmapped and is charged for every weight of its
  checkpoint, the dearer side.
  The mapping from a
  LoRA's keys to the weights it patches knows ComfyUI's spellings of a LoRA
  pair (`.lora.down.weight` and four more were missing, so one Qwen Image
  LoRA was counted as 1933 patched weights where the Worker counts 240).
  `StockFit` carries the term, so the Worker's two walls and the quote the
  driver reads charge one figure by one function, and every refusal prints
  it. The partial-load route's room rule charges the same term for the layers
  still to bake, by each layer's own format (2 GiB, and for each layer 7.5 MiB
  where it is scaled FP8 or full precision, 4 MiB where it is INT8 ConvRot,
  nothing where it is plain FP8), without the bytes guard and without the
  INT8 exception: its BF16 cell ran on hardware under the rule without the
  term. Charged by the line for every format, 371 INT8 layers still off the
  device were asked 8.4 GiB with 8.1 there (2026-11-12), the route was
  declined, and the render ran from cast-time weights with ComfyUI's backup
  kept, to a low of 5.97 GiB; by format the ask is 7.2. The default
  (`lora_low_rss` on) and FSDP,
  which takes no classic bake, are charged nothing new. The loader node's own
  preflight runs before any LoRA node and cannot know the stack.
- **The fallback** (`actor/in_place_fallback.py`). Baking in place on the
  loaded base is the default's own path and leaves almost none of that
  memory. So where the classic price, credited what dropping the slot's
  resident returns, does not fit, and the in-place price does (one copy of
  the stack and 2 GiB of working memory), the store bakes in place. The term
  and the fallback belong to a change that ends in a fresh load. A change
  ComfyUI's classic hot swap makes in place (the held stack's backup covers
  the new one, as for a strength change, or the other slot holds a model)
  has neither and keeps its price: such a change grew the Worker by 0.8 GiB
  on the 2026-11-05 hardware leg. So the store asks after the hot swap has
  declined and before it drops anything. The quote reads the same fact the
  hot swap acts on, for every format: whether ComfyUI's backup on the slot
  holds every weight the new stack patches (the held stack's own files, or
  for another file the weights its headers map to), and whether the other
  slot holds a model. It prices by the same arguments, so both pick the same
  bake. Both read "the slot holds this base" by the store's one comparison
  (`in_place_fallback.holds_base`): the file name and the loader options
  together. A request that keeps the file and changes an option (the weight
  dtype, for one) is a fresh load, and is priced as one. A slot whose
  checkpoint file has changed on disk since its load is set aside by the
  store: base and stack load fresh and no in-place bake is offered, and the
  quote prices that. A request whose key is the slot's own is quoted a reuse
  only while the checkpoint and the stack's LoRA files still carry the
  fingerprints the slot recorded, which is the store's own rule for a reuse.
  A LoRA file rewritten under its own name makes the same request a stack
  change on the held base, priced as any other, with ComfyUI's backup
  credited only where it still covers the stack. For the second model of a dual-model render the bake is chosen against the
  reading less the first model's charge, as the store chooses it beside the
  loaded first model, and the first slot is counted as held, as the store
  finds it. Three places still work out a second time what the store
  decides, and are known duplications until one function serves both sides.
  The pair's budget is the reading less the first model's peak price and not
  what its load leaves, which errs toward a refusal. The quote matches the
  slab-retry flag's base by file name where the store compares the whole key;
  no classic bake is in play there, since slab residency needs `lora_low_rss`
  on. And `in_place_fallback.loads_fresh` restates the gates of
  `store_bake.copies_off_the_gpu` (the other slot, stock residency, placed
  bytes) and does not share them. Where the slot holds the
  base bare the bake runs on it. Where it holds anything else (nothing,
  another model, this base with a stack of either kind) the slot is dropped,
  the base loads bare at the stock price and the bake runs on that: a stack
  is never baked on top of another, and a base counts as bare only when
  ComfyUI's backup of it is empty too. The baked slot keeps no backup and no
  un-bake record, so the next stack change reloads the base. One line says
  the fallback was taken, and the quote's measured block says which bake was
  priced. A refusal at the Worker's own wall comes after the slot is
  dropped, as with any refused fresh load.
- **The permission** (`nodes/in_place_bake_permission.py`). The in-place bake
  is another bake path, and a memory rule must not move a combination onto a
  path the identity gate has not passed. The partial-load route
  (`actor/lora_partial_load.py`) is not such a path and needs no gate: it
  loads the rest of the model through ComfyUI's own loader on the model's
  own patcher, so every key is baked by the function a whole classic load
  uses, and what changes is residency (the backup is freed, the slot is
  marked baked). The in-place bake restores and re-bakes each key by the
  pack's own protocol, and whether that equals ComfyUI's bake for a
  combination is what the gate proves. A graph that turns the switch off
  runs no ceremony, so the driver sends a permission and never a prohibition:
  the sample request carries it only when the gate ledger holds an exact PASS,
  and no FAIL, for the combination with the in-place bake on (the pair's key
  for a dual-model render, as the gate's own grant reads it). The PASS must
  sit under this graph's other worker arguments unchanged, so a graph whose
  Init node differs in more than the switch from the one that earned it gets
  none. It rides the
  per-dispatch request and not the worker arguments, which the gate's
  capability context holds whole. Absent means no fallback, on every path
  that does not set it: the classic price decides alone, and the refusal
  names both prices and both cures (free memory, or render the combination
  once with `lora_low_rss` on so the gate can prove it). The quote reads the
  same permission, so the fleet is not admitted to a change a Worker would
  refuse. A slot baked in place is not reused by a request with the switch
  off and no permission; it is dropped and loaded the classic way, and the
  quote prices it as that load. Only a sample dispatch carries the
  permission: a loader node or the scheduler node that loads a model with
  its stack keeps the classic bake.

Run on hardware on 2026-11-12 as first built (the line and the files for
every format): the refusal with no permission, the gate's PASS, the in-place
bake with the classic latent on a held base, on an empty slot and on the
pair, the classic bake admitted at its price, and the fleet's own refusal
where neither fits. The charge by format and by reading is built on the
second table, whose rows are that day's three hardware blocks. What no row
holds: a whole INT8 Wan-Animate 2 load at a reserve other than 8 GiB or at
a clip other than the template's; any other INT8 ConvRot model's whole load
read against ComfyUI's condition, or a second one loaded whole with many
patched keys; the second plain FP8 stack's whole load at its price's edge
(ComfyUI placed it partly at a reading 3.0 GiB above the price, so its one
whole load on record is on a roomy box); a full-precision stack with a low
read on a held base; a scaled FP8 file with weights over 64 MiB; and a
pair's rank loading the INT8 template whole on a tight box.

**No price knows the clip.** Every figure above is worked at the loader, before
a latent exists. With a classic bake's backup held, what a render takes grows
with the clip: the sampler's memory follows the latent, and it outruns
ComfyUI's own estimate by more as the latent grows (0.5 GiB at 81 frames, 2.6
at 241). ComfyUI's whole-load condition leaves only its reserve for that, so
the low at the tightest reading that still loads the model whole falls with
the clip: about 6 to 7 GiB at the shipped 81 frames and about 4.4 projected at
241, from one roomy render of 2026-11-12 that took 52.4 GiB where 81 frames
takes 39. Three shipped templates run this path as shipped (the Wan-Animate 2
template and its BF16 and memory-saver variants, with `lora_low_rss` off and a
LoRA), all at 81 frames. `lora_low_rss` on keeps no backup and is not on it.
The same is so on the last release, which priced this load lower. The Worker
now gives that backup up before the first step where a whole load leaves less
room than the sampler may ask (`lora_partial_load.free_backup_for_sampling` on
one GPU; on a pair the ranks together, `backup_agreement`; section 5.5), which
takes the projected low at the 241-frame edge from 4.4 to about 10 by the
route's record of the same step (5.8 GiB returned before the trim was added;
not yet read on hardware for a whole load). A pair's rank at its whole-load
edge read 4.29 GiB on the first Spark on 2026-11-13 with the backup kept; the
pair's agreement and its room check are the answer to that reading and have no
hardware reading of their own yet. What stays as above: a load placed partly
whose route could not run.

**Freed host memory goes back at once.** torch's wheels allocate host memory
through a bundled mimalloc, which gives freed pages back to the box only at a
purge, and a purge runs only when the thread that owns them allocates again
after the purge delay. A stock load frees its host copy as it places weights,
the last of those frees come just before the load returns, and nothing
allocates between them and the next price, so no positive delay can fire in
that window. torch 2.15.0 carries mimalloc 2.5.2, which purges far less than
the 2.2.6 in 2.12.0 even on later allocations: with flux1-dev placed, the head
actor's anonymous memory was 15.0 and 16.1 GiB on 2.15.0 against 2.5 GiB on
2.12.0 with the same source and ComfyUI (2026-11-05 and 2026-11-06), and the
actor held 17.8 GiB of resident memory after the Kandinsky 5 Pro video file's
placement and through the gate's drop, so the ceremony's re-price came up 6.9
GiB short on a box with room. The
worker loop therefore sets `MIMALLOC_PURGE_DELAY=0` in the environment every
cluster actor inherits, the driver's launch does the same for a local actor
(`cli/worker_process_env.HOST_ALLOCATOR_PURGE_ENV`; a value the loop or the
driver inherits stands, and the unit `dgxm update` installs starts from an
empty environment), and `scripts/comfy-driver.sh` exports the same for the
driver, so a free on the thread that allocated the memory gives it back at
once. That setting is for torch's own allocator. The weights of a quantized
file are the loader's own tensors, and a probe found them in the C library's
per-thread arenas (2026-11-13, section 5.5): there, from a process's second
load of a file on, the freed pages of blocks under 32 MiB come back only at a
trim. The code trims after the free of a backup for a sampler and after the
store drops a slot (the hard clear among them), and nowhere else in this
release: a load's placement, a released LoRA stack and a soft clear are not
followed by one, so a Worker's anonymous memory can rest about 10 GiB above a
fresh Worker's after its second load of the INT8 Wan-Animate 2 file, until a
drop, a hard clear or a recycle. A small block freed from another thread still waits for that thread's
next allocations. The cost is about 20 percent on a loop that does nothing but
allocate and free host memory (0.44 s against 0.36 s for ten rounds of 16
blocks of 128 MiB); the release test on this setting measures what renders pay.
Each actor logs the value it started with in its `UMA posture` line.

**A model in the way offers a clear, not a credit.** A refusal priced over a
Worker resident names the old model and its box: "the old model (flux1-dev)
is still loaded on the head". The driver then works out what a clear of the
attached mesh, the same teardown as Reset attached mesh, gives back on each
short box: the file bytes of each stock resident, whose GPU copy an in-process
drop returned on 2026-11-04, plus the actor's anonymous memory above what it
held at its first load (`actor/actor_footprint.py`). A load cast to another
weight dtype is credited a lower bound of the copy its cast built (below):
on 2026-11-05 the store's drop of a 26.6 GiB bf16-cast Wan 2.2 DiT raised the
head from 58.7 to 88.5 GiB, and from 57.9 to 88.5 GiB on a warm re-queue. Where
the price already credited the slot's own resident (the swap credit below), a
clear is weighed against the uncredited price, the row's figure plus its
credit, because a cleared box holds no copy to drop, and the refusal says that
figure is the whole price. When that resident is the only old model on a short
box, the card gives the memory earlier loads left behind there ("the head
still holds about 12.0 GiB that earlier loads left behind") rather than the model, which
the price already dropped; a refusal no clear can fix leaves that sentence out. A slab or comfy-managed resident
and an FSDP shard are credited the arena only. A slot the request reuses, and a model this graph's loader
nodes load, are neither old nor counted in what a clear gives back: a fresh fleet would only load them
again. One slot is the exception, the LoRA case of the swap credit: the slot a
change prices as a swap when its resident is the very file the change asks for
(`old_model.changes_in_place`). That copy is not reused as it is, and the host
copy of the weights its stack patches is what holds the memory: on 2026-11-05 a
BF16 Wan-Animate 2 actor kept about 34 GiB of anonymous memory after loading
base and LoRA, and a strength change was refused 14.3 GiB short, while a fresh
head reads about 95 GiB against the 69.1 GiB a fresh load needs. That change
now reuses the backup and is credited it (below); a LoRA the held stack lacks
still meets this card. That slot is
weighed like any credited one, and the card names the model asked for: "The
model loaded earlier (wan_animate_2_bf16) holds about 29.6 GiB on the head that
this change needs." Only a row the price credited the slot's own stock copy
counts: a slab, comfy-managed or FSDP resident of the same file keeps the plain
rule. The rule reads the file the agreement names, which at the sampler is the
cond model's, so an uncond row counts only when it holds that same file. A model an earlier sampler of the same graph loaded is
still old, since
the later sampler fits on a fresh fleet, so the card and the log say "loaded
earlier", not "an earlier render". Only when
every short box is short by less than that, with 2 GiB to spare for what a
fresh Worker takes back, does the refusal become "Old model in the way"
(`old_model.py`), at the agreement and at the loader windows. The loader asks
the Workers on the driver host for the same facts after its own estimate
refused, and a clear must make room in both of its windows, against their
unclamped budgets. The card's Clear and retry grants nothing: it writes no
memo and no ledger row and lifts no guard, so its kind stays outside the frozen
audit vocabulary. A card resets only the fleet that refused, and every reset,
from any door, answers every open card. With Always clear and retry ticked
(`auto_clear` in the consent memo) or `DGXM_ALLOW_CLEAR_RETRY` set on the
driver, the driver clears by itself, at most once per load attempt, under the
reset route's single-flight gate and never under a Pipeline or Gate session
pinned to its mesh, waits for the driver host's MemAvailable to settle,
attaches a fresh fleet and prices again. That price decides: a load that still
does not fit refuses in the plain words above. A model the load does not
replace, in another slot, is never credited, because nothing drops it. The Worker's own wall prices after its in-process drop, so it
names no old model; when the arena it still holds would cover its shortfall,
its refusal says so and names Reset attached mesh.

**A refusal names a box the way the user does.** The box the driver runs on is
"the head". With one other box, that box is "the sibling"; with more, each is
"the sibling at <host name>", so no two read alike. A fleet on the driver's box
alone, a local fleet or a cluster.toml that lists only that box, is "this box",
and a box that runs more than one rank names its GPU too ("GPU 1 on the head").
The head is the `[[hosts]]` entry that resolves to the driver's own machine, the
check rank 0's co-residency already makes, never the first entry by position;
when that check cannot resolve the driver's entry, every box reads "the sibling
at <host name>". One module words it (`box_names.py`). The agreement stamps the
name on every rank's answer, and the refusals under its guard (its own, and the
undeliverable-actor refusals that end its wait, `restart_reattach.py`), the Old
model in the way card, the agreement's log line, the FSDP clean-reload
refusal and the Fleet timeline take their words from it. So none of those
names a box by its rank number or by the address cluster.toml gives it. Machine fields keep the host
name, the address and the rank number for operators and parsers: the attach
trace, ledger rows, and the warning a reply that names another rank leaves in
the log.

**A Worker's own refusal says "here", and the driver names the box.** A
Worker words no sentence about a peer and does not know which box the driver
runs on, so its walls say where in its own voice: "stock residency cannot load
X here", "the LoRA bake for X cannot proceed here", "model X was not found
here", "sampler X is not registered with ComfyUI here". The driver knows which
box each rank sits on, and it names the box where it receives the error, before
any other reader sees it (`worker_box.py`, at `mesh_rpc.call_all_bound`,
`mesh_lease.collect_sample` and the Fleet drain): "On the sibling, stock
residency cannot load X here: ...", after the tag, with the words from
`box_names.py`. A fleet on the driver's box alone gets no prefix, since "here"
is already the box the user reads about. A Fleet job takes the box of the actor
the driver sent it to. At a collective call of more than one rank, Monarch's
wrapper says nothing of which rank raised, so the Worker adds one machine line
naming its rank (`actor/failure.mark_rank`), which the driver reads and takes
off, so no user reads the number. The line is no tag, a Worker one release
behind sends none and keeps its host name, and the partial-load refusal, which
every rank raises alike and which speaks for the fleet, gets none. The label
never rides the setup key.

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

**A swap over a stock resident is priced after its drop.** The quote reads a
request whose key differs from the slot's resident as a swap. When that
resident is stock, the store never holds both copies: it keeps the resident and
re-applies a stack in place (the LoRA case below), or it drops the slot before
it loads (`actor/store_load.fresh_load`), and an in-process drop of a stock
resident returns its GPU copy (2026-11-04, and the readings in the paragraph
on a model in the way). So the stock price at the agreement credits that copy,
which the Worker reports beside the slot's residency
(`capacity_lora_bake.credit_stock_swap`, fed by
`actor/worker_capacity._resident_bytes`), and still charges the new load's host
copy and the floor. A credit is never more than the drop returns, or the fleet
admits what a rank's own ladder refuses after its drop: an uncast resident is
credited its file, a cast one each float tensor at the smaller of its stored
and its cast size (ComfyUI builds fp32 auxiliaries at the cast's width on a
copying load, and a widening cast is unmeasured), and a cast one whose header
cannot be read nothing. The price's own cast figure is an upper bound and is
never a credit, and the sentences say the drop "frees at least" the credit. The
quote endpoint runs outside the GPU lock, so its figures are bound to the
request key the snapshot publishes for each slot: a slot whose resident changed
between the two reads gets no credit and no old-model figures. The arena a stock load retains is not credited, nor is a
slab, comfy-managed or FSDP resident, a slab rung's own price, or a slot whose
load or cleanup is in flight: those still price as held. Until 2026-11-05 every
swap priced as held, charging a same-slot swap its old resident twice; once a
cast was priced, `dgx-monarch-wan22-t2v`, whose two stages swap in the cond
slot, refused its stage 2 at 58.7 GiB available against 60.9 GiB, where the
hardware had rendered it with 39.9 GiB left at its lowest. Credited, stage 2
needs 34.3 GiB. The Worker's own ladder prices after its drop, so it never
needs the credit.

**The driver host's own prices count the same drop.** The loader preflight and
the render-site checks that charge a pending DiT load (the driver footprint,
the MiniMax H3 activation price, the reference and pose price) read the driver
host's MemAvailable, where the head's Worker shares the box. Until 2026-11-11
they priced the new DiT beside the model its slot was about to drop, so a user
who changed model in one session met the Old model in the way card with room
on the box: flux1-dev held and then LTX 2.3 needed 80.8 GiB against 67.5
available with 22.2 GiB about to be dropped, and MiniMax-H3-Image INT8 after
BF16 needed 46.0 GiB against 34.1 with the 37.5 GiB BF16 DiT in the same slot.
Now a price that does not fit asks the Workers on the driver host the
agreement's own quote and counts the copy the load's own slot drops first
(`swap_credit.py`, `nodes/loader_relief.py`). It counts only what the
Worker's own price credited, the row that names the drop and its bytes, so the
driver never admits on a drop the agreement would not count: a stock resident
only, never a slab-priced row, a reuse or a LoRA change that keeps its base.
The rule under the credit is one sentence: no credit where the drop copies
anything to the host before it frees the old model. A drop through ComfyUI's
global unload did that in two cases. The store pins only a whole load to its
device, so ComfyUI's detach copied the device weights of a model it held only
partly (`model_lowvram`) to the host. And only the dropped slot was pinned, so
a model the other slot held was unloaded to the host with it. Both copies are
waste on unified memory, where an offload frees nothing, and both are gone
there for a stock resident (`actor/resident_ledger.unload_dropped`; a discrete
GPU keeps the global unload, since an offload frees device memory). The drop
runs under the actor's GPU lock as before and touches fewer models than the
global unload did, so nothing a running render needs is moved. It takes only
the dropped model's records out of ComfyUI's loaded list: a whole load pinned
and unloaded by ComfyUI's own `model_unload`, a partly loaded one by the step
ComfyUI takes when it replaces a clone (`detach(unpatch_all=False)`), which
moves no weight. The slot that stays is left loaded, and it is pinned to its
device as slab and shard bases already are, so the unload ComfyUI runs at the
next load that does not use it moves nothing either (with smart memory off, as
on a Worker, `load_models_gpu` unloads every other model). Its next render
therefore moves no weight. This holds within one setup. A request whose
topology, attention kernel or `sync_ulysses` differs from the live setup's
rebuilds the process groups first, and that teardown unloads every model in
both slots on every rank
([TROUBLESHOOTING #33](TROUBLESHOOTING.md#33-a-live-topology-or-residency-policy-change-times-out-while-retiring-a-large-resident)), so
the drop then meets an empty store. After a single-model render in between, ComfyUI
loads the pinned model again in place; a LoRA stack in hot-swap mode is
applied again then, and its backup sits on the device from there on. ComfyUI prices the next load of an unloaded model at its whole size
whether its weights went to the host or stayed pinned, and reads free memory
live, so the pin changes no whole-or-partial load decision. The Worker reports
where its drop copies nothing (`drop_copy_free` in the slot's snapshot entry,
from the same predicate the drop runs on), and only there does the credit
stand beside a partly loaded resident or a held other slot. It does not where
the other slot is partly loaded or comfy-managed (the drop is the targeted one
there too, so that slot's host copy moves from the drop to the next load that
does not use it, unpriced there as it was at the drop), on a discrete GPU, for a
slab or shard build, which keep the global unload their owners close after,
or under `DGXM_DISABLE_COPY_FREE_DROP=1`, which restores the earlier drop. The
store-wide unload (`unload_all`, the clear) keeps the global unload too, since
ComfyUI must hold no model after it. A
resident with no credit is priced as held, at the agreement and here, and the
card's clear replaces it. The tests run the drop on ComfyUI's real classes on
CPU (the copy-free drop seam of the canary); the dip itself, the memory that
returns and the kept slot's next render are hardware legs.
It counts only the slot the load goes into, and only onto a price that charges
the DiT's own load: one that reads the weights as resident (the loader's memo;
an explicit preset at the render-site footprint and H3 prices) has no drop to
count. In a dual-model graph each of these prices charges one DiT alone, and
the old models, priced as held, used to hide that. So the credit for one slot
is taken less what the other slot's load adds to the host: that load's file
less what its own Worker's price credits for the model it drops, read from one
more quote that carries both loads as the sample-time agreement asks them
(`swap_credit.growth`). Where that leaves nothing, or the other load cannot be
read from the graph or priced, nothing is credited. A rank answers from the
topology it last held, so under an explicit preset other than cfg2, where
every rank loads both models, a rank that sends no row for the other slot
counts as no answer. A load that fits
asks no Worker. The credit is spent only against memory that is back before
it is needed, and that depends on when the Worker drops. An eager load drops
at the loader node, before the driver builds what it has not yet built. A
deferred load drops at the first render, after the driver has built the
graph's text encoder and VAEs beside the old model, so those models must fit
on the reading as it is, with no credit. Under `auto` the load is deferred.
Under an explicit preset it is eager unless an identity ceremony or the
dm-cfg2 split defers it (`nodes/loaders.py`), which is decided after the
price, on the live fleet; so there a credited price must hold both ways: the
load is priced as one that waits, and the driver models still to build must
fit beside the old model with no credit. The other slot's load is read from
the nodes ComfyUI will run (`graph_live.live_nodes`, the walk the graph price
uses), so a loader behind a lazy switch's unselected branch takes no room and
cancels no credit. The credit is tried before any card and before the
driver unloads ComfyUI's models, a refusal that still stands says what was
counted, and the card is weighed against the uncredited price as at the
agreement. A credited load forgets the dropped model's loader memo, so that
model is priced when it comes back. Nothing here clears a Worker: the store
drops the slot as it did before, and the agreement then prices the load on
every rank with its own credit.

When the credit does not make room, the loader also unloads the models ComfyUI
holds on the driver host, as the render-site checks do before they refuse, and
prices the whole load again from a fresh reading with the graph's driver
models charged whole. It does so once a load, never while a render is running
in this driver, and not under `DGXM_DISABLE_DRIVER_UNLOAD=1`. That switch
leaves the credit on at both sites, since the credit is a price and not a
rescue; `DGXM_DISABLE_SWAP_CREDIT=1` turns the credit off at both. The loader's
price already counts held models against the graph's own, so the unload makes
room only by what ComfyUI held beyond them, another graph's larger encoder;
in the two records above it would have added nothing, and the credit alone
makes the room. Where ComfyUI holds no model the same step still returns the
allocator's cached blocks, and the load is priced again if the reading rose
by a tenth of a GiB or more. The driver's ComfyUI keeps that cache for longer
since its free-memory reading follows MemAvailable
(`driver_uma_memory.install_driver`, above), and on the driver host the kept
blocks count against the Worker's load.

**A LoRA change on a held base is the same credit.** When the resident is a
stock copy of the same file and the store would keep it and re-apply the new
stack in place (`actor/store_bake.swaps_in_place`, published in the store
snapshot), the load places no second copy of the file. So the stock price
credits the file's own copy and still charges the host copy of the weights the
stack patches and the floor (`capacity_lora_bake.credit_hot_swap`). When the
in-place swap would first copy the model off the GPU, the store drops the copy
and loads base and stack fresh instead (section 5.5). That reload is the swap
above, so the credit is the slot's own copy either way: over a cast resident
it is the lower bound the Worker reports, never the price's cast figure, and
nothing when the Worker reports none. A rank whose drop returned less refuses
at its own wall before it reads a byte
([TROUBLESHOOTING #94](TROUBLESHOOTING.md#94-a-multi-rank-render-refuses-saying-a-peer-rank-could-not-load)).
The host copy a held stack keeps (ComfyUI's backup) is credited too, as a lower
bound, for a change to a stack of the held stack's own files: a strength, an
order, a LoRA removed. ComfyUI keeps each weight a stack patches, as it was, on
the host; every clone of the base shares that copy, and an in-place change puts
those weights back from it and places again only what it lacks. For the BF16
Wan-Animate 2 base and its distill LoRA the headers put the backup at the file:
1261 of its 1303 tensors, 32,785,804,928 bytes, and the 42 left out hold 3.8
MiB. Each in-place change also reads the new stack's LoRA files again, through
the pread backend into anonymous memory (`actor/pread_backend.py`). On
2026-11-05, on one Spark, a strength change on that base grew the actor by 0.8
GiB, about the 0.69 GiB LoRA file and the patch temporaries, took 1.7 s of
`gpu_load` and rendered a fresh load's latent bit for bit, where the agreement
charged it 38.6 GiB and, on the pair's head at 24.5 GiB, raised Old model in
the way. An in-place swap runs unpriced on the Worker, so for such a change the
agreement's price is the only guard, and the backup is reused, not dropped, so
this credit rests on that measured swap, not on the drop rule above. The Worker
reports the bytes of the backup's host tensors
(`actor/worker_capacity._backup_bytes`) only for that state: classic mode, a
stock, uncast, full-precision resident placed whole, the other slot empty, and
a held stack whose every patched weight the backup holds. The price credits
them up to the host copy it charges, net of the new stack's own files, and
credits no backup when those files cannot be sized or cost as much
(`capacity_lora_bake.held_backup_credit`,
`capacity_lora_bake.stack_file_bytes`): 1.1 times the file less the backup,
plus the files and the floor, about 8.7 GiB there, where a stack of three LoRAs
of 2.97 GiB prices 11.0 GiB. A LoRA the held stack lacks gets no backup credit,
covered or not: the quote cannot read ComfyUI's key map without loading the
file, and when its keys are not covered the store reloads, which keeps the
backup through the fresh load (the actor rose from 33.9 to 64.8 GiB). The
credit carries both copies, so a clear is still weighed against the price a
cleared box pays. A base the
low-RSS bake consumed with no un-bake record reloads after the store drops it,
so the swap credit above covers it with the same figure; explicit
`slab_weights=on` drops the stock resident and loads slab, whose price takes
no credit. On 2026-11-04 an explicit-preset
Wan-Animate 2 graph loaded its bare BF16 base at the UNET Loader and then asked
for the base and its LoRA: priced whole, the change needed 69.1 GiB against
61.8 GiB on the head, and credited it needs 38.6 GiB. The INT8 file took the
same path: the hot swap took about 16 GiB of the head's free memory against
22.1 GiB priced this way, and the one-step sample about 10 GiB more. The
Worker's own ladder never applies the credit, because `store_residency.resolve`
prices with no store snapshot, so a change the store cannot make in place falls
back to a fresh load that is priced whole there.

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
schedule against comfy master and the newest release tag; the signature tier is
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
| LoRA bake | `actor/lora_cast_time`, `actor/model_store` per-key bake | `bake_equivalence_canary.py` |
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
   - `dgxm update`: plain update refuses while ComfyUI runs or when any Worker
     runs from a release slot. It fetches, checks the fetched commit's exact
     torchmonarch pin and dependencies, and fast-forwards only after that check
     passes. It then installs the pin and dgx-monarch with `--no-deps` and, on
     a cluster, the pin on each Worker host, restarts the Workers and runs
     doctor. It never resolves torch/NCCL/xfuser.
     `--verify` selects the certainty-aware remote-cluster transaction split
     across inspection, remote release-slot, staged driver-pin, smoke, receipt,
     and state-machine leaves; its operational contract is canonical in
     [INSTALL.md](INSTALL.md#updating).
   - `dgxm top`: live TUI; `dgxm gate`: identity-gate the last prompt;
     `dgxm install-service`: refuses a unit it does not own, syncs the package,
     then installs and starts each host's Worker service as a systemd user
     unit, and fails a host whose user lingering is off.
   - `dgxm diagnostics`: read-only capture of the driver's existing
     `/dgxm/telemetry` endpoint into a new private JSON file named by
     `--output`, bounded by duration, event count and bytes. It never attaches,
     creates an admin listener or stops a Worker. A partial capture exits 3,
     and `--capabilities` reports the runtime's limits without connecting.
     Usage is canonical in [CLUSTER.md](CLUSTER.md#monarchdiagnostics).
   - `dgxm version`: prints the dgx-monarch version and its torchmonarch pin.
   - `dgxm launcher`: separately owned local CLI launcher installation, offered
     during onboarding. It plans by default, binds the selected executable and
     config, preserves unmanaged commands, and makes no Worker or dependency
     changes. Ordinary-shell PATH and dashboard endpoint verification belong
     to the launch handoff.
   - `dgxm sol-install`: stages per-host optional package plans and binds apply
     to their fleet digest under host lifecycle locks. Existing core runtime
     versions and files remain protected; other existing dependency changes
     require explicit names. `dgxm sol-smoke` separately runs the opt-in GPU
     dense control after passive idle checks, without creating a mesh.
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

On its own, the pack writes into ComfyUI's input folder in one place, at
import, and only the demo inputs its shipped templates name
(`example_workflows/input`, drawn by `tools/gen_demo_inputs.py`).
`demo_inputs.py` copies each one into that folder when its name is absent
there. It never creates the folder. It creates each file exclusively, never
replaces, truncates or follows a file, link or folder already there, removes
a copy it could not finish, logs every name it placed, and never raises into
ComfyUI's startup; a copy that stops part way logs one WARNING that names
what it placed. A demo file a user deletes returns at the next start.
`DGXM_DEMO_INPUTS=off` turns the copy off. pytest imports the entrypoint as
the root package before any fixture runs, so the test battery assigns that
switch when its conftest loads, and the entrypoint canary assigns it before it
runs the entrypoint, each whatever the shell exports. The tests of the copy
point it at scratch folders. So none writes into a shared ComfyUI tree. Two
other paths put files in that folder at a person's request, and this rule
does not cover them: a Studio upload, through ComfyUI's own upload route, and
the sweep's test media, when an operator runs `benchmark/sweep/media.py`.

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
Native latent RDMA remains default-off; §5.6 defines the selected-rail
qualification boundary and links the current evidence.

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
  hosts, and Gate authority refuses a pair whose torch or CUDA builds differ
  (section 5.2).
- torchmonarch is pinned and bumped with a changelog entry per bump; the
  developer skill documents the introspection-first upgrade process.
- xFuser is pinned to the local `0.7.0+dgxm.npuimport1` compatibility wheel
  until an official release replaces it (section 3.1, rule 3).
- ComfyUI: track nightly (project policy), with a CI canary job that runs the
  node-pack import and a CPU-only graph-build daily against comfy master and
  the newest release tag. The release entry changes only when ComfyUI tags a
  release, so a red master beside a green release indicates an upstream
  change, and a red release on an unchanged tag indicates a change in this
  repository.
- Templates follow master's widget order. A stock widget that ComfyUI
  inserts mid-node, and that a ComfyUI release this project supports still
  lacks, is a row in `stock_widget_drift.py`: the widget, the widget it
  follows, and the value every template holds, its default. The row leaves
  when the support floor passes the release that added the widget. A reader
  that maps `widgets_values` by a running ComfyUI's order re-reads with the
  row only when the values outrun that order, and drops the widget only at
  that value; the frontend extension restores the node by name. The canary
  requires that value on every ComfyUI it runs against.

---

## 7. Repository layout & maintainability

This is a selected map of the files that own a contract, not a full file
inventory. Add a file here when it takes ownership of a lifecycle,
compatibility, or release boundary that a maintainer must be able to find.

```
dgx-monarch/
├── LICENSE                    # Apache-2.0
├── NOTICE                     # copyright line, vendored-kernel notice
├── README.md                  # §8
├── pyproject.toml             # package metadata + pins, dgxm entrypoint
├── src/dgx_monarch/
│   ├── nodes/                 # thin nodes + setup binding
│   │   ├── abandoned_heal.py  # the #87 abandoned-lease reset, shared with Fleet
│   │   ├── action_security.py # same-origin checks for browser driver actions
│   │   ├── execution_event_routes.py # GET /dgxm/execution-events: every read
│   │   │                      #   renews the observer lease
│   │   ├── fleet_cleanup.py   # cancellation-safe Fleet resource retirement
│   │   ├── fleet_jobs.py      # Fleet job-list assembly and its per-prompt limit
│   │   ├── gate_capacity_stop.py # a ceremony's stop when its request cannot fit
│   │   ├── gate_ceremony.py   # one ceremony's render legs, frozen as evidence
│   │   ├── gate_process_state.py # process-lifetime first-use Gate state
│   │   ├── gate_process_verdicts.py # one process-verdict publication and its reads
│   │   ├── gate_provenance.py # mandatory all-rank Gate source bracket
│   │   ├── gate_verdict.py    # verdict derivation, lever quarantine, publication
│   │   ├── metrics_route.py   # bounded finite Prometheus projection
│   │   ├── model_patch_preflight.py # driver refusals for the per-render skip,
│   │   │                      #   context-window, shift and LoRA patches,
│   │   │                      #   the H3 image latent's shift, and
│   │   │                      #   MOSS-Image reference images and text bias
│   │   ├── h3_context_windows.py # the node that carries ComfyUI's MiniMax H3
│   │   │                      #   context windows to the Workers
│   │   ├── pending.py         # terminal sample-result ownership and cleanup
│   │   ├── pipeline.py        # bounded FIFO submission/abort orchestration
│   │   ├── recycle_guard.py   # reset single-flight/rate ownership past HTTP timeout
│   │   ├── render_session.py  # node-side names for the mesh_session.py API
│   │   ├── resolved_topology.py # what auto read and chose, for telemetry
│   │   ├── sampler_wire.py    # sampler output merge and stock noise rebuild
│   │   ├── strict_json.py     # non-finite-safe public JSON normalization
│   │   └── setup_binding.py   # follow-up RPC topology/token reassertion
│   ├── actor/                 # GPUWorker, model store, failure decorators
│   │   ├── comfy_custom_nodes.py # headless custom-node discovery/cleanup
│   │   ├── context_room.py    # free memory for the Worker's CUDA context
│   │   │                      #   before its first CUDA call, and the refusal
│   │   ├── pread_backend.py   # lazy safetensors backend shim
│   │   ├── load_profile_file.py # bounded per-host directory of Init
│   │   │                      #   load_profile reports
│   │   ├── capacity_quote.py  # one rank's price for one slot, as data: the
│   │   │                      #   ladder's arithmetic and the wire row it fills
│   │   ├── rescue_offer.py    # capacity rescue offer parts, shared by the
│   │   │                      #   ladder that raises one and the quote row
│   │   ├── rank_readiness.py  # the post-load readiness flag every whole-model
│   │   │                      #   topology crosses before the first collective
│   │   ├── sample_protocol.py # the GPUWorker sample body `_sample_impl` runs
│   │   ├── slab_wrap.py       # minimal DLPack view of slab-owned byte ranges
│   │   ├── resident_ledger.py # tells comfy what a slab or a shard already
│   │   │                      #   holds, so its sample-time load stops asking;
│   │   │                      #   and drops a slot with no host copy
│   │   ├── store_load.py      # fresh-load path of ensure: ladder, load, adoption
│   │   ├── store_load_ownership.py # exact fresh-load resource/slot transaction
│   │   ├── store_slab_admit.py # the slab capacity check, its card text, and
│   │   │                      #   the sentence printed when neither fits
│   │   ├── stored_model.py    # resident-slot resource identity and close order
│   │   ├── worker_capacity.py # the capacity_quote endpoint body: this rank's
│   │   │                      #   slots, one snapshot, one running budget
│   │   ├── worker_compile.py  # opt-in DiT block compilation boundary
│   │   ├── guidance_render.py # Skip Layer Guidance on the render clone, and
│   │   │                      #   its Worker refusal
│   │   ├── context_windows_render.py # MiniMax H3 context windows on the render
│   │   │                      #   clone, and the Worker refusals in the load guard
│   │   ├── model_sampling_render.py # the SD3 or MiniMax H3 shift node on the
│   │   │                      #   render clone, and the H3 shift's Worker refusals
│   │   ├── lora_shape_guard.py # the refusal of a LoRA patch that changes a
│   │   │                      #   weight's shape, before any bake
│   │   ├── lora_partial_load.py # a one-GPU partial load with a LoRA stack: the
│   │   │                      #   low-RSS bake taken for that load, or the
│   │   │                      #   refusal where an FP8 layer would differ
│   │   ├── in_place_fallback.py # a classic LoRA bake that does not fit is
│   │   │                      #   baked in place on the held base, when the
│   │   │                      #   driver sent the gate's permission (§5.9)
│   │   ├── vae_decode.py      # the vae_decode endpoint and the one resident
│   │   │                      #   decode VAE an actor may hold (§5.6)
│   │   └── rdma_handoff.py    # durable worker descriptor ownership + ACK
│   ├── adapters/              # §5.3 registry (one file per family)
│   │   ├── flux_shared.py     # control-span/shard helpers the flux forwards
│   │   │                      #   share with hunyuan, lens, and qwen_image
│   │   ├── attention_inputs.py # what ComfyUI hands a model-level attention
│   │   │                      #   op, and the closed keyword set it accepts
│   │   ├── stock_attention.py # stock attention left in place under sequence
│   │   │                      #   parallelism, run without an attention
│   │   │                      #   module's kernel pin
│   │   ├── index_cache.py     # bounded per-device cache of the Ulysses pad
│   │   │                      #   path's index tensors
│   │   ├── flux3_packing.py   # FLUX.3 Klein span clipping, stream shards, and
│   │   │                      #   pad rows
│   │   ├── moss_image_seams.py # MOSS-Image window cut, block-patch vetting,
│   │   │                      #   fused-norm route and attention ledger
│   │   └── minimax_h3_packing.py # MiniMax H3 per-token packing and checkpoint
│   │                          #   timestep embeddings
│   ├── kernels/               # proven-exact kernel replacements (§4)
│   │   ├── int8_gemm.py       # the int8_gemm option: knob, install points,
│   │   │                      #   fallbacks and their log lines
│   │   └── int8_gemm_triton.py # MIT, vendored (§3.1): the Triton INT8 GEMM,
│   │                          #   its proxy and the import hook
│   ├── fp8_compute.py         # [worker_args] fp8_matrix_mult on each actor
│   │                          #   build, and the `FP8 compute:` line a wrapper
│   │                          #   on comfy.ops.pick_operations logs per build
│   ├── demo_inputs.py         # copies the templates' demo inputs into ComfyUI's
│   │                          #   input folder where a name is absent (§6.1)
│   ├── comfy_surface.py       # runtime Comfy touchpoint/signature assertion
│   ├── stock_widget_drift.py  # stock widgets older supported ComfyUI
│   │                          #   releases lack, and what templates hold
│   ├── comfy_forward_contracts.py # aggregate/re-export forward compatibility map
│   ├── comfy_rebound_signatures.py # exact stock rebound-method contracts
│   ├── comfy_rebound_sites.py # declared static self.bind/Adapter.bind inventory
│   ├── comfy_rebound_validation.py # exact signature/requiredness comparator
│   ├── topology.py            # dataclasses + auto-mode resolution
│   ├── topology_table.py      # the auto-mode table rows (data)
│   ├── topology_cfg_evidence.py # dated evidence each cfg row needs
│   ├── skip_layer_guidance.py # Skip Layer Guidance wire, activity test and
│   │                          #   refusal reasons (stdlib leaf)
│   ├── h3_context_windows.py  # MiniMax H3 context-window wire, refusal reasons,
│   │                          #   window schedule and row count (stdlib leaf)
│   ├── h3_sigma_shift.py      # the MiniMax H3 shift node's refusal reasons and
│   │                          #   the one-frame image latent it guards (stdlib leaf)
│   ├── error_utils.py         # total diagnostics + cancellation-first composition
│   ├── retry_policy.py        # bounded retry outcomes without stale diagnostics
│   ├── actor_lifetime.py      # actor-side reaper (stdlib leaf; watcher installed
│   │                          #   at bootstrap, lease trigger armed by the client)
│   ├── client_lease.py        # driver-side lease renewer for live fleets
│   ├── mesh.py                # attach/spawn/shutdown, registry lock + callback
│   ├── mesh_attach.py         # transport enable, the pre-dial id gate, heal
│   │                          #   and the in-process retry
│   ├── worker_proc_id.py      # per-start Worker proc id: derive, dial, refuse
│   ├── worker_runtime.py      # each rank's torch/CUDA build against the
│   │                          #   driver's, the rule Gate authority holds
│   ├── delivery_faults.py     # the fault hook's delivery failures and the price
│   │                          #   wait that ends on one (stdlib leaf)
│   ├── restart_reattach.py    # the render's re-attach after every Worker loop
│   │                          #   restarted: id check, teardown, refusals
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
│   ├── mesh_setup_transition.py # worker-policy broadcast and process-group
│   │                          #   setup behind MeshHandle's methods
│   ├── mesh_safety.py         # lifecycle classification + pure safety policy
│   ├── mesh_teardown.py       # stop authority and supervision reconciliation
│   ├── monarch_semantics.py   # stdlib-only torchmonarch behavior probes
│   ├── monarch_touchpoint_spec.py # touchpoint record + signature comparison
│   │                          #   (stdlib leaf; keeps the manifest importable)
│   ├── monarch_policy.py      # validated Monarch process policy: installed
│   │                          #   before the runtime import, then read back
│   ├── monarch_policy_io.py   # policy, trust-fingerprint and driver-host
│   │                          #   readers that import neither Monarch nor Torch
│   ├── monarch_launch.py      # explicit process launch policy that keeps the
│   │                          #   controller's environment out
│   ├── monarch_capabilities.py # path-redacted Monarch artifact and native-load
│   │                          #   facts, without hardware probes
│   ├── monarch_diagnostics.py # opt-in Monarch metrics on a bounded daemon
│   │                          #   queue; no Jobs, servers or spans
│   ├── absorb_fire.py         # one greppable INFO line per teardown-absorption
│   │                          #   site (stdlib leaf; used by soak diagnostics)
│   ├── fault_repeats.py       # report an unhandled cluster fault once, then
│   │                          #   count its exact repeats (stdlib leaf)
│   ├── attach_trace.py        # one greppable line per cluster-attach phase:
│   │                          #   the vocabulary, the timer, and the parser
│   │                          #   the TROUBLESHOOTING #101 grep depends on
│   ├── attention_pins.py      # per-module kernel pins: one walk per load, the
│   │                          #   render record fields, the class-K waiver
│   ├── capacity_agreement.py  # driver-mediated pre-load fan out: every rank
│   │                          #   prices before any rank allocates
│   ├── capacity_memory.py     # one /proc/meminfo parser and what it may claim
│   ├── consent_observe.py     # worker refusal descriptors as consent-panel cards
│   ├── driver_footprint.py    # driver-side footprint policy (no torch or
│   │                          #   monarch import; render-path pending-load refusal)
│   ├── driver_unload.py       # free ComfyUI's driver models, then the actors'
│   │                          #   allocator caches, before a render-site
│   │                          #   capacity refusal, checking again after each
│   ├── swap_credit.py         # the copy a load's own slot drops first, counted
│   │                          #   at the driver host's prices from the Worker's
│   │                          #   own credit (nodes/loader_relief.py asks)
│   ├── driver_uma_memory.py   # on unified memory, ComfyUI's free-memory reading
│   │                          #   in the driver is MemAvailable less a floor (§5.9)
│   ├── vae_encode_item.py     # on unified memory, the driver encodes a batch
│   │                          #   of images one image a call (§5.9)
│   ├── capacity_classic_bake.py # what a LoRA stack's classic bake adds to the
│   │                          #   stock price, and the in-place bake's own
│   │                          #   price (§5.9)
│   ├── vae_decode_relief.py   # on unified memory, before a VAE decode ComfyUI
│   │                          #   prices above its free memory the driver has it
│   │                          #   free that much from its text encoders (§5.9)
│   ├── vae_decode_route.py    # Worker-assisted VAE decode (§5.6): the knob,
│   │                          #   the keyed stock-decode wrappers and the split
│   ├── vae_decode_plan.py     # unit plans read off ComfyUI's own loops, and
│   │                          #   their verified replays
│   ├── vae_decode_schedule.py # the two-ended unit board: the driver takes the
│   │                          #   front, the ranks the back (stdlib leaf)
│   ├── vae_decode_families.py # which VAEs split, through which stock decode,
│   │                          #   and why the rest do not
│   ├── vae_decode_env.py      # what a Worker's decode shares with the
│   │                          #   driver's: compared, or adopted for the decode
│   ├── vae_decode_remote.py   # the per-unit RPC, its wire shape and the VAE's
│   │                          #   file identity
│   ├── vae_decode_telemetry.py # one INFO line per decode, the fallback
│   │                          #   WARNING and the last record
│   ├── fsdp_proof_scope.py    # FSDP clean-reload scope + the topology test
│   │                          #   that selects it, below either caller
│   ├── operator_actions.py    # inert, safety-classed readiness remediation catalog
│   ├── operator_activity.py   # additive health/activity/maintenance presentation
│   ├── operator_profiles.py   # compile-time safe/balanced/advanced authoring policy
│   ├── operator_readiness.py  # canonical three-layer readiness normalizer
│   ├── rails.py               # RDMA rail byte counters and link state from
│   │                          #   sysfs (stdlib leaf; telemetry reads both)
│   ├── telemetry_events.py    # bounded strict-JSON event-tail projection
│   ├── telemetry_fleet.py     # the last Fleet wave's per-job rows: rank, host,
│   │                          #   box, duration, prompt digest, driver-clock start
│   │                          #   and end, and a marker when another box's host answers
│   ├── telemetry_steps.py     # per-render step marks: each render times a step
│   │                          #   against its own last one (stdlib leaf)
│   ├── telemetry_ceremony.py  # Gate ceremony windows and the first-render
│   │                          #   claim for the progress tracker (stdlib leaf)
│   ├── runtime_diagnostics.py # bounded advisory phase timings shared by the
│   │                          #   driver and actors (stdlib leaf)
│   ├── runtime_timing.py      # advisory queue-wait and GPU-phase timers; the
│   │                          #   asyncio lock and executor behave as before
│   ├── diagnostic_session.py  # finite sanitized dgxm diagnostics captures and
│   │                          #   their private export
│   ├── execution_events.py    # opt-in prompt-bound observation journal and its
│   │                          #   observer lease; never runtime authority
│   ├── runtime_observer.py    # render, Fleet and RDMA hooks that publish to
│   │                          #   that journal only while a lease holds
│   ├── finite_json.py         # dependency-free non-finite-to-null JSON
│   │                          #   normalization for the driver and companion
│   ├── studio_resource_validation.py # data-only check of packaged Studio files,
│   │                          #   run by setup, update and Worker cleanup
│   ├── studio/                # optional Studio companion and its optional
│   │                          #   user unit, with the compiled bundle
│   │                          #   (frontend source in studio/); see
│   │                          #   docs/STUDIO.md for its contracts
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
│   ├── native_return_latch.py # driver latch: messaging after a failed native
│   │                          #   read, cleared only by a driver restart
│   ├── occupancy_metrics.py   # opt-in occupancy up-down counters, sampled on
│   │                          #   the existing metrics thread (stdlib leaf)
│   ├── return_phases.py       # one native return's phase split for the result
│   │                          #   row, the render log line and the qualifier
│   ├── return_diagnostics.py  # observational return counters; never evidence
│   │                          #   for ACK, release or reuse
│   ├── tui/readiness_view.py  # pure canonical readiness renderer for the TUI
│   ├── tui/diagnostics_view.py # fixed, bounded display of recent driver timings
│   ├── tui/view_helpers.py    # dashboard rendering/recording helpers
│   └── cli/                   # dgxm command adapters and lifecycle leaves
│       ├── arguments.py        # public parser; command effects stay in leaves
│       ├── cluster_smoke.py    # owned DP/NCCL/source-cohort smoke + typed cleanup
│       ├── config_removal.py   # checked lock-bound durable config unlink
│       ├── diagnostics_command.py # dgxm diagnostics: bounded read-only capture
│       │                       #   of the driver's telemetry endpoint
│       ├── doctor_nccl.py      # loaded NCCL library probe and the cross-host row
│       ├── doctor_permissions.py # config-mode diagnostic row
│       ├── doctor_repair.py    # exact-file chmod-only repair transaction
│       ├── doctor_worker_context.py # the host's owned Worker release for
│       │                       #   Doctor's environment probe
│       ├── gate_last.py        # pure `gate --last` rewrite of one chosen
│       │                       #   history entry, shared with Studio
│       ├── launcher.py         # managed local command ownership and publication
│       ├── comfy_top.py        # foreground driver and owned dashboard supervisor
│       ├── comfy_top_process.py # bounded owned-process signal and settlement helpers
│       ├── comfy_top_signal.py # owned foreground signal observation and startup gate
│       ├── legacy_init.py      # compatibility interactive config writer
│       ├── legacy_update.py    # plain update: fetch, dependency check,
│       │                       #   fast-forward, pin install, restart
│       ├── legacy_update_dependencies.py # plain update's candidate dependency
│       │                       #   check before the fast-forward
│       ├── release_slot.py     # read-only release-slot probe for Doctor and plain update
│       ├── lifecycle.py        # Worker service lifecycle and package/pin effects
│       ├── lifecycle_generation.py # Worker generation marker/fence capabilities
│       ├── lifecycle_host.py   # host certainty, locality, and listener proofs
│       ├── lifecycle_lock.py   # shared per-user service-mutation serialization
│       ├── lifecycle_scripts.py # lock-held Worker start/stop payload builders
│       ├── lifecycle_systemd.py # canonical legacy unit + ownership proof
│       ├── listener_generation.py # which process holds a loop's LISTEN socket,
│       │                       #   how long it has held it, the id it serves,
│       │                       #   and the doctor row
│       ├── operator_receipt.py # bounded disclosure-safe receipt schema/writer
│       ├── receipt_sanitize.py # conservative receipt-note redaction leaf
│       ├── sol_attn_prerequisites.py # passive optional dependency/header/cache evidence
│       ├── sol_install.py      # per-host plans and reviewed fleet digest
│       ├── sol_install_payload.py # isolated wheel staging and protected package apply
│       ├── sol_staging.py      # removes the private Sol staging of abandoned
│       │                       #   Studio reviews on one host, under the lifecycle lock
│       ├── sol_install_guard.py # bounded passive maintenance idle guard
│       ├── sol_install_inventory.py # installed metadata and proven editable companions
│       ├── sol_install_protected_links.py # read-only protected link and target evidence
│       ├── sol_install_requirements.py # built-wheel dependency checks before mutation
│       ├── sol_smoke.py        # explicit model-free GPU dense kernel control
│       ├── setup_apply.py      # confirmed lock-held setup mutation/settlement
│       ├── setup_cli.py        # strict public setup parsing/presentation boundary
│       ├── setup_command.py    # plan/confirmation/config-lock coordinator
│       ├── setup_config_carry.py # operator tables a re-apply keeps from the existing file
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
│       ├── studio_command.py   # dgxm studio: config-derived launch, the Studio
│       │                       #   user unit, tunnel line
│       ├── update_activity.py  # read-only driver activity and opt-in
│       │                       #   privileged process checks
│       ├── update_cancellation.py # cancellation-precedence helper
│       ├── update_command.py   # concrete verified-update platform adapter
│       ├── update_dependency_contract.py # read-only dependency closure checks
│       │                       #   run in each target interpreter
│       ├── update_dependency_diagnostics.py # bounded local dependency remedies,
│       │                       #   never copied into receipts
│       ├── update_doctor_diagnostics.py # bounded Doctor categories for updater
│       │                       #   stderr
│       ├── update_driver_pin.py # driver-pin change: stage both wheels, switch or restore offline
│       ├── update_entrypoint.py # serialized execution and receipt publication
│       ├── update_inspect.py   # exact target dependency/source inventory checks
│       ├── update_lock.py      # per-checkout flock: one driver-side update at a time
│       ├── update_pin_diagnostic.py # bounded local ownership errors, separate from receipts
│       ├── update_receipt.py   # bounded update step-to-receipt adapter
│       ├── update_repository.py # read-only checkout and origin observations
│       │                       #   the update controllers share
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
│       ├── vae_decode_row.py   # doctor row: the decode knob and the driver's
│       │                       #   last VAE decode, read over HTTP
│       ├── driver_memory_row.py # doctor row: which free-memory reading the
│       │                       #   running driver has, from its /system_stats,
│       │                       #   and what it says of its encode rule
│       ├── update_worktree.py  # owned detached-worktree cleanup
│       └── update_worker_attestation.py # exact remote source/pin-payload proof
├── web/js/dgx_monarch_readiness_model.js # pure fail-closed browser normalizer
├── tests/canary/              # Comfy and torchmonarch surface canaries
├── tools/check_artifacts.py   # offline manifest check, and the only writer of
│                              #   example_workflows/artifacts.toml
├── tools/gen_studio_inventory.py # the only writer of
│                              #   src/dgx_monarch/studio/resource_hashes.py,
│                              #   the Studio data files and
│                              #   web/js/dgx_monarch_presentation.js; --check verifies
├── tools/fp8_paths.py        # ComfyUI's operations pick for a checkpoint, read
│                              #   from its header; the comfy canary checks the
│                              #   reproduced rule against comfy master
├── tools/gen_dgxm_top_image.py # renders docs/media/dgxm-top.svg from a
│                              #   synthetic recording
├── tools/gen_demo_inputs.py   # draws example_workflows/input from code
├── example_workflows/artifacts.toml # the model files each template needs
│                              #   staged: models subfolder, source, size, sha256
├── example_workflows/input/   # the demo images, clips and music the templates'
│                              #   input nodes name
├── benchmark/                 # matrix and sweep harnesses + quality gates
├── docs/                      # DESIGN.md (this), INSTALL, CLUSTER, MODELS,
│                              #   TROUBLESHOOTING, VALIDATION, ADAPTERS
├── skills/                    # §6.2
└── .github/workflows/         # lint+unit (x86 CPU), daily comfy canary on
                               #   master and the newest release tag, weekly
                               #   torchmonarch canary
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
latent-return posture and its qualification boundary in §5.6 F3, attach timeout and wedge
recovery in [docs/TROUBLESHOOTING.md #1](TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout) and [#2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach), and the attached-mesh self-heal
in [docs/TROUBLESHOOTING.md #87](TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

Line-rule exception ledger: `CEILINGS` in `tests/line_limit_helpers.py` lists
every module above the 500-line default cap with its ceiling.
`test_module_line_limit_exception_ledger`
(`tests/test_surface_remediation.py`) fails when the set of modules over 500
lines differs from that list or a module passes its ceiling, and
`test_design_ceiling_table_matches_the_ledger` beside it holds the table below
to that list. A ceiling
includes no allowance for future additions. Changing the set or a ceiling
requires a reviewed ledger edit; new behavior should prefer an extracted
module. An exception is not permission to fold a focused lifecycle, policy,
validation, or compatibility module back into the listed file. The table below
mirrors the ledger by hand; no test reads it, and its reasons are context, not
an enforced requirement. The current exceptions are:

| module | ceiling | cohesive reason |
|---|---:|---|
| `mesh.py` | 690 | driver mesh lifecycle, actor dispatch, teardown, and recovery orchestration; deliberate-recycle outcomes live in `mesh_recycle.py`, creation publication in `mesh_creation.py`, worker-policy broadcast and process-group setup in `mesh_setup_transition.py`, behavior probes in `monarch_semantics.py`, and diagnostic composition in `error_utils.py` |
| `cli/doctor.py` | 765 | user-facing preflight plus the hardware failure catalogue |
| `cli/actor_reaper.py` | 603 | procfs actor identity, orphan provenance, the termination sequence, the doctor report, and the read-only liveness answer the driver asks after a stop times out |
| `adoption_evidence.py` | 732 | opt-in, request-bound resident-adoption authority, full-file identity/cache, and strict worker/result validation; the render path retains only narrow integration hooks |
| `actor/unbake.py` | 692 | byte-verified checkpoint capture/restore engine |
| `config.py` | 638 | typed cluster schema, fabric profiles, validation, and TOML rendering; strict table-key vocabulary lives in `config_schema.py` |
| `actor/sampling.py` | 620 | worker sampler protocol and distributed execution paths; per-adapter data-parallel exemptions keep raw token/weight sequences intact across ranks |
| `actor/comfy_bridge.py` | 531 | Comfy worker bootstrap and narrowly-scoped runtime patching; DynamicVRAM state, latches, and policy live in `actor/comfy_dynamic.py`, the pread shim lives in `pread_backend.py`, and the INT8 GEMM option lives in `kernels/int8_gemm.py`, which bootstrap configures before ComfyUI imports comfy-kitchen |
| `actor/capacity_quote.py` | 539 | residency memory estimates and the response fields parsed by the driver, including slab LoRA bake memory and default LoRA-path selection |
| `capacity_agreement.py` | 567 | the driver's collection, its one pure decision function and the fleet card it composes |
| `actor/worker.py` | 509 | ordered checks and adapter installation for each freshly loaded base in `_inject_for_topology` |
| `actor/model_store.py` | 511 | the store's one `ensure`: reuse, the two swaps, the in-place fallback's single call before any drop, and the fresh load, in the order they must be tried |
| `gate_ledger.py` | 527 | the ledger's protocol number and its burn history live together, so the record of why a version was retired cannot drift from the number itself |
| `comfy_surface.py` | 533 | one executable inventory of every stock ComfyUI touchpoint, deliberately flat so a failing canary identifies the changed API |
| `adapters/fsdp.py` | 589 | the FSDP launch contract in one place: admitted kinds and profiles, the live precision rescan, the wrap with ignored islands and quantized shards, the streaming shard build hand-off, and the shard-build price |
| `nodes/gate_identity.py` | 583 | canonical normal/Fleet capability construction, proof-scope binding, authorization, and typed denial |
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
| Native latent RDMA | Scoped selected-rail checks; one-step at-threshold sampling | Disabled by default. Section 5.6 and [Validation](VALIDATION.md#native-rdma) separate CPU-staged protocol checks, one-step sampling and untested recovery paths. Messaging or NCCL results do not qualify native return. |
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
| ComfyUI nightly breakage | daily canary against Comfy master and the newest release tag; bind to model classes; wrap stock nodes instead of copying them (§5.7) |
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
3. **Recovery retires the whole fleet.** A supervision failure evicts the
   fleet, and the next Init or render brings up a fresh one once the old
   processes are confirmed gone; until then the attach refuses with its
   remedy. An endpoint error leaves the live fleet and its warm state in
   place. A Worker loop that stayed down longer than Monarch's message
   delivery timeout still needs a fresh driver (§5.1).
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

- Earlier measurements found cfg-parallel winning at or below about 1 MP and
  USP above about 1.5 MP (Krea2 and Chroma), so the `topology_table.py` rows
  that split by resolution switched at 1.2 MP. On 2026-10-14, after the 2026-10
  sequence parallel fixes, Campaign K measured cfg2 bit-identical to one GPU
  and faster than uly2 at every measured size of each new band: Chroma BF16,
  FP8 and INT8 from 1.2 to 1.64 MP (row 19), Qwen-Image FP8 from 1.2 to
  1.77 MP (row 68), Krea2 FP8 and BF16 from 1.2 to 2.36 MP (row 15), and
  Boogu FP8 and Anima BF16 below 1.2 MP (rows 79 and 47; since 2026-11-07
  these and Chroma's rows 20 and 19 take only a prompt pair whose text rows
  differ, and row 47 was withdrawn on 2026-11-10). Each upper band
  ends at the largest size measured; the two lower bands follow rows 20 and 28
  and rest on one size, 1.05 MP. Krea2 INT8 cfg2 was exact but ran 0.97x to
  1.08x against uly2, so it kept Ulysses; LongCat cfg2 above 1.2 MP was not
  bit-identical, and Kandinsky5's image and video checkpoints share one
  family, so neither gained a row (BENCHMARKS.md).
- FLUX.3 Klein Base cfg2, measured by Campaign K's method on 2026-10-15 and
  2026-10-16 with the first Spark capped at 2100 MHz, beat uly2 at every size
  from 1024 to 1536 square and was exact when the prompt encoded past one
  80-row block. On a one-block prompt one GPU folds cond and uncond into one
  batch-2 call and cfg2 did not match it, so FLUX.3 has no cfg2 row
  (VALIDATION.md).
- MOSS-Image cfg2 refuses on the shipped graphs, whose Skip Layer Guidance
  runs the two passes as separate calls. With the skip off, measured on
  2026-10-21 with the first Spark capped at 2100 MHz, cfg2 was exact and
  faster than uly2 on a pair of different lengths at 1024 square, and read
  one-step NRMS 0.023 to 0.026 on a pair of equal length, so MOSS-Image has
  no cfg2 row (VALIDATION.md).
- Ideogram4 head_dim 256, FP8 pair at 1024x1024: Ulysses2 matched
  DP2 at NRMS 0.000; Ring2 measured 0.102 against the 0.100 limit.
- Sage attention helps only fp8 with USP at high resolution (about 5-8% at
  1536 in the 2026-06 campaign) and loses on bf16 at 1024. Campaign K
  (2026-10-14) read the SAGE route 1 to 5 percent faster than Torch Flash
  uly2 on Krea2 and Chroma FP8, at 1-step NRMS 0.030 to 0.078 against one
  GPU, where Torch Flash uly2 was bit-identical.
- FSDP is a capacity tool, not a speed tool; NCCL_PROTO=LL can deadlock FSDP,
  so never set it.
- GB10 has GPUDirect RDMA off (`cuMemGdrSupport 0`), so NCCL stages through
  host memory: about 17 GiB/s busbw for a 2-rank 256 MiB all-reduce inside
  Monarch actors.
- On unified memory the page cache competes with GPU allocations, so doctor
  warns and suggests drop_caches. ComfyUI's own planner also read free memory
  there; the pack corrects that reading in its actors and in the driver
  (§5.9, "ComfyUI in the driver reads the box as it is"), and doctor's
  `driver memory reading` row says which reading the running driver has, and
  whether it encodes a batch of images one image a call.
- Monarch bring-up failures map to preflight checks and troubleshooting entries.
- `scripts/setup_env.sh` documents the aarch64 torchaudio constraint, which
  applies only to ComfyUI before v0.38.0.

Measurement scope and limitations are recorded in [VALIDATION.md](VALIDATION.md).
