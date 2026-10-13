# cluster.toml reference

One file describes the whole cluster. Search order:

1. the Init node's `config_path` widget / `dgxm --config`
2. `$DGXM_CLUSTER_TOML`
3. `./cluster.toml`
4. `~/.config/dgx-monarch/cluster.toml`

If an explicit path does not exist, loading fails rather than falling back to
local mode. A commented example ships as `cluster.example.toml`. `dgxm setup`
writes the file from a plan you review first; `dgxm init` is the legacy
interactive writer.

In cluster mode the Init node checks a fingerprint of the file's bytes on
every Queue, and any byte change, a comment included, retires the old fleet
before attaching a replacement. An edit the driver refuses keeps the old fleet
attached, and reverting the edit reuses it. In local mode the Init node does
not fingerprint the file, and a local mesh reads only `[worker_args]`. An edit
there does not reach a live local mesh on the next Queue; it applies when the
Init node runs again (a changed widget or a new ComfyUI process). If the old
processes are not confirmed stopped, replacement is blocked to prevent two
fleets from holding models or rendezvous ports at once.

Unknown keys fail config loading at the root, in `[cluster]`, `[[hosts]]`,
`[worker_args]`, and every `[fabric.<name>]` table, including unselected
profiles. This catches misspelled security and transport settings before attach.

## Guided setup profiles

[CONCEPTS.md](CONCEPTS.md#operator-words) defines the profile names; this table
shows the settings each profile writes. Setup checks the hardware and treats
missing information as unknown:

| setup choice | settings written |
|---|---|
| `safe` | native latent RDMA off, and stock residency (`lora_low_rss = false`, `slab_weights = false`) on every hardware class |
| `balanced` (default) | the `safe` residency settings on discrete, mixed, or unidentified hardware; on a positively identified homogeneous UMA fleet it removes both residency keys, so the worker's model-aware defaults decide |
| `advanced` | only on a positively identified homogeneous UMA fleet: `lora_low_rss = true` and `slab_weights = true`; mixed, discrete, and unknown fleets receive a typed refusal |

Every choice also recommends topology `auto`, `auto_gate=first_use` and
pipeline depth 1 for the graph; setup shows these and never writes them into
`cluster.toml`. It never writes `operator_profile`, `comfy_managed`,
`compile_dit` or `load_profile`, or the profile name. Review the rendered diff to see every setting that will change.

All three choices write `auto_heal = true` and `rdma_latent_return = false`
(see [`[cluster]`](#cluster)). A dual-Spark render still runs: NCCL carries the
split and actor messages return the latent.

In cluster mode the identity gate's capability context binds a fingerprint of
the settings that can change math, residency or transport
([DESIGN.md](DESIGN.md) §5.2). Setup shows the old-to-new diff and warns
whenever applying it changes the file. Only a changed fingerprint makes
contextual PASS and INCONCLUSIVE rows ineligible until the gate checks the new
configuration. A sticky FAIL stays in force: no profile change or config rewrite
clears it.

## `[cluster]`

| key | default | meaning |
|---|---|---|
| `client_bind` | none (**required** for any cluster-mode attach) | Monarch client bind on the driver's fabric IP, e.g. `tcp://192.0.2.11:0` (an RFC 5737 documentation address; use your own). It is required because the default transport advertises whatever the hostname resolves to, which is loopback on many boxes, and the attach then fails with `MESH_ATTACH_CONFIG_TIMEOUT` ([TROUBLESHOOTING #1](TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout)). |
| `fabric_profile` | `single-node` | named NCCL/GLOO/UCX env set, see [Fabric profiles](#fabric-profiles). A multi-host file that omits this key or selects `single-node` is rejected, and a name with no shipped profile needs its own `[fabric.<name>]` table. |
| `transport_security` | none | `"trusted_fabric"` or absent; any other value fails the load. See [Worker transport security](#worker-transport-security). |
| `nccl_master_addr` | first host's IP | literal-IP rendezvous address for the data plane |
| `nccl_master_port` | `29777` | rendezvous port |
| `python` | `python3` | interpreter with torchmonarch and the ComfyUI dependencies, at the same path on every host |
| `comfy_dir` | auto-detect | ComfyUI root; per-host override via `hosts[].comfy_dir` |
| `rdma_latent_return` | `false` | leader latents return via one-sided RDMA read, size-gated by `rdma_min_bytes`. Keep it off by default; selected-rail qualification is limited to the [recorded scope](VALIDATION.md#native-rdma), with native cancellation and fabric-failure recovery untested; [DESIGN.md](DESIGN.md) §5.6 and [TROUBLESHOOTING #75](TROUBLESHOOTING.md#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails) are canonical. |
| `rdma_min_bytes` | `8388608` (8 MiB) | RDMA size gate: latents smaller than this return via messaging, not RDMA. |
| `ssh_key` | agent default | key for `dgxm` lifecycle commands |
| `auto_heal` | `true` | Before each attach, restart every Worker service when a probe sees one down; a host the probe cannot reach stops the attach instead. After the last attach attempt fails, restart them so the next ComfyUI session starts clean. The one in-process attach retry does not depend on this key ([TROUBLESHOOTING #1](TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout), [#2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)). Set `false` to leave every restart to you. |

## `[monarch]` (optional)

The 0.7.0 integration validates these settings before runtime initialization.
`startup_timeout_s` defaults to `180` and accepts integers from `30` through
`3600`. After host attachment completes, ProcMesh initialization and actor
readiness share this observation budget. Host attachment and its configuration
push retain their separate existing timeouts and retry policy. A timeout is
not permission to reuse an unsettled mesh or restart a persistent Worker.
When the budget expires in cluster mode, the driver stops the processes it
spawned but keeps its transport poisoned, so every later queue refuses until
ComfyUI restarts. Raise `startup_timeout_s` before that restart if the budget
was too short. The edit does not touch the Worker units: only the driver reads
this budget.

### `[monarch.rdma]`

These settings configure the native runtime independently of NCCL's fabric
profile. They do not enable latent RDMA: `cluster.rdma_latent_return` remains
`false`. [Selected-rail checks](VALIDATION.md#native-rdma) qualify only their
recorded workload/runtime/hardware; positive-step qualification is limited to the recorded at-threshold case.

| key | default | meaning |
|---|---|---|
| `target` | `""` | `nic:<device>`, `cpu:<numa>` or `gpu:<ordinal>`; empty uses upstream selection. A device name uses letters, digits, `_`, `.` and `-`, does not start with `-` or `.` and does not contain `..`; a number has no leading zeros. Native return needs a `nic:<device>` target for every participant ([DESIGN.md](DESIGN.md) §5.6). |
| `host_targets` | `{}` | Table mapping configured host names to `nic:<device>` overrides; any other value, empty included, fails config loading. Use each host's actual device name; names need not match between machines. A host without an entry uses `target`. |
| `peer_device_affinity` | `"match_name"` | `any`, `match_name`, or two or more disjoint groups, as in `groups:dev0,dev1|dev2,dev3`. Affinity describes allowed NIC pairings; it does not discover reachable subnets. |
| `max_nics_per_buffer` | `1` | Maximum registered NICs per buffer, `1..8`. An explicit NIC target pins the upstream manager to that NIC. |
| `qps_per_peer` | `1` | Native QPs per NIC route, `1..8`. |
| `min_stripe_size_kb` | `512` | Native minimum stripe size, `4..65536` KiB. |
| `application_chunks` | `1` | dgx-monarch's application transfer chunks, `1..8`; separate from native QP striping. |
| `allow_tcp_fallback` | `false` | Permit upstream TCP fallback. A fallback transfer never establishes native-RDMA acceptance. |
| `qp_init_timeout_s` | `30` | Native queue-pair initialization timeout, `1..300` seconds. |

NIC count × QPs × application chunks may not exceed `32`. Start qualification
with the defaults and an explicit NIC. Increase one dimension at a time only
after measuring the selected rail. Upstream's H100/GB200 defaults do not
establish a GB10 performance gain.

`benchmark/rdma_return_bench.py` provides a source-bound CPU staged-return
qualifier. Its `--preflight-only` mode gathers passive source/process facts
without importing Torch or Monarch. An authorized active run names the exact
commit and source manifest, configuration, return mode, and a new private
output path. `--driver-host` is optional; when given, it must name the host the
driver resolves to (below), or the run refuses:

```bash
python -B benchmark/rdma_return_bench.py \
  --expect-sha FULL_COMMIT --expect-manifest SOURCE_MANIFEST \
  --config cluster.toml --driver-host CONFIGURED_HOST_NAME \
  --return-mode message --output /existing/private-directory/control.json
```

Messaging control requires native returns disabled. The separate native arm
requires explicit NIC targets on every host and no TCP fallback. Each policy
variation uses a fresh process. The qualifier owns its CPU actor processes and
tests threshold boundaries, reattachment and ownership settlement; it does not
replace the full sampling, cancellation or render-fidelity gates. An ambiguous
native result remains HOLD with backing retained; do not wrap it in an external
timeout that kills its owner.

The policy is resolved per host before Monarch imports it, checked against
native configuration, and bound into transport identity. Each Worker and actor
uses its own `[[hosts]]` entry. The driver uses the entry whose `address` IP
equals the `client_bind` IP, the rule the qualifier also applies; when none
matches, as on a driver box without a Worker, it uses `target` and logs one
WARNING. With native return on, the driver checks every host's resolved target
before it starts any process. A failed native read latches native return off
until ComfyUI restarts, so later setups use messaging
([TROUBLESHOOTING #115](TROUBLESHOOTING.md#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)).
Conflicting ambient
environment variables are refused. A transport-policy change requires a fresh
process; it cannot change an already initialized RDMA manager safely. Preserve
Worker ownership and follow the reviewed update/restart procedure in
[Changing policy after install](#changing-policy-after-install).

### `[monarch.diagnostics]`

`metrics = false` is the default. Enabling it publishes supported counters and
histograms through Monarch; gauges and observable instruments are unavailable
in this release. Local bounded phase timings remain advisory and never prove
transfer completion, cleanup or readiness.

`otlp_endpoint = ""` disables export. A configured HTTP(S) endpoint must have no
credentials, query, fragment or path other than a bare trailing `/` (the
exporter appends each signal's path, such as `/v1/metrics`) and requires
**`otlp_logs = true`** explicitly:
Monarch 0.7 exports upstream raw logs as well as metrics to this destination.
Those logs may contain more than dgx-monarch's sanitized metric labels. Use an
operator-controlled collector and review disclosure requirements before
opting in.

Monarch fixes its export destination when its native runtime first loads.
ComfyUI's driver loads it when the first mesh is built, after installing the
endpoint, and actors and Workers receive the endpoint before they start. When
the driver cannot start a requested metrics or OTLP export, the mesh build
refuses with a typed error that names the reason, before any process starts.
`otlp_restart_required` means Monarch loaded before the endpoint was set, which
happens when you add or change the endpoint after a mesh was built: restart
ComfyUI. `ambient_otlp_endpoint_conflict` means `OTEL_EXPORTER_OTLP_ENDPOINT` in
the ComfyUI environment names another endpoint: make `otlp_endpoint` match it,
or unset the variable and restart ComfyUI. Removing a configured endpoint also
needs a restart. An actor or Worker that cannot start a requested export logs
an ERROR with the reason and keeps running. `/dgxm/telemetry` reports the
driver's export state as `monarch_diagnostics`, beside the requested policy in
`monarch_policy`.

`dgxm diagnostics --capabilities` explains the exact runtime's diagnostic
limits. A finite capture reads existing telemetry only:

```bash
dgxm diagnostics --host 127.0.0.1:8188 --duration-s 30 --output /absolute/private/capture.json
```

The destination directory must already exist; the command creates a new
mode-0600 file and refuses overwrites and symlinks. Captures are bounded by
duration, event count and byte count; incomplete coverage stays explicit.
Attached-cluster admin, SQL and process-stack capture
remain unavailable because the upstream admin API does not return a stoppable
owned handle. The integration does not manufacture a Job around persistent
Workers to obtain those features.

### Changing policy after install

Each managed Worker unit carries `cluster.rdma_latent_return` and the policy
resolved from these tables, except `startup_timeout_s`. After you edit a setting
the unit carries, `dgxm restart` refuses with `POLICY_STALE` before it stops any
Worker, `dgxm up` refuses each stale host and names the cause, and
`dgxm doctor` marks the host's `worker policy` row `STALE`. When restart gets no
answer from a host's check, as after an SSH transport error, it keeps the old
order: it stops every Worker, and `dgxm up` then refuses the stale host. Stop
ComfyUI, then run `dgxm update --verify --target HEAD`: a verified update of
the installed commit renders the units from the current file, restarts the
Workers and keeps its rollback. [TROUBLESHOOTING #113](TROUBLESHOOTING.md#113-lifecycle-commands-refuse-with-policy_stale-after-a-policy-edit)
has the details.

## `[worker_args]` (optional)

ComfyUI settings applied inside every actor process. The graph's
Init-node values take precedence. The Init node always sends `mmap_fallback`
and `swap_verify`, so those two keys here have no effect on a graph. Loading
the file rejects unknown keys, and both the driver and
the actor reject a wrong type, such as the string `"off"` for a boolean. On
unified memory (a device that reports itself integrated) the worker fills in
the tested Spark defaults: `disable_pinned_memory`, `disable_async_offload`,
`disable_smart_memory` and `lora_low_rss` default on, `reserve_vram_gb` to
8.0, `slab_weights` to `"auto"`, and `safetensors_backend` to `pread` unless
`mmap_fallback` is set. A key you set wins, with two exceptions:
`reserve_vram_gb = 0` still becomes 8.0, and `slab_weights = true` beside
`lora_low_rss = false` falls back to off with a warning. Discrete GPUs keep
stock ComfyUI defaults. Allowed keys are:

* booleans: `disable_pinned_memory`, `disable_smart_memory`,
  `disable_async_offload`, `disable_custom_nodes`, `mmap_fallback`,
  `lora_low_rss`, `load_profile`, `compile_dit`, `comfy_managed`,
  `shared_act_scale` and `log_act_scale`;
* bounded numbers: `reserve_vram_gb` / `uma_reserve_gb` (0..1024),
  `swap_verify` (-1..64 integer) and `fsdp_prefetch_depth` (1..8 integer;
  above 1, FSDP prefetches that many blocks ahead, paid in reserved memory);
* `slab_weights` (boolean or `"auto"`) and `safetensors_backend` (`"pread"` or
  `"mmap"`).

The UMA guidance behind these values is in docs/VALIDATION.md.

`shared_act_scale` and `log_act_scale` cover the one activation scale a sharded
group shares for nvfp4 (docs/MODELS.md, [docs/TROUBLESHOOTING.md #95](TROUBLESHOOTING.md#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales)). The hook
is on by default, so `shared_act_scale = false` is the only setting worth
writing, and it belongs here rather than in a shell export because it decides
whether a rank issues a collective and every rank must agree. `log_act_scale =
true` traces one line per wrapped layer per rank for the first step. They set
`DGXM_SHARED_ACT_SCALE` and `DGXM_LOG_ACT_SCALE` in the worker.
Removing a key restores its default. If the key was never set, an environment
variable set by hand on a host stays in effect until you unset it. Changing
`shared_act_scale` during a live session unloads every resident model because
the scale hook is installed when the model loads.

`comfy_managed = true` enables ComfyUI's DynamicVRAM inside every mesh actor.
It is experimental and does not support LoRA; read [docs/TROUBLESHOOTING.md #62](TROUBLESHOOTING.md#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)
before setting it. Loading the file enforces two cross-key rules, as the Init
node does: `comfy_managed = true` is rejected beside `slab_weights = true` and
beside `lora_low_rss = true`, because this residency forces both settings off
inside the worker. `slab_weights = "auto"` is allowed beside it: `auto` lets the worker choose, and the worker turns slab off in this case.
Like `disable_custom_nodes`, it is a startup setting: changing it for an
actor that has already bootstrapped is refused until the Attached mesh is
reset.

**Omit the key when you do not want it. Never write `comfy_managed = false`.**
`[worker_args]` supplies defaults for the graph, and the merged dictionary is
part of every identity-gate capability context, so writing the key at
all, with either value, changes the context of every render on this box and
re-proves every PASS in the ledger. The Init node follows the same rule and
writes no key when the widget is off.

`disable_custom_nodes = true` is a startup setting for sealed validation: the
worker imports ComfyUI core and built-in nodes and runs no third-party pack. It
is a `[worker_args]` key, not a public Init-node widget; sealed runners reach
it through the private `_init_with_bootstrap_policy` seam. Changing it for an
actor that has already bootstrapped is refused until the Attached mesh is
reset. `DGXM_NO_CUSTOM_NODES` only skips the optional pack preload; it does not
set this startup setting. Leave it false unless you run a campaign with
no custom nodes. Nothing announces it at startup: a sealed worker logs `comfy
bootstrapped from <dir>` and no preload line at all, so when a pack the driver
loads is missing on a worker, check `[worker_args]`. With this setting false the
preload logs one count line (`preloaded N custom node pack(s)`, or that it
found no eligible packs) plus a warning for each pack that fails to import.

`DGXM_FAULT_LOAD_RANK` is a debug-only worker environment variable, excluded
from `[worker_args]`. It makes the rank it names refuse inside the model load,
after every rank has priced the load and before any weight is read, which is
the only way to exercise the post-load readiness exchange on hardware
([docs/TROUBLESHOOTING.md
#99](TROUBLESHOOTING.md#99-a-render-refuses-saying-an-acceptance-fault-was-injected-on-that-rank),
and the test in docs/VALIDATION.md). It stays off unless `DGXM_ACCEPTANCE=1` is
set on the same worker. It is an environment variable for two reasons: worker
args reach every rank, so they cannot name one, and they ride inside every
render's capability context, where a debug knob would re-prove every PASS in
the ledger. The worker reads both variables at setup and logs them at WARNING,
armed or ignored. Both names are on the worker loop's reviewed inheritance
list, so a unit drop-in reaches the actor; neither carries a credential, and
together they can only make one rank refuse a load. Never set either on a box
that is rendering.

```toml
[worker_args]
disable_pinned_memory = true
reserve_vram_gb = 8.0
```

## `[[hosts]]`

| key | default | meaning |
|---|---|---|
| `name` | none | ssh-reachable hostname or literal IP. `dgxm` reaches the host over ssh by this name. The driver also resolves the first host's name to tell whether rank 0 shares its box. Only when the name resolves to the driver's own box do the loader and render preflights charge rank 0's checkpoint weights to the driver's memory, and only then does the MiniMax H3 activation preflight run. A name the driver cannot resolve, such as an alias only `~/.ssh/config` knows, counts as remote, so the driver skips that charge and that preflight. `@` is rejected; `:` is accepted only in a literal IPv6 address, so SSH/rsync destination syntax cannot be rewritten. |
| `address` | **required** unless `name` is a literal IP (then `tcp://<name>:26600`) | Worker service bind. It must be the fabric IP. A hostname-derived bind is refused at config load: Ubuntu's `127.0.1.1 <hostname>` /etc/hosts line would bind the service to loopback (DESIGN §5.1). |
| `gpus` | `1` | GPUWorker actors to spawn on this host (1..64); every host must declare the same count |
| `ssh_user` | current user | lifecycle ssh user. `@` and `:` are rejected as destination delimiters. |
| `comfy_dir` | cluster default | per-host ComfyUI root |

Host order is rank order; the first host is the NCCL master.
`nccl_master_addr` only sets the address the ranks dial to reach it, so it
must be one of the first host's addresses. A cluster holds at most 64 hosts and
256 ranks, and the actors reject an invalid rank or device assignment before
CUDA or NCCL setup.

`dgxm` syncs the package source into a dedicated directory on each remote host,
deletes everything else there (compiled bytecode included), and starts workers
with bytecode reads and writes in that tree turned off. Strict acceptance
checks each worker's live source set and loaded module origins on its own.

## Worker transport security

Peer authentication is unavailable at the attach API dgx-monarch calls, and its
control plane executes actor code. Isolate that interface as
[SECURITY.md](../SECURITY.md) requires, then record the acknowledgement:

```toml
[cluster]
transport_security = "trusted_fabric"
```

Without it, lifecycle startup and cluster attach refuse and `dgxm doctor` fails
its transport row. Doctor also warns when a worker address is globally
routable.

## Fabric profiles

A `[fabric.<name>]` table extends the shipped profile of that name, overriding
any value it sets, or defines a new profile. It accepts only bounded scalar
`NCCL_*`, `GLOO_*`, and `UCX_*` tuning values. TOML booleans become the
conventional environment values `1` and `0`. Variables whose
underscore-delimited names contain `FILE`, `PLUGIN`, or `MODULE_DIR` tokens can
load code or name arbitrary paths and are rejected on the driver, then
revalidated inside each actor before they reach `os.environ`.

Shipped profiles:

* **`dgx-spark-pair`:** 2x DGX Spark over ConnectX-7 200G RoCE, both rails:
  `NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME=enp1s0f0np0`,
  `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`, `NCCL_IB_GID_INDEX=3`,
  `NCCL_NET_GDR_LEVEL=SYS` (GB10 has no GPUDirect RDMA, so NCCL stages through
  host memory), `NCCL_BUFFSIZE=16777216` (16 MiB),
  `UCX_NET_DEVICES=enp1s0f0np0`, and `NCCL_DEBUG=WARN`.
* **`generic-roce`:** starting point for RoCE clusters. It sets
  `NCCL_IB_GID_INDEX=3` and `NCCL_DEBUG=WARN`; add `NCCL_SOCKET_IFNAME` and
  `NCCL_IB_HCA` for your NICs.
* **`generic-ib`:** InfiniBand autodetect; it sets only `NCCL_DEBUG=WARN`.
* **`single-node`:** empty.

When a host's configured `NCCL_SOCKET_IFNAME` is absent or down, its worker
rewrites the profile's socket, UCX and HCA variables from the RDMA rails it
finds up and logs a warning; `dgxm doctor` warns when the host it runs on has
that interface is also unavailable locally.

`NCCL_PROTO` is not an operator fabric-profile knob; every configured value
is rejected. Leave it unset. `LL` breaks
FSDP (VALIDATION.md), so
`dgxm doctor` also normalizes case/whitespace and FAILs when an ambient driver
or worker-shell value includes `LL`.

Fabric environment settings apply only inside worker actor processes.

---
See [VALIDATION.md](VALIDATION.md) for fabric measurements and setup test
results. Successful setup does not establish native RDMA support or model
accuracy.
