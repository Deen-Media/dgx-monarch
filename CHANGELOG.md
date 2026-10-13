# Changelog

## [Unreleased]

- Keep the driver's metrics export through a config edit that does not
  produce a new fleet. An `otlp_endpoint` the driver cannot install now refuses
  before anything is written, so the running exporter is never stopped, and an
  exporter thread waiting on a slow telemetry backend carries on when the
  backend answers. When the driver admits an edit but cannot stop the old
  workers, for example because a render still holds them, it now puts back the
  old fleet's policy and export, so a revert that reuses that fleet finds both
  as they were. If an exporter stopped for that edit is still blocked in a
  backend call and cannot restart, the driver logs one ERROR and
  `/dgxm/telemetry` shows `exporter_still_running` in `monarch_diagnostics`
  until the next mesh build.
- Latch native latent return off for the rest of the driver session after a
  native read fails on the driver. Later setups ask the actors for messaging
  instead of repeating the fault and retaining another latent each time; the
  failed render, and a Fleet job whose own native read fails, raise
  `NativeReturnLatchedError`, the driver logs one warning, and `/dgxm/telemetry`
  shows the latch next to `return_ownership`. Ownership rules are unchanged;
  the latch clears when ComfyUI restarts
  ([TROUBLESHOOTING #115](docs/TROUBLESHOOTING.md#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)).
- Add `--timing` to `benchmark/rdma_return_bench.py`: first-use and warm timed
  returns at chosen sizes inside the checked CPU-staged run, in either
  message/native order, with `--timing-summary` pairing the reports into warmed
  medians, savings and ratios. Timing evidence refuses host names, addresses,
  PIDs and home paths, and a plan that would stage more than
  `--timing-budget-gib` (default 8) refuses. The recorded 2026-10-13 figures
  came from a private workspace tool; this mode implements the same protocol
  for later campaigns and has not yet run on hardware.
- Make plain `dgxm update` fetch first, check the fetched commit's dependencies,
  and fast-forward only after that check passes, so a refusal leaves the
  checkout at its current commit instead of at the new commit against the old
  torchmonarch.
- Attribute a normal render's progress steps to the render's own telemetry
  token without transferring ownership, and ignore a step for a finished render
  or one that arrives when no render is live. A late step no longer leaves
  `dgxm top` and the sidebar showing an active render until ComfyUI restarts.
- Stop `dgxm diagnostics` from marking a healthy capture partial because its
  last poll window was shorter than a fetch: a final window shorter than the
  slowest fetch so far is slept instead of polled. `--output` is checked before
  the capture starts, and a refusal prints its fixed reason instead of only the
  exception type.
- Report `dgxm_diagnostics_available` and `dgxm_return_ownership_available` as
  0 for a Worker whose source has no diagnostics block, as after a partial
  deploy, instead of omitting the series. The driver scope also reads 0 when
  its payload has no block, as when the first telemetry refresh fails.
- Cut the `dgxm top` driver phase summary to one row instead of wrapping it over
  the panels below on an 80-column terminal.
- Name why a host's source probe failed: a missing or non-executable Worker
  interpreter reads `FAILED_WORKER_INTERPRETER`, a lifecycle lock failure
  `FAILED_LIFECYCLE_LOCK`, any other exit without a verdict
  `FAILED_SOURCE_PROBE`, each with the exit code and a bounded, home-relative
  stderr tail. `dgxm restart` refuses on all of them before stopping any Worker.
- Run the driver's policy admission before recycling a live fleet, so a
  cluster.toml edit the driver refuses (native return without a `nic:` target
  for every host, an export it cannot start, a transport change that needs a
  fresh process) keeps the working fleet attached; reverting the edit reuses it.
  The refusal puts back the driver's earlier policy and export state, so a
  running metrics export keeps running.
- Tighten `[monarch]` validation: `nic:` names starting with `-` or `.` or
  containing `..`, `cpu:` and `gpu:` numbers with leading zeros, a `groups:`
  affinity with fewer than two groups, and an `otlp_endpoint` with a path
  other than a bare trailing `/` now fail config loading with a typed error
  naming the key. A Worker unit rendered from such a value reads
  `POLICY_STALE`: fix the value in cluster.toml, then run
  `dgxm update --verify --target HEAD` to re-render the unit. The update, like
  the probe, checks a unit's policy argument for its keys, value types and
  canonical encoding but not its values, so it accepts that unit as stale
  instead of refusing with `prior_worker_drift`.
- Treat a Worker unit that carries a well-formed Monarch policy other than
  the current one as stale rather than tampered. A `[monarch]` or
  `cluster.rdma_latent_return` edit on a managed installation no longer locks
  out lifecycle commands: the probe reads `POLICY_STALE`, `dgxm restart`
  refuses before stopping any Worker service, `dgxm up` names the cause, and
  `dgxm doctor` adds a `worker policy` row per host. When a host's probe goes
  unanswered, restart keeps the old order and `dgxm up` refuses the stale host
  after the stop. `dgxm update --verify --target HEAD` renders the current
  policy with the usual rollback. Worker units carry the default
  `monarch.startup_timeout_s`, which only the driver reads, so editing it
  leaves them current. A unit without policy arguments counts as current only
  under the default policy with native return off
  ([TROUBLESHOOTING #113](docs/TROUBLESHOOTING.md#113-lifecycle-commands-refuse-with-policy_stale-after-a-policy-edit)).
- Resolve the driver's own `[monarch.rdma]` host target by matching
  `client_bind` to a `[[hosts]]` address, the rule the RDMA qualifier now
  shares; its `--driver-host` becomes an optional check. With native return on,
  the driver refuses before starting any process unless it and every host
  resolve to a `nic:<device>` target, whatever `max_nics_per_buffer` allows,
  and a `host_targets` value other than `nic:<device>` fails config loading.
- Make driver Monarch metrics and OTLP export work: registering the nodes no
  longer loads Monarch, so the driver applies its endpoint first. A requested
  export that cannot start refuses the mesh build with a typed error naming the
  reason; actors and Workers log it. `/dgxm/telemetry` reports
  `monarch_diagnostics`.
- Narrow the Gate configuration fingerprint to settings that can change math,
  residency or transport, and the transport identity to `[monarch.rdma]`.
  Editing `auto_heal`, `ssh_key`, `monarch.startup_timeout_s` or
  `[monarch.diagnostics]` keeps PASS rows. Each combination re-proves once
  after this change
  ([TROUBLESHOOTING #114](docs/TROUBLESHOOTING.md#114-every-combination-runs-its-first-use-gate-again-after-an-update)).
- Stop recording digests of install paths and pip RECORD files in Monarch
  capability snapshots, remove them from the committed 0.7.0 pin baseline, and
  make the leak check report a digest of any private home path the scanning
  interpreter can see.
- Mark the Kroma INT8 artifact row as a file the reader supplies, record that
  the pinned upstream Qwen3-VL encoder matches the tested encoder tensor for
  tensor, and point the data-parallel batch refusal at the Kroma upscale
  graph's batch-one note.
- Fix the torchmonarch canary's Future probe to use a returned channel Future,
  avoiding the constructor removed in 0.7.0, and show comparison reports and
  failed stages in Actions.
- Check authenticated managed Worker releases in Doctor host environment probes,
  so a staged runtime upgrade does not falsely report the unchanged base pin as
  Worker version skew. Keep driver and plain-install checks separate. Report
  bounded failure categories from verified-update Doctor checks without private
  probe details; certainty and rollback behavior remain unchanged.
- Allow an explicit foreground process-inspector lifetime up to four hours;
  retain the launcher's 30-minute default and authenticated, bounded inspection.
- Add Kroma v0.3.1 Turbo OPD workflows for BF16, local INT8 conversions and
  two-pass 2x refinement using the existing Krea2 runtime. Document upstream
  sampling settings, required artifacts and native shift handling. Preserve
  the stock Krea2 final projection shape under pure Ulysses; the BF16 and local
  INT8 8-step graphs and BF16 2x refinement matched their own stock one-GPU
  references in conditioning, latents and decoded pixels. Acceptance covers
  the measured pure-Ulysses configurations. Manual `uly2+fsdp` also matched
  stock for BF16/local INT8 T2I and BF16 2x refinement at batch 1. Batch-2
  `dp2+fsdp` T2I remains CHECK: FSDP reload proofs passed, but native
  stock-output fidelity did not. Keep
  upscale at batch 1: native VAE encoding collapses an initial image batch of
  2 to one latent item, which DP2 correctly refuses at refinement.
- Upgrade to TorchMonarch 0.7.0 and advance the package identity to 1.2.0 so
  previous Gate PASS rows cannot authorize the new runtime. Preserve the
  existing Torch/CUDA build and validate the candidate's dependencies before
  pin installation in plain and verified updates. Report bounded package
  diagnostics when prerequisites are missing or incompatible.
- Let verified updates and recovery explicitly reuse the reviewed read-only
  process inspectors when protected process markers are unreadable. Require
  fresh authenticated reports at both activity checks and record the choice
  in the receipt; initial setup defaults remain unchanged.
- Preserve exact legacy Worker units during migration and verify policy-bearing
  Worker commands against their captured units. Finish inspector socket cleanup
  even when terminal shutdown delivers repeated signals.
- Validate native RDMA binding load and package/import ownership separately from
  public API signatures. Extend the dependency canary with real local CPU actors,
  concurrent dispatch, Future timeout semantics and ProcMesh shutdown. Run it
  against Torch 2.12.0+cu132; retain these checks as CPU-only evidence.
- Validate the unchanged shipped Krea2 workflow at 1536×1536 with its automatic
  topology and first-use Gate; record a slab cross-residency PASS and owned cleanup.
- Validate the unchanged shipped H3 workflow at 1344×768×124 with full CPU
  video/audio decode and owned cleanup. Its Gate remains INCONCLUSIVE/no_material;
  the recorded result establishes functional output, not a fidelity or speedup claim.
- Record exact-source GPU checks for the migration: adapted Krea2 Turbo pixel equality,
  explicit recovery after interruption, a RAW FSDP clean-reload Gate PASS and
  reduced-canvas H3 MP4 output. Matched latent decoding localized the pipeline/Fleet
  pixel differences downstream of sampling; clean cancellation remains unproven.
  See [the scoped results](docs/VALIDATION.md#gpu-checks-for-the-070-migration)
  for reused-output counts, recipes and timing limits.
- Add validated Monarch startup and RDMA policy with per-host target overrides,
  native readback and transport identity. Start with one NIC, one QP and one
  application chunk. Selected-rail CPU-staged production returns passed the
  recorded integrity and ownership cases, and one-step at-threshold latents
  matched stock and messaging in two fresh native sessions. Repeated large-return
  timings support opt-in use; native return stays off by default. See [the exact scope](docs/VALIDATION.md#native-rdma).
- Fix Fleet progress remaining active after completed jobs. Each receiver owns
  its progress token, retires it during cleanup and ignores late steps after
  retirement without clearing another render.
- Record healthy no-model reattach and controlled fresh-driver recovery, while
  preserving the reproduced same-driver stop/start and restart failures;
  [issue #2](https://github.com/Deen-Media/dgx-monarch/issues/2) remains open.
- Add bounded phase timings and opt-in supported Monarch metrics. Advanced
  attached-cluster profiling, SQL and stack tools remain explicitly unavailable
  where the upstream API cannot provide an owned, stoppable session.

## [1.1.0] - 2026-10-11

### Everyday use

- Show health separately from activity in the dashboard and browser. Healthy
  rendering reads READY / RENDERING with a wait instruction. Preserve legacy
  readiness JSON and recordings, and explain activity that CLI status does not
  observe. Unknown or blocked evidence remains visible.
- Add opt-in `dgxm launcher` access from a normal terminal, bound to the selected
  environment and cluster config. Preserve existing commands and make identical
  reruns a no-op. Setup verifies PATH resolution and the dashboard endpoint.
- Launch ComfyUI alongside its dashboard with `--with-top`, including from the
  desktop shortcut. Preserve foreground logs and terminal signals, and close
  only the driver and dashboard owned by that launch.
- Add a MiniMax H3 workflow that saves H.264/AAC MP4 at 24 fps. Keep the existing
  separate-frame/audio workflow. The catalog now contains 59 workflows.

### Installation and updates

- Add an optional fleet-wide Sol-Attention installer with a reviewed package
  plan and protected runtime checks. Keep the explicit GPU dense-control smoke
  separate from installation. Diagnose dependency requirements, Python headers
  and compiler caches; document the accuracy waiver and mesh recycling.
- Accept proven editable metadata companions and preserved core-library links
  during Sol maintenance. Keep installed-file ownership authoritative and reject
  ambiguous metadata, unsafe replacements and changes to protected packages.
- Derive the Torch install constraint from distribution metadata while retaining
  runtime/CUDA checks. Preserve working Torch builds and report useful package
  ownership details when verified updates refuse an installation.
- Support the verified color-matcher 0.6.0 test-data layout in TorchMonarch
  inventory checks, with a recovery command for older updaters. Keep checks for
  changed files, duplicate ownership and import shadows.
- Report Worker service health from fresh passive process and listener checks,
  separately from attached actor telemetry. Missing or failed observations
  remain unknown.
- Simplify guided setup and handoff: gather prerequisites before plan approval,
  validate private receipt paths and process-inspection capability early, and
  put launch commands and installation paths first. Detailed network auditing
  is optional; trusted isolation and operator acknowledgement remain required.
- Let users select a supported first-render model; Chroma is optional. Preserve
  recognized deletions of ComfyUI's two tracked placeholder files during setup,
  with a warning; other checkout changes still block it.

### Compatibility and verification

Existing node mappings, ordered inputs and defaults remain compatible. Core
runtime dependency pins are unchanged. An upgrade requires first-use checks
before cached identity-gate results can be reused.

[Operator acceptance](docs/VALIDATION.md#onboarding-and-operator-acceptance)
records the tested source, two-Spark execution and cleanup scope. Sol dense
kernel checks and a saved MiniMax MP4 are functional evidence, not sparse
attention or model-fidelity qualification. Native latent RDMA remains
**NOT RUN / HOLD**. The cuSPARSELt ARM64 packaging exception remains tracked in
[issue #5](https://github.com/Deen-Media/dgx-monarch/issues/5).

## 1.0.0

Initial release of DGX Monarch: distributed image and video generation in
ComfyUI, built for a pair of NVIDIA DGX Sparks.

### Rendering

- Split supported renders across both GPUs with sequence or CFG parallelism,
  or run independent prompts with Fleet.
- Shard supported model weights with FSDP when a checkpoint needs more memory.
  Text encoding, VAE decode and saving remain in the ComfyUI driver.
- Start from 58 example workflows, including Krea2, Ideogram4, MiniMax H3 and
  Qwen Image 2.1. [Model support](docs/MODELS.md) lists the tested checkpoints,
  precisions, inputs and parallel modes.
- Use automatic topology selection or choose an explicit layout. Slab
  residency and low-RSS LoRA handling offer memory savings where supported;
  first-use checks compare those paths before reuse.

### Installation

- Check dependencies with `tools/check_dependencies.py`. It verifies the
  official artifact before accepting the known upstream packaging exception;
  other dependency errors fail.
- Install from source into the Python environment used by ComfyUI, preserving
  its CUDA Torch build and building the pinned xFuser compatibility wheel.
- Configure both Sparks through guided setup with a reviewed plan, dependency
  and model checks, Worker service ownership checks, and saved receipts.
- Follow the repository's [setup skill](skills/dgx-monarch/SKILL.md), directly
  or through the [copyable README prompt](README.md#set-up-with-a-coding-agent).
  It covers discovery, approvals, installation, validation and recovery.
- Use the [first-render recipe](docs/QUICKSTART.md#first-distributed-render)
  for exact model files, pinned downloads, hashes and distributed settings.
  A fresh-agent trial on two Sparks completed saved Chroma renders and a
  non-destructive setup rerun. [Installation results](docs/VALIDATION.md#fresh-agent-installation)
  describe the tested baseline, reused components and limits.

### Operations

- Inspect the pair with Doctor, status, the ComfyUI sidebar and the terminal
  dashboard. Diagnostics distinguish failed checks from missing information.
- Apply verified updates with coordinated Worker activation, source checks,
  Doctor and a cluster smoke test. Prior releases and transaction records
  support recovery, including when a Worker shares the driver's Spark.
- Check capacity before loading and confirm owned actor cleanup before reuse.
  Worker services remain running when ComfyUI closes; stopping them is a
  separate operator action.

### Development

- Run CPU checks on GitHub-hosted x64 runners with Python 3.11 and 3.12, and
  ARM64 with Python 3.12. The x64 jobs also check package contents and CLI
  installation; ComfyUI canaries check upstream compatibility.

### Limits

- PyTorch still pins the NVIDIA cuSPARSELt ARM64 wheel with a platform-tag
  error. The [dependency checker](docs/TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform)
  verifies and reports this exception without changing installed packages.
- Support and measured speedups apply to the configurations listed in
  [Model support](docs/MODELS.md), [Benchmarks](docs/BENCHMARKS.md) and
  [Validation](docs/VALIDATION.md). A workflow template does not establish
  accuracy for every precision, LoRA, attention kernel or memory mode.
- FSDP is a capacity option and can be slower. Optional memory optimizations
  have their own checks and restrictions; failed comparisons do not grant
  support for them.
- The inter-node network must be isolated and source-restricted. The Worker
  API has no peer authentication; follow the
  [fabric requirements](SECURITY.md#fabric-trust-boundary).
- Native latent RDMA remains **NOT RUN / HOLD** and is off by default.
  Latents return through actor messages; distributed rendering uses NCCL.
- A failed reply channel can require a fresh ComfyUI or driver process after
  confirmed actor cleanup. See [Troubleshooting](docs/TROUBLESHOOTING.md).
