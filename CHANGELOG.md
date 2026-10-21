# Changelog

## [Unreleased]

- Add an optional Studio companion with bundled workflow controls, durable job
  records, project libraries, reviewed imports, grid experiments and job-bound
  evidence; read the [acceptance record](docs/STUDIO_ACCEPTANCE.md) first.
- Studio update needs `--config` and `--driver-service` and defaults to verified
  mode. The owner token prints only to a terminal, otherwise to a 0600 file
  ([Studio](docs/STUDIO.md#install-and-start), [#123](docs/TROUBLESHOOTING.md#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token)).
- `dgxm top` and the ComfyUI sidebar show memory in GiB, draw the memory bar on
  hosts with no slab-resident model, give the six 16-color segments six colors
  and show RX and TX rails for every box ([TUI](docs/TUI.md#what-it-shows)).
- Drivers register GET `/dgxm/execution-events`, which grants the Studio
  companion a 30 s observer lease; the [threat model](docs/THREAT_MODEL.md#browser-to-comfyui-driver)
  lists it with the other driver routes.
- Verified update, release staging and setup refuse a target whose packaged
  Studio files are missing, extra or differ from their recorded SHA-256 table.

## [1.2.0] - 2026-10-16

**Upgrading from 1.1.0.** Follow [INSTALL](docs/INSTALL.md#upgrading-from-110): the first update runs the 1.1.0
updater, so this release's checks start with the next one. If a host's Workers
run from a release slot (remote hosts of guided multi-node setups, every host
after a verified update), use only `dgxm update --verify`; plain update leaves
them on 1.1.0 ([TROUBLESHOOTING #118](docs/TROUBLESHOOTING.md#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)). From 1.2.0 on, plain update refuses such a
host and Doctor's `worker release version` row fails a stale one. Before
updating, turn `rdma_latent_return` off, add no `[monarch]` table and clear
`DGXM_RDMA_QP_SPLIT` where Workers start. Then install `xxhash` and what
`pip check` names on every host, clear what [TROUBLESHOOTING #119](docs/TROUBLESHOOTING.md#119-a-mesh-build-or-worker-start-refuses-on-an-ambient-environment-variable) names, add a
`[monarch.rdma]` target before re-enabling native return, rerun the editable
install after a verified update, and restart ComfyUI. The torchmonarch pin moves
to 0.7.0; FLUX.3 Klein needs ComfyUI 0.41.0 or newer, or commit 20b46065e.

### Everyday use

- A ComfyUI that stays up attaches again after its Workers restart within the
  30 s delivery timeout, and refuses older Workers before it dials (closes [#2](https://github.com/Deen-Media/dgx-monarch/issues/2)).
  So does a sampler render that meets restarted Workers, recording `reattach`
  and `reattach_s` (closes [#47](https://github.com/Deen-Media/dgx-monarch/issues/47)); the two-Spark recheck ran on 2026-10-16.
- Eager setup and Fleet queues heal an abandoned lease without **Reset attached
  mesh**. The sidebar shows the topology and each host's rail rates, red when a
  rail is down, and progress no longer sticks after a render.
- Add three Kroma v0.3.1 Turbo workflows, which matched stock at batch 1, and
  four FLUX.3 Image Klein workflows, for 66 in all.

### Installation and updates

- From 1.2.0 on, plain `dgxm update` refuses a release-slot host before it
  fetches (`release_slot_worker`, `release_slot_unknown`), checks the fetched
  commit's dependencies on every host, fast-forwards only if they pass, and
  refuses an uncommitted `pyproject.toml` edit or a linked worktree.
- Doctor, `dgxm up` and `dgxm restart` read a managed Worker's torchmonarch from
  its release; the skew warning sends a slot to `--verify`; Doctor checks that
  `/usr/bin/python3` is 3.9 or newer on each host.
- Verified updates take `--privileged-process-inspection`, keep exact legacy
  Worker units, and accept a foreground inspector of up to four hours.
- Every manifest row pins a Hugging Face commit, size and sha256 or names the
  file you supply (101 pinned, 6 supplied), and CI checks it. The LTX 2.5
  Gemma 4 row pins the vendor's current file, checked again against stock.
  `setup_env.sh --comfy` no longer needs torchaudio.

### Compatibility and verification

Node mappings, ordered inputs and defaults stay compatible. The new pin and
version make each combination re-run its first-use Gate ([TROUBLESHOOTING #114](docs/TROUBLESHOOTING.md#114-every-combination-runs-its-first-use-gate-again-after-an-update));
later edits that cannot change math, residency or transport keep PASS rows.
[Migration checks](docs/VALIDATION.md#gpu-checks-for-the-070-migration) ran the shipped Krea2 workflow at 1536 square on that build's
uly2 route (row 11; slab Gate PASS) and the shipped H3 workflow (output only) on
ComfyUI e8665ca. Later records ran on ComfyUI 29e28e2b4 (0.41.0); the
[FLUX.3 Klein record](docs/VALIDATION.md#evidence-flux-3-klein) is the release's acceptance record. The [merged pull requests](https://github.com/Deen-Media/dgx-monarch/compare/v1.1.0...v1.2.0)
hold the detail, and TROUBLESHOOTING numbers refer to [that file](docs/TROUBLESHOOTING.md).

### Rendering

- Support FLUX.3 Image Klein, step-distilled and base, for text to image and
  reference editing ([MODELS](docs/MODELS.md#model-flux-3-klein), [TROUBLESHOOTING #121](docs/TROUBLESHOOTING.md#121-a-flux3-checkpoint-fails-to-load-on-comfyui-older-than-0410)). Both rows are **HW**. uly2
  matched one GPU exactly on text to image from 1024 to 1536 square and, since
  each fused norm runs at stock's row count, on edits and small sizes
  (2026-10-16). ring2 and base cfg2 are measured, not exact; NVFP4 refuses FSDP.
- `auto` takes cfg2 where it was exact and beat uly2 in [Campaign K](docs/BENCHMARKS.md#campaign-k-cfg2-against-uly2-2026-10-14-head-capped): Krea2 FP8
  and BF16 from 1.2 to 2.36 MP, Chroma BF16, FP8 and INT8 to 1.64 MP, Qwen-Image
  FP8 to 1.77 MP, Boogu FP8 and Anima BF16 under 1.2 MP, and CFG++ at CFG 1.
- A checkpoint or LoRA that pins an approximate attention kernel refuses class K
  where the pin runs, until you grant the Sol-Attn waiver ([TROUBLESHOOTING #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel),
  closes [#30](https://github.com/Deen-Media/dgx-monarch/issues/30)). Sequence parallelism drops the pin. A BF16 or FP16 checkpoint's
  own pins refuse class P on `+fsdp`, and `first_use` refuses a running pin on
  cfg2+fsdp and dp2+fsdp even with the waiver.
- Refuse a checkpoint whose weight scales ComfyUI would drop, such as FLUX.3
  Klein's `fp8r` file, class P with no waiver ([TROUBLESHOOTING #120](docs/TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
- A Gate FAIL quarantines only its own combination, LoRA keys on FSDP-replicated
  modules bake, Anima runs sequence-parallel again, padded Ulysses ran 2.2%
  faster in a Chroma test, and FSDP reload pricing credits freed shards.

### Operations

- Add a validated `[monarch]` policy for startup, RDMA and diagnostics. Config
  loading names a bad key, native return needs a `nic:` target per host, a
  policy edit reads `POLICY_STALE` (TROUBLESHOOTING #113) instead of locking out
  lifecycle commands, and a refused edit keeps the fleet and its export.
- A mesh build or Worker start refuses an ambient OTLP endpoint with export off
  and `MONARCH_RDMA_*` or `DGXM_RDMA_QP_SPLIT` values the policy did not set
  (TROUBLESHOOTING #119); so does an export that cannot start.
- Add opt-in Monarch metrics with ownership occupancy, OTLP export, and phase
  timings that split each native return. `/dgxm/metrics` adds step time,
  native-return latch and faults, and per-port rails; the finish record adds
  topology, table row, quant, kernel and pins. Monarch's profilers stay off.
- A failed native read latches native return off for the session
  (TROUBLESHOOTING #115). Doctor adds `listener generation` and `worker policy`
  rows and, like `dgxm restart`, names why a source probe failed. Fix partial
  `dgxm diagnostics` captures, `dgxm top` rates and phases, sidebar refresh
  after a reset, a Reset that raced a loader preflight and stale ledger temp
  files. Load profiles live in `~/.cache/dgx-monarch/profiles/`.

### Development

- CI type-checks under `check_untyped_defs`; canaries cover pins, scale replay,
  LoRA bakes and local actors. The sweep labels pinned cells and CFG 1 cfg cells
  under FSDP as refusals; the matrix runner records waivers and refusal classes.
  `tools/make_pin_probe_lora.py` and `rdma_return_bench.py --timing` are new.

### Limits

- Native latent RDMA stays off by default; [selected-rail checks](docs/VALIDATION.md#native-rdma) keep their
  recorded scope. Not yet run on hardware: FSDP LoRAs on H3, LTXAV or Anima, the
  credited Flux2 proof, return timing, clean cancellation and Kroma's batch-2 `dp2+fsdp`.
- Kroma keeps Ulysses: its cfg2 failed the same-math bar on 2026-10-16, so Krea2's cfg2 row
  excludes it, even when `family_adapter` names `krea2`, and the sweep labels Kroma cells to match.

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
