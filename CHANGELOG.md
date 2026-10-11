# Changelog

## [Unreleased]

- Support proven editable metadata and preserved core-library links during
  optional Sol maintenance. Keep installed-file ownership authoritative, bind
  each protected link and target, and refuse ambiguous metadata or replacement
  through links.

- Show health separately from rendering or other active work in the dashboard
  and browser. Preserve legacy readiness JSON and recordings, and explain the
  render activity that CLI status does not observe. Keep unknown mesh evidence
  and blocking fields visible when aggregating multiple meshes or drivers.

- Add an opt-in `dgxm launcher` for normal-terminal access to a selected
  environment and config. Preserve existing commands, refuse unsafe or edited
  targets, and make identical reruns a no-op. Setup guidance verifies actual
  PATH resolution and the intended dashboard endpoint.

- Add an optional combined ComfyUI and dashboard launch with the selected
  interpreter, config and endpoint. Preserve foreground logs and signal
  handling, and close only the dashboard and driver owned by that launch.
  Desktop shortcuts can save the same options.

- Build pip's Torch constraint from distribution metadata while preserving
  separate runtime/CUDA checks. Name safe package-ownership details locally
  when verified update refuses; keep receipts disclosure-safe.

- Add a MiniMax H3 template that saves H.264/AAC MP4 at 24 fps, retaining the
  existing separate-frame/audio template. Check native export with synthetic
  CPU video and audio in the ComfyUI canary.

- Extend optional Sol-Attention diagnostics on each configured interpreter
  with installed dependency requirements, Python headers and bounded cache
  observations. Explain the accuracy waiver and post-install mesh recycling;
  package detection does not claim GPU compilation or kernel execution.

- Add a plan-first optional Sol installer for every configured Worker
  interpreter. Bind apply to reviewed wheel plans, protect the existing core
  runtime and refuse unsafe package changes. Keep the explicit GPU dense-control
  smoke separate from installation and model accuracy validation.

- Accept the verified color-matcher 0.6.0 test-data layout during pinned
  TorchMonarch checks. Preserve checks for changed files, duplicate ownership
  and executable import shadows. Add a recovery command for installations
  whose older updater cannot pass this check.

- Report live Worker service readiness from fresh passive process and listening
  socket checks, separately from actor telemetry. Refresh in the background;
  missing, expired or failed observations remain unknown.

- Put start commands, browser access and installation paths first in the setup
  handoff. Offer the optional dgxm TUI and a simple launch script in the user's
  chosen location. Keep permanent installations outside temporary or chat-output
  folders.

- Use the default private setup receipt directory and check custom paths early.
  Add a process-inspection capability check and a maintained foreground helper
  command so required administrator steps can be planned before apply.

- Gather setup prerequisites before asking for one installation plan approval,
  and reuse approval for its listed actions. Make detailed network auditing
  optional while retaining the required trusted isolation and operator
  acknowledgement. Add private, read-only reports for requested audits. Keep isolated acceptance tests separate from normal
  installation.
- Permit unstaged deletions of ComfyUI's tracked `input/example.png` and
  `output/_output_images_will_be_put_here` during setup. Preserve the missing
  files, modified checkout status and warning; other changes still block setup.

- Let users choose a supported model for their first distributed render.
  Chroma remains an optional recommendation; setup still requires a saved
  output, proof that both Sparks participated and a safe rerun.

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
