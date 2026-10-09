---
name: dgx-monarch-dev
description: >-
  Contribute to dgx-monarch: author a new model-family adapter with its
  acceptance gates, follow independent-source contribution rules, run the
  CPU test suite and benchmark harness, and execute the release checklist.
  Use when working on dgx-monarch internals, porting a new model family,
  bumping torchmonarch, or preparing a release.
---

# dgx-monarch development

Work in `$COMFY_DIR/custom_nodes/dgx-monarch`, either the installation
checkout or a symlink to it. `docs/DESIGN.md` is the specification; section
numbers below refer to it.

Maintainability rules (§7): annotate public and internal interfaces with
types, keep modules
within 500 lines unless `tests/line_limit_helpers.py` records a reviewed
ceiling, put model-specific conditionals in adapters, keep the auto table as
data, and cite dated evidence for empirical constants.

## Independent-source checklist (§3)

1. Implement from the repository spec, current ComfyUI source, and public
   dependency APIs. Do not copy external implementations.
2. xfuser stays external. Do not vendor it. The temporary compatibility wheel
   (`tools/build_xfuser_compat_wheel.py`) is rebuilt from a SHA-verified
   official wheel and changes only the version metadata and the optional NPU
   import, which it makes lazy. Retain this workaround until official xFuser
   and yunchang releases build CUDA attention without it.
3. For new behavior, identify the source in the PR, such as a specification
   section or public API document.
4. Public node and package identities stay `dgx-monarch` and `DGXMonarch*`.
5. Sign off external contributions under the DCO.

## Adapter authoring (§5.3)

Read `docs/ADAPTERS.md` first. Summary:

1. Study the family's comfy model source (`comfy/ldm/<family>/`).
2. Write the sharded forward (shard_seq after embeddings, sp_gather before the
   head; shard every per-token tensor).
3. Route self-attention through USP: `usp_options()` for pure-self-attn block
   loops, module bind for mixed blocks. Cross-attention stays stock.
4. Declare padding and mask rules; raise typed errors for unsupported USP behavior.
5. Register in `ADAPTERS` (most specific first); a subclass whose forward
   contract differs raises `UnsupportedModelError` from `matches()`.
6. Complete the acceptance gates in `docs/ADAPTERS.md`: cross-rank latent
   identity, single-GPU reference fidelity, execution or typed refusal for
   every auto-table topology, a logged `hot-swap` after a LoRA tweak, and a
   passing lazy-swap identity gate for a LoRA stack.
7. Hardware gate before shipping: the template's derived cells run through the
   sweep harness on the pair at every preset and quant the matrix derives,
   scored against the single-GPU reference. Attach cell IDs, verdicts and
   elapsed times to the PR. `docs/SWEEP.md` section 2 gives the commands.

Implement families in this order: wan shape, flux shape, packed single-stream,
then video with per-token timesteps (`docs/ADAPTERS.md` "Family implementation
order").

## Actor invariants

* **Concurrent endpoints, serial compute (§5.2):** every `GPUWorker` endpoint
  is `async def`; the blocking GPU/CUDA/NCCL body runs on one `max_workers=1`
  executor (`_gpu_exec`) via `_on_gpu`, serialized by a single `asyncio.Lock`
  (`_gpu_lock`). Every endpoint that takes `_gpu_lock`, or otherwise holds
  the actor for more than an instant (`artifact_identity`, `capacity_quote`),
  must be `@concurrent_endpoint`; the dispatch comment at the top of
  `actor/worker.py` says why and names the instant endpoints, `status` among
  them, that stay plain and lock-free so supervision answers during a render.
  `_on_gpu` copies contextvars: RDMABuffer reads the actor context, which is
  unset off-thread and would misbind to the client. Worker setup
  (`actor/worker_env.setup_impl`) pins `torch.set_num_threads(16)`
  (`DGXM_NUM_THREADS` overrides it) because using all 20 Spark cores starves
  host IO.
* **Cross-render pipeline (F2, §5.6):** `mesh.submit_sample` and
  `collect_sample` split eager send from deferred collect; `RenderPipeline`
  (`nodes/pipeline.py`) keeps at most the Init `pipeline_depth` renders in
  flight. The sequential path, `nodes/common.run_render`, is the depth-1 form:
  `submit_render(...)`, then `.result()`. The actor lock keeps compute serial,
  so latents stay byte-identical with or without overlap. Overlap helps queued
  shared-model video, large latents and many fast renders; compute-bound image
  renders gain little.
* **RDMA latent return (F3, §5.6):** default `off`, and held for
  requalification pending the checks in docs/DESIGN.md §5.6, which explains
  the hazard and the requirements for enabling it again. When enabled it is
  size-gated at `DEFAULT_RDMA_MIN_BYTES` (8 MiB in constants.py; config
  `rdma_min_bytes`, env `DGXM_RDMA_MIN_BYTES`). Never force-free RDMA backing
  in `unload` or `clear_vram`: a pipelined one-sided read may still be
  running. A `HandoffRegistry` entry keeps its backing until its exact
  generation and token ACK or an actor recycle. Live entries are capped at
  the pipeline depth; when the registry is full, the worker returns the latent
  by messaging and evicts no entry.
* **FSDP plus sequence parallelism (§5.4):** `uly2+fsdp` puts two NCCL
  communicators on the same two ranks (FSDP all-gather over the world group,
  USP all-to-all over the Ulysses group), and they deadlock at world=2 without
  `NCCL_LAUNCH_ORDER_IMPLICIT=1` (NCCL >= 2.26). Every worker setup sets it
  and nothing clears it, because NCCL reads it once per process, at the first
  launch. At the default `fsdp_prefetch_depth` of 1,
  the empty FSDP2 forward-prefetch list in `adapters/fsdp.py` is a defensive
  pin, not part of the fix. Measured 2026-07-04 at world=2: byte-identical to
  resident, about 1/world of the weights per rank, about 2x slower.
  `ring2+fsdp` raises a typed refusal. `NCCL_PROTO` stays unset: the `PROTO=LL` rule
  (Appendix A) is a separate hazard and still holds.

## Test / validation ladder

```bash
ruff check __init__.py src tests benchmark tools
mypy -p dgx_monarch
bash scripts/test_cpu.sh                           # CPU, timing + slow-test report
python tools/gen_templates.py --check
# hardware (Spark pair): follow the smoke contract in docs/THREAT_MODEL.md
dgxm doctor
# first run: cp benchmark/matrix.example.toml benchmark/matrix.toml (gitignored)
python benchmark/run_matrix.py --matrix benchmark/matrix.toml
```

Public-source CI covers lint, unit tests, artifacts, and canaries on hosted
runners. It never routes a job to protected hardware. Hardware validation
uses hand-run scripts from a separately reviewed private control repository;
docs/THREAT_MODEL.md records the contract. Verify the exact public source commit
and source manifest on every host, obtain operator authorization, and passively
check idle state and ownership before execution. No CI workflow, Actions
runner, or scheduled job may execute on protected hardware. For a check without
a clean recorded run, report the hardware gate as **NOT RUN**. Worker services
persist; stopping one requires explicit maintenance authorization and verified
restoration. Native latent RDMA remains **NOT RUN / HOLD**.

## torchmonarch bump procedure (§6.3)

1. Introspection first: run `tests/canary/monarch_surface_canary.py snapshot`
   against the candidate in a scratch environment and compare it with the
   checked baseline. A `dir()` diff shows additions, not behavior. Review
   semantic changes such as dispatch behavior and warning classes explicitly.
2. Run `tests/dispatch_contract_probe.py` under the candidate. An API-name
   comparison cannot detect changes to dispatch behavior.
3. Update the pin in every declaration (`pyproject.toml`, `requirements.txt`,
   `scripts/setup_env.sh`, `TORCHMONARCH_PIN`, and the tests that assert the
   old version) and bump `__version__` in the same commit. The gate ledger
   binds trusted `PASS` rows to the package version, so a pin-only change can
   reuse a PASS measured on a different runtime. Regenerate the checked baseline
   on aarch64 as `tests/fixtures/torchmonarch_pin_<version>.json`, with the
   version's dots written as underscores (`tests/test_monarch_surface.py`
   builds the name from `TORCHMONARCH_PIN`; for 0.6.0 it is
   `torchmonarch_pin_0_6_0.json`). Update `.github/workflows/ci.yml` and
   `.github/workflows/torchmonarch-canary.yml`, which name that file, and
   review `PIN_SEMANTICS` in `tests/test_monarch_surface.py`.
4. After active work has stopped, deploy the same version atomically on every
   host. A mixed-version mesh can attach
   and spawn before every endpoint call times out without a version message.
   Update the source checkout and exact dependency pin on all hosts, then
   restart and run doctor.
5. Hardware gate: measure `status` latency during a live render (poll
   `dgxm status` or `dgxm top` while a render holds the GPU) and compare it
   against the dated concurrency result in docs/VALIDATION.md. The
   Cluster Status (DGX Monarch) node cannot measure concurrent latency:
   ComfyUI queues its prompt until the render ends. Also test mid-render
   cancel and recycle, a null A/B across two driver sessions with fresh pixel
   baselines per pin, and one combination's Gate run against its ledger rows
   before repeating Gate checks across the fleet.
6. Add the CHANGELOG entry.

`git grep -lE '(from|import) monarch\b|import_module\("monarch' src` lists the
modules that import torchmonarch. `git grep -l 'monarch\._src' src` adds the
`cli/` modules that name its private bootstrap module as a string and find
actor processes by it: process identity (`proc_identity`,
`process_inspector`), the reap ledger (`actor_ledger`) and the two
setup-service script modules. Review both sets on a bump.

## Release checklist

1. `ruff`, mypy, pytest, and template generation pass. Smoke-test a
   fresh Git checkout directly under ComfyUI `custom_nodes`: install it with an
   explicit `COMFY_PYTHON` and the docs/INSTALL.md recipe, which builds the
   xFuser compatibility wheel, then runs `"$COMFY_PYTHON" -m pip install -e` on
   the checkout with that wheel and the torch constraint.
2. From the separately reviewed private control repository, run the authorized
   hand-run hardware checks against the exact public source SHA, including each
   feature-specific validation. Do not arm a source-repository hardware route.
   Record unperformed checks as **NOT RUN**. Keep raw evidence private and attach
   only reviewed, sanitized conclusions to release notes. Compare manually
   maintained tables with those results; operator checks do not validate model
   families.
3. Write the CHANGELOG section and bump `__version__` in
   `src/dgx_monarch/__init__.py` (pyproject reads the version from it).
4. Node API diff review: mapping keys and ordered input names/types/defaults
   are semver-governed and snapshotted in `tests/node_input_contract.json`.
   New inputs remain optional or hidden and append-only.
5. Measured claims identify the actual run date and supporting evidence.
   Editing documentation does not establish hardware validation.
6. Tag the exact commit whose required checks passed and repeat the
   source-checkout smoke from that
   tag.
