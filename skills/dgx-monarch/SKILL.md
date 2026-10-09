---
name: dgx-monarch
description: >-
  Install, validate, update, and troubleshoot DGX Monarch on Linux ComfyUI
  installations, including a two-DGX-Spark cluster. Use for agent-led setup,
  first distributed renders, and safe reuse of an existing installation.
---

# DGX Monarch setup and operation

Work from the repository checkout the user selected. Repository paths below
are relative to that checkout. Use the existing `dgxm` commands and linked
procedures; do not create another installer or bypass a failed check.

Read [INSTALL](../../docs/INSTALL.md) before installing dependencies or services,
[SECURITY](../../SECURITY.md#fabric-trust-boundary) before networked setup, and
[QUICKSTART](../../docs/QUICKSTART.md#first-distributed-render) before rendering.
Use [CLUSTER](../../docs/CLUSTER.md) for configuration keys and
[TROUBLESHOOTING](../../docs/TROUBLESHOOTING.md) for a specific failure.

## 1. Discover before changing anything

Start with read-only inspection. Identify which checkout, ComfyUI, environment
and model directories need read access, and which destinations need writes.
For new paths, request the narrowest existing parent, such as `custom_nodes`
for a node-pack checkout rather than the whole ComfyUI tree. Request the known
folder permissions together. Do not request blanket filesystem access or
disable the sandbox. Record the following for each selected host:

- OS and CPU architecture; Python interpreter used by ComfyUI; Python version,
  virtual-environment path and whether it imports system packages.
- ComfyUI path, Git origin, exact commit and local changes; existing custom-node
  installation or symlink; imported `dgx_monarch` path if already installed.
- Torch version, CUDA build and import path; installed Monarch, xFuser and
  other required dependency versions. Read current requirements from
  `pyproject.toml` and `requirements.txt`; do not substitute remembered pins.
- Config location, Worker service state, source installation, active ComfyUI
  sessions, attached actors and GPU jobs. Keep full diagnostics private.
- Free disk and memory; model files available in the selected ComfyUI model
  directories or a model cache the user permits you to inspect.
- The user-approved SSH hosts, fabric interfaces and addresses, and the basis
  for trusted isolation: existing verification or informed operator
  confirmation under INSTALL. Do not scan the LAN or assume private IPs imply trust.

Use launcher/service metadata or the user's supplied paths to identify the
ComfyUI interpreter. A shell's default `python` is not sufficient evidence.
Ask only for facts that safe inspection cannot establish: the intended hosts,
ambiguous ComfyUI environments, missing access, network approval or storage
choices. Never read private-key or token contents into tool output or chat.
Collect missing prerequisites and decisions into one short request after
discovery, rather than asking one question per command. Reuse the existing
ComfyUI and environment where suitable. Propose durable locations for anything
new and include the user's choice in the plan. Source, environments, models
and permanent launchers do not belong in temporary or chat-output directories
unless the user explicitly requested a temporary test. Ask whether the user
wants the optional dgxm TUI and an easy launch script, and where to save the
script, alongside the initial questions. If the TUI is accepted, include the
`[tui]` extra in the initial constrained dependency plan using INSTALL's
[optional dashboard procedure](../../docs/INSTALL.md#optional-dashboard-and-worker-services).
If declined, skip it; do not install extras without approval. Follow INSTALL's [launch handoff](../../docs/INSTALL.md#launch-and-handoff)
for its contents and verification. Use existing approved
credential helpers; let the user complete authentication,
license acceptance, sudo prompts and SSH host-key verification themselves.

Before apply, check the [user service session](../../docs/INSTALL.md#check-the-agents-user-service-session)
and [process inspection capability](../../docs/INSTALL.md#protected-process-inspection)
on each host. Include a required privileged inspector in the installation plan
rather than discovering it after an apply fails. This ownership requirement
is separate from the optional network audit. Reuse a helper only after checking
its authenticated reply; protected metadata may still require a foreground
sudo session on each host.

Use the default [receipt directory](../../docs/INSTALL.md#operator-receipts),
not a custom `--receipt` in the chat workspace. Check a customized
`XDG_STATE_HOME`, or an explicitly requested receipt path, during discovery.
Do not change permissions on unrelated directories to make it pass.

Select one source checkout for each ComfyUI installation, directly at
`"$COMFY_DIR/custom_nodes/dgx-monarch"` or a symlink to the working checkout.
Do not install from a second source tree. Verify the requested repository or
an explicitly approved mirror, the selected commit, a clean working tree,
and the symlink target before using an existing path. Preserve local changes
and unrelated custom nodes. ComfyUI's two missing example files may be
preserved under the narrow [INSTALL exception](../../docs/INSTALL.md#inspect-before-installing);
keep its modified status and warning in the record. Other mismatches need a
decision, not a reset or overwrite.

## 2. Agree on the installation plan

Describe what will be reused and what will be created or changed. Include
Python/ComfyUI paths, dependency changes, model storage, config destination,
services, caches, network access and recovery. List package installation,
Worker installation or restart, automatic service recovery, lingering and any
security change explicitly when needed. These can be approved together in one
plan; an explicit security approval does not require a separate conversation.
Include the required permissions for new private config and state directories,
and create them with those permissions from the start.
Ask for approval before stopping work, replacing an environment, deleting
anything, changing security/access,
or installing or restarting persistent Worker services. Use authorization
already given for the same concrete plan; do not ask again at each command.
Ask again only for a new action or risk outside that plan. A sudo password or
provider login may still need the user in their own terminal.

Separate these cases:

- **New Monarch installation on existing ComfyUI:** preserve its working CUDA
  Torch build and other custom nodes. Choose a recovery method for the actual
  changes; this does not require archiving every environment or restoring the
  whole pair afterward. Install only after checking the dependency
  plan and confirming the affected environment is idle.
- **ComfyUI missing:** include this prerequisite phase using
  the [official manual-install guide](https://docs.comfy.org/installation/manual_install),
  with an isolated environment and a compatible version. Include this phase
  in the initial approval if discovery already showed it was needed. Monarch's
  setup does not install ComfyUI, the GPU driver or the OS.
- **Already installed:** inspect first. If source, dependencies, configuration,
  model files and service ownership match, reuse them and validate. Do not
  reinstall packages or rerun service installation just to repeat setup.
- **Fresh-install acceptance on a working pair:** agree on an isolated baseline
  and restoration plan before maintenance. A new config path is not enough:
  `dgxm-worker.service` and the managed source/state paths use the real user's
  home. Include `~/.cache/dgx-monarch` in that boundary: it holds consent state
  (`consent_memo.json`) and model-family information (`family_memo.json` by
  default). Inspect the configured paths; a new environment or config does not
  make those records new. Agree whether to preserve, reuse, or isolate them,
  and retain any existing records for restoration. `HOME` or XDG overrides
  do not safely isolate a second installation.
  Do not move existing state, remove units or start a second fleet without
  the specific approved plan.

Normal onboarding does not require the fresh-install acceptance procedure.
Use that procedure only when the user asks for an isolated test on an occupied
pair. Keep model files, credentials, unrelated work and the existing working setup.
A copied environment or `--system-site-packages` environment must be reported
as reused dependencies, not a fresh dependency installation. Global package,
model-download, compiler and GPU caches may be reused when the approved plan
allows it; record which were reused. Do not clear them to make the test appear
cleaner. A fresh package installation does not imply a cold-cache test. Do not
claim a clean-machine test when the OS, driver, ComfyUI or other prerequisites
were reused.

## 3. Install through the maintained source procedure

Set the discovered paths explicitly for each host:

```bash
export COMFY_DIR="/absolute/path/to/ComfyUI"
export COMFY_PYTHON="/absolute/path/to/python-used-by-ComfyUI"
```

Follow [INSTALL's dependency plan and apply steps](../../docs/INSTALL.md#install-from-source).
That is the canonical command sequence. Do not use `scripts/setup_env.sh` for
new-user onboarding: it is a developer-lab bootstrap with fixed paths and a
separate prerequisite history.

The required safeguards are:

1. Verify compatible Python, architecture, ComfyUI and CUDA Torch on every
   host. The Python minimum alone does not prove that compatible binary
   dependencies exist. Align the ComfyUI commit and required package versions.
2. Record Torch version, CUDA build and import path before installation. Apply
   its exact constraint to every relevant dependency resolution, including
   ComfyUI prerequisite work. Reject a plan that replaces the working Torch
   build; do not silently change system CUDA, drivers or NCCL.
3. Build the required xFuser compatibility wheel with
   `tools/build_xfuser_compat_wheel.py`. It verifies the official wheel and
   makes the optional NPU import lazy. Do not hand-edit an installed dependency
   or substitute the unpatched upstream package for the repository's pin.
4. Review the dry-run dependency report before applying it. Save reports and
   resolved versions privately. Keep versions consistent across hosts; open
   dependency ranges are not a promise of identical future resolution.
5. Apply the reviewed install using the chosen ComfyUI interpreter, then verify
   the original Torch build/import path, CUDA availability, the actual project
   import path and the exact Monarch/xFuser pins. Stop on incompatible binary
   imports. Do not copy a donor environment's packages to hide missing wheels.

Run `tools/check_dependencies.py` with the chosen ComfyUI interpreter as
described in INSTALL. It verifies the [known cuSPARSELt platform-tag failure](../../docs/TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform)
before accepting it. Retain any printed exception with the installation
record; do not describe that result as an error-free raw pip check. Stop if
the tool fails.

`dgxm` is installed beside `COMFY_PYTHON`; use that executable explicitly when
multiple environments exist. Establish `DGXM` as described in INSTALL before
using the guided setup examples. On a cluster, the configured Python path must
work on every host; ComfyUI paths can be specified per host. If existing paths
cannot meet that contract, propose an approved environment plan rather than
pointing workers at an arbitrary Python.

## 4. Configure the selected pair

Use [guided setup](../../docs/INSTALL.md#guided-multi-node-setup) as the default
config and service path. It probes only the hosts you name. Set `--python-bin`,
ComfyUI paths and explicit config `--output`. For first-render files stored
inside the ComfyUI tree, include their relative paths with `--artifact`.
The probe rejects paths that resolve outside that tree, including symlinks.
For an existing external model folder, independently verify the resolved files
on both hosts and record that setup's artifact check did not cover them; do
not move or duplicate the user's files without approval. Start with the `safe`
profile and inspect the complete dry-run plan before an approved apply.

Read the [fabric requirements](../../SECURITY.md#fabric-trust-boundary).
The Worker API has no peer authentication. The dedicated interface must be
source-restricted to trusted peers, including dynamic actor ports; opening only
port 26600 is insufficient. `--acknowledge-trusted-fabric` records the user's
confirmed isolation, not a network repair. Follow INSTALL's
[standard setup and optional network audit](../../docs/INSTALL.md#inspect-the-cluster-network).
The detailed audit is optional; record operator-confirmed and agent-verified
boundaries accurately. Do not require sudo collection for every setup or ask
the user to interpret firewall rules. Known exposure still needs correction.
Do not edit firewall rules, create keys, disable host-key checks or enable
lingering without explicit approval for that action. An approved item in the
installation plan is sufficient; do not request it again.
Keep native latent RDMA off.

All setup profiles write `auto_heal=true`. It can restart Worker services,
so service approval must cover that recovery behavior. If the approved plan
requires manual recovery, follow INSTALL's documented config-edit step before
starting the driver. The in-process attach retry is separate from `auto_heal`.

An existing unit, managed source or unknown ownership is a reason to inspect
and reuse or stop for a decision. Setup intentionally refuses to take over it.
Do not delete it to make an apply pass. Retain the applied setup's default
receipt and follow its failed/partial status before retrying.

Use the explicit config path for every `dgxm` command and the ComfyUI process.
Pass it to the graph's Init node too. Do not let discovery pick another user's
config or silently turn the intended distributed render into a local one.

## 5. Verify a real first render

Use a supported model the user already has or chooses. Check its exact
checkpoint, precision and two-Spark configuration in [MODELS](../../docs/MODELS.md),
then use the corresponding repository workflow and its documented settings.
If the user has no preference, recommend the optional
[Chroma recipe](../../docs/QUICKSTART.md#first-distributed-render). Its
[file table](../../docs/QUICKSTART.md#first-render-files) provides a complete
starting point; Chroma is not required for installation acceptance.

For the selected workflow, identify every model file, destination, official
source and license requirement. Check published hashes where available and
verify matching actual bytes on both hosts. Record when no publisher hash is
available; do not invent one. Reuse matching files and investigate mismatches
without overwriting them. Do not bypass download gates or licensing conditions.

After setup, run Doctor and status using the explicit config. Resolve failures
without weakening checks. Doctor and setup's short cluster smoke are useful
prerequisites, but neither proves a completed distributed render.

Start ComfyUI in a tracked foreground terminal/session with the selected
interpreter and config. Use explicit `mode=cluster`, a documented two-rank
configuration for the selected model, and `auto_gate=first_use`. Keep native
latent RDMA off. Prefer stock residency and modest inputs where supported;
follow the model's documented attention and memory requirements. Chroma's cfg2
layout, sampler and step count apply only to its recipe. Use the repository
workflow and retain its required inputs. Do not enable untested optimizations
or bypass a failed gate to make setup pass.

Require a completed queue and a saved image, video or audio output that can be
opened in an appropriate viewer or player. Save its path and hash,
the submitted workflow/settings, and per-rank source and render evidence that
identifies two ranks on two distinct Sparks. Correlate it with the same queue.
A local render, imports, two reachable services or green Doctor checks alone
do not satisfy this requirement. Keep host identities and raw logs private;
publish only reviewed, sanitized results.

For setup reruns, repeat discovery and read-only validation first. A healthy
matching installation should need no package, unit, config or model rewrites.
Compare source, package versions, config/unit hashes and Worker identities
before and after. Queue the same recipe with a different seed to avoid counting
a cached image as another successful render. Record any required intervention.

## 6. Recover and hand back control

On failure, retain logs and receipts. A failed import is not permission to
replace Torch; an attach timeout is not permission to kill every Python
process. Use [troubleshooting](../../docs/TROUBLESHOOTING.md) for the observed
error and preserve ownership across cleanup. If a service change is partial
or unknown, inspect its recorded state before another mutation.

Stop only a test driver and mesh you own. Confirm actor teardown before reuse.
Do not automatically drop OS caches, restart services, change clocks, clear
failure records or remove dependencies. Those actions require a concrete
reason and approval. Never stop unrelated work or bypass a comparison failure.

For an acceptance test on a working pair, confirm whether the user wants to
retain the test installation or restore the original setup. Carry out that
approved choice and verify it, including source, configuration, dependencies,
service health and absence of test actors. A rollback plan must name saved
paths and how to restore them; do not invent one after a partial failure.

Normal setup retains the working installation. Hand over a usable foreground
session in the user's terminal when possible. If an agent-owned test driver
was stopped for cleanup, say it is stopped and supply the verified start
command; do not imply its browser URL is currently available.

Start the final response with the [launch handoff](../../docs/INSTALL.md#launch-and-handoff):
start command or shortcut, browser access, actual running state and permanent
installation/workflow/output locations. Keep evidence-directory paths separate.
For an accepted TUI, prefer `dgxm top` once its environment and discovery are
verified; explain activation once and add explicit paths or options only when needed.
Then report the tested commit, Python/ComfyUI/Torch/CUDA and dependency
versions, reused components, saved output and two-host proof, rerun result,
manual steps, failures/fixes, final retained/restored state and limitations.
Label unperformed checks **NOT RUN**. Keep credentials and machine-specific
records out of the repository. Do not merge, publish or create releases unless
the user has authorized those actions.

## Daily operation after setup

ComfyUI stays in a visible foreground terminal. Never launch it with `&`,
`nohup`, `disown`, or an untracked process. Include terminal use in the approved
launch plan. If the agent cannot retain a foreground session and handle its
signals, ask the user to run the launcher in their own terminal.

Worker services are persistent. After an operation that can change them, run
`dgxm status` and tell the user exactly which services remain running. Include
`dgxm down` as the intentional stop command. Do not stop services merely
because ComfyUI exits.

Use `scripts/comfy-driver.sh` for the optional foreground launcher and
`scripts/dgxm-desktop.sh` for an explicitly requested desktop shortcut.
[INSTALL](../../docs/INSTALL.md#updating) documents verified updates and recovery.
[MODELS](../../docs/MODELS.md) documents additional workflows and settings;
[CLUSTER](../../docs/CLUSTER.md) and [TUI](../../docs/TUI.md) cover configuration
and monitoring. The optional `comfy_managed` Init widget changes how workers
place weights; leave it disabled for the first render. Read
[docs/TROUBLESHOOTING.md #62](../../docs/TROUBLESHOOTING.md#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)
before enabling it or changing residency. Do not generalize the first render
test to other models, precisions, memory modes or native RDMA.
