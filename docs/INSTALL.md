# Install from source

dgx-monarch installs from a Git checkout inside ComfyUI's `custom_nodes`
directory. ComfyUI is a prerequisite. If it is missing, follow the
[official ComfyUI manual-install guide](https://docs.comfy.org/installation/manual_install)
before continuing. The dgx-monarch setup commands do not install ComfyUI.

## Inspect before installing

Requirements: Linux, Python 3.11 or newer, and the CUDA-enabled torch build
already used by ComfyUI. The Python minimum alone does not guarantee compatible
wheels for every architecture. Check the selected interpreter, CUDA support and
available dependency wheels before changing an environment. Both Sparks need
matching Python, torch, torchmonarch and ComfyUI versions; guided setup also
requires clean Git ComfyUI checkouts except for the missing example files
described below. Use the current requirements in
`pyproject.toml` and `requirements.txt`, not a version copied from another rig.

First discover the ComfyUI checkout and the interpreter that actually starts
it. Inspect its launcher or service configuration without printing credentials
or whole process environments. Record the ComfyUI commit and working-tree
status, Python version, torch version, `torch.version.cuda`, and package import
locations. Inspect the existing `custom_nodes/dgx-monarch` path, including any
symlink target, Git origin, commit and local changes. Confirm that an existing
import of `dgx_monarch` comes from that intended installation. Do not print a
Git remote URL containing a token; record only its host and repository name.

Reuse a matching, healthy installation. Check its configured cluster, model
files and health without reinstalling packages or repeating service setup.
If the checkout is unexpected, modified, broken or points elsewhere, explain
the difference and agree on a repair before writing to it. Never overwrite an
existing path or create a second copy to hide an import conflict.

Guided setup permits only unstaged deletions of ComfyUI's tracked regular
files `input/example.png` and `output/_output_images_will_be_put_here`.
It preserves those deletions, records the checkout as modified and emits a
warning. This exception does not cover staged changes, other missing files,
new files, renames, symlinks or Git flags that hide changes. Do not reset a
checkout just to remove this warning.

If ComfyUI is missing, treat its installation as a separate prerequisite phase
with the user's approval, included in the initial plan when known. Follow the
official guide linked above and verify
that it runs on CUDA before returning here. Do not replace system Python,
global torch, or another application's environment. Do not copy packages from
a donor environment. `scripts/setup_env.sh` is a developer-lab helper with
fixed environment and torch choices, not this onboarding procedure.

When reviewing a prerequisite's dependency report, check each artifact's URL
and hash. An official Torch index can link to dependencies hosted by NVIDIA or
the Python Package Index as well as PyTorch. Verify those sources against the
selected official installation instructions rather than requiring every wheel
to share the index's hostname. Keep credentials out of saved URLs.

### Caches and scratch during an isolated test

This section applies to an explicitly requested isolated acceptance test on
a working pair. Normal installation reuses existing caches and does not
require a full environment archive or restoration afterward.

Agree on cache reuse before the first CUDA prerequisite check, since even that
check can create cache files. Set supported cache locations before importing
Torch or probing CUDA, and record which existing caches remain in use.
`~/.cache/dgx-monarch` holds consent and model-family memos by default; these
paths do not follow `XDG_CACHE_HOME`. Preserve them under the approved test and
restoration plan rather than assuming a new environment isolates them.

Driver exports do not establish Worker cache isolation: Worker services filter
their inherited environment. They also replace caller `TMPDIR` with private
scratch under the trusted user runtime directory, falling back to the user's
home state directory or system temporary directory. Native dependencies can
write logs or caches at other default paths. Record the observed locations and
reuse; do not weaken the environment filter, edit a managed unit, or clear
unrelated caches to claim a cleaner installation.

<a id="install-from-source"></a>

## Install the node pack

Before changing an existing environment, choose a recovery plan matched to
the proposed changes, such as a backup of the affected environment or an
isolated copy. Include the known prerequisite, package, service and access
changes in one plan for approval. Reuse that approval for the same listed
work. Stop ComfyUI and any services using that environment only within the
approved maintenance window. Keep model files, credentials,
other custom nodes and unrelated work intact.

Set these paths explicitly. `COMFY_PYTHON` must be the interpreter that starts
this ComfyUI checkout. If the intended node-pack path is absent, clone
`https://github.com/Deen-Media/dgx-monarch.git` there and select the same reviewed
commit on both hosts. An existing checkout or symlink must pass the inspection
above before reuse.

```bash
export COMFY_DIR="/absolute/path/to/ComfyUI"
export COMFY_PYTHON="/absolute/path/to/python-used-by-ComfyUI"
export REPO_DIR="$COMFY_DIR/custom_nodes/dgx-monarch"
test -f "$COMFY_DIR/main.py" && test -x "$COMFY_PYTHON"
"$COMFY_PYTHON" -c 'import torch; assert torch.cuda.is_available(), torch.__version__; print(torch.__version__, torch.version.cuda, torch.__file__, "cuda ok")'
```

Stop if either check fails. Do not substitute another interpreter to make the
check pass. Prepare the xFuser wheel and torch constraint before any pip
resolution, then save a dry-run report:

```bash
XFUSER_WHEELS="$(mktemp -d)"
"$COMFY_PYTHON" "$REPO_DIR/tools/build_xfuser_compat_wheel.py" \
  --output-dir "$XFUSER_WHEELS" &&
  TORCH_BUILD="$("$COMFY_PYTHON" -c 'import torch; print(torch.__version__)')" &&
  printf 'torch==%s\n' "$TORCH_BUILD" > "$XFUSER_WHEELS/torch-constraint.txt" &&
  "$COMFY_PYTHON" -m pip install --dry-run \
    --report "$XFUSER_WHEELS/install-plan.json" -e "$REPO_DIR" \
    --find-links "$XFUSER_WHEELS" \
    --constraint "$XFUSER_WHEELS/torch-constraint.txt"
```

Review the report privately before applying it. Refuse a plan that installs,
replaces or removes torch, including a different build with the same public
version. Review other proposed dependency changes against ComfyUI's needs.
If resolution fails, diagnose the conflict; do not remove the constraint or
use `--no-deps` to bypass missing runtime dependencies. Do not put repository
or model-host tokens in commands, URLs or reports. Use an existing credential
helper or have the user authenticate through the provider's normal flow.

After reviewing the plan, run the matching install command:

```bash
"$COMFY_PYTHON" -m pip install -e "$REPO_DIR" \
  --find-links "$XFUSER_WHEELS" \
  --constraint "$XFUSER_WHEELS/torch-constraint.txt"
```

The builder downloads the official xfuser 0.7.0 wheel unless `--wheel` names a
local copy. It checks the SHA-256 and wheel contents, applies the lazy
optional-NPU-import patch, and produces `xfuser==0.7.0+dgxm.npuimport1`.
The editable install also brings in the exact `torchmonarch` pin
([DESIGN.md §6.3](DESIGN.md#63-versioning-and-compatibility-policy)). First
installation needs network access; it must not replace ComfyUI's CUDA torch.
Use the same constraint and dry-run review for later optional extras too.

After installation, compare torch version, CUDA version and import location
with the recorded values, confirm CUDA availability, and run the dependency
checker with this interpreter:

```bash
"$COMFY_PYTHON" "$REPO_DIR/tools/check_dependencies.py"
```

Then check imports for Monarch, xFuser and ComfyUI. Record the
resolved package versions and node-pack commit. Preserve the install report
with the private test record; remove only the temporary directory created
above when it is no longer needed. This procedure pins Monarch and xFuser;
it does not lock every transitive dependency.

The checker verifies the [known cuSPARSELt platform-tag failure](TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform)
before accepting it on Linux ARM64. Save any printed exception with the
installation record. A failed check must be resolved before continuing.

The installer places `dgxm` beside `$COMFY_PYTHON`. Invoke that executable so a
shell's unrelated `dgxm` cannot select another environment:

```bash
export DGXM="$(dirname "$COMFY_PYTHON")/dgxm"
```

Start ComfyUI with its normal foreground command. Do not hide or background
the process. The optional [driver launcher](../scripts/comfy-driver.sh) also
keeps ComfyUI attached to the invoking terminal. ComfyUI registers node packs
and their template folders when the server starts, so start it after the
install.

On a two-host rig, install on both hosts and run guided setup before you start
ComfyUI: setup refuses to install or start a Worker service while a driver
runs ("Multi-node source installation" below).

### Launch and handoff

Keep permanent files in the user's chosen locations. Reuse suitable existing
ComfyUI and Python installations; agree on durable paths for new ones during
planning. Temporary and chat-output folders are for evidence or explicitly
requested temporary tests, not the normal installation.

Offer an optional easy launch script and ask whether the user wants it and
where to save it. Their home directory, `~/.local/bin` or another folder they
choose can work; do not silently choose the Desktop or a chat folder. Check
for an existing file and ask before replacing it. The script should only set
`DGXM_CLUSTER_TOML` to the resolved config and `exec bash` the maintained
`scripts/comfy-driver.sh` with the resolved repository, ComfyUI and Python
paths and the selected launch options. Forward arguments with `"$@"` so
`--check` works. Preserve foreground execution and signals; do not add Worker
restarts or a second installer.

Validate the exact launch command, or the chosen script, with `--check` before
using the same paths, config and options for the first render. This checks
launcher arguments and paths; it does not prove a running browser service.
Keep any required user terminal open for normal use. If cleanup stops the
agent's test driver, tell the user that ComfyUI is stopped and provide the
working command to start it.

Lead the handoff with these details, before the test report:

- The exact start command or shortcut path, and which Spark runs it.
- Whether ComfyUI is currently running, plus the actual browser URL verified
  while it ran. A localhost URL is for a browser on that Spark. For another
  machine, use an approved SSH tunnel if needed and verify its endpoint; never
  widen listening addresses or firewall access silently. If stopped, label
  the URL as available after launch.
- If the user chose the TUI, lead with `dgxm top` after verifying that `dgxm`
  resolves to the selected environment and default discovery finds the intended
  config and driver. Give the exact environment-activation command once if
  needed. Use the full executable path or explicit `--config` and `--host`
  options only when that setup needs them, with actual values rather than
  placeholders. Live render data needs ComfyUI serving the Monarch telemetry
  endpoint; running Workers alone do not supply it. See [TUI](TUI.md).
- The saved selected workflow and configured render-output folder.
- The actual Monarch checkout, ComfyUI, Python environment and cluster config
  paths, separately from the private evidence directory.
- How to stop or restart ComfyUI from its foreground terminal. Worker services
  persist when ComfyUI closes; use `dgxm down` only for an intentional Worker
  stop, and report their observed state.

A normal installation remains installed and ready for use. Restoration belongs
to an explicitly requested temporary acceptance test or an approved rollback.

If the user wants a desktop entry with the project icon, install it after the
source install:

```bash
bash "$COMFY_DIR/custom_nodes/dgx-monarch/scripts/dgxm-desktop.sh" install \
  --comfy-dir "$COMFY_DIR" --python "$COMFY_PYTHON"
```

The shortcut opens a terminal, starts only ComfyUI, and opens the browser once
the server is ready. It does not save a custom `DGXM_CLUSTER_TOML` value passed
to the installer; use the chosen launch script when a custom config path is
needed. See [scripts/README.md](../scripts/README.md) for options.

For the first distributed render, [choose a supported model and workflow](QUICKSTART.md#first-distributed-render)
you already have or want to use. Chroma is optional; its
[file list](QUICKSTART.md#first-render-files) is a starting point if you have no
preference. Use the selected workflow's files, download sources and access
requirements. Start ComfyUI with
the explicit cluster config selected below. `mode=auto` without a config can
run locally, so a saved image alone is not proof that both Sparks rendered it.
Keep the workflow, saved output and evidence of both ranks' participation.

Local mode is separate: it uses no persistent Worker service and creates its
actors on the driver. A local render is useful for diagnosis but does not
complete two-Spark installation acceptance.

Cluster mode has a separate lifecycle. `dgxm up`, guided setup with
`--start-worker-service`, auto-heal, and installed user services can leave
Worker services running after ComfyUI exits. The launcher neither starts nor
stops those services. Run `dgxm status` after a lifecycle action and use
`dgxm down` only for an intentional stop.

`dgxm up` succeeds only once the exact Worker process owns its configured
listener and its orphan-actor sweep has settled; `down` also settles the
stopped service's actor sweep. Transport loss, a timeout, a missing terminal
marker or an unavailable sweep produces an unknown result and a nonzero exit status. `restart` syncs and starts nothing unless `down` definitely
succeeded. `status` exits nonzero for a dirty or replacement-blocked Attached
mesh even when every per-host process and listener row is healthy.

## Model files required by each template

`example_workflows/artifacts.toml` lists model files for the public examples
and generated test fixtures under `tests/fixtures/workflows/generated/`.
For installation, use the files listed for your selected public workflow.
Test placeholders are for contributors and are not downloads. Each file has
a models subfolder, source URL, size and sha256; each workflow lists the files
it needs. Shared files,
such as the Flux VAE and UMT5 text encoder, have one row each. Demo image,
video and audio inputs are excluded.

Check which files are already installed:

```bash
python tools/check_artifacts.py --models-dir ~/ComfyUI/models
```

The check is offline: it opens no socket and downloads nothing. It prints one
line per row (present, absent, unpinned or placeholder), except a staged file
whose size disagrees with the manifest: the check names that file on stderr
and fails. `--hash` also hashes each staged file whose row carries a `sha256`,
reading each in full, and treats a hash mismatch the same way.
Without `--models-dir`, it performs CI's repository checks: valid rows, a row
for every template model file, and a template reference for every row.

A row with a blank `url` or `sha256`, or a `size` of 0, is pending. The check
reports pending rows and passes; `--pending` names them, and `--strict` fails
on them once the set is filled. `--capture` with the same `--models-dir` first
syncs, then sets `size` and `sha256` from each staged file, replacing any value
already there, and writes the manifest back; it never fills `url`. Rows
marked `placeholder = true` name example LoRAs to replace with your own files;
they remain placeholders and do not count as pending.

After a template changes which files it loads, rebuild the rows and
per-template lists with `python tools/check_artifacts.py --sync` and commit
the result.

## Multi-node source installation

On every host (driver and workers):

1. The source checkout and its `"$COMFY_PYTHON" -m pip install -e` command,
   each checkout directly under that host's ComfyUI `custom_nodes` directory.
   The ComfyUI path may differ per host, and `COMFY_PYTHON` names that host's
   ComfyUI interpreter. Every Worker service runs the one `python` path in
   `cluster.toml`, so the interpreters must share a path: absolute, or under
   each home (`~/`). Keep the torch and CUDA builds aligned: `dgxm doctor`
   reports `FAIL` for torch version skew across hosts or against the driver.
   It also reports `FAIL` for a torchmonarch version off the pin and for
   `NCCL_PROTO=LL` in the driver environment or a worker shell. Fabric
   profiles reject every `NCCL_PROTO` value at config load; leave protocol
   selection unset.
2. A ComfyUI checkout on every host. Workers import ComfyUI code but do not run
   its server.
3. The model files under the same relative names (`models/diffusion_models/…`,
   `models/loras/…`). Workers resolve model names against their local
   directories. Check all files required by the selected workflow on both
   hosts; the [Chroma file list](QUICKSTART.md#first-render-files) applies only
   to that optional recipe. Compare existing bytes before transferring anything, copy
   only missing approved files, and never replace a different file silently.
   Pass each required file to setup with `--artifact`, relative to ComfyUI.
4. Passwordless ssh from the driver to each worker host.
5. A dedicated, trusted network for all Monarch transport traffic, declared
   with `transport_security = "trusted_fabric"` under `[cluster]`; `dgxm doctor`
   fails without it. The attach API dgx-monarch calls has no peer
   authentication, and its control plane runs actor code.
   [SECURITY.md](../SECURITY.md#fabric-trust-boundary) gives the isolation
   rules, including why a rule for port 26600 alone is not enough.

### Inspect the cluster network

**Standard setup:** identify the selected fabric interfaces and addresses,
then reuse existing verified controls or obtain the operator's informed
confirmation that the [documented boundary](../SECURITY.md#fabric-trust-boundary)
is in place: the complete Monarch transport surface uses the dedicated fabric,
all traffic on that interface is source-restricted to the trusted driver and
peers, and it is not exposed to a shared LAN, Wi-Fi or the Internet. A direct
cable or private IP address alone is not that confirmation.

Record whether the basis is operator-confirmed or agent-verified. Operator
confirmation does not mean the agent audited the firewall. Standard setup does
not require a firewall dump, sudo collection or a new audit each time. The
trusted-fabric acknowledgement remains required and never configures isolation.

**Optional network audit:** use this when the user requests it, is unsure about
the boundary, or a specific observation raises a concern. Begin with the local
read-only collector from each host's repository checkout:

```bash
python3 tools/collect_network_state.py
```

It writes a private report and prints its path. The default uses no sudo and
may report incomplete firewall access. If the specific check needs
administrator access, give the user one command per host with that host's
actual checkout path:

```bash
python3 /absolute/path/to/dgx-monarch/tools/collect_network_state.py --sudo
```

The user runs it in their terminal and supplies the saved path or confirms
completion. The agent reads and interprets the report locally. Do not ask the
user to paste raw firewall rules into chat or decide whether the rules are
safe. Keep reports private. The collector makes no network or firewall changes.

A complete report is not a safety verdict. Interpret the configured rules,
routes, selected fabric interfaces and physical topology together against
SECURITY. Forwarding being enabled or UFW being off alone does not establish
exposure and does not require sudo collection. If a check is inconclusive,
name the specific missing fact. Known exposure or evidence contradicting the
required boundary cannot be overridden by an acknowledgement; explain it and
request approval for a narrow fix with rollback. Never apply firewall changes
as a side effect of collection.

Use guided setup below for the usual installation. As a manual alternative,
`cluster.example.toml` is a complete config for a back-to-back DGX Spark pair.
After checking that the destination is absent or approved for replacement, copy it to `~/.config/dgx-monarch/cluster.toml`, replace every `192.0.2.x`
address (six sites, `client_bind` and `nccl_master_addr` among them), and set
`python` to the shared interpreter path above and `ssh_key` to a key every
host accepts. The attach refuses a config with no `client_bind`, because the
default transport would advertise whatever the driver's hostname resolves to,
often loopback, and the attach would time out ([TROUBLESHOOTING #1](TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout)). The
example names one QSFP port set's interfaces. Each worker detects its active
RDMA rails as it builds its actor environment and rewrites the names when the
other set is active; it logs the change, and the `fabric ifaces` doctor row
reports it.

### Guided multi-node setup

Choose one config writer: edit
`cluster.example.toml` by hand, run `dgxm setup`, or run the legacy
interactive `dgxm init`. After a hand-edited or `dgxm init` config, run
`dgxm up`, then `dgxm doctor`.

Stop ComfyUI on the driver before an apply that installs or starts a Worker
service. Setup refuses that transaction while a local driver, a Render session
or a lease is active, and also whenever it cannot prove all three inactive.
Start ComfyUI again once `dgxm doctor` is green.

During discovery, check the [user service session](#check-the-agents-user-service-session)
and [process inspection capability](#protected-process-inspection) on each host.
Include any required helper in the installation plan before apply. For normal
setup, omit `--receipt` to use the maintained private receipt directory; do not
choose a path under the chat workspace. Check customized state paths using
[Operator receipts](#operator-receipts).

Review the setup plan on the driver before applying it:

```bash
export CLUSTER_CONFIG="/absolute/path/to/cluster.toml"
"$DGXM" setup \
  --host 'spark-a,192.0.2.11,1' \
  --host 'spark-b,192.0.2.12,1' \
  --client-ip 192.0.2.11 \
  --python-bin "$COMFY_PYTHON" \
  --comfy-dir "$COMFY_DIR" \
  --profile safe \
  --output "$CLUSTER_CONFIG" \
  --acknowledge-trusted-fabric       # probe + plan/diff; no mutation
```

Once approved, repeat the reviewed command with
`--apply --install-service --start-worker-service --verify`.
After it succeeds, check the installed cluster:

```bash
"$DGXM" --config "$CLUSTER_CONFIG" doctor
"$DGXM" --config "$CLUSTER_CONFIG" status
```

Use `safe` for the first setup. Both `safe` and `balanced` write
`auto_heal = true`, which can restart Worker services during attachment;
starting a render therefore needs the agreed service authority too. If a test
requires all restarts to remain manual, review an explicit config edit to
`auto_heal = false` after setup, before rendering. That edit is a separate
manual boundary, not a setup flag, and changes the config digest recorded by
the setup receipt. Record the edited config and rerun its validation.

Pass `--config "$CLUSTER_CONFIG"` to subsequent `dgxm` commands and select the
same config in the ComfyUI process:

```bash
DGXM_CLUSTER_TOML="$CLUSTER_CONFIG" \
  bash "$REPO_DIR/scripts/comfy-driver.sh" \
  --comfy-dir "$COMFY_DIR" --python "$COMFY_PYTHON"
```

Keep ComfyUI in this foreground terminal. Use existing approved SSH access and
verify the destination host identities. Ask before creating keys, changing
SSH access, enabling lingering, installing privileged helpers or changing
firewall rules. Explicitly listed actions may share the installation-plan
approval; already approved actions do not need another prompt. The trusted-fabric
acknowledgement does not configure or verify a firewall; follow [SECURITY.md](../SECURITY.md#fabric-trust-boundary)
before granting it. Never print credentials or copy them into a receipt.

The `192.0.2.x` values are documentation addresses; replace them with the
literal addresses on the dedicated, source-restricted cluster fabric. This
example makes `spark-a` a driver-colocated Worker. `dgxm update --verify`
supports this two-Spark layout (see [Updating](#updating)). The
[operator checks](VALIDATION.md#operator-checks) record its tested scope.

`dgxm setup` does not scan the LAN. You name every candidate host, fabric
address, GPU count and optional artifact path, and setup probes only those
targets, locally or over SSH. A dry run is the default: it validates the
strict rendered config, reports reachability, accelerator and UMA
observations, the fabric recommendation, service state and cross-host artifact
agreement, then prints the exact config diff. `--apply` makes the changes,
after a confirmation prompt unless `--yes` is given.

Each `--host` value is `NAME,FABRIC_IP[,GPUS[,SSH_USER[,COMFY_DIR]]]`. `NAME`
cannot contain `@` and may contain `:` only as a literal IPv6 address;
`SSH_USER` may contain neither. With no `--host`, setup asks for each host
interactively; it never discovers hosts. `--output` sets the config
destination, `--fabric-profile` overrides the inferred fabric profile, and
each `--artifact` names a path relative to ComfyUI whose bytes must match on
every host. Its resolved target must stay inside that ComfyUI tree; an external
model cache or symlink to one is rejected. Independently verify external model
files on both hosts and record that setup's artifact check did not cover them.
`--python-bin` sets the interpreter every host shares (default
`python3`; pass the interpreter that has torch and torchmonarch), `--ssh-key`
the SSH identity, and `--comfy-dir` the default ComfyUI path, which a
per-host `COMFY_DIR` field overrides. The terminal diff can show the key and
checkout paths, so keep it private. `--json` prints a reduced plan and result
without the config diff, but it still holds versions, commits, model
fingerprints, counts and artifact, config and source hashes: review it before
you share it. A JSON apply requires `--yes` so that stdout stays one JSON
object; read the terminal plan first. Use the default receipt destination
unless a custom path is needed; see [Operator receipts](#operator-receipts).

#### Check the agent's user service session

Before interpreting a Worker service warning or changing a service, verify
that the current shell can query the intended user's service manager:

```bash
systemctl --user show dgxm-worker.service \
  --property=LoadState,ActiveState,SubState,MainPID
```

If this fails with `Failed to connect to bus: No medium found`, the shell may
lack the login session's `XDG_RUNTIME_DIR` or `DBUS_SESSION_BUS_ADDRESS`.
An agent shell can have this problem while the Worker service is healthy.
In that case, a `dgxm status` warning such as `STRAY nohup loop` is not enough
evidence to restart, remove, or replace the service.

Repeat the read-only query through an existing approved SSH or login session
on that host, as the same user who owns the service. Keep normal SSH host-key
verification enabled. Alternatively, use session metadata to discover that
user's runtime directory and bus address, verify the directory and socket
belong to that user, and set only the current shell's session variables before
querying again. Do not guess a UID or borrow another user's bus. Compare the
reported service PID and state with the observed Worker process, then rerun
`dgxm status` with the explicit cluster config from the working session. If no
valid user session is available, ask the user to establish one; do not create
a replacement service or change lingering to hide a failed query.

#### Protected process inspection

Some Linux sessions cannot read the environment markers of protected
processes. Setup then reports `service_ownership_unknown`, because a process
name alone cannot prove a process is unrelated to Monarch. Keep the receipt
and leave the ownership guard in place.

Check the capability early on each host, from the selected repository:

```bash
python3 tools/process_inspection.py --check
```

Run the check as the installation user in the same approved host or SSH login
session used by setup. If an agent sandbox blocks `/proc`, retry through an
existing approved, unprivileged host session before concluding root access is
needed. Sandbox denial alone is not a host permission failure; do not disable
the sandbox or bypass access restrictions.

This checks whether the current session can read process metadata. It does not
prove process ownership or that no work is running. A denied or incomplete
check means the required ownership check may need privileged inspection; it
does not justify killing a process or weakening the guard.

When needed, include the read-only inspector in the approved installation
plan. Review `tools/process_inspection.py` and
`src/dgx_monarch/cli/process_inspector.py` first. Reuse an existing helper only
when its authenticated reply passes setup's identity and freshness checks;
a socket path or old ready message is not sufficient.

Otherwise, give the user this command with each host's actual repository path.
Run it as the normal user in a foreground terminal on each host that needs it:

```bash
python3 /absolute/path/to/dgx-monarch/tools/process_inspection.py --serve
```

The launcher prompts for sudo, copies the inspector to a private root-owned
directory, checks its hash and runs it with isolated system Python. It changes
no sudo policy or kernel setting. Protected process metadata can still require
one open foreground terminal per host; do not background the helper.

The agent should verify the authenticated reply through setup rather than
asking the user to confirm it when that reply is available. Add
`--privileged-process-inspection` to both the reviewed setup command and its
apply in the setup terminal. The plan and receipt record this choice; cluster.toml never does. The
helper reads only the invoking user's process markers and returns bounded
Worker and actor identities and endpoint fingerprints. It cannot start or stop
processes or expose command lines or environment values. Before setup uses a
reply, it checks that the peer is root, that the helper source hash, UID and
boot id match, and that the reply is fresh. Missing, expired, incompatible or
incomplete inspection stays unknown; nothing skips the protected processes.

The launcher uses `--lifetime 1800`, the helper's 30-minute maximum;
Ctrl-C stops it sooner. Keep it running until setup and any recovery have
settled. If it expires during an uncertain transaction, keep that
transaction's receipt and recovery state: a new helper alone does not
authorize repeating an ambiguous change. Setup never runs sudo or starts the
helper itself.

The sidebar and `dgxm top` receive fresh passive service observations from
the driver. An initial, stale or failed observation can show UNKNOWN even
after a successful setup. Follow the [readiness explanation](CONCEPTS.md#operator-words)
and inspect the reported cause; UNKNOWN alone does not mean a privileged
helper or service restart is needed. A past setup result never replaces a
fresh service-identity check.

#### Prerequisites checked by setup

Guided setup does not install or repair Python, CUDA, torch, torchmonarch,
ComfyUI, rsync, systemd or lingering. Every candidate must already have
Python 3.11 or newer, the declared GPU count, CUDA-capable torch, the exact
torchmonarch pin and a clean Git ComfyUI checkout, apart from the
[narrow missing-example exception](#inspect-before-installing). The Python,
torch and torchmonarch versions and the ComfyUI commit must match across hosts. A unit
installation, started or not, also needs rsync, systemd user services and
lingering on each host. The dry run reports each missing prerequisite as a
blocker against its host; fix it there and rerun the dry run.

Setup profiles (`--profile safe|balanced|advanced`, default `balanced`)
compile into existing strict config keys and graph advice; the receipt
records the profile and cluster.toml does not.
[CLUSTER.md](CLUSTER.md#guided-setup-profiles) holds the exact mapping and the
native-RDMA restrictions.

#### Service installation and verification

Writing the config alone does not install or start Worker services.
`--install-service` installs the unit and leaves it inactive;
`--start-worker-service` also starts it and requires `--install-service`;
`--verify` requires both. So the `--verify` smoke's attach may use configured
auto-heal to restart an unhealthy Worker service only as part of the service changes you approved. Every probed unit and enablement link must be definitely
absent and every Worker service definitely inactive. Setup never takes over a
pre-existing unit: an existing or stale enabled unit is kept and refused even
when inactive, because setup may start only the unit and remote source slot it
created, or the local source binding it verified, in the same transaction. The
managed `~/.local/share/dgx-monarch/src` path must be absent too: a `dgxm up`
deployment already there belongs to an earlier deployment, and setup does not take it over.
If setup cannot determine the state, it makes no changes. `--yes` skips only the
final plan prompt; it bypasses no unreachable-host, trust, artifact, profile,
ownership or verification check.

A successful installation is not recreated on every rerun. Inspect and reuse
its exact config, source, interpreter and services, then run doctor, status and
model checks. Repeating `--apply --install-service` against it is expected to
refuse. If a fresh test is required, agree on the maintenance and rollback
plan first. A different config, virtual environment or `XDG_STATE_HOME` does
not isolate the fixed `dgxm-worker.service` name, managed source link,
generation markers or lifecycle locks under the same user. Preserve those
existing artifacts and the working installation before any approved changes;
never delete them just to get past a setup refusal.

With `--verify`, setup runs doctor and a short cluster test. The test attaches
a data-parallel mesh, establishes NCCL, and checks every rank's status and
source version before and after the collective. It then retires only the
ProcMesh it created. Cleanup never calls `hosts.shutdown()`.

After confirmation and the config lock, and before changing any config or service, setup repeats the checks on the
reviewed hosts, model files and local source manifest. A changed or missing
result stops the apply.
The config file itself is rechecked when it is published (below). A service
change also needs definite evidence that no Worker process, listener or actor
remains on any target host, and the driver activity check above is strict: an
incomplete local process scan reports `service_activity_unknown`, a
Comfy-shaped candidate process counts as busy, and only a complete scan that
finds none, with a definite answer from every ComfyUI port, proves the driver
idle. Setup never turns an unknown observation into cleanup authority, and it
refuses, before any effect, a target it cannot place as local or remote.

When service changes are requested, setup finishes installing and starting
its source and units before writing the new config. Uncertain
activity or ownership, a failure to prepare or activate the source, or an unresolved
service outcome therefore leaves the old config bytes in place; the new config is written atomically only after service success is confirmed. A definite service
failure can settle as `failed`; owned state whose cleanup cannot be confirmed
settles as `partial` and stays for inspection.

#### Recovery after an interrupted setup

Service installation and rollback must refer to the same process generation. The private
marker binds the setup token to the host boot and the exact Worker PID birth;
its inactive form is accepted only after complete evidence of no Worker and no
listener. A persistent fence covers start-time revocation, marker publication,
and compensation readback and mutation. Before its first owned stop, setup
fsyncs a stop intent bound to the transaction. If that stop's outcome is lost,
later settlement only reads state back and never issues a second stop; exact
inactive readback (`inactive`/`dead`, `MainPID=0`, no queued job, and zero
Worker/actor/listener runtime) may finish removing owned artifacts. Missing or
replaced generation or intent authority stays `partial` and keeps the
artifacts for inspection.

To bring an existing managed source or unit under guided setup, stop and
inspect that deployment, then use the verified update where it applies, or
retire or move the managed source yourself before you rerun setup. Setup
automates neither the cleanup nor the migration.

When the Worker and driver share a host, guided setup copies no source. It
verifies the imported `dgx_monarch` tree, publishes an owned managed `src` link to that
tree's source directory (for an editable install, the checkout's `src`), and
binds the new unit to that directory. Remote hosts instead get exact copies in
immutable release slots named for the transaction token, before their owned
`src` links and units are published. Verified update can migrate the
colocated unit to an immutable release slot ([Updating](#updating)).

Any change to the config bytes changes the Gate capability context, and the
plan says so before confirmation: earlier contextual `PASS` or `INCONCLUSIVE`
results cannot authorize optimized residency with the new config. A FAIL stays durable and
source-independent only for the same canonical model and configuration context, so an
old-context FAIL does not govern the new config digest.

Once mutation has begun, a `KeyboardInterrupt` or `SystemExit` first settles
any owned cleanup setup can prove, records unresolved steps as `unknown`,
attempts the receipt, and then re-raises the original interruption.
Cancellation after the smoke is dispatched neither erases a compensation
already observed nor authorizes a new rollback. A failed publication can still
leave settled state without a receipt: keep the private output and inspect
config, unit and source state before you retry. "Operator receipts" below says
when setup must write a receipt and when it may return without one.

Verification distinguishes failure from uncertainty. A doctor result of exactly
`False`, or a typed smoke failure with confirmed smoke cleanup, is
`verification_failed`: setup may compensate its owned services and, after
exact settlement, roll back the config. A doctor exception or non-boolean
result, a smoke transport exception, malformed or incomplete evidence, or
unconfirmed smoke cleanup is `verification_unknown`: the result is `partial`,
config and Worker service stay in place for inspection, and nothing is
compensated or rolled back.

#### Locks and config recovery

Worker service changes share one owner-only lifecycle lock per user on each
target host. `up`, `down` and `restart`, service install and removal, package
and pin syncs, guided setup publication and compensation, and the stop and
start legs of updates all take it. It serializes one host-side effect at a
time, not a whole multi-host command, and it cannot exclude a manual
`systemctl`, a signal, or hostile code under the same account. Status, plain
doctor, readiness and planning change nothing. Guided setup also binds remote
rsync to the exact reserved transaction and slot with a separate lock.
Contention, or lock evidence that is unsafe or changed, stops the operation before overlapping changes can occur.

Config changes use a separate owner-only lock for the normalized absolute
config path. Commands refuse if another writer holds it; they do not wait. Guided setup, `dgxm init`, safe doctor repair
and `dgxm uninstall` share it; uninstall holds it across service removal and
any confirmed, checked config unlink. Contention or an unsafe lock refuses
before any config or service change, and setup and repair also attempt a
failure receipt. Setup takes the lock after confirmation and holds it through
the recheck, the config and service effects, verification, the final config
readback, the receipt and any rollback. Its lock file and state directory are
private, opened without following symlinks and owned by the current user, and
every parent directory is checked for safe traversal.

An absent config is published atomically, without overwrite. For an existing
config, setup rechecks the device and inode, mode, digest and bytes just
before replacing it, and both publication and the final success check require
exact readback. Setup records the prepared inode before publication, so an
interruption between the call and its return can still recover the installed
file, and rollback runs only while that inode and digest are still current
under the same lock. Setup retains its private config and lock directories and
the exclusive `0600` backup of a replaced config, named by digest. Rollback
restores the active config and service state without deleting these recovery
artifacts.

This guarantee covers cooperating `dgxm` writers under a trusted account, not
any change under that account. The checks reject the drift they observe, but
Linux has no portable replace-if-inode-still-equals primitive: a manual or
hostile writer under the same account that lands between the final check and
the rename (or the checked unlink) of an existing target can be overwritten or
removed.

### Optional dashboard and Worker services

Install the final safetensors release (>= 0.8.0), not a pre-release. ComfyUI
brings safetensors in, and `0.8.0rc0` satisfies `>= 0.8.0` but lacks the pread
backend; the `safetensors pread` doctor row checks this and names the fix.

Offer the optional dgxm TUI during the initial setup questions, alongside the
launch-script choice. If accepted, prefer one combined install with the `tui`
extra. Follow the same dry-run and constrained installation procedure above,
using the verified xFuser wheel and preserved Torch constraint, changing
`-e "$REPO_DIR"` to
`-e "$REPO_DIR[tui]"` in both commands. If the temporary wheel directory has
already been removed, prepare it again first. Review the dependency plan and
verify torch afterward; do not run an unconstrained extras install. If the
user declines, leave the extra out. Use the chosen ComfyUI environment rather
than creating or overwriting another one.

`dgxm update` installs with `--no-deps`, so it never adds the extra.

Guided setup or `dgxm install-service` installs the systemd user units. If
lingering is absent, ask the user before enabling it with
`sudo loginctl enable-linger "$USER"`; it lets Worker services survive logout. With the unit in place, lifecycle commands go through systemd and
verify its state; without it they fall back to `nohup`. Plain remote package
syncs write to `~/.local/share/dgx-monarch/src`, which guided setup instead
publishes as an owned link to its release slot.

`dgxm install-service` and `dgxm uninstall` act only on an absent unit, or on
a regular, non-symlink unit whose first line is the dgx-monarch managed-unit
marker or whose bytes match the pre-marker canonical unit for this
configuration. A symlink or any other `dgxm-worker.service` is kept and
refused before any source sync or lifecycle change. Install also refuses
while a drop-in for the unit exists in one of the user's own systemd unit
directories, because a drop-in can replace `ExecStart`. Install rewrites a
pre-marker unit to the marked form, whose marker allows normal config
migration, and publishes by replacing the unit path, never by writing through
a symlink. A hand-edited or older managed unit
takes the manual path in [TROUBLESHOOTING #90](TROUBLESHOOTING.md#90-install-service-or-uninstall-refuses-an-existing-worker-unit), which also names
both pre-marker shapes, every drop-in directory the check reads, and why
install refuses a unit setup wrote.

The one exception is a unit `dgxm setup` wrote: `dgxm uninstall`, guided
setup's only teardown, removes it, while `dgxm install-service` refuses it.
Uninstall leaves setup's `~/.local/share/dgx-monarch/src` link into the
release slot in place. Remove that link on every host that carried the setup
unit before you rerun either installer: `dgxm setup` refuses an existing
managed source path, and plain sync refuses a release link without its
matching unit. Both source layouts
belong to the SSH target user and never assume the driver's checkout path
exists on a remote host.

Both launch modes, systemd and `nohup`, filter the Worker service's
environment before Monarch is imported. Only runtime identity, locale and path
settings and a short exact list of reviewed GPU and runtime knobs reach actor
launch options; unknown and secret-looking names are dropped without reading
their values. Keep the generated unit's explicit `PYTHONPATH` assignment and
add no broad `PassEnvironment=` rule. Put NCCL, GLOO and UCX tuning in
`cluster.toml`, which validates it and applies it inside each actor; never put
service or cloud credentials in a Worker unit.

Set the Init node's `mode` to `cluster`, or leave it on `auto`, which picks
cluster mode whenever it finds a cluster.toml.

### Swap on unified-memory hosts

Consider disabling swap on every Spark; `dgxm` does not require it.
Unified memory is the only memory the box has. A load or render near the
limit pages the model's own weights through the 16 GiB `/swap.img` that DGX
OS enables and can freeze the box; with swap off, the same overrun is a clean
OOM kill of one process.
`dgxm doctor` warns while swap is active on the driver or any worker host. To
turn swap off and keep it off across reboots:

```bash
sudo swapoff -a && sudo sed -i 's|^/swap.img|#/swap.img|' /etc/fstab \
  && printf 'vm.swappiness=10\n' | sudo tee /etc/sysctl.d/99-swap.conf \
  && sudo sysctl -q -p /etc/sysctl.d/99-swap.conf && swapon --show
```

`swapon --show` prints nothing once swap is off. To undo, uncomment the fstab
line, run `sudo swapon -a`, and delete `/etc/sysctl.d/99-swap.conf` to restore
the default swappiness at the next boot.

## Updating

Stop ComfyUI first. Both modes refuse while a local driver answers, and also
when the driver probe gets no answer, because a running driver would keep the
old Python modules while restarted workers load the new ones. Start ComfyUI
again afterwards; that restart loads the updated modules, so the node set on
the canvas matches the tree.

```bash
dgxm update    # git pull, the repo's exact torchmonarch pin on the driver and
               # workers, package install and sync, restart, doctor
dgxm update --verify --help  # transactional cluster path and exact target options
```

`--host HOST:PORT` adds that address to the ComfyUI activity guard in either
mode; discovery of other local driver ports still runs. In verified mode an
explicit `--host` with no confirmed driver behind it counts as unknown
activity, so `dgxm update --verify --host ...` always refuses, with
`activity_unknown` when no driver answers and nothing else blocks. The other
flags need `--verify`: `--target` selects the exact commit or ref
(default: origin's current default branch), `--yes` skips only the final
activation prompt, and
`--receipt` selects an absolute path for the sanitized receipt. None of them
bypasses an activity, source, config, dependency, or ownership guard.

Plain `dgxm update` is the compatibility path. It installs both torchmonarch
and dgx-monarch with `--no-deps`. Neither mode resolves or replaces torch,
NCCL, xfuser, yunchang, or optional extras; each moves only torchmonarch, to
the repository's exact pin. Upgrade other dependencies separately.
Plain and verified updates share one file lock per driver checkout, so a second
update from any process, in either mode, refuses before it changes anything.

For a cluster running from a Git source checkout, `dgxm update --verify`
supports remote Workers and one Worker on the driver's host. It first copies
the running update code into a private controller snapshot and launches an
isolated process from it. That process keeps using the captured code while
the live checkout and local Worker release change. The controller verifies
the saved config and source before starting. See the
[recorded update results and limits](VALIDATION.md#operator-checks).

It fetches `origin` and
resolves the requested ref to one full commit. By default it reads origin's
current default branch directly from the remote, rather than a cached local
`origin/HEAD`. It requires a clean driver checkout that the target fast-forwards and an
unchanged config, and refuses while ComfyUI, a Render session, a lease, or an
actor is active, or when the evidence to decide that is missing. It then puts
the target in a detached private worktree, confirms that every declared target
dependency other than torchmonarch is already satisfied on the driver and on
every host, builds a private release of the package and its exact torchmonarch
pin, stages it on every Worker host, and reads each copy back. It downloads the
prior and target exact torchmonarch wheels and installs each into a private
verification site with offline `--no-index --no-deps`. The live driver pin
must match the current checkout. Every host's live release must match the
prior driver's version, source and complete torchmonarch package. If that
check fails or cannot finish, the updater refuses: it cannot safely restore
an unverified release. Last,
the command asks for final confirmation, rechecks the checkout and activity,
and only then stops Worker services.

### Verified activation and recovery

Before stopping services, the updater captures each existing unit and source
layout and matches the full unit against supported generated forms. This
includes recognized legacy, hardened and guided-setup units. A marker alone
does not establish ownership; changed units, overrides or uncertain ownership
refuse migration.

Activation journals the exact prior unit and source layout, then switches
each stopped Worker to its staged release and writes a unit bound to that
release. A prior source directory moves to a backup; a prior link's target is
recorded. The live driver checkout stays at its prior commit. The command
starts Workers without syncing, runs target-code doctor and the client-owned
cluster smoke, and checks release, dependency and all-rank source agreement.
Only after those checks pass does it promote the verified driver pin,
fast-forward and read back the driver checkout, then finalize the transaction.
Prior release slots, source backups and installation journals are retained.

Migrated units include the canonical generation-invalidating `ExecStartPre`.
Compensation restores the captured unit bytes, mode and source layout before
a guarded prior start. It does not replace an unknown installation with an
assumed default. A failed start becomes definite only after two fence-held
readbacks both prove `inactive`/`dead`, `MainPID=0`, and no queued job.
Activating/deactivating state, malformed manager output, or a manager error
remains `unknown`.

A worker stop that definitely fails ends the update before activation and
starts nothing; any host that did stop stays down. A failed activation
compensates the remote slots and, if that succeeds, starts the prior release
without syncing. A failed start, doctor, smoke, readback, or driver commit first confirms the new
services stopped (for a driver commit, also that the driver still reads back at
its prior commit and pin), then compensates the slots and starts the prior
release; each recovery step runs only after the one before it succeeded, and a
prior start that definitely fails gets one more stop. If a stop, activation,
target start, compensation, or prior restart is unknown or throws, the command
performs no later mutation.

After target start is confirmed, an unknown, thrown, or cancelled doctor,
cluster smoke, exact readback, or driver-commit step authorizes exactly one
generation-bound Worker-service stop attempt to stop the newly started Workers.
Under the lifecycle lock, every host must still carry the exact update
generation recorded after its listener became ready, bound to that host boot
and exact listener-owner PID birth. A persistent generation fence stays held
across marker validation, stop/readback, and settlement. A missing, changed, or
ABA-mismatched marker makes the stop `unknown` and authorizes no mutation. The
attempt is recorded as `fail_closed_stop`. An interruption can land before that
step or the receipt is published; it is re-raised unchanged, and the latch set
before the stop is dispatched still forbids a second stop. After that stop the
command never compensates release slots, restarts the prior release, or cleans
the retained stage/journal, and it never retries an interrupted stop. A
finalization/cleanup failure after the driver commit reports a partial update
rather than undoing an already verified running release. These unknown paths use
schema-v2 step status `unknown` and top-level `partial`; v1 receipts remain
readable. The updater can restore only the release and service state it changed. It
cannot roll back unrelated third-party dependencies.

Verified update refuses unknown host locality, duplicate Worker aliases and
a local Worker configured for a different SSH user. A separate controller is
not required for the two-Spark layout.

`dgxm up` and package sync check a managed release link against its exact
setup or update unit, owner, parent directories and release target. Sync
leaves an authenticated release unchanged. It refuses foreign links or unit
overrides rather than writing into a release slot. Plain source directories
retain the normal sync path.

A clean successful update removes its temporary controller snapshot. Failed
or interrupted runs retain it and print its recovery location. Preserve it
with the installation journals until the transaction is understood; retained
state is not permission to restart services or delete prior releases.

Source activation invalidates Gate results tied to the previous source. The next use
runs the check again; sticky FAIL evidence remains authoritative.
Once it holds the per-checkout lock, every verified attempt that returns
normally writes a sanitized receipt, including a planned refusal or a partial
or fail-closed settlement. `KeyboardInterrupt` and `SystemExit` are
re-raised after best-effort settlement; the wrapper first tries to publish the
attached interruption receipt and suppresses a publication failure, so the
original interruption survives. Preserve the private terminal log and inspect
service/release state before retrying an interrupted update.

### A saved workflow refuses after an update

A saved workflow can fail at queue time with `missing_node_type` even after
installation and doctor succeed. The message gives the node title and id; the
title may differ from its registered class. Find that id in the saved JSON and
read `type` (`class_type` in an API export). Rewire it to the replacement in
the release's CHANGELOG.

Node mapping keys remain registered throughout a major version. Only a major
release removes one, and its CHANGELOG must name the removed class. This error
indicates an unavailable class, not a damaged workflow file.

## Safe doctor repair

`dgxm doctor` stays passive around live GPU state. It runs no load for a clock
reading, so on a box with a CUDA device the `GPU clock under load` row reports
its load probe as **NOT RUN**. Its one GPU workload is the `sageattention`
row's kernel probe, a bounded child process that runs only when `sageattention`
imports and no GPU compute process is running. A busy GPU skips the probe, and
a GPU-process check that does not answer reports it as **NOT RUN**. Run any
invasive diagnostic only in an operator-controlled maintenance window.

`dgxm doctor --repair` repairs only one allowlisted local condition: group or
other permission bits on the exact regular `cluster.toml` inode. It hashes and
inspects the file without following a symlink. After confirmation it takes the
shared config lock, rechecks the inode, bytes, and prior mode, and changes only
the mode, to `0600`; it reruns doctor only after the receipt is published.
Lock contention refuses and attempts a failed receipt instead of racing another
cooperating config writer. Network, SSH, packages, services, processes, meshes,
and power remain manual. `--yes` confirms without a prompt and `--receipt`
selects an absolute receipt path; `--repair` cannot be combined with `--json`.

## Operator receipts

Setup, safe repair, and verified update use the versioned
`dgx-monarch.operator-receipt` JSON schema. Schema v2 adds the step status
`unknown` so transport loss, malformed/incomplete evidence, cancellation, or
unconfirmed cleanup is not mislabeled `failed` or `partial`; the top-level
operation status remains `planned`, `succeeded`, `failed`, or `partial`.
A setup receipt carries the cluster-smoke step when setup ran with `--verify`;
a verified update records it whenever the update reaches the smoke. The default
destination is `$XDG_STATE_HOME/dgx-monarch/receipts/`, falling back to
`~/.local/state/dgx-monarch/receipts/`. The directory is private and owned by
the current user; publication is atomic, no-follow, non-overwriting, and the
file mode is `0600`.

Use this default for normal setup. During discovery, inspect any customized
`XDG_STATE_HOME` and the existing directory ancestry before applying changes.
Do not redirect receipts into a chat workspace just to keep reports together.
The default destination uses the same safe-parent checks as a custom path.

For a normally returned applied setup, safe repair, or verified update, receipt
publication is required before the command reports success. If the operation settles but its
required receipt cannot be published, the command exits nonzero and reports a
partial/receipt failure; it does not call an unrecorded mutation fully
successful. Doctor may already have tightened the config mode before receipt
publication fails; the permission change is not undone. A setup dry-run makes
no config, service, or cluster change and writes a local audit receipt only
when one was explicitly requested. These may return without a default receipt:
early setup validation, a declined setup confirmation, a repair refused by its
arguments or unable to inspect the config, and a verified update given a
relative `--receipt` or unable to acquire its per-checkout lock. After setup
confirmation, a config-lock refusal or interruption still attempts the required
setup receipt before returning or re-raising. An applied setup or verified
update interrupted by `KeyboardInterrupt` or `SystemExit` may settle without a
published receipt, as its section above describes.

Receipts keep stable step names, outcomes, counts, timestamps, selected
profile or exact target hash when applicable, and bounded source/config
hashes. They also distinguish an unattempted publication from a rollback: when
a pre-config service refusal or unknown outcome settles owned service cleanup,
`setup_compensation` counts record `config_unchanged` and
`service_compensated`, not `config_rolled_back`. They reject or redact
host/network identity, commands, environment assignments, credentials, paths,
URLs, and raw exceptions. Timing, operation names, counts, target commits, and
hashes can still reveal deployment metadata. Review receipts before sharing
them. `dgxm doctor --json` is different: it is a full diagnostic
snapshot rather than a receipt. It can include host names, addresses, paths,
versions and raw check details; keep it private or redact it by hand before sharing.

If the user needs a custom `--receipt`, check its absolute path and existing
ancestors before apply. No parent may be a symlink. Group- or world-writable
ancestors are rejected unless sticky; the final directory must belong to the
current user. The command creates missing parents and sets the final directory
to `0700`, so choose a dedicated private directory. Never change permissions
on a shared or unrelated directory to make receipt publication pass. An
optional dry-run receipt has the same path requirements.

## Uninstalling

```bash
dgxm uninstall                      # removes Worker services/units; offers reviewed config removal
rm -rf "$COMFY_DIR/custom_nodes/dgx-monarch"   # a symlink here: only the link goes
```

Uninstall holds the shared config lock across service/unit removal, rereads the
config, and asks separately before its checked unlink. Lock contention or
observed config drift refuses; the trusted-account limit on the final
check-to-unlink window, described under guided setup, still applies. If any
host's unit is not one it owns, uninstall refuses before any worker stops. It
removes no unit or config unless `dgxm down` definitely succeeds on every host;
a failed or unknown stop leaves the installation in place for inspection.
Models and venvs are left alone.

---
See [VALIDATION.md](VALIDATION.md) for setup, repair and cluster test results
and their limits. FSDP with sequence parallelism requires NCCL >= 2.26.
