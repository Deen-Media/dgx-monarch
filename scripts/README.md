# scripts/

`comfy-driver.sh` launches one ComfyUI driver with the validated DGX Spark flag
profile. It does not run `dgxm up`, `down`, or `restart`, and it never stops a
persistent Worker service. Without `--with-top`, ComfyUI replaces the launcher process, stays attached
to the terminal, and receives signals normally.

Pass an existing ComfyUI checkout explicitly. Select its virtual environment or
interpreter when the current shell is not already using it:

```bash
export COMFY_DIR="/absolute/path/to/ComfyUI"   # the checkout dgx-monarch installs under
bash "$COMFY_DIR/custom_nodes/dgx-monarch/scripts/comfy-driver.sh" \
  --comfy-dir "$COMFY_DIR" \
  --venv "$COMFY_DIR/.venv"
```

All checkout, venv, interpreter, and log paths must be absolute. Place or symlink
this repository directly under the selected ComfyUI checkout's `custom_nodes`
directory. The example uses that location and passes its parent checkout as
`$COMFY_DIR`.

The launcher verifies the installation path before checking the port. It reuses
a listener only if its live node schema exposes `DGXMonarchInit`. A new driver
uses ComfyUI's `--auto-launch` to open a browser after server setup. Use
`--no-browser` for terminal-only or remote launches.

Pass additional ComfyUI arguments after `--` to preserve spaces and shell
quoting:

```bash
bash scripts/comfy-driver.sh \
  --comfy-dir /opt/ComfyUI \
  --python /opt/ComfyUI/.venv/bin/python \
  --no-browser \
  -- --preview-method auto --output-directory "/srv/Comfy Output"
```

The launcher owns `--listen`, `--port`, `--auto-launch`, and
`--disable-auto-launch`. Supply its `--listen`, `--port`, `--no-browser`, and
`--log` options instead; `--log` appends the visible output to an absolute path
while leaving the terminal attached. The launcher refuses ComfyUI's TLS flags,
because its confirmation probe speaks plain HTTP. `EXTRA_ARGS` accepts a list
split on whitespace; pass real arguments when a value contains spaces.

## FP8 compute flag

`--fast fp8_matrix_mult` after `--` gives the models this ComfyUI process
builds, such as a stock graph's, ComfyUI's FP8 matmul path for plain FP8
checkpoints. It does not reach the DGX Monarch actors, which start ComfyUI
without its arguments: `[worker_args] fp8_matrix_mult` in cluster.toml is
their switch ([MODELS.md](../docs/MODELS.md#fp8-compute-paths)). Name the
feature: `--fast` alone turns on every ComfyUI performance feature, and
autotune among them keeps ComfyUI's asynchronous allocator off.

```bash
bash scripts/comfy-driver.sh \
  --comfy-dir /opt/ComfyUI \
  --python /opt/ComfyUI/.venv/bin/python \
  -- --fast fp8_matrix_mult
```

The launcher never passes `--bf16-unet`, which upcasts FP8 weights to BF16 at
load so that no FP8 path runs. Keep it out of any wrapper used to measure one.

## Optional dashboard alongside ComfyUI

Add `--with-top --cluster-config /absolute/path/to/cluster.toml` to the
foreground launcher. It uses installed GNOME Terminal (`--wait`) or falls
back to xterm. It requires a graphical display and an interactive terminal;
no terminal or Python packages are installed automatically. Install the TUI
extra through the constrained procedure in [INSTALL.md](../docs/INSTALL.md).

```bash
bash scripts/comfy-driver.sh \
  --comfy-dir /opt/ComfyUI --python /opt/comfyui-env/bin/python \
  --cluster-config /absolute/path/to/cluster.toml --with-top
```

The dashboard opens in a separate window using the exact selected Python,
config, and driver address/port. Wildcard bind addresses map to loopback for
local monitoring. Both processes use the same config. `--check` validates
paths, config, imports, and the supported terminal prerequisite without
opening a window or starting ComfyUI; graphical/interactive access is checked
when actually launching.

A new driver starts only after the dashboard process confirms its terminal
startup. ComfyUI retains the invoking terminal, visible logs, signals, shell
job control, and exit status. Closing the dashboard leaves ComfyUI running;
when ComfyUI exits, its dashboard closes. Cancellation gives the owned driver
group up to 180 seconds for graceful teardown (matching the mesh recycle
window, with a final TERM attempt), then forces only that group to stop and
bounds the final reap. A forced stop explicitly leaves Attached-mesh and actor
cleanup unconfirmed; inspect cluster state before reuse. Dashboard shutdown
and the local terminal-client reap are bounded separately. Shared terminal
servers and reused drivers are never termination targets. If a confirmed driver already
exists, only a dashboard opens and the launcher waits for that window; it
never acquires ownership of or stops the existing driver. Failure to open the
dashboard refuses a new driver start. Failures after startup are reported
without stopping ongoing ComfyUI work.

Cleanup uses a private connection to the dashboard created by this launch.
It never closes another terminal window or a shared terminal server. Worker
services remain unchanged. For headless use, keep the ordinary foreground
launcher and run `dgxm top` in a second terminal yourself.

## Optional desktop entry

The installer validates paths and creates a per-user desktop entry with the
repository icon. The shortcut opens a terminal and starts the foreground
ComfyUI driver without sudo or Worker-service actions.

```bash
bash scripts/dgxm-desktop.sh install \
  --comfy-dir /opt/ComfyUI \
  --venv /opt/ComfyUI/.venv

bash scripts/dgxm-desktop.sh uninstall
```

The desktop installer also accepts `--with-top --cluster-config PATH` and
persists the config in the shortcut, so later launches use the same selection.

The entry lives under `${XDG_DATA_HOME:-$HOME/.local/share}/applications`.
Install and uninstall refuse to replace or remove an entry they do not own.

Do not add wrapper scripts for cluster operations here: `dgxm` is the
interface for Worker-service lifecycle and setup.

## Developer environment helper

`scripts/setup_env.sh` is a developer-lab helper, not the portable installer.
It creates or changes `~/monarch-env`, assumes `~/ComfyUI`, selects a fixed
Torch/CUDA build, and for a ComfyUI before v0.38.0, which still needs
torchaudio, can copy it from a donor environment. Its rerun can change
packages already installed there. Do not run it against a user's
working environment as part of agent onboarding.

Use [INSTALL.md](../docs/INSTALL.md) for dependency installation and
[the setup skill](../skills/dgx-monarch/SKILL.md) for discovery, approved
setup, verification and recovery. Missing ComfyUI or CUDA dependencies need a
separate reviewed prerequisite plan.

## CPU test wrapper

`bash scripts/test_cpu.sh` runs `pytest tests/ -q --durations=25` from the
checkout root with this checkout's `src` first on `PYTHONPATH`,
`HYPOTHESIS_PROFILE=ci`, and OMP and MKL at one thread unless already set. It
prints the total time and the 25 slowest tests, and exits with pytest's
status. `DGXM_TEST_PYTHON` selects the interpreter. The default is `python`,
which some systems lack, so set it or activate the environment first.

Extra arguments are added to `pytest tests/`. Options such as `-k` or
`--ignore` narrow the run, but a test file path does not, because `tests/` is
always collected; run pytest directly for a subset. A run longer than 120
seconds prints a warning, a workflow annotation in GitHub Actions. The
warning is advisory and leaves the exit status unchanged.
[CONTRIBUTING.md](../CONTRIBUTING.md#check-your-change) lists the other
checks.

## Optional normal-terminal CLI launcher

The package installs `dgxm` inside the selected Python environment. To make
that exact executable available from an ordinary terminal, use its local
launcher command. Planning performs no installation:

```bash
/opt/comfyui-env/bin/dgxm launcher \
  --executable /opt/comfyui-env/bin/dgxm \
  --cluster-config /absolute/path/to/cluster.toml \
  --destination "$HOME/.local/bin/dgxm"
```

Repeat with `--apply` to install. The default destination is
`~/.local/bin/dgxm`. Identical reruns leave the file unchanged. The command
refuses an unmanaged command, a modified launcher, or a symlink destination
or parent. It only replaces a recognized managed launcher after checking its
identity and contents under a cooperative lock. That lock does not prevent
manual writers; POSIX replacement still has a final check-to-rename window.
A newly created launcher is published atomically without overwriting any
existing command.

The launcher executes the selected environment's `dgxm`, forwards arguments,
and preserves exit status and signals. It passes the saved cluster config
explicitly, refusing a missing saved config instead of falling through to another
cluster. A leading global `--config PATH` or `--config=PATH` overrides the
saved selection. Explicit overrides pass unchanged to the CLI, preserving its
home-directory expansion and command-specific validation.

Inspect the PATH report, then verify in your ordinary terminal with
`command -v dgxm` and `dgxm version` (run `hash -r` first in Bash). Aliases,
functions, and an earlier PATH entry can shadow the file. If its directory is
absent from PATH, use the full path or add that directory in your own shell
configuration. The installer does not edit shell startup files, install
packages, start the dashboard, or change Worker services. This is entry-point
selection; cluster operations remain implemented only by the CLI itself.
