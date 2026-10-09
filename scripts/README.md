# scripts/

`comfy-driver.sh` launches one ComfyUI driver with the validated DGX Spark flag
profile. It does not run `dgxm up`, `down`, or `restart`, and it never stops a
persistent Worker service. ComfyUI replaces the launcher process, stays attached
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

The entry lives under `${XDG_DATA_HOME:-$HOME/.local/share}/applications`.
Install and uninstall refuse to replace or remove an entry they do not own.

Do not add wrapper scripts for cluster operations here: `dgxm` is the
interface for Worker-service lifecycle and setup.

## Developer environment helper

`scripts/setup_env.sh` is a developer-lab helper, not the portable installer.
It creates or changes `~/monarch-env`, assumes `~/ComfyUI`, selects a fixed
Torch/CUDA build, and can copy torchaudio from a donor environment. Its rerun
can change packages already installed there. Do not run it against a user's
working environment as part of agent onboarding.

Use [INSTALL.md](../docs/INSTALL.md) for dependency installation and
[the setup skill](../skills/dgx-monarch/SKILL.md) for discovery, approved
setup, verification and recovery. Missing ComfyUI or CUDA dependencies need a
separate reviewed prerequisite plan.
