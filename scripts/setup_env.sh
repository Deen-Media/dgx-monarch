#!/usr/bin/env bash
# Create or refresh ~/monarch-env on a Spark. Idempotent.
#   bash scripts/setup_env.sh                     # this box (actors + torch layer)
#   bash scripts/setup_env.sh --comfy             # + the ComfyUI and xFuser layer
#   bash scripts/setup_env.sh --comfy --worker    # same on both boxes
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
XFUSER_BUILDER="$SCRIPT_DIR/../tools/build_xfuser_compat_wheel.py"
MONARCH_PIN="torchmonarch==0.6.0"   # the 0.x API changes between releases; bump on purpose
# --worker needs DGXM_SIBLING, the other box's fabric IP. It has no default
# because the address is your rig's. DGXM_SSH_KEY names an ssh key the sibling
# accepts; empty uses your ssh config or agent.
KEY="${DGXM_SSH_KEY:-}"
SIB="${DGXM_SIBLING:-}"
# Optional donor env for the aarch64 torchaudio workaround (see below); any
# env with a torch-2.12-compatible torchaudio works.
TORCHAUDIO_DONOR="${DGXM_TORCHAUDIO_DONOR:-}"

setup_base() {
  local PY="$HOME/monarch-env/bin/python"
  local P="$HOME/monarch-env/bin/pip"
  if [ ! -x "$PY" ]; then
    python3 -m venv "$HOME/monarch-env"
  fi
  "$P" install -q --upgrade pip
  "$P" install -q "$MONARCH_PIN" numpy
  "$P" install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
  "$PY" -c "import monarch.actor, monarch.rdma, torch; import monarch.rdma as r; print('monarch+torch OK on', __import__('socket').gethostname(), '| ibverbs:', r.is_ibverbs_available())"
}

setup_comfy_layer() {
  # Install ComfyUI requirements, diffusers and xFuser without replacing torch.
  # torchvision is not pinned here; its explicit install uses --no-deps and
  # is skipped when requirements already installed it. On aarch64, the cu132
  # index lacks a compatible torchaudio build, so copy one from
  # DGXM_TORCHAUDIO_DONOR. The tested donor build is 2.11.0+cu132 with torch 2.12.
  # ComfyUI before v0.38.0 imports torchaudio at startup; refuse a missing donor
  # rather than leave that installation unable to import ComfyUI.
  local PY="$HOME/monarch-env/bin/python"
  local P="$HOME/monarch-env/bin/pip"
  local SP XFUSER_WHEELS TORCH_CONSTRAINT TORCH_BUILD
  if [ ! -r "$XFUSER_BUILDER" ]; then
    echo "FAIL: cannot read tools/build_xfuser_compat_wheel.py; run setup_env.sh from a full dgx-monarch checkout" >&2
    return 1
  fi
  SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  "$P" install -q -r "$HOME/ComfyUI/requirements.txt"
  "$P" install -q diffusers==0.38.0 accelerate
  # Install xFuser's declared dependencies (distvae, peft, av and others) while
  # a constraint holds pip to the CUDA torch build already in this venv. Do not
  # use --no-deps here: xFuser 0.7 adds required runtime distributions.
  TORCH_BUILD="$("$PY" -c 'import torch; print(torch.__version__)')"
  XFUSER_WHEELS="$(mktemp -d)"
  TORCH_CONSTRAINT="$XFUSER_WHEELS/torch-constraint.txt"
  printf 'torch==%s\n' "$TORCH_BUILD" > "$TORCH_CONSTRAINT"
  if ! "$PY" "$XFUSER_BUILDER" --output-dir "$XFUSER_WHEELS"; then
    rm -rf "$XFUSER_WHEELS"
    return 1
  fi
  "$P" install -q --upgrade-strategy only-if-needed \
    --constraint "$TORCH_CONSTRAINT" --find-links "$XFUSER_WHEELS" \
    xfuser==0.7.0+dgxm.npuimport1 yunchang==0.6.4
  rm -rf "$XFUSER_WHEELS"
  "$P" install -q --no-deps torchvision --index-url https://download.pytorch.org/whl/cu132
  if ! "$PY" -c "import torchaudio" 2>/dev/null; then
    local DONOR_PY="$TORCHAUDIO_DONOR/bin/python"
    local DONOR_SP=""
    if [ -n "$TORCHAUDIO_DONOR" ] && [ -x "$DONOR_PY" ]; then
      DONOR_SP="$("$DONOR_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    fi
    if [ -n "$DONOR_SP" ] && [ -d "$DONOR_SP/torchaudio" ]; then
      "$P" uninstall -q -y torchaudio 2>/dev/null || true
      cp -r "$DONOR_SP/torchaudio" "$DONOR_SP"/torchaudio-*.dist-info "$SP/"
      echo "torchaudio copied from $TORCHAUDIO_DONOR"
    else
      echo "FAIL: no working torchaudio (as of 2026-07-02, the cu132 index served only torchaudio 2.2.0 for aarch64)" >&2
      echo "      set DGXM_TORCHAUDIO_DONOR to a venv with a torch-2.12-compatible" >&2
      echo "      torchaudio (2.11.0+cu132 is proven ABI-compatible)" >&2
      return 1
    fi
  fi
  echo "comfy layer done on $(hostname)"
}

WANT_COMFY=0
WANT_WORKER=0
for a in "$@"; do
  case "$a" in
    --comfy) WANT_COMFY=1 ;;
    --worker) WANT_WORKER=1 ;;
    *) echo "FAIL: unknown argument: $a (the options are --comfy and --worker)" >&2; exit 2 ;;
  esac
done

setup_base
if [ "$WANT_COMFY" = 1 ]; then
  setup_comfy_layer
fi

if [ "$WANT_WORKER" = 1 ]; then
  if [ -z "$SIB" ]; then
    echo "FAIL: --worker needs DGXM_SIBLING=<other box's fabric IP>" >&2
    exit 1
  fi
  if ! "$HOME/monarch-env/bin/python" -c \
    'import ipaddress,sys; ip=ipaddress.ip_address(sys.argv[1]); sys.exit(ip.is_unspecified or ip.is_multicast)' \
    "$SIB"; then
    echo "FAIL: DGXM_SIBLING must be the other box's fabric IP address, not a hostname; unspecified and multicast addresses are refused" >&2
    exit 1
  fi
  SSH_ARGS=(-o BatchMode=yes)
  [ -n "$KEY" ] && SSH_ARGS+=(-i "$KEY")
  # Send the program over stdin and quote each value with bash's %q, so no
  # env-provided path lands in the remote command line, where a quote or
  # newline would become code.
  {
    printf '%s\n' 'set -euo pipefail'
    printf 'MONARCH_PIN=%q\n' "$MONARCH_PIN"
    printf 'TORCHAUDIO_DONOR=%q\n' "$TORCHAUDIO_DONOR"
    printf 'WANT_COMFY=%q\n' "$WANT_COMFY"
    # The peer may not have this checkout: stream the stdlib-only builder with
    # the two functions, not a second copy or a prebuilt wheel.
    # Expand these variables on the peer after SSH reads this script.
    # shellcheck disable=SC2016
    printf '%s\n' 'XFUSER_BUILDER="$(mktemp)"'
    # shellcheck disable=SC2016
    printf '%s\n' 'trap '\''rm -f "$XFUSER_BUILDER"'\'' EXIT'
    # shellcheck disable=SC2016
    printf '%s\n' 'cat > "$XFUSER_BUILDER" <<'\''PY'\'''
    cat -- "$XFUSER_BUILDER"
    printf '%s\n' 'PY'
    declare -f setup_base setup_comfy_layer
    printf '%s\n' 'setup_base' \
      "if [ \"\$WANT_COMFY\" = 1 ]; then setup_comfy_layer; fi"
  } | ssh "${SSH_ARGS[@]}" -- "$SIB" bash -s
fi
