#!/usr/bin/env bash
# Launch one ComfyUI driver with the validated DGX Spark flag profile.
#
# Owns only the foreground ComfyUI driver; never changes Worker services.
# exec replaces this shell so terminal logs and signals reach ComfyUI.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  comfy-driver.sh --comfy-dir ABSOLUTE_PATH [launcher options] [-- COMFY_ARGS...]

Launcher options:
  --comfy-dir PATH   Required absolute ComfyUI checkout
  --venv PATH        Absolute virtual environment containing bin/python
  --python PATH      Absolute Python interpreter; conflicts with --venv
  --listen ADDRESS   ComfyUI bind address (default: 127.0.0.1)
  --port PORT        ComfyUI port (default: 8188)
  --reserve-vram GB  GiB kept free for the OS (default: 2)
  --sage             Enable ComfyUI's native Sage attention path
  --no-browser       Do not ask ComfyUI to open the browser after readiness
  --log PATH         Also append visible stdout and stderr to an absolute file
  --check             Validate paths and arguments, then exit without probing or starting
  -h, --help         Show this help

COMFY_DIR, VENV, PYTHON_BIN, LISTEN, PORT, RESERVE_VRAM, SAGE, LOG, and
EXTRA_ARGS remain available as explicit environment inputs. COMFY_DIR and any
provided VENV, PYTHON_BIN, or LOG path must be absolute. Prefer arguments for
desktop shortcuts and for values containing spaces.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

need_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        fail "$1 needs a value"
    fi
}

reject_control_chars() {
    case "$2" in
        *$'\n'*|*$'\r'*) fail "$1 contains a newline or carriage return" ;;
    esac
}

resolve_directory() {
    local label=$1 value=$2 resolved
    reject_control_chars "$label" "$value"
    [[ "$value" = /* ]] || fail "$label must be an absolute path"
    [ -d "$value" ] || fail "$label is not an existing directory: $value"
    resolved=$(realpath -e -- "$value") || fail "$label could not be resolved: $value"
    printf '%s\n' "$resolved"
}

resolve_executable() {
    local label=$1 value=$2 parent base
    reject_control_chars "$label" "$value"
    [[ "$value" = /* ]] || fail "$label must be an absolute path"
    if [ ! -x "$value" ] || [ -d "$value" ]; then
        fail "$label is not an executable file: $value"
    fi
    parent=$(realpath -e -- "$(dirname -- "$value")") || fail "$label parent could not be resolved"
    base=$(basename -- "$value")
    printf '%s/%s\n' "$parent" "$base"
}

repository_is_discoverable() (
    local entry resolved
    shopt -s nullglob dotglob
    for entry in "$COMFY_DIR/custom_nodes"/*; do
        [ -d "$entry" ] || continue
        resolved=$(realpath -e -- "$entry" 2>/dev/null) || continue
        [ "$resolved" = "$REPOSITORY_ROOT" ] && return 0
    done
    return 1
)

validate_extra_args() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --listen|--listen=*|--port|--port=*|--auto-launch|--disable-auto-launch)
                fail "$arg is managed by this launcher; use the matching launcher option"
                ;;
            --tls-keyfile|--tls-keyfile=*|--tls-certfile|--tls-certfile=*)
                fail "ComfyUI TLS flags are not supported: the launcher's driver probe uses plain HTTP"
                ;;
        esac
    done
}

COMFY_DIR=${COMFY_DIR:-}
VENV=${VENV:-}
PYTHON_BIN=${PYTHON_BIN:-}
LISTEN=${LISTEN:-127.0.0.1}
PORT=${PORT:-8188}
RESERVE_VRAM=${RESERVE_VRAM:-2}
SAGE=${SAGE:-0}
LOG=${LOG:-}
OPEN_BROWSER=1
CHECK_ONLY=0
COMFY_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --comfy-dir)
            need_value "$@"
            COMFY_DIR=$2
            shift 2
            ;;
        --venv)
            need_value "$@"
            VENV=$2
            shift 2
            ;;
        --python)
            need_value "$@"
            PYTHON_BIN=$2
            shift 2
            ;;
        --listen)
            need_value "$@"
            LISTEN=$2
            shift 2
            ;;
        --port)
            need_value "$@"
            PORT=$2
            shift 2
            ;;
        --reserve-vram)
            need_value "$@"
            RESERVE_VRAM=$2
            shift 2
            ;;
        --sage)
            SAGE=1
            shift
            ;;
        --no-browser)
            OPEN_BROWSER=0
            shift
            ;;
        --log)
            need_value "$@"
            LOG=$2
            shift 2
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            COMFY_ARGS+=("$@")
            break
            ;;
        *)
            fail "unknown launcher option: $1; place ComfyUI arguments after --"
            ;;
    esac
done

if [ -n "${EXTRA_ARGS:-}" ]; then
    # EXTRA_ARGS is a legacy input: it splits on whitespace and treats quotes as
    # plain characters. Prefer arguments after --, which keep spaces and quoting.
    read -r -a LEGACY_EXTRA <<< "$EXTRA_ARGS"
    COMFY_ARGS=("${LEGACY_EXTRA[@]}" "${COMFY_ARGS[@]}")
fi

[ -n "$COMFY_DIR" ] || fail "--comfy-dir or COMFY_DIR is required: pass the absolute path of an existing ComfyUI checkout"
COMFY_DIR=$(resolve_directory "ComfyUI checkout" "$COMFY_DIR")
[ -f "$COMFY_DIR/main.py" ] || fail "no main.py in ComfyUI checkout: $COMFY_DIR"
[ -f "$COMFY_DIR/comfy/cli_args.py" ] || fail "not a ComfyUI checkout (no comfy/cli_args.py): $COMFY_DIR"
SCRIPT_DIR=$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")
REPOSITORY_ROOT=$(realpath -e -- "$SCRIPT_DIR/..")
[ -d "$COMFY_DIR/custom_nodes" ] || fail "ComfyUI has no custom_nodes directory: $COMFY_DIR"
if ! repository_is_discoverable; then
    fail "this dgx-monarch checkout is not discoverable under $COMFY_DIR/custom_nodes, the only directory this launcher checks; clone it there or add a direct symlink to $REPOSITORY_ROOT"
fi

if [ -n "$VENV" ] && [ -n "$PYTHON_BIN" ]; then
    fail "choose either --venv or --python, not both"
fi
if [ -n "$VENV" ]; then
    VENV=$(resolve_directory "virtual environment" "$VENV")
    PYTHON_BIN=$(resolve_executable "virtual environment Python" "$VENV/bin/python")
    export VIRTUAL_ENV=$VENV
    export PATH="$VENV/bin:$PATH"
    unset PYTHONHOME 2>/dev/null || true
elif [ -n "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(resolve_executable "Python interpreter" "$PYTHON_BIN")
else
    DEFAULT_PYTHON=$(command -v python 2>/dev/null || true)
    [ -n "$DEFAULT_PYTHON" ] || fail "no Python on PATH; pass --venv or --python"
    [[ "$DEFAULT_PYTHON" = /* ]] || DEFAULT_PYTHON=$(command -v "$DEFAULT_PYTHON")
    PYTHON_BIN=$(resolve_executable "Python interpreter" "$DEFAULT_PYTHON")
fi

reject_control_chars "listen address" "$LISTEN"
[[ "$LISTEN" =~ ^[A-Za-z0-9._:%-]+$ ]] || fail "listen address has unsupported characters: $LISTEN"
[[ "$LISTEN" != *,* ]] || fail "listen address must be one address, not a comma-separated list"
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "port must be an integer from 1 to 65535"
(( PORT >= 1 && PORT <= 65535 )) || fail "port must be an integer from 1 to 65535"
[[ "$RESERVE_VRAM" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "reserve-vram must be a non-negative number of GiB"
case "$SAGE" in
    0|1) ;;
    *) fail "SAGE must be 0 or 1" ;;
esac
validate_extra_args "${COMFY_ARGS[@]}"

if [ -n "$LOG" ]; then
    reject_control_chars "log path" "$LOG"
    [[ "$LOG" = /* ]] || fail "log path must be absolute"
    LOG_PARENT=$(realpath -e -- "$(dirname -- "$LOG")") || fail "log parent does not exist"
    [ -d "$LOG_PARENT" ] || fail "log parent is not a directory"
    if [ -e "$LOG" ] && { [ ! -f "$LOG" ] || [ -L "$LOG" ]; }; then
        fail "log path must be a regular, non-symlink file"
    fi
    LOG="$LOG_PARENT/$(basename -- "$LOG")"
fi

if [ "$CHECK_ONLY" = 1 ]; then
    printf 'launcher check: ready\n'
    printf '  comfy: %s\n' "$COMFY_DIR"
    printf '  python: %s\n' "$PYTHON_BIN"
    printf '  driver: http://%s:%s\n' "$LISTEN" "$PORT"
    exit 0
fi

# Reuse only a listener whose /object_info/DGXMonarchInit exposes this pack.
# Reject other listeners, including ComfyUI without DGX Monarch.
set +e
"$PYTHON_BIN" - "$LISTEN" "$PORT" <<'PY'
import json
import socket
import sys
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None

host = sys.argv[1]
port = int(sys.argv[2])
connect_host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
try:
    with socket.create_connection((connect_host, port), timeout=0.5):
        pass
except OSError:
    raise SystemExit(10)

url_host = f"[{connect_host}]" if ":" in connect_host else connect_host
request_url = f"http://{url_host}:{port}/object_info/DGXMonarchInit"
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirect(),
)
try:
    with opener.open(request_url, timeout=2.0) as response:
        if response.geturl() != request_url:
            raise ValueError("driver probe changed endpoint")
        payload = json.load(response)
except Exception:
    raise SystemExit(11)

node = payload.get("DGXMonarchInit") if isinstance(payload, dict) else None
valid = (
    isinstance(node, dict)
    and node.get("name") == "DGXMonarchInit"
    and node.get("category") == "DGX Monarch"
    and "DGXM_MESH" in node.get("output", ())
)
raise SystemExit(0 if valid else 11)
PY
PROBE_STATUS=$?
set -e

case "$PROBE_STATUS" in
    0)
        BROWSER_HOST=$LISTEN
        [ "$BROWSER_HOST" = "0.0.0.0" ] && BROWSER_HOST=127.0.0.1
        [ "$BROWSER_HOST" = "::" ] && BROWSER_HOST=::1
        case "$BROWSER_HOST" in
            *:*) BROWSER_URL="http://[$BROWSER_HOST]:$PORT" ;;
            *) BROWSER_URL="http://$BROWSER_HOST:$PORT" ;;
        esac
        printf 'DGX Monarch driver already running at %s; not starting a duplicate.\n' "$BROWSER_URL"
        if [ "$OPEN_BROWSER" = 1 ]; then
            "$PYTHON_BIN" - "$BROWSER_URL" <<'PY'
import sys
import webbrowser

webbrowser.open(sys.argv[1])
PY
        fi
        exit 0
        ;;
    10) ;;
    *) fail "port $PORT has a listener that is not a confirmed DGX Monarch driver; stop it or choose another --port" ;;
esac

# CUDA, Triton and OpenMP settings validated on DGX Spark.
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_MODULE_LOADING=LAZY
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# NCCL_P2P_DISABLE is fine on one box but breaks cross-Spark NCCL, and doctor
# fails NCCL_PROTO=LL on every layer. Worker fabric settings belong in cluster.toml.
unset NCCL_P2P_DISABLE NCCL_PROTO 2>/dev/null || true

ARGS=(
    --listen "$LISTEN"
    --port "$PORT"
    --disable-pinned-memory
    --disable-async-offload
    --reserve-vram "$RESERVE_VRAM"
    --bf16-vae
    --bf16-text-enc
    --dont-upcast-attention
)
[ "$SAGE" = 1 ] && ARGS+=(--use-sage-attention)
if [ "$OPEN_BROWSER" = 1 ]; then
    # ComfyUI invokes its browser callback only after server setup.
    ARGS+=(--auto-launch)
else
    ARGS+=(--disable-auto-launch)
fi
ARGS+=("${COMFY_ARGS[@]}")

cd "$COMFY_DIR"
if [ -n "$LOG" ]; then
    # The process substitution copies output to the log and the terminal, and the
    # exec below still makes Python this process, so terminal signals reach it.
    exec > >(tee -a -- "$LOG") 2>&1
fi
printf 'Starting ComfyUI with DGX Monarch at http://%s:%s\n' "$LISTEN" "$PORT"
printf 'Worker services are not changed by this launcher.\n'
exec "$PYTHON_BIN" main.py "${ARGS[@]}"
