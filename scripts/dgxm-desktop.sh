#!/usr/bin/env bash
# Install or uninstall the optional DGX Monarch desktop entry for this user.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  dgxm-desktop.sh install --comfy-dir ABSOLUTE_PATH [--venv PATH | --python PATH]
                       [--listen ADDRESS] [--port PORT] [--no-browser]
  dgxm-desktop.sh uninstall

The installed entry opens a terminal and runs the foreground ComfyUI driver.
It never starts or stops dgx-monarch Worker services and never uses sudo.
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

desktop_quote() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//\`/\\\`}
    value=${value//\$/\\\$}
    value=${value//%/%%}
    printf '"%s"' "$value"
}

ACTION=${1:-}
case "$ACTION" in
    install|uninstall) shift ;;
    -h|--help|"") usage; exit 0 ;;
    *) fail "first argument must be install or uninstall" ;;
esac

XDG_ROOT=${XDG_DATA_HOME:-${HOME:?HOME is not set}/.local/share}
reject_control_chars "XDG data directory" "$XDG_ROOT"
[[ "$XDG_ROOT" = /* ]] || fail "XDG_DATA_HOME must be absolute, and so must HOME when XDG_DATA_HOME is unset or empty"

if [ "$ACTION" = uninstall ]; then
    [ "$#" -eq 0 ] || fail "uninstall takes no other arguments"
    if [ ! -d "$XDG_ROOT/applications" ]; then
        printf 'DGX Monarch desktop entry is not installed.\n'
        exit 0
    fi
    APPLICATIONS_DIR=$(realpath -e -- "$XDG_ROOT/applications")
    DESKTOP_FILE="$APPLICATIONS_DIR/dgx-monarch.desktop"
    if [ ! -e "$DESKTOP_FILE" ] && [ ! -L "$DESKTOP_FILE" ]; then
        printf 'DGX Monarch desktop entry is not installed.\n'
        exit 0
    fi
    if [ ! -f "$DESKTOP_FILE" ] || [ -L "$DESKTOP_FILE" ]; then
        fail "refusing to remove a desktop entry that is a symlink or not a regular file"
    fi
    grep -qx 'X-DGX-Monarch-Managed=true' "$DESKTOP_FILE" || fail "refusing to remove an unmanaged desktop entry (no X-DGX-Monarch-Managed=true line)"
    rm -- "$DESKTOP_FILE"
    printf 'Removed %s\n' "$DESKTOP_FILE"
    exit 0
fi

COMFY_DIR=
VENV=
PYTHON_BIN=
LISTEN=127.0.0.1
PORT=8188
NO_BROWSER=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --comfy-dir) need_value "$@"; COMFY_DIR=$2; shift 2 ;;
        --venv) need_value "$@"; VENV=$2; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
        --listen) need_value "$@"; LISTEN=$2; shift 2 ;;
        --port) need_value "$@"; PORT=$2; shift 2 ;;
        --no-browser) NO_BROWSER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown install argument: $1" ;;
    esac
done

[ -n "$COMFY_DIR" ] || fail "install needs --comfy-dir"
COMFY_DIR=$(resolve_directory "ComfyUI checkout" "$COMFY_DIR")
if [ ! -f "$COMFY_DIR/main.py" ] || [ ! -f "$COMFY_DIR/comfy/cli_args.py" ]; then
    fail "not a ComfyUI checkout (no main.py or comfy/cli_args.py)"
fi
if [ -n "$VENV" ] && [ -n "$PYTHON_BIN" ]; then
    fail "choose either --venv or --python, not both"
fi

SCRIPT_DIR=$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")
LAUNCHER=$(resolve_executable "launcher" "$SCRIPT_DIR/comfy-driver.sh")
ICON=$(realpath -e -- "$SCRIPT_DIR/../docs/media/dgx-monarch.svg") || fail "desktop icon docs/media/dgx-monarch.svg is missing"
if [ ! -f "$ICON" ] || [ -L "$ICON" ]; then
    fail "desktop icon is not a regular file"
fi

PYTHON_ARGS=()
if [ -n "$VENV" ]; then
    VENV=$(resolve_directory "virtual environment" "$VENV")
    resolve_executable "virtual environment Python" "$VENV/bin/python" >/dev/null
    PYTHON_ARGS=(--venv "$VENV")
elif [ -n "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(resolve_executable "Python interpreter" "$PYTHON_BIN")
    PYTHON_ARGS=(--python "$PYTHON_BIN")
else
    DEFAULT_PYTHON=$(command -v python 2>/dev/null || true)
    [ -n "$DEFAULT_PYTHON" ] || fail "no Python on PATH; pass --venv or --python"
    [[ "$DEFAULT_PYTHON" = /* ]] || DEFAULT_PYTHON=$(command -v "$DEFAULT_PYTHON")
    PYTHON_BIN=$(resolve_executable "Python interpreter" "$DEFAULT_PYTHON")
    PYTHON_ARGS=(--python "$PYTHON_BIN")
fi

CHECK_ARGS=(--comfy-dir "$COMFY_DIR" "${PYTHON_ARGS[@]}" --listen "$LISTEN" --port "$PORT" --check)
[ "$NO_BROWSER" = 1 ] && CHECK_ARGS+=(--no-browser)
"$LAUNCHER" "${CHECK_ARGS[@]}" >/dev/null

EXEC_PARTS=(
    "$(desktop_quote "$LAUNCHER")"
    --comfy-dir "$(desktop_quote "$COMFY_DIR")"
)
if [ "${PYTHON_ARGS[0]}" = --venv ]; then
    EXEC_PARTS+=(--venv "$(desktop_quote "${PYTHON_ARGS[1]}")")
else
    EXEC_PARTS+=(--python "$(desktop_quote "${PYTHON_ARGS[1]}")")
fi
EXEC_PARTS+=(--listen "$(desktop_quote "$LISTEN")" --port "$PORT")
[ "$NO_BROWSER" = 1 ] && EXEC_PARTS+=(--no-browser)
printf -v EXEC_LINE '%s ' "${EXEC_PARTS[@]}"
EXEC_LINE=${EXEC_LINE% }

mkdir -p -- "$XDG_ROOT/applications"
APPLICATIONS_DIR=$(realpath -e -- "$XDG_ROOT/applications")
DESKTOP_FILE="$APPLICATIONS_DIR/dgx-monarch.desktop"
if [ -e "$DESKTOP_FILE" ] || [ -L "$DESKTOP_FILE" ]; then
    if [ ! -f "$DESKTOP_FILE" ] || [ -L "$DESKTOP_FILE" ]; then
        fail "refusing to replace a desktop entry that is a symlink or not a regular file"
    fi
    grep -qx 'X-DGX-Monarch-Managed=true' "$DESKTOP_FILE" || fail "refusing to replace an unmanaged desktop entry (no X-DGX-Monarch-Managed=true line)"
fi

umask 077
TEMP_FILE=$(mktemp "$APPLICATIONS_DIR/.dgx-monarch.desktop.XXXXXX")
cleanup() {
    rm -f -- "$TEMP_FILE"
}
trap cleanup EXIT HUP INT TERM
cat > "$TEMP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=DGX Monarch
Comment=Launch the supervised DGX Monarch ComfyUI driver
Exec=$EXEC_LINE
Icon=$ICON
Terminal=true
Categories=Graphics;Utility;
StartupNotify=true
X-DGX-Monarch-Managed=true
EOF
chmod 0644 "$TEMP_FILE"
mv -f -- "$TEMP_FILE" "$DESKTOP_FILE"
trap - EXIT HUP INT TERM
printf 'Installed %s\n' "$DESKTOP_FILE"
printf 'The shortcut starts only ComfyUI; Worker services are unchanged.\n'
