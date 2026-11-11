#!/usr/bin/env bash
# Run the gate checks CI runs beside the suite, then the CPU-only pytest
# checks, and report total time and slowest tests.
#
# The gates are the CI lint-and-unit steps that can run on a developer's box.
# A green run here does not promise a green CI run: tests/test_cpu_gates.py
# names the steps only CI runs (its pinned install, the wheel build, the smoke
# environments, dgxm smoke), CI's tools can be other versions than the ones
# here, and a gate whose tool is absent here is skipped and named. Every gate
# runs, even after one fails, and the suite runs after them; the exit status
# is non-zero if either failed. CI runs each gate as its own step, so this
# skips them under GitHub Actions, and DGXM_SKIP_GATES=1 skips them here.
# No gate writes a tracked file: each compares in memory or works in its
# run's own scratch folder, so two runs side by side cannot cross.
# The gate helper calls the check functions below by name. ShellCheck 0.9.0,
# which CI's runner carries, reports that as SC2317; 0.11 reports it as SC2329.
# shellcheck disable=SC2317,SC2329
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
test_python=${DGXM_TEST_PYTHON:-python}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

gates_ok=0
gates_failed=()
gates_skipped=()
gate_tmp=""  # the gates' one scratch folder, outside the checkout; removed at exit

gate() {  # gate <name> <command...>: run one check, report it, never stop the run
  local name=$1
  shift
  local began=$SECONDS
  if "$@"; then
    gates_ok=$((gates_ok + 1))
    printf 'gate %s: ok (%ss)\n' "$name" "$((SECONDS - began))"
  else
    gates_failed+=("$name")
    printf 'gate %s: FAILED (%ss)\n' "$name" "$((SECONDS - began))"
  fi
}

skip() {  # skip <name> <reason>
  gates_skipped+=("$1")
  printf 'gate %s: skipped, %s\n' "$1" "$2"
}

sources_compile() {  # compileall writes its caches under the scratch folder, not the tree
  PYTHONPYCACHEPREFIX="$gate_tmp/pycache" \
    "$test_python" -m compileall -q __init__.py src benchmark tests tools studio/tests
}

source_archive_is_complete() {  # CI's sdist census, built from a copy so the checkout is untouched
  local scratch=$gate_tmp/sdist
  mkdir -p "$scratch/tree" || return 1
  git ls-files -z > "$scratch/tracked" || return 1
  # A tracked file deleted here and not yet staged draws a warning, and the census leaves it out.
  tar --null -T "$scratch/tracked" --ignore-failed-read -cf - | tar -xf - -C "$scratch/tree" || return 1
  if ! "$test_python" -m build --sdist --no-isolation --outdir "$scratch/dist" "$scratch/tree" \
    > "$scratch/build.log" 2>&1; then
    tail -20 "$scratch/build.log"
    return 1
  fi
  "$test_python" - "$scratch/tree" "$scratch/tracked" "$scratch"/dist/*.tar.gz << 'PY'
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

tree = Path(sys.argv[1])
tracked = {name for name in Path(sys.argv[2]).read_text().split("\0")
           if name and os.path.lexists(tree / name)}
generated = {
    "PKG-INFO",
    "setup.cfg",
    "src/dgx_monarch.egg-info/PKG-INFO",
    "src/dgx_monarch.egg-info/SOURCES.txt",
    "src/dgx_monarch.egg-info/dependency_links.txt",
    "src/dgx_monarch.egg-info/entry_points.txt",
    "src/dgx_monarch.egg-info/requires.txt",
    "src/dgx_monarch.egg-info/top_level.txt",
}
with tarfile.open(sys.argv[3]) as archive:
    members = {PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
               for member in archive.getmembers()
               if member.isfile() and len(PurePosixPath(member.name).parts) > 1}
expected = tracked | generated
if members != expected:
    print(f"sdist_missing: {sorted(expected - members)}")
    print(f"sdist_extra: {sorted(members - expected)}")
    print("MANIFEST.in decides what the source archive holds; CI runs this census on every push.")
    sys.exit(1)
PY
}

shell_scripts_parse() {
  local script
  for script in scripts/*.sh; do
    bash -n "$script" || return 1
  done
}

browser_assets_parse() {
  local asset
  for asset in web/js/*.js; do
    node --check "$asset" || return 1
  done
  node tools/check_dgx_monarch_segments.mjs
}

run_gates() {
  gate_tmp=$(mktemp -d)
  trap 'rm -rf "$gate_tmp"' EXIT
  set +e
  gate "protected test inventory" "$test_python" tools/check_protected_tests.py --check
  gate "refusal ledger" "$test_python" tests/test_refusal_classes.py --check
  gate "trust citations" "$test_python" tools/repin_trust_citations.py --check
  gate "leak check" "$test_python" tools/leak_check.py
  gate "ruff" "$test_python" -m ruff check __init__.py src tests benchmark tools studio/tests
  gate "mypy" "$test_python" -m mypy -p dgx_monarch
  gate "compileall" sources_compile
  gate "templates" "$test_python" tools/gen_templates.py --check
  if "$test_python" -c 'import cairosvg' 2> /dev/null; then
    gate "brand assets" "$test_python" tools/gen_brand_assets.py --check
    gate "template cards" "$test_python" tools/gen_template_cards.py --check
  else
    skip "brand assets" "CairoSVG is not installed (CI runs it)"
    skip "template cards" "CairoSVG is not installed (CI runs it)"
  fi
  gate "artifact manifest" "$test_python" tools/check_artifacts.py --strict
  if ! "$test_python" -c 'import build' 2> /dev/null; then
    skip "source archive" "the build module is not installed (CI runs it)"
  elif ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    skip "source archive" "this is not a Git checkout (CI runs it)"
  else
    gate "source archive" source_archive_is_complete
  fi
  gate "shell syntax" shell_scripts_parse
  if command -v shellcheck > /dev/null; then
    gate "shellcheck" shellcheck scripts/*.sh
  else
    skip "shellcheck" "shellcheck is not installed (CI runs it)"
  fi
  if command -v node > /dev/null; then
    gate "browser assets" browser_assets_parse
  else
    skip "browser assets" "node is not installed (CI runs it)"
  fi
  set -e
  printf 'gates: %s ok, %s failed%s, %s skipped%s\n' "$gates_ok" "${#gates_failed[@]}" \
    "$( [ "${#gates_failed[@]}" -gt 0 ] && printf ' (%s)' "$(IFS=,; echo "${gates_failed[*]}")")" \
    "${#gates_skipped[@]}" \
    "$( [ "${#gates_skipped[@]}" -gt 0 ] && printf ' (%s)' "$(IFS=,; echo "${gates_skipped[*]}")")"
}

if [ "${GITHUB_ACTIONS:-}" != "true" ] && [ "${DGXM_SKIP_GATES:-}" != "1" ]; then
  run_gates
fi

started=$(date +%s)
set +e
HYPOTHESIS_PROFILE=ci "$test_python" -m pytest tests/ -q --durations=25 "$@"
status=$?
set -e
elapsed=$(( $(date +%s) - started ))
printf 'CPU battery completed in %ss\n' "$elapsed"
if [ "$elapsed" -gt 120 ]; then
  warning='CPU battery exceeded the 120s target; inspect the duration report.'
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    printf '::warning::%s\n' "$warning"
  else
    printf 'WARNING: %s\n' "$warning" >&2
  fi
fi
if [ "${#gates_failed[@]}" -gt 0 ] && [ "$status" -eq 0 ]; then
  status=1
fi
exit "$status"
