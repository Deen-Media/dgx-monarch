#!/usr/bin/env bash
# Run the gate checks CI runs beside the suite, then the CPU-only pytest
# checks, and report total time and slowest tests.
#
# Locally, a green run means CI's gates are green too, except the checks
# listed in tests/test_cpu_gates.py as CI-only (install pins, wheel and sdist
# builds, dgxm smoke). Every gate runs, even after one fails, and the suite
# runs after them; the exit status is non-zero if either failed. CI runs each
# gate as its own step, so this skips them under GitHub Actions, and
# DGXM_SKIP_GATES=1 skips them here.
# The gate helper calls the check functions below by name.
# shellcheck disable=SC2329
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

refusal_ledger_is_current() {
  local ledger=tests/refusal_class_ledger.json saved status
  saved=$(mktemp)
  cp "$ledger" "$saved"
  PYTHONPATH=src "$test_python" tests/test_refusal_classes.py --write > /dev/null
  cmp --silent "$saved" "$ledger"
  status=$?
  if [ "$status" -ne 0 ]; then
    diff -u "$saved" "$ledger" || true
  fi
  cp "$saved" "$ledger"
  rm -f "$saved"
  return "$status"
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
  set +e
  gate "protected test inventory" "$test_python" tools/check_protected_tests.py --check
  gate "refusal ledger" refusal_ledger_is_current
  gate "trust citations" "$test_python" tools/repin_trust_citations.py --check
  gate "leak check" "$test_python" tools/leak_check.py
  gate "ruff" "$test_python" -m ruff check __init__.py src tests benchmark tools studio/tests
  gate "mypy" "$test_python" -m mypy -p dgx_monarch
  gate "compileall" "$test_python" -m compileall -q __init__.py src benchmark tests tools studio/tests
  gate "templates" "$test_python" tools/gen_templates.py --check
  if "$test_python" -c 'import cairosvg' 2> /dev/null; then
    gate "brand assets" "$test_python" tools/gen_brand_assets.py --check
    gate "template cards" "$test_python" tools/gen_template_cards.py --check
  else
    skip "brand assets" "CairoSVG is not installed (CI runs it)"
    skip "template cards" "CairoSVG is not installed (CI runs it)"
  fi
  gate "artifact manifest" "$test_python" tools/check_artifacts.py --strict
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
