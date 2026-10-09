#!/usr/bin/env bash
# Run CPU-only pytest checks and report total time and slowest tests.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
test_python=${DGXM_TEST_PYTHON:-python}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

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
exit "$status"
