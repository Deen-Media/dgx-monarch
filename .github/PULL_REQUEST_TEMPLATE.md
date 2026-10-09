## What changed, and why

## Provenance (DESIGN.md §3)

- [ ] Submitted code is original work or material I have the right to contribute
- [ ] Relevant external specifications, APIs, or behavior sources are identified
- [ ] No third-party implementation code is vendored or copied into this change
- [ ] xfuser stays external: no xfuser code is committed, and only `tools/build_xfuser_compat_wheel.py` changes its wheel
- [ ] DCO sign-off (`git commit -s`)

## Validation

- [ ] `ruff check __init__.py src tests benchmark tools` + `bash scripts/test_cpu.sh` green
- [ ] Adapter changes: acceptance gates run (cross-rank identity + fidelity against the single-GPU reference)
- [ ] Node API diff reviewed (mappings/inputs/defaults are semver-governed)
