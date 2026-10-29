## What changed, and why

## Provenance (DESIGN.md §3)

- [ ] Submitted code is original work or material I have the right to contribute
- [ ] Relevant external specifications, APIs, or behavior sources are identified
- [ ] No third-party implementation code is vendored or copied into this change, apart from the compiled Studio bundle that `npm run build` produces and the vendored MIT INT8 GEMM kernel file; a change to a bundled package's locked version updates `THIRD-PARTY-NOTICES.txt`, and a change to the kernel file keeps its notice and its list of changes
- [ ] xfuser stays external: no xfuser code is committed, and only `tools/build_xfuser_compat_wheel.py` changes its wheel
- [ ] DCO sign-off (`git commit -s`)

## Validation

- [ ] `ruff check __init__.py src tests benchmark tools studio/tests` + `bash scripts/test_cpu.sh` green
- [ ] Adapter changes: acceptance gates run (cross-rank identity + fidelity against the single-GPU reference)
- [ ] Node API diff reviewed (mappings/inputs/defaults are semver-governed)
