# Contributing

Keep each PR focused. Explain the problem, resulting behavior and validation.
Read the [developer skill](skills/dgx-monarch-dev/SKILL.md) for development or
the [setup skill](skills/dgx-monarch/SKILL.md) for installation.

## Set up a development environment

Follow the verified xFuser wheel and Torch-constraint procedure in
[docs/INSTALL.md](docs/INSTALL.md). Use `-e "$REPO_DIR[dev]"` in both the
reviewed dry run and the install command. Keep the temporary wheel directory
until installation finishes. The dev tools are listed in `pyproject.toml`.

Use the Python environment you selected throughout. Preserve ComfyUI's working
CUDA Torch build and the repository's exact Monarch pin; an older Monarch can
lack required APIs such as `concurrent_endpoint`. For development without a
GPU, follow the dependency setup in [.github/workflows/ci.yml](.github/workflows/ci.yml)
in a separate environment. Its CPU runner uses a CUDA-linked Torch wheel
because FakeTensor tests exercise CUDA device guards. This does not run GPU
workloads.

## Check your change

Run these from the checkout in the development environment for Python changes:

```bash
ruff check __init__.py src tests benchmark tools
mypy -p dgx_monarch
bash scripts/test_cpu.sh
```

`DGXM_TEST_PYTHON` selects the interpreter. The wrapper uses this checkout's
source, defaults OMP/MKL to one thread and reports the 25 slowest tests; its
120-second warning is advisory. `HYPOTHESIS_PROFILE=ci` uses deterministic
generation without deadlines or an example database. Bare pytest has different
defaults.

Choose additional checks based on the files and behavior you changed:

| Change | Validation |
|---|---|
| Documentation or comments | Run the affected documentation tests, check links and examples, and verify any source citations. No hardware run is needed for wording alone. |
| Nodes or browser UI | Run the node contracts and relevant UI tests; preserve saved-workflow compatibility. Check JavaScript syntax and segment mappings. |
| Workflow templates or model-file lists | Regenerate changed templates, then run the template and artifact checks. Do not invent model hashes or treat a pending artifact as verified. |
| Logos or workflow cards | Run the brand-asset and template-card generators' checks. |
| Runtime or distributed behavior | Run the CPU suite and relevant regression tests. A render-path change also needs an authorized hardware run before claiming it works on hardware. |
| Model adapters or topology | Run the required cross-rank and reference comparisons described below. CPU success does not establish model accuracy. |

CI still runs its full required checks for every PR. The workflow is the
complete command reference; its additional checks include:

```bash
python tools/leak_check.py
python tools/repin_trust_citations.py --check
python tools/check_protected_tests.py --check
python tools/gen_templates.py --check
python tools/gen_brand_assets.py --check
python tools/gen_template_cards.py --check
python tools/check_artifacts.py
python -m compileall -q __init__.py src benchmark tests tools
for script in scripts/*.sh; do bash -n "$script"; done
shellcheck scripts/*.sh
for asset in web/js/*.js; do node --check "$asset"; done
node tools/check_dgx_monarch_segments.mjs
```

Run `python tools/check_dependencies.py` in the test environment. CI runs
the same tool; see the [verified upstream exception](docs/TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform)
if it prints a known platform-tag failure. CI also checks the refusal ledger
and builds and inspects both package artifacts on x64. The hosted ARM64 job
`lint-and-unit-arm64` uses Ubuntu 24.04 and Python 3.12 for dependency,
Torch/CUDA-build and full CPU checks; x64 covers the wheel build. To check the ledger locally, regenerate it with
`PYTHONPATH=src python tests/test_refusal_classes.py --write` and review any
diff in `tests/refusal_class_ledger.json`. An unexpected diff must be resolved,
not committed just to make the check pass.

### Test and citation maintenance

Tests must restore `os.environ`: use `monkeypatch.setenv` or, when the tested
code writes it, `isolated_environ`. The test guard reports and restores leaks.
Placeholder hosts such as `worker` skip DNS; the worker import policy resets
after each test. Synchronize processes through events with timeouts, not fixed
sleeps, and share expensive deterministic setup across tests.

Protected tests are listed in `tests/protected_test_inventory.json`. Their
digests cover normalized test source and the module-level helpers, fixtures,
marks and tables they use. After reviewing a deliberate change, run
`python tools/check_protected_tests.py --write`, include the inventory diff,
and list added, removed or changed protected tests in the PR. A digest detects
change; it does not prove equivalent coverage or track imported helpers,
conftest fixtures or runtime code.

For moved source or test citations, run
`python tools/repin_trust_citations.py --base <rev>` and review the
`docs/TRUST.md` diff. The optional `pre-commit install` hook runs the same
citation check as CI. Gate changes must also be reviewed against the numbered
invariants in [TRUST](docs/TRUST.md).

## Preserve the public API and dependency boundaries

The 1.x API must keep existing workflows loading. `NODE_CLASS_MAPPINGS` keys
and ordered input names, types and defaults are public API. Review
`tests/node_input_contract.json` and `tests/test_node_contract.py` when adding
a node or changing its definition. New inputs must be optional or hidden and
appended at the end. Patch releases fix bugs; minor releases add compatible
features. A breaking API change requires a major version and a migration plan.
See [versioning policy](docs/DESIGN.md#63-versioning-and-compatibility-policy).

Submit original work or material you have the right to contribute. Identify
external specifications, APIs or behavior that informed the change. Do not
copy or vendor dependency implementations. xFuser remains external: its
compatibility wheel is built locally from the verified official wheel, and
neither the wheel nor dependency source belongs in the repository.

External contributions require a DCO sign-off on each commit with
`git commit -s`. There is no Contributor License Agreement. See
[LICENSE-NOTES.md](LICENSE-NOTES.md).

Keep machine addresses and other host-specific settings out of `src/`; use
`cluster.toml` or a fabric profile. Approximate caching or staleness
accelerators such as TeaCache are outside project scope.

## Pull requests and hardware validation

Add the change under `[Unreleased]` in the changelog, creating that section if
needed. Document new failure modes in a numbered [troubleshooting entry](docs/TROUBLESHOOTING.md). A fix for
an upstream ComfyUI break follows [DESIGN §5.10](docs/DESIGN.md): add a behavioral
test for the affected stock API to `tests/canary/comfy_seam_contracts.py`.

Name the checks, hardware class, configuration and limits. Support correctness
claims with identity-gate results or fidelity measurements as required by
[ADAPTERS](docs/ADAPTERS.md). A new family's template-derived cases must run through the
[sweep harness](docs/SWEEP.md) on a real pair against a single-GPU reference.
Attach reviewed, sanitized results.

The public repository has no hardware-runner route. Follow the manual
[hardware-smoke contract](docs/THREAT_MODEL.md#github-actions-to-trusted-hardware):
use reviewed hand-run scripts from the private control repository, with
operator authorization, exact source commit and manifest verification on every
host, and passive idle and ownership checks. CI, Actions runners and scheduled
jobs must never execute on protected hardware. A client must not stop persistent
Worker services as cleanup. Service stops or restarts require explicit
maintenance authorization and verified restoration.

Keep raw evidence private. Record **NOT RUN** for a check without a clean
recorded run. Publish measured claims in [VALIDATION](docs/VALIDATION.md) with
the actual run date and settings; a documentation edit is not a hardware test.
Cross-rank agreement alone does not prove agreement with native ComfyUI.
Native latent RDMA remains **NOT RUN / HOLD**; actor messaging is the supported
return path.

## Release checks

Release maintainers must complete these checks on the intended release commit:

1. CI and the [ComfyUI-master canary](.github/workflows/comfy-canary.yml) pass.
   The canary checks node-pack import, schemas, adapters, bake/slab behavior,
   every committed template and the stock API contracts in
   `tests/canary/comfy_seam_contracts.py` on CPU.
2. `python -m build` succeeds. Inspect both artifacts and install them only in
   disposable validation environments. The wheel contains the package and CLI;
   the source checkout provides the complete ComfyUI node pack. This project
   does not publish package artifacts.
3. Complete the authorized manual hardware checks above, including reference
   comparisons for render-path or topology changes. Preserve their scope and
   report anything unperformed as **NOT RUN**.
4. Review `[Unreleased]`, regenerate workflow templates, and check that
   dependency declarations and runtime checks agree. `dgxm update` changes only
   the exact torchmonarch pin and dgx-monarch, both with `--no-deps`; it must not
   replace Torch, NCCL, xFuser, yunchang or optional extras.
5. Follow the [developer skill's release checklist](skills/dgx-monarch-dev/SKILL.md#release-checklist)
   for the version, changelog, final node API review and source-install smoke.
   Tag only the checked commit and repeat the source-install smoke from that tag.

## Report a problem

Use the bug-report form with sanitized Doctor output, relevant Worker logs,
model, precision, topology and troubleshooting steps already tried. Follow its
redaction instructions before posting diagnostics.

Report security vulnerabilities through the private GitHub Security Advisory
process in [SECURITY.md](SECURITY.md), not a public issue. This is a
solo-maintainer project; responses are best effort, with security reports and
regressions on documented hardware paths taking priority.
