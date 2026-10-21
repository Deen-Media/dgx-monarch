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

On Ubuntu, install `libcairo2` and `fonts-dejavu-core`, as CI does, for the
brand and card checks. In an environment without the dev extra's CairoSVG,
pass `--ignore=tests/test_brand_assets.py --ignore=tests/test_template_cards.py`
to `scripts/test_cpu.sh` and rely on CI for those two files.

## Check your change

Run these from the checkout in the development environment for Python changes:

```bash
ruff check __init__.py src tests benchmark tools studio/tests
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
| Nodes or browser UI | Run the node contracts and relevant UI tests; preserve saved-workflow compatibility. Check JavaScript syntax and segment mappings. Then regenerate and check the Studio inventory, as described below. |
| Workflow templates or model-file lists | Regenerate changed templates, then run the template checks and `python tools/check_artifacts.py --strict`. Fill each new manifest row from a public file that matches the tested file byte for byte, or mark it a placeholder with a note. Never invent a model hash. Then regenerate and check the Studio inventory. |
| Studio interface (`studio/`) | Follow the [Studio development steps](docs/STUDIO.md#development), which build the bundle and then regenerate and check the Studio inventory. |
| Logos or workflow cards | Run the brand-asset and template-card generators' checks. After a `dgxm top` layout change, run `python tools/gen_dgxm_top_image.py` and render `docs/media/dgxm-top.png` again as [BRANDING](docs/BRANDING.md) describes. |
| Runtime or distributed behavior | Run the CPU suite and relevant regression tests. A render-path change also needs an authorized hardware run before claiming it works on hardware. |
| Model adapters or topology | Run the required cross-rank and reference comparisons described below. CPU success does not establish model accuracy. |

The Studio inventory records the CLI flags, config fields, panel actions,
workflow templates and shared presentation rules that Studio presents, and the
hashes of the bundled frontend. A change to any of them, or to the Studio
frontend, makes it stale. Regenerate it, then check it:

```bash
python tools/gen_studio_inventory.py
python tools/gen_studio_inventory.py --check
```

Commit the regenerated `src/dgx_monarch/studio/data/*.json`,
`src/dgx_monarch/studio/resource_hashes.py` and
`web/js/dgx_monarch_presentation.js`, and the rebuilt bundle under
`src/dgx_monarch/studio/static/`, with the source change. Verify
`dgxm-studio --check` from both installed package formats after a Studio
change. [Studio acceptance](docs/STUDIO_ACCEPTANCE.md) remains separate from
CPU and mocked HTTP coverage.

CI still runs its full required checks for every PR. The workflows under
`.github/workflows/` are the complete command reference; their additional
checks include:

```bash
python tools/leak_check.py
python tools/repin_trust_citations.py --check
python tools/check_protected_tests.py --check
python tools/gen_templates.py --check
python tools/gen_brand_assets.py --check
python tools/gen_template_cards.py --check
python tools/check_artifacts.py --strict
python tools/gen_studio_inventory.py --check  # after regenerating it, as above
python -m compileall -q __init__.py src benchmark tests tools studio/tests
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
copy or vendor dependency implementations; the compiled Studio bundle is build
output, listed in [LICENSE-NOTES.md](LICENSE-NOTES.md#studio-frontend-bundle).
xFuser remains external: its
compatibility wheel is built locally from the verified official wheel, and
neither the wheel nor dependency source belongs in the repository.

External contributions require a DCO sign-off on each commit with
`git commit -s`. There is no Contributor License Agreement. See
[LICENSE-NOTES.md](LICENSE-NOTES.md).

Keep machine addresses and other host-specific settings out of `src/`; use
`cluster.toml` or a fabric profile. Approximate caching or staleness
accelerators such as TeaCache are outside project scope.

## Pull requests and hardware validation

Add a changelog bullet as [Changelog entries](#changelog-entries) describes.
Document new failure modes in a numbered [troubleshooting entry](docs/TROUBLESHOOTING.md). A fix for
an upstream ComfyUI break follows [DESIGN §5.10](docs/DESIGN.md): add a behavioral
test for the affected stock API to `tests/canary/comfy_seam_contracts.py`.

Name the checks, hardware class, configuration and limits. Support correctness
claims with identity-gate results or fidelity measurements as required by
[ADAPTERS](docs/ADAPTERS.md). A new family's template-derived cases must run through the
[sweep harness](docs/SWEEP.md) on a real pair against a single-GPU reference.
Attach reviewed, sanitized results.

The public repository has no hardware-runner route. Follow the manual
[hardware-smoke contract](docs/THREAT_MODEL.md#github-actions-to-trusted-hardware):
use reviewed hand-run scripts and tools from this repository, with
operator authorization, exact source commit and manifest verification on every
host, and passive idle and ownership checks. CI, Actions runners and scheduled
jobs must never execute on protected hardware. A client must not stop persistent
Worker services as cleanup. Service stops or restarts require explicit
maintenance authorization and verified restoration.

Use the validation protocol in [VALIDATION](docs/VALIDATION.md), the
[sweep harness](docs/SWEEP.md) and the tools under `benchmark/` and `tools/`;
no archive or separate hardware-control repository is required. Keep
host-specific wrappers, configuration and raw evidence in a private workspace.
Review any wrapper before execution.

Keep raw evidence private. Record **NOT RUN** for a check without a clean
recorded run. Publish measured claims in [VALIDATION](docs/VALIDATION.md) with
the actual run date and settings; a documentation edit is not a hardware test.
Cross-rank agreement alone does not prove agreement with native ComfyUI.
Native latent RDMA remains off by default; actor messaging is the normal
return path. [Selected-rail qualification](docs/VALIDATION.md#native-rdma)
covers only its recorded scope, not untested cancellation or fabric-failure
recovery. Do not generalize one passing configuration to another workload.

### Changelog entries

Give each change a user can see one bullet under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md); add that heading if it is missing. A change
earns a bullet when a user or operator would notice it: a new model,
workflow, node input, command, flag, refusal, Doctor row, metric or config
key; a fixed defect they could hit; a changed default, dependency pin or
ComfyUI minimum; or a new support or validation claim. Refactors, test-only
changes and doc wording fixes earn none.

Keep each bullet to two or three lines at 80 columns. Say what changed and
what the user does about it, in the active voice. Link the troubleshooting
entry, issue or doc section that holds the detail; the pull request keeps the
narrative, the test list and the measurements. A later pull request that
changes the same feature before release edits that bullet instead of adding
one.

At release, fold `[Unreleased]` into the version's section, grouped as
Everyday use, Installation and updates, Compatibility and verification,
Rendering, Operations, Development and Limits. Open it with an "Upgrading
from" paragraph when an update needs manual steps. Every user-visible change
keeps a clause; merge related bullets, fold internal work into one line per
group or drop it, and keep the section within about 110 lines.

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
4. Fold `[Unreleased]` into the release section as
   [Changelog entries](#changelog-entries) describes, regenerate workflow
   templates, and check that dependency declarations and runtime checks agree.
   `dgxm update` changes only the exact torchmonarch pin and dgx-monarch, both
   with `--no-deps`; it must not replace Torch, NCCL, xFuser, yunchang or
   optional extras.
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
