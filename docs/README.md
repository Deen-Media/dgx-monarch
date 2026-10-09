# Documentation

Start with the installation and first-render guides. The reference documents
are available when you need details about a model, setting or failure.

## Install and use

| Goal | Guide |
|---|---|
| Install on two Sparks | [Installation](INSTALL.md) |
| Let a coding agent help | [Copyable prompt](../README.md#set-up-with-a-coding-agent) and [setup skill](../skills/dgx-monarch/SKILL.md) |
| Save the first distributed render | [Quickstart](QUICKSTART.md) |
| Choose a model and GPU layout | [Model support](MODELS.md) |
| Understand Workers, meshes and rendering | [Concepts](CONCEPTS.md) |
| Configure the pair | [Cluster setup](CLUSTER.md) and [network requirements](../SECURITY.md#fabric-trust-boundary) |
| Watch renders and memory | [Terminal dashboard](TUI.md) |
| Resolve a problem | [Troubleshooting](TROUBLESHOOTING.md) and [FAQ](FAQ.md) |

## Check results

[Benchmarks](BENCHMARKS.md) gives measured render times and memory use with the
settings needed to interpret them. [Validation](VALIDATION.md) records model
comparisons, installation results and known limits. These are references for a
specific configuration; you do not need to read them before installing.

[Campaign results](CAMPAIGN_RESULTS.md) summarizes the retained datasets,
including failures and retries. The linked exports and column definitions are
available for closer analysis.

## Contribute

Start with [Contributing](../CONTRIBUTING.md) and the
[developer skill](../skills/dgx-monarch-dev/SKILL.md).

| Work | Reference |
|---|---|
| Understand the implementation | [Design and architecture](DESIGN.md) |
| Add or maintain a model adapter | [Adapters](ADAPTERS.md) |
| Compare render results | [Sweep harness](SWEEP.md) |
| Change first-use checks or reuse rules | [Gate trust model](TRUST.md) |
| Review deployment boundaries | [Security](../SECURITY.md) and [threat model](THREAT_MODEL.md) |
| Update logos or workflow cards | [Brand assets](BRANDING.md) |
| Maintain generated test graphs | [Workflow fixtures](../tests/fixtures/workflows/README.md) |

[License notes](../LICENSE-NOTES.md) explain dependency and contribution terms.
[Changelog](../CHANGELOG.md) describes releases.
