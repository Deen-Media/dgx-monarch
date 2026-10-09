# License notes

dgx-monarch is licensed under the Apache License, Version 2.0. See `LICENSE`
for the terms. The copyright holder is Deen Media, LLC.
This file explains which components the repository distributes and which users
install separately. It is not legal advice.

## Direct and optional runtime dependencies

| Component | License | Relationship |
|---|---|---|
| dgx-monarch | Apache-2.0 | Source distributed by this repository; copyright Deen Media, LLC |
| torchmonarch | BSD-3-Clause | Runtime dependency installed separately |
| xfuser / xDiT | Apache-2.0 | External runtime dependency; the temporary local compatibility wheel changes only the version metadata and makes the optional-NPU import lazy |
| yunchang | Apache-2.0 | Runtime dependency installed separately; also required by xfuser |
| ComfyUI | GPL-3.0 | Runtime host installed separately by the user |
| Textual | MIT | Optional TUI dependency installed separately |
| nvidia-ml-py | BSD | Optional TUI telemetry dependency installed separately |
| sol-attn | see its own notices | Optional sparse-attention kernel installed separately; its distribution declares no single license expression, and its `THIRD_PARTY_NOTICES.md` names BSD-3-Clause and Apache-2.0 components |
| apache-tvm-ffi | Apache-2.0 | Optional; the Sol-Attn path needs it beside sol-attn, installed separately |
| nvidia-cutlass-dsl | NVIDIA Software License Agreement | Optional CuTe compiler for the Sol-Attn path, installed separately |

Dependency projects retain their own licenses, notices, and distribution
terms. The repository's `NOTICE` file contains the dgx-monarch copyright line.
The generated wordmarks contain DejaVu Sans Bold glyph outlines; their font
notice is preserved in `THIRD-PARTY-NOTICES.txt`. The repository includes no
vendored dependency implementations or font binaries. Redistributions under
Apache-2.0 section 4(d)
keep `NOTICE` with the source. Packaging or redistributing dgx-monarch together
with other software may create additional obligations; redistributors are
responsible for assessing the combination they ship.

## Source boundary

dgx-monarch integrates with separately installed projects through their public
or runtime interfaces. Model adapters implement distributed execution for
supported models without including dependency implementations.

Contributions must be the contributor's original work or material the
contributor has the right to submit under this repository's license. When a
change depends on an external specification, API, or behavior, identify that
source in the pull request. Do not copy third-party implementation code into
this repository. Install xfuser and other runtime dependencies separately. The
temporary xFuser wheel is rebuilt locally from a SHA-verified official wheel;
no third-party wheel or source is committed.

## Contributions

External contributions require a Developer Certificate of Origin sign-off via
`git commit -s`. The project does not require a Contributor License Agreement.
