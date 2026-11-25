# Studio workflow compatibility fixtures

The two App Mode graphs below are synthetic graph fixtures, not captured
successful Comfy executions. The tests provide their node schemas. Unknown
metadata deliberately contains inert markup to ensure round trips do not
discard extension data.

`app_mode_1_41_13.json` exercises the `[nodeId, widgetName]` input tuples used
by the [v1.41.13 App Mode store](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.41.13/src/stores/appModeStore.ts).

`app_mode_canonical.json` exercises the canonical widget identity and optional
configuration contract inspected on 2026-10-19 in the upstream
[widget identity implementation](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/types/widgetId.ts)
and [App Mode store](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/stores/appModeStore.ts).
Its frontendVersion field is a fixture contract label, not a released version.

Tests derive deleted-node, duplicate-label, nested-graph, legacy-path,
unknown-widget, image/video-output, rebinding and cycle cases from these graphs.
Nested graphs are inspected and retained; executing subgraph serialization and
custom JavaScript widgets requires a separately validated adapter. They are
refused here. A frontend-exported API prompt is the supported execution input
for graphs whose UI-only transformations are not implemented by Studio.

## Recorded driver schema

`object_info_bundled.json` is not synthetic. It is trimmed from a real
`/object_info` response that a driver returned on 2026-11-15, running ComfyUI
0.45.0 at commit 4d64a8bc0, ComfyUI frontend package 1.58.5 and dgx-monarch
dev f297c70, with the rig's other node packs loaded. It replaces a 2026-10-20
recording (ComfyUI d52a7a5ef, v0.42.0-14, frontend 1.55.18, dgx-monarch dev
61f446f) that lacked the node classes of the templates added since: Wan 2.2
TI2V (`Wan22ImageToVideoLatent`), p-h3-image and MiniMax-H3-Image (their
conditioning nodes, `DGXMonarchBasicGuider`,
`DGXMonarchModelSamplingMiniMaxH3`, `ComfySwitchNode`, `TextGenerateV2`,
`PreviewAny`, `ResizeImageMaskNode` and the string nodes `StringCompare`,
`StringReplace`, `StringTrim`, `RegexExtract` and `RegexReplace`). Rebuilt,
the two also differ where ComfyUI changed between the releases: Create Video's
`chroma_subsampling` and `prores` codec, Save Video's `mov` format, the CLIP
type `fibo2`, a hidden `prompt` input on the DGX Monarch samplers, the new
templates' model files and the Init node's `minimax_h3_image` family.
`tests/test_studio_bundled_workflows.py`
inspects and compiles every bundled workflow against it, so it covers the
frontend's upload helper slots, V3 dynamic combos, union widgets and Autogrow
sockets as a real driver declares them.

`build_object_info_bundled.py` made it from the saved response, and running it
again on a new recording replaces it. The script:

- keeps only the node classes the bundled workflows use;
- keeps only the fields Studio reads (`input`, `input_order`, `output`,
  `output_is_list`, `output_name`, `output_node`, `name` and `python_module`)
  and drops every tooltip and description;
- cuts each file-list choice (model, LoRA and input file names) down to the
  names the bundled workflows reference, all of them public in
  `example_workflows/`, and leaves other choice lists, such as sampler names,
  whole;
- stops if a referenced name is missing from the recording, except for the
  example names it lists in `NOT_IN_RECORDING`.

The recording rig had not installed two example files that public workflows
name, so the script adds them to their loader's choices:
`your_first_lora.safetensors` and `your_second_lora.safetensors`
(DGXMonarchLoraLoader). The other example inputs the 2026-10-20 recording
lacked, six images, three videos and one audio file, are installed on the rig
now and come from the recording. Whether a file exists stays the driver's
check at execution.

The script also writes `stock_text_classes.json`: every stock class in the
recording (its `python_module` is `nodes`, `comfy_extras.*` or
`comfy_api_nodes.*`) with a text input and a text or conditioning output, with
its text inputs and whether each box is multi-line. Studio's text-through rule
(`studio/src/textThrough.ts`) holds a row for each, read from the class's
source, and `tests/test_studio_template_layouts.py` fails when a listed class
has no row. A new recording that adds a stock text class needs its row.

`compiled_template_graphs.json` holds each bundled template as the API prompt
a job records, the form Reuse settings and Upscale load back into a draft: per
node its class, its links and its text inputs, with the values emptied.
`build_compiled_template_graphs.py` writes it from the bundled workflows and
`object_info_bundled.json`, and `tests/test_studio_fixture_builder.py` fails
when it is stale, so run the script after a template or the recorded schema
changes. `studio/src/fieldPresentation.test.ts` checks that every template
names the same Prompt and Negative prompt fields in this form as saved.

## Recording a new driver schema

A template that adds a node class the recording lacks does not fail the
suite. `tests/test_studio_bundled_workflows.py` skips that template, and the
coverage test, with a reason that names each missing class, and so does
`tests/test_studio_template_layouts.py` for its layout checks. To cover it, record
a new schema on a driver that has those nodes installed:

1. Start ComfyUI with the dgx-monarch custom node and every node pack the
   bundled workflows use, then save its schema to a durable directory outside
   the repository, for example
   `curl -sf http://127.0.0.1:8188/object_info -o ~/studio-evidence/object_info.json`.
2. Run `python tests/fixtures/studio/build_object_info_bundled.py ~/studio-evidence/object_info.json`.
   If it stops on a missing name, install that model or input file on the
   driver and record again, or add the name to `NOT_IN_RECORDING` when it is
   an example file that public workflows ship.
3. Update the recording date and the ComfyUI, frontend and dgx-monarch
   versions in the section above, then run
   `pytest tests/test_studio_bundled_workflows.py` and check that nothing is
   skipped. Run `pytest tests/test_studio_template_layouts.py` too: a stock
   class that the new `stock_text_classes.json` lists without a row in
   `studio/src/textThrough.ts` and the layout test's copy fails it. Then run
   `python tests/fixtures/studio/build_compiled_template_graphs.py`.
