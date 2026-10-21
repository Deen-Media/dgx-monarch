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
`/object_info` response that a driver returned on 2026-10-20, running ComfyUI
commit d52a7a5ef (v0.42.0-14), ComfyUI frontend package 1.55.18 and the
dgx-monarch tree that adds MOSS-Image on dev 61f446f, with the rig's other node
packs loaded. It replaces a 2026-10-16 recording (ComfyUI 29e28e2b4, frontend
1.55.17, dgx-monarch dab81e3bb02c) that lacked the MOSS-Image nodes, files and
CLIP type; rebuilt, the two differ only in those additions.
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

The recording rig had not installed thirteen example files that public
workflows name, so the script adds them to their loader's choices:
`your_first_lora.safetensors` and `your_second_lora.safetensors`
(DGXMonarchLoraLoader); `blue_dancer.png`, `qwen21_canvas_rgba.png`,
`qwen21_mask_rgba.png`, `qwen21_source.png`, `reference_character.png` and
`reference_character_mask.png` (LoadImage); `driving_video.mp4`,
`driving_video_mask.mp4` and `source_video.mp4` (LoadVideo); and
`wan_dancer_reference_audio.mp3` (LoadAudio). Whether a file exists stays the
driver's check at execution.

## Recording a new driver schema

A template that adds a node class the recording lacks does not fail the
suite. `tests/test_studio_bundled_workflows.py` skips that template, and the
coverage test, with a reason that names each missing class. To cover it, record
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
   skipped.
