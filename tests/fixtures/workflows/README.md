# Workflow test fixtures

The `generated/` workflows exercise cases used by the automated checks and distributed
render sweeps. They are not shown in ComfyUI’s template browser. For ready-to-use
examples, start with [the quickstart](../../../docs/QUICKSTART.md).

The fixtures cover LoRA connections, prompt lengths, guide masks, multiple
guides and staged sampling. Some require synthetic media or a user-supplied
LoRA in place of `test_lora_placeholder.safetensors`. A fixture’s presence does
not establish model support; see [supported models](../../../docs/MODELS.md).

Edit `tools/gen_templates.py`, then run from the repository root:

```bash
python tools/gen_templates.py
python tools/gen_templates.py --check
python tools/check_artifacts.py --sync
```

The artifact sync preserves recorded download links, file sizes and checksums.
See [the sweep guide](../../../docs/SWEEP.md) for media preparation and hardware
validation requirements. Do not queue these workflows on a running cluster
without the required checks and approval.

The API-format JSON files in this directory are separate regression fixtures.
They are not produced by the workflow generator.
