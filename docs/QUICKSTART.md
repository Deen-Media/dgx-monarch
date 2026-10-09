# Quickstart

This page assumes ComfyUI is installed, the dgx-monarch Git checkout lives in
its `custom_nodes` directory, and `dgxm doctor` is green. Follow
[INSTALL.md](INSTALL.md) for the source installation. If ComfyUI is missing,
start with the
[official ComfyUI manual-install guide](https://docs.comfy.org/installation/manual_install).
Terms are defined in [CONCEPTS.md](CONCEPTS.md).

Keep ComfyUI in the foreground. Leaving ComfyUI does not stop the Worker
services on the cluster hosts. `dgxm up`, guided setup with
`--start-worker-service`, auto-heal, or an installed user service can leave
them running. Check `dgxm status`; use `dgxm down` only for an intentional stop.

For your first distributed render, [choose a supported model](#first-distributed-render)
you already have or want to use. The optional Chroma recipe below provides
exact files and settings if you have no preference. Either way, verify that
both Sparks took part.
For agent-led installation, follow [the setup skill](../skills/dgx-monarch/SKILL.md).

## 1. Add the Init node

Add **DGX Monarch Init** (`Add Node -> DGX Monarch -> DGX Monarch Init`). It
attaches the cluster and returns one `mesh` handle. Other distributed nodes
receive that handle directly or through the loader. A graph with no DGX Monarch
nodes renders on the driver as usual, and queue advice lists what to add and
swap; a graph that uses DGX Monarch nodes without Init cannot run, and the
advice says so.

For the first two-Spark check, use explicit `mode=cluster` and a documented
two-rank configuration for your [selected model](#first-distributed-render). For ordinary use, `auto`
selects a topology for the requested resolution and logs its choice. Without
`cluster.toml`, `auto` runs locally, so it is not proof of a distributed render.
[MODELS.md](MODELS.md) lists family support, and
[CLUSTER.md](CLUSTER.md) covers the config file.

## 2. Only some nodes move to the cluster

DGX Monarch versions of model operations run on the mesh: the diffusion-model
loader, LoRA loader, samplers, guiders, Model Sampling SD3, and Basic
Scheduler. Templates include the DGX Monarch shift node where needed; the
stock node cannot accept a mesh model. The CLIP loader, text encodes, empty
latent, VAE decode, and save nodes remain in the ComfyUI process on both paths
and have no cluster variant.

On Queue, the driver reports each stock-node replacement by node id and whether
Init is missing. The advice neither blocks the queue nor edits the graph.
[TROUBLESHOOTING.md #63](TROUBLESHOOTING.md#63-a-toast-at-queue-time-says-the-workflow-is-missing-dgx-monarch-nodes)
contains the replacement list, the three nodes with no one-for-one swap, and
the one swap that carries a condition.

## 3. Open a template instead of building one

`Workflow -> Browse Templates -> dgx-monarch` contains starting graphs for
the supported image and video workflows, plus five general examples:
quickstart, a LoRA stack in low-RSS mode, Fleet, the Identity Gate, and a
dual-Spark split render. The [MODELS.md](MODELS.md) support matrix states each
variant's tested settings and identifies variants with no template yet.
Choose the template for your selected model; `dgx-monarch-chroma-t2i` is an
optional starting point. ComfyUI registers the route that serves a pack's
template folder only when the server starts, so after an install while
ComfyUI runs, this menu lists the new graphs but none of them open until you
restart ComfyUI. `dgxm update` refuses while ComfyUI runs, so an update ends
with a ComfyUI start too.

Each Note node describes the output and constraints. Family graphs also name
the required model files and directories. The five general graphs share this
Flux 1 Dev stack, without listing it in their Notes: `flux1-dev.safetensors`
in `models/diffusion_models`, `clip_l.safetensors` and
`t5xxl_fp16.safetensors` in `models/text_encoders`, and `ae.safetensors` in
`models/vae`. Three LTX 2.5 graphs name no files either,
`dgx-monarch-ltx25-i2v`, `dgx-monarch-ltx25-flf2v` and
`dgx-monarch-ltx25-i2v-guide`; take theirs from `dgx-monarch-ltx25-t2v`.

Templates carry placeholder filenames, not model downloads. Put the files in
place and select them on the loader widgets before you queue. If a family is
blocked, its Note says so and names the
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) entry that explains the hold; the Wan
SCAIL Preview graph includes this warning.

<a id="first-distributed-render"></a>

## 4. First distributed render

Use a supported model you already have or choose. Check the exact checkpoint,
precision, inputs and two-Spark configuration in [Model support](MODELS.md),
then follow its workflow notes and file list. Verify matching model files on
both hosts and any published hashes, and follow the model's access and license
requirements. Use `mode=cluster` and `auto_gate=first_use`; leave native latent
RDMA off. Follow the selected model's topology, attention, sampler and memory settings.

Chroma is optional. If you have no preference, the recipe below gives a tested
starting point. Its filenames and render settings apply only to Chroma. For
any selected model, require a saved output, [evidence from both Sparks](#collect-evidence-from-the-same-render),
and a successful setup rerun. The recorded fresh-agent trial used Chroma;
choosing another model does not extend that trial's results.

### Optional recipe: Chroma 1 HD

Use the existing [Chroma workflow](../example_workflows/dgx-monarch-chroma-t2i.json).
This is a small three-file starting point among the supported workflows. The
recipe uses the tested FP8-mixed checkpoint and cfg2 configuration described in
[Model support](MODELS.md#model-chroma). It does not establish support for other
precisions, parallel modes, or model families.

Finish setup on both Sparks first. Use matching DGX Monarch and ComfyUI
revisions, compatible dependencies, and identical model bytes under the same
relative model names on each host. Choose the setup `safe` profile for stock
residency and no native latent RDMA. All setup profiles enable `auto_heal`, which
can start or restart Worker services; the agent needs approval for that service
scope before rendering. See [setup profiles](CLUSTER.md#guided-setup-profiles)
and the setup skill for discovery, configuration, service ownership, and recovery.
Do not replace a working configuration merely to try this graph.

<a id="first-render-files"></a>

### Model files

Place these files below each host's ComfyUI `models` directory, or use its
existing model-path configuration after checking that it resolves the same
files. The links select fixed upstream revisions. The sizes and SHA256 values
come from those revisions' published file metadata; verify the actual bytes on
each host before use. Together they occupy 14,686,032,392 bytes, about 14.69 GB,
plus any download cache or temporary copies.

| File and destination below `models/` | Official download | Bytes | SHA256 |
|---|---|---:|---|
| `diffusion_models/Chroma1-HD-fp8mixed.safetensors` | [Comfy-Org repack](https://huggingface.co/Comfy-Org/Chroma1-HD_repackaged/resolve/409cfff960d66598d4d0ff46cb932f53492cc6d9/split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors) | 9,193,379,316 | `a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575` |
| `text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors` | [ComfyUI maintainer](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn_scaled.safetensors) | 5,157,348,688 | `a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a` |
| `vae/ae.safetensors` | [Chroma publisher](https://huggingface.co/lodestones/Chroma/resolve/2f3b2730d7b5edbd02cdff12f72610af5787300b/ae.safetensors) | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

The [Chroma repack](https://huggingface.co/Comfy-Org/Chroma1-HD_repackaged),
[text encoder](https://huggingface.co/comfyanonymous/flux_text_encoders), and
[publisher's VAE repository](https://huggingface.co/lodestones/Chroma) list Apache
2.0 licensing and were accessible without account approval when this recipe
was checked. Read their license and model cards and retain applicable notices.
The repack credits the original Chroma model and its FP8 conversion separately.

These files need no token. If an upstream source later requires access approval,
stop for the user to review and accept its conditions. Other templates, including
Flux Dev, have their own access and license requirements. Use the user's approved
credential store or interactive login when needed; never print a token, put it
in a command or committed file, or use another source to bypass a gate.

Check existing files before downloading anything. This read-only check assumes
`COMFY_DIR` is the discovered ComfyUI directory and its models are stored there:

```bash
(
  cd "$COMFY_DIR/models" || exit
  sha256sum --check <<'SHA256'
a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575  diffusion_models/Chroma1-HD-fp8mixed.safetensors
a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a  text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38  vae/ae.safetensors
SHA256
)
```

For configured external model folders, hash the resolved files against the same
values on both hosts. Guided setup's `--artifact` check only accepts files whose
resolved paths remain inside ComfyUI; external folders and symlinks to them need
this independent hash check, recorded separately from setup's receipt.
A missing file can be downloaded to a new temporary file, checked, and
then placed at its destination. A mismatch requires investigation, not an
overwrite of the existing model or a new expected hash. Do not use the artifact
tool's `--capture` option to validate a download: it records whatever bytes are
present as the expected result. The three expected entries are also maintained
in [artifacts.toml](../example_workflows/artifacts.toml).

### Workflow settings

Open the Chroma template and change widget values in your own saved copy:

| Node | Setting |
|---|---|
| DGX Monarch Init | `mode=cluster`, `topology=cfg2`, `attention=TORCH_FLASH`; set `config_path` to the absolute path of the approved two-Spark config |
| DGX Monarch Init, advanced | Keep `auto_gate=first_use`, `compile_dit=false`, `comfy_managed=off`, and `pipeline_depth=1`; set `lora_low_rss=off` and `slab_weights=off` for stock residency |
| Diffusion, text encoder, VAE loaders | Select the three exact filenames above; text encoder type stays `chroma` |
| Positive and negative text encodes | Keep both prompts supplied by the template |
| Empty SD3 Latent Image | Width 1024, height 1024, batch size 1 |
| DGX Monarch KSampler | Seed 0, seed control `fixed`, 26 steps, CFG 3.5, `euler`, `beta`, denoise 1.0 |
| Save Image | Use a recognizable output prefix, such as `dgx_monarch_first_render` |

Keep LoRAs and FSDP out of this first render, and leave native latent RDMA off.
Explicit `cluster` mode refuses a missing cluster configuration instead of
falling back to local execution. Explicit `cfg2` selects the two-rank path tested
with this graph's positive and negative prompts. Do not substitute Schnell's
four-step schedule: this is a different checkpoint.

Queue once after the approved service startup and successful Doctor checks.
Allow the first-use gate to finish; do not disable it to avoid a wait. Save the
output PNG, workflow, prompt ID, and relevant logs. Queue the same saved workflow
with a different seed to check that a second render succeeds without reusing
a cached image.

For this stock, LoRA-free recipe, the first-use gate can report `INCONCLUSIVE`
with `inconclusive_kind=no_material`: there is no optimized residency or LoRA
path to compare. This specific result is expected, leaves the render on stock
residency, and does not prevent installation acceptance when the image and
both-Spark evidence below are complete. Record the result as reported, not as
`PASS`. Other inconclusive reasons, errors, or failures need investigation;
do not disable the gate or enable optimizations to remove the message.
[Troubleshooting #66](TROUBLESHOOTING.md#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)
explains the distinction and reuse of this result.

### Submit the saved workflow through the API

An agent can use ComfyUI's API without building a different graph. Put only the
saved first-render UI JSON in a dedicated directory. Create a private conversion
TOML with the actual driver address and paths; choose a new output directory so
conversion cannot overwrite earlier evidence:

```toml
driver = "http://127.0.0.1:8188"
comfy_dir = "/absolute/path/to/ComfyUI"
out_dir = "/absolute/path/to/private-acceptance/conversion"
template_dirs = ["/absolute/path/to/private-acceptance/ui-workflows"]
```

From the selected repository checkout, run the existing converter:

```bash
"$COMFY_PYTHON" -m benchmark.sweep.convert \
  --config /absolute/path/to/private-acceptance/conversion.toml --source driver
```

It reads widget definitions from that driver's `/object_info` and writes
`<out_dir>/graphs/<saved-workflow-name>.json`. Inspect that graph against the
selected workflow's settings, including the step count (26 for the Chroma
recipe), seed and explicit cluster config. Notes and
the frontend seed-control widget are omitted; the numerical seed remains.
Use only the converter here, not the sweep runner, which can shorten schedules
for probes.

Submit the converted graph as `prompt` in `POST /prompt`, with a unique
`client_id`. Include the saved UI JSON as `extra_data.extra_pnginfo.workflow`
to retain it in a PNG output. For other output formats, save the workflow
separately and retain any metadata the save node supports. Save the exact request and returned `prompt_id`,
then collect the evidence below. Preserve validation errors instead of silently
altering the graph or retrying it.

### Collect evidence from the same render

These checks apply to the selected workflow. The values in parentheses are
for the optional Chroma recipe.

Use one queued job at a time during acceptance, with no other driver using the
pair. Keep the foreground ComfyUI log. Before queuing, record the time and its
current log position; after completion, save the end time and the new log
segment. This matters because individual rank sampling messages do not carry
the ComfyUI prompt ID.

1. Save the submitted graph and returned `prompt_id`. Read ComfyUI's
   `GET /history/<prompt_id>` on the same approved driver address. Save the
   entry showing `status.completed=true`, `status.status_str=success`, its
   execution messages, and the saved output entry. Check that execution did
   not merely reuse a cached sampler or output. Use the output entry's filename,
   subfolder and type to identify the file. Open or play it, check the expected
   dimensions and duration where applicable (a 1024 by 1024 PNG for Chroma),
   and record its SHA256. Retain the original workflow and history separately.
2. In the captured log interval, find the selected render topology
   (`render topology: cfg2` for Chroma) and both
   `rank 0/<host>: sample ...` and `rank 1/<host>: sample ...` messages.
   They must name the two intended, distinct Spark hosts. Save the full interval,
   not just selected lines, so startup or another job cannot be mistaken for
   this render. The one-job interval and matching history timestamps provide
   the correlation; a rank message alone does not identify a prompt.
3. Before tearing down the mesh, save `GET /dgxm/telemetry` from that driver.
   Its `workers` rows should identify `host`, `rank`, `world`, `topology`, and
   `source_manifest_sha256`. Require ranks 0 and 1 on the intended hosts,
   `world=2` and matching source manifests. In each `topology` object, require
   the selected model's documented layout (for Chroma: `cfg=2`, `ulysses=1`,
   `ring=1`, `dp=1`, and `fsdp=false`). Before the first
   Init, `workers` can be empty. During sampling, status queries may time out
   or return cached data; obtain a successful post-render snapshot and retain
   any error or stale-data markers rather than treating them as current proof.

ComfyUI may add bookkeeping fields such as `is_changed` to the prompt stored
in PNG metadata. Preserve that metadata and compare the actual node types,
inputs and settings with the submitted graph. Account for each added field;
do not require whole-object equality or discard unexplained differences.

Telemetry establishes which source and ranks are attached. The sampling log
and successful saved output establish that they performed this render.
Idle Worker readiness, two live services, or a telemetry snapshot alone cannot
replace that evidence. Keep addresses, host identities and raw logs private.
For the second render, change the seed and collect a new prompt ID, log
interval, history and output; do not count the first output twice.

### Acceptance record

A new installation has passed only when its own record contains:

- The DGX Monarch commit, ComfyUI commit, Python/Torch/CUDA/Monarch/xFuser versions,
  and matching model hashes from both hosts.
- A completed render and a saved output that opens or plays correctly, with its
  exact workflow and settings.
- Render-time evidence naming both Spark hosts and their participating ranks.
  Idle Worker services, imports, and a green Doctor report alone do not show
  distributed rendering.
- A second successful render and a setup rerun that preserves the working
  installation without replacing environments, configuration, or model files.
- The reused components, approvals or manual steps, failures and fixes, and
  confirmation that the agreed retained or restored setup is healthy.

The Chroma recipe passed installation, two distributed renders and a read-only setup
rerun with newly installed Python environments, ComfyUI and Monarch on existing
Sparks. The [fresh-agent installation record](VALIDATION.md#fresh-agent-installation)
gives the tested versions, reused components and final restoration status.
These are installation checks, separate from model-fidelity comparisons; they
do not establish a clean-machine installation.

## 5. Account for first-use identity gates

The first render of a model, residency and LoRA combination runs an identity
check if no current Gate result applies. Expect a wait of several minutes
before an image appears. Later renders of the same combination reuse the
stored result and skip that check. A final status other than `PASS`
does not stop your render: the driver turns unverified memory optimizations (slab
weights, low-RSS LoRAs) off and renders on the stock path. Under FSDP, which
has no stock fallback, it refuses instead. [TROUBLESHOOTING.md
#61](TROUBLESHOOTING.md#61-the-first-render-took-minutes-and-the-next-one-took-seconds)
documents latency, notifications, and remediation.

## Troubleshooting

* Run `dgxm doctor`. Include its failing check and remediation in issue reports.
* If Nodes 2.0 shows an older template without input controls, reopen its updated
  template. [Entry 105](TROUBLESHOOTING.md#105-nodes-20-shows-nodes-and-links-but-no-input-widgets)
  explains how to repair a saved copy while preserving your edits.
* [TROUBLESHOOTING.md](TROUBLESHOOTING.md) is indexed by symptom rather than by
  subsystem. Search the error text.
* [FAQ.md](FAQ.md) has an "Install and first hour" section for the questions
  that are not failures.
* The **DGX Monarch** sidebar tab shows per-box memory, live render progress
  and recent gate verdicts without leaving the browser. `dgxm top` is the same
  view for the whole rig, in a terminal ([TUI.md](TUI.md)).
* If the issue persists, open an issue with the
  [bug-report form](../.github/ISSUE_TEMPLATE/bug-report.yml). Include topology,
  model, quant, version, the relevant log window, and ruled-out troubleshooting
  entries.

---
See [VALIDATION.md](VALIDATION.md) for setup test results and remaining
limits. Successful setup does not establish model accuracy.
