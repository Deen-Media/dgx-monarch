# The sweep harness

The harness under `benchmark/sweep/` checks runtime behavior on hardware. It
derives a cell for each combination allowed by the templates and sweep config,
predicts the runtime's response, runs selected cells on a live pair, and compares
the results with those predictions. Use it to validate family additions,
ComfyUI upgrades and adapter changes.

## 1. Cells and expected results

A **cell** is one template crossed with one checkpoint quant, one topology
preset, one attention kernel, one memory or performance setting and one LoRA choice. The matrix
builds these combinations from the templates on disk, the checkpoint files in
ComfyUI's `models/diffusion_models`, and the sweep config. Add a template to the generator and the next matrix run derives its cells
automatically.

Every cell carries a **label** predicting the runtime's response.
Where a guard can be called, the label comes from calling it and reading the
class out of the message it raises, so the harness does not restate that rule.
The labels are `render`, `refuse:P`, `refuse:K`, `refuse:untyped`, and the
three skips: `skip:no-artifact`, `skip:unknown-family`, `skip:upstream-gated`.
The refusal classes are the ones in [DESIGN.md](DESIGN.md) section 5.9. Each
cell also records the **basis** for its label. A **leg** is one execution of
that cell, such as a cold load, warm render, or short comparison probe:

| basis | meaning |
|---|---|
| `disk` | the checkpoint is not on disk, or its header names no family |
| `name-token` | a block-scaled layout, whose kind lives in the loaded model rather than the header |
| `header` | a rule read from the checkpoint and LoRA headers (the fp32-weight LoRA rule below) |
| `config` | a template the sweep config gates |
| `token-count` | a rule that reads a prompt's encoded length |
| `token-count-unavailable` | the same rule when no tokenizer file can count the prompt |
| `per-cond-dispatch` | a cfg-parallel cell with two differing prompts on a family that publishes the real caption length as a constant, so the two can never fold into one model call |
| `cond-fold-unknown` | the same cell on a family with no such constant, so the matrix cannot say which cfg-parallel path it takes |
| `open-defect` | a known runtime defect the matrix carries until its fix lands; its cells read FINDING until then, so the defect stays visible. No current rule uses it |
| `guard` | every other cell |

The basis records the prediction's source; it does not affect the verdict.

Class C never appears as a label. A capacity check compares the required bytes with available
memory, which the matrix cannot predict, so a cell that may meet one is flagged
`capacity_risk`, and on it a class C answer or a message from either memory
check in `CAPACITY_WALLS` (`run.py`) scores PASS-capacity. A local cell's flag
also prices its resident CLIP/text-encoder loader beside its checkpoint: local mode runs the driver and the worker on one box, so the
loader's bytes are already spent when the stock load prices itself, while a
cluster cell puts that loader on the driver host. The 33.0 GiB Flux2 fp8mixed file has a measured capacity limit. Its two-host
cluster cells carry `capacity_risk` when `auto_gate=first_use` on auto, cfg2,
uly2, ring2 and `uly2+fsdp`. Tests also reached this limit on six uly2 and ring2
LoRA cells.

A LoRA cell on an FSDP preset is labelled by the checkpoint's own admission
property (`adapters/detect.fsdp_lora_admission_property`), not by the coarse
quant and `lora_low_rss` checks alone: a comfy-kitchen quantized checkpoint
(scaled fp8, fp8mixed, int8, with or without convrot) never bakes a LoRA stack
under FSDP, whatever `lora_low_rss` is set to.

The matrix distinguishes these FSDP refusal rules:

- **Block-scaled FSDP is class P.** An mxfp8, nvfp4 or nvfp4_mixed
  checkpoint on an FSDP preset reads `refuse:P`, basis `name-token`. The
  runtime refuses each with a class P card: a uniform file at the FSDP launch
  contract, and a mixed file at the shard-layout check.
- **The fp32-weight LoRA refusal under FSDP is class P.** A
  LoRA cell on an FSDP preset whose stack patches an fp32-stored base weight
  reads `refuse:P`, basis `header`. The worker's in-bake dtype check refuses
  it, and the matrix counts the patched keys from the checkpoint and LoRA
  headers.
- **That prediction covers only families tested for this cast.** Whether
  comfy casts a stored fp32 weight to BF16 at load is family code the header
  cannot show, so the rule applies only to `FSDP_LORA_LIVE_CAST_FAMILIES`,
  which contains Krea2. It reads after the low-RSS guard, so
  a cell with `lora_low_rss` off records that guard first.

**How a render is scored.** A two-rank cell is compared against the
single-GPU render of the same graph and seed, at a one-step probe. Full
trajectories can amplify numerical differences, so fidelity is scored on the
probe leg and full renders supply elapsed times. An FSDP cell is compared instead
against its resident twin, the same preset without the shards, and must match
exactly. A lever cell on a preset without FSDP is compared instead against the
auto cell of its row (same template, checkpoints, kernel and LoRA choice): an
exact match is required when its settings change only weight placement or
work scheduling, and otherwise at the probe, or at the full-render bar where it
has no probe leg. Section 5 gives the bars and what a cell records.

**CI and hardware validation.** The sweep needs an available pair, running
worker loops and a driver serving one transport; a full pass takes weeks.
CPU tests check logic in minutes. Both are required because CPU results do
not establish hardware behavior, and hardware runs do not replace CI.

## 2. The family acceptance gate

A new family PR is complete when the derived cells
of its template have run through the sweep on the pair, at every preset and
quant the matrix derives for it, scored against the single-GPU reference, with
the record attached to the PR: the cell ids, the verdicts and the cold and
warm render times. That is a scoped sweep of one template and costs about an
hour of pair time.

From the repository root, with the sweep config at `<cfg>`:

```bash
# 1. Convert the templates through the live driver, then derive the cells.
python -m benchmark.sweep.convert --config <cfg> --source driver
python -m benchmark.sweep.matrix  --config <cfg>

# 2. References first, on a driver started for the local transport.
python -m benchmark.sweep.run --config <cfg> --session local --cells <template>

# 3. Stop that driver after the run exits. Start a cluster driver.
python -m benchmark.sweep.run --config <cfg> --session cluster --cells <template>

# 4. The sanitized summary to paste on the PR.
python -m benchmark.sweep.report --config <cfg> --session cluster
```

`python -m benchmark.sweep.matrix --config <cfg> --convert` runs step 1 in the
same call, but falls back to the repository's widget map when the driver does
not answer. A template token in `--cells` keeps the template's waivable cells.

Every PR must pass the CPU and doctrine tests. Family PRs must also pass this
hardware gate.

A family added without sweep cells is a coverage gap to close, not an
exception to the rule. Mage Flow, Qwen
Image 2.1 and Wan Animate 2 were tested with dedicated hardware scripts, and no
matrix-derived sweep cell of the three has run. Their support in MODELS.md is limited to
those tests, recorded in VALIDATION.md. Their templates still need scoped
sweeps.

## 3. Adding a template

1. Add the spec to `tools/gen_templates.py` and regenerate. A shipped template
   goes in `example_workflows/`; a graph used only for a test goes in
   `tests/fixtures/workflows/generated/`. `python tools/gen_templates.py --check`
   is what CI runs.
2. Convert, then derive. The order matters: the matrix without `--convert`
   reads whatever graphs are on disk, so a template edited without a
   re-convert derives cells from the old graph.
3. Read the derived cells for the new template in `cells.jsonl` and the counts
   table in `cells_counts.md`. Every cut the derivation made is printed in the
   pruned block, so a template that derived nothing says why.
4. Check the labels against what the runtime would answer. A label that
   disagrees is a matrix gap, not a rig fault, and the fix is a label rule in
   `matrix.py` that calls the guard rather than restating it. Fix it in its own
   PR before the cells run.

**The probe rewrite.** A cell gets a probe leg only when the harness can cut
its schedule to one step. One table decides it: `STEP_WIDGETS` in `convert.py`
names the widget each node type keeps its step count in. The same table
decides which graphs get a probe leg and what the leg rewrites, so the matrix
and the runner cannot disagree. A node type that holds its count under a
widget name the table does not know is invisible: its graph reads as
probe-capable and renders its full schedule under a probe leg. Add each new
type to the table when introducing it. A known step widget on a type the table lacks, or
a count that arrives on a link, is caught: it drops the probe for the whole
graph and names the node type. Tests pin that every shipped and testing-only
template carries a step count the rewrite can set, so a template that loses its
probe leg fails on CPU before it reaches the rig.

A staged graph takes the probe count in every stage, so a two-stage probe is
two sampler steps. The step-nrms floor is a one-step number. Re-run the null
A/B (section 4) on a staged template before reading a staged probe against it.

## 4. Running a scoped sweep

**The config.** Copy `benchmark/sweep/sweep.example.toml` somewhere rig-local
and edit it. It names the driver, the second box, the ComfyUI and output
directories, the world, the template directories, the presets, kernels and
levers to cross, the memory floor, the timeouts, the LoRA files per template,
the artifact groups for checkpoints the stem matcher misses, and the probe
steps and floor. Nothing in your copy is committed: real host names, real LoRA
names and real paths live there and only there.

**Sessions.** One driver process serves one transport, so `--session local` and
`--session cluster` are separate runs against separate drivers. The local
session renders the single-GPU references the cluster cells are scored
against, so it runs first. The runner opens a mesh at the session's transport
before it scores anything, and ends the session rather than filing a findings
list if the driver was started for the other one.

**Selecting cells.** `--cells` takes cell ids, labels or template names, comma
separated. A cell id or template token keeps every cell it names. Selecting a label skips cells that can render only after a waiver, unless
`--include-waivers` is also set. `--limit N` runs only the first N selected
cells; without it, every selected image cell runs and only the first video
cell. A stale reference that a selected cell reads runs as well, on top of
either count ("Isolating a crash-prone cell" below). `run_pruned_<session>.json`
names the cells left out. `--cooldown-s` and `--timeout-s` override the config
for one run.

**Isolating a crash-prone cell.** A named run pulls in, ahead of each
dependent, any reference whose record would not resume ("What re-runs" below),
whatever it last answered. A caller that runs an OOM-prone cell alone in each
session, to contain a wedge, then meets the reference's crash again in every
dependent's session. `--skip-crashed-references` leaves such a reference alone
when its current record is a crash for this exact question (the same pin,
label, capacity-risk basis, reference plan and converted graph a resume would
ask for), the crash type is not in `RECOVERABLE_CRASHES` (`run.py`: a dirty
mesh an earlier cell left), and the record is not a wedge. Any other record
reads as an ordinary stale reference and is pulled in again. A dependent is
recorded blocked, naming the crashed reference, only when its label
can still reach a fidelity compare: `render`, or a refusal with a waiver guard
that can render on its waived leg. Any other dependent never reads the
reference, so it runs normally and nothing is pulled for it. Blocking carries
down the reference chain: a cell two or more links from the crash is blocked
too.

`--timeout-s`, or the config's image or video timeout, bounds each leg from
submission through the last history poll. Each HTTP request gets the time left
in that budget, so a long LoRA bake is not cut off by a fixed poll limit. A
poll failure after the driver returned a prompt ID is recorded and drained. If
submission loses its response, the runner stops the session: the driver may
have queued the prompt, but no ID is available to settle it. Check the
driver's queue and settle that work before restarting the sweep.

**Launch it detached with a log.** A wave runs longer than a login shell:

```bash
setsid nohup python -m benchmark.sweep.run --config <cfg> --session cluster \
    --cells <template> > sweep-cluster.log 2>&1 &
```

Stop the driver only after the run exits. The runner recycles the fleet at
session close, because a driver killed with a live mesh strands its actors and
the next session's first render waits in a collective until the stall guard
evicts it.

**What re-runs.** A resume keeps a record when all of these match: the ComfyUI
commit, the digest of the runtime source under `src/dgx_monarch`, torch, NCCL
and the world; the label and capacity-risk basis the cell was scored against;
the reference spec (its kind, the cell it names, the bar and the probe steps);
and the converted graph's own digest. The record must also have settled: the
cell rendered or refused. A timeout, a crash or a transport error is the rig
saying it was unavailable, not the cell answering, so those re-run. So does a
record a wedged driver wrote, whatever class it carries. The resume reads the
runner's own wedge signals (section 5) back off the record, so the session that
ends on one and the record that carries one cannot disagree. The repository
commit is not part of that identity, because a harness-only commit would then
re-run a whole wave. The NCCL stamp identifies the library loaded by the harness process. Older
records without that stamp rerun on resume and count as another build for
reference comparisons. Start a new out
directory for a new wave.

**Planning a full sweep.** About three weeks of pair time, video roughly two
thirds of it in the original planning estimate. It runs in waves: the contract
cells first, since they cost seconds each and prove the refusal table both ways
before any render is trusted; then a null comparison of one cell against itself
across two driver sessions, which must produce identical output before relying on later
measurements; then the local references; then image renders; then video;
then the levers and LoRA; then the waived kernels.

The recorded stable-source campaign selected 2,632 of 7,192 matrix cases
using fixed ComfyUI, torch and NCCL versions. It covered local references,
image renders on Torch Flash, cuDNN and SAGE_AUTO, LoRA image cells, and a
26-cell video sample. [VALIDATION.md](VALIDATION.md) records the results and
4,560 omitted cells, including every waiver and Sol cell.

## 5. Reading results

**Verdicts.**

| verdict | what it means |
|---|---|
| PASS | the observed class matched the label, and any bar the cell carries held |
| PASS-capacity | a memory-capacity check refused the run first on a cell flagged `capacity_risk`; that is not a fault of the cell |
| CHECK | a measured difference above its limit, a full-render difference, or a measurement the harness could not prove, on a cell that otherwise passed |
| FINDING | an expected refusal rendered, an expected render refused, a leg crashed or timed out, a warm leg, probe leg or waived leg did not render, a leg after the cold one rendered and saved no frame, or a `bit-identical` comparison differed |

A typed refusal the matrix predicted is a PASS. No PASS survives a finding
filed against the same cell: the findings list is the defect list. Two kinds
of line report something unproven rather than a fault, and those read CHECK: a
reference compare the harness could not make (no reference record, a reference
rendered on another build, missing or unequal frames, an audio envelope it
could not correlate) and a warm leg the execution cache may have served. An
FSDP cell whose resident twin does not render keeps its verdict, with the
reason in its notes and no fidelity number.

Every cell carries `cfg_path`, the cfg-parallel path its renders were expected
to take: `slice` cuts one batched cond and uncond call across the ranks,
`dispatch` gives each rank a whole conditioning where the two do not
concatenate, `either` is a cell the matrix cannot call, and `none` is every cell
whose topology resolves to no cfg split. The runtime decides per step, by
comfy's own concat rule over the cond list comfy built. The matrix holds two
facts: whether the prompt texts differ, and whether the family publishes the
caption length as a constant. Two prompts of one text fold on any family. Two
that differ dispatch on a family with the constant; without it they may still
fold, so the cell reads `either`. The record carries the expected path beside the render time used to detect
an ineffective split.

One more finding reads CHECK. A cfg cell whose warm leg takes at least nine
tenths of its single reference's warm render time did not split: two ranks that each
render the whole cond and uncond batch take as long as one GPU. The pixels are
right either way, so only the timing can show it. The line reads
"cfg2 did not split" and names the reference cell it was read against. A cell
with no single reference, an FSDP cell among them, makes no claim: its
reference is the same render at another residency, which prices nothing about
one rank's work.

**Bars.**

| bar | what it compares | floor |
|---|---|---|
| `step-nrms` | this cell's one-step probe against the reference's own | 0.10, over it reads CHECK |
| `bit-identical` | an FSDP cell against its resident twin, or a residency or inert lever against its auto cell, on the full warm render | any difference is a FINDING |
| `waived-nrms` | a render the runtime already knows to be wrong, against the reference: a kernel the matrix names as approximate, or any leg that rendered under a granted class K card, whether this cell granted it or an earlier cell of the same shape left it live | none, the number is the deliverable |
| `full-render-nrms` | two full warm renders, where no probe leg is available | none; the number is reported, and the cell reads CHECK unless the two renders match to the pixel |

The `step-nrms` limit was calibrated on one step and cannot be applied to a
full render. Comparisons without a limit report the measured difference;
waived renders are already expected to differ. The report lists each floorless
bar under its own heading, apart from the findings, so its number is never
read as a fault. A cell whose batch size was toggled is compared against
nothing at all: no other cell renders the same frame count.

A cell labelled `render` can meet a class K guard only the driver sees, because
it reads token counts; the runner grants the card and queues the graph again.
That render moves to the `waived-nrms` bar only when the record holds a granted
consent, a waived leg that rendered, and a class K waiver `use` row in the gate
ledger naming a dispatch ID the leg's own sampler published in its ComfyUI
prompt history. A waiver stamp the input carried from an earlier render does
not count. A cold leg that rendered under a grant an earlier cell left live
moves on that `use` row alone. A cell labelled anything but `render` or
`refuse:K`, or lacking that evidence, keeps its ordinary bar; a
`bit-identical` bar never moves.

**What a step-nrms number contains above cfg 1.0.** The probe scores the latent
one step produces, which is the cfg combination of the cond legs, so the number
carries each leg at its cfg weight. At cfg 1.0 the combination is the positive
leg exactly, and the negative leg is not scored at all. At cfg 3.5 both enter,
weighted 3.5 and 2.5. That matters when comfy does not fold the two
conditionings into one model call: two calls round independently, and the
combine scales both roundings. Comfy's concat rule decides the fold, and it can
fold a cross-attention pair of unequal length too; see
[VALIDATION.md](VALIDATION.md). So one template's probes at
two cfg values are two different measurements, and a template whose prompts do
not fold is read against its own cfg gain rather than against a folding
template's number. Earlier Chroma tests illustrate the limit of this explanation:
the shipped BF16 uly2 template measured 0.150, while equal prompts measured
0.031 on the same build and kernels. Follow-up tests in
`benchmark/reports/chroma_cfg_amplification_matrix.toml` measured small
differences, so independent rounding and the short negative prompt did not
explain the original gap. After the text projection fix, pure Ulysses BF16
matched one GPU exactly. [VALIDATION.md](VALIDATION.md) records those tests.
Compare probe results only with their prompt and CFG settings in mind.

**The record.** One JSON per cell under `cells/<id>.json` in the output
directory, named by the twelve-hex cell id. It carries the cell as derived, the
pin it ran on, every leg with its outcome, duration, message tail, frame count and
memory summary, the fidelity block, the gate ledger rows the cell wrote, the
worker journal lines from each box of its session, the findings and the notes,
and the verdict. The legs are `cold`, `warm` at a moved seed, `resident` for an
FSDP cell's twin, `probe` where the cell has one, and `waived`, the same graph
queued again after a waivable guard fired and the runner asked to grant its
card. An attempt that did not settle is kept beside it as
`<id>.attempt<N>.json`, because the crash a fresh fleet answers is how the
fleet is known to have been dirty. The eighteen /proc/meminfo fields, sampled
at 1 Hz on both boxes across every leg, live under `mem/<id>_<leg>.jsonl`.

A cell `--skip-crashed-references` left without a reference carries no legs:
its record reads `observed: "blocked"`, verdict FINDING, and one finding
naming the crashed reference it was not run against. The runner never writes
it over a record that already resumes.

**The report.** `python -m benchmark.sweep.report --config <cfg> --session
<session>` writes `summary_<session>.md` and `.json` beside the records. One
row per cell with the cell id, template, preset, kernel, label, observed
answer, per-leg outcomes, cold and warm times, the fidelity numbers, the frame count and
the verdict, then the findings, then the floorless measurements under their own
headings. The report is redacted; generation fails if a private value remains.
The per-cell records are not redacted: they carry the driver address and are
keyed by host name, so they stay rig-local.

**Comparing two output directories.** Cell ids are a digest of the cell's own
definition, so the same cell has the same id in every output directory, and
that is the join key. Report both sessions and compare the `cold` and `warm`
columns row by row for speed. For memory, compare the per-leg summaries in the
two records, or read the raw samples with `python -m
benchmark.sweep.memsample --summary mem/<id>_<leg>.jsonl`, which prints the
minimum, maximum and last of available memory, shared memory, anonymous pages
and file pages per box. Two directories are only comparable when their pins
agree, which is why every record carries one.

**Wedge signals.** A wedged driver answers every later cell in about two
seconds, so a runner that kept going would file findings against cells that
never ran. Two signals say the rig, not the cell, is the answer: a recycle the
driver refuses twice in a row (`http 503`), and a driver message naming a fleet
no attach can replace or a transport a partial bring-up poisoned. These end the
session too: a clear graph or leg whose queue will not drain, a prompt
submission whose response is lost, a recycle still rate limited after three
tries, and a transport probe that cannot open the session's mesh. The runner
writes the reason into `run_summary_<session>.json` beside the records and
exits non-zero, so the shell around it can restart the driver and the worker
loops. Restarting both is what clears it.

## 6. Rig rules

**Change runtime code only between waves, with the driver stopped.** Between waves:
pull on both boxes, `dgxm restart`, run the doctor, relaunch the driver, then
regenerate the matrix. A resume re-runs every settled cell whose label or
reference moved, so a deployed label rule costs re-runs, never stale verdicts.
Never edit the checkout a live runner is reading. The runner stamps the digest
of the runtime source once, at session open, and every replacement fleet
imports the runtime fresh off the disk, so an edit mid-wave runs in the next
fleet while the records still name the old digest. The pin then names a
runtime that did not answer.

**The memory floor.** Before it queues anything, every cell waits within a
600-second window for available memory on every box of its own session to reach
the configured floor, and prints what it is waiting for. A floor that has not
moved for twelve polls (about three minutes) has always been a worker fleet
holding the previous checkpoint, never a box still releasing memory. The runner
then replaces the fleet once and reads again for what is left of the window, at
least 60 seconds, so a wait with a replacement can pass 600 seconds. A cell
still runs when the window ends under the floor; `memory_floor` in its record
says so.

**Fleet recycles.** The runner replaces the fleet at session open, on every
template change, whenever the comfy-managed lever moves, after any cell that
did not render, and at session close. A cold leg that crashes with a type in
`RECOVERABLE_CRASHES` gets one retry on a new fleet. If that recycle does not
answer `ok`, the runner skips the retry and records why. Only a rendered cell
can leave its fleet standing: a refusal that arrives after the weights are on
the box leaves the fleet holding them, and the next cell then answers on the
residue instead of on its own guard. After a render on the same template the
residue reaches the next cell only through memory, because the guard prices the
checkpoint against MemAvailable. So the runner reads the box that cell handed
on, the lowest MemAvailable any host read on its last sampled leg, and keeps
the fleet only if that clears both the memory floor and this cell's own price:
what the runtime charges a stock load of the checkpoint
(`capacity_fit.stock_required_bytes`). The harness cannot know which residency
the load will take, so it charges stock, the dearest price it can compute; a
comfy-managed load under pinned staging costs more, but pinned staging is
driver state the harness cannot read. Under the floor, the replacement is the
one the floor wait would make about three minutes later. Under a price above
the floor, nothing else would catch it: the cell would clear the floor wait and
then refuse on the residue. The record states the reason, and a cell that kept
the fleet carries a `fleet_kept` stamp with the reading and the bar instead.
Read that stamp before interpreting the cell's cold render time: a kept fleet already holds the
model, so that leg is a first render, not a first load. A kept fleet also
carries every worker lever but the comfy-managed latch into the next cell; the
runtime makes that safe, because applying the new policy drops each resident
the new policy cannot serve.

**Use a fresh seed for every manual test.** ComfyUI serves an identical
prompt from its execution cache, so a leg re-queued unchanged measures the
cache. The runner moves the seed on the warm leg and flags a warm leg whose
seeded node the prompt history says the cache served. A record without that
signal falls back to the duration: a warm leg under a fifth of its resident leg, or
of its cold leg where it has none. Cold is the wrong baseline for an FSDP cell,
whose cold leg pays the first-use clean-reload check and whose warm leg
never does. Recycle the fleet before a manual test too, so the previous load does not affect the result.

**Video tests.** Run one cell per invocation, starting cold on an idle host.
Raise the memory floor for long sequences. This pacing does not enforce a
temperature limit.

**Capacity legs.** These run by hand between waves rather than as cells.
`python -m benchmark.sweep.memhog --target-gib N` holds this box's available
memory at a figure while one checkpoint loads, allocating and touching pages
until the kernel's own reading lands within half a GiB of the target, and frees
everything on a signal. A target under 8 GiB is refused, because a box held
that low stops answering and the leg could not be ended by hand.
`python -m benchmark.sweep.memsample` writes the runner's own record shape for
a leg it did not queue, so one reader answers for a swept cell and a
hand-driven leg alike.

## 7. Sharing results

Records stay in the configured output directory on the rig. They are private,
unsanitized and uncommitted. Share cell ids and the redacted report.

Cite a cell by its twelve-hex id, with the template, preset and kernel from the
report row, the label, the observed answer and the verdict. That is enough for
anyone with the output directory to open the record, and enough for a reader
without it to know what was run. Paste the report row rather than the record:
the report is redacted, the record is not. If a record has to travel, take the
lines you need through the report first.

Include the affected cell ids when filing an issue from a wave. Validate a fix
by rerunning those cells with `--cells <id>,<id>`.
