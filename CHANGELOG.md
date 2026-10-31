# Changelog

## [Unreleased]

### Everyday use

- Studio is a supported interface from this release, beside the CLI, the TUI
  and the ComfyUI panel. Its docs and the capability contract in Settings no
  longer call it a draft. The
  [acceptance record](docs/STUDIO_ACCEPTANCE.md), the
  [measurement record](docs/STUDIO_PERFORMANCE.md#runtime-acceptance) and the
  [interface QA record](docs/STUDIO_INTERFACE_QA.md#remaining-acceptance) still
  list every open item ([Studio](docs/STUDIO.md)).
- Studio's Workflow needs attention list follows the draft. The companion's
  inspection now checks the draft's edited values, not the saved template, and
  after a value edit Create inspects the draft again once the edits pause. A
  repaired value, such as a model file changed to one the driver has, leaves the
  list and the footer, and a problem that remains stays ([Studio](docs/STUDIO.md#working-with-workflows)).
- Create's footer names the field that keeps Generate off, by the name the
  field shows, and the problem, with its node only when another field shows the
  same name, and a Show field button that opens Inputs or All inputs, scrolls to
  the field and focuses it. With more than one issue, a list of them all, each
  with its own Show field, stays usable while Generate and Review settings are
  off ([Studio](docs/STUDIO.md#working-with-workflows)).
- Under the Random seed policy each seed field reads Random each run and takes
  no edits, with a line that says Fixed sets it, and Fixed makes it editable
  again. The seed under a result and Use this seed are unchanged ([Studio](docs/STUDIO.md#working-with-workflows)).
- Create's Operation select is now labelled Job category, with one line under
  it: it labels the job in Queue, the gallery and its record, and the workflow
  decides what runs, so Image and Workflow run the same graph ([Studio](docs/STUDIO.md#working-with-workflows)).
- Studio's Send to edit puts the output into the image input of a workflow
  that edits images. It keeps the workflow in Create when that one supports
  Edit, loads the one editing workflow the driver runs, lists them when there
  are several, and says why and changes nothing when there is none. The
  companion copies the output into the driver's input store, so the 2 MiB
  upload limit does not apply, and the Operation becomes Edit. Under Edit,
  Generate stays off until the image input holds an image. Before, Send to
  edit reloaded the text to image graph as "Edit (not supported)" with only
  lineage attached and Generate on ([Studio](docs/STUDIO.md#working-with-workflows)).
- Over an image in Studio's output viewer, the mouse wheel zooms around the
  point under the pointer. Anywhere else the wheel still scrolls the page, and
  Fit, 1:1, the zoom buttons and drag to pan work as before, except that a
  zoomed image smaller than the frame now stays in its middle instead of at the
  left edge ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's telemetry panel keeps the jobs, one Readiness line and both hosts in
  view, and folds the readiness rows, metric history, sample inspection, export
  and raw sample into Detailed telemetry, which the browser keeps open or closed
  with the layout ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's video tiles in Recent Outputs, the Gallery and Queue show the
  video's first frame and its length instead of "Preview unavailable". A tile
  with no frame says No thumbnail, and only a video the browser cannot decode
  says Cannot play in this browser ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's Generate checks the draft and starts the job in one action, and the
  job keeps the same immutable submission record. Review settings shows that
  record first. The review dialog opens by itself only for a warning, which
  then has its own box, or for a refused draft, with a button to the Create tab
  that holds the cause. The general confirmation box is gone, the checks fit
  one line, and the expiry line says what expires and why. The companion's
  review no longer reports its dispatch checks as a warning. A brief line under
  Generate replaces the Submitted banner, and the submission record stays
  closed in a short scrolling box in the dialog and in Queue's job details ([Studio](docs/STUDIO.md#working-with-workflows)).
- Studio's sidebar groups Grids, Fleet and Operations under Utilities, one
  page with a tab for each and a line on what each is for. The tabs keep their
  route ids, so Grant and retry still opens Operations, and the command palette
  lists each tab by name ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's telemetry panel shows the followed job's step and node, then every
  reporting host side by side with available memory, GPU memory, GPU
  utilization, power and temperature, then the driver's readiness rows for the
  Worker service, attached mesh and render session. When only the driver's host
  reports, as before the first Monarch render attaches a mesh, it says so, and
  readings kept after telemetry stops are labelled stale with their age. Its
  chart picks a host and a metric separately and shows a choice with one option
  as text instead of a one-item list ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's Grids page explains what a grid compares with a worked example and
  walks a first comparison, 20 steps against 40. The Fleet page says what Fleet
  does, when to use it and how to start. Its button, now Set up Fleet in
  Create, loads the packaged Fleet workflow and names the next step; before,
  Configure Fleet workflow only set the draft's Operation label and opened
  Create with nothing to act on ([Studio](docs/STUDIO.md#run-many-prompts-with-fleet)).
- Studio's artifacts route lists outputs with equal creation times most
  recently stored first, as the snapshot does, so the viewer shows a finished
  job's last output even when the snapshot leaves it out ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Studio's viewer shows an image chosen after a video at once. Before, it could
  report "Selected media cannot be decoded by this browser" until another
  output was chosen ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- When a new generation starts, Studio's viewer stops showing the previous
  output and names the job's workflow and progress until the new output lands.
  Recent Outputs keeps the earlier outputs, and the snapshot's job rows now
  carry the draft's workflow name ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- Left and Right on a Studio Recent Outputs tile select the previous or next
  output and show it in the viewer ([Studio](docs/STUDIO.md#workspace-controls-and-media)).
- The Init node's tooltip and description say `sync_ulysses` has no effect on
  the pinned xfuser 0.7.0; changing it still reruns setup and the identity gate
  ([Adapters](docs/ADAPTERS.md#what-an-adapter-provides)).
- Studio's Operation select lists only the operations the loaded workflow
  supports, read from its input, upscale and save nodes, so a text to image
  workflow no longer offers Edit. A draft that holds an unsupported operation
  says which input it needs, and loading a workflow picks one it supports ([Studio](docs/STUDIO.md#working-with-workflows)).
- Create says what a run saves, such as PNG frames plus separate FLAC audio or
  one MP4 video with sound, read from the graph's save nodes ([Studio](docs/STUDIO.md#working-with-workflows)).
- MiniMax H3 and LTX workflows get one Duration control per clip, in frames
  and seconds, that writes the clip's frame count to every node that holds it.
  Their frame count fields name their node, unit and relation ([Studio](docs/STUDIO.md#working-with-workflows)).
- The MiniMax H3 templates say what they save: "MiniMax H3 text to video, PNG
  frames plus separate FLAC audio", "MiniMax H3 text to video, one MP4 with
  sound" and "MiniMax H3 with a guide anchored mid video, PNG frames plus
  separate FLAC audio". File names and graph ids stay, so saved drafts and
  records keep their inputs.
- Studio's guide says which workflows Studio lists, what an imported workflow
  needs, and what an update does to bundled and saved copies ([Studio](docs/STUDIO.md#workflow-sources-and-updates)).
- Studio's Create pane no longer scrolls sideways at any pane width. Long
  lineage ids, file names and field names wrap, fields narrow to fit, and each
  lineage id has a Copy button that copies the whole id. On a short window the
  field list keeps a usable height and the pane scrolls instead, with Generate
  pinned at its foot ([Studio](docs/STUDIO.md#working-with-workflows)).
- Studio names the prompt boxes Prompt and Negative prompt, from the inputs
  chosen for prompt styles or else the sampler inputs their text reaches, and
  names other text boxes by their node. The prompt box starts taller and grows
  with its text, an empty negative prompt waits behind Add negative prompt on
  Inputs, and raw types such as STRING show only on All inputs ([Studio](docs/STUDIO.md#working-with-workflows)).

### Rendering

- `[worker_args] fp8_matrix_mult` (`"auto"`, `"on"` or `"off"`) chooses whether
  the models the actors build take ComfyUI's FP8 matmul path. ComfyUI's
  `--fast fp8_matrix_mult` reaches only the driver's own loads, because the
  actors start ComfyUI without its argument list: `"on"` gives every actor build
  the loader option the flag stands in for, `"off"` builds a loader's
  `fp8_e4m3fn_fast` as `fp8_e4m3fn`, and `"auto"`, the default, keeps the
  previous behaviour. Every model build in the driver and in each actor logs
  one `FP8 compute:` line naming the operations class ComfyUI picked.
  `dgxm doctor` shows the setting, `dgxm status` prints it when cluster.toml
  sets the key and `dgxm status --json` always carries it, and
  `tools/fp8_paths.py` reads from a checkpoint's header which class it takes
  under each setting.
  The FP8 matmul path's speed and fidelity on GB10 are NOT RUN
  ([Models](docs/MODELS.md#fp8-compute-paths),
  [VALIDATION](docs/VALIDATION.md#fp8-compute-paths),
  [TROUBLESHOOTING #141](docs/TROUBLESHOOTING.md#141-fp8-renders-run-no-faster-than-bf16)).
- The two probes of #93 item 6 ran on the head and changed nothing: comfy-kitchen's
  own NVFP4 GEMM was the fastest path at every MiniMax H3 linear shape, and its
  built-in Sol-Attn kernel ran as fast as the sol_attn package's with about twice
  its error against exact attention, so the lever keeps the package
  ([VALIDATION](docs/VALIDATION.md#item-6-probes-nvfp4-gemm-and-sol-attn-2026-10-31)).
- INT8 ConvRot linears run a Triton GEMM kernel in place of comfy-kitchen's
  CUTLASS kernel, in the ComfyUI process and in every actor, with identical
  output; in the 2026-10-28 and 29 comparison MiniMax H3 and Wan Animate 2
  rendered 16 to 25 percent faster. `[worker_args] int8_gemm = "cutlass"` turns it off in the actors and
  `DGXM_INT8_GEMM=cutlass` in the ComfyUI process. A process without Python.h,
  Triton or a validated comfy-kitchen release (0.2.37 or 0.2.40) runs CUTLASS
  and logs one WARNING ([Models](docs/MODELS.md#int8-gemm-kernel),
  [TROUBLESHOOTING #138](docs/TROUBLESHOOTING.md#138-the-log-shows-an-int8-gemm-warning-and-int8-renders-run-stock-cutlass)).
  The kernel option's install points ran on hardware, one short render per
  process kind ([VALIDATION](docs/VALIDATION.md#int8-gemm-kernel-comparison-2026-10-28-and-29)).
- MiniMax H3 image to video now has speed and memory records at 5, 10 and 15
  seconds (1344x768, 20 steps) on the pair and on one Spark. On the pair a
  15 second clip samples in about 45 minutes with exact attention and 26 with
  Sol-Attn under its waiver, and a session admits it twice. One rank on the
  sibling alone rendered it in 83:31 with 75 GiB free at the lowest, after
  the one-hour sample timeout cut off a first attempt. A single RTX 5090's
  reported 8.6 minutes for the same clip stays beside them
  ([BENCHMARKS](docs/BENCHMARKS.md#minimax-h3-speed-and-memory-by-clip-length-2026-10-29-to-2026-10-30-head-capped)).
  The activation charge is 2.7 to 2.9 times the measured activation on the
  pair and 4.7 to 4.8 times on one rank, with no constant changed, and a BF16
  capacity rescue on the pair ends refused with no card
  ([VALIDATION](docs/VALIDATION.md#h3-capacity)).
- The MiniMax H3 activation preflight charges each rank a fixed 7.5 GiB floor
  plus 634,368 bytes per row on the rank, fitted to four hardware points, in
  place of 1,032,192 bytes per row with no floor. On the pair it now admits
  the 15 second clip's second request in a session and the gate's slab path,
  which the old charge of 60.1 GiB per rank refused against the 17.6 GiB the
  render measured; one Spark still refuses the clip, 11.2 GiB short
  ([VALIDATION](docs/VALIDATION.md#h3-capacity),
  [TROUBLESHOOTING #56](docs/TROUBLESHOOTING.md#56-minimax-h3-render-refused-activation-footprint-preflight)).
- `compile_dit` no longer forces stock residency on the families it leaves
  eager, such as Qwen-Image, Qwen Image 2.1, LTX and Ideogram4; the Init
  tooltip names the families it compiles ([TROUBLESHOOTING #14](docs/TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on)).
- Before a render-site capacity check refuses, the driver now also asks the
  actors to release their allocator caches once ComfyUI's own models are
  unloaded, so a session's second render no longer counts the first render's
  retained activations as spent. The identity gate runs the same checks for
  the whole request before its proof render: a request that cannot fit the
  slab path gets one INCONCLUSIVE capacity row and no proof render, renders on
  stock residency, and the next request checks again ([TROUBLESHOOTING #56](docs/TROUBLESHOOTING.md#56-minimax-h3-render-refused-activation-footprint-preflight)).
- The driver's deadline for a render's sample call grows with the work. For
  MiniMax H3 and Wan SCAIL, SCAIL-2 and Animate 2 it allows 600 s per step at
  109,466 packed rows per rank, scaled linearly with the rows on each rank, at
  least 120 s per step and at most six hours in all. `DGXM_SAMPLE_TIMEOUT`
  stays the floor, 3600 s unless set, and wins over the cap. A 15 second H3
  clip on one Spark, which sampled for 80 minutes and was cut off at the hour,
  now gets 200 minutes; other families keep the larger of the floor and 120 s
  per step. Each render logs its deadline and the terms behind it at INFO
  ([TROUBLESHOOTING #139](docs/TROUBLESHOOTING.md#139-a-long-render-fails-with-timeouterror-when-the-sample-deadline-passes)).
- A capacity-rescue consent now keeps slab residency when stock residency
  cannot load the checkpoint. The identity gate records INCONCLUSIVE with
  `cross_mode=CAPACITY`, takes no lever, and the render proceeds under slab;
  on 2026-10-30 the 61.7 GiB MiniMax H3 bf16 file was refused under stock
  instead. `DGXM_ALLOW_SLAB_RESCUE=1`, `DGXM_AUTO_RESCUE=1` and the Auto-allow
  toggle give the same consent record as the panel click. The panel now lists
  the rescue card a refusal offers after such a verdict or after an update, and
  a Worker's rescue offer becomes that card when the driver holds no rank
  ([TROUBLESHOOTING #5](docs/TROUBLESHOOTING.md#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).
- Worker-assisted VAE decode, off by default: with
  `[worker_args] worker_vae_decode = "auto"` the stock VAE Decode and VAE
  Decode (Tiled) nodes split a video decode over the Workers, the driver
  decoding its own share and each rank on another GPU the rest. ComfyUI's own
  decode loop runs once on the driver with only its per-unit decode swapped,
  so the output is byte-identical to the one-host tiled decode: MiniMax H3
  through both nodes, and the Wan 2.1 and 2.2, HunyuanVideo 1.0 and 1.5 and LTX
  video VAEs through VAE Decode (Tiled). A Worker loads the driver's VAE file
  in the driver's dtype, decodes under the driver's attention and torch
  backend flags, holds one decode VAE that `status()` reports and `clear_vram`
  drops, and declines any unit it cannot decode exactly, which the driver then
  decodes with one WARNING; a reply of any other shape or dtype than the stock
  function's is refused the same way. Replies waiting on the driver stay
  within a 2 GiB budget. Each decode logs whether it split and over how many
  ranks, and `dgxm doctor` shows the last one. The key stays out of the
  Gate's contexts and the trust fingerprint. CPU tests compare bytes against
  real ComfyUI; the hardware compare is NOT RUN
  ([Models](docs/MODELS.md#worker-assisted-vae-decode),
  [VALIDATION](docs/VALIDATION.md#worker-assisted-vae-decode),
  [TROUBLESHOOTING #140](docs/TROUBLESHOOTING.md#140-a-vae-decode-did-not-split-over-the-workers)).

### Operations

- `dgxm doctor` adds an `INT8 GEMM` row for this box and one per configured
  host: the configured kernel and whether the Python headers, Triton and a
  validated comfy-kitchen are there. The rows warn and never fail
  ([TROUBLESHOOTING #138](docs/TROUBLESHOOTING.md#138-the-log-shows-an-int8-gemm-warning-and-int8-renders-run-stock-cutlass)).
- Worker services keep torch inductor's compile cache in
  `~/.cache/dgx-monarch/inductor`, so `compile_dit` no longer repeats its
  autotune after a reboot; set `TORCHINDUCTOR_CACHE_DIR` to place it elsewhere ([TROUBLESHOOTING #14](docs/TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on)).

### Rendering

- Before a render-site capacity check refuses, the driver unloads the models
  ComfyUI holds on its host, such as the last render's text encoder and VAE,
  and checks again; set `DGXM_DISABLE_DRIVER_UNLOAD=1` to turn this off ([TROUBLESHOOTING #56](docs/TROUBLESHOOTING.md#56-minimax-h3-render-refused-activation-footprint-preflight)).

## [1.4.0] - 2026-10-28

**Upgrading from 1.3.0.** Follow [INSTALL](docs/INSTALL.md#upgrading-from-130): the first update runs the 1.3.0
updater, whose process scan can refuse an idle host while a zombie ComfyUI
process lingers; wait for it to go and run the update again. If a host's
Workers run from a release slot, use only `dgxm update --verify`
([TROUBLESHOOTING #118](docs/TROUBLESHOOTING.md#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)). The torchmonarch pin, the required
dependencies and the Gate protocol do not change. SCAIL and SCAIL-2 now refuse
`cfg2`, every cfg composite and ring beyond world 2
([TROUBLESHOOTING #136](docs/TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)), and on ComfyUI 0.43.0 a
MOSS-Image edit with an image strength above 0 and other than 1.0 refuses under
sequence parallelism ([TROUBLESHOOTING #135](docs/TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)).
`dgxm setup --apply` keeps the existing file's `[fabric.<name>]`, `[monarch]`
and `[studio]` tables and your `[worker_args]` keys, so stop adding them back
([CLUSTER](docs/CLUSTER.md#re-running-setup)).

### Everyday use

- Studio history settlement stores a finished job's artifacts before it
  writes the job's terminal state, so a snapshot never shows a succeeded job
  without its outputs. This fixes the rare HTTP smoke test failure in which
  the snapshot route read between those two commits.
- The Studio observer publishes its driver state once per completed cycle,
  so a snapshot never reports the driver connected before the cycle that
  observed it has finished, and a lasting event-watcher fault no longer
  blinks off and on mid-cycle.
- Studio marks a job succeeded only after history settlement has stored its
  outputs, not when ComfyUI's live success event arrives, and the event
  watcher wakes the observer so settlement runs as soon as the driver writes
  its history entry. The observer publishes its telemetry report once per
  cycle, after both of its requests, so a snapshot never pairs one cycle's
  runtime event status with an older telemetry sample.
- Studio's output viewer follows the job this tab submitted, or the newest
  active job, with a status line that shows its state, one node's step count,
  the elapsed time on the companion's clock or the driver's own total, and a
  failure's code and first sentence with a link to Queue. When that job
  succeeds the viewer shows its last final output unless you chose another
  result or opened a comparison, fetching the output when the snapshot window
  omits it, and a reload never shows an earlier result as new. The recent
  outputs strip shows thumbnails of job outputs, job summaries carry the
  driver's start and end times and the companion's first running observation,
  the shared presentation contract gains job-state tokens and an m:ss duration
  rule, and Create counts unknown outcomes until Queue reconciles them.
- Studio's review dialog opens with a plain summary: whether the job is ready
  or what blocks it, the model files, seeds, prompt text, requested topology,
  the checks the backend still runs and the expiry, with the immutable record
  closed below and a reason under a disabled Submit. New drafts default to the
  Random seed policy; the seeds a job ran with show in the review, the viewer
  footer and Queue rows, ride on job summaries as `submission.effective.seeds`,
  and copy alone through the gallery's Copy seed. Use this seed puts an
  output's seed into the draft with the Fixed policy as one undoable change,
  and the dialog closes after every submit so a refusal or an unknown outcome
  shows its banner. Summaries carry seeds above 2**53 - 1 as exact digits,
  imported jobs show their prompt's seeds, and an expired review says so and
  keeps Submit off.
- Studio's Create view lists packaged workflows by their Note titles in title
  order, with a filter box that matches words in the title or file name, a
  Clear button, Enter and Escape keys, and the chosen file name shown under
  the list. The command palette can also load a packaged workflow by title,
  and `GET /api/v1/workflows` gains a `title` field beside the unchanged
  file-stem `name`.
- Studio's Create panel explains how ordered prompt styles combine, with a
  worked example, and lets you move each applied style up or down; Library
  drops its Style mode select, which never changed the result. Grids opens
  with a how-to, starts with empty value fields and says why Validate every
  cell is disabled, and the telemetry panel and event console lose their
  close buttons in favor of the footer toggles and Settings.
- Every bundled template carries a reviewed App Mode layout, so ComfyUI's App
  Mode and the Studio Inputs tab open on the prompt, negative prompt, seed,
  size, sampling settings, input media, LoRA strength and model file instead
  of every input. `tools/gen_templates.py` writes the layouts from one checked
  table, and a new test inspects each template in Studio to confirm its
  inputs exist and follow the approved order.
- Studio reads a failed job for people: the viewer status line and Queue rows
  show the refusal class, the first sentence, the remedy, the TROUBLESHOOTING
  entry and, for a consent request, the driver's reason and action, with
  Grant and retry opening Operations on consent.apply when the companion
  allows management and a plain pointer when it does not. Queue rows show a
  thumbnail of each job's newest final output, fetched as the row scrolls into
  view.
- Studio's viewer status line and Queue rows show a step count only while the
  node that reported it runs. The companion clears the count when ComfyUI
  starts the next node, so a VAE decode after a sampler no longer reads as
  the sampler's last step; the bar sweeps until the running node reports
  steps of its own.
- The status line's Open Queue and Reconcile in Queue links open Queue on
  that job's row, expanded and scrolled into view, turning to the history
  page that holds it. When Queue's first history load leaves the job out,
  Queue reads that one job by its identity and shows it above the history. A
  link applies once, so a later visit to Queue opens on no row.
- The gallery's Copy named inputs, Copy API prompt and Export provenance keep
  every digit of a seed above 2**53 - 1, in the prompt, the frozen seed list
  and the workflow's widget values: Studio reads the job's full record with
  an exact parser instead of the browser's JSON reader, which rounds such
  numbers. Reuse settings, Send to edit, Upscale and Load workflow refuse a
  job whose recorded prompt or workflow holds such a number, name each input
  with its digits and leave the draft unchanged, since a draft field holds
  whole numbers exactly only up to 2**53 - 1.
- Studio's gallery Actions menu opens above its card's buttons on a solid
  background and stays inside the card. Before, it lay over its own Actions
  toggle, so a click meant to close it ran the menu item underneath, such as
  Copy named inputs or Save output membership. On Grids, the Saved
  prompt-style axes list wraps a style's record id, so Use as axis A and Use
  as axis B stay on screen at phone widths instead of pushing the page
  sideways.
- Studio's viewer keeps following this tab's job after later jobs push it out
  of the companion's 200-job snapshot. The tracker then reads the job by its
  identity, once per snapshot while it may still change and once after it has
  ended, so its status line, failure text and Open Queue link stay, and the
  link makes Queue read the job by its identity and pin its row.

### Installation and updates

- `dgxm setup --apply` keeps the existing file's `[fabric.<name>]`, `[monarch]`
  and `[studio]` tables and your `[worker_args]` keys, so stop re-adding them;
  fix any it refuses ([CLUSTER](docs/CLUSTER.md#re-running-setup)).
- `dgxm update`, guided service setup and the Comfy updater's verify step no
  longer refuse an idle host because a subprocess somewhere on the box exited
  during the ComfyUI process scan. The scan skips zombie and exited processes;
  a denied read of a command line still leaves the result unknown.

### Compatibility and verification

- The `moss_image` adapter follows ComfyUI v0.43.0's MOSS-Image forward
  (commit 9fe7ef764), which takes a text attention bias that the Text Encode
  MOSS-Image Edit node attaches when its `image_strength` is above 0 and
  other than 1.0. Under sequence parallelism such a render now refuses class
  P on every Worker before the first block, because the sharded attention
  kernel cannot add the bias; 1.3.0 dropped it and rendered as if the
  strength were 1.0
  ([#135](docs/TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)).
  An all-zero bias still renders, single, cfg2 and dp2 apply the bias in
  ComfyUI's own forward, and text-to-image renders are unaffected. The comfy
  canary's signature contract and seam checks follow the same tree, so the
  daily canary runs green on ComfyUI master again.
- A hardware run on ComfyUI v0.43.0 on 2026-10-27 observed the MOSS-Image
  text attention bias refusal (#135) at the driver before dispatch on uly2 and
  ring2, and the next render on the same fleet matched stock with no Recycle.
  With the bias, renders on single and cfg2 and row 0 of a dp2 batch were
  bit-identical to stock on one GPU at one step and over 50 steps, and
  text-to-image uly2 stayed bit-identical to one GPU at one step and over
  50 steps, and to stock at one step
  ([VALIDATION](docs/VALIDATION.md#moss-image-bias-on-comfyui-426f5eea)).
- BENCHMARKS records a four-Spark owner's published 1.3.0 matrix, reported and
  not reproduced here: uly2+cfg2 ran fastest for CFG models and uly4 for
  MiniMax H3 and LTX ([BENCHMARKS](docs/BENCHMARKS.md#community-four-node-results-2026-10-27)).

### Rendering

- SCAIL and SCAIL-2 refuse cfg-parallel class P on the Worker, as PixelDiT
  does: one-step cfg2 probes of SCAIL-2 on 2026-10-28 read NRMS 0.116 and
  0.085 against one GPU while uly2 and ring2 were byte-identical. Use uly2 or ring2
  ([#136](docs/TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)).
- SCAIL and SCAIL-2 refuse ring attention at any sequence-parallel world
  above two at injection, and the sweep matrix labels both refusals, after a
  four-Spark owner reported ring4 and auto's uly2+ring2 fold outside the
  fidelity limit at world 4; uly4 is the explicit choice there
  ([#136](docs/TROUBLESHOOTING.md#136-a-scail-or-scail-2-render-refuses-cfg2-and-ring-beyond-world-2)).
- Ideogram4 refuses a Ulysses degree that does not divide its 18 attention
  heads after the load and before any forward binds, as FLUX.3 and MOSS-Image
  do, where 1.3.0 went on into the all-to-all and crashed there; the sweep
  matrix labels it
  ([#137](docs/TROUBLESHOOTING.md#137-an-ideogram4-render-on-a-ulysses-degree-that-does-not-divide-18-heads-refuses-after-the-load-and-before-any-forward-binds)).
- The driver refuses a MOSS-Image render whose prompt carries an effective
  text attention bias before dispatch when Ulysses or ring is above one, on
  the same predicate the Worker's sharded forward uses, so no Worker loads
  the model for a render it would refuse. It reads both prompts, so on
  uly2+cfg2 a bias on one prompt refuses the whole render instead of one cfg
  group refusing while the other waits. single, cfg2 and dp2 still run stock's
  forward with the bias ([#135](docs/TROUBLESHOOTING.md#135-a-moss-image-edit-with-an-image-strength-other-than-10-refuses-under-sequence-parallelism)).
- Every Worker rank reads both prompts of a MOSS-Image render for the edit
  node's text attention bias after the load and before sampling, and refuses
  it class P under sequence parallelism, so a bias on one prompt alone refuses
  every rank together on uly2+cfg2 even when the driver cannot read the
  checkpoint header. The driver preflight and the sharded forward behave as
  before and share one scan.
### Development

- The comfy canary runs daily against both ComfyUI master and the newest
  ComfyUI release tag, as two entries of one job. The release entry changes only when
  ComfyUI tags a release, so a red master beside a green release points at an
  upstream change, and a red release on an unchanged tag points at this
  repository.
- The torchmonarch canary uploads its reports with actions/upload-artifact
  v7.0.2, the update Renovate proposed against main, and Renovate now opens
  its pull requests against dev, since main only ever fast-forwards to dev.
- A headless Firefox round on the Studio release candidate re-ran the twelve
  browser rows of the test procedure and added rows for the controls that had
  none: the viewer status line and step bar, the newest result and its strip,
  the review summary and its expiry, seeds, the workflow filter and palette
  group, prompt styles, the Grids how-to, App Mode inputs, failure details
  and the Queue link, Queue thumbnails, exact large seeds and the panel
  toggles. [STUDIO_INTERFACE_QA](docs/STUDIO_INTERFACE_QA.md#browser-evidence-2026-10-28)
  holds the record. `studio/tests/README.md` lists all 26 rows, one to one
  with that record's table, and `studio/tests/event_driver.py`, a CPU fixture
  that sends ComfyUI's driver events, refuses marked prompts and answers with
  the recorded template schema, runs the rows the plain fixture cannot.

## [1.3.0] - 2026-10-23

**Upgrading from 1.2.0.** Follow [INSTALL](docs/INSTALL.md#upgrading-from-120): the first update runs the 1.2.0
updater, so this release's packaged Studio file check starts with the next one.
If a host's Workers run from a release slot, use only `dgxm update --verify`
([TROUBLESHOOTING #118](docs/TROUBLESHOOTING.md#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)). The torchmonarch pin and the required dependencies do
not change. Studio needs the `studio` extra, which neither update mode installs
([INSTALL](docs/INSTALL.md#optional-studio-companion)). Releases before this one refuse a cluster.toml that holds
`[studio]`, so add the table only after the driver and every checkout that
reads the file run this release ([Studio](docs/STUDIO.md#configure-studio-in-clustertoml)). MOSS-Image and Kromoss need ComfyUI 0.42.0 or newer on the
driver and every Worker ([TROUBLESHOOTING #125](docs/TROUBLESHOOTING.md#125-a-moss-image-checkpoint-fails-to-load-on-comfyui-older-than-0420)). A graph that wires Model
Sampling SD3 (DGX Monarch) to a checkpoint that samples at multiplier 1.0 now
refuses where it rendered noise ([TROUBLESHOOTING #128](docs/TROUBLESHOOTING.md#128-model-sampling-sd3-dgx-monarch-refuses-on-a-family-whose-comfyui-sampling-multiplier-is-10)).

### Everyday use

- Add an optional Studio companion, `dgxm-studio`, with bundled workflow
  controls, durable job records, project libraries, reviewed imports, grid
  experiments and job-bound evidence. It is a draft; read the
  [acceptance record](docs/STUDIO_ACCEPTANCE.md) first.
- `dgxm studio` starts Studio with one word. A new `[studio]` table in
  cluster.toml names the driver origin, the port, an optional driver service
  and an optional start command ([CLUSTER](docs/CLUSTER.md#studio-optional)); `--driver` and `--config`
  override it. It passes the config file `dgxm` itself uses, prints where each
  value came from and never probes ports. The first start on a fresh state
  directory needs `[studio] driver` or `--driver`, and Studio reads only a
  cluster.toml no other account can write ([Studio](docs/STUDIO.md#configure-studio-in-clustertoml)).
- Studio opens signed in. On a terminal with a display, `dgxm studio` and
  `dgxm-studio` send the browser to `sign-in.html`, an owner-only page in the
  state directory that carries the link `http://127.0.0.1:<port>/#token=...`,
  when `.html` files and `http` links open in the same browser. The page keeps
  the token in the tab's session storage and removes it from the address bar;
  the browser's history keeps the link, so treat it like a password. The token
  never appears in a process argument dgxm starts, a query string, a log or
  the journal. `--no-open` starts no browser, and a companion started with
  `--allow-management` makes no link; its token is typed into the form.
- `dgxm studio tunnel <host>` prints the one SSH line that reaches Studio from
  another computer,
  `ssh -o ExitOnForwardFailure=yes -L 127.0.0.1:<port>:127.0.0.1:<port> <host>`,
  the address to browse and what to do when ssh cannot forward the port. The
  line names the address the forward listens on, so ssh exits instead of
  opening a session without the forward; `--run` runs it ([#133](docs/TROUBLESHOOTING.md#133-the-sign-in-link-does-not-open-studio)).
- When no driver accepts a connection, Studio shows Driver not running with the
  command that starts it, disables Review generation and connects by itself
  when the driver answers ([#131](docs/TROUBLESHOOTING.md#131-studio-shows-driver-not-running)). The README front page shows Studio,
  captured against the CPU test fixture.
- Studio's Review generation takes empty text in a required input, as ComfyUI
  does. It had blocked 23 of the 71 bundled workflows as shipped, among them
  the MOSS-Image ones, which leave `double_layers` empty, and those with an
  empty negative prompt. A missing value or an unavailable choice still blocks
  review, and an Operations field the companion requires still asks for a value.
- A library import that the companion refuses with `record_window_limit`,
  because the catalog grew past the browser window after the review, clears
  the pending import, so file selection and a new review work again. The page
  had kept the import pending, and each retry resent the refused plan until it
  expired or the catalog shrank.
- `dgxm top` and the ComfyUI sidebar show memory in GiB, draw the memory bar on
  hosts with no slab-resident model, give the six 16-color segments six colors
  and show RX and TX rails for every box ([TUI](docs/TUI.md#what-it-shows)).
- Add five MOSS-Image workflows, for 71 in all.

### Installation and updates

- Verified update, release staging and setup refuse a target whose packaged
  Studio files are missing, extra or differ from their recorded SHA-256 table.
  Studio's own update needs `--config` and `--driver-service` and defaults to
  verified mode. The owner token prints only to a terminal, otherwise to a 0600
  file ([Studio](docs/STUDIO.md#install-and-start), [#123](docs/TROUBLESHOOTING.md#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token)).
- The artifact manifest pins the six MOSS-Image DiT files, the BF16 and INT8
  ConvRot encoders and the FLUX.2 small decoder at Comfy-Org's and lodestones'
  commits, and the two Kromoss LoRAs.
- `dgxm studio up` installs and starts `dgxm-studio.service`, a user unit with a
  fixed port that also starts the `[studio] driver_service` unit when one is
  set. `dgxm studio down` removes it and keeps the unit when its stop fails;
  `dgxm studio open` prints the sign-in link, opens it when the session has a
  display and says when the service runs code older than the checkout;
  `dgxm studio token` prints the token file and, with `--rotate`, restarts the
  service for a new token. `dgxm uninstall` refuses while the service is
  installed ([Studio](docs/STUDIO.md#keep-studio-running), [#132](docs/TROUBLESHOOTING.md#132-dgxm-studio-up-refuses-or-the-always-on-studio-does-not-start)).

### Compatibility and verification

Node mappings, ordered inputs and defaults stay compatible; Skip Layer Guidance
(DGX Monarch) is a new node. The consent route records an optional `source`,
`panel` (the default) or `studio`, as the waiver row's `consent_source` and
refuses any other value with a 400; Studio's accept sends `studio`
([threat model](docs/THREAT_MODEL.md#companion-to-driver-systemd-and-cli)). The new consent source moves the Gate protocol to 14, which
burns 13, so with the new version each combination re-proves once on first use
([TRUST](docs/TRUST.md#protocol-invariants), [TROUBLESHOOTING #114](docs/TROUBLESHOOTING.md#114-every-combination-runs-its-first-use-gate-again-after-an-update)). The [MOSS-Image record](docs/VALIDATION.md#evidence-moss-image-sft)
is the release's acceptance record; on 2026-10-22 its identity legs ran again
on ComfyUI a071826fd and matched ([VALIDATION](docs/VALIDATION.md#moss-image-on-comfyui-a071826fd)). On 2026-10-21 a Studio lifecycle pass on
two Sparks ran Recycle, a respawn render and Fleet under the observer lease,
then `dgxm restart`, with no failure event ([Studio acceptance](docs/STUDIO_ACCEPTANCE.md)). The
[merged pull requests](https://github.com/Deen-Media/dgx-monarch/compare/v1.2.0...v1.3.0) hold the detail, and TROUBLESHOOTING numbers refer to
[that file](docs/TROUBLESHOOTING.md).

### Rendering

- Support MOSS-Image, the OpenMOSS team's 5B single-stream text-to-image
  model, with its SFT and Base checkpoints and lodestones' Kromoss fine-tune
  ([MODELS](docs/MODELS.md#model-moss-image-sft)). It needs ComfyUI 0.42.0 or newer, the first release with
  ComfyUI's support commit e087d8f6c ([TROUBLESHOOTING #125](docs/TROUBLESHOOTING.md#125-a-moss-image-checkpoint-fails-to-load-on-comfyui-older-than-0420)). The `moss_image`
  adapter shards the joint `[image, text]` stream in one contiguous window per
  rank, keeps the prelude and both text-embedder blocks whole on every rank,
  runs each block's fused-norm calls at stock's row count, names pad rows on
  every topology and refuses a padded ring class K. Under
  sequence parallelism it honors only an identity or `sequence_shard_local`
  block replacement, such as ComfyUI's Skip Layer Guidance skip, and refuses
  any other class P. Five BF16 templates ship at the vendor's settings: 50
  Euler steps at CFG 3 with the layer-22 skip for SFT, Base and Kromoss, a JSON
  layout graph on SFT, and 8 steps at CFG 1 for Kromoss with its DMD2 LoRA.
  Nine testing graphs cover the INT8 ConvRot files, an odd grid, a 1536 square
  canvas, the skip turned off, the second DMD LoRA, the DMD2 LoRA on Base and a
  LoRA slot.
- Promote MOSS-Image SFT, MOSS-Image Base and Kromoss to **HW** on the hardware
  ladder of 2026-10-21 at `ff8dab8300aa` with ComfyUI `d52a7a5ef`, re-measured
  on 2026-10-21 and 2026-10-22 at `9caa6aae660b`, the head capped at 2100 MHz
  ([VALIDATION](docs/VALIDATION.md#evidence-moss-image-sft)). BF16 uly2 was bit-identical to one GPU at one step and over
  50 steps from 1024 to 2048 square on SFT and at 1024 and 1536 square on Base
  and Kromoss, and on SFT to stock at one step from 512 to 2048 square; auto
  keeps it ([BENCHMARKS](docs/BENCHMARKS.md#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped)). FSDP renders matched resident ones, so FSDP is
  supported. cfg2 keeps refusing on the shipped graphs, which carry the skip,
  and gets no row; ring2 is measured, not granted. The Kromoss DMD LoRAs bake to
  stock's merge on every key; their renders differ from stock's because stock
  applies a LoRA at cast time.
- The four CFG 3 MOSS-Image templates use ComfyUI's `normal` scheduler, within
  0.0005 of the vendor pipeline's sigmas at 50 steps where `simple` sits 0.069
  off; on 2026-10-21 the SFT graph on `normal` rendered on uly2 bit-identical to
  one GPU and to stock. The 8-step Kromoss graph and the testing graphs keep
  `simple`.
- The MOSS-Image INT8 ConvRot files pin comfy-kitchen's INT8 attention on all
  42 blocks, so single, cfg2 and dp2 ask for the Sol-Attn waiver and uly2 drops
  the pins ([TROUBLESHOOTING #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)). On uly2, ring and cfg2 the family admits only
  BF16 and INT8 ConvRot.
- Run every INT8 linear of the sequence-parallel MOSS-Image and FLUX.3 Klein
  forwards through comfy-kitchen's INT8 GEMM, as stock does. Since the
  fused-norm parity change some dequantized the weight instead, and MOSS-Image
  INT8 uly2 read one-step NRMS 0.027 to 0.036 against the pin-stripped one-GPU
  render. On 2026-10-21 both families' INT8 uly2 renders were bit-identical to
  it, MOSS-Image ran 0.78 to 0.79 s a step at 1024 square against 1.01 to
  1.03 s, and every BF16 latent rerun was unchanged.
- Refuse a LoRA on a MOSS-Image INT8 file on every topology, class P, before
  any weight moves, in the driver, Fleet KSampler and every Worker. On
  2026-10-21 the Kromoss INT8 file with its DMD2 LoRA read one-step NRMS 0.985
  against stock on uly2.
  dgx-monarch bakes as stock's classic loader does, which with DynamicVRAM off
  renders the pair about 1.0 NRMS from stock's default, so the message names
  that default loader ([TROUBLESHOOTING #129](docs/TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)). The quantization refusal says why
  INT8 ConvRot is admitted under sequence parallelism.
- The `moss_image` adapter follows ComfyUI master's MOSS-Image forward of
  2026-10-22 (commit 60aade0bb) and keeps d52a7a5ef supported; in CPU tests the
  sharded text-to-image forward is bit-identical to stock on both trees
  ([MODELS](docs/MODELS.md#model-moss-image-sft)). Reference images, which the new Text Encode MOSS-Image Edit
  node attaches, refuse class P on every topology but one GPU, at the driver
  and on every rank before any exchange ([TROUBLESHOOTING #130](docs/TROUBLESHOOTING.md#130-a-moss-image-render-with-reference-images-refuses-on-every-topology-but-one-gpu)).
- A one-GPU MOSS-Image edit computes its reference images' keys and values once
  per conditioning, as stock ComfyUI does, instead of on every model call.
  Stock caches them only in inference mode, which a Worker's sampling thread is
  not in, so on 2026-10-22 a one-GPU edit on hardware matched stock's latent at
  1.46 times stock's time a step. The adapter now runs stock's `extra_conds` in
  inference mode on every load, as the FLUX.3 adapter does, and stock's own rule
  still picks the branch ([ADAPTERS](docs/ADAPTERS.md#what-an-adapter-provides)). Off one GPU that step refuses reference
  images first, so no Worker computes them for a render it refuses. On
  2026-10-22 every edit and reference-free leg was bit-identical to stock, and
  the edit ran 2.119 s a step against stock's 2.142 ([VALIDATION](docs/VALIDATION.md#moss-image-on-comfyui-a071826fd)).
- The `qwen_image21` adapter follows ComfyUI master's Qwen Image 2.1 forward
  of 2026-10-22 (commit 503d785f0), which added a `prefix_cache` parameter for
  a writable prefix K/V cache, and keeps a071826fd supported; the
  `import-against-comfy-master` canary passes again. In CPU tests the sharded
  forward is bit-identical to stock on both trees, with the cache off and on
  ([MODELS](docs/MODELS.md#model-qwen-image-2-1)). Cache device `auto` refuses class P on Ulysses topologies under
  dynamic VRAM on ComfyUI 503d785f0 or later, where stock moved `auto` to the
  new writable cache, on every rank alike before any exchange; `gpu`, `cpu` and
  `off` are not affected, and `auto` on a071826fd passes as before
  ([TROUBLESHOOTING #134](docs/TROUBLESHOOTING.md#134-a-qwen-image-21-ulysses-render-with-the-cache-device-auto-refuses-on-comfyui-503d785f0-or-later)).
- Add Skip Layer Guidance (DGX Monarch), which carries stock
  SkipLayerGuidanceDiTSimple's widgets to every Worker, where ComfyUI's own
  node applies them; the graph advisor suggests it in place of the stock node.
  It refuses on a cfg topology, with a two-checkpoint guider and under sequence
  parallelism on any family but MOSS-Image. Model Sampling SD3 (DGX Monarch)
  refuses after it with a step window other than 0 to 1 ([TROUBLESHOOTING #126](docs/TROUBLESHOOTING.md#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider)).
- Refuse a LoRA whose patch changes a weight's shape, on every family and
  topology, before any bake, naming the key and both shapes. ComfyUI master
  since e3bee7ffc applies such a patch, such as the MOSS-Image color and depth
  control LoRA, where the bake failed with an untyped error ([TROUBLESHOOTING #127](docs/TROUBLESHOOTING.md#127-a-lora-that-changes-a-weights-shape-refuses-before-any-bake)).
- Refuse Model Sampling SD3 (DGX Monarch), in the driver and Fleet KSampler, on
  a checkpoint whose ComfyUI config samples at multiplier 1.0: FLUX.1 Schnell,
  Chroma and Radiance, Z-Image and Lumina 2, OmniGen2, Boogu, Mage Flow,
  Qwen-Image, Qwen Image 2.1, Ideogram 4, Krea 2 and Kroma, MOSS-Image and
  Kromoss, and Anima. The shipped graphs wire it only to Wan and Kandinsky 5,
  and a config the driver cannot read is let through ([TROUBLESHOOTING #128](docs/TROUBLESHOOTING.md#128-model-sampling-sd3-dgx-monarch-refuses-on-a-family-whose-comfyui-sampling-multiplier-is-10)).
- Copy each component when the un-bake restore rebuilds a quantized weight on a
  CPU module, where an un-bake to an empty LoRA stack could leave one key
  holding another key's bytes.

### Operations

- Drivers register GET `/dgxm/execution-events`, which grants the Studio
  companion a 30 s observer lease ([threat model](docs/THREAT_MODEL.md#browser-to-comfyui-driver)).
- Studio shutdown waits for the threads that settle history and write
  previews, and the store closes its database under its lock; a close under a
  running settlement crashed the companion with a segmentation fault. Studio's
  job, media, grid and runtime history pages no longer serve a row twice or
  skip one at a page end: each cursor carries the key SQLite sorted by.
  `dgxm-studio` binds its port before it prints or writes the token, so a busy
  port exits with one line and no token.
- A driver started with ComfyUI's `--base-directory` stamps the commit of the
  ComfyUI checkout it runs, found from the directory that holds ComfyUI's
  `folder_paths` module. The stamp read Git state under the user-data root the
  flag names and read `unknown`, so the artifact parity check refused every
  render the driver sent to Workers and Gate lookups read every PASS as stale.
  The runtime provenance snapshot, the telemetry and `dgxm top` now show the
  right commit too.
- A Worker's runtime directory falls back from `XDG_RUNTIME_DIR` to
  `~/.local/state`, then to `/tmp/dgx-monarch-<uid>`, when its path would pass
  64 bytes, so a long home directory no longer stops Workers from starting on
  Monarch's Unix socket path limit.

### Development

- CI builds the Studio frontend and checks that the bundled assets reproduce
  and that `tools/gen_studio_inventory.py` matches the source.
  `benchmark/run_matrix.py` takes Skip Layer Guidance's widgets as
  `slg_double_layers`, `slg_single_layers`, `slg_start_percent` and
  `slg_end_percent`, and the sweep matrix predicts MOSS-Image INT8 LoRA cells
  `refuse:P`. The Studio inventory waives the `dgxm studio` rows, because the
  page cannot start or stop the companion that serves it, and the original
  Studio specification gains a dated amendment for the always-on unit.

### Limits

- Studio is a draft: live update, Worker and Fleet coverage through Studio,
  end-to-end, accessibility and performance acceptance are **NOT RUN**
  ([acceptance record](docs/STUDIO_ACCEPTANCE.md)). On a rig, `dgxm studio token --rotate`, a restart after the
  process is killed and a Studio unit that starts a driver unit are **NOT RUN**.
- MOSS-Image RoPE scaling and the color and depth control, on ComfyUI master
  only on 2026-10-20, ship no template; the control LoRA refused on both ranks
  and the next render ran with no Recycle. No cluster edit has been compared
  with stock. cfg2 with the skip off is exact on a prompt pair of different
  lengths and measured on one of equal length. ring2 with a LoRA is not
  supported: every first-use Gate ceremony of that pair that reached a verdict
  failed.
- The Qwen Image 2.1 forward on ComfyUI 503d785f0 has not been re-run on
  hardware.

## [1.2.0] - 2026-10-16

**Upgrading from 1.1.0.** Follow [INSTALL](docs/INSTALL.md#upgrading-from-110): the first update runs the 1.1.0
updater, so this release's checks start with the next one. If a host's Workers
run from a release slot (remote hosts of guided multi-node setups, every host
after a verified update), use only `dgxm update --verify`; plain update leaves
them on 1.1.0 ([TROUBLESHOOTING #118](docs/TROUBLESHOOTING.md#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)). From 1.2.0 on, plain update refuses such a
host and Doctor's `worker release version` row fails a stale one. Before
updating, turn `rdma_latent_return` off, add no `[monarch]` table and clear
`DGXM_RDMA_QP_SPLIT` where Workers start. Then install `xxhash` and what
`pip check` names on every host, clear what [TROUBLESHOOTING #119](docs/TROUBLESHOOTING.md#119-a-mesh-build-or-worker-start-refuses-on-an-ambient-environment-variable) names, add a
`[monarch.rdma]` target before re-enabling native return, rerun the editable
install after a verified update, and restart ComfyUI. The torchmonarch pin moves
to 0.7.0; FLUX.3 Klein needs ComfyUI 0.41.0 or newer, or commit 20b46065e.

### Everyday use

- A ComfyUI that stays up attaches again after its Workers restart within the
  30 s delivery timeout, and refuses older Workers before it dials (closes [#2](https://github.com/Deen-Media/dgx-monarch/issues/2)).
  So does a sampler render that meets restarted Workers, recording `reattach`
  and `reattach_s` (closes [#47](https://github.com/Deen-Media/dgx-monarch/issues/47)); the two-Spark recheck ran on 2026-10-16.
- Eager setup and Fleet queues heal an abandoned lease without **Reset attached
  mesh**. The sidebar shows the topology and each host's rail rates, red when a
  rail is down, and progress no longer sticks after a render.
- Add three Kroma v0.3.1 Turbo workflows, which matched stock at batch 1, and
  four FLUX.3 Image Klein workflows, for 66 in all.

### Installation and updates

- From 1.2.0 on, plain `dgxm update` refuses a release-slot host before it
  fetches (`release_slot_worker`, `release_slot_unknown`), checks the fetched
  commit's dependencies on every host, fast-forwards only if they pass, and
  refuses an uncommitted `pyproject.toml` edit or a linked worktree.
- Doctor, `dgxm up` and `dgxm restart` read a managed Worker's torchmonarch from
  its release; the skew warning sends a slot to `--verify`; Doctor checks that
  `/usr/bin/python3` is 3.9 or newer on each host.
- Verified updates take `--privileged-process-inspection`, keep exact legacy
  Worker units, and accept a foreground inspector of up to four hours.
- Every manifest row pins a Hugging Face commit, size and sha256 or names the
  file you supply (101 pinned, 6 supplied), and CI checks it. The LTX 2.5
  Gemma 4 row pins the vendor's current file, checked again against stock.
  `setup_env.sh --comfy` no longer needs torchaudio.

### Compatibility and verification

Node mappings, ordered inputs and defaults stay compatible. The new pin and
version make each combination re-run its first-use Gate ([TROUBLESHOOTING #114](docs/TROUBLESHOOTING.md#114-every-combination-runs-its-first-use-gate-again-after-an-update));
later edits that cannot change math, residency or transport keep PASS rows.
[Migration checks](docs/VALIDATION.md#gpu-checks-for-the-070-migration) ran the shipped Krea2 workflow at 1536 square on that build's
uly2 route (row 11; slab Gate PASS) and the shipped H3 workflow (output only) on
ComfyUI e8665ca. Later records ran on ComfyUI 29e28e2b4 (0.41.0); the
[FLUX.3 Klein record](docs/VALIDATION.md#evidence-flux-3-klein) is the release's acceptance record. The [merged pull requests](https://github.com/Deen-Media/dgx-monarch/compare/v1.1.0...v1.2.0)
hold the detail, and TROUBLESHOOTING numbers refer to [that file](docs/TROUBLESHOOTING.md).

### Rendering

- Support FLUX.3 Image Klein, step-distilled and base, for text to image and
  reference editing ([MODELS](docs/MODELS.md#model-flux-3-klein), [TROUBLESHOOTING #121](docs/TROUBLESHOOTING.md#121-a-flux3-checkpoint-fails-to-load-on-comfyui-older-than-0410)). Both rows are **HW**. uly2
  matched one GPU exactly on text to image from 1024 to 1536 square and, since
  each fused norm runs at stock's row count, on edits and small sizes
  (2026-10-16). ring2 and base cfg2 are measured, not exact; NVFP4 refuses FSDP.
- `auto` takes cfg2 where it was exact and beat uly2 in [Campaign K](docs/BENCHMARKS.md#campaign-k-cfg2-against-uly2-2026-10-14-head-capped): Krea2 FP8
  and BF16 from 1.2 to 2.36 MP, Chroma BF16, FP8 and INT8 to 1.64 MP, Qwen-Image
  FP8 to 1.77 MP, Boogu FP8 and Anima BF16 under 1.2 MP, and CFG++ at CFG 1.
- A checkpoint or LoRA that pins an approximate attention kernel refuses class K
  where the pin runs, until you grant the Sol-Attn waiver ([TROUBLESHOOTING #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel),
  closes [#30](https://github.com/Deen-Media/dgx-monarch/issues/30)). Sequence parallelism drops the pin. A BF16 or FP16 checkpoint's
  own pins refuse class P on `+fsdp`, and `first_use` refuses a running pin on
  cfg2+fsdp and dp2+fsdp even with the waiver.
- Refuse a checkpoint whose weight scales ComfyUI would drop, such as FLUX.3
  Klein's `fp8r` file, class P with no waiver ([TROUBLESHOOTING #120](docs/TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
- A Gate FAIL quarantines only its own combination, LoRA keys on FSDP-replicated
  modules bake, Anima runs sequence-parallel again, padded Ulysses ran 2.2%
  faster in a Chroma test, and FSDP reload pricing credits freed shards.

### Operations

- Add a validated `[monarch]` policy for startup, RDMA and diagnostics. Config
  loading names a bad key, native return needs a `nic:` target per host, a
  policy edit reads `POLICY_STALE` (TROUBLESHOOTING #113) instead of locking out
  lifecycle commands, and a refused edit keeps the fleet and its export.
- A mesh build or Worker start refuses an ambient OTLP endpoint with export off
  and `MONARCH_RDMA_*` or `DGXM_RDMA_QP_SPLIT` values the policy did not set
  (TROUBLESHOOTING #119); so does an export that cannot start.
- Add opt-in Monarch metrics with ownership occupancy, OTLP export, and phase
  timings that split each native return. `/dgxm/metrics` adds step time,
  native-return latch and faults, and per-port rails; the finish record adds
  topology, table row, quant, kernel and pins. Monarch's profilers stay off.
- A failed native read latches native return off for the session
  (TROUBLESHOOTING #115). Doctor adds `listener generation` and `worker policy`
  rows and, like `dgxm restart`, names why a source probe failed. Fix partial
  `dgxm diagnostics` captures, `dgxm top` rates and phases, sidebar refresh
  after a reset, a Reset that raced a loader preflight and stale ledger temp
  files. Load profiles live in `~/.cache/dgx-monarch/profiles/`.

### Development

- CI type-checks under `check_untyped_defs`; canaries cover pins, scale replay,
  LoRA bakes and local actors. The sweep labels pinned cells and CFG 1 cfg cells
  under FSDP as refusals; the matrix runner records waivers and refusal classes.
  `tools/make_pin_probe_lora.py` and `rdma_return_bench.py --timing` are new.

### Limits

- Native latent RDMA stays off by default; [selected-rail checks](docs/VALIDATION.md#native-rdma) keep their
  recorded scope. Not yet run on hardware: FSDP LoRAs on H3, LTXAV or Anima, the
  credited Flux2 proof, return timing, clean cancellation and Kroma's batch-2 `dp2+fsdp`.
- Kroma keeps Ulysses: its cfg2 failed the same-math bar on 2026-10-16, so Krea2's cfg2 row
  excludes it, even when `family_adapter` names `krea2`, and the sweep labels Kroma cells to match.

## [1.1.0] - 2026-10-11

### Everyday use

- Show health separately from activity in the dashboard and browser. Healthy
  rendering reads READY / RENDERING with a wait instruction. Preserve legacy
  readiness JSON and recordings, and explain activity that CLI status does not
  observe. Unknown or blocked evidence remains visible.
- Add opt-in `dgxm launcher` access from a normal terminal, bound to the selected
  environment and cluster config. Preserve existing commands and make identical
  reruns a no-op. Setup verifies PATH resolution and the dashboard endpoint.
- Launch ComfyUI alongside its dashboard with `--with-top`, including from the
  desktop shortcut. Preserve foreground logs and terminal signals, and close
  only the driver and dashboard owned by that launch.
- Add a MiniMax H3 workflow that saves H.264/AAC MP4 at 24 fps. Keep the existing
  separate-frame/audio workflow. The catalog now contains 59 workflows.

### Installation and updates

- Add an optional fleet-wide Sol-Attention installer with a reviewed package
  plan and protected runtime checks. Keep the explicit GPU dense-control smoke
  separate from installation. Diagnose dependency requirements, Python headers
  and compiler caches; document the accuracy waiver and mesh recycling.
- Accept proven editable metadata companions and preserved core-library links
  during Sol maintenance. Keep installed-file ownership authoritative and reject
  ambiguous metadata, unsafe replacements and changes to protected packages.
- Derive the Torch install constraint from distribution metadata while retaining
  runtime/CUDA checks. Preserve working Torch builds and report useful package
  ownership details when verified updates refuse an installation.
- Support the verified color-matcher 0.6.0 test-data layout in TorchMonarch
  inventory checks, with a recovery command for older updaters. Keep checks for
  changed files, duplicate ownership and import shadows.
- Report Worker service health from fresh passive process and listener checks,
  separately from attached actor telemetry. Missing or failed observations
  remain unknown.
- Simplify guided setup and handoff: gather prerequisites before plan approval,
  validate private receipt paths and process-inspection capability early, and
  put launch commands and installation paths first. Detailed network auditing
  is optional; trusted isolation and operator acknowledgement remain required.
- Let users select a supported first-render model; Chroma is optional. Preserve
  recognized deletions of ComfyUI's two tracked placeholder files during setup,
  with a warning; other checkout changes still block it.

### Compatibility and verification

Existing node mappings, ordered inputs and defaults remain compatible. Core
runtime dependency pins are unchanged. An upgrade requires first-use checks
before cached identity-gate results can be reused.

[Operator acceptance](docs/VALIDATION.md#onboarding-and-operator-acceptance)
records the tested source, two-Spark execution and cleanup scope. Sol dense
kernel checks and a saved MiniMax MP4 are functional evidence, not sparse
attention or model-fidelity qualification. Native latent RDMA remains
**NOT RUN / HOLD**. The cuSPARSELt ARM64 packaging exception remains tracked in
[issue #5](https://github.com/Deen-Media/dgx-monarch/issues/5).

## 1.0.0

Initial release of DGX Monarch: distributed image and video generation in
ComfyUI, built for a pair of NVIDIA DGX Sparks.

### Rendering

- Split supported renders across both GPUs with sequence or CFG parallelism,
  or run independent prompts with Fleet.
- Shard supported model weights with FSDP when a checkpoint needs more memory.
  Text encoding, VAE decode and saving remain in the ComfyUI driver.
- Start from 58 example workflows, including Krea2, Ideogram4, MiniMax H3 and
  Qwen Image 2.1. [Model support](docs/MODELS.md) lists the tested checkpoints,
  precisions, inputs and parallel modes.
- Use automatic topology selection or choose an explicit layout. Slab
  residency and low-RSS LoRA handling offer memory savings where supported;
  first-use checks compare those paths before reuse.

### Installation

- Check dependencies with `tools/check_dependencies.py`. It verifies the
  official artifact before accepting the known upstream packaging exception;
  other dependency errors fail.
- Install from source into the Python environment used by ComfyUI, preserving
  its CUDA Torch build and building the pinned xFuser compatibility wheel.
- Configure both Sparks through guided setup with a reviewed plan, dependency
  and model checks, Worker service ownership checks, and saved receipts.
- Follow the repository's [setup skill](skills/dgx-monarch/SKILL.md), directly
  or through the [copyable README prompt](README.md#set-up-with-a-coding-agent).
  It covers discovery, approvals, installation, validation and recovery.
- Use the [first-render recipe](docs/QUICKSTART.md#first-distributed-render)
  for exact model files, pinned downloads, hashes and distributed settings.
  A fresh-agent trial on two Sparks completed saved Chroma renders and a
  non-destructive setup rerun. [Installation results](docs/VALIDATION.md#fresh-agent-installation)
  describe the tested baseline, reused components and limits.

### Operations

- Inspect the pair with Doctor, status, the ComfyUI sidebar and the terminal
  dashboard. Diagnostics distinguish failed checks from missing information.
- Apply verified updates with coordinated Worker activation, source checks,
  Doctor and a cluster smoke test. Prior releases and transaction records
  support recovery, including when a Worker shares the driver's Spark.
- Check capacity before loading and confirm owned actor cleanup before reuse.
  Worker services remain running when ComfyUI closes; stopping them is a
  separate operator action.

### Development

- Run CPU checks on GitHub-hosted x64 runners with Python 3.11 and 3.12, and
  ARM64 with Python 3.12. The x64 jobs also check package contents and CLI
  installation; ComfyUI canaries check upstream compatibility.

### Limits

- PyTorch still pins the NVIDIA cuSPARSELt ARM64 wheel with a platform-tag
  error. The [dependency checker](docs/TROUBLESHOOTING.md#109-pip-check-reports-cusparselt-is-not-supported-on-this-platform)
  verifies and reports this exception without changing installed packages.
- Support and measured speedups apply to the configurations listed in
  [Model support](docs/MODELS.md), [Benchmarks](docs/BENCHMARKS.md) and
  [Validation](docs/VALIDATION.md). A workflow template does not establish
  accuracy for every precision, LoRA, attention kernel or memory mode.
- FSDP is a capacity option and can be slower. Optional memory optimizations
  have their own checks and restrictions; failed comparisons do not grant
  support for them.
- The inter-node network must be isolated and source-restricted. The Worker
  API has no peer authentication; follow the
  [fabric requirements](SECURITY.md#fabric-trust-boundary).
- Native latent RDMA remains **NOT RUN / HOLD** and is off by default.
  Latents return through actor messages; distributed rendering uses NCCL.
- A failed reply channel can require a fresh ComfyUI or driver process after
  confirmed actor cleanup. See [Troubleshooting](docs/TROUBLESHOOTING.md).
