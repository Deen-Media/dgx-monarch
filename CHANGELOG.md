# Changelog

## [Unreleased]

### Everyday use

- MiniMax-H3-Image (ComfyUI 0.45.0) runs on its own sharded adapter: stock's
  own packing on every rank, sequence shards with exact pad exclusion on
  Ulysses, and the stacked output head on the checkpoint's grid at stock's
  rows. cfg2, dp2, a padded ring, a batch above one and a head whose depth the
  grid does not match refuse with a plain sentence, as do the MiniMax H3
  sampling-shift and context-window nodes on this family. Both H3 adapters now
  set stock's `block_index` on each block's options. The header names the
  family, so auto takes uly2 for it and the driver prices its packed rows and
  its text encoder before dispatch. A row-check tool records the three
  row-keyed calls at a rank's rows for the hardware run.
- MiniMax-H3-Image ships two templates, text to image and edit, on the files
  ComfyUI's own templates load: the INT8 ConvRot DiT and text encoder, the
  FP16 VAE and the W4A8 prompt expander, with KSampler at 6 steps, CFG 1,
  `euler` and `simple`. The edit scales its picture with ComfyUI's two Resize
  Image/Mask nodes. ComfyUI's prompt expander chain is wired as ComfyUI wires
  it, behind an If/Else switch that ships off in both templates. Its task
  prompt and system prompt boxes hold this project's own texts, written from
  public research on the model; the text ComfyUI's templates carry there is
  MiniMax's, and this project does not ship it. Turn the switch on to have
  the expander write the prompt, or replace the texts with your own. An
  answer that holds no caption falls back to the request as written, where
  ComfyUI's own graph would render from an empty prompt. A testing graph
  samples the
  checkpoint's own sigma grid through Sampler Custom (DGX Monarch). The sweep
  labels the family's cfg2 and dp2 cells from the driver's own preflight.
  [MODELS](docs/MODELS.md#model-minimax-h3-image) lists the eight published
  files with their sizes and names the licence MiniMax publishes them under.
  No render of this family has run on hardware yet
  ([VALIDATION](docs/VALIDATION.md#minimax-h3-image)).
- A graph whose prompt expander is on with an empty task prompt, or with one
  that lacks the marker Replace Text fills, refuses class P at the UNET
  loader, the samplers and Identity Gate, before the DiT loads or a picture
  renders. ComfyUI raises no error there, and the render would follow a caption
  unrelated to the request, or spend the expander's run and fall back to the
  request
  ([TROUBLESHOOTING #153](docs/TROUBLESHOOTING.md#153-a-render-refuses-because-the-prompt-expander-is-on-with-an-empty-task-prompt)).
- The sweep converter, which the template release test shares, reads the
  child widgets a ComfyUI dynamic combo reveals, under their dotted names,
  from a driver's schema or from the template generator's table. Every
  committed graph converts as it did, except the nodes that hold such a
  widget, the first to do so: the new edit graph's two resize nodes and
  Generate Text in the three MiniMax-H3-Image graphs
  ([SWEEP](docs/SWEEP.md#3-adding-a-template)).
- Every image, clip and music track a shipped template's input nodes name now
  comes with the pack, in `example_workflows/input`. At start the pack copies
  each one ComfyUI's input folder lacks, and never replaces a file, link or
  folder already there, so a template such as the Qwen Image 2.1 masked edit
  no longer opens on a fresh install with a missing `qwen21_source.png`. They
  are drawn shapes, a stick figure and a synthetic music loop that show the
  graph runs; load your own for real work. A demo file you delete comes back
  at the next start unless `DGXM_DEMO_INPUTS=off` is set, which skips the
  copy ([INSTALL.md](docs/INSTALL.md#install-from-source)), and
  `tools/gen_demo_inputs.py` draws the files again.
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
- The UNET loader now prices every model the graph loads on the driver. Under
  `auto` the DiT loads at the first render, after the driver has loaded the
  text encoders, VAEs and any prompt upsampler, and the loader's load window
  used to charge the DiT alone: on 2026-11-05 the BF16 p-h3-image template
  passed it on one Spark, ran its 18 GB upsampler, and refused at the sampler
  2.0 GiB short. That window now charges those models too, so the template
  refuses before anything loads, and the refusal opens with one plain sentence
  ("p-h3-image_bf16.safetensors needs 119.8 GiB on this box and 110.8 GiB is
  available, 9.0 GiB short") and names what would fit: the same model in
  another precision on disk here (the INT8 ConvRot and W4A8 DiT and upsampler
  files), a switch whose other setting skips a model, and a pair with the Init
  node's topology set to `uly2+fsdp`, which `auto` never picks, whose first
  render runs the gate's clean-reload proof. It never names a file the graph would then refuse:
  no INT8 MOSS-Image file beside a LoRA, no W4A8 file on a `*+fsdp` preset, no
  file that pins an attention kernel off Ulysses and ring, and no DiT file no
  graph here names. It also offers the Load with slab residency card, whose
  render of this graph has no hardware record yet. Models ComfyUI already
  holds are credited up to their size, where any one held model used to credit
  the whole stack, so a second graph's upsampler is priced too. On a fresh
  driver the price errs high: LTX 2.3 text to video now needs 80.9 GiB where
  it needed 65.8. A model only a fixed switch's other branch reaches, or no
  output node, is no longer charged, so the template with the upsampler
  switched off loads. MOSS-Image, MOSS-Image Base and Kromoss
  get the loader's price too, their 19.31 GB encoder included. A loader
  `weight_dtype` cast, which used to stand the loader's check down whole, is
  priced there at the copy the cast builds, the figure the Worker's stock
  price uses, where the load runs stock. A refusal a
  Worker raises at the sampler, the loader or Identity Gate now opens the prompt error, card
  sentence included, with the traceback in the log, where the error used to
  begin "Actor call dgxm_worker.sample failed.". With `slab_weights=on` and no
  Identity Gate PASS, after a failed authorization, for a second model or on
  Fleet KSampler, the refusal says the driver ran stock and why, and names the
  gate where a PASS would help, where it used to ask for `slab_weights=on`. The Old model in
  the way card's loader text keeps the sentence that turns the preflight off,
  and its no-room sentence quotes the same window and figure as the lead
  ([TROUBLESHOOTING #52](docs/TROUBLESHOOTING.md#52-driver-side-footprint-preflight-refused-a-render),
  [#53](docs/TROUBLESHOOTING.md#53-a-render-failed-and-the-panel-is-asking-for-one-click)).
- A load that does not fit because a model loaded earlier is still
  held now says so: "the old model (<name>) is still loaded on <box>". When a
  clear of the attached mesh would make room on every short box, the panel
  raises an Old model in the way card whose Clear and retry runs the same reset
  as Reset attached mesh, then you queue again. Its Always clear and retry
  checkbox, the toggle of that name under Consents and waivers, or
  `DGXM_ALLOW_CLEAR_RETRY=1` on the driver lets the driver clear and retry by
  itself, once per load. The card writes no consent and no ledger row, and the
  load is priced again on the fresh Workers. Both the cross-rank agreement and
  the loader's own check raise it, and the agreement's slab rescue card now
  names a held model too. A one-GPU fleet or a Fleet job, whose Worker prices
  its own load, names the memory a model loaded earlier left behind and Reset
  attached mesh instead. On 2026-11-04, 19 of 20 template errors were this
  case ([TROUBLESHOOTING #53](docs/TROUBLESHOOTING.md#53-a-render-failed-and-the-panel-is-asking-for-one-click),
  [#93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box)).
- A load whose UNET loader casts the weight dtype is now priced before it
  loads. The capacity checks used to skip a cast, because file size does not
  say how big the cast copy is, and admit the load with no figure; on
  2026-11-05 `dgx-monarch-wan22-t2v`, which casts both DiTs to bf16, loaded its
  second DiT on a head with 12.2 GiB free and only a test watchdog stopped
  it. The price now reads the file's header for the size of the copy the cast
  builds and charges it in place of the file, beside the host copy and the
  floor, so a short box refuses before anything loads and says what would fit.
  For a cast that keeps the width, as wan22's fp16 to bf16 does, the figure is
  the one an uncast load gets (60.9 GiB for each wan22 DiT). A cast load offers
  no slab rescue, since a slab under a cast keeps the cast copy beside the
  file; `slab_weights=on` refuses a cast it cannot price, and the explicit
  stock refusal no longer suggests it. An old model held in the way still
  raises its card, which now credits a cast model the copy its cast built.
- A load that replaces the stock model in its own slot is now priced after
  the Worker drops that model, which it always does first. The agreement used
  to count both copies; with its cast priced, wan22-t2v's stage 2, a swap over
  stage 1's DiT, would then refuse at 58.7 GiB available on a fresh pair, and
  each stage of a warm re-queue would ask for a clear. A LoRA change on the
  same file is the same credit. A model in the other slot, a slab resident and
  the arena a stock load keeps still count
  ([TROUBLESHOOTING #5](docs/TROUBLESHOOTING.md#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc),
  [#93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box)).
- The capacity refusals, the Old model in the way card and the driver's
  capacity log line name a box as the head (where the driver runs) or the
  sibling, the way the docs do. Before, one refusal could name the same box by
  its host name, by the address cluster.toml gives it, and as "rank 0", so on a
  pair configured by address the card read "is still loaded on" and an
  address. Now it reads "The old model (flux1-dev) is still loaded on the head
  and qwen_image_2512_fp8_e4m3fn does not fit beside it. Clear the Sparks and
  run again?". With more than two boxes each other box is "the sibling at" its
  host name, and a fleet on the driver's box alone is "this box". The FSDP
  clean-reload refusal and the refusal for a Worker actor a restart left
  undeliverable name their boxes the same way, and the loader's charge note
  says "a Worker on this box" where it said "co-resident rank 0"
  ([TROUBLESHOOTING #93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box),
  [Concepts](docs/CONCEPTS.md#the-machines-and-the-processes)).
- A Worker's own refusal names no host either. Its sentence says "here", and
  the driver opens it with the box that refused: "On the sibling, stock
  residency cannot load p-h3-image_bf16.safetensors here: the load needs ...".
  On one box, where the loader card says "this box", it reads "here" with
  nothing before it. The slab, comfy-managed, rescue and LoRA bake walls read
  the same way, and so do a model file missing on one box, whose error opens
  "On the sibling, actor call dgxm_worker.load_model failed." above "model ...
  was not found here", and a sampler or scheduler one box's ComfyUI lacks. A
  Fleet job takes the box of the actor it went to, and the Fleet timeline's
  rows read "the head" and "the sibling" where they read "box 1" and "box 2"
  ([TROUBLESHOOTING #93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box),
  [Concepts](docs/CONCEPTS.md#the-machines-and-the-processes)).
- The error a sampler, Fleet KSampler, a UNET loader or Identity Gate raises for
  a refusal ends with its sentence and the card's plain words. It no longer
  ends with the measured numbers and the consent block as JSON: the card is
  already raised when the driver receives the refusal, and ComfyUI's log keeps
  both blocks. Studio's Queue row offers no Grant and retry for such a refusal;
  grant it from Operations, consent apply, which lists the driver's pending
  requests ([TROUBLESHOOTING #53](docs/TROUBLESHOOTING.md#53-a-render-failed-and-the-panel-is-asking-for-one-click),
  [Studio](docs/STUDIO.md#read-a-failure-and-grant-a-consent)).
- A LoRA change on a BF16 model the Workers already hold, with `lora_low_rss`
  off as the Wan-Animate 2 BF16 templates set it, no longer stalls, and no
  longer ends in a refusal with nothing to click. Adding a LoRA to a bare base
  loads base and LoRA fresh instead of copying the whole model off the GPU
  first, at the same price: on the memory-saver template that copy took 230 s,
  and a fresh load of base and LoRA 17 s. And a change refused because the host copy the held LoRA keeps holds the
  memory now offers the Old model in the way card, "The model loaded earlier
  (wan_animate_2_bf16) holds about 29.6 GiB on the head that this change needs.
  Clear the Sparks and run again?", when a cleared box would hold a fresh load.
  This is the LoRA case of the credit for a load that replaces the stock model
  in its own slot
  ([TROUBLESHOOTING #18](docs/TROUBLESHOOTING.md#18-first-lora-change-on-a-quant-model-is-one-time-slow-historical),
  [#93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box)).
- A new strength, a new order or a LoRA removed on a BF16 model the Workers
  hold with its LoRA, with `lora_low_rss` off, is admitted without a click
  where it fits. The change reuses ComfyUI's host copy of the patched weights,
  which the Workers now report, and the price counts it as held, while it
  still charges the LoRA files the change reads again: on the pair, the
  Wan-Animate 2 strength change from 1.0 to 0.8 needs about 8.7 GiB where it
  needed 38.6 GiB and raised Old model in the way. On one Spark the same change
  took 0.8 GiB, about the LoRA file it reads again, and rendered a fresh load's
  latent. A LoRA the held stack lacks keeps the old price and the card. And a
  re-queue whose UNET loader ComfyUI kept from an earlier queue is now priced
  at the sampler or at Identity Gate (DGX Monarch), by the loader's own price
  for what is not yet held, so a short box gets the loader's refusal and what
  would fit, where only the Worker's wall answered before ([TROUBLESHOOTING #52](docs/TROUBLESHOOTING.md#52-driver-side-footprint-preflight-refused-a-render),
  [#93](docs/TROUBLESHOOTING.md#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-box)).
- The Ideogram 4 template keeps its JSON caption in a Text (Multiline) node
  wired to CLIP Text Encode. Typed into CLIP Text Encode's own box, the
  browser's dynamic prompt pass stripped every brace pair on Queue, so a
  browser render sent broken JSON while API submissions, the sweep and Studio
  sent the caption as written. App Mode and Studio show the caption as the
  prompt field ([MODELS](docs/MODELS.md#model-ideogram4)).
- Template Notes describe the identity gate as it runs: proof renders of at
  most two steps on the graph's own mesh and topology, before and after the
  swap or reload under test, plus one with slab off when slab is on, never a
  stock single-GPU load. The Identity Gate's Note and steps tooltip
  call a PASS a finite check instead of saying 2 steps prove as much as 20.
  Notes no longer say FSDP refuses fp8, int8 or fp16 files or that Anima's
  adapter makes it refuse; the Flux 2, Kandinsky 5 Pro and Anima Notes state
  the FSDP launch and LoRA rules, and each keeps its own hardware limits.
- Studio waits up to 280 s for actors recycle and consent apply, 20 s past the
  260 s the driver allows an attached-mesh reset, so a reset that takes longer
  than 20 s records the driver's receipt instead of unknown. Other driver
  requests still wait 20 s, and a lost answer still records unknown
  ([Studio](docs/STUDIO.md#install-and-start)).
- The ComfyUI panel rebuilds a consent card or an active grant that the driver
  replaced under the same key and count, so Accept, Dismiss and Revoke send the
  id the driver serves now, and the card shows the driver's current text.
  Before, after another tab dismissed a card and the render raised it again,
  the first Accept in the first tab was refused
  ([TROUBLESHOOTING #53](docs/TROUBLESHOOTING.md#53-a-render-failed-and-the-panel-is-asking-for-one-click)).
- When Studio cannot cancel a running job because the driver has no
  prompt-specific route, every request that joined the cancellation stays
  unspent, as the first one did. Before, a second request that joined while the
  first waited for the driver replayed as not found
  ([TROUBLESHOOTING #124](docs/TROUBLESHOOTING.md#124-studio-refuses-a-submission-retry-upload-cancellation-operation-or-update)).
- Studio's grid cell Inspect full submission and Export cell, and the grid
  exports, keep every digit of a whole number above 9007199254740991, such as
  the third seed of an incrementing grid that starts at 9007199254740991. The
  parameter differences under a comparison tell two such seeds one apart, and
  compare a field nested deeper than 12 levels as one value and say so, where
  such a field used to drop out of the table
  ([Studio](docs/STUDIO.md#working-with-workflows)).
- Studio's Validate every cell stays off, and the line beside it says why,
  while a grid axis value is a whole number outside -9007199254740991 to
  9007199254740991, or the base seed of an incrementing grid is above
  9007199254740991. The browser would have sent it rounded, so the grid ran
  other seeds than the ones typed
  ([Studio](docs/STUDIO.md#working-with-workflows)).
- In `dgxm top`, a paused or scrubbed view shows the run story of its own tick
  once the one-hour history is full. Before, pausing again at the same place
  could show the story from the earlier pause ([TUI](docs/TUI.md#record-and-replay)).
- Studio's metric history leaves out a reading whose host or rail clock runs
  more than 5 s ahead of the companion's, as it already left out such a sample.
  Before, one Worker clock ahead of the driver put a future point in that
  host's chart and hid every corrected point until real time caught up
  ([Studio](docs/STUDIO.md#workspace-controls-and-media)).

### Compatibility and verification

- This release needs ComfyUI 0.45.0 or newer. ComfyUI 0.45.0 changed the
  MiniMax H3 output head to take one sigma grid per stream, and the sharded
  H3 forward now passes those grids, built as stock builds them. It reads the
  head's form when it binds: under sequence parallelism, a stacked PDD head on
  an older ComfyUI, or a head form it does not know, refuses class P before the
  render, where 1.4.0 failed inside the head. Every listed H3 and p-h3-image
  file has a one-block head and renders as before. The sharded block loop also
  pops the prefetch queue with stock's block scope, so ComfyUI's malloc graph
  under comfy-managed residency plans each block as stock's forward does ([TROUBLESHOOTING #147](docs/TROUBLESHOOTING.md#147-a-minimax-h3-render-refuses-or-fails-in-its-output-head-after-a-comfyui-update)).
- The comfy canary's outside forward modules seam passes on ComfyUI 0.45.0
  again. UltraTex made Flux's `extra_conds` call `preprocess_token_masks`, and
  the seam now holds a table of such calls that touch no module and no
  parameter: it runs the call on a tiny real Flux with every parameter watched,
  so the canary fails the day upstream makes it reach one. The conditioning
  wire seam also holds UltraTex's view grid, masks and view list
  ([ADAPTERS](docs/ADAPTERS.md#flux-family-view-conditioning-ultratex)).
- docs/CAMPAIGN_COLUMNS.md types `legs.probe.message` and
  `legs.resident.message` as text: their values open with a plain `[dgxm:...]`
  tag and are not JSON arrays. A column now types as JSON only when every value
  parses as one ([CAMPAIGN_COLUMNS](docs/CAMPAIGN_COLUMNS.md)).
- The hardware checks of 2026-11-03 to 05 are in the docs. p-h3-image and
  Wan 2.2 TI2V 5B are HW: all three p-h3-image DiT files on one rank match
  stock and on uly2 match one rank, its LoRA route matches stock, and the
  BF16 template refuses on one Spark before anything loads and renders each
  way the refusal names but the INT8 ConvRot DiT; at its 20 steps Wan 2.2
  TI2V on one GPU matches stock and on uly2 matches one GPU. VALIDATION gains
  the template release
  test of 2026-11-04 and what #139 fixed, the other 32 FP8 compute cells
  (HunyuanImage 2.1's plain FP8
  stays inside the floor on `fp8_ops` and the Ideogram4 fp8_scaled pair does
  not, against BF16; `fp8_matrix_mult = "on"` stays unsafe), Worker-assisted
  VAE decode on Wan 2.2 TI2V and both HunyuanVideo VAEs with its fail-open,
  warm, short-of-memory, 362-frame and Clear VRAM checks, the MiniMax H3
  context windows, the ComfyUI 0.44.0 rows, the BF16 rescue's other forms,
  the pipelined unload, the Old model in the way card, the cast and swap
  credits, and a LoRA change on a held base. The torch 2.15.0 ladder passed
  steps 1 to 10 on the pair on 2026-11-05 and 06, and this release ships on
  torch 2.15.0+cu132. Ideogram4's MODELS row now says its fp8_scaled pair
  reads 0.161 against the BF16 pair, from the file's quantization, and the
  p-h3-image template's note says what its edit's hardware record shows.
  Still NOT RUN: the release test on torch 2.15, LTX 2.3's decode split, a
  Worker on ComfyUI 0.43 with context windows, every layout at world 4 and the
  full sweep runbook
  ([VALIDATION](docs/VALIDATION.md#release-test-2026-11-04),
  [VALIDATION](docs/VALIDATION.md#torch-2-15),
  [MODELS](docs/MODELS.md#model-p-h3-image),
  [MODELS](docs/MODELS.md#model-wan-2-2-ti2v-5b)).
- comfy-kitchen 0.2.41, the release ComfyUI master now pins, joins the INT8
  GEMM kernel's validated set, so an install that follows that pin keeps the
  kernel on instead of running every INT8 linear on CUTLASS with a WARNING.
  0.2.41 changes no INT8 or W4A8 code, and the kernel's GPU check matched 240
  of 240 cases on it. The render comparison was not rerun: a release now needs
  one only when it changes the INT8 path. The comfy canary's comfy-kitchen check, red against ComfyUI
  master, passes again ([Models](docs/MODELS.md#int8-gemm-kernel),
  [VALIDATION](docs/VALIDATION.md#int8-gemm-kernel-comparison-2026-10-28-and-29)).
- The video templates follow ComfyUI 0.44.0's Create Video, which gained
  `chroma_subsampling` before `codec` (ComfyUI #17463); they hold its default,
  `auto`. On ComfyUI 0.43 and older, which lack the widget, this pack's
  frontend extension restores Create Video by name, and Studio and the sweep
  converter skip the slot, so `codec` stays `none` there. The comfy canary's
  template step, red against ComfyUI master since 2026-11-01, passes again,
  and it now checks Save Video's format and codec against the options the
  running ComfyUI offers ([MODELS](docs/MODELS.md#model-minimax-h3-t2va-fl2va-ref2va-av)).
- Every actor starts with `TORCH_DIST_USE_NCCL2=0`, so torch 2.15 keeps its
  legacy NCCL backend, the one torch 2.12 ships. torch 2.15's new `nccl`
  backend takes an allocator snapshot that ComfyUI's cudaMallocAsync allocator
  refuses, which ended each actor at its first process group; torch 2.12
  ignores the setting. `dgxm doctor` gains an `actor NCCL backend` row that
  names the torch the actors import and checks the setting. Each Gate PASS now
  also names the driver's torch and CUDA build, so a torch change runs the gate
  again on each gated combination's first render, and the notice names both
  builds when only the build moved. A row written before this change carries
  no build and reads stale. INSTALL now says what the ComfyUI guide's torch
  2.15.0 changes, gives the import check its commands and notes the torchaudio
  builds that no longer load against torch 2.12.0+cu132. On 2026-11-05 and 06
  the torch 2.15 ladder passed on the pair with the setting
  ([TROUBLESHOOTING #143](docs/TROUBLESHOOTING.md#143-under-torch-215-an-actor-dies-when-it-builds-its-first-process-group),
  [VALIDATION](docs/VALIDATION.md#torch-2-15),
  [INSTALL](docs/INSTALL.md#install-from-source)).
- The docs now say what the code does where an outside review of 2026-11-05
  found them wrong. The support matrix marks cfg2 refused for PixelDiT / PiD,
  Wan SCAIL Preview and Wan SCAIL2, whose adapters refuse it class P for every
  file, and says Z-Image's cfg2 trims unequal prompts though only equal-token
  prompts were compared. DESIGN says a supervision failure retires the whole
  fleet, as the runtime does. INSTALL, Studio and
  [TROUBLESHOOTING #132](docs/TROUBLESHOOTING.md#132-dgxm-studio-up-refuses-or-the-always-on-studio-does-not-start)
  name 1.3.0 as the first release that reads `[studio]`. The INT8 GEMM records
  give the measured 0.92 to 1.20 range at 1024 rows and call the H3 audio
  difference a likely decode effect, not a finding; the SCAIL-2 cfg2 record
  drops its claim that every other family's cfg2 is bit-identical; INSTALL
  names the specifier that admits safetensors `0.8.0rc0`; and THREAT_MODEL
  says how reviewed timings reach the benchmark records while a gate
  conclusion carries none ([MODELS](docs/MODELS.md#support-matrix),
  [DESIGN](docs/DESIGN.md#11-release-quality-bar)).
- The torchmonarch canary blocks each snapshot whose private ownership probe
  fails, so a pin and a latest stable that fail it the same way no longer
  compare clean ([TROUBLESHOOTING #27](docs/TROUBLESHOOTING.md#27-torchmonarch-canary-reports-compatibility-or-semantic-drift)).
  The comfy canary's template widget check also fails a template whose
  dynamicPrompts widget holds text the browser would rewrite on Queue.
- README, STUDIO, STUDIO_ACCEPTANCE and DESIGN give the inventory's current
  counts, 73 workflows, 21 nodes and 143 declared inputs, and a test ties them
  to the generated inventory. The acceptance record adds the browser round of
  2026-11-03 and keeps the gaps it did not repeat.

### Rendering

- A LoRA that patches a weight ComfyUI renames at load now bakes under FSDP.
  A FLUX checkpoint stores a q/k norm weight as `<...>_norm.scale` and the
  model holds it as `<...>_norm.weight`; a LoRA with diffs for those weights
  (the FLUX.2 Klein 9B distill LoRA carries 27) stopped a `*+fsdp` render in
  the Worker with a bare `no checkpoint tensor for key` error on 2026-11-11.
  The shard bake now finds the tensor under the file's name, from ComfyUI's
  own rename, and verifies its bytes as for any key. That holds for a family
  whose config also builds weights as it renames (PixelDiT strips `net.` and
  splits a modulation weight in two): the keys it only moves are found, and a
  weight it builds from part of a tensor is not in the file. A patched key the
  file holds under no name refuses with a class P sentence before any shard is
  written. With `lora_low_rss` on and no FSDP the same keys used to stay as
  host copies behind a warning that they "did not verify against the
  checkpoint"; they are now restored from the file like every other key, and
  the warning is gone.
- A Worker that ComfyUI loads only partly now computes a LoRA's patched weights
  as its full load does. ComfyUI's classic patcher gives every patched module
  it leaves off the device a cast-time patch, and that patch is the full
  load's weight only for a plain weight above 16 KiB stored in the forward's
  dtype. Every other patched key (a module of 16 KiB or less such as a norm
  weight, a weight stored in another dtype than the forward such as a plain
  FP8 file, a quantized Linear) now takes the weight the full load bakes when
  its layer casts. On GB10 the old patch rounded a BF16 norm weight with a
  diff apart on about 7 percent of its values, and a patched INT8 ConvRot
  layer's output landed about 0.01 NRMS from the full load's. On an FP8, NVFP4
  or MXFP8 file the weight is not enough: ComfyUI's classic forward leaves a
  patched layer's input unquantized in that state. With `lora_low_rss` off, a
  one-GPU render that would run such a layer off the device now stops before
  sampling with a class C message that names `lora_low_rss` on (the Spark
  default, which was never affected); it used to render with another output
  than a full load and say nothing. A GPU with memory of its own keeps the
  baked weight in host RAM instead, where there is room, and renders. Found by
  reading and held by CPU tests against ComfyUI's own patcher; no hardware
  render isolates any of it, and the kept path cannot run on a Spark.
- A forced `family_adapter` no longer binds a model its adapter refuses under
  `auto`. This changes what a forced family does. Before, forcing bound by
  class ancestry alone: `wan` took the Wan VACE, Camera, HuMo, S2V, Animate,
  RelayVSR and CausalAR models, `hunyuan` took HunyuanVideo 1.0 and its two
  image-to-video models, and `zimage` took Lumina 2.0 and Ming-Image. Under
  sequence parallelism the family's sharded forward then replaced the
  variant's own and dropped the inputs it did not know, so a render could be
  wrong with no error. A forced family now binds a model exactly when its
  adapter would take the model's nearest ComfyUI class under `auto`. A class a
  custom node derives from a supported one still binds; the models above
  refuse class P, as they do under `auto`
  ([TROUBLESHOOTING #154](docs/TROUBLESHOOTING.md#154-a-forced-family_adapter-refuses-a-model-its-adapter-does-not-take-under-auto)).
- An extra tensor in a checkpoint no longer moves the loader's header
  refusals. Before, one tensor named like another family's key
  (`aux.dec_net.x.weight` in a Z-Image L2P file, `aux.audio_patch_proj.weight`
  in a MiniMax-H3-Image file with no sigma grid, or a key of any earlier row of
  the detector's table) made the header detector read another family, and the
  file passed the gate while ComfyUI still built the same model from it. The
  gate now names both families by the exact keys ComfyUI's own detection tests
  ([TROUBLESHOOTING #91](docs/TROUBLESHOOTING.md#91-a-z-image-render-refuses-with-an-upstream-gate-or-takes-the-host-down)).
- A damaged safetensors header no longer turns the loader's header refusals
  off. Before, a header the strict reader refused (a tensor entry or metadata
  key written twice, an empty tensor name, tensors sharing bytes, metadata
  that is not all strings) got no verdict, so a Z-Image L2P file, a file whose
  weight scales ComfyUI drops or a MiniMax-H3-Image file loaded as if the gate
  were not there, while stock ComfyUI still read it. The gate now reads such a
  header again as ComfyUI's own reader does, and the driver and every Worker
  refuse it with the class and sentence its intact header gets. A
  MiniMax-H3-Image `config` nested past the JSON parser's limit, which also
  got no verdict, now counts as no sigma grid, and a `_quantization_metadata`
  of JSON `null`, which ComfyUI reads as a table that names no layer, no
  longer turns the dropped-scale refusal off. A file no stock reader can read
  keeps ComfyUI's own load error, and a header read that ends with no verdict
  on the file refuses class P
  ([TROUBLESHOOTING #152](docs/TROUBLESHOOTING.md#152-a-render-refuses-because-the-header-gate-could-not-read-the-checkpoints-header)).
  A forced `family_adapter=minimax_h3` no longer binds the MiniMax H3 video
  adapter to a MiniMax-H3-Image model
  ([MODELS](docs/MODELS.md#running-finetunes)).
- UltraTex, the FLUX.2 Klein 9B LoRAs that ComfyUI 0.45.0 paints a mesh
  texture with, runs on one GPU, uly2 and uly2+fsdp. Its conditioning tiles a
  sample's views into one image and drops background tokens inside stock's
  Flux forward, and on CPU against ComfyUI 0.45.0 the uly2 forward returns
  stock's bits in the Flux 1, Flux 2 and LongCat layouts: with the view grid
  alone, the token masks alone, a view reference on a tile away from the front
  and all three together, batch 1 and 2. The FSDP wrap runs the token-mask step
  with nothing to gather. cfg2, dp2, ring
  and any degree above uly2 split a sample's views or are not stock's math, so
  a Flux, Flux 2 or LongCat render with UltraTex conditioning refuses there
  before dispatch, and auto skips the cfg rows for it. No hardware render has
  run yet ([MODELS](docs/MODELS.md#model-ultratex),
  [TROUBLESHOOTING #148](docs/TROUBLESHOOTING.md#148-a-render-with-ultratex-conditioning-refuses-on-cfg2-dp2-ring-or-a-wider-mesh)).
- MiniMax-H3-Image, which ComfyUI 0.45.0 added, is detected from its header
  as `minimax_h3_image` and runs on its own sharded adapter. A file
  of the family whose quantization metadata cannot be read refuses class P at
  the loader, before any load
  ([TROUBLESHOOTING #150](docs/TROUBLESHOOTING.md#150-a-minimax-h3-image-file-refuses-because-its-quantization-metadata-cannot-be-read)).
  A file of the family whose config metadata has no sigma grid
  (`transformer.pdd_sigmas`), which stock ComfyUI fails to build, refuses
  first ([TROUBLESHOOTING #149](docs/TROUBLESHOOTING.md#149-a-minimax-h3-image-file-refuses-because-its-header-has-no-sigma-grid)).
  The family's BF16 file reads `bf16` (its 51 F16 adaLN linears are islands),
  and its INT8 ConvRot and W4A8 files read `int8` and `w4a8`. MiniMax H3 video
  and p-h3-image files read and render as before. For every family, the FSDP
  launch check no longer counts a checkpoint's uint8 attention pins
  (`<block>.comfy_attention.config`), so a bf16 or fp16 file that pins proves
  its precision where it refused class P. On every `+fsdp` preset the shard
  build then clears those pins: the Init widget's kernel runs, no card asks,
  and the record says dropped, as ADAPTERS describes. No listed bf16 or fp16
  file pins except MiniMax-H3-Image's, which the loader refuses first.
- A change in behaviour in a world of one GPU (one Spark by itself, or
  `mode=local` with one GPU): a checkpoint's own attention pins now run as
  stock ComfyUI runs them, with no Sol-Attn waiver card, no stamp and no
  ledger row, and the first-use ceremony runs every leg with them, slab
  residency included. The FLUX.3 Klein and MOSS-Image INT8 ConvRot files (the
  SFT, Base and Kromoss checkpoints) render there with their publisher's INT8
  attention as stock does; on uly2 they render with exact attention, which
  stock cannot do with those files. The two differ a little, and the render
  record says which ran. A pin a LoRA sets, restates or changes still asks in
  a world of one GPU, as does a live attention module that differs from the
  checkpoint's. A `single` preset on a two-rank mesh resolves to dp2 on the
  Worker and keeps the card. cfg2, dp2 and their `+fsdp` forms still ask for
  the waiver wherever the pins run; a bf16 or fp16 file under `+fsdp` renders
  with its pins dropped and recorded, with no card ([TROUBLESHOOTING
  #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)).
- The render record's pin line says "kept" where it said "ran": the record
  reads the pins from the load and the topology and does not count attention
  calls. The field names are unchanged.
- Fix: the Workers prepare Anima's text as stock ComfyUI does, so a Worker's
  one-GPU Anima render is expected to equal stock's. Stock runs a prompt in
  inference mode, where Anima's `extra_conds` runs the text adapter and pads
  the text to 512 rows once per conditioning. A Worker samples outside it, so
  it took the branch ComfyUI keeps for training: it ran the adapter on every
  model call and split a prompt pair whose rows differ into two batch-1 calls
  where stock makes one batch-2 call. On 2026-11-10 the shipped graph's
  one-GPU Worker render read one-step NRMS 0.028 against stock, which this
  difference is expected to explain. The adapter now runs stock's method in
  inference mode on every load, as FLUX.3's and MOSS-Image's do, and on the
  CPU the Worker's model calls now equal stock's; a canary seam lists every
  inference-mode branch in ComfyUI's source. The hardware comparison against
  stock has not run yet
  ([TROUBLESHOOTING #151](docs/TROUBLESHOOTING.md#151-an-anima-render-on-the-workers-differs-from-stock-comfyui)).
- Change: Anima under `auto` renders on uly2 at every size, where it took cfg2
  below 1.2 MP on BF16. With the fix one GPU folds the prompt and the negative
  into one batch-2 call, as stock does, and cfg2's two batch-1 calls are not
  that call, so `auto` table row 47 is withdrawn; explicit cfg2 still renders
  and is not the same math. The shipped Anima template sets `auto`, so it
  now renders on uly2. The speed cost against cfg2 is NOT RUN on hardware.
- uly2 renders of Flux 1, Flux 2, Chroma, LongCat and Qwen Image 2.1 fill each
  fused-norm call of their sharded blocks to the rows one GPU normalizes, as
  FLUX.3 and MOSS-Image already did. comfy-kitchen's fused AdaLN picks its
  kernel by a call's rows, and at 512 square a rank's call fell on the other
  side of that switch: on 2026-11-07 uly2 there read one-step NRMS 0.006 to
  0.062 against one GPU, while the shipped 1024 square stayed bit-identical.
  Sizes below about 1024 square are expected to match one GPU exactly now; a
  hardware re-read is pending
  ([fused AdaLN switch](docs/VALIDATION.md#fused-adaln-switch-other-families)).
- LTX under sequence parallelism runs the two row-count-keyed calls in each
  block, comfy-kitchen's fused AdaLN and the per-head gate projection, at one
  GPU's row count, and the sharded self-attention applies RoPE through the call
  stock makes. On 2026-11-07 LTX 2.3 on uly2 read one-step latent NRMS 0.012 to
  0.014 against one GPU at every geometry tried, under the 0.10 floor but not
  identical, where Wan 2.1 on the same build was identical; an op probe traced
  it to the gate projection at every geometry and the fused AdaLN at 1024 rows
  a rank, on torch 2.15 and 2.12 alike. Each rank now does
  stock's row count of work for those two calls. At batch 2 each rank's share
  of a stream, and a padded attention's output, now reach the next linear
  contiguous, as stock's do, so its bias is added in the same fused call. A ComfyUI whose LTX blocks no
  longer read the globals this binds refuses a sharded render at load, class P,
  and canary seam 51 names the change ([VALIDATION](docs/VALIDATION.md#evidence-ltx-2-3-t2v)).
- `auto` takes cfg2 on Boogu FP8 below 1.2 MP and on Chroma below 1.64 MP
  only when the prompt and the negative encode to different text rows, as on
  the shipped graphs, and Ulysses for an equal pair; Anima's row had the same
  scope until its withdrawal (the Anima entry above). One GPU folds
  an equal pair into one batch-2 call, which cfg2's two calls do not
  reproduce: on 2026-11-07 such pairs read one-step NRMS 0.014 to 0.042
  against one GPU on Anima and Boogu, and no full schedule matched, while the
  shipped pairs stayed bit-identical; Chroma BF16 had read NRMS 0.034 on an
  equal pair. The sampler nodes tag the pair on the latent, and the render's
  telemetry records it as `pair_differs`. Explicit `cfg2` on an equal pair
  still renders. At world 4 an equal pair on these rows no longer refuses at
  batch 1: it takes the family's Ulysses row, folded into `uly2+ring2`, and
  at batch 2 `uly2+dp2`
  ([cfg2 rows](docs/VALIDATION.md#campaign-k-equal-pairs)).
- Sampler Custom renders run the first-use identity gate, as KSampler renders
  do. The six LTX graphs and p-h3-image used to render on stock residency until
  someone ran the Identity Gate node, and refused on a `*+fsdp` preset. Now the
  first render of a new combination runs the check on its own call: the same
  guider, sampler, noise and seed, on the first two steps of its sigma schedule.
  It records the verdict, the render follows, and later renders skip the check.
  A graph with no LoRA, whose family slab residency does not cover, records
  INCONCLUSIVE with nothing to compare, as a KSampler graph does; a LoRA,
  explicit slab, comfy-managed residency or a `*+fsdp` preset gets a real
  comparison ([TROUBLESHOOTING #61](docs/TROUBLESHOOTING.md#61-the-first-render-took-minutes-and-the-next-one-took-seconds)).
- A dual-model render (Ideogram 4) runs the first-use gate over both of its
  models. The proof is the dual render itself, and its PASS binds both
  checkpoints and LoRA stacks under a combination key of its own, so it grants
  only that pair and a single-model PASS never grants it; the Worker checks
  both before it uses the grant. On cfg2, where each rank holds one of the two
  models, a dual render keeps stock residency. On a `*+fsdp` preset it still
  refuses under `auto_gate=first_use`, now because the clean-reload proof
  reloads one model, and comfy-managed residency still refuses it. A dual check
  that cannot finish turns slab and low-RSS off for that pair alone, and a FAIL
  recorded for the pair or for its conditional model turns them off for the
  dual render ([TROUBLESHOOTING #61](docs/TROUBLESHOOTING.md#61-the-first-render-took-minutes-and-the-next-one-took-seconds)).
- A model on an explicit topology loads once when its first render runs the
  identity gate. The UNET loader placed it on every rank, the gate dropped it
  a moment later, and the first proof render loaded the file again: about half
  a minute to a minute and a second pass over the file per rank, and on
  2026-11-06 a capacity
  refusal for the Kandinsky 5 Pro template, priced while the first copy's host
  memory still counted. The loader now leaves the load to the gate's first
  proof render when the gate will run. A graph with a BasicScheduler (DGX
  Monarch) still loads twice, because the scheduler loads the model to compute
  its sigmas before the gate; no shipped template uses one. The gate, its
  verdict and the render are unchanged ([TROUBLESHOOTING #61](docs/TROUBLESHOOTING.md#61-the-first-render-took-minutes-and-the-next-one-took-seconds)).
- A LoRA renders as it does in stock ComfyUI. The Workers baked a LoRA with
  ComfyUI's classic merge, the math stock runs only with DynamicVRAM off,
  while stock's default applies a LoRA when a layer casts its weight, in the
  forward's dtype and with requantization seeded by the module's name. The
  pruned INT8 MiniMax H3 file with the p-h3-image LoRA read NRMS 0.077 against
  stock at one step and 0.257 at 8 steps, and Kromoss BF16 with a DMD LoRA
  read 0.0365 to 0.0635 at one step. Every bake now computes the weight stock's cast
  computes; a probe found it bit-identical to stock on the file's INT8 ConvRot
  and BF16 layers. MOSS-Image INT8 with a LoRA still refuses (#129) until a
  hardware run measures it
  ([TROUBLESHOOTING #145](docs/TROUBLESHOOTING.md#145-a-lora-render-on-a-worker-differs-from-stock-comfyui)).
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
  under each setting. On 2026-10-31 the FP8 matmul path rendered plain FP8
  files as noise on GB10, in the actors under `"on"` and in the driver under
  `--fast fp8_matrix_mult` alike, though it sampled 24 to 30 percent faster,
  so `"on"` is unsafe there and the default stays `"auto"`. FP8 speed on GB10
  comes from comfy-kitchen files: the Krea2 Turbo fp8_scaled file sampled in
  65 s against 73 s for BF16 at one-step NRMS 0.068, while a BF16 file cast to
  `fp8_e4m3fn_fast` drifted past the 0.10 floor
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
  Triton or a validated comfy-kitchen release (0.2.37, 0.2.40 or 0.2.41) runs
  CUTLASS and logs one WARNING ([Models](docs/MODELS.md#int8-gemm-kernel),
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
  The activation charge runs 1.4 to 2.2 times the pair's measured activation
  and 1.9 to 3.3 times one rank's
  ([VALIDATION](docs/VALIDATION.md#h3-capacity)).
- The MiniMax H3 activation preflight is refitted to five hardware points and
  Config A's measured use: it charges each rank an 11.0 GiB floor and 129,024
  bytes per row on the rank, in place of 1,032,192 bytes per row with no
  floor, and keeps 73,728 bytes per row of the whole sequence. On the pair it
  admits the 15 second clip's second request in a session and the gate's slab
  path, which the old charge of 60.1 GiB per rank refused against the 17.6 GiB
  the render measured. The one-rank BF16 shape that the 2026-08-05 kill made a
  refusal rendered on 2026-10-31 beside the driver, holding 5.5 GiB, so the
  preflight now admits it, and it admits the 15 second clip on one Spark with
  36.7 GiB to spare. Each completed render clears by 6.5 GiB or more, and the
  charge stays above what every measured render used
  ([VALIDATION](docs/VALIDATION.md#h3-capacity),
  [TROUBLESHOOTING #56](docs/TROUBLESHOOTING.md#56-minimax-h3-render-refused-activation-footprint-preflight)).
- The refitted price passed its hardware checks on 2026-10-31 with both
  driver preflights on. The 15 second MiniMax H3 clip rendered on one Spark
  in both layouts: one rank on the sibling alone in 99:31, and one rank on
  the head beside the driver in 105:51, where the earlier charge refused it
  at the loader. The 61.7 GiB BF16 file rendered at 5 seconds as one rank beside
  the driver through the capacity rescue in 22:30
  ([VALIDATION](docs/VALIDATION.md#activation-across-clip-lengths-measured-2026-10-29-and-30),
  [BENCHMARKS](docs/BENCHMARKS.md#minimax-h3-speed-and-memory-by-clip-length-2026-10-29-to-2026-10-30-head-capped)).
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
- MiniMax H3 Context Windows (DGX Monarch) carries ComfyUI 0.44.0's MiniMax
  H3 context windows (#17425) to the Workers. It takes the stock node's
  widgets, and each Worker applies ComfyUI's own node to its render clone, so
  each window is one model call on a shorter clip, which Ulysses splits as it
  splits a whole clip. The sample deadline counts every window's rows. It
  refuses class P on a Worker whose ComfyUI predates 0.44.0, on another
  family and on a two-checkpoint guider's negative model. The activation
  preflight still prices the whole clip. A comfy canary seam holds the stock
  contract, and no windowed render has run on hardware
  ([MODELS](docs/MODELS.md#model-minimax-h3-t2va-fl2va-ref2va-av),
  [VALIDATION](docs/VALIDATION.md#minimax-h3-context-windows),
  [TROUBLESHOOTING #142](docs/TROUBLESHOOTING.md#142-a-render-with-minimax-h3-context-windows-dgx-monarch-refuses)).
- A capacity-rescue consent now keeps slab residency when stock residency
  cannot load the checkpoint. The identity gate records INCONCLUSIVE with
  `cross_mode=CAPACITY`, takes no lever, and the render proceeds under slab.
  On 2026-10-30, before this change, the 61.7 GiB MiniMax H3 BF16 file was
  refused under stock. Since the change it has rendered on one Spark under
  `DGXM_AUTO_RESCUE=1` (2026-10-31), and on the pair and on one Spark under
  `DGXM_ALLOW_SLAB_RESCUE=1` and under the panel's card (2026-11-04). The pair
  under `DGXM_AUTO_RESCUE=1` and the Auto-allow toggle have no render record
  ([VALIDATION](docs/VALIDATION.md#bf16-capacity-rescue-2026-10-30-a-recorded-limitation)).
  `DGXM_ALLOW_SLAB_RESCUE=1`, `DGXM_AUTO_RESCUE=1` and the Auto-allow
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
  real ComfyUI. On the pair, MiniMax H3, Wan 2.1 and LTX 2.5 passed the
  hardware byte compare, and the other families are unchecked
  ([Models](docs/MODELS.md#worker-assisted-vae-decode),
  [VALIDATION](docs/VALIDATION.md#worker-assisted-vae-decode),
  [TROUBLESHOOTING #140](docs/TROUBLESHOOTING.md#140-a-vae-decode-did-not-split-over-the-workers)).
- A Wan 2.2 TI2V 5B template, `dgx-monarch-wan22-ti2v.json`, runs the one
  FP16 checkpoint with the Wan 2.2 VAE at the settings of ComfyUI's own TI2V
  template: 1280x704, 121 frames at 24 fps, 20 uni_pc steps at cfg 5 and
  shift 8. auto runs it on uly2. A Load Image wired to `start_image` makes it
  image to video, and the Wan adapter shards the per-frame timesteps that
  image brings. Its VAE splits under Worker-assisted VAE decode through VAE
  Decode (Tiled), and on CPU the real file splits byte-identical to the stock
  tiled decode. No hardware comparison has run, so the row is impl
  ([MODELS](docs/MODELS.md#model-wan-2-2-ti2v-5b),
  [VALIDATION](docs/VALIDATION.md#evidence-wan-2-2-ti2v-5b)).
- p-h3-image, PrunaAI's 8 MP image model on MiniMax H3 (ComfyUI #17506),
  runs on the MiniMax H3 adapter: ComfyUI builds its DiT as MiniMax H3. The
  `dgx-monarch-p-h3-image.json` template follows the vendor's text-to-image
  workflow on the BF16 DiT, text encoder and VAE, with the prompt upsampler,
  the 8-step grid and both shifts at 1.0. ModelSamplingMiniMaxH3 (DGX
  Monarch) carries ComfyUI's ModelSamplingMiniMaxH3 to the Workers, which
  apply the stock node to the render clone; it refuses class P on another
  family, and a one-frame image latent that no H3 shift reaches refuses
  rather than sample at 12.0 and 3.0. The W4A8 file reads as `w4a8`, not
  `fp8`, and FSDP refuses it. On CPU the H3 sharded forward returns stock's
  bits on the image graph; no hardware has run it, so the row is impl. Studio
  names the text that reaches a Basic Guider the prompt. MODELS also
  documents the Sol-H3-VAE decoder, which decodes on the driver
  ([MODELS](docs/MODELS.md#model-p-h3-image),
  [VALIDATION](docs/VALIDATION.md#evidence-p-h3-image),
  [TROUBLESHOOTING #144](docs/TROUBLESHOOTING.md#144-a-p-h3-image-render-or-modelsamplingminimaxh3-dgx-monarch-refuses)).
- Worker-assisted VAE decode keeps three ComfyUI 0.44.0 decodes on the
  driver, with one line each: the Sol-H3-VAE decoder, which decodes the whole
  clip in one pass, the p-h3-image VAE, which decodes one frame, and a VAE
  whose compute dtype the VAE Compute Dtype node set, which a Worker cannot
  build the same way ([TROUBLESHOOTING #140](docs/TROUBLESHOOTING.md#140-a-vae-decode-did-not-split-over-the-workers)).
  [MODELS](docs/MODELS.md#model-minimax-h3-t2va-fl2va-ref2va-av) says what
  this project does with ComfyUI 0.44.0's p-h3-image model, Sol-H3-VAE
  decoder and MiniMax H3 context windows, and
  [VALIDATION](docs/VALIDATION.md#comfyui-0440-additions) lists the
  behaviours no hardware has run.

### Operations

- A Worker logs one line when it accepts an identity gate grant, naming the model
  slots the grant covers, for example `identity gate grant accepted on rank 1: the
  gate PASS covers both model slots, [cond] ideogram4_fp8_scaled.safetensors with 1
  LoRA and [uncond] ideogram4_unconditional_fp8_scaled.safetensors with no LoRA`.
  A dual-model render's grant over both slots can now be checked from a Worker
  journal. The line names checkpoints as the model store's lines do and counts
  LoRAs without naming them; a single-model grant logs the same line with its one
  slot.
- `dgxm doctor` adds an `INT8 GEMM` row for this box and one per configured
  host: the configured kernel and whether the Python headers, Triton and a
  validated comfy-kitchen are there. The rows warn and never fail
  ([TROUBLESHOOTING #138](docs/TROUBLESHOOTING.md#138-the-log-shows-an-int8-gemm-warning-and-int8-renders-run-stock-cutlass)).
- Worker services keep torch inductor's compile cache in
  `~/.cache/dgx-monarch/inductor`, so `compile_dit` no longer repeats its
  autotune after a reboot; set `TORCHINDUCTOR_CACHE_DIR` to place it elsewhere ([TROUBLESHOOTING #14](docs/TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on)).

- After a worker death and a timed-out stop, a host whose process table
  cannot be listed answers unknown and keeps the replacement latch. It used to
  count nothing and answer gone, which released the latch. `dgxm reap` and the
  lifecycle sweeps report such a host as unaccounted for, and doctor's orphan
  row says unobserved ([TROUBLESHOOTING #87](docs/TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)).
- `dgxm doctor`'s Worker service row reads unobserved, not FAIL, when the host
  answered but could not tell whether the loop runs or listens, so doctor exits
  75 and `dgxm update --verify` keeps the new release instead of restoring the
  old one. A listener check whose `awk` is missing or fails reads unknown, not
  closed, so auto-heal no longer restarts a running fleet on it
  ([TROUBLESHOOTING #1](docs/TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout)).
- `dgxm doctor` reads only a `last` and an `nvidia-smi` that succeed: a failed
  run, or `nvidia-smi`'s own error text, reads unobserved instead of a clean
  boot history or a GPU process. A worker host whose `/proc/swaps` is
  unreadable or malformed reads unobserved instead of swap off, and one that
  cannot read its fabric state in sysfs reads unobserved instead of failing.
- `dgxm doctor`'s `FP8 matmul (actors)` row warns when `fp8_matrix_mult = "on"`
  and a host reports a GB10 GPU, the case MODELS.md documents as unsafe, and
  names those hosts ([TROUBLESHOOTING #141](docs/TROUBLESHOOTING.md#141-fp8-renders-run-no-faster-than-bf16)).
- `dgxm doctor --repair` records a mode change that ran but could not be
  confirmed as `unknown` with one change, in a `partial` receipt, instead of
  saying nothing changed ([INSTALL](docs/INSTALL.md#operator-receipts)).
- `/dgxm/metrics` writes each series once on a host with more than one GPU:
  the GPU series and the Worker actor's counters carry `worker_index`, and the
  memory and rail series appear once per host ([CLUSTER](docs/CLUSTER.md#metrics-reference)).
- An actor ledger whose actor list is not a list reads as corrupt and empty,
  and a row with an infinite, NaN or fractional pid or start time is dropped,
  instead of raising out of the liveness check and the sweep.
- `dgxm setup` accepts the same `--artifact` twice: the plan it prints and the
  plan it validates now count the paths once.
- `dgxm setup`'s cleanup of a reserved release slot refuses, and keeps the
  slot, when its site holds anything beside the package.
- The desktop entry quotes paths for both of the Desktop Entry format's escape
  layers, so a path with a backslash, a quote, a backtick or a dollar sign
  reaches the launcher unchanged, and the icon path keeps its backslashes.
- `scripts/setup_env.sh --comfy` stops with the repair when torchaudio is
  installed but does not import, whatever the ComfyUI version, and checks a
  donor copy imports before it reports done.
- Setup profiles say what native return's qualification covers, CPU-staged
  returns on one selected rail, and what it does not, instead of saying it
  awaits requalification; they keep it off.
- Workers and the driver give freed host memory back at once: the worker
  loop sets `MIMALLOC_PURGE_DELAY=0` for every cluster actor, the driver's
  launch for a local actor, and `scripts/comfy-driver.sh` exports it, because torch 2.15.0's bundled mimalloc 2.5.2 purges so late
  that an actor kept 15 to 18 GiB of host memory after a stock load and the
  next load was priced short ([TROUBLESHOOTING #146](docs/TROUBLESHOOTING.md#146-a-worker-keeps-a-dropped-models-host-memory-and-the-next-load-is-priced-short)).

### Rendering

- Before a render-site capacity check refuses, the driver unloads the models
  ComfyUI holds on its host, such as the last render's text encoder and VAE,
  and checks again; set `DGXM_DISABLE_DRIVER_UNLOAD=1` to turn this off ([TROUBLESHOOTING #56](docs/TROUBLESHOOTING.md#56-minimax-h3-render-refused-activation-footprint-preflight)).
- Three fixes from the 2026-11-04 release test. The identity gate's stock
  comparison frees what the driver host can reload before it gives up, so the
  Flux 2 template renders on a fresh pair instead of refusing with its 33 GiB
  text encoder on the head ([TROUBLESHOOTING #66](docs/TROUBLESHOOTING.md#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)).
  A LoRA change on a base a Worker already holds is priced as the in-place
  swap the store makes, without a second copy of the file, so the Wan-Animate 2
  memory-saver template no longer refuses under an explicit topology. And
  where a model call is sharded, every rank batches cond and uncond the way
  the leanest rank does, so two boxes with different free memory no longer
  stall on mismatched collectives
  ([TROUBLESHOOTING #79](docs/TROUBLESHOOTING.md#79-a-cross-spark-render-refuses-because-the-ranks-disagree-about-weight-residency)).
- A data-parallel render keeps its progress bar and stall guard alive until
  the last group finishes. Each group's result leader sends its own done, and
  the driver stopped reading at the first one, so a slower group's steps were
  dropped and a gap longer than `DGXM_SAMPLE_STALL_TIMEOUT_S` marked a healthy
  mesh stalled ([TROUBLESHOOTING #86](docs/TROUBLESHOOTING.md#86-a-render-error-says-the-fleet-stalled-or-a-worker-refuses-with-a-poisoned-cuda-context)).
- The loader's early batch check reads only the samplers that take its model.
  A dp2 mesh feeding the Fleet KSampler with a batch of one, or a batch-one
  branch on another loader, no longer refuses a valid load; a sampler the
  check cannot tie to the loader gets its answer at render submit
  ([TROUBLESHOOTING #16](docs/TROUBLESHOOTING.md#16-single-preset-refuses-because-the-latent-batch-does-not-divide-dp)).
- Chroma's cfg2 trim removes only the equalizer's hard padding. A custom
  conditioning whose attention mask carries a finite bias on some text tokens
  keeps those tokens and its mask, as a one-GPU render does; before, the trim
  dropped them and the mask.
- Gate authority holds every box to the driver's torch and CUDA build. Each
  Worker reports its build at setup and in the Gate ceremony's provenance, and
  checks a grant's build against its own. A pair on two builds refuses a risky
  render and names each box's build, where before a Worker recreated with only
  a new torch reused a PASS proven under the old one. A matched pair keeps its
  PASS rows ([TROUBLESHOOTING #8](docs/TROUBLESHOOTING.md#8-version-skew-between-hosts)).
- A sharded FLUX.3 render refuses at load, class P, when the installed
  ComfyUI's block no longer reads the norm global the parity route binds; it
  used to warn and render with a possible bf16 difference. A topology
  without sequence parallelism, such as `single`, still renders ([TROUBLESHOOTING #121](docs/TROUBLESHOOTING.md#121-a-flux3-checkpoint-fails-to-load-on-comfyui-older-than-0410)).

### Development

- `scripts/test_cpu.sh` runs the gate checks CI runs beside the suite before
  it: the protected test inventory first, then the refusal ledger, the TRUST
  citations, the leak check, ruff, mypy, compileall, the generator and artifact
  checks, CI's source archive census on a copy of the tracked files, and the
  shell and browser-asset checks. A local green run now covers every CI gate
  except the CI-only steps `tests/test_cpu_gates.py` names, where before the
  gates surfaced only after a push; CI's tools can still be other versions
  than yours. A gate is skipped and named when CairoSVG, the build module, a
  Git checkout, shellcheck or node is missing, `DGXM_SKIP_GATES=1` skips them
  all, and `tests/test_cpu_gates.py` holds every CI step to a local gate or a
  named CI-only reason. No gate writes a tracked file: the refusal ledger tool
  gains a check mode, `python tests/test_refusal_classes.py --check`, which
  compares in memory and prints any difference, and CI's step runs the same
  command where it used to rewrite the ledger and put it back.
- The sweep runner ends the session when a waived retry's queue will not
  clear, as it already did after every other leg, and when the cache reset
  before a waived, resident or probe leg leaves the queue busy or gets no
  answer. The next leg or the next cell's reset no longer queues behind work
  that may still be running, and such a record re-runs on resume
  ([SWEEP](docs/SWEEP.md#5-reading-results)).
- A session-close recycle that does not answer `ok` makes the sweep runner
  exit non-zero, and `run_summary_<session>.json` names its answer under
  `close_recycle`. The cell records stand. A recycle whose response cannot be
  read, a body that is not JSON or one cut short, is such an answer at session
  open, at close and at every reset, instead of ending the runner without a
  summary ([SWEEP](docs/SWEEP.md#4-running-a-scoped-sweep)).
- The sweep's audio compare reads every audio file of both legs and keeps each
  file's sample rate, channel count and length. Legs that differ in file
  count, rate, channels or length read CHECK instead of correlating what they
  share, and so does a file soundfile cannot decode, which used to end the
  session. The report lists audio results in a section of their own, and
  envelope correlation, which no calibrated threshold backs, changes no
  verdict ([SWEEP](docs/SWEEP.md#5-reading-results)).
- The comfy canary's MageFlow two-rank test stops at once when one rank fails,
  reports that rank's error and leaves no rank thread behind. The cond-batch
  agreement test kills and reaps every child process it started when a rank
  fails or times out. The Boogu key-order test reports a refusing rank's own
  error, not the broken barrier its peer was left with. The dual-model cfg
  test now hands its gather a non-contiguous view, so it fails if the
  production conversion goes.

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
