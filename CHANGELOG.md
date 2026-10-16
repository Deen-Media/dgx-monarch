# Changelog

## [Unreleased]

**Upgrading from 1.1.0.** The first update from 1.1.0 runs the 1.1.0 updater,
so the update checks this release adds apply from the next update on. If any host's
`~/.local/share/dgx-monarch/src` resolves into a release slot, as on the remote
hosts of every guided multi-node setup and on every host after a verified
update, update only with `dgxm update --verify`: plain update leaves those
Workers on 1.1.0, and
[TROUBLESHOOTING #118](docs/TROUBLESHOOTING.md#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)
is the repair. Turn `rdma_latent_return` off before updating, add the
`[monarch.rdma]` target afterwards, then turn it back on. Do not add a
`[monarch]` table before updating, because the 1.1.0 loader rejects it. Clear
`DGXM_RDMA_QP_SPLIT` from the Worker environments before updating, and the
other ambient variables
[TROUBLESHOOTING #119](docs/TROUBLESHOOTING.md#119-a-mesh-build-or-worker-start-refuses-on-an-ambient-environment-variable)
names before the first render. The 1.1.0 updater does not install torchmonarch 0.7.0's own
requirements (`xxhash`, `pyarrow>=17`, `numpy>=1.26`). Install `xxhash`, and
any of the others that `pip check` names, on the driver and every Worker host.
Until you do, `pip check` reports it and the next update in either mode
refuses. After a verified update, rerun the editable install once so its
metadata reads the new version
([INSTALL](docs/INSTALL.md#upgrading-from-110)).

- Support FLUX.3 Image Klein, the step-distilled and the base checkpoint, at
  status `impl` with no hardware test yet
  ([MODELS](docs/MODELS.md#model-flux-3-klein)). It needs ComfyUI 0.41.0 or
  newer, the first release to carry ComfyUI's support commit 20b46065e of
  2026-10-15
  ([TROUBLESHOOTING #121](docs/TROUBLESHOOTING.md#121-a-flux3-checkpoint-fails-to-load-on-comfyui-older-than-0410)).
  The `flux3` adapter shards the five image blocks and then the joint
  `[image, references, text]` stream in contiguous chunks, clips stock's
  modulation spans to each rank's rows, names pad rows on every topology, and
  keeps the text and reference blocks whole on every rank. Every Worker runs
  the base model's text and reference streams once per conditioning, as stock
  ComfyUI does. Four BF16 templates ship: text to image and reference editing
  for each checkpoint, at the vendor's settings (16 Euler steps at CFG 1 for
  the step-distilled model, 25 at CFG 3.5 for the base). The edit graphs
  express the vendor's Set Reference Multi Latent node in stock nodes with the
  same arithmetic. Six testing graphs cover the vendor's INT8 ConvRot and
  NVFP4 files, an odd grid and a two-reference edit, whose two 4:3 synthetic
  references `benchmark/sweep/media.py` now writes. The artifact manifest pins
  the six DiT files and the INT8 encoder at the gated vendor repository's
  commit; the BF16 encoder and the VAE reuse their pinned rows.
- A ComfyUI that stays up attaches again after a Worker service restarts
  under it (`dgxm restart`, a crash that systemd restarts, or a hand restart)
  when the service comes back within Monarch's 30 s message delivery timeout
  (closes [#2](https://github.com/Deen-Media/dgx-monarch/issues/2)). Each
  Worker start serves a new Monarch proc id, and the driver reads each host's
  id before every attach and dials it. The first attach to a restarted Worker
  waits about 2 s before it dials. After a longer outage the fallback is still
  a ComfyUI restart. An id the driver cannot read refuses before anything dials and
  names its remedy, and `dgxm doctor` fails the `listener generation` row of a
  Worker that publishes none. The id read waits up to 10 s per host, up from
  3 s. Upgrading: a driver on this release refuses Workers still on an older
  one before dialing, and a ComfyUI started before the update needs a restart
  once its Workers run this release. No unit, config or firewall change.
  CPU-tested and checked on the pair on 2026-10-15
  ([TROUBLESHOOTING #2](docs/TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)).
- A sampler render after the Worker services restart under a ComfyUI that
  stayed up attaches again within that render, instead of waiting out the
  30 s capacity price deadline and asking for Reset attached mesh (closes
  [#47](https://github.com/Deen-Media/dgx-monarch/issues/47)). The driver
  records each delivery failure its fault hook reports, and the price wait
  ends within half a second of one that names a fleet host and the mesh the
  price waits on; a rank that has not answered after 2 s is asked once more,
  because a restarted loop's old channel session loses the first request with
  no report. A render through KSampler, KSampler Advanced or Sampler Custom
  that owns its session and keeps the last render's setup, or an explicit
  loader's eager load, then reads every loop's id. When every loop serves a
  new id, it stops the stale mesh with a 2 s bound on each teardown wait,
  releases it on the liveness proof, attaches by the new ids and renders.
  Some loops moved and others not, no loop moved, an id it cannot read, or
  old processes not proven gone each refuse at once, class C, naming the
  remedy. The driver log prints `restart_reattach outcome=...`, the attach
  trace adds `event=restart_reattach`, and the render's finish record carries
  `reattach` and `reattach_s`. KSampler Pipeline renders and the Identity
  Gate node stop at the fast refusal, which names one KSampler render as the
  way to attach again; an automatic identity gate that meets the stale mesh
  aborts without pushing a stock policy to the gone actors. The Fleet node, a
  one-host fleet and a render whose setup changed meet the old actors as
  before. After an outage past the delivery timeout the fallback is still a
  ComfyUI restart. CPU-tested on 2026-10-15; the two-Spark recheck is pending
  ([TROUBLESHOOTING #2](docs/TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)).
- Fix `benchmark/run_matrix.py` for renders under an accuracy waiver. The
  samplers then return a dict with `ui` and `result` entries, which the
  runner unpacked as a tuple, so each such cell rendered and was then recorded
  as N/A. The runner now unwraps the return and lists the waiver dispatch ids
  in the row's new `waiver_runs` field; other rows are unchanged.
- Label a cfg topology on a CFG 1 graph `refuse:P` under FSDP as well. The
  sweep matrix predicted an untyped refusal there, but the cfg wrapper's typed
  guard answers before the FSDP admission, as the FLUX.3 sweep of 2026-10-16
  showed on two cells (`cfg2+fsdp` at CFG 1 under plain Euler). A cfg++
  sampler still renders.
- Refuse a checkpoint whose weight scales ComfyUI would drop, before anything
  loads
  ([TROUBLESHOOTING #120](docs/TROUBLESHOOTING.md#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales)).
  The upstream gate replays, on the header's key names, the conversions and
  the one prefix strip of ComfyUI's diffusion-model load. It then reads, per
  layer, whether a scale beside an fp8, fp4, int8 or packed 4-bit weight ends
  with a record ComfyUI reads: a `comfy_quant` key, an entry in the header's
  `_quantization_metadata`, or the record ComfyUI's legacy `scaled_fp8`
  conversion writes for each `scale_weight` it renames. Without one, stock
  ComfyUI drops the scale as an unexpected key, and an fp8 file such as
  `variants/fp8r/model.safetensors` of FLUX.3 Image Klein loads without an
  error and computes unscaled. The refusal is class P with no waiver, raised
  at the driver's loader preflight and again at the Worker's load funnel,
  both through `upstream_gate.raise_refusal`, from one header read per file
  in each process. `upstream_gate.refuses` becomes `refusal_for`, which
  returns the text and its TROUBLESHOOTING number, and the sweep matrix
  labels such a cell `refuse:P` through it. The comfy canary job checks the
  replay against ComfyUI's own load, key for key.
- Bake a LoRA key on a module the FSDP wrap replicates. The wrap keeps every
  module a family runs outside its diffusion forward whole on each rank (H3's
  `condition_proj` and `token_refiner`, LTXAV's caption projections and
  connectors, Anima's `llm_adapter`), and a LoRA key on one raised
  `UnbakeError` on every `*+fsdp` topology. The shard-aware bake now treats
  such a parameter as a world-1 shard: each rank reads the pristine weight,
  runs ComfyUI's merge and key-seeded rounding, and writes the whole result,
  which is the resident bake. The replicated set comes from the census call
  the wrap makes, and any other unsharded parameter still raises. A CPU test
  in the comfy canary job holds bit equality with ComfyUI's own merge; no
  hardware run covers a LoRA under FSDP on H3, LTXAV or Anima yet, so MODELS
  makes no claim for them.
- Add a slice-tuple case to the bake-equivalence canary: LoRA units routed
  into thirds of one fused qkv weight through `(key, (dim, offset, size))`
  entries bake to the same bits through the store's per-key bake and
  ComfyUI's `patch_weight_to_device`, in fp32 and bf16, and the rows outside
  a patched third keep their bits.
- Add a Doctor row, `<host> worker release version`, that reads the package
  version inside each host's release slot (`s-` from guided setup, `u-` from a
  verified update) and fails, naming the host, both versions and
  TROUBLESHOOTING #118, when it differs from the driver's. It parses the
  slot's `dgx_monarch/__init__.py` and needs no config key, so it also runs
  when the 1.1.0 updater starts this release's Doctor. A plain source
  directory, or guided setup's link to a colocated checkout, reads OK. The
  existing env row keeps its torchmonarch skew check.
- Plain `dgxm update` now reads every host's source link before it fetches and
  refuses with `release_slot_worker`, naming the host, when one resolves into a
  release slot, because plain update cannot refresh a slot. The message gives
  `dgxm update --verify --target <ref>` and TROUBLESHOOTING #118. A host whose
  link cannot be read (no SSH answer, or its configured Python failed) refuses
  with `release_slot_unknown`. Guided setup's link to a colocated checkout
  follows the pull and does not refuse.
- The torchmonarch skew warning for a Worker in a release slot now sends you
  to `dgxm update --verify`, not plain update, and names TROUBLESHOOTING #118.
  A plain host, or guided setup's link to a colocated checkout, keeps the
  plain-update advice, which now names cluster.toml's `[cluster]` `python`
  key.
- Document the ambient environment refusals that already stop a mesh build or
  a Worker start: `OTEL_EXPORTER_OTLP_ENDPOINT` while export is off, a
  `MONARCH_RDMA_*` name the policy does not set, and a policy variable such as
  `DGXM_RDMA_QP_SPLIT` set to another value. Each message now names its
  remedy and TROUBLESHOOTING #119, and CLUSTER names the three variables.
- `scripts/setup_env.sh --comfy` no longer fails on a ComfyUI that does not
  need torchaudio. It runs the donor step only when ComfyUI's
  `requirements.txt` lists torchaudio or its version is below 0.38.0, and
  otherwise prints a warning and continues.
- Correct the Kroma scope: a BF16 Kroma file reads as family krea2, so at CFG
  other than 1 (or under a CFG++ sampler) from 1.2 to 2.36 MP auto takes Krea2
  cfg2 row 15, which no Kroma run has tested. MODELS marks the Kroma cfg2 cell
  as selected and untested, the Krea2 rows say row 15 covers BF16 as well as
  FP8, and VALIDATION lists Kroma BF16 among the untested files that reach a
  cfg2 row. `uly2` keeps the tested path.

- Fill every pending row of `example_workflows/artifacts.toml`. 71 rows now
  carry a url pinned to a Hugging Face commit, with the file's size and
  sha256, and CI runs `tools/check_artifacts.py --strict`, so a row added
  without them fails. For 69 of them the vendor's sha256 at that commit
  matches the copy staged on the test host. The LTX 2.5 Gemma 4 encoder row
  pins the vendor's current revision: the vendor replaced the file in August
  2026 to align it with the LTX-2.5 checkpoints after the August ladders ran,
  so the four LTX 2.5 template shapes were rendered again with the current
  file at one denoise step against a stock single-GPU reference and passed
  the same gate ([VALIDATION](docs/VALIDATION.md#evidence-ltx-2-5-t2v-av-packed)).
  The two HunyuanImage 2.1 refiner files were downloaded again from the
  pinned commit and hash to their rows, which also match the full-file hashes
  a July 2026 acceptance run took of the earlier staged copies on both test
  hosts. The two CogVideoX 1.5 checkpoints become
  placeholder rows: the tested files were local ComfyUI-layout conversions of
  the diffusers-layout download, and no public copy matches them. Each row's
  note and template note name the source download and the layout to convert
  it to, and the troubleshooting entry now gives the full key rename. The LTX
  2.5 template note also says that a download needs a Hugging Face token from
  the account that accepted the license
  ([INSTALL](docs/INSTALL.md#model-files-required-by-each-template),
  [TROUBLESHOOTING #102](docs/TROUBLESHOOTING.md#102-a-cogvideox-checkpoint-reads-family-unknown-or-comfyui-will-not-load-it)).
- Detect, record and waive a per-module attention kernel pin where it runs,
  on topologies without sequence parallelism (closes issue #30). ComfyUI lets
  a checkpoint, or a LoRA with `.comfy_attention.config` keys, pin
  comfy-kitchen's Sol or int8 attention kernel on each attention module.
  Under sequence parallelism the pack already drops every pin; on single,
  cfg2, dp2, cfg2+fsdp and dp2+fsdp every stock call runs it as on one GPU,
  yet the render record named only the Init widget's kernel and the pack
  asked for no waiver. The worker now walks each fresh load once, before
  injection and before any LoRA is staged, keeps every `comfy_attention`
  module whose config names a kernel, and adds the pins a LoRA stages (the
  patcher's object patches) at each render; nothing runs per step
  (`attention_pins.py`). The render finish record gains `attention_pins`
  (pinned attention modules per kernel, Sol's tau in the name),
  `attention_pins_ran` and `attention_pins_dropped` beside the unchanged
  `attention` fact, so no combination gates again; the sidebar's last-render
  line, the driver log and the `benchmark/run_matrix.py` console show them,
  and run_matrix rows carry them. Under dual-model cfg2 the fields add both
  models' modules; when both models pin, `attention_pins_models` keeps each
  model's own fields and the line names each. Both kernels are approximate by
  construction (Sol pools the key blocks it does not route to; int8 rounds
  q, k, v and the attention weights to 8 bits), so a pinned one that would
  run refuses with class K under the new guard `sol_attn:checkpoint_pin`
  before any sampling, until the operator grants the existing Sol-Attn
  waiver
  ([TROUBLESHOOTING #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)).
  The card now says "an approximate attention kernel", names the pinned
  kernel, its module count and that a sequence-parallel topology drops the
  pin, and states no measurement for a pinned kernel. Every rank of a fleet
  reaches the same verdict from the pins, the LoRA stack and the topology,
  never from whether its host could bind the kernel, because cfg2+fsdp and
  dp2+fsdp run no readiness exchange. An fp8 or int8 checkpoint takes the
  FSDP direct wrap and keeps its pins. A bf16 or fp16 checkpoint whose own
  header pins refuses class P at the FSDP launch check before any shard
  build, because its uint8 config tensors leave the precision unproven. Under
  `auto_gate=first_use` a pin that runs on cfg2+fsdp or dp2+fsdp refuses even
  after the grant, because the Gate ceremony renders without waivers and FSDP
  has no stock residency to fall back to; uly2+fsdp, `auto_gate=off` or a
  topology without fsdp renders it. Under dual-model cfg2 the card names the
  conditional checkpoint, on which the driver keys every grant, and no
  dispatch's card can name a checkpoint an earlier dispatch bound. ComfyUI
  decodes the config with `Tensor.numpy()`, so slab residency and the
  low-RSS LoRA staging now keep the config tensor on the host, and slab
  residency prices a LoRA of configs alone as no bake.
  `tools/make_pin_probe_lora.py` writes a LoRA that pins a kernel on one
  family's attention modules (Krea2, Anima, Kandinsky 5, Wan 2.1 or Qwen
  Image 2.1) from a checkpoint's header, for a hardware check without a
  patched checkpoint. CPU tests cover detection on nested stand-in models, a
  LoRA's pin, the record and the refusal under each topology, the fp8 FSDP
  wrap, the dual-model cfg2 card and the cost; canary seam 39 loads a tiny
  real Wan 2.1 model with real config tensors, applies a real LoRA pin
  through ComfyUI's own patcher, pins the attribute and function names the
  detector reads, and checks the probe tool's patterns against tiny real
  models of all five families.
- Label the sweep cells of a checkpoint that pins an attention kernel. The
  matrix had no rule for these pins, so it would have labelled every such
  cell `render`. It now reads each `comfy_attention` config tensor from the
  header of every checkpoint a graph loads and hands the pins to the
  runtime's own guard (`attention_pins.assert_waived`), basis `header`. On
  single, cfg2, dp2, cfg2+fsdp and dp2+fsdp the cell reads `refuse:K` under
  `sol_attn:checkpoint_pin`. It is waivable only where a grant lets the cell
  render: not on an FSDP preset at `auto_gate=first_use`, where the Gate
  ceremony renders without the waiver, and not on cfg2 or cfg2+fsdp at CFG
  1.0 without a cfg++ sampler, where a granted leg meets the cfg refusal.
  A pinned Chroma nvfp4 file on cfg2 meets the sharded nvfp4 bar first and
  carries no waiver either, because the runner grants one card per cell and
  the granted leg would meet the pin guard.
  The runner now dismisses the card such a refusal raises, so a later cell's
  grant still finds exactly one card of its kind. A bf16 or fp16 file that
  pins refuses class P at the FSDP launch check first, and a sol kernel
  without sequence parallelism refuses class P before any load. uly2 and
  ring2 drop the pins and render, but their single-GPU reference runs the
  pinned kernel, so every cell scored against that reference, cfg2 and dp2
  included, takes the `waived-nrms` bar: the difference is recorded and no
  floor applies. The rule reads checkpoint headers only, so the matrix drops
  the lora-on arm of a template whose LoRA carries a pin config, and the
  artifact set of a checkpoint whose config it cannot predict, each with a
  note. ADAPTERS, MODELS and TROUBLESHOOTING #117 no longer say the FSDP
  shard build clears a bf16 or fp16 checkpoint's own pins: the launch check
  refuses such a file first. No current template uses a pinned checkpoint,
  so no existing cell's label or bar changes
  ([SWEEP](docs/SWEEP.md#1-cells-and-expected-results)).
- Export ownership occupancy as Monarch up-down counters when
  `[monarch.diagnostics] metrics = true` (issue #35):
  `dgxm.return.live_handoffs`, `dgxm.return.pending_read_jobs` (the read
  leases), `dgxm.return.poison_owners`, `dgxm.mesh.active_leases`,
  `dgxm.mesh.abandoned_samples` and `dgxm.native_return.latched`. Each reads the
  source of the `/dgxm/metrics` series with the matching name. The thread that
  already exports phase metrics samples the levels at most once a second and
  adds the change since its last export, so no new thread, timer or service
  starts, and the next sample resends a failed add. With metrics off the
  counters make no instrument and read no source. A meter without up-down
  counters logs one DEBUG line, and the phase metrics and renders go on
  ([CLUSTER.md](docs/CLUSTER.md#monarchdiagnostics)).
- Record the phases of each native latent return (issue #35). The Worker's
  result row carries `return_phases`: the child-process preflight, timed as the
  new `rdma_preflight` phase, the registration, any rollback drops
  (`rdma_drop`) and the rest of the pack. A preflight child that runs past its
  15 s bound records the phase as a timeout. For each completed native return
  in a normal render, the driver logs one `native return:` line with that split
  and its own read, drops, ACK and rest. Fleet jobs keep the Worker's split in
  the result row only; no Fleet log line or ledger shows it. The qualifier's
  `--timing` mode stores both splits with each native call and in its summary.
  When a record reads complete, its phases add up to its total within half a
  millisecond for each rounded value. Nothing about the return changes, and
  native return stays opt-in. The split of the 2 s first native use is NOT RUN
  until the next hardware leg ([VALIDATION.md](docs/VALIDATION.md#native-rdma)).
- Decide against an inspector mode for Monarch's own observability services
  for now (issue #35). In the 0.7.0 wheel, telemetry, `Job.profile`,
  `monarch query` and the dashboard each need a Job, whose attach pushes a
  second client's config to the Workers. Telemetry also spawns procs on every
  Worker host, and the dashboard, when enabled, binds every interface. The mesh
  admin agent needs no Job and can bind loopback, but its handle has no stop.
  The snapshot actor has no stop handle and runs until its procs stop.
  DESIGN.md section 5.8 cites each fact and the upstream changes that would
  unblock it; the TROUBLESHOOTING escape hatch lists what to use today.
- Remove the temporary files that killed actor ledger writers leave in
  `~/.local/state/dgx-monarch` (issue #36, which counted 164 of them on one
  host). The Worker service's tracker thread writes a temporary file and then
  renames it over the ledger. A writer killed between the two steps, for
  example when a stop kills the Worker service, runs no cleanup and leaves
  the file. The file is now named
  `actor-procs.json.<pid>.<starttime>.<thread id>.tmp`, and each ledger write
  removes the files whose writer no longer runs with that pid and start time.
  The start time makes the check exact within a boot: a live writer's file
  always stays, a later process that took a dead writer's pid no longer
  keeps its file, and the sweep reads no clock or file time. Files in the
  older `actor-procs.json.<pid>.<thread id>.tmp` shape go once no process
  holds their pid, so one whose pid a later process took stays until that
  process exits. A failed rename now removes its own file too. Before, the
  next write from the same thread reused the name and failed, so one refused
  rename blocked every later ledger write from that loop. The write stays
  atomic, and a debug line counts the files each sweep removed
  ([TROUBLESHOOTING #51](docs/TROUBLESHOOTING.md#51-a-worker-actor-outlives-its-client-and-holds-tens-of-gib)).
- Drop a checkpoint's per-block kernel pin at every stock attention call the
  pack leaves in place under sequence parallelism (issue #30). ComfyUI runs a
  block's pinned kernel unless the call's transformer options carry an
  attention override, which it checks first. Cross-attention in Wan 2.1 and
  2.2, Wan SCAIL, WanDancer, Wan-Animate 2, LTX, Kandinsky 5 and Anima, and the
  refiners every rank runs whole in Z-Image, OmniGen2, Boogu, HunyuanImage 2.1
  and HunyuanVideo 1.5, MiniMax H3, Kandinsky 5's text blocks, Krea2 and Kroma,
  got the original options or none, so a checkpoint that pinned comfy-kitchen
  Sol ran it there under the widget's stamp, and the pack asked for no class-K
  card. Those forwards now hand the stock modules a copy of the options whose
  override runs ComfyUI's default backend with the call's own arguments, as
  stock runs it with no pin (`adapters/stock_attention.py`), so a checkpoint
  with no pin renders bit for bit as before. The source read found two sites
  the issue did not list: WanDancer's music encoder, which calls attention
  with no options and now runs a bound copy of its forward that passes no
  pin, and the LTX audio-video text connectors, which ComfyUI runs from
  `extra_conds` with no options. It also found that ComfyUI runs MiniMax H3's
  token refiner, which the issue did list, from `extra_conds` with no options
  as well as from the forward, so no options copy can reach that call. The
  attention modules of the connectors and the token refiner now get a bound
  forward that adds the override. The override cannot log the pin, because
  ComfyUI removes it before the call; the bound forwards log it once at debug
  level and say that ComfyUI's default backend runs the call. The copy keeps
  an override the graph installs where the forward admits one; LTX and
  Wan-Animate 2 refuse a foreign override before they build the copy
  ([TROUBLESHOOTING #72](docs/TROUBLESHOOTING.md#72-a-distributed-render-refuses-a-comfy-attention-patch-or-nag-style-hook)).
  Topologies without sequence parallelism keep every stock call and honor a
  pin, as stock ComfyUI does. CPU tests drive each family's injected forward
  with stand-in pinned sites, and the canary's attention keyword handoff seam
  drives tiny real ComfyUI models of Wan 2.1, LTX, Kandinsky 5, Anima's cosmos
  DiT, Z-Image and Krea2, and the three sites the forward's options copy
  cannot reach, through the same check ([ADAPTERS](docs/ADAPTERS.md) lists
  each site).
- Stop a checkpoint's per-block kernel pin from bypassing the Sol-Attn
  waiver in the attention the pack runs on Qwen Image 2.1 and in Wan-Animate 2
  self-attention (issue #30). Their sequence-parallel forwards call ComfyUI's
  native attention themselves and handed it the block's preferred kernel, so
  a checkpoint that pinned comfy-kitchen Sol ran it under a TORCH_FLASH stamp
  and the pack asked for no class-K card. Both now do what Anima's op does:
  they log the pin once at debug level and hand ComfyUI none, so ComfyUI's
  default backend runs. In a worker that cannot import xformers, that is
  torch's scaled dot product attention, which tries flash first and picks
  another exact backend where flash cannot take the call, so no approximate
  kernel runs under the stamp. The pack asks for the Sol-Attn waiver only
  when the widget selects Sol. Qwen's cache-miss prefix fill, which runs
  ComfyUI's own callback, drops the pin too, and a CPU test now drives that
  branch. The daily canary's attention keyword handoff seam checks the Qwen
  Image 2.1 attention and the Wan-Animate 2 self-attention against the real
  ComfyUI tree. The entry above covers the stock attention the pack leaves in
  place.
- Close the small follow-ups from the 2026-10-12 review (issue #23).
  `dgxm doctor` now names a failed source probe as `dgxm restart` and a
  syncing `dgxm up` do: a missing or non-executable Worker interpreter reads
  `FAILED_WORKER_INTERPRETER` with its exit code instead of
  `FAILED_SOURCE_LAYOUT`, and any other exit without a verdict reads
  `FAILED_SOURCE_PROBE`, since Doctor runs the probe without the lifecycle
  lock. Plain `dgxm update` from a linked git worktree now refuses with a
  message that names it, where it used to skip the fetch; a copy without Git
  still skips it ([INSTALL](docs/INSTALL.md#updating)). The mesh row and
  `SampleResultBusyError` now say an abandoned sample may still own
  worker-side latent backing, and `LifecycleBusyError` that it may still own a
  result or RDMA descriptor of its setup generation; neither says it may still
  be running. A Monarch 0.7.0 collective call returns an actor error only after
  every rank has replied, so a sample whose rank raised has stopped, though a
  rank that raised after packing its latent can still own the backing. After a
  timeout, an interrupt or a worker death the sample may still be running
  ([TROUBLESHOOTING #87](docs/TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)).
  [TROUBLESHOOTING #115](docs/TROUBLESHOOTING.md#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)
  records why latents held by failed native reads get no cap and no operator
  release: one fault holds at most one Fleet wave or one pipeline window of
  them, and one restart frees them all. The worker-policy broadcast and
  process-group setup behind `MeshHandle` moved from `mesh.py` to
  `mesh_setup_transition.py` with no change in behavior, which lowers the
  `mesh.py` line ceiling from 898 to 690; the `cli/doctor.py` ceiling drops
  from 774 to 770 to match that file.
- Stop the padded Ulysses path from waiting on the device. The keep index
  for pad exclusion, the rank-major joint-order permutation and its inverse
  depend only on integer row counts, so the path now builds each once per
  layout on the device and reuses it from a small cache keyed by shape and
  device, which drops its least recently used entry past the bound. The
  pad-row remap reads a host copy of the inverse, computed on the CPU,
  instead of reading the device once per dropped row, and the keep index no
  longer calls `nonzero`. Each padded attention call used to wait on the
  device about six times, roughly six thousand times in a 20-step Chroma
  render. `index_select` and the scatter receive the same integers as
  before, so the output does not change. On Chroma1-HD FP8-mixed at uly2
  with 20 steps, the 1176 square render, which pads, ran 2.2 percent faster,
  the 1152 square render, which does not, stayed within noise, and every
  seed's pixels matched the previous build's
  ([VALIDATION](docs/VALIDATION.md#padded-ulysses-index-cache)).
- Let `auto` take cfg2 where Campaign K (2026-10-14, head capped at 2100 MHz)
  measured it bit-identical to one GPU and faster than the route it replaced:
  Krea2 FP8 and BF16 from 1.2 to 2.36 MP (row 15), Chroma BF16, FP8 and INT8
  from 1.2 to 1.64 MP (row 19), Qwen-Image FP8 from 1.2 to 1.77 MP (row 68),
  and Boogu FP8 (row 79) and Anima BF16 (row 47) below 1.2 MP. Each row's
  dated evidence record grants only the precisions that read bit-identical and
  beat the incumbent at every measured size in its band (Krea2 INT8 was exact
  but no faster, so it keeps Ulysses), and the granted cfg2 ran 1.17x to 1.66x
  faster than the incumbent
  ([BENCHMARKS](docs/BENCHMARKS.md#campaign-k-cfg2-against-uly2-2026-10-14-head-capped)).
  A cfg++ sampler at CFG 1 keeps its uncond pass, so `auto` no longer skips a
  cfg2 row when a DGX Monarch KSampler node names it, and the render finish
  record now carries the fields the table audit needs, with the sampler wall
  and mean step keyed to their own render so a pipelined render in flight
  beside it cannot lend them.
- Price the FSDP clean-reload proof against what the proof frees. The initial
  preflight and the post-baseline reprice now credit each rank's resident shard
  pool, read from the worker's reserved figure and capped at one share, because
  the proof unloads that shard set before it builds the next one. The Worker's
  own check after the real unload stays decisive. On the 60 GiB bf16 Flux2
  checkpoint the preflight refused at 39.1 GiB available against 39.8 required
  while 30 GiB of shards it was about to drop were resident. With the credit the
  price fits, but the Flux2 proof has not been rerun on hardware with it.
  Every refusal names the credit it applied
  ([TROUBLESHOOTING #88](docs/TROUBLESHOOTING.md#88-the-first-render-of-an-fsdp-topology-refuses-with-a-capacity-figure-and-no-proof)).
- Clear a set of small defects and hygiene items an external review and the
  2026-10-13 audit turned up. `dgxm setup` closes each config-directory
  descriptor once and never retries a released number. `dgxm top` rail rates
  and the driver's telemetry cache TTL use a monotonic clock, so a wall-clock
  step cannot print a 1000x rate or hold a stale payload. The whole package now
  type-checks under mypy's `check_untyped_defs`, and CI prints the mypy version;
  the TUI attribute that shadowed Textual's `App.animate` is renamed. The
  sidebar refreshes once when a reset outcome lands, and a revoked consent no
  longer hides the re-raised card for 15 s. The lifecycle helpers name the
  system interpreter through one `SYSTEM_PYTHON` constant with its reasons, a
  new doctor row checks that `/usr/bin/python3` on each host starts and is 3.9
  or newer, CLUSTER.md states the requirement, the generation helper is
  assembled from two constants instead of a string split, and the Sol-Attn
  install script refuses to ship a relative import. Load profiles are written
  under `~/.cache/dgx-monarch/profiles/` with the newest 20 kept,
  `/proc/meminfo` is read with a context manager, a dead duplicate
  `wait_for_mesh_creation` is gone, ADAPTERS.md and VALIDATION.md say that
  `sync_ulysses` has no effect on the pinned xfuser, and the ComfyUI canary's
  path filter lists every module its real-Comfy steps import.
- Render Anima under sequence parallelism again on current ComfyUI builds.
  ComfyUI now hands a model's attention op its q, k and v in single-use
  containers and names a preferred kernel for each block; Anima's op refused
  both with a TypeError inside the worker. The op now takes the containers and
  ignores the preference, because the Init node's attention setting picks the
  kernel, and a new seam contract holds that handoff against the real ComfyUI
  tree so the daily canary catches the next change of this kind.
- Heal the abandoned-lease wedge before eager setup too. After a worker crash,
  changing the checkpoint, topology or levers no longer needs a manual Reset:
  Init with an explicit topology, the UNET Loader and the BasicScheduler now
  run the same one-shot reset that renders and Fleet queues run, and log it.
- Show more of the cluster where operators already look. `/dgxm/metrics` adds
  `dgxm_render_sec_per_step`, `dgxm_native_return_latched` (0 or 1) and
  `dgxm_native_return_fault_count`; under a stale telemetry payload the latch
  series appear only when the cached latch is set, so a cached clear is never
  reported. The render finish record in `/dgxm/telemetry` now carries the
  resolved topology label, the auto table row, the latent megapixels, the quant
  kind, the attention kernel and the final seconds per step, so readers of
  `/dgxm/telemetry` and the sidebar can see where the table's choice is slow.
  `dgxm top` and `dgxm diagnostics` captures do not show these fields yet. Each
  render now times a step against its own previous step, so the second leader
  of a dp>1 render, or another render in the same Fleet wave, no longer turns
  the gap between two messages into the seconds per step. Telemetry publishes
  the RDMA rail link state next to the byte counters, read from the same sysfs
  files `dgxm doctor` uses, and the time the box read them, so the sidebar and
  `dgxm top` time each rate by the box's own readings. A device with more than
  one cabled port now reports each port as `<device>/<port>` in the counters,
  the link state and the `/dgxm/metrics` rail label. The sidebar draws the
  active topology above the host cards and one rails line per host with RX and
  TX rates, and in red the link state of any rail that is not ACTIVE. A box
  that runs two ranks has its rails read once per poll, so its two cards show
  the same rates. The idle render line names the topology, the table row and
  the step rate. The rail readers moved to `rails.py`, the resolved topology
  facts live in `nodes/resolved_topology.py` and the per-render step marks in
  `telemetry_steps.py`.
- Fix five defects an external review of the dev branch confirmed. A
  measured or persisted identity-gate FAIL now quarantines only its own
  checkpoint combination, applied on each dispatch through a per-dispatch
  policy clone, so other checkpoints on the same Init node keep the residency
  levers the graph asked for and run their own first-use Gate; `comfy_managed`
  still turns off for the whole Init node, and an unreadable ledger still
  fails closed for the graph
  ([TROUBLESHOOTING #67](docs/TROUBLESHOOTING.md#67-what-auto_gateoff-costs-and-what-it-does-not)).
  Because two checkpoints on one Init node can now need different worker
  settings, a pipelined render whose setup differs from the live one first
  collects the renders still in flight, whose result leases would otherwise
  refuse the change. The three verdict helpers that `auto_gate.py` held moved unchanged to
  `nodes/gate_process_verdicts.py`. Quantized FSDP shards keep every
  row-aligned layout parameter, such as an int8 per-row scale, whole on every
  rank and gather only the weight bytes, so a weight whose shard holds a
  single row, or whose rows do not split evenly across ranks, no longer hangs
  the all-gather or rebuilds with the wrong scale; no supported checkpoint
  reaches either case today. The Chroma cfg-pad forward hook calls the class
  `_forward`, as the other adapters do, so installing it twice replaces the
  wrapper instead of nesting it. The driver's scratch threads for off-loop
  waits, RDMA read jobs and Fleet drains are daemon threads regardless of the
  thread that created them. The loader residency memos and the consent audit
  set are cleared and written under locks, so a **Reset attached mesh** that
  lands during a loader preflight can no longer raise out of a memo eviction or
  carry an audit token into the next generation, and an audit whose write
  failed releases only its own claim, never a newer one on the same token.
- Run the abandoned-lease reset before a Fleet queue too. When abandoned
  samples are the only obstruction, Fleet recycles the attached mesh once
  before its first dispatch, drops the driver's residency memos and dispatches
  on the fresh fleet, so a Fleet queue after a failed Fleet wave no longer
  needs **Reset attached mesh**. Both paths check and reset the fleet whose
  dispatch refused, so a later wedge in the same ComfyUI session heals as the
  first one did. An active sample lease, a pending RDMA read or an
  unconfirmed stop leaves the refusal as it was
  ([TROUBLESHOOTING #87](docs/TROUBLESHOOTING.md#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)).
- Record the cold-cache readiness measurement in VALIDATION.md. With the page
  cache dropped on both Sparks, the `readiness` phase peaked at 4.151 s
  (3.519 s warm) and each actor bootstrapped in 5 to 6 s, against Monarch's
  30 s hold on messages to a starting actor, so the hold still gets no policy
  setting.
- Bring the docs and three small code paths in line with the pinned
  TorchMonarch behavior. DESIGN.md now says that an endpoint Future survives a
  timed-out `.get(timeout)` while a `returns_future` body such as
  `ProcMesh.stop` is cancelled, keeps abandoning a timed-out sample because a
  later read cannot settle who owns its result, and names concurrent status,
  cancellation and lease renewal as the reason every endpoint is async.
  [TROUBLESHOOTING #1](docs/TROUBLESHOOTING.md#1-attach-fails-with-mesh_attach_config_timeout),
  the `client_bind` row in CLUSTER.md and the missing-`client_bind` refusal now
  say that a loopback-only hostname fails at once and a LAN or Tailscale
  address leaves the fabric. The power-user escape hatch points at `dgxm top`,
  the sidebar and `dgxm diagnostics`, because `monarch-tui` needs a mesh admin
  server that the integration never starts. The supervision string net no longer
  lists `ProcessExited`, which the pinned binding no longer carries. The CPU
  smoke canary now times out a `ProcMesh.stop` at once and requires the stop to
  stay cancelled and its procs to answer until a fresh stop. VALIDATION.md
  records that ownership review and the actor-start readiness measurement
  against Monarch's 30 s hold on messages to a starting actor; no policy setting
  was added for the hold, and CLUSTER.md names it beside the readiness budget.
  The Monarch diagnostics adapter loses its unreachable trace branch and span
  sink, so its status no longer reports `traces`. `dgxm up` and `dgxm restart`
  now read a managed host's torchmonarch version from its Worker release, as
  `dgxm doctor` does, so a stale base interpreter behind a current release no
  longer warns, and a stale release names `dgxm update` as its only fix.
- Keep the driver's metrics export through a config edit that does not
  produce a new fleet. An `otlp_endpoint` the driver cannot install now refuses
  before anything is written, so the running exporter is never stopped, and an
  exporter thread waiting on a slow telemetry backend carries on when the
  backend answers. When the driver admits an edit but cannot stop the old
  workers, for example because a render still holds them, it now puts back the
  old fleet's policy and export, so a revert that reuses that fleet finds both
  as they were. If an exporter stopped for that edit is still blocked in a
  backend call and cannot restart, the driver logs one ERROR and
  `/dgxm/telemetry` shows `exporter_still_running` in `monarch_diagnostics`
  until the next mesh build.
- Latch native latent return off for the rest of the driver session after a
  native read fails on the driver. Later setups ask the actors for messaging
  instead of repeating the fault and retaining another latent each time; the
  failed render, and a Fleet job whose own native read fails, raise
  `NativeReturnLatchedError`, the driver logs one warning, and `/dgxm/telemetry`
  shows the latch next to `return_ownership`. Ownership rules are unchanged;
  the latch clears when ComfyUI restarts
  ([TROUBLESHOOTING #115](docs/TROUBLESHOOTING.md#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)).
- Add `--timing` to `benchmark/rdma_return_bench.py`: first-use and warm timed
  returns at chosen sizes inside the checked CPU-staged run, in either
  message/native order, with `--timing-summary` pairing the reports into warmed
  medians, savings and ratios. The summary needs both orders and refuses a set
  that lacks one, naming the missing order. Timing evidence refuses host names,
  addresses, PIDs and home paths, and a plan that would stage more than
  `--timing-budget-gib` (default 8) refuses. The recorded 2026-10-13 figures
  came from a private workspace tool; this mode implements the same protocol
  for later campaigns and has not yet run on hardware.
- Make plain `dgxm update` fetch first, check the fetched commit's dependencies,
  and fast-forward only after that check passes, so a refusal leaves the
  checkout at its current commit instead of at the new commit against the old
  torchmonarch. A checkout whose HEAD already contains its upstream, as with
  local commits, stays where it is and is checked at HEAD. A staged, unstaged or
  index-hidden edit to `pyproject.toml` refuses before the fetch, because the
  fast-forward would otherwise install it without checking it.
- Attribute a normal render's progress steps to the render's own telemetry
  token without transferring ownership, and ignore a step for a finished render
  or one that arrives when no render is live. A late step no longer leaves
  `dgxm top` and the sidebar showing an active render until ComfyUI restarts.
- Stop `dgxm diagnostics` from marking a healthy capture partial because its
  last poll window was shorter than a fetch: a final window shorter than the
  slowest fetch so far is slept instead of polled. `--output` is checked before
  the capture starts, and a refusal prints its fixed reason instead of only the
  exception type.
- Report `dgxm_diagnostics_available` and `dgxm_return_ownership_available` as
  0 for a Worker whose source has no diagnostics block, as after a partial
  deploy, instead of omitting the series. The driver scope also reads 0 when
  its payload has no block, as when the first telemetry refresh fails.
- Cut the `dgxm top` driver phase summary to one row instead of wrapping it over
  the panels below on an 80-column terminal.
- Name why a host's source probe failed: a missing or non-executable Worker
  interpreter reads `FAILED_WORKER_INTERPRETER`, a lifecycle lock failure
  `FAILED_LIFECYCLE_LOCK`, any other exit without a verdict
  `FAILED_SOURCE_PROBE`, each with the exit code and a bounded, home-relative
  stderr tail. `dgxm restart` refuses on all of them before stopping any Worker.
- Run the driver's policy admission before recycling a live fleet, so a
  cluster.toml edit the driver refuses (native return without a `nic:` target
  for every host, an export it cannot start, a transport change that needs a
  fresh process) keeps the working fleet attached; reverting the edit reuses it.
  The refusal puts back the driver's earlier policy and export state, so a
  running metrics export keeps running.
- Tighten `[monarch]` validation: `nic:` names starting with `-` or `.` or
  containing `..`, `cpu:` and `gpu:` numbers with leading zeros, a `groups:`
  affinity with fewer than two groups, and an `otlp_endpoint` with a path
  other than a bare trailing `/` now fail config loading with a typed error
  naming the key. A Worker unit rendered from such a value reads
  `POLICY_STALE`: fix the value in cluster.toml, then run
  `dgxm update --verify --target HEAD` to re-render the unit. The update, like
  the probe, checks a unit's policy argument for its keys, value types and
  canonical encoding but not its values, so it accepts that unit as stale
  instead of refusing with `prior_worker_drift`.
- Treat a Worker unit that carries a well-formed Monarch policy other than
  the current one as stale rather than tampered. A `[monarch]` or
  `cluster.rdma_latent_return` edit on a managed installation no longer locks
  out lifecycle commands: the probe reads `POLICY_STALE`, `dgxm restart`
  refuses before stopping any Worker service, `dgxm up` names the cause, and
  `dgxm doctor` adds a `worker policy` row per host. When a host's probe goes
  unanswered, restart keeps the old order and `dgxm up` refuses the stale host
  after the stop. `dgxm update --verify --target HEAD` renders the current
  policy with the usual rollback. Worker units carry the default
  `monarch.startup_timeout_s`, which only the driver reads, so editing it
  leaves them current. A unit without policy arguments counts as current only
  under the default policy with native return off
  ([TROUBLESHOOTING #113](docs/TROUBLESHOOTING.md#113-lifecycle-commands-refuse-with-policy_stale-after-a-policy-edit)).
- Resolve the driver's own `[monarch.rdma]` host target by matching
  `client_bind` to a `[[hosts]]` address, the rule the RDMA qualifier now
  shares; its `--driver-host` becomes an optional check. With native return on,
  the driver refuses before starting any process unless it and every host
  resolve to a `nic:<device>` target, whatever `max_nics_per_buffer` allows,
  and a `host_targets` value other than `nic:<device>` fails config loading.
- Make driver Monarch metrics and OTLP export work: registering the nodes no
  longer loads Monarch, so the driver applies its endpoint first. A requested
  export that cannot start refuses the mesh build with a typed error naming the
  reason; actors and Workers log it. `/dgxm/telemetry` reports
  `monarch_diagnostics`.
- Narrow the Gate configuration fingerprint to settings that can change math,
  residency or transport, and the transport identity to `[monarch.rdma]`.
  Editing `auto_heal`, `ssh_key`, `monarch.startup_timeout_s` or
  `[monarch.diagnostics]` keeps PASS rows. Each combination re-proves once
  after this change
  ([TROUBLESHOOTING #114](docs/TROUBLESHOOTING.md#114-every-combination-runs-its-first-use-gate-again-after-an-update)).
- Stop recording digests of install paths and pip RECORD files in Monarch
  capability snapshots, remove them from the committed 0.7.0 pin baseline, and
  make the leak check report a digest of any private home path the scanning
  interpreter can see.
- Mark the Kroma INT8 artifact row as a file the reader supplies, record that
  the pinned upstream Qwen3-VL encoder matches the tested encoder tensor for
  tensor, and point the data-parallel batch refusal at the Kroma upscale
  graph's batch-one note.
- Fix the torchmonarch canary's Future probe to use a returned channel Future,
  avoiding the constructor removed in 0.7.0, and show comparison reports and
  failed stages in Actions.
- Check authenticated managed Worker releases in Doctor host environment probes,
  so a staged runtime upgrade does not falsely report the unchanged base pin as
  Worker version skew. Keep driver and plain-install checks separate. Report
  bounded failure categories from verified-update Doctor checks without private
  probe details; certainty and rollback behavior remain unchanged.
- Allow an explicit foreground process-inspector lifetime up to four hours;
  retain the launcher's 30-minute default and authenticated, bounded inspection.
- Add Kroma v0.3.1 Turbo OPD workflows for BF16, local INT8 conversions and
  two-pass 2x refinement using the existing Krea2 runtime. Document upstream
  sampling settings, required artifacts and native shift handling. Preserve
  the stock Krea2 final projection shape under pure Ulysses; the BF16 and local
  INT8 8-step graphs and BF16 2x refinement matched their own stock one-GPU
  references in conditioning, latents and decoded pixels. Acceptance covers
  the measured pure-Ulysses configurations. Manual `uly2+fsdp` also matched
  stock for BF16/local INT8 T2I and BF16 2x refinement at batch 1. Batch-2
  `dp2+fsdp` T2I remains CHECK: FSDP reload proofs passed, but native
  stock-output fidelity did not. Keep
  upscale at batch 1: native VAE encoding collapses an initial image batch of
  2 to one latent item, which DP2 correctly refuses at refinement.
- Upgrade to TorchMonarch 0.7.0 and advance the package identity to 1.2.0 so
  previous Gate PASS rows cannot authorize the new runtime. Preserve the
  existing Torch/CUDA build and validate the candidate's dependencies before
  pin installation in plain and verified updates. Report bounded package
  diagnostics when prerequisites are missing or incompatible.
- Let verified updates and recovery explicitly reuse the reviewed read-only
  process inspectors when protected process markers are unreadable. Require
  fresh authenticated reports at both activity checks and record the choice
  in the receipt; initial setup defaults remain unchanged.
- Preserve exact legacy Worker units during migration and verify policy-bearing
  Worker commands against their captured units. Finish inspector socket cleanup
  even when terminal shutdown delivers repeated signals.
- Validate native RDMA binding load and package/import ownership separately from
  public API signatures. Extend the dependency canary with real local CPU actors,
  concurrent dispatch, Future timeout semantics and ProcMesh shutdown. Run it
  against Torch 2.12.0+cu132; retain these checks as CPU-only evidence.
- Validate the unchanged shipped Krea2 workflow at 1536×1536 with its automatic
  topology and first-use Gate; record a slab cross-residency PASS and owned cleanup.
- Validate the unchanged shipped H3 workflow at 1344×768×124 with full CPU
  video/audio decode and owned cleanup. Its Gate remains INCONCLUSIVE/no_material;
  the recorded result establishes functional output, not a fidelity or speedup claim.
- Record exact-source GPU checks for the migration: adapted Krea2 Turbo pixel equality,
  explicit recovery after interruption, a RAW FSDP clean-reload Gate PASS and
  reduced-canvas H3 MP4 output. Matched latent decoding localized the pipeline/Fleet
  pixel differences downstream of sampling; clean cancellation remains unproven.
  See [the scoped results](docs/VALIDATION.md#gpu-checks-for-the-070-migration)
  for reused-output counts, recipes and timing limits.
- Add validated Monarch startup and RDMA policy with per-host target overrides,
  native readback and transport identity. Start with one NIC, one QP and one
  application chunk. Selected-rail CPU-staged production returns passed the
  recorded integrity and ownership cases, and one-step at-threshold latents
  matched stock and messaging in two fresh native sessions. Repeated large-return
  timings support opt-in use; native return stays off by default. See [the exact scope](docs/VALIDATION.md#native-rdma).
- Fix Fleet progress remaining active after completed jobs. Each receiver owns
  its progress token, retires it during cleanup and ignores late steps after
  retirement without clearing another render.
- Record healthy no-model reattach and controlled fresh-driver recovery, while
  preserving the reproduced same-driver stop/start and restart failures;
  [issue #2](https://github.com/Deen-Media/dgx-monarch/issues/2) stayed open
  after these checks. Per-start Worker ids (above) close it; the two-Spark
  recheck ran on 2026-10-15.
- Add bounded phase timings and opt-in supported Monarch metrics. Advanced
  attached-cluster profiling, SQL and stack tools remain explicitly unavailable
  where the upstream API cannot provide an owned, stoppable session.

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
