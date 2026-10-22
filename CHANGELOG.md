# Changelog

## [Unreleased]

- Add an optional Studio companion with bundled workflow controls, durable job
  records, project libraries, reviewed imports, grid experiments and job-bound
  evidence; read the [acceptance record](docs/STUDIO_ACCEPTANCE.md) first.
- Studio update needs `--config` and `--driver-service` and defaults to verified
  mode. The owner token prints only to a terminal, otherwise to a 0600 file
  ([Studio](docs/STUDIO.md#install-and-start), [#123](docs/TROUBLESHOOTING.md#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token)).
- `dgxm top` and the ComfyUI sidebar show memory in GiB, draw the memory bar on
  hosts with no slab-resident model, give the six 16-color segments six colors
  and show RX and TX rails for every box ([TUI](docs/TUI.md#what-it-shows)).
- Drivers register GET `/dgxm/execution-events`, which grants the Studio
  companion a 30 s observer lease; the [threat model](docs/THREAT_MODEL.md#browser-to-comfyui-driver)
  lists it with the other driver routes.
- Verified update, release staging and setup refuse a target whose packaged
  Studio files are missing, extra or differ from their recorded SHA-256 table.
- Studio shutdown waits for the threads that settle history and write live and
  operation previews, and the store closes its database under the lock every
  other use holds. Closing it under a running settlement crashed the companion
  with a segmentation fault.
- Studio's job, media, grid and runtime history pages no longer serve a row
  twice or skip one where a page ends. Each cursor now carries the key SQLite
  sorted the page by, not a value read back from the last row: the job and
  grid lists re-render `created_at` at 15 significant digits, and SQLite on
  x86-64 parses a few stored timestamps one unit in the last place high.
- `dgxm studio` starts the Studio companion with one word. It reads the
  driver origin, the port, an optional driver service and an optional start
  command from a new `[studio]` table in cluster.toml
  ([CLUSTER](docs/CLUSTER.md#studio-optional)), passes the config file `dgxm`
  itself uses, prints where each value came from, and never probes ports.
  `--driver` and `--config` override them. The first start on a fresh state
  directory needs `[studio] driver` or `--driver`. Studio reads only a
  cluster.toml no other account can write. Releases before this one refuse a
  cluster.toml that holds `[studio]`, so add it only after every checkout
  that reads the file and the driver run this release
  ([Studio](docs/STUDIO.md#configure-studio-in-clustertoml)).
- Studio opens signed in. On a terminal with a display, `dgxm studio` and
  `dgxm-studio` send the browser to `sign-in.html`, an owner-only page in the
  state directory that carries the link `http://127.0.0.1:<port>/#token=...`,
  when `.html` files and `http` links open in the same browser; the page
  keeps the token in the tab's session storage and removes it from the
  address bar. The browser's history keeps the link, so treat it like a
  password. The token never appears in a process argument dgxm starts, a
  query string, a log or the journal. `--no-open` starts no browser. A
  companion started with `--allow-management` never makes a link; its token
  is typed into the form.
- `dgxm studio up` installs and starts `dgxm-studio.service`, a user unit
  with a fixed port; `dgxm studio down` removes it; `dgxm studio open` prints
  the sign-in link and opens it when the session has a display, and says
  when the service runs code older than the checkout; `dgxm studio token`
  prints the token file and, with `--rotate`, restarts the service for a new
  token. With `[studio] driver_service` the service also starts that driver
  unit. `dgxm uninstall` refuses while the service is installed
  ([Studio](docs/STUDIO.md#keep-studio-running),
  [#132](docs/TROUBLESHOOTING.md#132-dgxm-studio-up-refuses-or-the-always-on-studio-does-not-start)).
- `dgxm studio tunnel <host>` prints the one SSH line that reaches Studio
  from another computer,
  `ssh -o ExitOnForwardFailure=yes -L 127.0.0.1:<port>:127.0.0.1:<port> <host>`,
  the address to browse and what to do when ssh cannot forward the port; the
  line names the address the forward listens on, so with that option ssh
  exits instead of opening a session without the forward.
  `--run` runs it
  ([#133](docs/TROUBLESHOOTING.md#133-the-sign-in-link-does-not-open-studio)).
- When no driver accepts a connection, Studio shows Driver not running with
  the command that starts it, disables Review generation, and connects by
  itself when the driver answers
  ([#131](docs/TROUBLESHOOTING.md#131-studio-shows-driver-not-running)).
- `dgxm-studio` binds its port before it prints or writes the token, so a
  busy port exits with one line and no token.
- The README shows Studio, captured against the CPU test fixture.
- The Studio inventory waives the `dgxm studio` rows: the page cannot start
  or stop the companion that serves it.
- The original Studio specification gains a dated amendment for the
  always-on unit.
- Support MOSS-Image, the OpenMOSS team's 5B single-stream text-to-image
  model, with its SFT and Base checkpoints and lodestones' Kromoss fine-tune
  ([MODELS](docs/MODELS.md#model-moss-image-sft)). It needs ComfyUI 0.42.0 or
  newer, the first release to carry ComfyUI's support commit e087d8f6c of
  2026-10-16 ([TROUBLESHOOTING #125](docs/TROUBLESHOOTING.md#125-a-moss-image-checkpoint-fails-to-load-on-comfyui-older-than-0420)). The `moss_image` adapter
  shards the joint `[image, text]` stream in one contiguous window per rank,
  keeps the prelude and the two text-embedder blocks whole on every rank,
  launches both fused-norm calls of each block at stock's row count, names
  pad rows on every topology and refuses a padded ring class K. Under
  sequence parallelism it honors a block replacement only when it is an
  identity or marked `sequence_shard_local`, which admits ComfyUI's own Skip
  Layer Guidance skip, and refuses any other class P, as it does a marked
  replacement that hands the block other rows than the rank's window. Five BF16 templates
  ship at the vendor's settings: 50 Euler steps at CFG 3 with the layer-22
  skip for SFT, Base and Kromoss, a JSON layout graph on SFT, and 8 steps at
  CFG 1 for Kromoss with its DMD2 LoRA. Nine testing graphs cover the INT8
  ConvRot files, an odd grid, a 1536 square canvas, the skip turned off, the
  second DMD LoRA, the DMD2 LoRA on Base and a LoRA slot. The artifact
  manifest pins the six DiT files, the BF16 and INT8 ConvRot encoders and the
  FLUX.2 small decoder at Comfy-Org's and lodestones' commits, and the two
  Kromoss LoRAs.
- The MOSS-Image INT8 ConvRot files pin comfy-kitchen's INT8 attention on all
  42 blocks, so single, cfg2 and dp2 ask for the Sol-Attn waiver and uly2
  drops the pins ([TROUBLESHOOTING #117](docs/TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)). On uly2, ring and cfg2 the
  family admits only BF16 and INT8 ConvRot. RoPE scaling and the color and
  depth control, on ComfyUI master only on 2026-10-20, ship no template.
- Add Skip Layer Guidance (DGX Monarch), which carries stock
  SkipLayerGuidanceDiTSimple's widgets to every Worker, where ComfyUI's own
  node applies them to the render. The graph advisor suggests it in place of
  the stock node. It refuses on a cfg topology, with a two-checkpoint guider,
  and under sequence parallelism on any family but MOSS-Image, and Model
  Sampling SD3 (DGX Monarch) refuses when it follows the node with a step
  window other than 0 to 1
  ([TROUBLESHOOTING #126](docs/TROUBLESHOOTING.md#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider)). `benchmark/run_matrix.py` takes the same
  widgets as the optional `slg_double_layers`, `slg_single_layers`,
  `slg_start_percent` and `slg_end_percent` keys.
- Refuse a LoRA whose patch changes a weight's shape, on every family and
  topology, before any bake. ComfyUI master since e3bee7ffc can apply such a
  patch, for example the
  MOSS-Image color and depth control LoRA, which widens `video_embedder` from
  128 to 384 input channels; the bake used to fail on it with an untyped
  error. The refusal names the key and both shapes
  ([TROUBLESHOOTING #127](docs/TROUBLESHOOTING.md#127-a-lora-that-changes-a-weights-shape-refuses-before-any-bake)).
- Refuse Model Sampling SD3 (DGX Monarch) on a checkpoint whose ComfyUI
  config samples at multiplier 1.0, where the node's multiplier-1000 sampling
  renders noise: FLUX.1 Schnell, Chroma and Radiance, Z-Image and Lumina 2,
  OmniGen2, Boogu, Mage Flow, Qwen-Image, Qwen Image 2.1, Ideogram 4, Krea 2
  and Kroma, MOSS-Image and Kromoss, and Anima. Such graphs rendered noise
  before and now refuse before dispatch. The shipped graphs wire the node
  only to Wan and Kandinsky 5, whose configs sample at multiplier 1000, and a
  checkpoint whose config the driver cannot read is let through
  ([TROUBLESHOOTING #128](docs/TROUBLESHOOTING.md#128-model-sampling-sd3-dgx-monarch-refuses-on-a-family-whose-comfyui-sampling-multiplier-is-10)). Fleet KSampler runs the same check.
- Run every INT8 linear of the sequence-parallel MOSS-Image and FLUX.3 Klein
  forwards through comfy-kitchen's INT8 GEMM, as stock does. Since the
  fused-norm parity change, the rows `at_stock_rows` keeps carried the filled
  call's leading stride at batch 1. Outside inference mode, where a Worker
  samples, ComfyUI then sent the next linear on an INT8 weight down a route
  that dequantizes the weight and skips activation quantization: MOSS-Image's
  `qkv` and `ffn_up_gate`, and FLUX.3 Klein's `qkv` (or `q`, `k` and `v`) and
  `mlp_in`. On 2026-10-21 MOSS-Image INT8 ConvRot on uly2 read one-step NRMS
  0.027 to 0.036 against the pin-stripped one-GPU render for that reason.
  FLUX.3 Klein's INT8 uly2 row was measured before that change. The kept rows
  now carry a fresh tensor's strides, and the weight-only layouts (INT8 with
  or without ConvRot, ConvRot W4A4, W4A8 and W6A8) always get a 3-D
  activation as 2-D rows. A BF16 linear after the norm now takes stock's
  matrix product too, where it ran as a batched product of batch 1; NVFP4
  already got 2-D rows from ComfyUI. On 2026-10-21, after the change,
  MOSS-Image INT8 uly2 was bit-identical to the pin-stripped one-GPU render
  at 1024 and 1536 square and ran 0.78 to 0.79 s a step at 1024, against
  1.01 to 1.03 s before; FLUX.3 Klein's INT8 uly2 renders were bit-identical
  to the pin-stripped one-GPU render too, and every BF16 latent rerun was
  unchanged.
- Copy each component when the un-bake restore rebuilds a quantized weight.
  On a module that lives on the CPU, the rebuilt weight and scale shared the
  pooled read buffer that the reader refills for a later key, so an un-bake
  to an empty LoRA stack could leave one key holding another key's bytes. A
  module on the GPU already got a copy.
- Refuse a LoRA on a MOSS-Image INT8 file (SFT, Base or Kromoss) on every
  topology, class P, before any weight moves. On 2026-10-21 the Kromoss INT8
  ConvRot file with its DMD2 LoRA on uly2 read one-step NRMS 0.985 against
  stock ComfyUI. A probe of stock ComfyUI alone found the cause upstream:
  with DynamicVRAM off, stock's classic loader bakes the LoRA into the INT8
  weights and renders the pair about 1.0 NRMS from its own default, and
  dgx-monarch's bake is that classic bake. The driver checks
  before dispatch and Fleet KSampler before its first one, every Worker's
  model store before any LoRA build or bake, and the sweep matrix predicts
  those cells `refuse:P`. Stock ComfyUI on one GPU runs the pair with its
  default DynamicVRAM loader
  ([TROUBLESHOOTING #129](docs/TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)).
  The MOSS-Image quantization refusal now says why INT8 ConvRot is admitted
  under sequence parallelism.
- Promote MOSS-Image SFT, MOSS-Image Base and Kromoss to **HW** on the
  hardware ladder of 2026-10-21, run at dgx-monarch `ff8dab8300aa` with
  ComfyUI `d52a7a5ef` and the head capped at 2100 MHz, and re-measured on
  2026-10-21 and 2026-10-22 at `9caa6aae660b`
  ([VALIDATION](docs/VALIDATION.md#evidence-moss-image-sft)). BF16 uly2 was
  bit-identical to one GPU at one step and over the full 50 steps from 1024
  to 2048 square on SFT and at 1024 and 1536 square on Base and Kromoss, and
  on SFT to stock ComfyUI at one step from 512 to 2048 square; auto keeps it
  ([BENCHMARKS](docs/BENCHMARKS.md#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped)).
  INT8 ConvRot uly2 read one-step NRMS 0.027 to 0.036 against a pin-stripped
  one-GPU render at the first release and was bit-identical to it at the
  second. FSDP renders matched resident ones, so FSDP becomes supported. cfg2
  keeps refusing on the shipped graphs, which carry the skip; with the skip
  off it is exact on a prompt pair of different lengths and measured on a
  pair of equal length, so MOSS-Image gets no cfg2 row. ring2 is measured,
  not granted, and ring2 with a LoRA is not supported: every first-use Gate
  ceremony of that pair that reached a verdict failed. The Kromoss DMD LoRAs
  bake to stock ComfyUI's merge on every key, and their renders differ from
  stock's because stock applies a LoRA at cast time. The color and depth
  control LoRA refused on both ranks, and the next render ran with no
  Recycle.
- The four CFG 3 MOSS-Image templates, SFT, Base, the layout graph and
  Kromoss at 50 steps, use ComfyUI's `normal` scheduler in place of `simple`.
  `normal` lands within 0.0005 of the vendor pipeline's sigmas at 50 steps,
  where `simple` sits 0.069 off. On 2026-10-21 the SFT graph on `normal`
  rendered on uly2 bit-identical to one GPU and to stock; against `simple` at
  the same seed it read latent NRMS 0.140 and PSNR 31.3 dB, a different image
  of the same prompt. The 8-step Kromoss graph keeps `simple`, its card's
  recipe, and the MOSS-Image testing graphs keep `simple`, the schedule the
  sweep measured.
- The MOSS-Image INT8 LoRA refusal says what the probe found: dgx-monarch
  bakes the LoRA into the quantized weights as stock ComfyUI's classic loader
  does, and stock with DynamicVRAM off renders the Kromoss INT8 file with its
  DMD2 LoRA as far from its own default. Its remedy names stock's default
  DynamicVRAM loader
  ([TROUBLESHOOTING #129](docs/TROUBLESHOOTING.md#129-a-moss-image-int8-render-with-a-lora-refuses-on-every-topology)).
- The consent route takes an optional `source` field, `panel` or `studio`,
  and records it as the waiver row's `consent_source`. A body without it
  records `panel`, and any other value refuses with a 400. Studio's accept
  sends `studio`, so the ledger no longer reads it as a panel click
  ([threat model](docs/THREAT_MODEL.md#companion-to-driver-systemd-and-cli)).
  `studio` joins the frozen consent-source vocabulary, so the Gate protocol
  moves to 14 and each combination re-proves once on first use
  ([TRUST](docs/TRUST.md#protocol-invariants)).
- The `moss_image` adapter follows ComfyUI master's MOSS-Image forward of
  2026-10-22 (commit 60aade0bb), which moved each block's norm behind
  `modulate` and added three reference parameters to `_forward` and two
  arguments to `rope`. The sharded blocks route whichever global the block
  reads, the rope call follows all three forms, and the touchpoint registry
  admits the forward with and without the reference parameters, so the
  previous tree, d52a7a5ef, stays supported. In CPU tests the sharded
  text-to-image forward is bit-identical to stock on both trees; it has not
  been re-run on hardware ([MODELS](docs/MODELS.md#model-moss-image-sft)).
- Refuse MOSS-Image reference images, which the new Text Encode MOSS-Image
  Edit node attaches for edit LoRAs, class P on every topology but one GPU:
  no hardware test has compared a cluster edit with stock. The driver refuses
  before dispatch and the sharded forward on every rank before any exchange;
  `mode=local` with `gpus_per_host=1` and Fleet KSampler run ComfyUI's own
  forward
  ([TROUBLESHOOTING #130](docs/TROUBLESHOOTING.md#130-a-moss-image-render-with-reference-images-refuses-on-every-topology-but-one-gpu)).
- A driver started with ComfyUI's `--base-directory` stamps the commit of
  the ComfyUI checkout it runs. The stamp read Git state under the user-data
  root that the flag names, found none and read `unknown`, so the artifact
  parity check refused every render the driver sent to Workers, whose stamp
  names the real commit, and Gate lookups read every PASS as stale. The stamp
  and the runtime provenance snapshot now find the checkout one way, from the
  directory that holds ComfyUI's `folder_paths` module, so the telemetry and
  `dgxm top` show the right commit too.
- A one-GPU MOSS-Image edit computes its reference images' keys and values
  once per conditioning, as stock ComfyUI does, instead of on every model
  call. Stock caches them only in inference mode, which a Worker's sampling
  thread is not in, so on 2026-10-22 a one-GPU edit on hardware matched
  stock's latent at 1.46 times stock's time a step. The adapter now runs
  stock's `extra_conds` in inference mode on every load, as the FLUX.3
  adapter does, and stock's own rule still picks the branch. Off one GPU
  that step refuses reference images first, so no Worker computes them for
  a render it refuses
  ([ADAPTERS](docs/ADAPTERS.md#what-an-adapter-provides)).
- The `qwen_image21` adapter follows ComfyUI master's Qwen Image 2.1
  forward of 2026-10-22 (commit 503d785f0), which added a `prefix_cache`
  parameter for a writable prefix K/V cache. The touchpoint registry admits
  the forward with and without it, so the previous tree, a071826fd, stays
  supported, and the `import-against-comfy-master` canary passes again. In
  CPU tests on this rig the sharded forward is bit-identical to stock on both
  trees, with the cache off and on; it has not been re-run on hardware
  ([MODELS](docs/MODELS.md#model-qwen-image-2-1)).
- Refuse Qwen Image 2.1 cache device `auto` class P on Ulysses topologies
  under dynamic VRAM on ComfyUI 503d785f0 or later, where stock moved `auto`
  to the new writable cache and no hardware test has compared a sharded render
  with it. Every rank refuses alike before any exchange; `gpu`, `cpu` and
  `off` are not affected, and `auto` on a071826fd passes as before
  ([TROUBLESHOOTING #134](docs/TROUBLESHOOTING.md#134-a-qwen-image-21-ulysses-render-with-the-cache-device-auto-refuses-on-comfyui-503d785f0-or-later)).

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
