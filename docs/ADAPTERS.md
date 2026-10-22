# Adapter authoring guide

An adapter implements one model family (DESIGN.md §5.3). Family-specific
branches belong in adapters. Code outside `src/dgx_monarch/adapters/` must not branch
on model class.

## Independent implementation rules

Work from [DESIGN.md §5](DESIGN.md), the current ComfyUI model source, and the
public xfuser and Monarch APIs. Keep project logic independently expressed.
In each PR, identify the source used for each implemented behavior.

## What an adapter provides

```python
class MyFamilyAdapter(Adapter):
    family = "myfamily"
    model_base_classes = ("MyModelBase",)     # comfy.model_base names
    cfg_parallel_supported = True
    cfg_cond_padding = "none" | "pad" | "pad+mask" | "pad+text-mask"
    cfg_split_seam = "diffusion_model" | "apply_model"
    cfg_batch_constant = None | "<the CONDConstant key>"

    def inject_usp(self, diffusion_model, ctx): ...
```

`cfg_parallel_supported` defaults to `True`. Set it to `False` when the family
executes conditional and unconditional guidance as separate model calls, so no
batched call exists for cfg-parallel to split. `GPUWorker` then raises a typed
physics refusal immediately after adapter detection, before wrapper setup or
sampling; wrapper construction repeats the guard as a backstop. A family that
also sets `dual_model_cfg_supported` (Ideogram4) is exempt at cfg2 with no
sequence parallelism on a two-rank fleet without FSDP: there it runs the split
guider (dm-cfg2), one model per rank ([MODELS.md](MODELS.md)). The operator
remedy is `auto`, a family- and world-compatible topology with physical cfg
degree 1, or a true one-GPU run; do not name one ring degree as always
available. For supported families the wrapper slices every forward argument
whose tensors carry the batched call's dim-0 size, and two declarations decide
where it runs and when it can run at all.

`cfg_split_seam` names the comfy wrapper type the split installs on. The
family's forward must build the `DIFFUSION_MODEL` executor. If it does not,
the installed split wrapper is never called: every rank renders the whole
batch, preserving correctness but losing the parallel split. Read the family's
forward in the installed ComfyUI. If it calls
`get_all_wrappers(WrappersMP.DIFFUSION_MODEL, ...)`, keep the default; if it
does not, declare `"apply_model"`, the seam `BaseModel.apply_model` applies
for every family. The `cfg split` seam in
`tests/canary/comfy_seam_contracts.py` holds the declaration to the installed
comfy in both directions, so an incorrect declaration fails the CPU tests.

`cfg_batch_constant` names a per-cond `CONDConstant` the family publishes, such
as a caption's real token count. ComfyUI compares a constant by value, so two
prompts whose values differ are never folded into one call and no pad rule can
change that. Declaring it is optional. It makes the dispatch below certain for
a differing pair rather than decided per render, so the driver can skip the
pad and the sweep can label the cell without a tokenizer.

**The two cfg seams, and which one a render takes.** Every family on a cfg
topology installs both wrappers and the decision is per render, not per family:

| conds | seam | what each rank does |
|---|---|---|
| concatenate | `cfg_split_seam` | runs a contiguous half of the batched call's rows |
| do not concatenate | `WrappersMP.CALC_COND_BATCH` | runs one whole conditioning, alone |

The dispatch is in `adapters/cfg_dispatch.py`. ComfyUI hands every family's
cond list to one central seam before any model call, so the wrapper there can
give each cfg rank a contiguous group of that list, mask the rest to `None`,
and all-gather the per-cond outputs back into the list the sampler expects.
Stock `_calc_cond_batch` allocates one output per cond index before it reads
that cond, so a masked index returns exact zeros and each rank's model call is
the call a single GPU makes for that conditioning. Both ranks decide the same
branch from the cond list, the latent shape and `cfg_world()` alone, before any
collective, preventing one rank from gathering while another skips it. The slice wrapper
passes a dispatched call straight through: a rank running one cond owns every
row of it already.

**The fold rule is ComfyUI's own.** `can_concat_cond` in `comfy/samplers.py`
takes every conditioning against the first, and its `cond_equal_size` half
compares the keys and then asks each cond object's own `can_concat`. The
dispatch mirrors that over the cond list and calls those same methods, so the
shape rule for a regular cond, the repeat rule for a cross-attention cond of
unequal length, and the value rule for a constant are defined by ComfyUI. A
declared constant is checked first and determines the result
regardless of other conditioning fields. Otherwise, unsupported conditioning
features retain the slice path: an `area`, a `mask`, a `gligen` patch, a
timestep window, a hook group, or two conditionings on different control
objects. Those change what ComfyUI compares, or what a rank running one cond
alone would patch, and every family's cfg2 was measured on the slice.

ComfyUI's free-memory reading can differ across ranks. It controls how many
concatenable conditionings share a call, not whether they can concatenate.
The dispatch rule does not use it, so both ranks decide from identical inputs.

A family declaring a constant is never cond-padded: `equalize_cond_lengths` is
skipped for it in `actor/sample_protocol.py`. The pad could never make a
constant fold, and under the dispatch it would hand exactly one rank a padded
cond, whose forward can refuse where the other rank's returns and leave the
survivor alone in an all-gather. The skip reads a static family attribute, so
every rank takes it. `inject_cfg_pad_forward` stays installed and is a no-op
with no pad rows.

For concatenable conditionings, slicing retains the faster single two-row
call instead of two one-row calls. Dispatch runs only where slicing would
refuse. It costs one all-gather of `world * ceil(conds / world)`
latent-sized tensors per sampler step. No family declares the dispatch seam;
comfy applies it itself. The `cond dispatch` seam in
`tests/canary/comfy_seam_contracts.py` holds the stock properties it relies on,
the cond classes' own `can_concat`, and the pack's rule against ComfyUI's own
`can_concat_cond` on real conds.

A family that declares a pad rule for asymmetric cfg prompts (`"pad"` /
`"pad+mask"`) shapes how the driver equalizes cond and uncond so ComfyUI keeps
them in one batched call (`equalize_cond_lengths`, actor/sampling.py):
`"pad"` appends zero rows to the shorter text tensor; `"pad+mask"` also
attaches an additive attention bias. A `"pad"` family whose stock forward cannot
see the synthetic rows adds `inject_cfg_pad_forward`, a cfg-only forward that
finds those zero rows and, on a call that arrives without a mask, trims a
uniform all-pad tail rather than masking it, so the common cfg-split batch=1
case runs mask-free on the flash SDPA fast path. Lens's shipped template takes
the masked path instead: its GPT-OSS text encoder always returns a keep-mask,
which the driver extends over the pad rows and the forward passes to stock.
A float mask disqualifies flash, and masked SDPA measured about 3x slower
per attention call on GB10 (14.75 ms against 4.94 ms, VALIDATION.md).
The trim is exact for the kept image tokens because the
synthetic rows are removed before attention and never become keys. A ragged
batch may keep an additive bias over only the residual per-entry pad when the
stock path can honor it; maskless families reject the ragged case. Flux2 needs
no pad forward: ComfyUI front-pads its text conds to 512 tokens, so the concat
is already shape-equal.

A `"pad+mask"` family without its own `inject_cfg_pad_forward` lets its stock
forward apply the driver-attached bias as-is (Flux 1.x and LongCat, over the
joint [text, image] keys). That bias pays the masked-SDPA cost, and because
the mask sends the call to a different SDPA kernel than an unmasked one, it is
a fidelity gap against a true one-GPU call as well as a speed one: Chroma's
shipped pair read 0.09 bf16 and 0.16 fp8 against one GPU before its trim and
0.0 after. Chroma is the one `"pad+mask"` family that
adds the forward: `ChromaAdapter.inject_cfg_pad_forward` reads the real length
off the driver's own bias, its zero prefix rather than the context's content,
then trims the context and drops the bias before stock's `_forward` runs, so
stock gets the unpadded, maskless shape a true single-GPU call would carry. A
batch whose rows disagree on that length refuses rather than guessing one
shape for all of them.

An adapter may also define `prepare_base_model(base_model)`. The worker calls
it from `_inject_for_topology` on every load path (resident, the dual-model
slots, the Gate cycle and the FSDP Gate cycle), right after it selects the
adapter and before any sequence-parallel or cfg install, so it precedes the
first `extra_conds` of every sample; an adapter without the attribute is left
as it was. FLUX.3 and MOSS-Image use it to bind `extra_conds` on the base
model (their paragraphs below). A hook that binds through `Adapter.bind` on
the shared `BaseModel` instance survives the patcher clones a LoRA makes and
the baked hot-swap arm, and one that reads the class function replaces rather
than stacks on a repeat call.

`inject_usp` binds, on the live diffusion model instance:

1. **A sharded top-level forward.** Implement the family's distributed forward with
   `shard_seq` after the token embeddings and `sp_gather` before the output
   head. Every per-token tensor (RoPE freqs, per-token time embeddings,
   control adds) shards identically. Divisibility padding is zero-pad + trim
   as transport: every aligned per-token tensor receives the same pad and
   `sp_gather` removes it before the output head. A pad row is still a key:
   measured on Krea2, post-norm a zero embedding is a real key at
   RoPE (0,0,0), producing left-edge artifacts at odd token counts. cfg_pp
   samplers evaluate the short uncond even at cfg 1.0, so the surface is every
   render with an empty negative. A forward that excludes the pad rows passes
   `padded_row_indices(segments)` as `drop_rows`, through `usp_options` for a
   joint sequence or straight to a patched attention, and the Ulysses path
   then excludes exactly those rows. Ring and hybrid have no full-sequence
   point to drop them at, so each family names its rows on one of two scopes:

   * every topology, so a padded sequence on ring or hybrid raises a typed
     refusal instead of rendering known-wrong math: Krea2, Lens, MiniMax H3,
     LTX, Mage Flow, FLUX.3, MOSS-Image and the flux family (Chroma, Flux,
     Flux2, LongCat);
   * pure Ulysses only, so ring and hybrid attend the pad and never refuse:
     Qwen-Image, Hunyuan, Boogu, Ernie-Image and Ideogram4 (each in its own
     paragraph below).

   The listed families outside the flux-family forward wire the exclusion
   into their own forward and declare no `usp_pad_exclusion_probe`; that
   attribute applies to the flux family alone (below).

   That ring/hybrid refusal
   carries an expert class K waiver: default off, never automatic, output
   stamped rendered-under-waiver, one permanent ledger row per grant and per
   use ([docs/TROUBLESHOOTING.md #55](TROUBLESHOOTING.md#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means)). Under an accepted waiver the exact
   exclusion is skipped entirely rather than applied on one slice, because
   `drop_rows` indices address the whole SP sequence while a ring rank holds
   one ring slice, so the waived render attends the pad rows and remains
   mathematically incorrect. Unpadded sequences remain byte-identical. A family
   that shards two streams passes a second row set as `kv_drop_rows`: LTX pairs
   video queries with audio keys and the reverse, each stream pads on its own,
   and one row set applied to both sides would drop real keys at the other
   stream's coordinates. Adapters that pass no `drop_rows` keep their own
   measured contracts. Wiring another family onto the exclusion requires its
   own probe, which may be a named acceptance test while the family still
   has status `impl`: MiniMax H3's probe was included in its acceptance checks
   ([VALIDATION.md](VALIDATION.md)), FLUX.3's is the odd-grid case of
   `tests/test_flux3_comfy_exactness.py`, and MOSS-Image's is the odd-stream
   case of `tests/test_moss_image_comfy_exactness.py`. An adapter with a
   stronger fail-closed invariant may opt into `shard_seq(..., allow_padding=False)`. Omnigen2 does
   so for its joined ring stream. It has no exact pad-row exclusion evidence,
   so a non-divisible stream typed-refuses instead of attending a synthetic
   key. This is a capability reduction, not hardware evidence; see [TROUBLESHOOTING #77](TROUBLESHOOTING.md#77-a-chroma-omnigen2-or-z-image-usp-render-refuses-a-mask-or-sequence-length).

   In the flux-family forward, whether a family pads or refuses is one
   declared attribute rather than a family name written into the forward:
   `Adapter.usp_pad_exclusion_probe` names the probe whose measured result
   validates pad exclusion for that family, and defaults to `None`, which preserves the
   refusal. Chroma names `benchmark/reports/chroma_pad_fidelity_matrix.toml`
   and therefore pads, drops and restores like Krea2; Flux, Flux2 and LongCat
   run the same forward under one inherited
   attribute naming `benchmark/reports/flux_family_pad_fidelity_matrix.toml`.
   The benchmark sweep derives its label rule from that attribute too, so a
   family that gains a probe moves from a shard-time class P refusal to a
   render on pure Ulysses and to the waivable ring card on ring or hybrid,
   with nothing renamed. If a family's acceptance leg fails, remove its probe attribute so the
   shared forward refuses again. Chroma's measured legs read 1-step NRMS
   0.014 to 0.034 on four padded uly2 cells; the flux-family
   legs passed on every gate, each padded leg inside its
   own divisible companion's band in the same run ([VALIDATION.md](VALIDATION.md)).

   A family whose `single` preset is unusable (H3's derives dp, which its
   packed latent refuses) owns its own ring refusal in its adapter, so the
   remedy names something the operator can run; the generic
   `assert_ulysses_only_padding` in `base.py` stays underneath as the backstop
   and keeps its `uly*/single` wording for everyone else. Both guards are class
   K and both carry the same waiver, scoped per family so a grant for one
   family's guard can never silence another's.

   **RoPE alignment under sharding.** Compute each rank's positional embedding
   from that rank's local ID/position shard. Shard the IDs, then embed locally.
   Do not chunk an
   embedded full sequence: whenever the token layout differs from the layout
   the ids were embedded in (separately sharded streams attending jointly, a
   re-concatenated [txt, img] stream with a text length that does not divide
   the SP degree), the chunk hands the rank a contiguous run of the wrong
   positions and every image token shifts by the padding, producing a vertical
   artifact band with no error raised. `flux_family.py` shows both cases.

   **Flux-family double-stream order.** Chroma, Flux, Flux2 and LongCat double
   blocks share this path. Pure Ulysses restores canonical
   `[text, image]` order after its head all-to-all, before calling attention.
   The separately sharded streams otherwise arrive as
   `[text0, image0, text1, image1]`; that permutation preserves real-number
   attention algebra but changes Flash rounding enough to reproduce the
   measured Chroma fp8_scaled/int8_convrot failures.
   Query and key pad coordinates are remapped independently, and the inverse
   permutation restores the original rank layout before the return all-to-all.
   The permutation, its inverse and the pad exclusion's keep index depend
   only on integer row counts, so the order and the pad exclusion build them
   on the device once per layout and keep them in a small cache keyed by
   shape and device, which drops its least recently used entry past the
   bound (`adapters/index_cache.py`). The pad-row remap reads a host copy
   of the inverse, computed by the same arithmetic on the CPU, and the keep
   index comes from `arange` runs rather than a `nonzero`, so no attention
   call waits on the device
   ([VALIDATION.md](VALIDATION.md#padded-ulysses-index-cache)). A
   self-attention that names one row set for both sides remaps it once and
   shares one keep index between queries and keys, and the output restore
   reuses the cached inverse.
   The single-block stream is already canonical. Ring and hybrid topologies
   retain their existing order. With the order restored,
   every Flux 1 Dev, Flux2 fp8mixed and LongCat pad-matrix leg matches
   its dp2 reference exactly, and the Flux 1 Dev and LongCat
   templates matched one GPU exactly (VALIDATION.md).
2. **USP attention routing** for every attention over the sharded sequence:
   * blocks that are pure self-attention: thread
     `usp_options(transformer_options, ctx.usp_attention)` into the block loop;
     ComfyUI's `optimized_attention_override` hook completes the routing;
   * blocks that mix self- and cross-attention (wan, ltx): bind a replacement
     forward on self-attention modules only. Cross-attention needs no
     patch: local q against full/replicated k/v is already correct.

   The callable behind `ctx.usp_attention` comes from
   `make_usp_attention(kernel, sync_ulysses)`. The kernel name picks the
   inner attention (TORCH_FLASH validated on GB10, re-selectable per render
   for the fp8 sage gate). The Init `sync_ulysses` widget (default True)
   reaches `xFuserLongContextAttention.__init__` as `use_sync`, but the
   pinned xfuser 0.7.0 overrides `forward` and never passes `use_sync` to
   `SeqAllToAll4D.apply`, so the widget has no effect on the Ulysses
   all-to-all. The widget stays, because removing a node key needs a major
   release.

   Qwen Image 2.1 is an exception: its block-causal prefix path gathers
   target K/V and calls native Comfy attention, so its Ulysses forward
   rechecks the mutable dispatcher on every call and admits only
   `TORCH_FLASH`. Other selected kernels and a foreign
   `optimized_attention_override` refuse before sequence construction.

   Wan-Animate 2 likewise gathers frame-indexed K/V and calls native Comfy
   attention. Its outer forward admits only mutable `TORCH_FLASH` before patch
   embedding or collectives; other selected kernels and a foreign
   `optimized_attention_override` refuse there too.

   **Per-module attention preference and q/k/v containers.** ComfyUI lets a
   model file pin an attention kernel on each attention module. Models pass
   `preferred_attention=self.comfy_attention` to their attention call and box
   q, k and v in `AttentionTensorContainer`. Comfy's `wrap_attn` drops the
   keyword and takes the tensors out of their boxes before it calls
   `optimized_attention_override`, so the `usp_options` overrides see
   neither. An op or callback the pack installs on the model itself runs
   before `wrap_attn` and sees both. Anima's `attn_op` is one such op: it
   takes the boxes once and ignores the keyword
   (`adapters/attention_inputs.py`). It accepts no other keyword. The
   dispatcher drops an unknown keyword for an ordinary kernel, so the op
   refuses one with a class P refusal that names it
   (docs/TROUBLESHOOTING.md #116) rather than let it pass unapplied. Qwen
   Image 2.1's attention callbacks also receive both, but they do not refuse
   an unknown keyword as Anima's op does: the two callbacks its forward
   builds drop any keyword but the preference without a word, and the
   prefix-fill wrapper hands one to comfy's own callback, which accepts no
   keyword but the preference.
   Under sequence parallelism the Init `attention` widget and the dispatcher
   choose the kernel, so the pack logs the preference at debug level once
   per call site where the pack runs the attention and otherwise ignores it.
   Qwen Image 2.1 and Wan-Animate 2, whose
   distributed paths call native Comfy attention themselves, follow the same
   rule: they log the preference once and pass `preferred_attention=None`,
   so comfy's default backend runs, and these paths admit no widget value
   but `TORCH_FLASH`. A worker starts ComfyUI with no attention flags, so on
   NVIDIA, when the worker interpreter cannot import xformers, that default
   is torch's scaled dot product attention. With xformers importable, comfy
   picks xformers attention instead. For a call with at least 131,072 query
   elements, comfy's wrapper asks torch to try flash first, then cuDNN, the
   efficient kernel and math; it leaves smaller calls to torch's own order.
   Flash takes no mask, so Qwen's masked text rows run one of the other
   backends. Each of these is exact, so no approximate kernel runs under the
   `TORCH_FLASH` stamp, though the stamp does not name the one kernel that
   ran. Qwen's cache-miss prefix fill runs comfy's own
   `block_causal_attention` callback, which hands the preference on, so the
   adapter wraps that callback to pass none. A checkpoint that pins a kernel
   on its attention modules (the comfy-kitchen int8 kernel, or the
   comfy-kitchen Sol kernel, `comfy_kitchen_sol`, with its tau) therefore renders this
   attention with the widget's kernel where stock ComfyUI on one GPU would
   use the pinned one. Under sequence parallelism the pack asks for the
   class-K Sol-Attn waiver only when the widget selects Sol.

   Under sequence parallelism no attention call honors a module's pin. The
   pack also leaves some stock attention in place, and those calls run
   comfy's default backend, as stock runs them for a checkpoint with no pin;
   the widget picks only the sharded attention's kernel. A read of every
   adapter against ComfyUI e8665ca8 on 2026-10-14 found these sites; on
   2026-10-15 the inventory was checked again against 29e28e2b4, whose
   commits since e8665ca8 change none of the modules below except to add
   FLUX.3. On 2026-10-20 the MOSS-Image sites were added from d52a7a5ef; the
   other sites were not read again against that tree:

   * cross-attention: every block's `cross_attn` in Wan 2.1 and 2.2, Wan
     SCAIL, WanDancer and Wan-Animate 2; `attn2` in LTX, and `audio_attn2`
     as well in LTX audio-video; the visual blocks' `cross_attention` in
     Kandinsky 5; and Anima's cross-attention, which runs cosmos' stock
     `torch_attention_op`;
   * self-attention that every rank runs whole before the sharded block
     loop: the context and noise refiners in Z-Image; the context, noise and
     reference-image refiners in OmniGen2; Boogu's context refiner; the
     token refiner (`txt_in`) in HunyuanImage 2.1 and HunyuanVideo 1.5;
     Kandinsky 5's text transformer blocks; the text fusion blocks in Krea2
     and Kroma (all of them under pure Ulysses, the layerwise blocks under
     ring or hybrid); WanDancer's music encoder; and MOSS-Image's two
     text-embedder blocks;
   * text refiners that comfy runs from `extra_conds` on every rank, before
     the forward: MiniMax H3's token refiner, which also runs inside the
     forward when the text arrives unrefined, and the video and audio text
     connectors in LTX audio-video;
   * FLUX.3's five text blocks and five reference blocks, which comfy runs
     from `extra_conds` for the base model and at the start of the forward
     for the step-distilled one, in both cases without options.

   The forward hands each such module a copy of `transformer_options` that
   carries an override (`adapters/stock_attention.py`). Comfy's `wrap_attn`
   checks the override before the preference, and the override hands the
   call, with its arguments unchanged, to the backend function `wrap_attn`
   would run itself with no pin. An unpinned checkpoint therefore renders
   exactly as before, and a pinned one renders as stock does unpinned.
   `wrap_attn` removes the preference before it calls the override, so
   these sites cannot log the pin. Where comfy hands a module no options, as
   for the `extra_conds` refiners, the pack binds the module's forward to add
   the override itself; that bound forward sees the module and logs its
   preference once per site at debug level. WanDancer's music encoder passes
   attention no options at all, so the pack binds a copy of its forward that
   passes `preferred_attention=None` and logs the preference, as the Qwen
   Image 2.1 and Wan-Animate 2 paths do. These bound forwards log that
   comfy's default backend runs the call. The copy keeps an override the
   graph installed already, such as an attention backend node's, and
   `wrap_attn` runs it ahead of the pin, as stock does, where the forward
   admits one: LTX and Wan-Animate 2 refuse a foreign override before they
   build the copy, as Qwen Image 2.1 does
   ([docs/TROUBLESHOOTING.md #72](TROUBLESHOOTING.md#72-a-distributed-render-refuses-a-comfy-attention-patch-or-nag-style-hook)),
   and in the families that admit one the pack's sharded attention never
   runs it, so it runs only at these stock sites. One default backend would
   differ: with `--use-ck-attention` comfy runs the comfy-kitchen int8
   kernel, whose own path for boxed q, k and v the override would skip. A
   worker never passes that flag.

   The other families leave no stock attention call under sequence
   parallelism: their sharded forwards call the pack's kernel directly,
   install an override that `wrap_attn` checks before the preference, or,
   as in Qwen Image 2.1, hand comfy no preference.
   Every topology without sequence parallelism (its ulysses and ring
   degrees are both 1; on a two-rank pair that is single, cfg2, dp2,
   cfg2+fsdp and dp2+fsdp) leaves every stock attention call in place, so a
   pin runs there as it does in stock ComfyUI on one GPU, with the FSDP
   cases below. Dropping the pin there would break the cfg2
   same-math gate, which compares against one GPU, so the pack detects,
   records and waives it instead (`attention_pins.py`):

   * **Detection.** Once per fresh load, before injection and before
     ComfyUI stages any LoRA, the pack walks the loaded model and keeps every
     `comfy_attention` module whose `config` names a kernel, with the kernel
     ComfyUI bound in `function`. Injection, which includes the FSDP shard
     build, runs after that walk, so the pack then rereads only the pinned
     modules. Each render reads the cached walk and adds the pins a LoRA
     stages: ComfyUI's patcher stages a LoRA's `.config` key as an object
     patch at `diffusion_model.<module>.comfy_attention`, and `patch_model`
     puts it in place before the render. A model with no pin costs that one
     walk per load and a read of the patcher's few object patches per
     render; nothing runs per step.
   * **Record.** The render finish record carries `attention_pins` (pinned
     attention modules per kernel; Sol's tau is part of the kernel's name),
     `attention_pins_ran` and `attention_pins_dropped` (modules per reason a
     pin does not run). `attention_pins_ran` is true when ComfyUI dispatches
     every pinned module to its pinned kernel, as on one GPU; the kernel
     itself still runs exact attention on a masked call (Sol) or on a float32
     call with `low_precision_attention` off (int8). The count is per
     attention module, so a block whose self- and cross-attention are both
     pinned counts twice. The fields sit beside the `attention` fact, which
     still names the widget's kernel. The widget's kernel, which also feeds
     the gate ledger's capability context, is unchanged, so no combination
     gates again. The sidebar's last-render line and the
     `benchmark/run_matrix.py` console show them as, for example, "pinned:
     comfy_kitchen_sol tau=0.7 x 40 modules, ran"; run_matrix result rows
     carry the three fields. Under dual-model cfg2 each cfg rank renders its
     own checkpoint, so the fields add the two models' modules, as one rank
     holding both does. When both models pin, `attention_pins_models` keeps
     each model's own three fields under `cond` and `uncond`, run_matrix rows
     carry it too, and the line gives one clause per model, for example
     "pinned: cond model: comfy_kitchen_sol tau=0.7 x 20 modules, ran; uncond
     model: comfy_kitchen_sol tau=0.7 x 10 modules, dropped (kernel
     unavailable on this host)".
   * **Waiver.** Both kernels ComfyUI binds are approximate by construction.
     `comfy_kitchen_sol` attends a routed subset of key blocks exactly and
     pools the rest, so it is not identity preserving. `comfy_kitchen_int8`
     rounds q, k and v to signed 8-bit integers and the softmax weights to
     unsigned ones (comfy-kitchen's `int8_attention`; the rotation it applies
     first keeps q.k exact, and the rounding that follows does not). A SAGE
     kernel runs because the operator picked it on the Init widget, or
     because auto picked it on a measured combination. A pinned int8 kernel
     is neither: the checkpoint or LoRA picked it, and it runs on both legs
     of the cfg2 same-math gate, so that gate cannot see it. So it takes the
     waiver too. A kernel this project has not classified counts as
     approximate. When a pinned approximate kernel would run, the render
     refuses with class K under the guard `sol_attn:checkpoint_pin` before
     any sampling, and runs, stamped, once the operator grants the Sol-Attn
     waiver for that checkpoint, topology and world
     ([docs/TROUBLESHOOTING.md #117](TROUBLESHOOTING.md#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel)).
     The card says the kernel is approximate by construction and claims no
     measurement for it. The pack decides from the pins, the LoRA stack and the
     topology, never from whether ComfyUI could bind the kernel on this host,
     so every rank of a fleet reaches the same verdict. That matters under
     cfg2+fsdp and dp2+fsdp, which run no readiness exchange: a rank that
     refused alone there would leave its peer waiting at the first collective
     until the group timeout. On the other multi-rank topologies the check
     also sits inside the sample path's load guard, so a rank that refuses
     for any reason still crosses the readiness flag and its peer refuses
     with it. A render on one GPU has no peer, so it reads its own binding: a
     pin whose kernel ComfyUI could not bind there runs ComfyUI's default
     backend and asks nothing. A pinned exact kernel would only be recorded;
     ComfyUI ships none today.

   Under sequence parallelism the pack asks nothing and the record says it
   dropped the pins. The record names two other cases where a pin does not
   run. The FSDP shard build of a bf16 or fp16 checkpoint reloads the
   sharded weights through `load_state_dict`, and `ComfyAttention` clears
   its config on any load whose state dict lacks the key, so a pin that
   reached that build would not run, and the record would name it. No
   safetensors checkpoint whose own header pins reaches it today: its uint8
   config tensors leave the FSDP launch check unable to prove a bf16 or fp16
   core (`fsdp_islands.header_core_kind`), so every `+fsdp` preset refuses
   that file class P (`validate_fsdp_launch_quant`) before any shard build.
   An fp8 or int8 checkpoint takes the direct wrap, which reloads nothing, so
   its pins stay, run and ask for the waiver as on a resident topology. A
   LoRA's pin is
   expected to run under either wrap, since `patch_model` puts it in place
   after the wrap; no hardware run has checked that yet. A kernel ComfyUI
   cannot bind on a host leaves ComfyUI's default backend running there, as
   on one GPU.

   ComfyUI decodes the config tensor with `Tensor.numpy()`, which a GPU
   tensor refuses, so slab residency and the low-RSS LoRA staging keep these
   uint8 tensors on the host. Without that, a pinned checkpoint under slab
   residency and a pinned LoRA under low-RSS staging would each fail to load:
   a CPU reproduction shows the decode error, though no hardware load has
   checked either path yet. Slab residency prices a LoRA that holds only
   these tensors as no bake, because it patches no weight.

   Under dual-model cfg2 each rank checks the checkpoint it holds and its
   LoRA stack, and the dm-cfg2 readiness exchange makes the other rank
   refuse with it. The driver keys every grant of that render on the
   conditional checkpoint, so the card names it even when only the
   unconditional checkpoint pins, and the card's detail line names the
   checkpoint that pins.

   The first-use ceremony carries no waiver, so a ceremony on a pinned
   combination ends on this guard, as it does for every class-K guard, and
   changes no residency setting. On a resident topology a risky combination
   (a LoRA stack, or slab residency requested) then renders under stock
   residency, the limit TROUBLESHOOTING #84 describes for Sol. FSDP has no
   stock residency to fall back to. Under `auto_gate=first_use`, a render on
   cfg2+fsdp or dp2+fsdp needs a Gate PASS, and the ceremony that would earn
   one ends on this guard, so a pin that runs there (a LoRA's, or an fp8 or
   int8 checkpoint's) refuses even after the grant. Render it on uly2+fsdp,
   where dgx-monarch drops the pin, with `auto_gate=off`, or on a topology
   without fsdp. `tools/make_pin_probe_lora.py` writes a LoRA that pins a
   kernel on one family's attention modules, for a hardware check without a
   patched checkpoint.

   The "attention keyword handoff" seam in
   `tests/canary/comfy_seam_contracts.py` holds the Anima op, the
   `usp_options` override, the Qwen Image 2.1 attention and the Wan-Animate 2
   attention against the real ComfyUI tree, and with tiny real models the
   stock sites in the Wan 2.1, LTX, Kandinsky 5, Anima (on its cosmos DiT),
   Z-Image and Krea2 forwards, WanDancer's music encoder, MiniMax H3's token
   refiner, the LTX audio-video text connectors, FLUX.3's text and
   reference blocks, where a tiny real FLUX.3 model carries real config
   tensors on two early and two sharded blocks, and MOSS-Image's text-embedder
   and sharded blocks, where a tiny real MOSS-Image model carries a real pin
   on every block. The other sites (SCAIL,
   WanDancer cross-attention, LTX audio-video `attn2` and `audio_attn2`,
   OmniGen2, Boogu and the Hunyuan token refiner) have CPU tests with
   stand-in models (`tests/test_stock_attention_pins.py`). The same seam
   loads a tiny real Wan 2.1 model with two real config tensors and applies a
   real LoRA pin through ComfyUI's own patcher, and the detector must report
   both; it also checks the probe tool's key patterns against tiny real
   Krea2, Anima, Kandinsky 5, Wan 2.1, Qwen Image 2.1, FLUX.3 and MOSS-Image
   models, FLUX.3 for both the `flux3` and the `flux3_early` pattern and
   MOSS-Image for both `moss_image` and `moss_image_text`. The
   detection, record and waiver have CPU tests with stand-in modules
   (`tests/test_attention_pins.py`, `tests/test_attention_pins_wiring.py`).

   **Lens joint order, key bias and text rows.** Each Lens block joins
   `[image, text]`, so Ulysses' head all-to-all gathers `[image0, text0,
   image1, text1]` while stock attends `[image0, image1, text0, text1]`. Under
   pure Ulysses the forward passes `LensJointOrder`, a `RankMajorJointOrder`
   that lists the real rows in stock's order and the divisibility pad rows
   after them; Lens already names its pad rows as `drop_rows` in every mode,
   and the descriptor maps them to that tail. Stock Lens also always builds
   its joint key mask, all zeros for an unmasked prompt, and hands it to
   comfy's attention, so torch's SDPA runs a masked backend rather than flash.
   An all-ones mask is no mask in real arithmetic but not in bits. xFuser's
   kernels take no mask, so the forward passes stock's bias as one query-bias
   group over every real row, and the full-axis call runs comfy's own SDPA
   wrapper with it (`usp_query_bias`). A uniform trailing text pad stays in
   the sequence, its keys under stock's float32 minimum, because the kernel's
   bits depend on how many keys it reduces over; ring and hybrid trim it.
   Each block's `txt_qkv`, `to_add_out` and SwiGLU `txt_mlp` run on
   all text rows in stock's layout through the `chroma_text.py` helper;
   `txt_mlp` is wrapped whole, so its three Linears share one gather, and
   `to_add_out` gets stock's joint batch stride. Only `torch.nn.Linear`
   weights are wrapped, so the mxfp8 checkpoint keeps its sharded text GEMMs.
   Ring and hybrid get none of these and keep the ring pad refusal.

   The bias call replaces the kernel the Init node names for Lens's joint
   attention under pure Ulysses, as the two exceptions above do for their
   native paths, though without refusing another selection. The selected name
   never described stock Lens's kernel either: comfy's sage entry hands a
   masked call to `attention_pytorch` unless the installed sage accepts
   `attn_mask`, which sageattention 2.2.0 does not, so one-GPU Lens runs
   masked torch SDPA under comfy's pytorch and sage attention alike, whichever
   kernel the Init node names. The gate ledger and the capability context
   still record the selected name. Recording Lens as kernel-agnostic remains a
   maintainer decision pending hardware evidence.

   Hardware results: the shipped BF16 graph's uly2 one-step
   render matched one GPU exactly on the template route. A test on comfy's
   own `LensTransformer2DModel` in float32 shows every rank's kernel call
   receiving stock's queries, keys, values and bias for its heads in stock's
   order, and the output equal to stock's, with no mask, an all-ones mask and
   a trailing pad. On the CPU the output already matches once the order is
   restored for unmasked prompts; the bias and the text rows show only in the
   operands there. The image-stream GEMMs still run on half the rows, and
   comfy's masked SDPA runs on half the heads. At the shipped graph's shapes
   on GB10 the template reading, a uly2 one-step image equal pixel for pixel
   to one GPU's, shows no pixel change from either; no latent comparison and
   no other shape has a reading. The comfy-canary job runs
   these files against comfy master: the exactness test, which skips on a
   runner whose GEMM gives a row other bits at half the rows; a check that
   stock still hands its kernel the bias the forward rebuilds; and the text
   module selection.

   Performance cost (not timed), pure Ulysses only. Per block per model call: the order
   treatment copies q, k, v and the output once each and reads nothing back
   from the device; even streams take the full-axis call too. The text
   treatment adds three `all_gather` calls with their copies and a
   joint-stride buffer, beside the four
   all-to-alls each block already makes. The bias adds one small tensor cast
   per call, as stock casts its own, and one output copy, and the masked
   backend's speed against flash is unknown. The shipped graph makes 40
   model calls of 48 blocks.

   **Qwen-Image joint order, pad rows and text rows.** Each rank joins `[text,
   image]` from its own two shards, so Ulysses' head all-to-all gathers
   `[text0, image0, text1, image1]` while stock attends `[text0, text1,
   image0, image1]`. Under pure Ulysses the forward passes the same
   `RankMajorJointOrder` that Chroma and Mage Flow use, so the kernel sees
   stock key order. It also passes `padded_row_indices` for both streams as
   `drop_rows`, so the zero rows that padding adds are not attended and the
   kernel sees stock's key set. The image stream's length counts reference
   tokens, so an Edit render's pad follows image plus references. Ring and
   hybrid topologies get neither: they keep their existing order and attend
   the pad rows. Every pure-Ulysses render, padded or not, names an order, so
   it takes the full-axis call and never reaches xfuser's own. A shard also
   runs the text-stream Linears on half the text rows, and the BLAS can pick
   another kernel for another row count, so the bits can differ. Under pure
   Ulysses each double block's `add_q_proj`, `add_k_proj`, `add_v_proj`,
   `to_add_out` and both `txt_mlp` projections gather the text rows, run on
   all of them in stock's layout, and re-shard (the helper in
   `chroma_text.py`, shared with Chroma). `to_add_out` gets stock's joint
   batch stride. Only `torch.nn.Linear` modules are wrapped, so plain fp8 and
   bf16 weights are covered and checkpoints with quantization metadata are
   not. A block replaced through `patches_replace["dit"]` is wrapped too. The
   image-stream Linears stay sharded. The selector locates the six text
   Linears by attribute path. A missing path
   or a module that is neither a Linear nor a quantized Linear leaves that
   projection sharded and logs a warning once per process. The comfy-canary
   job checks those paths against ComfyUI master. The hardware
   rows measure all treatments together; they do not isolate which of the six
   projections changes bits at sharded row counts.

   Hardware results: under pure Ulysses the base 2512 FP8 file
   matched its dp2 reference exactly at 1328 and 1024 with odd and even
   image and text streams, and at batch 2 (VALIDATION.md); the 1328 shape with
   both streams odd read 1-step NRMS 0.047777 before the treatments, and the
   shipped 2512 FP8 template's uly2 one-step render read pixel-identical to
   one GPU on the template route. The tests
   pin the key set, key order, pad rows and Linear layout against stock's,
   but they run no CUDA kernel. The image-stream GEMMs (`to_q`, `to_k`,
   `to_v`, `to_out.0` and both `img_mlp` projections) still run on half the
   rows, an odd row count at the default 1328 canvas (6889 image tokens, 3445
   per rank), and the measured latents show they keep stock's bits there.
   Checkpoints with quantization metadata, whose text Linears stay sharded,
   and Edit-2511 have not been measured.

   Performance cost (not timed), pure Ulysses only. Per block per model
   call, with 60 blocks:
   six text `all_gather` calls (`add_q_proj`, `add_k_proj` and `add_v_proj` each
   gather the same activation, and the `txt_mlp` output projection gathers a
   four times wider input), each with a contiguous copy and a re-shard copy;
   `to_add_out` also fills a joint-stride buffer; the text GEMMs run on every
   row; the order treatment copies q, k, v and the output once each; and a
   stream with a pad row adds a row selection of q, k and v plus a zero-filled
   output and a scatter, which the default 1328 canvas triggers because 83 by
   83 is odd. That makes nine full-length copies per padded call. Pad rows
   cost no device to host sync: the order and keep indices come from the
   per-layout cache, and the pad-row remap reads its host copy of the
   inverse. None of these copies or gathers has a timing yet. Time a warm
   1328 render before and after on hardware. Candidate reductions meant to
   keep every bit, neither applied nor tested: one row selection that folds
   the order permutation into the keep index, five full-length copies instead
   of nine; and one gather of the q, k, v input with `txt_mlp` run whole on
   full rows, three gathers instead of six.

   **Ideogram4 pad row.** Ideogram4 packs `[text, image]` into one stream and
   shards it in contiguous chunks, so Ulysses' head all-to-all already hands
   the kernel stock's key order and the family needs no order descriptor. Its
   text projection runs on every text row before the shard, so it needs no
   full-row projection either. An odd packed total, which the shipped template
   makes (283 text rows plus 4096 image rows), adds one zero pad row at the
   tail. The family's modulation only scales and RMSNorm keeps a zero row at
   zero, so the row reaches the first block's attention as zeros. It is still
   a key there: its logit of 0 takes softmax weight, and its zero query takes
   the mean of the values, so from the second block on the row carries
   content. Under pure Ulysses the forward passes `padded_row_indices` for
   the stream as `drop_rows`, so the full-axis call drops that row and the
   kernel sees stock's key set. The unconditional model's image-only call
   takes the same path when its grid is odd. An even stream names an empty
   set and keeps xfuser's own call. Ring and hybrid name nothing and attend
   the pad row. For this family ring2 stays an explicit diagnostic that auto
   never picks: the tested FP8 pair's ring2 one-step comparison read
   NRMS 0.102 against the 0.100 limit (DESIGN.md Appendix A). FSDP shards
   weights, not the sequence, so uly2+fsdp is pure Ulysses and takes the same
   exclusion, while ring2+fsdp does not (and refuses at topology validation,
   DESIGN.md §5.4). A padded multi-prompt text mask is still dropped with a
   warning (mask rules below).

   Hardware results: under pure Ulysses the fp8-scaled
   conditional model at CFG 1 matched its dp2 reference exactly at 1024
   and 1040 with 283 and 284 text rows, padded or not, and under `uly2+fsdp`
   at 1024 (VALIDATION.md); the padded 1040 shape read 1-step NRMS 0.087852
   before the fix. On the template route the shipped dual-model graph at
   CFG 7 rendered one uly2 step pixel-identical to one GPU on bf16, fp8 and
   int8; its unconditional image-only call is even at 1024, so a padded
   image-only call has not been measured. The tests pin the key set, key order and
   pad row against stock's with stub kernels, and one runs comfy's own model
   at toy size in float32 against the sharded forward: every kernel call gets
   stock's q, k and v and the latent equals stock's exactly, on a CPU whose
   GEMM gives a row the same bits at either row count. On GB10 the block
   Linears still run on half the rows (2190 of 4379 at the template) and the
   head-split flash call runs at head dimension 256, and the measured latents
   show neither changes a bit at those shapes.

   Performance cost (not timed), pure Ulysses with an odd packed total. Per block per
   model call: a row selection of q, k and v, a zero-filled output and a
   scatter, about 100 MB of extra writes per block at the template's 4380 by
   9 by 256 bf16 rows. The query and key sides share one keep index from the
   per-layout cache, so the block waits on no device read. The all-to-all
   count is unchanged. Over the template's 34 blocks that is about 3.4 GB per
   conditional call, and a render makes 20 conditional calls; its
   unconditional calls are even at 1024 and pay nothing. Neither this path
   nor xfuser 0.7's own forward passes `use_sync`
   to the all-to-all. Time a warm template render before and after on
   hardware.

   **Krea2 joint order and text rows.** Krea2 shards text and image apart and
   each rank joins `[text, image]` for its single-stream blocks, so Ulysses
   gathers `[text0, image0, text1, image1]` while stock attends all text then
   all image. Under pure Ulysses the joint blocks pass the same
   `RankMajorJointOrder`, so the kernel sees stock key order, and the descriptor
   remaps the pad rows Krea2 already drops. On a shard the text path
   (`txtfusion`: two layerwise blocks, the projector and two refiner blocks;
   then `txtmlp`) would run its Linears and norms on half the text rows. The
   conditioning is replicated on every rank, so under pure Ulysses each rank
   runs stock's text path on all rows, with native attention in the refiner as
   stock has, and shards only its output. Nothing is wrapped, so the
   fp8_scaled and int8 checkpoints, whose Linears load through
   `mixed_precision_ops` and fall outside the `chroma_text` selection, take the
   same path as bf16: unlike Chroma and Qwen-Image, their text Linears see all
   text rows under pure Ulysses. Ring and hybrid keep the shard-first text
   path with refiner USP attention and the rank-major joint order, and still
   name their drop rows, so a padded render there meets the ring pad refusal.

   Hardware results: on TORCH_FLASH under pure Ulysses every
   gate matched its dp2 reference exactly: RAW BF16 at 1536 and 1448
   with odd and even text and image streams, a 440-row prompt, and batch-2
   calls with and without a text pad, and the shipped template on RAW BF16,
   RAW INT8 ConvRot and Turbo FP8-scaled (VALIDATION.md). Before the
   treatments the template read 1-step NRMS 0.015695 on INT8 and 0.034555 on
   FP8, and an odd-text BF16 gate 0.009848. Auto takes the SAGE route (row 11)
   for FP8 from 1.2 MP only where cfg2 on row 15 does not apply: above 2.36 MP,
   or at CFG 1 with no CFG++ sampler keeping the unconditional pass. Campaign K
   read that route on Turbo FP8 at 1152 to 1536 square at 1-step NRMS 0.030 to
   0.058 against one GPU, and its full-schedule latents did not equal the
   reference ([VALIDATION.md](VALIDATION.md#campaign-k-cfg2)).
   `tests/test_krea2_comfy_exactness.py` builds comfy's own `SingleStreamDiT`
   and shows that every joint and text attention and every text Linear gets
   stock's operands, and that each rank's output equals stock's on the CPU.
   The joint blocks' Linears and the final layer still run on local rows (4665
   of 9329 per rank for the shipped 1536 canvas and 113-token prompt), and the
   measured latents show they keep stock's bits at those row counts.

   Performance cost (not timed), pure Ulysses only. Per joint block per model call, the
   order treatment copies q, k, v and the output once each, with its
   permutation and inverse from the per-layout cache, and every joint call
   takes the full-axis path even with no pad row. A padded call reads nothing
   back from the device. With `compile_dit` on, the full-axis call is compiler-disabled, so
   every joint attention breaks the compiled block there, padded or not. The text
   path costs each rank the whole text compute once per call instead of half,
   adds no gather, and drops the refiner's eight all-to-alls and their pad
   selections. Its Linears do about 4.6 GFLOP per text token per call (two
   layerwise blocks over 12 layer rows, two refiner blocks and `txtmlp`), and
   each rank does about 2.3 GFLOP more per text token than on a shard,
   so the cost grows with prompt length: about 0.26 TFLOP per call for the
   113-token prompt and 1.0 TFLOP for a 440-token one. The shipped 1536 graph
   makes 16 model calls per render (8 steps; its 113-token prompt and 5-token
   negative cannot batch), so 28 blocks give 448 joint reorders per rank.
   Time a warm render at the 113-token and 440-token prompts, before and
   after, on hardware.

   **Hunyuan joint order, pad rows and text rows.** HunyuanImage 2.1 and
   HunyuanVideo 1.5 share one forward. Its double blocks join `[text, image]`
   from each stream's shard, so the all-to-all gathers `[text0, image0, text1,
   image1]` while stock attends `[all text, all image]`. Under pure Ulysses the
   double loop passes `RankMajorJointOrder`, as Chroma and Qwen-Image do, so the
   kernel sees stock key order. The text stream is everything that joins it
   before the shard: Qwen rows, byt5 glyph rows and Video 1.5's vision rows; the
   image stream counts reference rows. A render with no text rows names no
   order, since its gathered order is already stock's. The single loop shards
   one joined stream in contiguous chunks, which gather in stock order, so it
   takes options without the descriptor. Those are a copy of the double-loop
   options, as stock carries one dict through both loops. Under pure Ulysses
   each loop also names its own pad rows as `drop_rows`: the double loop the
   pads of the text and image streams, the single loop the tail pad of the
   joined stream. Qwen text has no length floor, so about half of all prompts
   pad; the shipped image template's prompt is 169 text rows (149 Qwen and 20
   byt5, counted with comfy's tokenizer), which pads both loops
   at 2048 by 2048. Ring and hybrid topologies get neither treatment: they keep
   their existing order and attend the pad rows. A shard also runs the four
   text-stream Linears of each double block (`txt_attn.qkv`, `txt_attn.proj`,
   `txt_mlp[0]` and `txt_mlp[2]`) on half the text rows, and a BLAS may pick
   another kernel for another row count. Under pure Ulysses each gathers the
   text rows, runs on all of them in stock's layout, and re-shards (the helper in `chroma_text.py`,
   with Hunyuan's own selector). `txt_attn.proj` gets stock's joint batch
   stride, which counts reference rows. Only `torch.nn.Linear` modules are
   wrapped, so the plain fp8 file and bf16 weights are covered and checkpoints
   with quantization metadata are not. A block replaced through
   `patches_replace["dit"]` is wrapped too.

   Hardware results: under pure Ulysses every gate matched
   its dp2 reference exactly: the FP8 HunyuanImage 2.1 file at 2048
   and 2080 with 52, 169 and 172 text rows, with the prompt as negative and
   with an empty one, and the FP16 720p HunyuanVideo 1.5 file at 1280x720,
   121 frames and 77 text rows (VALIDATION.md). Before the treatments the 2048
   template shape read 1-step NRMS 0.013841 and the Video gate 0.055786. The
   refiner and SR checkpoints have not been measured. The tests
   pin the key set, key order, pad rows and text Linear layout against stock's,
   but they run no CUDA kernel. One of them builds comfy's own `HunyuanVideo`
   at toy sizes in both layouts and checks, against stock's own forward, every
   kernel's queries, keys and values, every text Linear's rows and strides, and
   the exact output. That holds only on a CPU whose GEMM gives a row the
   same bits at a shard's row count as at the full one, so each case checks
   its own row counts first and skips where they differ. The ordinary CI job
   has no ComfyUI and skips the file; the comfy-canary job runs it. The
   hardware rows measure the three treatments together, at Hunyuan's widths
   (3584 and MLP 14336 for HunyuanImage 2.1, 2048 and MLP 8192 for
   HunyuanVideo 1.5), so they do not show which one each shape needs. The
   image-stream Linears (`img_attn.qkv`, `img_attn.proj` and both `img_mlp`
   projections) and the single blocks' `linear1` and `linear2` still run on
   half the rows, and the measured latents show they keep stock's bits at
   those row counts.

   Performance cost (not timed), pure Ulysses only. HunyuanImage 2.1 has 20
   double and 40
   single blocks. Per double block per model call: four text `all_gather`
   calls, each with a contiguous copy and a re-shard copy, plus a joint-stride
   buffer for `txt_attn.proj`; the four text GEMMs run on every text row on
   both ranks; the order treatment takes its permutation and inverse from the
   per-layout cache and indexes q, k, v and the output once each. A padded
   stream adds, in each block of the loop it pads, a row selection of q, k
   and v, a zero-filled output and a scatter; neither the keep index nor the
   order remap reads the device. The template's canvas and prompt pad both
   loops. Time a warm 2048 render before and after on hardware. The
   HunyuanVideo 1.5 720p checkpoint has 54 double blocks and no single
   blocks. Its template's prompt is 77 text rows (counted with comfy's
   tokenizer) and its empty negative 6, so only the cond call's text stream
   pads.

   **Boogu joint order, pad rows and instruct rows.** Boogu shards five
   streams. The noise and reference refiners each shard their own stream, the
   double blocks shard the instruct stream and the `[reference, noise]` image
   stream apart, and the single blocks re-shard the joined `[instruct,
   reference, noise]` stream; the caption refiner stays replicated. Each
   double block makes two attention calls on one `transformer_options`: the
   joint `[instruct, image]` attention, which Ulysses gathers rank-major as it
   does Qwen-Image's, and the image self-attention, which it gathers in stock
   order. Under pure Ulysses the double loop's override picks each call's
   options by its local query length (`adapters/boogu_ulysses.py`): the joint
   call gets a `RankMajorJointOrder` and both streams' pad rows, and the image
   call gets the image stream's pad rows alone. A joint pad coordinate can
   fall inside the image axis, where it would drop a real image row, so the
   two calls never share a set. The noise, reference and single loops name
   their own tail pad rows. The seven instruct Linears of each double block
   (`instruct_to_q`, `instruct_to_k`, `instruct_to_v`, `instruct_out` with
   stock's joint batch stride, and the three `instruct_feed_forward` Linears)
   run on every instruct row through the helper in `chroma_text.py`. The
   selection takes `torch.nn.Linear` and comfy's mixed-precision Linear with
   an fp8 layout, which the shipped fp8_scaled checkpoint loads through.
   ComfyUI quantizes that layer's input against the checkpoint's
   `input_scale`, or 1.0 when there is none as in that file, so a full-row
   call quantizes every row as stock does; layouts that scale from the input
   itself, NVFP4 and MXFP8 among them, keep the shard
   (tests/test_boogu_quant_projections.py).
   Ring and hybrid get none of this: they keep rank-major order and attend
   their pads.

   Hardware results: with the fp8 instruct Linears selected,
   every uly2 gate of the shipped fp8_scaled file matched its
   dp2 reference exactly at 1024 and 1040, odd and even streams, empty and long
   prompts; before that step six of seven gates read 0.0116 to 0.0177
   (VALIDATION.md). The shipped template's one-step render on the sweep route
   matched one GPU exactly on the same build. ComfyUI's own
   `BooguTransformer2DModel` at toy size in float32 runs two ranks equal to
   its stock forward exactly, over odd and even row counts in every
   sharded stream, with and without reference latents, at batch 1 and 2
   (tests/test_boogu_comfy_exactness.py). The kernel there is comfy's SDPA on
   both sides, so the test proves operands and order, not a GPU kernel. Two
   things stay untreated: the image-stream, `to_out.0` and single-block
   Linears run on half the rows, and the sharded kernel attends K and V
   expanded to 28 heads where stock passes 7 with `enable_gqa`. The latents
   show neither changes a bit at the measured shapes.

   Performance cost (not timed), pure Ulysses only. The template renders 25 steps at cfg
   4.0 as two model calls a step, because the 109-row cond and the 66-row
   uncond never batch, and each call runs 8 double, 32 single and 2 + 2
   refiner blocks. In every call each double block's joint attention takes the
   full-axis path, copies q, k, v and the output once each for the order, and
   reads the permutation and its inverse from the per-layout cache, which
   `restore` reuses. Every padded call selects the real rows of q, k and v,
   fills a zeroed output and scatters into it, and reads nothing back from
   the device. At 1024 the cond call pads the instruct stream, so each double
   block's joint call and each of the 32 single blocks over the 4205-row
   stream pay that; the uncond call pads nothing. At 1040 the template prompt
   pads both double streams, so each joint call, each image self-attention
   and each noise refiner block pay it, and the 4334-row single stream pads
   nothing. For
   selected Linears, including the shipped fp8_scaled checkpoint, the instruct
   wrapper adds seven `all_gather` calls per double block and runs those GEMMs
   on every row on both ranks. Three gathers duplicate earlier work because
   `instruct_to_q`, `instruct_to_k` and `instruct_to_v` read one tensor and
   `linear_1` and `linear_3` another, so a per-block gather cache in a new
   helper would cut them to four. At batch 1 neither `.contiguous()` call in
   the helper copies; `instruct_out` still copies into its joint-stride
   buffer, and an odd instruct stream adds a pad copy to each re-shard. Time a
   warm template render before and after on hardware.

   **Ernie-Image pad row and q/k branch.** Ernie shards one joint `[image,
   text]` stream as one chunk, so Ulysses' head all-to-all already hands the
   kernel stock's key order and the forward names no order. An odd joint
   length takes one zero pad row, and after modulation that row is a real key.
   At 1024 square the 4096 image rows plus the one-token empty negative make
   4097, so the template's uncond call meets it in all 36 blocks. Stock's
   block calls attention with no options, so under pure Ulysses the forward
   computes `padded_row_indices` for the joint stream before the layer loop
   and the bound attention passes it as `drop_rows`; an even length names no
   row and keeps xfuser's own call. Under pure Ulysses the bound attention
   also builds q and k on stock's own branch. At inference stock hands the
   cast norm weights to comfy-kitchen's fused `rms_rope_split_half`, while the
   unfused body runs RMSNorm then `apply_rope_split_half`, stock's training
   branch, and the fused kernel's own reduction can round differently on
   CUDA. The sharded rotary tensor is made contiguous there, as stock's is,
   and so are the token shard before the first block and the gathered rows
   before the final norm: at batch 2 the shard's chunk and an odd stream's trim
   are strided views where stock's norms read contiguous rows. On CPU the
   strided reads give the same bits; on CUDA nothing has measured them. Ring
   and hybrid get none of these: they attend the pad row and keep the unfused
   branch and the strided views. No Linear is wrapped: every block Linear runs
   on the joint shard, and none reads a short text stream.

   Hardware results: under pure Ulysses the BF16 file matched
   its dp2 reference exactly at 1024 and 1040 for the conditional
   call alone, the template pair and an equal-length pair in one batch-2 call,
   and under `uly2+fsdp` at 1024 (VALIDATION.md); the 1040 conditional call
   read 1-step NRMS 0.015621 before the treatments. A real-model test builds
   comfy's `ErnieImageModel` at toy size, shows every kernel gets stock's q, k
   and v and the output matches stock with max abs difference 0.0 on an
   aarch64 CPU, and shows each rank makes stock's fused call. The comfy canary
   runs it against comfy master on x86, where a value fed by a GEMM only has
   to agree to float32 rounding. On CPU the fused and unfused kernels agree
   exactly, so the CUDA rounding difference is inferred, not shown. The
   seven block Linears (`to_q`, `to_k`, `to_v`, `to_out.0`, `gate_proj`,
   `up_proj`, `linear_fc2`) run at 2049 to 2118 rows against stock's 4097 to
   4236 at widths 4096 and 12288, and the measured latents show they keep
   stock's bits there. The hardware rows measure all three treatments together
   and do not isolate which each shape needs.

   Performance cost (not timed), pure Ulysses only. An even joint stream
   adds nothing per
   block. An odd one adds, per block per call, a row selection of q, k and v,
   a zero-filled output and a scatter; its keep index comes from the
   per-layout cache and reads nothing back from the device. The fused branch
   runs one kernel where the unfused one ran three. The rotary and
   token-shard copies happen once per forward and the final-norm copy once
   per odd forward, all at batch 2 only.

   **Mage Flow joint order, pad rows and text rows.** Mage Flow runs
   Qwen-Image's double block, so the same three departures apply. Under pure
   Ulysses the forward passes `RankMajorJointOrder` and both streams'
   `padded_row_indices`, so the kernel sees stock's keys in stock's order with
   no pad row, and the image stream counts reference rows. Unlike Qwen-Image,
   Mage names its drop rows on every topology, so a ring or hybrid render that
   pads meets the ring pad refusal. Each double block's `add_q_proj`,
   `add_k_proj`, `add_v_proj`, `to_add_out` and both `txt_mlp` projections,
   the six Linears Qwen-Image wraps, gather the text rows, run on all of them
   in stock's layout, and re-shard through the helper in `chroma_text.py`;
   `to_add_out` gets stock's joint batch stride. The three `add_*`
   projections take one input, so the forward hands the helper a gather that
   gathers it once. Every text Linear with unquantized weights is wrapped:
   bf16, fp16 and plain fp8 weights, and the dense layers inside a quantized
   file, which comfy loads through its mixed-precision Linear with no
   `layout_type` (the BF16 `to_add_out` of the MXFP8 and NVFP4 native-view
   files, the BF16 layers of FP8 mixed and mixed NVFP4). A Linear with a
   quantized `layout_type` keeps its shard; for NVFP4 the row scope in
   `mage_nvfp4_scale.py` keeps pad rows out of its activation scale. A block
   replaced through `patches_replace["dit"]` is wrapped too. The image-stream
   Linears stay sharded.

   Hardware results for T2I: under pure Ulysses every gate
   matched its dp2 reference exactly: the BF16 file at 1024 and 1040
   with 27 and 28 text rows, an equal-length pair, a long prompt, batch 2,
   `uly2+fsdp` and a plain FP8 weight cast, and the Turbo INT8 ConvRot file
   at 1024 (VALIDATION.md); the BF16 1040 gate read 1-step NRMS 0.049301
   before the treatments. At 1024 the MXFP8 native-view and FP8 mixed files
   read 0.081165 and 0.028193 while their unquantized text Linears stayed
   sharded, and identical once those layers also ran on all text rows;
   the NVFP4 native-view and mixed NVFP4 files read identical both ways.
   A real-ComfyUI test runs the stock model and the two-rank forward and
   finds the kernel operands, the wrapped Linear operands and the output
   equal to stock's on the CPU. The image GEMMs at half rows keep stock's
   bits at the measured canvases; Edit graphs with reference latents have
   not been measured.

   Performance cost (not timed), pure Ulysses only. Per block per model call: four text
   `all_gather` calls (one for the `add_*` input, one for `to_add_out` and
   one for each `txt_mlp` projection, the `net[2]` input four times wider),
   six text GEMMs on every text row, and the joint-stride fill for
   `to_add_out`. Each gather takes up to one contiguous copy, and each GEMM
   output up to one pad copy and one re-shard copy; none of these copies
   happens at batch 1 with no text pad row. A quantized file does this work
   only for its unquantized text Linears. A render runs 12 blocks per model
   call and two calls per step when the two prompts differ in length, as the
   shipped T2I pair does.

   **FLUX.3 Klein two-phase shard, spans and early streams.** FLUX.3 runs
   five image blocks over the image rows, then 28 single blocks over the
   joint `[image, references, text]` stream, and modulates by span: each span
   is a `(start, end, modulation)` row range, the first modulates every row
   and each later one redoes its own rows. The sharded forward replaces
   `Flux3._forward` and calls stock's method for every computation of the
   prelude (the time, image, id and rope embeddings), so only the
   orchestration is the pack's. The image phase shards the image rows in
   contiguous chunks and gathers them after the five blocks; the joint phase
   rebuilds the stream and shards it again in contiguous chunks. Ulysses' head
   all-to-all then hands the kernel stock's key order with the pads at the
   tail, so no order descriptor is needed (the Ideogram4 and Boogu shapes).
   Each rank clips stock's spans to its own rows: the first span is kept
   even when it covers no local row, because the block reads its modulation
   for every row, and a later span with no local row is dropped rather than
   handed to comfy-kitchen's fused `adaln` as an empty launch. That fused
   `adaln` picks its CUDA kernel by a call's kernel rows (batch times rows),
   so a rank's piece of a span normalized in a call sized to the piece can
   round differently from stock's call for the whole span. Each of the 33
   sharded blocks therefore gets a per-instance `forward`
   (`adapters/flux3_adaln_parity.py`) that runs stock's own `Block.forward`
   code with its `modulated_norm` global resolved to a wrapper: each call is
   filled with zero rows to stock's row count for its span (the whole stream
   for the first call, the stock span for each later one), runs once, and
   keeps the rank's rows. The op is row-local, so those rows are stock's bits
   whatever row count comfy-kitchen branches on. The class, its module and
   the early blocks are untouched, and a stock block that stops reading that
   global runs unrouted with one warning, which the CPU exactness test turns
   into a failure. Every shard
   chunk is made contiguous, since the block writes its gated residual in
   place and stock's kernels read contiguous rows. Ids shard with the same pad
   and chunk as their tokens and embed locally through stock's `rope`. The
   final layer runs on the gathered joint stream's image rows, stock's call on
   stock's rows.

   Pad rows are named as `drop_rows` on every topology. A pad row takes the
   first span's shift, so it is a nonzero key after the first `adaln`; under
   pure Ulysses the full-axis call drops it, and a padded ring or hybrid
   render refuses once, before the first block, through the generic
   `assert_ulysses_only_padding`, whose `uly*` or `single` remedy FLUX.3 can
   run. ComfyUI pads the text stream to a multiple of 80 tokens with real
   tokens the model attends, so text never makes a stream odd at an even
   degree. An odd image grid pads the image phase, and at world 2 the joint
   phase pads when the joint total is odd: a text-to-image render on an odd
   grid, or an edit whose second or later references make the total odd. One
   reference doubles the image rows, so a one-reference edit pads only its
   image phase. An edit takes its grid from the first reference, so a 4:3
   source pads where a square one does not.

   The five text blocks and five reference blocks stay in stock's
   `preprocess_conds`, replicated on every rank: the text stream is short, and
   sharding it would add all-to-alls and run its Linears at half rows. ComfyUI
   calls those blocks without options, so under sequence parallelism the
   adapter binds each to comfy's default attention (the stock sites above);
   the INT8 files pin none of them. On every load `prepare_base_model` binds
   `Flux3.extra_conds` to run stock's method under `torch.inference_mode()`,
   as stock ComfyUI does when it executes a prompt.
   A Worker samples on its own thread, where inference mode is off, so without
   the bind the base model would rerun its early blocks on every step at the
   call's batch. Stock's own predicate still decides the branch: the base
   model computes its text and reference states once per conditioning, and
   the step-distilled flow map, whose early modulation depends on each step's
   target sigma, keeps stock's per-step path, as does any model with
   conditioning hooks. The bind clones every inference tensor it returns, so
   no later in-place write meets one. It reads the class function, so a
   repeat bind replaces rather than stacks, and it sits on the shared
   `BaseModel` instance, so LoRA clones and the baked hot-swap arm keep it;
   the worker installs it before any sample's first `extra_conds`. It caches
   nothing stock does not: no denoising state is reused across steps. On the
   step-distilled model every rank reads the whole sampler schedule from
   `transformer_options["sample_sigmas"]`, and a call without it refuses
   class P before the first block, where stock would raise `KeyError`. A
   Ulysses degree that does not divide the 24 heads refuses class P at
   injection. FSDP wraps the single blocks and `content_mode_blocks.image`
   per block and keeps the early streams whole through the class-scoped
   census (below).

   Hardware results (2026-10-15 and 2026-10-16): uly2 one-step latents
   equaled stock ComfyUI on one GPU at 1024 and 1536 square and on the odd
   1040 and 1552 grids, for both checkpoints. The five image blocks and the
   28 single blocks still run their Linears on a shard's rows, and the
   measured latents show they keep stock's bits at those row counts for text
   to image. One-reference edits read one-step NRMS 0.019 to 0.055 against
   stock, under the floor but not identical, while the two-reference edit,
   which pads both phases, was identical. A probe on 2026-10-16 traced the
   edits, and uly2 text to image at 704 square and below, to the fused
   `adaln` kernel switch above; the parity route makes them bit-identical to
   one GPU, confirmed on hardware on 2026-10-16
   ([VALIDATION](VALIDATION.md#evidence-flux-3-klein)). On the CPU,
   `tests/test_flux3_comfy_exactness.py` builds ComfyUI's own `Flux3` at toy
   size in float32 and shows every kernel call of the two-rank forward
   receiving stock's q, k and v for its heads in stock order with no pad row,
   and the output equal to stock's: an even grid, an odd grid that pads both
   phases, one and two references, batch 2, the flow map and the fused q, k
   and v layout the INT8 files use. That proves operands and order, not a GPU
   kernel.

   Performance cost, sequence parallelism only. Per model call:
   four all-to-alls in each of the five image blocks, one all-gather of the
   image rows (about 25 MB at 1 MP in bf16) and one final all-gather of the
   joint stream, beside the 28 times four joint all-to-alls every Ulysses
   family pays; a padded phase adds a row selection and a scatter per block.
   The fused-norm parity route gives each rank stock's full `adaln` work, the
   first call of every block over the whole stream: op-level timings put it
   at 1 to 31 ms per model call, about 2 to 4 percent of a measured uly2 step
   at 1024 square, pending the hardware walls.
   Every rank runs the replicated prelude whole. The step-distilled edit also
   runs the five reference blocks every step on every rank, about 14 percent
   of each rank's work with one 1 MP reference; the base model computes them
   once per conditioning. By block rows, an estimate and not a timing, uly2
   leaves each rank half of one GPU's work for text to image and the base edit
   and about 54 percent for the step-distilled edit. Measured on 2026-10-15 and
   2026-10-16, with the first Spark capped at 2100 MHz, uly2 sampler walls ran
   1.33x to 1.57x faster than one GPU on the base model, 1.41x to 1.55x on the
   step-distilled model and 1.50x on the step-distilled edit
   ([BENCHMARKS](BENCHMARKS.md#flux3-klein-one-gpu-uly2-and-cfg2-2026-10-15-to-2026-10-16-head-capped)).

   **MOSS-Image single stream, shared modulation and text embedder.**
   MOSS-Image runs 42 identical blocks over one joint `[image, text]` stream,
   the image rows first. One modulation per model call comes from the
   timestep: `shared_block_modulation` gives six chunks that every block reads
   for every row, and `video_head_modulation` gives the head's pair. The final
   RMSNorm, modulation and linear run on the image rows only. There are no
   spans, so FLUX.3's span clipping does not apply: every rank's block call
   reads the same `(B, 1, dim)` vectors. The sharded forward replaces
   `MOSSImage._forward` and calls stock's methods for the whole prelude, on
   stock's rows on every rank: the rotary table for all N rows, the time
   embedding, both modulations, the image embedding and the text embedder,
   whose two blocks run whole on the last T rows of the table, as stock runs
   them. The text stream is short, and sharding it would add all-to-alls.
   Under sequence parallelism those two blocks are a stock attention site
   (above), and the INT8 files pin neither of them.

   The joint stream and its rotary table then shard together, one contiguous
   window per rank (`adapters/moss_image_seams.py`), so Ulysses' head
   all-to-all hands the kernel stock's key order with the pads at the tail and
   no order descriptor is needed. Each block runs stock's own `Block.forward`
   code on its window, and the head runs after the gather, stock's call on
   stock's rows. MOSS-Image calls comfy-kitchen's fused `rms_adaln` twice per
   block and its fused q and k norm and RoPE, `rms_rope_split_half`, once in
   every block. Each block gets a per-instance `forward` that runs stock's code
   with the global it reaches the norm by sent through FLUX.3's `at_stock_rows`
   (`adapters/flux3_adaln_parity.py`): `modulated_rms_norm` before ComfyUI
   60aade0bb (2026-10-22), and `modulate` since, stock's own function, which
   with no reference split makes the same norm call with the same scale and
   shift. The block's `residual`, which then makes the same gated-residual
   call, is row-local and not routed. Both norm calls are filled with zero
   rows to the stream's N rows, the rank's window at the front, run once, and
   keep the rank's rows, so each launch is stock's launch whatever row count
   comfy-kitchen branches on. The q and k call is not routed: on 2026-10-20,
   with comfy-kitchen 0.2.39, it returned stock's bits on every window with no
   fill ([VALIDATION](VALIDATION.md#evidence-moss-image-sft)). A fused-norm
   call stock's block does not make, another call count or a missing norm
   global refuses class P, and canary seam 43 names the change. The kept
   rows go on with a fresh tensor's strides, as stock's norm output does
   (`adapters/int8_linear_layout.py`): outside inference mode, where a
   Worker samples, ComfyUI picks a quantized linear's route by its input's
   strides, and rows that kept the filled call's stride sent `qkv` and
   `ffn_up_gate` on an INT8 file down a route that dequantizes the weight
   and skips activation quantization. On 2026-10-21 that read one-step NRMS
   0.027 to 0.036 against a pin-stripped one-GPU render; with fresh strides
   INT8 uly2 was bit-identical to it, and the INT8 linears run comfy-kitchen's
   INT8 GEMM as stock's do.

   An odd joint total pads at world 2, from an odd image grid or from a
   prompt length that makes N = HW + T odd. A pad row takes the shared shift,
   so it is a nonzero key after the first norm; pad rows are named as
   `drop_rows` on every topology, and a padded ring or hybrid render refuses
   class K once, before the first block, through FLUX.3's
   `refuse_padded_ring`. Under sequence parallelism the loop sets
   `block_index`, `total_blocks` and `block_type="single"` as stock does and
   honors a `patches_replace["dit"][("single_block", i)]` replacement only
   when it is exact on a window: an identity function, read from its
   bytecode, which is what ComfyUI's Skip Layer Guidance installs, or a
   callable marked `sequence_shard_local`. Any other replacement refuses
   class P before any compute, and so does a marked one that hands the block
   other rows than the rank's window or its rotary rows. The adapter declares
   `sp_honours_block_replace = True`, the attribute the Skip Layer Guidance
   carrier reads before it admits a skip under sequence parallelism
   ([TROUBLESHOOTING #126](TROUBLESHOOTING.md#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider)).
   The prefetch queue mirrors stock's, with its block-scope pops, and the
   malloc graph stock opens around its block loop is not opened, as in
   FLUX.3's sharded forward.

   Injection checks the live tree against stock's MOSS-Image on every path,
   the forced `family_adapter` path included, and refuses class P where it
   differs, as `matches()` refuses a `MOSSImage` subclass. It chooses the
   `rope` call once, from the loaded class's signature:
   `(h, w, text_len, device)` in the 0.42 releases, the form that adds
   `transformer_options` on ComfyUI master since b9b290e43 (2026-10-19), which
   reads `rope_options`, and the form that also adds `ref_shapes` and
   `f_offset` since 60aade0bb, called as stock calls it with no reference: an
   empty shape list and offset 0. Any other form refuses class P. A Ulysses degree that
   does not divide the 24 heads refuses class P at injection. Under sequence
   parallelism a latent wider than the checkpoint's, which only a color or
   depth control adapter feeds, refuses class P
   ([TROUBLESHOOTING #127](TROUBLESHOOTING.md#127-a-lora-that-changes-a-weights-shape-refuses-before-any-bake)),
   and so does a forward in which a block attended outside the sharded
   override. The sharded forward keeps FLUX.3's attention-hook and keyword
   guards. Since 60aade0bb `_forward` takes the reference images of edit
   LoRAs as three parameters of its own; the sharded forward names them, and
   non-empty reference latents or cached reference keys and values refuse
   class P on every rank before any exchange, as the driver refuses them on
   every topology but one GPU before dispatch
   ([TROUBLESHOOTING #130](TROUBLESHOOTING.md#130-a-moss-image-render-with-reference-images-refuses-on-every-topology-but-one-gpu)).
   A block called with no row count, as stock's `preprocess_refs` calls each
   block, runs stock's own code with its reference arguments.
   single, cfg2 and dp2 bind no forward and run stock's. On every load
   `prepare_base_model` binds `MOSSImage.extra_conds` to run stock's method
   under `torch.inference_mode()`, as FLUX.3's does (its paragraph above).
   Stock computes the references' keys and values once per conditioning
   through `preprocess_refs` only in inference mode, with no hooks and a
   method other than `index` or `index_timestep_zero`; otherwise its forward
   recomputes them on every model call. Without the bind a one-GPU edit took
   that path on a Worker: on 2026-10-22 it matched stock's latent at 1.46
   times stock's time a step. Stock's predicate still picks the branch, the
   keys and values are stock's own, and every inference tensor in the output
   is cloned. The adapter declares `cfg_cond_padding = "none"`: every image
   row's rotary position depends on the prompt length and the joint attention
   has no mask, so no driver pad is exact, and a prompt pair of different
   lengths dispatches one condition per rank.
   On uly2, ring and cfg2 only BF16 and INT8 ConvRot run
   (`sp_validated_quants`); NVFP4 would reduce its shared activation scale
   over a rank's pad rows. FSDP needs no table row: `layers` is a standard
   block list, so the 42 blocks wrap one by one and the text embedder and
   every other module fall in the root group, and with no reference image
   `extra_conds` runs no diffusion-model module. Since 60aade0bb it runs
   `preprocess_refs`, every block, for reference images, and reference
   images refuse off one GPU, so the family's one outside-forward row is in
   the refused-call table, which replicates nothing. The block list is
   not named `blocks`, so `compile_dit` compiles nothing for this family and
   keeps requested slab residency.

   Hardware results (2026-10-21): uly2 one-step latents equaled stock
   ComfyUI on one GPU from 512 to 2048 square, on both joint parities where
   both ran, on the odd 1040 grid and at batch 2, for SFT, Base and Kromoss
   where each ran, and full 50-step renders equaled one GPU from 1024 to 2048
   square with the skip on, and at 1024 with it off. The 42 sharded blocks
   run their Linears on a window's rows, and the measured latents show they
   keep stock's bits at those row counts. INT8 ConvRot on uly2 equaled a
   one-GPU render of the pin-stripped copy after the stride fix above. ring2
   misses one GPU's bits with or without a pad, so it is measured, never
   granted: without a LoRA it read one-step NRMS 0.023 to 0.026 against
   stock on an even joint stream, which needs no pad, and 0.0025 to 0.031
   against one GPU on padded streams under the waiver, where the ring also
   attends the pad ([VALIDATION](VALIDATION.md#evidence-moss-image-sft)).
   On the CPU,
   `tests/test_moss_image_comfy_exactness.py` builds ComfyUI's own
   `MOSSImage` at toy size in float32 and shows every kernel call of the
   two-rank forward receiving stock's q, k and v for its heads in stock order
   with no pad row, and the output equal to stock's: an even stream, odd
   streams cut inside the image rows and inside the text rows, four ranks
   with a rank of pads only, batch 2, fused and separate q, k and v, both
   `rope` forms, ComfyUI's own Skip Layer Guidance skip and a marked
   row-local patch. That proves operands and order, not a GPU kernel; the
   kernel facts the route relies on are in the VALIDATION record.

   Performance cost, sequence parallelism only. Per model call: 42 times
   four joint all-to-alls and one final all-gather, the replicated prelude
   and text embedder on every rank, and the fused-norm route, which runs
   each block's two `rms_adaln` calls over all N rows on every rank. By
   block rows uly2 leaves each rank about half of one GPU's Linear and
   attention work. Measured on 2026-10-21 with the first Spark capped at
   2100 MHz, BF16 uly2 sampler walls ran 1.48x to 1.59x faster than one GPU
   from 1024 to 2048 square, and the 8-step Kromoss graph with its DMD2
   LoRA 1.51x ([BENCHMARKS](BENCHMARKS.md#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped)).

   Comfy model-option attention hooks are a separate contract. Stock patch
   nodes such as NAG consume `MODEL`, while DGX loaders publish `DGXM_MODEL`,
   so they cannot be wired into an ordinary DGX graph. If an out-of-tree
   carrier places those hooks on a resident worker model, the distributed
   adapters fail closed: Flux-family, Chroma, Hunyuan, and Wan-Animate 2 reject
   `attn1_patch` and `attn1_output_patch` hooks without a shard-local
   capability declaration before sequence
   sharding, and Krea2 rejects both under sequence parallelism whatever they
   declare; Wan-Animate 2 applies the same requirement to `attn2_patch` because
   its cross-attention patch observes spatially local rows. Qwen Image 2.1
   rejects Comfy block and attention hooks outright because they can rewrite its
   replicated prefix. CFG parallel rejects an `attn1_output_patch` that does
   not declare `condition_shard_local`, since such a hook may compare both
   condition branches.
   Exact support needs an explicit shard/condition-local capability contract;
   preserving a Python callable while changing the tensor domain is not
   support. Adapter authors declare the applicable
   `sequence_shard_local`/`condition_shard_local` strings in a collection on
   the callable's private `_dgxm_attention_patch_capabilities` attribute; every
   installed hook must opt in. See [docs/TROUBLESHOOTING.md #72](TROUBLESHOOTING.md#72-a-distributed-render-refuses-a-comfy-attention-patch-or-nag-style-hook).
3. **Mask rules**: USP kernels honor no attention bias. If the family carries
   one (padding masks, guide attenuation), either warn-and-drop when the pad
   rows are already content-zeroed and that degradation is part of the
   measured adapter contract, or raise `UnsupportedModelError` naming the fix.
   Chroma and Z-Image reject an effective mask before sharded work and accept
   a noop mask. Never change semantics without a signal.

Bindings go through `Adapter.bind` (idempotent instance-method replacement).
The worker calls `inject_usp` exactly once per fresh base load; clones made
for LoRA hot-swaps share the injected instance.

## Adapter-adjacent worker transforms

The worker applies two capacity and performance transforms in
`_inject_for_topology` after adapter injection. Adapters do not implement them:

* **Opt-in DiT block compile** (`actor/worker_compile.py:maybe_compile_dit`;
  the Init `compile_dit` widget). When enabled, the worker `torch.compile`s
  each block of `diffusion_model.blocks` in place with
  `max-autotune-no-cudagraphs`, which unlocks inductor's fp8 Triton GEMM;
  plain cudagraphs crash the dynamic-guidance layer on GB10. In-place
  `block.compile()` preserves state_dict keys for the LoRA hot-swap; it is
  idempotent and fails open to eager. It applies to any topology, including
  world-1/local, whenever the model exposes a `.blocks` list and the base is
  not slab-resident, and adds a one-time autotune warmup on the first render
  (docs/VALIDATION.md,
  [docs/TROUBLESHOOTING.md #14](TROUBLESHOOTING.md#14-first-render-pauses-9-s-with-compile_dit-on)).
* **FSDP capacity mode** (`adapters/fsdp.py:apply_fsdp_capacity_mode`), when
  the topology carries `fsdp`. It shards the standard block-list attributes
  (`blocks`, `transformer_blocks`, `double_blocks`, `single_blocks`, `layers`,
  `double_stream_layers`, `single_stream_layers`, `context_refiner`,
  `noise_refiner`, `ref_image_refiner`, `pixel_blocks`, `visual_transformer_blocks`,
  `text_transformer_blocks`, `patch_blocks`, and `content_mode_blocks.image`, a
  dotted path the wrap resolves one attribute at a time, because iterating a
  `ModuleDict` yields its keys) plus the top-level module with
  `fully_shard`, so each rank holds ~1/world of the weights and all-gathers
  per block. This is a capacity tool and is about 2x slower. Register every new
  family block-list attribute. No matching name raises `UnsupportedModelError`
  ("no shardable block lists found"). A partial match silently reduces capacity
  because an omitted list enters only the catch-all `fully_shard(dm)` wrapper
  instead of per-block sharding.

  It admits bf16 and fp16 cores, fp32 islands, and fp8 and int8 files
  (`adapters/fsdp_quant.py`). It also takes a LoRA stack, only under `lora_low_rss` (the shard-aware bake in
  `actor/fsdp_lora.py`). It leans on the
  worker's `NCCL_LAUNCH_ORDER_IMPLICIT` (NCCL >= 2.26; every worker setup sets
  it, since NCCL reads it once per process) to keep the world-group
  all-gather and the ulysses-group all-to-all from co-scheduling into a
  deadlock on world=2
  ([docs/TROUBLESHOOTING.md #13](TROUBLESHOOTING.md#13-uly2fsdp--ring2fsdp-render-deadlocks-hangs-no-error)).
  At the default `fsdp_prefetch_depth` of 1 it pins FSDP2's explicit
  forward-prefetch list empty, where torch already starts it: a defensive pin,
  not the guard. A worker_args depth above 1 names that many blocks ahead
  instead. Never set NCCL_PROTO=LL: it can independently
  deadlock FSDP.

  Parameters that must not shard go to every `fully_shard()` call as
  `ignored_params` and stay replicated plain tensors
  (`adapters/fsdp_islands.py`): fp32 islands, zero-dimensional parameters, and
  the modules a family runs outside its diffusion forward, which
  `OUTSIDE_FORWARD_MODULE_ATTRS` names per ComfyUI class (Anima's
  `llm_adapter`, see `adapters/anima.py`; MiniMax H3's `condition_proj` and
  `token_refiner`; LTXAV's caption projections and connectors). ComfyUI calls
  those modules from `extra_conds` before any pre-forward hook gathers a
  weight, so a sharded one
  fails the render on its first matmul; kept out of
  the wrap, they can never deadlock an all-gather. A new family that runs a
  module from `extra_conds` adds it to that table, which the `outside forward
  modules` canary seam holds to the installed ComfyUI.

  A second table, keyed by comfy class, holds dotted paths for a class whose
  module names other families also build. FLUX.3's `txt_in`, `time_in`,
  `emb_in`, `early_stream_modulations`, `txt_mode_blocks` and
  `content_mode_blocks.image_cond` live there, because Flux, Chroma and others
  build `txt_in` and `time_in` and run them inside their forwards; a flat entry
  would replicate theirs too. Only the FLUX.3 base model reaches these modules
  from `extra_conds`; the step-distilled model runs them inside its forward,
  where replication is still correct but not needed. The same seam reads the
  class-scoped rows with a reader that follows subscripted attributes and
  checks them against forward hooks on a tiny real model.

## Registration

Instantiate in `adapters/__init__.py:ADAPTERS`, most specific families first
(the first isinstance match wins). Subclasses of a supported base whose
forward contract differs (such as VACE, Camera, or S2V) must raise
`UnsupportedModelError` from `matches()`. Use exact-class allowlists so new
ComfyUI subclasses remain unsupported until reviewed. The LTX adapter
supports LTXV and LTXAV.

A new checkpoint generation is not automatically a new adapter. LTX 2.5 binds
the same two classes and sniffs as the same family, so it extends the LTX
adapter rather than registering beside it; a second adapter naming the same
exact classes would be shadowed by registry order and never dispatch. Its
three new surfaces each land on one side of the sharding boundary. The
spatio-temporal-guidance flag arrives in `transformer_options` and degrades a
flagged self-attention to its value projection, which is per token, so the
bound replacement honors it from local rows with no collective. The keyframe
position marker is applied in `_process_input`, ahead of the block loop the
adapter wraps, so it costs the workers nothing. The model geometry travels in
the checkpoint's `__metadata__`, not in the weight names, so every load path
that reaches a worker has to carry the header with the tensors.

## Acceptance gates (before HW promotion)

1. **Cross-rank latent identity**: every multi-rank render is gated in
   production. `run_render` (`nodes/common.py`) finishes in
   `nodes/render_result.py`, which calls
   `nodes/render_validation.py:verify_cross_rank_signatures`. It compares
   every rank's latent stats and raises on divergence, with a two-tier tolerance:
   identical, or <1e-3 relative for large models under comfy offload.
   Before comparison, every rank, including a singleton DP group, must
   provide the complete canonical shape/dtype/count/statistics/digest/projection
   signature, and its reported DP coordinate must match the pinned global-rank
   topology. Malformed signatures or incorrect topology grouping block output.
2. **Fidelity**: latents match the single-GPU reference within the campaign
   noise floor. Same-math comparisons (identical topology + kernels) use
   `gate_fidelity`; cross-topology comparisons use `gate_step_fidelity` at
   steps=1 (multi-step trajectories diverge across topologies;
   see `benchmark/gates.py:DEFAULT_STEP_NRMS` for the floor's provenance).
3. Every topology the auto table can emit is covered for its explicitly claimed
   input scope. A row either executes successfully for that scope or carries a
   documented typed capability refusal and remains `impl`; selecting an auto
   row does not promise that arbitrary prompt lengths or masks are runnable.
4. A LoRA-tweak re-render logs `hot-swap`, never `load`.
5. **Lazy-swap identity** (required for every family): a representative LoRA
   stack passes the identity gate. Run one render plus `dgxm gate`, or use the
   first-use auto-gate, and confirm that the ledger records `PASS`.

For campaign-level family promotion, gates 4 and 5 may be recorded explicitly
N/A, with a reason, when no compatible public LoRA exists for the exercised
model. That N/A records only the unavailable campaign leg: it claims no
hot-swap or lazy-swap proof and does not weaken the per-combination runtime
gate. Any requested model+LoRA combination must still earn
its own canonical identity-gate `PASS` before low-RSS or slab residency is
trusted.

## Family implementation order

Implement families in this order: the wan-family shape (proven end to end),
flux, a packed-sequence single stream (pixeldit- or lumina-like), then video
with per-token timesteps (ltx-like). Complete each family's acceptance gates
before starting the next.

Adapter hardware claims require the acceptance-gate evidence described above.
