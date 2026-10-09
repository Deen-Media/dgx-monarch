# FAQ

Common installation, workflow, hardware, and recovery questions.

Definitions of slab, gate, Fleet, and Worker service are in
[CONCEPTS.md](CONCEPTS.md). Test results are in
[MODELS.md](MODELS.md) and [VALIDATION.md](VALIDATION.md). If something breaks,
search the error text in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

An implementation or example alone does not mean a model or command has been
tested on the cluster.

## What this is

**What is dgx-monarch?**
A ComfyUI node pack plus a small cluster runtime. You still use ComfyUI in the
browser. Each host runs one Worker service in the background. When you attach
a mesh, each GPU gets an actor. Those actors either split one render
([xDiT](https://github.com/xdit-project/xDiT) + NCCL) or each finish their own
render (Fleet). [README.md](../README.md) has the longer version.

**Is this just two ComfyUI windows and something splitting the queue?**
No. Separate ComfyUI instances run independent jobs. dgx-monarch can split
one job across both hosts (cfg2, Ulysses, or ring), or run one full job per GPU
(Fleet).

**Did you write a faster sampler?**
No. The split uses stock xDiT / xfuser over stock NCCL. This project provides
the ComfyUI integration, [PyTorch Monarch](https://github.com/meta-pytorch/monarch)
actors, unified-memory handling, per-host errors, identity gate, setup and
doctor, and per-family adapters.

**Why Monarch and not Ray, torchrun, or a script?**
Monarch provides persistent actors, named GPU failures, and status queries
within the ComfyUI queue. torchrun launches raw xDiT from a shell.

**Is this a vendor cluster tool?**
No. A single-box ComfyUI guide tells you how to run one Spark well, and a
cable/network helper gets two boxes talking. Neither splits a sampler. That
is the part this pack does.

**Is this vLLM tensor parallel, but for pictures?**
No. ComfyUI does not use vLLM's `tensor_parallel_size=2` setting.

## Hardware

**Do I need two Sparks?**
Splitting a render requires at least two GPUs that the actors can group. One
Spark still runs the pack. Without `cluster.toml`, Init's `mode=auto` runs
locally ([QUICKSTART.md](QUICKSTART.md)). You get the memory tools, doctor, and
the sidebar. You do not get cfg2 or Ulysses across a cable.

**Do I need the QSFP / ConnectX-7 cable?**
For two hosts, it is the only measured link in this project. The shipped pair profile
is ConnectX-7 200G RoCE across both rails; [CLUSTER.md](CLUSTER.md) has the
profile and its settings. USB4 and Thunderbolt are untested. With no RDMA rail
up, the shipped profile still names the ConnectX-7 port, and doctor warns that
NCCL will likely fail. NCCL would use TCP sockets over such a link once a
`[fabric.<name>]` table names that link's interface, and no record here
measures that path.

**What about three or four Sparks?**
The design target is a 2 to 8 node Linux NCCL fabric
([DESIGN.md](DESIGN.md)); recorded timings cover two Sparks. Three- and
four-host deployments remain unmeasured.

**Will this make my 4090 or 5090 faster?**
Recorded timings cover two DGX Sparks (GB10, ARM, unified memory). Multiple
GPUs in a PC should work in principle, but no run is recorded here. A single
GPU cannot split a render.

**Another GB10 box, or a DGX Station?**
Other GB10 hosts share the chip but have no recorded run here. DGX Station
uses GB300 and is also untested; the measurements cover Sparks.

**Docker? Windows? Mac?**
Install is a Git checkout in ComfyUI `custom_nodes`, and the requirement is
Linux with Python 3.11 or newer. See [INSTALL.md](INSTALL.md). Other
installation environments are untested.

**Is a second Spark worth it for ComfyUI?**
If you already have two and you do big images or video, the cable is how you
split a render. If you have one box and you mostly do small images, a second
Spark may not make those images faster. Fleet does nothing useful on one GPU.

## Using ComfyUI

**Do I have to rebuild my workflows?**
No. Drop in **DGX Monarch Init**. Only a few nodes have a cluster version: the
diffusion-model loader, the LoRA loader, the samplers and guiders, Model
Sampling SD3, and Basic Scheduler. CLIP, text encode, empty latent, VAE
decode, and save stay in normal ComfyUI ([QUICKSTART.md](QUICKSTART.md)). When
you hit Queue it tells you which stock nodes to swap. It does not edit the
graph for you. The list is [TROUBLESHOOTING.md #63](TROUBLESHOOTING.md#63-a-toast-at-queue-time-says-the-workflow-is-missing-dgx-monarch-nodes).

**Can I keep my other custom node packs?** Yes, as long as they are not also
wrapping the DiT forward. If two packs patch the same attention, they can
conflict, and a distributed render refuses a Comfy attention patch it cannot
honor ([TROUBLESHOOTING.md
#72](TROUBLESHOOTING.md#72-a-distributed-render-refuses-a-comfy-attention-patch-or-nag-style-hook)).
Compatibility with an arbitrary collection of custom nodes is unverified.

**Where are the templates?**
`Workflow -> Browse Templates -> dgx-monarch`. What ships, and which graphs
work with any family, is in [QUICKSTART.md](QUICKSTART.md). A missing template does
not necessarily mean the model is unsupported. See [MODELS.md](MODELS.md).

**I closed ComfyUI. Is the cluster down?**
Not necessarily. Worker services can keep running after `dgxm up`, guided
setup, auto-heal, or an installed user service. Check `dgxm status`. Use
`dgxm down` only when you mean to stop them.

**SwarmUI or some other frontend?**
The driver exposes the ordinary ComfyUI API. Other frontends may work through
it, but are untested here.

## Speed and topology

**Why isn't every image twice as fast?**
A sampler runs on one GPU unless you split it. cfg2 only has two CFG passes to
place. Ulysses and ring exchange attention data every step, so they pay off on
long sequences (big image, video). How much time a split saves, and on which
jobs, is in [VALIDATION.md](VALIDATION.md).

**cfg2 vs Ulysses vs ring vs Fleet vs FSDP?**

- **cfg2.** Prompt on one GPU, negative on the other. Smaller images.
- **Ulysses / ring (USP).** Split the tokens. Big images and video.
- **Fleet.** Each GPU does a whole image. More pictures per hour, not a faster
  single picture.
- **FSDP.** Split the weights so each GPU holds part. That is a capacity tool,
  not a speed tool, and [CONCEPTS.md](CONCEPTS.md) says how much slower. Use
  it when the model does not fit, or you want room for a local LLM.
- **auto.** A rule table (family, quant, megapixels, how many GPUs). It logs
  the row it matched and why. It does not guarantee the fastest option.

Leave Init on `auto` the first time. If you just want two pictures at once,
open the Fleet template.

**SageAttention?**
See the topology tables in [MODELS.md](MODELS.md) for automatic
SageAttention selection.

**TeaCache, EasyCache, FBCache?**
These accelerators change the model's calculations, which this project does
not support
([MODELS.md](MODELS.md)). Use stock ComfyUI for workflows that need them.

**PipeFusion?**
Unavailable on this fabric. The topology slot is reserved; there is no
PipeFusion mode ([DESIGN.md](DESIGN.md)).

**Native RDMA for sending latents back?**
It is off by default and marked **NOT RUN / HOLD**. Latents come back over
actor messages; the split itself still uses NCCL. The terms of the hold are in
[DESIGN.md](DESIGN.md) 5.6, and the failure an operator can see is
[TROUBLESHOOTING.md #75](TROUBLESHOOTING.md#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails).

**I already split my queue with an agent.**
Fleet serves that use case. Ulysses or cfg2 splits one job across both hosts.

**NCCL between my Sparks is only a few GB/s.**
Fix the link first: RoCE, both rails, current ConnectX-7 firmware. Ulysses on
a TCP fallback looks like a broken sampler. [CLUSTER.md](CLUSTER.md) covers
the fabric profile and what the settings mean.

## Accuracy

**How closely does output match stock ComfyUI?**
Each HW row in [MODELS.md](MODELS.md) lists the tested configuration and
comparison. [VALIDATION.md](VALIDATION.md) records exact matches or the measured
difference, including NRMS where applicable.

**What is the identity gate?**
The first time you run a new model and LoRA combination, it compares reference renders
and stores the result. A `PASS` unlocks the optimized residency, and later
renders reuse that verdict. Anything else keeps that residency off and renders
through the stock path where a fallback is allowed
([TROUBLESHOOTING.md #66](TROUBLESHOOTING.md#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)). Terms are in
[CONCEPTS.md](CONCEPTS.md). If the first render is slow, or the check will not
finish: [TROUBLESHOOTING.md #61](TROUBLESHOOTING.md#61-the-first-render-took-minutes-and-the-next-one-took-seconds).

**Does Fleet run that check?**
No. Fleet uses stock residency unless the exact model combination and
configuration already has a `PASS`. Render it once through a normal DGX Monarch
KSampler to run the check. Fleet can then reuse that `PASS` for slab and low-RSS.

**What is the red waiver?** A pending capacity rescue card is amber. An
accuracy waiver is red, is headed "accuracy waiver", and says that it permits
known-wrong math ([TROUBLESHOOTING.md
#55](TROUBLESHOOTING.md#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means)).
Once you accept a capacity rescue it turns green and reads "Allowed". Accept a
red waiver only if you intend to permit the stated accuracy loss.

**Will I get the same image if I queue twice?**
Templates ship with the seed control on randomize, so a second Queue is not a
silent repeat. A gate `PASS` confirms that the tested load and swap paths matched. It does
not promise identical output for every seed and configuration.

**Will my pictures match stock ComfyUI?**
On the HW rows, yes, within the limit written down for that row. A `FAIL`
turns the optimization off and can fall back to stock
([CONCEPTS.md](CONCEPTS.md)).

## Models

**What works?**
[MODELS.md](MODELS.md). **HW** means the listed configuration passed its stated
comparison on two Sparks. **impl** means the implementation exists but that
comparison is incomplete or failed. Neither a screenshot nor a related
checkpoint extends a row's evidence.

**GGUF?**
No. The cluster path does not load GGUF checkpoints. [MODELS.md](MODELS.md)
lists this and the other unsupported formats.

**ControlNet, IP-Adapter, random Civitai LoRAs?**
The split covers the DiT. Driver-side preprocessors and extra networks are
not validated here, and there is no cluster ControlNet loader.
LoRA on FSDP needs `lora_low_rss` on (Init node, or `worker_args` in
`cluster.toml`); with it off the run refuses and names the switch, because the
sharded bake reads pristine weights from the checkpoint and never touches
comfy's backup copy. A comfy-kitchen quantized checkpoint (scaled fp8,
fp8mixed, int8) refuses LoRA on FSDP whatever that switch says. XLabs-format
Flux LoRAs fail silently instead: their keys do not match, so nothing
errors and the render ignores them; use a converted release. Row by row:
[MODELS.md](MODELS.md).

**I have a finetune or a merge.** Init has a `family_adapter` widget for the
case where the file header does not say what the model is. Naming the child
family for a finetune of that child is what works; naming a parent family for
a child model is refused, because it would compute the wrong math. A named
family declares the file's family; it neither inherits the support row's
evidence nor enables automatic slab residency. [MODELS.md](MODELS.md) has the
rules.

## Memory and running an LLM too

**Can I run vLLM on the same two Sparks?** A big tensor-parallel LLM wants
both boxes. What works in practice: Comfy on one, vLLM on the other; FSDP or
slab so a smaller LLM can share; or **Reset attached mesh** to give memory
back to the OS. Two heavy jobs on one pair compete for the same memory pool.

**I thought Spark ComfyUI only saw 64 GB. What are slab and low_rss?**
On a Spark the CPU and GPU share one memory pool, so how a loader spends it
decides what fits. Slab residency and low-RSS LoRA handling are what this pack
does about that, and both are explained in [CONCEPTS.md](CONCEPTS.md). The
measured loader numbers are in [VALIDATION.md](VALIDATION.md).

**What does Reset attached mesh do?**
It retires the actor processes and the models they were holding, and the next
render spawns fresh ones. Worker services stay up. Use it when you want that
memory back for an LLM. It refuses while a render or a sample-result lease is
still active, and the panel asks you to retry after the work finishes.

## Install and first hour

**How do I install?**
Install ComfyUI, put this repo in `custom_nodes`, get `dgxm doctor` green.
[INSTALL.md](INSTALL.md) has the steps and
[QUICKSTART.md](QUICKSTART.md) is the first graph. For two boxes, run
`dgxm setup` as a dry run, read the plan, then run it again to apply. It does
not scan the LAN: you type every host and fabric address.

**What is an operator receipt?**
A JSON file written by setup, `dgxm doctor --repair`, or `dgxm update
--verify`. It records what the command did; it is not signed machine identity
or remote attestation ([INSTALL.md](INSTALL.md#operator-receipts),
[THREAT_MODEL.md](THREAT_MODEL.md)).

**Which operator commands have been tested on the pair?** Setup apply with
its attached smoke, readiness, repair and standalone cluster smoke have
recorded runs. The readiness checks also correctly reported a stopped Worker. See [operator checks](VALIDATION.md#operator-checks).

Verified update supports a Worker on the driver's Spark. See the
[recorded update results and limits](VALIDATION.md#operator-checks) and
[update requirements](INSTALL.md#updating).
These operator checks do not establish model accuracy; use [MODELS.md](MODELS.md)
for those results.

**doctor looks fine and nothing works.**
Doctor warns rather than saying OK when a probe never ran, and a Worker service
that is down fails its check. Start at
[TROUBLESHOOTING.md #65](TROUBLESHOOTING.md#65-dgxm-status-and-dgxm-doctor-are-green-and-every-attach-still-fails). Paste the sanitized doctor rows
rather than a screenshot of the UI.

**I put ComfyUI on the LAN. Is the sidebar safe?** Reset attached mesh
requires a confirmation, an action header, and matching Host/Origin browser
metadata. Those are CSRF checks, not a login: a network-exposed driver still
depends on ComfyUI's own authentication. The free-memory button calls
ComfyUI's own `/free` route and inherits whatever protects that. The fabric
between the boxes needs separate isolation, described in
[SECURITY.md](../SECURITY.md). Do not expose
it through port forwarding without those protections.
[THREAT_MODEL.md](THREAT_MODEL.md) has the rest.

**License?**
Apache-2.0 for this repo. ComfyUI is the GPL-3.0 host, xDiT and xfuser are Apache-2.0,
Monarch is BSD-3-Clause, each installed separately. This tree vendors no
third-party code. [LICENSE-NOTES.md](../LICENSE-NOTES.md).

## When it breaks

**One Spark died mid-render.**
Once the box is back, close ComfyUI, run `dgxm restart` so both Worker
services restart together, and start ComfyUI again ([TROUBLESHOOTING.md #2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)).
A ComfyUI process cannot attach again once its fleet changed under it: a box
that died, a Worker service that restarted, or old workers the driver had to
confirm had stopped. A dead actor process retires the
whole attached mesh, and its error names the endpoint and host
([TROUBLESHOOTING.md #4](TROUBLESHOOTING.md#4-supervisionerror--process-exited-with-non-zero-code)). A worker exception is different: it comes back to the
caller and leaves its actor alive.

**I hit Stop in ComfyUI.**
It stops at a denoise-step boundary, not mid-kernel, and a pipeline abort
cancels the whole queued window first. If a GPU kernel still has not come back
after the abort deadline, close ComfyUI, then run `dgxm restart`: a Worker
service restarted under a live session cannot be attached from it again
([TROUBLESHOOTING.md #2](TROUBLESHOOTING.md#2-every-attach-times-out-after-one-failed-attach)).

**It said it refused. Did it crash?**
No. Things like packed CFG, LoRA on FSDP with `lora_low_rss` off, a pad that
ring cannot take, or a guide that will not fit are meant to stop before a
kernel runs. Follow the error's remedy, then Queue again; these refusals normally need no
reset.

**Where do I look?**
`dgxm doctor`, the DGX Monarch sidebar, `dgxm top`, then
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) by what you saw. A bug report needs:
topology, family and quant (no filenames), versions, sanitized doctor output, a
short log slice, and which troubleshooting entries you already ruled out.

---
See [VALIDATION.md](VALIDATION.md) for measurements and test limits.
