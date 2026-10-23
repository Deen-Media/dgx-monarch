# Troubleshooting

See [CONCEPTS.md](CONCEPTS.md) for project terminology. Search this page for
the error text or symptom; report failures not covered here.

> **A cold combination on a normal KSampler path can wait roughly 2-4 minutes
> on the tested hardware before showing first pixels.** Model size, storage,
> and hardware change that timing. This is expected with the default
> `auto_gate=first_use`: DGX Monarch KSampler and KSampler Advanced prove a new
> model/residency/LoRA combination before dispatching the user's first render,
> rather than rendering once through an unverified optimized path. Do not restart
> a healthy job during this proof. **Every dgx-monarch release version
> invalidates earlier PASS results.** A same-version package-source change seen
> by a fresh process also invalidates earlier results for that configuration.
> Each combination is checked again on first use after an upgrade or source change.
> A change to the artifact bytes, the ComfyUI commit, the gate protocol, the
> canonical package-source manifest, or the effective worker, fleet or topology
> context also needs a fresh proof. Each process reads the source manifest once,
> on first use, and does not watch it: restart the driver and the workers after
> changing an editable checkout.
> The check uses two short test renders and compares slab against stock when
> slab is active. A new model may also need a render to identify its family
> before reloading into slab, for up to four renders total. Fleet skips
> the inline identity check: an unverified Fleet combination runs with
> slab and low-RSS residency forced off. Single-model custom samplers
> require the explicit Identity Gate node. Dual-model custom requests are
> always dispatched with both residency optimizations off; the single-model
> Gate cannot authorize their second model slot.

## 1. Attach fails with `MESH_ATTACH_CONFIG_TIMEOUT`

dgx-monarch sets torchmonarch's attach timeout to 60s before the first
monarch import (`dgx_monarch/__init__.py`; hyperactor snapshots the
environment at that import, so a later set never applies). Export
`HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT` before ComfyUI starts to override it.
The driver waits 70s for the attach to initialize, strictly above the
config-push budget, so monarch's typed per-host error wins over a bare
TimeoutError. The wait follows the budget in force, not the shipped one: an
export of 180s waits 190s. Concurrent Init calls for the same mesh key share
one creator, and a wedged creator raises `MeshAttachError` after 180 seconds
rather than parking every waiter; `DGXM_MESH_CREATION_TIMEOUT` may override
that wait with a finite positive value up to 3600 seconds. Creation is
single-flight across the process: a different mesh key is refused with
`MeshAttachError` while an attempt is active; retry it after that settles.
Changing the limit does not make an unconfirmed old fleet safe to replace.

If attaches still time out, check the causes below. After any upgrade, rule
out a mixed-pin mesh first: driver and Worker services on different
torchmonarch builds attach and spawn cleanly, then every endpoint call times
out with no version message in either direction. Run `dgxm doctor` and check
both hosts' pin rows before treating it as a wedge.

**Symptom:** a Monarch client started without an explicit bind fails at once
with `hostname resolved to no non-loopback, non-link-local address; configure an
explicit transport address`, or Init node (or probe) times out attaching while
Worker service logs show a connection that never completes.

**Cause:** without an explicit bind, Monarch's default transport advertises
whatever the driver's own hostname resolves to. A name that resolves only to
loopback (`/etc/hosts` maps it to `127.0.0.1` on many distros) is refused at
once with the message above. A name that resolves to a LAN or Tailscale address
passes that check, so the Workers answer over that network, or never reach the
driver when that address is not routable from them.

**Fix:** set `cluster.client_bind = "tcp://<driver fabric IP>:0"` in
cluster.toml; it pins the client to the fabric address whatever the hostname
resolves to. dgx-monarch refuses a cluster attach without it, so mapping each
box's hostname to its fabric IP in `/etc/hosts` is not enough on its own. If an
attach timed out, see [#2](#2-every-attach-times-out-after-one-failed-attach): the failed attach has left the Worker services wedged.

`dgxm doctor`'s `hostname resolution` row reads that setting. When the
hostname resolves to a routable address, the row reports `[ ok ]` with that
address. When it resolves to loopback or not at all (normal on minimal server
images and in containers), a usable `client_bind` (a unicast IP that is not
loopback) still gives `[ ok ]` and the row names the IP it pins, because the
client transport advertises the bind and never resolves the hostname. Without
one the row warns.

### Self-heal (default on)

Before attaching, the driver checks the exact Worker service process and the
host's passive LISTEN table, then restarts dead services, so a cold fleet can
start from a queue press. It never opens a protocol-blind connection to a
Monarch worker endpoint (docs/VALIDATION.md). If an attach fails anyway, the
driver retries once in-process (the current pin evicts the stale attach
session, meta-pytorch/monarch#4067). When the retry also fails, it restarts the
Worker services so the next attempt starts clean, and that ComfyUI session
refuses further attaches: restart ComfyUI (the fleet is already healthy). The
retry dials each loop's id as read again after the first failure. It does not
run when that reading finds no id, or when the attach is a released fleet's
replacement and no loop moved: see
[#2](#2-every-attach-times-out-after-one-failed-attach). A refusal that starts `no attach ran`
dialed nothing and leaves the session usable ([#2](#2-every-attach-times-out-after-one-failed-attach)).
Set `auto_heal = false` in cluster.toml to opt out.

## 2. Every attach times out after one failed attach

**Symptom:** Worker services look alive (`dgxm status`: process running, listener open)
but attaches keep timing out.

**Cause:** the Worker service coordinator died while its TCP listener survived.
Two known triggers: a reply-timeout attach (see [#1](#1-attach-fails-with-mesh_attach_config_timeout)), and anything calling
`hosts.shutdown()` on an attached Worker service. dgx-monarch's own detach
never calls it; it stops its procs and leaves the service to `dgxm`. Normal
client exits and crashes did not wedge Worker services in the two-Spark
checks: repeated attach, NCCL bring-up and detach cycles ran without service
restarts ([VALIDATION.md](VALIDATION.md)).

**Fix:** auto-heal (`auto_heal` in cluster.toml, [#1](#1-attach-fails-with-mesh_attach_config_timeout)) heals dead Worker service
listeners before the first attach. A failed attach retries once in-process; a
second failure restarts the Worker services so the next attempt starts clean.
`dgxm restart` remains the manual fallback. One residual case needs more: [#15](#15-attaches-still-time-out-after-dgxm-restart-driver-side-wedge).

**A Worker service restarted under a live driver.** Each Worker loop start
serves a new Monarch proc id: it listens on `<proc-id>@tcp://<ip>:<port>`,
with an id derived from the boot id, its pid and its start time. The address
in cluster.toml, the systemd unit and the firewall rules stay the bare URL.
The loop writes the id into its actor ledger record before it listens, and
the listener-generation probe, which the driver runs on every host before
every attach, reads it back. The driver dials only that id. So after `dgxm
restart`, a crash that systemd restarts, or a hand restart of the service, a
ComfyUI that stayed up can attach again without a restart of its own. A loop
that served the bare URL kept Monarch's singleton proc identity across
restarts, and the driver's next attach stalled until its config push timed
out.

**A render after the restart.** The first render after the restart meets the
old actors. A message to one of them comes back to the driver's fault hook as
undeliverable, and never to the call that sent it, so before 1.2.0 the
render's cross-rank price request
([#93](#93-a-multi-rank-render-refuses-before-any-rank-loads-and-names-a-rank))
waited its whole 30 s deadline and refused with `did not answer in time`,
naming Reset attached mesh. Now the price wait ends within half a second of
the first delivery failure that names a fleet host and the mesh the price
waits on, and the refusal says the message was undeliverable. A late failure
from a mesh that Recycle, Reset or an earlier re-attach retired names the same
address under another mesh id, so it never ends a later wait.

When nothing has reached a restarted loop since its restart, the render's own
price request is the message the old channel session loses (below), and no
failure comes back for it. So the wait asks a rank that has not answered once
more after 2 s; that second request comes back undeliverable at once, and the
wait ends about 2.5 s in.

**A sampler render attaches again by itself.** A render through KSampler,
KSampler Advanced or Sampler Custom (DGX Monarch) that owns its render
session, and whose topology, attention and worker settings are unchanged
since the last render, carries on after that refusal. So does the eager load
of a Load Diffusion Model (DGX Monarch) node with an explicit topology, on the
same condition:

1. It reads every loop's id, the reading every attach takes, and compares it
   with the id this driver dialed.
2. When every loop serves a new id, it stops the stale mesh through the
   deliberate teardown, giving each of the teardown's two waits 2 s. In CPU
   runs a stop sent to the old ids was never acknowledged, so each wait ran
   out its 2 s; the old processes died with their loops. On the pair,
   Monarch later reported the old actors stopped; the recheck of this step
   is pending. As after any stop that times out, the liveness proof in
   [#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)
   releases the stale mesh only when every host reports its old processes
   gone.
3. It attaches by the new ids, through the ordinary attach and its budgets,
   and dispatches once more on the new mesh.

No click is needed. The driver log prints one line for it, `restart_reattach
outcome=attached: every Worker loop serves a new proc id (...); this render
retired the stale mesh (liveness_release) and attached again by the new ids in
<n> s, with no reset`. The replacement attach's trace opens with
`released_predecessor=true` after a liveness release and adds one
`event=restart_reattach` line
([#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out)). The
render's finish record in `/dgxm/telemetry` carries `reattach=restart_reattach`
and `reattach_s`, the seconds the detour took, and the `dgxm top` ticker shows
a `restart_reattach` event with its `outcome`, `verdict`, `moved`, `teardown`
and `wall_s`. The undeliverable messages and late faults from the stale mesh
(`cluster fault`), and a stale-mesh stop that times out (`proc stop on detach
timed out`), still log at ERROR and can raise browser toasts; next to
`outcome=attached` they are expected.

The render refuses at once instead, with a class C `cross_rank_capacity`
refusal that names the remedy, when the reading finds:

| what the reading found | remedy |
|---|---|
| some loops serve a new id and others still serve the id this mesh dialed | `dgxm restart`, so every loop restarts together, then queue again |
| no loop serves a new id: the actor is gone or unreachable for another reason | Reset attached mesh, then queue again |
| a loop whose id it cannot read | the id gate's remedies below, then queue again |
| a host where the liveness proof does not find the old processes gone | the after-death cause in [#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn); `dgxm restart` sweeps survivors ([#51](#51-a-worker-actor-outlives-its-client-and-holds-tens-of-gib)) |

**Paths that do not attach again by themselves.**

- KSampler Pipeline (DGX Monarch) renders and the Identity Gate (DGX Monarch)
  node run in a session pinned to the mesh they started on. They stop at the
  fast refusal, and queuing them again, or running `dgxm restart` again, does
  not clear the stale mesh. Run one render through KSampler (DGX Monarch)
  with the loader and Init inputs unchanged, which attaches again by the new
  ids, then queue the workflow again.
- An automatic first-use identity gate stops at the fast refusal too. Its
  abort turns that combination's residency levers off, as for any abort, but
  sends no stock policy to the gone actors, and the combination's next Gate
  run can restore the levers. The render it was proving then runs under stock
  residency, and attaches again by itself only when that leaves its worker
  settings as the last render's were (the next item). Under FSDP the gate's
  refusal ends the render instead.
- Fleet KSampler (DGX Monarch) dispatches without a cross-rank price, a
  one-host fleet prices nothing, and a render whose topology, attention or
  worker settings differ from the last render's sends that setup to the old
  actors before it prices. A gate's own policy changes can do the same.
  These meet the old actors directly.
- A render that reaches the stale mesh after Monarch's supervision already
  reported the old actors dead takes the supervision path. On the pair the
  panel's status poll drew that report about a minute after the restart. A
  render that arrives while the supervision stop is still running refuses
  with `the cached mesh cannot be used or replaced`; queue again once it ends.

On those paths the fallback is the one from before 1.2.0. The call fails on
the old actors (a `SupervisionError`,
[#4](#4-supervisionerror--process-exited-with-non-zero-code), or a call
timeout), and the driver retires the old fleet; when the stop of the old
fleet does not confirm, the liveness proof in
[#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)
releases it. The next render or Recycle then attaches a replacement and reads
the new id first; after a release the trace's `open` line carries
`released_predecessor=true`. That replacement attach is the path the first
CPU test below covers.

The Worker log prints the id in Monarch's own spelling and then in hex:
`worker service listening on service<...>@tcp://<ip>:<port> (proc_id=<16 hex
digits>)`. The probe, the attach trace and doctor print the same value as the
16 hex digits.

The first message this driver sends to a restarted loop's address is lost:
it rides the channel session the old process accepted, the new process
refuses that session, and only the next message opens a fresh one. A lost
config push would leave the new host agent waiting for it forever. So before
the first dial to a loop whose id changed since this driver last dialed it,
the driver sends that address one throwaway message and waits 2 s; the trace
writes a `proc_id_prime` line naming those loops. That first attach after a
restart therefore takes about 2 s.

That holds when the loop came back within Monarch's message delivery timeout
(`HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT`, 30 s by default in the pinned release).
After a longer outage, such as a stop, a crash loop or a reboot, the driver's
channel to that address stays closed and its attach fails at the config-push
budget, the attach within a render included: restart ComfyUI. dgx-monarch
leaves the delivery timeout at its default.

**The id gate.** When the first reading of an attach carries no id for a
loop, the attach refuses before anything dials, and the ComfyUI session stays
usable:

```
no attach ran: the id each Worker loop serves on this start is unknown for <host> (tcp://<ip>:26600): <reason>; <remedy>. This ComfyUI session stays usable (docs/TROUBLESHOOTING.md #2).
```

| reason | remedy |
|---|---|
| nothing holds its listener | start the Worker services (`dgxm up`), then render again |
| its loop runs an older release that serves the bare address | run the Worker services on this driver's release (`dgxm update`, or `dgxm restart` on a plain install; for Workers in release slots see [#118](#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)), then render again |
| no loop record publishes the id of the process holding its listener (`unpublished` or `malformed`) | render again, which reads the id again; if it repeats, `dgxm restart` |
| its actor ledger file would not read (`unreadable`) | check that `~/.local/state/dgx-monarch/actor-procs.json` on that host is a readable JSON file, then render again; if it repeats, `dgxm restart` |
| the probe did not read it (`unknown`, or the probe's own error) | fix the host or ssh fault, then render again |

`dgxm doctor` fails the `<host> listener generation` row for the second and
third reasons and warns for the last two
([#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out)).

A loop that cannot write its record says so in its log:

```
worker service could not publish its id in the actor ledger; driver attaches refuse until it can (docs/TROUBLESHOOTING.md #2)
```

Check that `~/.local/state/dgx-monarch/` on that host is writable by the
user the Worker service runs as. The loop's tracker writes the record again
every 5 s, so the next render after the fix attaches.

A loop that restarts between the read and the dial costs one attempt: the
host bounces the old id at once, but the attempt still waits out the config
push. The driver then reads the ids again, and a loop that serves a new id
gets one more attempt with it. When that reading after a failed first attempt
finds no id, no second attempt runs. The first dial already ran, so the
session is poisoned: the error names the first failure and the loops the
reading found no id for (`no second attempt ran, because the id reading after
it found none for <host> (...)`; a released fleet's error says `the id reading
after it found none for`), and the trace closes `attach_1_reread_refused` (a
released fleet closes `attach_1_released`;
[#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out)).

**Upgrading.** A ComfyUI started before the update that brought per-start ids
dials the bare address. `dgxm update` refuses in either mode while ComfyUI
runs, so this case arises only when the Workers restart on the new code under
a running ComfyUI: `dgxm restart` after a pull, or a hand restart of the
service. Its next attach then fails at the config-push budget, retries once,
and names `dgxm doctor` and a ComfyUI restart (older builds end that message
`the in-process retry already ran`): restart ComfyUI. A new driver against
Worker services still on an older release refuses before dialing, as the
table above says; Workers left in release slots by a plain update from 1.1.0
need [#118](#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update).

One case fails fast instead of retrying. When the driver has just retired a
dead fleet's replacement latch on positive proof that its worker processes are
gone, its attach to the replacement makes one attempt, plus one more only when
a loop came back with a new id after that attempt dialed. When it fails, the
error says no in-process retry ran and names the restart fallback above
([#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out)). Every
other attach keeps two attempts, and its second failure poisons the session.

**What the CPU test shows (2026-10-15).**
`tests/test_worker_proc_id_integration.py` starts two real Worker loops on one
machine and attaches from a driver process. After the driver's traffic has
been acknowledged, it restarts one loop, and the same process attaches again
through the replacement path at once, with nothing else sent to that loop
first: about 2.1 s, of which 2 s is the wait after the throwaway message. In
mutation runs of that test, serving the bare URL, reusing an id, or skipping
the throwaway message each made the same re-attach time out at its
config-push budget. In review runs with the loop held down 15 s and 25 s, the
re-attach still took about 2.1 s; held down 40 s, past the delivery timeout,
it failed at its budget with the restart fallback.

A third case in the same file restarts both loops under a live mesh, with
nothing sent to the loops first, then sends a render's capacity price and
calls the same re-attach that `submit_render` calls; it does not run
`submit_render` itself. The liveness scan, policy admission and client lease
are stood in, and the liveness stand-in answers gone at once. The price wait
ended 2.5 s in, on the second request's delivery failure, which named the
mesh the price waited on. The detour took 6.6 s (the id reading, the two 2 s
teardown waits and a 2.1 s attach, with no real liveness probe), and the
price answered on the new mesh 9.1 s after it was first sent, where the same
price had waited 30 s. On hardware the liveness release reads every host
under its 20 s budget; that cost is not measured yet. In mutation runs,
removing the wait's short circuit made the price wait 30.1 s, and removing
the re-attach left it refused.

**What the two-Spark recheck showed (2026-10-15).** With per-start ids and
both loops restarted together under a live driver, the first render waited
the 30 s price deadline and refused, the failure this release removes. The
render after it attached by the new ids in 2.8 s and rendered with no ComfyUI
restart. After a 15 s outage the next render also rendered. After a 40 s
outage the attach timed out at its config-push budget and poisoned the
session, and a relaunched driver rendered. The pair recheck of the attach
within the render is pending; it adds a render 45 to 90 s after `dgxm
restart` with the panel open, which may meet the supervision path instead.

**What the hardware tests showed before per-start ids (2026-10-01).** Changing
the fleet under a live driver left that driver unable to attach again. The
tests covered four failure cases and a successful recovery:

| What was tried | Result |
|---|---|
| A worker actor died while its loop was stopped, then the driver tried to attach (reproduced 4 times) | Every attach from that driver failed |
| One worker loop alone restarted under a live session, then the driver tried to attach | Failed, and stayed unattachable for every later driver until both loops restarted together |
| Both worker loops restarted together under a live session, the full 190 s pace waited | Still failed from that driver |
| `HostMesh.stop()` on the retired fleet | Timed out |
| Both loops restarted together (`restart_after_failed_attach`, auto-heal on), then a fresh driver process (ComfyUI restart) | Attached in 0.75 s |

The rule stands: **restart worker loops together (`dgxm restart`), not one
loop alone while a ComfyUI session is attached.** A render after one loop
restarted alone refuses and names this remedy. Any restart with no driver
alive is fine. When a re-attach still fails, recovery is both loops
restarting together, then a fresh driver process.

## 3. Next render hangs at NCCL rendezvous after a crash

**Symptom:** after a driver crash or an interrupted render, the next run's NCCL
init hangs on the master port.

**Cause:** stranded model-holding procs from the crashed client keep the NCCL
master port. dgx-monarch shuts procs down on client death and in try/finally
paths, but a SIGKILLed driver cannot.

A headless validation run exposed the same cleanup failure: its results had
been collected, but Monarch's transport panicked during interpreter shutdown,
and an actor stayed alive. Treat that as teardown residue, not as evidence for
or against the completed render's distributed math. Headless runners must
recycle their client-owned mesh before exit and confirm that no actor remains;
they must never shut down the persistent Worker services.

**Fix:** `dgxm restart`. The `nccl master port` row of `dgxm doctor` warns when
the port accepts connections outside a render.

## 4. `SupervisionError: … process exited with non-zero code`

A worker actor died. The error identifies its endpoint and host. Check the
Worker service log on that host: `~/.local/state/dgx-monarch/worker-loop.log`
for nohup-started Worker services (`dgxm up`), `journalctl --user -u
dgxm-worker` for systemd-managed Worker services. On UMA boxes the usual cause
is OOM; see [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc). The driver
evicts the poisoned mesh automatically, and the next Init or render respawns
the actors. That needs no ComfyUI restart. A Worker service that died and came
back within the delivery timeout serves a new id. When every service
restarted, the first sampler render attaches again by itself (CPU-tested; the
pair recheck is pending). When only one did, that render refuses and names
`dgxm restart` ([#2](#2-every-attach-times-out-after-one-failed-attach)). When
the attach to the replacement fleet fails anyway, after a longer outage, run
`dgxm restart` so both Worker services restart together, unless the error
says they already did, then restart ComfyUI.

## 5. Loads fail / OOM on UMA boxes (DGX Spark etc.)

**Symptom:** ComfyUI refuses a load that appears to fit, or the host swaps
heavily during loading. On UMA, the page cache and GPU allocations share memory.

**What to do:** before a large load, use
`sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` and set `reserve_vram_gb`
on Init. `dgxm doctor` warns when page cache crowds available memory. For a
capacity refusal, follow the alternatives in the message: slab residency,
an `*+fsdp` preset, a pruned or quantized checkpoint, or more free host memory.

**`stock residency cannot load...`** is a class C refusal on integrated/UMA
devices. The estimate is 2.1 times the checkpoint size for weights plus their
host placement copy, with a 5 GiB floor to avoid ComfyUI partial loading.
A 60.02 GiB BF16 checkpoint therefore requires about 131.0 GiB. The driver
uses the same placement charge. The previous 1.3x estimate admitted a load
that killed a host; measurements and calibration are in
[VALIDATION.md](VALIDATION.md).

**If `capacity rescue available` appears:** open the DGX Monarch sidebar,
click "Load with slab residency", then queue again. Nothing large has loaded
at the refusal. [#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)
covers headless consent, automatic rescue, and revocation.

A slab-only load has no stock reference on this hardware. Its identity result
stays `INCONCLUSIVE` with `cross_mode=CAPACITY`, even when consent permits the
render. Each load verifies checkpoint bytes
([#54](#54-slab-byte-verify-certificate-mismatch)), and the ledger records the
bypass and capability context. Neither consent nor a byte certificate grants
PASS. You do not need `auto_gate=off` for capacity rescue
([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)).

**If no card appears:** the refusal names the blocker: a non-safetensors
checkpoint, FSDP, `compile_dit`, `lora_low_rss` off, `slab_weights=off`, Gate
quarantine, or insufficient memory even for slab. Comfy-managed residency
forces slab off and cannot offer this rescue
([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)).
A loader `weight_dtype` cast disables this preflight because file size no
longer predicts resident size. Quarantine overrides consent; only a fresh
Identity Gate retest can clear it
([#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass)).

**If quarantine state is unknown:** consent cannot override it. The message
identifies the missing evidence:

| Reported condition | Next step |
|---|---|
| Incomplete artifact identity | Resolve the artifact identity before retrying. |
| Unreadable ledger | Check the ComfyUI output directory on that host. |
| Damage above the newest matching row, or damage not attributable to a matching capability context | Rerun Identity Gate for the combination. An unscoped read cannot order damage against a match; a scoped read may still be clear. |
| Matching row from another protocol, package version, or source build | Rerun Identity Gate on this build. |
| Unrecognized verdict | Rerun Identity Gate for the combination. |

Unknown state is nonwaivable class C, not class K. Only a persisted Gate FAIL
records an observed mismatch.

## 6. cfg2 render errors with "cfg-parallel got a model call of batch=…"

**Symptom:** typed error at render start under a cfg-parallel topology.

**Cause:** ComfyUI runs cond and uncond as separate calls when CFG == 1.0 or
the conditioning shapes are not concatenable, so there is nothing to split.
CFG 1.0 has one exception: a cfg++ sampler, whose name carries `cfg_pp`, sets
`disable_cfg1_optimization` and keeps the unconditional pass, so comfy folds
cond and uncond into one call of batch 2 and both cfg ranks work. That is why
the shipped krea2 template renders under `cfg2` at CFG 1.0 while the flux1
template, on the same preset and the same CFG but asking for `euler`, refuses.

**Fix:** for a family that supports cfg-parallel, raise CFG above 1.0, or keep
CFG 1.0 and pick a `cfg_pp` sampler, and use conditioning that can share one
batched call. On a pure cfg topology,
dgx-monarch applies the family-declared asymmetric-prompt padding contract;
combined USP+cfg presets can still refuse a real mask that the USP kernel
cannot honor. If the family has no batched cond/uncond call, use `auto`, a
documented family- and world-compatible topology whose physical cfg degree is
1, or a true one-GPU run. `uly2` or `ring2` is not a universal remedy.

## 7. Warning: "attention mask … IGNORED" under USP

yunchang/flash USP kernels support no arbitrary attention masks. Padded text
batches and regional prompts may degrade under ulysses/ring. Mask-critical
workflows need a family path that preserves their mask. For Chroma,
use `cfg2` when the graph makes a real batched cond/uncond call, or a true
one-GPU run with `mode=local` and `gpus_per_host=1`. Z-Image's stock Lumina
forward passes `cap_mask` as a hard `None`, so it has no documented
arbitrary-mask-preserving topology. See [#77](#77-a-chroma-omnigen2-or-z-image-usp-render-refuses-a-mask-or-sequence-length) for the typed guards and
family-specific remediation.

## 8. Version skew between hosts

**Symptom:** doctor row `torchmonarch=…; version skew vs driver pin`, a
failing `torch version match: skew across hosts` row, or pickling errors
mid-attach.

**Fix:** For a torch-version mismatch, install the same torch/CUDA build in
the driver and every worker environment; doctor treats the skew as a failure
because serialized objects and distributed kernels cross that boundary.
`dgxm update` uses `--no-deps`, so it does not replace torch.
For torchmonarch skew on plain source directories, run `dgxm update`
(synchronize the repo's exact pin with `--no-deps`, install/sync dgx-monarch
with `--no-deps`, restart, then doctor). When a host's
`~/.local/share/dgx-monarch/src` links into a release slot, as after guided
setup or a verified update, run `dgxm update --verify` instead: plain update
never changes a release slot. If the driver has already moved without those
Workers, repair that first
([#118](#118-workers-in-release-slots-still-run-the-previous-version-after-a-plain-update)).
Neither path resolves or replaces torch, NCCL, xfuser, or optional extras.
torchmonarch is pinned; bump it on all hosts at once.

## 9. Worker host death mid-render

An earlier host-death test observed `SupervisionError` after about 69 seconds,
responsive healthy hosts and same-mesh respawn after the host returned. That
result does not establish general same-driver recovery. The later 0.7.0
single-Worker stop/start check on 2026-10-13 (UTC) reproduced a replacement-attach
timeout and a poisoned driver transport; see [the dated lifecycle results](VALIDATION.md#lifecycle).
A poisoned transport requires a fresh driver after confirmed owned cleanup;
resetting its attached mesh does not clear that transport state. That check
came before per-start Worker ids; see [#2](#2-every-attach-times-out-after-one-failed-attach) for
re-attach after a Worker restart within the delivery timeout.

## 10. First RDMA latent return is slow

The [selected-rail measurements](VALIDATION.md#native-rdma) recorded a
1.989-2.139-second first-use native return cost. Repeated 32/64-MiB returns
benefited, while 8-MiB returns showed no consistent material gain. Those totals
combine pack RPC and read/drop/ACK; they are not network-only read timings.
Native return remains off by default. A failed or ambiguous native operation
must retain ownership until settlement; first-use cost never authorizes retry.
To measure your own selected rail, run the qualifier's `--timing` mode in
`benchmark/rdma_return_bench.py` in both orders and pair the reports with its
`--timing-summary` mode.

## 11. Custom sampler renders as euler / "sampler … is not registered on this worker"

**Symptom (0.1.0 and earlier):** a custom sampler picked in the KSampler
dropdown (RES4LYF `res_2s`, `res_3s`, …) renders at euler cost and quality;
multi-stage samplers silently under-compute. **Symptom (current):**
the render fails typed with "sampler … is not registered with ComfyUI on
worker host …", or the SamplerCustom path fails deserializing a SAMPLER
object.

**Cause:** the pack providing the sampler did not import inside the worker
procs, so its names never reached the worker's `KSampler.SAMPLERS`, and
stock comfy substitutes euler for unknown names (dgx-monarch raises
instead). The two known import blockers, both fixed or diagnosable here:
packs that register server routes at import time hit a missing
`PromptServer.instance` in headless actors (workers install an inert stub),
and packs whose python requirements are absent from the worker env
(RES4LYF: `pywavelets`, `opencv-python-headless`, `matplotlib`).

**Fix:** check the Worker service log for `custom node pack … failed to import`.
The message names the missing dependency. Install the pack's requirements
into the worker python (`python` in cluster.toml) on every host (e.g.
`~/monarch-env/bin/pip install pywavelets opencv-python-headless matplotlib`),
`dgxm restart`, re-render. The log line
`preloaded N custom node pack(s)` counts successful imports.

## 12. bf16 / large model load hangs at "0 MB usable" (UMA)

**Symptom:** a large or bf16 model never finishes loading; the Worker service
log shows comfy sizing it at ~0 MB and offloading, and it recurs on every
render rather than being a one-off OOM.

**Cause:** comfy's `get_free_memory` reads `torch.cuda.mem_get_info`, which
under-reports free memory on GB10 unified memory (measured: about
61 GiB reported vs 118 GiB actually free at idle, worse mid-render). The
estimate loader sizes the model at ~0 MB usable and offloads it on every
`prepare_sampling` re-management, so a one-shot force-full-load cannot fix it.

**Fix:** automatic. The worker patches `get_free_memory` at comfy bootstrap
to `psutil.virtual_memory().available` on integrated CUDA devices (host RAM
and VRAM are one physical pool), divided by the number of GPUWorker actors on
the host. bf16 krea2 then renders single-copy. Confirm the log line
`patched get_free_memory for GB10 unified memory`. Discrete GPUs are
untouched and fp8 is unchanged (it always fit the estimate). Still tight?
Drop the page cache too ([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)) so host-available is not crowded.

## 13. uly2+fsdp / ring2+fsdp render deadlocks (hangs, no error)

**Symptom:** an FSDP topology combined with sequence parallelism (uly2+fsdp;
ring2+fsdp is refused) shards the model correctly, then the render hangs
forever with no error raised.

**Cause:** FSDP all-gather runs over the world group while USP's Ulysses
all-to-all runs over the ulysses group: two NCCL communicators on the same
2 ranks. On world=2 their collectives co-schedule and deadlock. This is not
NCCL_PROTO=LL (a separate FSDP-killer; never set it).

**Fix:** automatic. Every worker setup sets NCCL_LAUNCH_ORDER_IMPLICIT=1 before
any communicator exists, whatever the topology (needs NCCL >= 2.26). NCCL reads
the variable once, at a process's first launch, and keeps it, so setting it
only for fsdp with sequence or CFG parallelism leaves a worker first set up
plain unguarded for later fsdp+SP renders; this caused stalls in testing. The env var is the whole guard. The fsdp adapter's empty FSDP2
forward-prefetch list pins a torch default and is no part of the fix
(worker_args `fsdp_prefetch_depth` above 1 fills it: a measured speed
setting, not a guard). The same guard covers `cfg2+fsdp` and `dp2+fsdp`. The
render is then byte-identical to the resident result, sharded to ~1/world
weights per rank, about 2x slower (the measured all-gather cost;
[VALIDATION.md](VALIDATION.md)). Still deadlocking? Confirm NCCL >= 2.26 on every
host and that no fabric profile sets NCCL_PROTO.

## 14. First render pauses ~9 s with `compile_dit` on

**Symptom:** with the Init `compile_dit` widget enabled, the first render
pauses (looks hung) before the first step; later renders are fast.

**Cause:** `max-autotune-no-cudagraphs` compiles each DiT block on its first
forward, causing a one-time inductor autotune warmup (about +9 s measured for
the fp8 Triton GEMM search; [VALIDATION.md](VALIDATION.md)). Expected, not a
hang.

**Fix:** let it finish once; the compiled blocks persist for the actor's life
(later renders and LoRA hot-swaps reuse them). It is opt-in and per-model:
the worker fails open to eager if a block cannot compile (log
`DiT block compile skipped (...); running eager`), and a successful pass logs
`DiT: N blocks compiled`. Turn the widget off to skip warmup entirely (the
compile flag is cleared on the next setup).

**Which models it compiles:** only a diffusion model with a `blocks` list.
krea2 is the one measured; other `blocks` families (wan, minimax) compile
without a fidelity record. Flux, Flux2, FLUX.3, Chroma, LongCat and MOSS-Image
name their lists otherwise, so they compile nothing and the worker logs
`compile_dit: ... has no blocks list, so nothing was compiled; running eager`.
For a checkpoint whose live-model family has already been detected, those
known no-op families leave requested slab residency available. Other families
with differently named lists stay stock until their compile surface is
covered. The first load of unfamiliar bytes also stays stock: the
worker never uses a header guess to combine slab residency with a potentially
compilable model.

## 15. Attaches still time out after `dgxm restart` (driver-side wedge)

**Symptom:** Worker services freshly restarted, `dgxm doctor` fully green, yet every
attach from ComfyUI still hits `MESH_ATTACH_CONFIG_TIMEOUT`, and a worker may
die hard (supervision: "no status received") on the first load after restart.

**Cause:** the long-lived driver ComfyUI process is holding poisoned Monarch
client state from an earlier crashed mesh. The worker-side self-heal
(defunct-mesh eviction) covers actor deaths, but a client whose transport state
was corrupted by successive crashes can keep emitting attaches that cannot
succeed, which then wedge the fresh Worker services again ([#2](#2-every-attach-times-out-after-one-failed-attach)).

**Fix:** restart the driver ComfyUI process as well as the Worker services:
`dgxm restart`, then kill and relaunch the comfy process (by PID or
`fuser -k <port>/tcp`; never an inline `pkill -f "main.py …"`, which
self-matches the shell issuing it). A fresh driver and Worker services attach cleanly.

**After a reply-channel timeout:** restored connectivity may leave the driver
unusable even when both Worker services remain healthy. In the reply-channel
fault test, Monarch closed the client reply channel after its 30-s
acknowledgement-delivery timeout. Collection timed out at its 10-s deadline,
but both reattach attempts in the same process timed out. After confirmed
owned-actor cleanup, a fresh driver completed two render and cleanup cycles
without restarting either Worker. Restart ComfyUI or the headless driver for
this case; do not treat traffic restoration as proof that the existing client
can be reused. The [test scope and failed same-process
recovery](VALIDATION.md#reply-channel-fault) are recorded separately from
successful recovery in the new process.

## 16. `single` preset refuses because the latent batch does not divide dp

**Symptom:** topology `single` on a multi-rank cluster rejects a batch-1
render, saying the preset leaves dp2 but latent batch 1 is not divisible by 2.

**Cause:** explicit presets tile the whole world: `single` at world=2 derives
dp=2 (each rank renders a batch slice), which needs `batch % 2 == 0`. Only
`auto` folds spare ranks into ring for batch-1 renders; explicit presets are
always interpreted literally.

**Fix:** use batch size 2 (two images, one per rank), or Init `mode=local` +
`gpus_per_host=1` for a true one-GPU render, or leave topology on `auto`.
A batch that VAEEncode turns into video frames, as in the Kroma upscale graph,
stays at 1 whatever the batch widget says, so only the last two fixes apply
there; see [the Kroma notes](MODELS.md#model-kroma).

The same refusal also appears at the loader node, before anything loads,
whenever the graph states a literal batch on an empty-latent node ComfyUI
ships, including the Flux 2, Chroma Radiance, Hunyuan Image and HunyuanVideo
1.5 latents. It is class P at both places, with one message: no consent and no
extra memory changes what a batch of 1 can be split into, so no card appears,
and a graph that is both over budget and batch indivisible is told about the
batch rather than offered a rescue it cannot use. A batch wired from another
node, or one the loader cannot read as a literal, gets its answer at render
submit from the real tensor. The MiniMax H3 audio+video latent states no batch
at all; the packed-latent rule refuses its packed shape above dp1 instead.

## 17. FSDP load refused: LoRA or non-bf16 checkpoint

**Symptom:** an FSDP preset (`uly2+fsdp`, `cfg2+fsdp`, `dp2+fsdp`) refuses
checkpoint precision or LoRA loading. A changing checkpoint may instead raise
`ArtifactBindingError` after the driver header preflight.

**What to do:** use a stable bf16, fp16, fp8, or int8 safetensors checkpoint.
For LoRAs, enable `lora_low_rss` and use bf16, fp16, or plain-dtype fp8 without
a comfy-kitchen wrapper. Otherwise use a resident topology (`uly2`, `cfg2`,
or `ring2`). These are class P refusals, with no waiver.

| Refusal | Meaning and remedy |
|---|---|
| `LoRA on FSDP-wrapped models needs lora_low_rss on` | Enable `lora_low_rss`. The shard-aware bake reads pristine checkpoint weights and retains each rank's dim-0 chunk; ComfyUI's load/backup path must not touch those shards. |
| `LoRA on an FSDP-wrapped model is not admitted for this checkpoint` | Comfy-kitchen scaled fp8, fp8mixed, and int8 (with or without convrot) are unsupported for this bake. Neither `lora_low_rss` nor `auto_gate` can admit them. A plain `float8_e4m3fn` file, such as Qwen-Image, is different and can be admitted. |
| `FSDP capacity mode supports bf16, fp16, fp8 and int8 checkpoints at launch (got quant=…)` | Use an admitted source precision. mxfp8/nvfp4 lack registered shard layouts; `weight_dtype=fp8_*` casts lack file-backed shard bytes. |
| `ArtifactBindingError` | Finish downloads, syncs, renames, and symlink changes on every host. Confirm the logical model name resolves to the intended stable file, then reselect or rerun the loader. |

If ComfyUI's default turns a BF16 source into FP16, select explicit `bf16` in
the DGX Monarch UNET loader. This changes live dtype only; it cannot convert an
FP16/FP32 source into valid BF16 evidence or bypass the header check.

**Precision limits:** uniform bf16/fp16 safetensors are admitted. A bf16 core
may include fp32 islands totaling at most 5% of file bytes; these are
replicated, with `quant=bf16` and
`live_dtype_profile=bf16_core_fp32_islands_v1`, plus island count and bytes.
Wan's exact small FP32 input convolution has its separate
`wan_fp32_patch_embedding_v1` profile, audited only with a bf16 core. The fp16
Wan pair with fp32 ingress layers must use resident topology or a bf16 source.
Above the 5% ceiling, mixed fp16/fp32 and all-fp32 files report `quant=fp32`
and refuse. Unrecognized precision or custom operation/quant wrappers report
`quant=unknown`. Legacy `.ckpt`/`.bin` cannot prove source precision; obtain or
convert a readable safetensors file. An explicit bf16 cast cannot admit them.

**LoRA checks can also fail during baking.** Quantized-key detection uses
`scaled_fp8`, I8 tensors, `*.comfy_quant`, `_quantization_metadata` in
`__metadata__`, or a `*.weight_scale` sidecar beside F8/I8 weights. Missed
markers are checked again by `capture_record`/`bake_key` in
`actor/fsdp_lora.py`.

FP32 islands need live/file dtype agreement too, which the header and coarse
profile cannot establish. Krea2 RAW bf16 casts its island to bf16 and fails
`_file_tensor`; ChromaRadiance's two islands remained fp32 in the measured
base FSDP census ([VALIDATION.md](VALIDATION.md#evidence-radiance)). The per-key
check raises a rank-symmetric, class P `UnbakeError` when agreement fails.

**Checkpoint and cleanup ownership:** the driver binds the descriptor to
device, inode, size, mtime, ctime, and the `diffusion_models` resolver result,
then re-resolves before any cache reuse, mesh lifecycle, setup, or load RPC.
Replacement, retargeting, disappearance, metadata drift, or remapping refuses.
Each worker independently checks the same identity before and after ComfyUI
loads it.

An aborted FSDP Gate reload, incomplete rank evidence, failed A/B comparison,
or rejected provisional PASS unloads all ranks exactly once. Cleanup requires
exactly one positive `unloaded` response per expected rank. Otherwise the mesh
becomes DIRTY; failure to publish DIRTY makes the handle defunct. Recycle the
client-owned mesh before retrying.

## 18. First LoRA change on a quant model is one-time slow (HISTORICAL)

**Symptom (older builds):** the first int8/fp8 LoRA change was 2-4x slower,
with a large `gpu_load` phase such as
`sample 117.44s (hot-swap gpu_load 67.4s)`. Later swaps returned to about 10-20 s.
This was quant-kernel warmup on the reload, not the swap itself, a leak, or
model eviction (`disable_smart_memory` defaults on).

**What to do:** update dgx-monarch. Retesting with slab weights, lazy un-bake,
and ComfyUI nightly found flat int8 swap times (10.9/9.7/9.7 s), with only the
initial bake still slow. If the first-swap slowdown returns, enable Init's
`load_profile`: it profiles each `load_models_gpu` call lasting at least 2 s
into a private, exclusively created
`~/.cache/dgx-monarch/profiles/dgxm-load-profile.<pid>.<t>.<random>.txt`; the
newest 20 reports are kept.

A large `gpu_load` on *every* render indicates eviction instead. Check
`uma_reserve_gb` pressure and co-resident workloads.

## 19. Blank ComfyUI canvas (grid renders, no menu/sidebar) after an update

**Symptom:** ComfyUI serves, the litegraph grid and FPS counter render, but
the Vue UI never mounts: no menu, no sidebar, stacked error toasts, a
full-screen block overlay. The browser console shows vue-i18n/vite errors
inside the frontend bundles. dgx-monarch's routes and extensions all answer
200, so the only doctor row that sees the fault is `frontend/backend skew`.

**Cause:** frontend/backend version skew. The served ComfyUI frontend is
newer than what the backend declares in `required_frontend_version`. The
usual trigger is launching with `--front-end-version
Comfy-Org/ComfyUI_frontend@latest`, which re-downloads the newest published
frontend on every launch: it works until the newest frontend is newer than the
version the backend declares, then the canvas goes blank with no change on
your side.

**Fix:** run `dgxm doctor` with the driver up; the `frontend/backend skew`
row names both versions and the fix: pin the launcher flag to the version
the backend requires (`--front-end-version Comfy-Org/ComfyUI_frontend@X.Y.Z`)
or update the ComfyUI checkout so the backend targets the newer frontend.
Pin the frontend version in persistent launchers; doctor warns on `@latest`
even while the versions match.

## 20. "value... is not available" on a DGX Monarch node after an update

**Symptom:** a workflow saved before an update fails to queue with errors
like `The value "first_use" for DGX Monarch Init's slab_weights is not
available`, with values that belong to a different widget on the same node.

**Cause:** ComfyUI stores a node's widget values by position, and a node
update inserted a widget mid-list, shifting every saved value after the
insertion point. Combo widgets reject the shifted value with an error;
numeric and boolean widgets accept it silently, so the whole node can be
misconfigured, not just the flagged rows.

**Fix:** update dgx-monarch and reload the workflow. Every DGX Monarch node
also saves a name-keyed value map, and on load the map wins, so a later
reorder cannot shift those saves. For an Init node saved before the map
existed, the frontend extension realigns the values from the Init widget-order
history; a toast reports the migration, and re-saving the workflow keeps it.
If a workflow still trips it (another node, or a layout that history does not
list), recreate the node: add a fresh one and rewire it, set your values, and
re-save. A right-click `Fix node (recreate)` item does the same and keeps the
links, but an optional extension adds it; ComfyUI's own frontend package has
none.

<a id="21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid"></a>

## 21. "sequence length needs divisibility padding... ulysses-only" on ring/hybrid

A render's token count does not divide the sequence-parallel degree, so
`shard_seq` must pad, and the exact pad exclusion exists only on pure-ulysses
topologies. Ring attention would attend the synthetic pad rows (measured
left-edge corruption), so the render refuses instead. The count is the joint
image and text count for Krea2 or Lens, the packed audio, video and text total
for MiniMax H3 (see [#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)), and either of the two streams for LTX. Mage Flow and
the flux family refuse here too. [ADAPTERS.md](ADAPTERS.md) names the
families that pass their pad rows as `drop_rows` on every topology, and so
refuse here, and those that pass them under pure Ulysses only and attend them
on ring without refusing; other families keep their own measured contracts.
The text side counts too: the Krea2 TE strips its chat template, and Lens's
GPT-OSS TE also produces small, prompt-dependent context lengths, so either
family's text stream can need padding on its own; cfg_pp samplers evaluate the
negative prompt even at cfg 1.0 (an empty negative is 5 tokens).
`auto` picks from the rule table, whose rows carry different evidence
levels; [MODELS.md](MODELS.md) states which are hardware-tested.
At world 4 with batch 1, `auto` folds a uly2 row's spare ranks into ring
(uly2+ring2) until uly4 passes a 4-rank acceptance gate; the explicit `uly4`
preset is available as an operator choice there.
Fixes, any one of: use a `uly*` or `single` topology at this resolution; pick
a resolution whose token grid divides the SP degree; adjust prompt/negative
lengths to divisible counts. On MiniMax H3 the `single` preset is not one of
the fixes: above world 1 it derives dp, and H3's packed latent refuses dp
([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)). Use `uly2` there, or `mode=local` with `gpus_per_host=1` for a true
single-GPU reference. H3 says so in its own wording, because the adapter reads
the ring degree itself before the first block runs. LTX reads it the same way,
and either of its streams can trip this alone: video rows go odd when the
latent frame count and both latent canvas edges are odd, and audio rows are
odd for about half of all clip length and frame rate pairs, so the fix there is
usually a frame count or a rate rather than a canvas.

**An expert waiver exists, and it is off by default.** This refusal is class K,
incorrect output: the synthetic pad row produced left-edge artifacts on Krea2,
and the same mechanism reaches every family that hits this guard. If you
intentionally need that math (a comparison leg, a bug repro, an image you
accept), the DGX Monarch panel raises a card in the accuracy style whose one
button reads "Render under waiver (output stamped)"; headless, set
`DGXM_WAIVE_KNOWN_WRONG=ring_pad` on the driver process. The card states the
measured error before you authorize the bypass. Every render that then goes
through is stamped rendered-under-waiver and writes a permanent WAIVER record
to the gate ledger, and no accuracy waiver is ever granted automatically. Full
flow, stamp and revoke in
[#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means).

## 22. "a previous model load could not be cleaned up" / model reuse is blocked

**Symptom:** after any model load fails, every later load or reuse on that worker
refuses with `a previous model load could not be cleaned up`, or an explicit
unload reports `failed-load cleanup remains blocked`. Worker status shows
`failed_load_cleanup_pending: true`. `retained_failed_load_slabs` is nonzero
only when the failed load owns a slab. `retained_failed_load_resources`
separately counts pins and other non-slab owners; both counts can remain zero
when cleanup is poisoned without a concrete retained resource.

**Cause:** global model unload, cache cleanup, collection, or closing an owned
load resource was interrupted or failed. A slab-backed load can leave tensors
whose Parameters point directly into its mapping; a stock/FSDP load can leave
uncertain Comfy model state or a checkpoint pin that Comfy may still use. The
worker retains any concrete resource and latches cleanup poison even when no
resource exists, then refuses all further model work. This prevents a use after
close, or reuse of process state whose cleanup was never confirmed. The original
load error remains primary; the driver log records the cleanup failure
separately.

The ownership boundary begins before acquisition, not only after a load throws.
One process-rooted fresh-load transaction owns each pin or slab through the
interruptible Comfy load and records the exact model-store slot candidate before
publication. It releases those children only after that exact object is
confirmed in that exact slot, so recovery cannot close a resource under an
already-published resident. A slab load also prepublishes restoration of both
temporary process-global Comfy hooks. If either hook cannot be restored to its
exact original object, the hook guard and slab remain rooted and arena close is
blocked until a later cleanup confirms both restorations.

A low-RSS LoRA lazy swap is one form of the same ownership problem. It restores
and re-bakes checkpoint weights in place, so a cancellation, native panic, or
other non-`Exception` interruption can leave a resident only partly changed.
The worker discards that slot before it propagates the interruption, and the
next request does a canonical load. If the discard is also interrupted, the
slot reports `a previous unload failed` and remains blocked; old request
metadata never makes that resident reusable.

There are two retry classes. Failure during process-wide unload/cache cleanup,
FSDP alias unlink, or `mmap.close()` with a live exported view occurs before
numeric descriptor release; the worker still owns the original FD or mapping,
so a later explicit global unload may retry it. Once an arena, checkpoint pin,
or temporary slab checkpoint-source owner enters `close(2)`, any exception is
ambiguous: the kernel may already have released and reused that numeric FD. The
worker marks the owner uncertain before the syscall, never retries that number,
and keeps cleanup poisoned. A slab attempts both its primary and optional annex
arenas, but either arena becoming uncertain makes the slab unusable and
retained.

**Fix:** if the worker log already says a checkpoint, slab-source, or slab-arena
descriptor close outcome is uncertain, skip in-process cleanup retries and use
**Reset attached mesh** at once. Otherwise, first run an explicit all-model
unload with a DGX Monarch ClearVRAM node at level `hard`. A successful
process-wide unload/cache/collection retry closes retry-safe retained resources
and clears the poison. If it still reports
`failed-load cleanup remains blocked`, use the sidebar **Reset attached mesh**
button: it stops the mesh-owned actor processes without unloading their models
first, because process exit frees them, and the next render attaches fresh
actors. Do not keep retrying renders against the blocked actor: only an
actor-process reset can safely discard an uncertain descriptor table. If the
reset itself reports incomplete, leave the mesh blocked and inspect the driver
and Worker service logs before using `dgxm restart` as the deployment-level
recovery.

## 23. Status or dashboard polling seemed to destabilize Worker services

**Symptom:** after running an older `dgxm status`, `dgxm doctor`, auto-heal, or
`dgxm top`, a later attach times out even though the worker port still appears
to listen.

**Cause:** older builds used protocol-blind TCP connect/disconnect readiness
probes against Monarch worker endpoints. Monarch owns connection lifecycle, so
a bare TCP handshake is not a valid health request. The failure is
state/order-dependent rather than a guaranteed result of every connection.

**Fix:** update dgx-monarch, then run `dgxm restart` once if the old probe
already left a Worker service unusable. Lifecycle checks verify the exact
process and read the host's passive LISTEN table through the local/SSH runner.
`dgxm top` uses driver telemetry and shows `?` when Worker service state is
unavailable; it never connects to worker ports or SSHes to every host on its
polling interval.

## 24. An older worker log contains inherited environment values

**Symptom:** a Monarch launch diagnostic in `worker-loop.log` or its runtime
scratch log includes values inherited from the shell or service that started
the loop.

**Cause:** older dgx-monarch launches passed the worker loop's ambient
environment into the pinned runtime, whose native process-launch diagnostic
can render child launch options. Older nohup logs also followed the caller's
umask.

**Fix:** treat any credential present in an old log as exposed and rotate it;
permissions or deletion do not revoke it. Update dgx-monarch and run
`dgxm restart`. Worker services replace ambient inheritance with a small exact
allowlist before Monarch imports, take fabric settings only from validated
`cluster.toml`, route runtime scratch into an owned `0700` directory, and open
nohup/systemd logs under `0077`. Do not add broad `PassEnvironment=` entries to
the generated service.

## 25. CI reports `leak-check[...]` for a tracked file

**Symptom:** the leak-assertion CI step fails with a repository-relative
file, line, column, and rule, but does not print the suspected value.

**Cause:** a Git-tracked working-tree file or filename matches a private
identifier rule (such as a home path, a rig-style hostname or a private network
address) or a credential-shaped value. The diagnostic omits the value so CI
logs do not turn a local mistake into a second disclosure.

**Fix:** run `python tools/leak_check.py` locally and inspect the named location.
Remove or replace the private value; rotate it if it was a real credential.
Do not add a credential or rig identifier to the allowlist: the code accepts
allowances only for the fictional `GPU-deadbeef` / `examplebox-*` fixture
namespaces, each pinned to an exact file and match count. Write example
addresses from `192.0.2.0/24`, which the address rule does not match.

## 26. Comfy canary reports `Comfy touchpoint missing` or `incompatible`

**Symptom:** the scheduled/PR ComfyUI-master job, or a local run of
`tests/canary/comfy_entrypoint_canary.py`, exits with a dotted Comfy path and
possibly a missing signature parameter.

**Cause:** the Comfy checkout no longer exposes an internal API that
dgx-monarch imports, patches, or calls. The manifest checks these before a
model load, so upstream drift fails at the named contract instead of later as
a worker crash or a silent sampler substitution.
For rebound forwards, the surface assertion imports every declared target and
compares its exact parameter names, kinds, required/optional status, and
variadics; drift is incompatible even if the replacement has `**kwargs` and
Python would accept it silently.

**Fix:** first reproduce against the same Comfy commit with
`COMFYUI_DIR=/path/to/ComfyUI python tests/canary/comfy_entrypoint_canary.py`.
If the two hosts run different Comfy revisions, align them and retry. If master
changed, update the affected dgx-monarch call site and its
`src/dgx_monarch/comfy_surface.py` or rebound-contract declaration together,
run the full CPU and Comfy canary battery, then leave hardware render/identity
validation to the rig owner. Do not delete or relax the assertion without
removing the corresponding production dependency; the repository AST inventory
test also refuses an undeclared `Adapter.bind`/`self.bind` site or a declaration
with no matching bind.

## 27. torchmonarch canary reports compatibility or semantic drift

**Symptom:** the weekly/manual `torchmonarch-canary` workflow fails its final
gate. Its `torchmonarch-surface-*` artifact contains exact-pin and latest-stable
JSON snapshots, install logs, comparison JSON, and Markdown reports. A failure
names a missing/incompatible dotted Monarch surface or an event-loop semantic
change.

**Cause:** either the installed exact pin no longer matches the checked pin
snapshot, latest stable changed a production call shape, or raw `PythonTask` /
public `Future.get()` active-asyncio behavior changed. The workflow treats
semantic drift as review-required even when it might be an improvement,
because dgx-monarch's off-loop and bounded-wait paths are built on the old
behavior.

**Fix:** inspect `checked-pin-report.md` first. If it fails, reproduce the exact
pin before trusting the latest comparison. Then inspect
`latest-vs-pin-report.md` and the two snapshots; annotation-only signature
changes are informational, while missing call shapes and semantic changes are
blockers. Adapt the production caller and manifest together, run the full CPU
suite plus real Monarch/hardware gates appropriate to the affected surface,
and only then bump every exact-pin mirror by hand, with a changelog entry.
The canary never auto-bumps or writes to the repository.

## 28. Operator-armed setup-kill handoff does not reach PASS

Setup-failure tests use reviewed private scripts under the manual validation
contract in [THREAT_MODEL.md](THREAT_MODEL.md). CI never runs them.

**Symptom:** an authorized setup-failure test does not complete. It may report an unwitnessed TCPStore
listener, an untyped worker death, an exceeded shared bound,
`replacement_blocked`, a reused PID, failed world-1 setup, or an unconfirmed
stop of the recovered ProcMesh.

**Cause:** the probe kills its one client-owned worker during a real
incomplete NCCL setup. Every later assertion is a separate part of the
lifecycle contract; a healthy respawn does not excuse unconfirmed ownership or
cleanup, and a generic timeout is not typed supervision evidence.

**Fix:** establish the outcome of the previous run and confirm cleanup of its
owned processes before starting another test. Verify the exact reviewed source
on every host and obtain authorization for the destructive test. Match the failing
phase to its action: verify the test environment's Comfy/CUDA/xFuser install for a
missing setup witness, inspect production supervision and teardown for
typed-failure/retirement failures, and restart the driver only after process
ownership is known when `replacement_blocked` appears. Never
relax the PID ownership, typed exception, fresh-PID, timeout, or recovered-stop
assertions. A passing result requires every phase and confirmed cleanup of the
test's recovered ProcMesh. It does not authorize stopping a persistent Worker
service or qualify model fidelity.

## 29. Large RDMA latent return fails after workers finish rendering

**Symptom:** a large-latent prompt fails on the driver after workers have
finished and may already report matching latent signatures. The driver log
mentions a Monarch PythonTask or `Future.get()` being used while an asyncio
event loop is active; an attempted RDMA registration release may report the
same refusal. Small latents can appear unaffected because they stay below the
RDMA threshold and return through actor messaging.

**Cause:** older driver-side materialization called a blocking Monarch read (and
then blocking release cleanup) from ComfyUI's active event loop. Current
dgx-monarch runs the complete local read/drop and descriptor ACK transaction in
one bounded scratch job. Its join envelope contains a 120 s native-read budget,
a 15 s margin on the future get, one 120 s drop budget per descriptor part, up
to 2 ACK broadcast attempts with a 120 s budget per attempt, and a final 30 s
outer join margin. The generation lease is published before the thread handoff,
so a delayed target or caller timeout cannot authorize recycle over work that
may still start.

On the worker, each production descriptor and its CPU backing are owned under
the exact setup generation and a random 128-bit token. The registry is capped at
pipeline depth; saturation uses actor messaging and never evicts an unACKed
entry. A successful driver read does not by itself free that backing. After a
completed read, or an off-loop scratch proof that the native operation never
started, the driver must confirm every local native drop and read-authority exit
before broadcasting an ACK. The response set must contain exactly the configured
worker count, exactly one
`released`/`already_released` owner, and `unknown` everywhere else; rank numbers
are not used as ownership identity. Only that validated ACK retires worker
registry ownership in-process.

**Fix:** update dgx-monarch consistently, restart the ComfyUI driver so it
imports the updated source, and verify the configured hosts with `dgxm doctor`
before retrying. If an update cannot be deployed immediately, set
`rdma_latent_return = false` in the cluster configuration and restart the
driver to use the slower actor-message return path; on a managed installation
that edit reads `POLICY_STALE` until `dgxm update --verify --target HEAD`
re-renders the Worker units (#113). Do not infer a model or
distributed-math failure from this transport error; verify the per-rank latent
signatures separately.

**A failed native read also switches native return off.** Any native read that
fails on the driver latches native return off for that ComfyUI process: later
renders return their latents by actor messaging, and only a ComfyUI restart
turns native return back on
([#115](#115-a-render-fails-with-nativereturnlatchederror-and-later-renders-return-by-messaging)).

**If a started native read is incomplete.** dgx-monarch performs zero native
drops and sends zero ACK; it retains the destination and every part because the
operation may still be live. If a read completed, or an off-loop contender
proved that it never started, but a failed or ambiguous drop occurs, that handle
is not retried and the driver again sends zero ACK. `RDMA registration ownership
POISONED` identifies this local process-lifetime ownership. Constructor
ambiguity on the worker is stricter: production RDMA is refused with
`RDMA registration ownership is ambiguous; Attached mesh reset required` rather
than treating the generation as reusable; the operator action is
**Reset attached mesh**.

**If both scratch threads fail to start.** When both prebuilt scratch threads
fail to start before either claims the job while ComfyUI's event loop is
active, dgx-monarch performs zero native reads, drops, or ACKs on that caller
thread. The bounded `PENDING` job, generation lease token, descriptor parts,
and worker backing remain rooted; lifecycle and recycle refuse the handle. In
that state, restart the ComfyUI driver process. A ProcMesh recycle is not
enough, because the driver-local scratch-job authority is what could not be
transferred.

**If an ACK response is lost.** An ACK response can be lost after the worker
applied it, so retries repeat only the idempotent generation/token ACK. A
partial or malformed response set, zero or multiple visible owners, or two
unconfirmed ACK attempts fails collection and abandons/blocks the generation;
do not infer whether remote backing survived an ambiguous response. Multi-DP
collection still drains every unique leader token before it raises the
strongest failure. Operational cancellation outranks ordinary errors; within
the same class, the first error remains primary and later failures are
diagnostic notes.

Preserve that selected registration/read/ACK error as the diagnosis and leave
`rdma_latent_return = false` while recovering. Production does not recycle
automatically. After pending work is retired, explicitly recycle the
client-owned ProcMesh with the DGX Monarch panel's **Reset attached mesh**
action or the authenticated `/dgxm/recycle` endpoint. Restart the driver only
for driver-local process poison or when the
process-stop outcome cannot be confirmed. Do not force-clear a retained owner
or retry a numeric native handle in place.

## 30. Sampler Custom fails before model loading with `ModuleNotFoundError`

**Symptom:** a workflow using stock `RandomNoise` + `KSamplerSelect` into
`DGXMonarchSamplerCustom` fails before any checkpoint load. The exception names
`comfy_extras/nodes_custom_sampler` as a module that cannot be imported on the
worker.

**Cause:** current ComfyUI loads built-in extra-node files with a source-path
module identity. Cloudpickle can serialize an object while that path key is
present in the driver, but a separate worker cannot import the path as a Python
module.

**Fix:** update dgx-monarch on every host and restart the driver/workers. Stock
RandomNoise/DisableNoise objects are then rebuilt under Comfy's canonical
importable module before the existing by-reference transport. Samplers and
third-party NOISE objects keep their prior behavior; use [#11](#11-custom-sampler-renders-as-euler--sampler--is-not-registered-on-this-worker) when a custom pack
itself fails to deserialize on workers.

## 31. Kandinsky5 output does not match the official workflow

**Symptom:** a Kandinsky5 render completes, but its denoising trajectory or
output differs from the official Image/Video workflow even though the seed,
steps, scheduler, conditioning, and checkpoint match. Video is the common case:
the current checkpoint default reports SD3 sampling shift 10 while the official
workflow patches it to shift 5.

**Cause:** the SD3 sampling shift is render-time model math, not a checkpoint
loader option. A plain DGX Monarch model handle therefore keeps Comfy's
checkpoint default unless the graph applies the sampling node. The shift does
not belong in loader options, where it would change model residency, and
patching the shared resident in place could leak it into later warm renders.

**Fix:** place `DGXMonarchModelSamplingSD3` after the DGX Monarch model/LoRA
chain and before the sampler. [MODELS.md](MODELS.md) carries the per-variant
shift values and what the patch does and does not touch. The Kandinsky5
validation runs checked the shift lifecycle on a reused resident model,
certified Ulysses/ring/cfg2/FSDP raw-latent fidelity, and ran the full official
Image/Lite/Pro settings. If a render still differs, first verify that the
sampling node is present and connected after every model/LoRA branch.

## 32. Hunyuan Image 2.1 refiner is auto-detected as LongCat

**Symptom:** Init with topology `auto` selects a LongCat policy row for a
Hunyuan Image 2.1 refiner checkpoint, even though the matching base checkpoint
selects Hunyuan.

**Cause:** the public refiner export wraps tensor names in `model.model.` and
identifies the variant with TokenRefiner plus `time_r_in` keys. Older
dgx-monarch releases lacked the exact nested-wrapper normalization and knew
only the Hunyuan Image base and Video 1.5 markers. Without the refiner marker,
the shared Flux-style block names fell through to the broader LongCat
signature.

**Fix:** update dgx-monarch on every host and restart the driver/workers. Auto
then normalizes the exact nested ComfyUI wrapper and recognizes the refiner
signature before Flux/LongCat. Until the update is deployed, choose an
explicit supported topology such as `uly2` for this refiner instead of
`auto`; this workaround does not certify hardware fidelity by itself.

## 33. A live topology or residency-policy change times out while retiring a large resident

**Symptom:** a warm render completes, then changing distributed topology or
switching the same topology to a different residency policy eventually raises
`TopologyTransitionError`. A topology replacement logs `topology change:
tearing down old NCCL groups on live actors`; a policy-only change retires the
resident without rebuilding groups. The error reports the setup generation and
a cleanup outcome such as `timeout_unknown`; later renders or setup attempts on
that fleet are refused until it is recycled.

**Cause:** injected models and attention dispatchers belong to one process-group
generation, so a topology change must unload them before destroying and
rebuilding the groups. A policy-only change must also unload models built under
the prior residency knobs. Older releases put either retirement behind an
opaque 120-second wait. Large residents could take longer to unload than that timeout. The driver
could time out before group destruction began while the worker operation
continued in the background. The old explicit teardown path also kept
its local setup key published until every destructive stage succeeded.

**Fix:** update dgx-monarch on every host. Current releases consume the next
setup generation and retire setup identity before cleanup, allow a bounded 600
seconds for resident retirement on both policy-only and topology-replacement
paths, and require every rank to acknowledge an `UNSETUP` state before
replacement setup starts. Outstanding sample results also keep their
generation through raw-latent materialization, every local RDMA drop, and the
descriptor-owner ACK ([#29](#29-large-rdma-latent-return-fails-after-workers-finish-rendering)); an incomplete read, ambiguous drop, or unresolved
ACK abandons and blocks that generation until an explicit ProcMesh recycle.
dgx-monarch treats a collection timeout as terminal: it requests cancellation
and abandons the unread result, so it cannot be collected later. A raw read
already running keeps the active lease until its scratch thread exits, and
that abandoned work needs a recycle before a topology change.
If a typed DIRTY result still occurs, recycle the client-owned mesh with the
DGX Monarch panel's **Reset attached mesh** action or the authenticated
`/dgxm/recycle` endpoint and confirm cleanup. A confirmed recycle permits a
retry in the same driver. Use a fresh driver process only when transport is
poisoned ([#2](#2-every-attach-times-out-after-one-failed-attach)) or the process-stop outcome is unconfirmed ([#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)).
Do not queue more renders on the DIRTY handle, and never restart Worker services
underneath a live driver mesh.

## 34. A render is refused because another render session owns the mesh

**Symptom:** a direct render, Pipeline, Fleet, Identity Gate, loader, scheduler,
or ClearVRAM operation raises `ConcurrentRenderSessionError` saying another
render session did not drain within the bounded timeout. If an earlier
residency RPC timed out or its dispatch result was lost, the refusal may instead
be a `TopologyTransitionError` reporting a DIRTY `... RPC completion` phase.

**Cause:** one mesh has one setup and model-residency state, and each live
sample owns setup-generation, descriptor-read, and ACK authority. Independent
logical sessions cannot safely change topology or worker policy, or unload
backing while another result/RDMA transaction or Identity Gate check is still
live. `DGXM_RENDER_SESSION_TIMEOUT` bounds both handle-lock acquisition and
owner draining; increasing it only changes how long a contender waits. A
timed-out or return-lost mutating RPC has unknown remote completion, so the
handle latches DIRTY rather than let a new session overlap it.

**Fix:** let the current queue, Gate, Fleet wave, or PendingRender finish; do not
bypass the owner or run two independent queues against the same mesh. If the
error identifies a DIRTY RPC-completion phase, use the authenticated DGX
Monarch **Reset attached mesh** action and confirm the actor processes stopped
before retrying.
Only classified supervision reconciliation can bypass session ownership; the
public lifecycle API cannot force it. A confirmed recycle permits a fresh setup
in the same driver, while an unconfirmed process-stop outcome still requires
operator verification as described in [#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

The panel's **Reset attached mesh** action and its internal `/dgxm/recycle`
route preserve that rule: a reset requested while a session or sample lease is
active stops nothing and returns a retryable busy result, so wait for the owner
to drain, then retry. See [#78](#78-the-dashboard-records-unknown-or-mesh-reset-says-busy-or-outcome-unknown) for a reset that reports busy or an unknown
outcome, and [#69](#69-worker-bring-up-is-interrupted-during-spawn-rollback-or-client-lease-startup) for when a failed stop allows another and why a timed-out one
never does.

## 35. Identity Gate reports `ERROR`/`INCONCLUSIVE` after a prior PASS

**Symptom:** an ordinary combination that previously passed is forced to stock
residency, an FSDP request raises a typed clean-reload refusal, or the ledger
contains a `RETESTING` row and no newer final verdict. Logs may also say that
the Gate ledger is unreadable, damaged, or not writable.

**Cause:** positive Gate authority fails closed. Before a retest starts, the
Gate writes one durable `RETESTING` transaction, a revocation that survives a
crash. An ordinary residency check revokes its normal, Fleet, and eligible
explicit-slab contexts; a no-LoRA FSDP check revokes only its separately
scoped normal-render context. A killed check or failed final append leaves
that denial in force. Likewise, an invalid-schema or unreadable row newer than
an older PASS may have been a revocation, so the older PASS is not resurrected.
A valid exact row written after earlier damage heals only that context;
parseable FAIL quarantine remains an availability block until a complete
explicit retest from the current source records a newer `PASS` in that same
exact tested configuration. A PASS in another context cannot clear it. A
later `INCONCLUSIVE` or `RETESTING` row does not supersede that compatible
FAIL, and a foreign-source terminal row cannot mask it. Without a governing
FAIL, either status still denies. Contextual `PASS`, `INCONCLUSIVE`, and
`RETESTING` rows, and the process, normal-render, and Fleet tokens, bind the
package-source manifest cached on first use in that process. A missing or
different source identity is stale and authorizes nothing; a sticky FAIL
quarantine does not depend on the source.

**Fix:** correct output-directory ownership/storage errors first; do not delete
the ledger to regain optimized residency. Run the explicit Identity Gate again
for the exact model/LoRA/config/topology context and let it finish. A
current-source PASS clears the older FAIL boundary only for that exact context.
If the checkpoint or LoRA bytes, ComfyUI commit, package version or source
manifest, protocol, setup, or physical configuration changed, a fresh proof is
required. Restart every process from the same source before re-gating. Ordinary
failures remain stock; FSDP failures must complete all-rank unload (or recycle
a DIRTY/defunct handle) before another FSDP attempt.

## 36. Packed audio/video sampling fails after denoising

**Symptom:** an LTX audio/video workflow finishes denoising and then raises on
`.detach()`, `.cpu()`, `torch.equal`, `.abs()`, or latent-shape access. A
custom sampler may instead return a flat or missing second output. Older Gate
builds can appear to inspect the video-shaped first modality without proving
the audio modality.

**Cause:** ComfyUI represents packed audio/video samples as `NestedTensor`.
Its public shape describes the first modality, while Tensor-only operations do
not validate the complete wrapper. The custom sampler's second output is
rebuilt from the callback x0, which arrives either as one flat pack or as the
nested view of it, and either way must come back as the original ordered
modality shapes.

**Fix:** update dgx-monarch on every host, restart the driver and Worker
services from the same source, and rerun the exact check. The Gate compares
every finite modality in both proof legs and includes every modality in the
immutable transaction snapshot. Sampler leaders preserve a detached, contiguous
CPU wrapper; packed returns use actor messaging, never RDMA. SamplerCustom
takes either x0 shape and returns a distinct shape- and dtype-matched denoised
wrapper; a flat x0 must carry the exact packed width. Pipeline and Fleet
aggregate modalities independently. Mixed, malformed, non-finite, aliased, or
structurally drifting outputs fail closed. Packed DP is a typed refusal, so
choose a topology whose model-parallel degree consumes the full world. Every
contextual PASS row from an older protocol is stale and re-proves once under
the current one.

## 37. Strict provenance refuses setup or an operation snapshot

**Symptom:** a sealed validation run fails before rendering with
`strict provenance requires custom nodes disabled at bootstrap`, reports that
the custom-node bootstrap policy changed, lacks its READY setup snapshot, or
rejects a Git query, source, or loaded-module origin as dirty, inexact, or
outside the attested tree.

**Cause:** strict evidence is bound to one worker process lifetime and one
source-only bootstrap policy. Changing `disable_custom_nodes` after ComfyUI was
imported, relying on the legacy `DGXM_NO_CUSTOM_NODES` preload switch, running
different or dirty source on either host, retaining stale managed-source
siblings/bytecode, or loading an owned module from another path invalidates the
claim. These refusals guard evidence; they are not render fallbacks. Only the
top-level ComfyUI `custom_nodes` directory is exempt from Git status,
hidden-index-flag, and source-inventory checks, and only when the fixed
`custom_nodes_disabled` bootstrap policy kept it from loading. Every other
checkout path stays strict, as do hidden index flags on included paths, tracked
sources, and loaded origins.

**Fix:** stop the live driver and recycle its owned mesh before touching Worker
services. For a sealed campaign, set `[worker_args] disable_custom_nodes = true`,
restore clean exact dgx-monarch and ComfyUI checkouts, then run `dgxm restart`
so managed source is exact-synced and both worker processes start under the
source-only bytecode policy. Confirm `dgxm doctor` and source parity before
retrying from a fresh driver. Do not substitute `DGXM_NO_CUSTOM_NODES`.
Interactive workflows that need third-party nodes leave the strict policy false
and cannot claim strict campaign provenance. Git installed under a prefix works:
the worker looks it up once on its bootstrap `PATH`, pins the resolved absolute
path, and runs it with a sanitized environment. Put it on that `PATH` before the
restart and do not replace it while the process lives; a changed executable
identity fails closed.

## 38. FSDP exact checkpoint pinning requires Linux procfs

**Symptom:** an FSDP load refuses before Comfy opens the checkpoint with
`FSDP exact checkpoint pinning requires Linux with procfs mounted at
/proc/self/fd`.

**Cause:** the exact-inode load contract uses Linux's `/proc/<pid>/fd/<fd>`
path to give Comfy a suffix-preserving alias for the already-proven open file.
Non-Linux workers do not provide that contract, and a Linux container can hide
or omit procfs. Continuing would reopen the user-visible path and bring back
the checkpoint-replacement race that the FSDP proof closes.

Teardown removes the alias before it closes the checkpoint descriptor. A failed
alias removal can be retried, because the worker still owns the descriptor.
Once the descriptor close begins, any reported error counts as an ambiguous
release that the worker never retries; use **Reset attached mesh** as [#22](#22-a-previous-model-load-could-not-be-cleaned-up--model-reuse-is-blocked)
describes.

**Fix:** run FSDP workers on Linux with procfs mounted so `/proc/self/fd`
exists, then retry from a fresh load; otherwise use a resident non-FSDP
topology. The refusal comes before the checkpoint descriptor or the temporary
alias directory exists.

## 39. Plain Wan Ring2 refuses the TORCH_FLASH q/k/v dtype

**Symptom:** a plain Wan `ring2` render raises
`Wan ring TORCH_FLASH is validated for matching FP16/BF16 q/k/v only` before
sampling enters ring communication.

**Cause:** the bounded pure-Ring2 path first proves that sequence-parallel rank
order matches the bound ring-group slots and their global-rank mapping. For
each local batch it then places both K/V shards directly into that verified
order and runs one native ATen Flash Attention call with local Q and
full-context K/V. It does not use yunchang's low-precision block outputs or
block-LSE merge. On the validated kernel contract, matching FP16 or BF16 q/k/v
preserve their input dtype; FP32 activations or mixed q/k/v dtypes are outside
that contract. The refusal comes before ring P2P; the path never casts
activations or picks an unverified fallback.

**Fix:** leave stock Comfy Wan inference dtype selection in place: default
quantized loads normally compute attention in FP16, while BF16 loads remain
supported. Remove any custom operation that forces FP32 or mixed attention
projections. A uniformly FP32 workflow must use `single` so stock Comfy can
select an FP32-capable attention backend; mixed projections must be corrected.
Do not cast FP32 to a lower dtype to get past this guard. Full-context K/V
uses more peak memory than the dependency's blockwise ring algorithm, but the
temporary allocation is bounded to one local batch. If that allocation does
not fit, use `uly2` or `single`; the pure-Ring2 path raises the failure and
never falls back to blockwise merging. This is the runtime contract, not a
hardware test result: rerun the sealed Ring2 matrix before promoting a
new build.

## 40. Cross-rank identity refuses malformed or topology-misgrouped evidence

**Symptom:** a distributed render raises `cross-rank latent identity FAILED`
and reports either malformed latent signatures or a global-rank/DP-rank
mismatch. No leader output is returned.

**Cause:** the driver validates each worker's complete canonical latent
signature before grouping or comparison, even when a DP group has only one
rank. It also derives the only valid DP coordinate from the global rank and the
pinned model-parallel width. Missing fields, inconsistent shape/count data,
non-finite statistics, malformed digests or projections, and balanced but
swapped DP groups are damaged evidence rather than a render result.

**Fix:** keep the exact same topology snapshot on the driver and every worker,
then recycle any worker that retained stale setup state and retry. If the
failure persists on a clean mesh, preserve the generic error and worker logs
for diagnosis; do not copy latent-signature payloads into reports or relax the
check. A custom worker or adapter must emit the canonical signature the stock
worker path produces, never partial evidence of its own.

## 41. LoRA reloads after `unbake capture skipped`

**Symptom:** `unbake capture skipped (reason)` appears during a quantized LoRA
load. Rendering may succeed, but subsequent LoRA changes reload the model;
status lacks `low_rss` and authoritative unbake keys.

**What to do:** update every host and restart driver/workers from the same
source. Do not edit checkpoint markers or bypass the low-RSS status gate.
Unsupported markers or mismatched live conversion must fall back to reload.

Two valid formats were rejected by older builds:

* **Wan FP8-scaled:** a zero-length root `scaled_fp8` tensor. Current builds
  accept zero-, one-, or two-element legacy markers; only two enables
  `full_precision_matrix_mult`. Dtype, namespace, scales, live quantization
  state, and checkpoint bytes are still checked.
* **MiniMax H3 int8-convrot:** per-layer `<layer>.comfy_quant` markers rather
  than `_quantization_metadata`, with recipe
  `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
  Current builds accept the direct recipe flat or under `params`, retaining
  byte, live-contract, and owning-module checks. The defect was marker format,
  not pruning; embedded Krea2 metadata already worked.

An unsupported recipe key still forces reload and is named in the log, for
example `the marker names 'per_row', which the int8_tensorwise restore does
not replay`.

## 42. A resident-adoption proof render is refused

**Symptom:** an explicitly instrumented proof render fails with a
`resident-adoption` context, artifact, setup, request, rank, or cross-rank
evidence error. A second render in the same context is refused even when the
first attempt failed. Pipeline samplers reject the context before submission.
Ordinary renders that do not opt into this private proof scope are unaffected.

**Cause:** the proof contract authorizes exactly one render submission and
requires every rank to report the same request-bound resident model and
full-content artifact identities. A reused or malformed context, missing rank,
changed artifact, stale setup generation, request drift, or disagreeing worker
evidence is damaged proof rather than permission to continue. Multi-render
pipeline surfaces cannot preserve this one-shot authority.

**Fix:** preserve the exact error and per-rank worker logs, contain the failed
attempt, and audit the independently authorized prompt, artifact hashes,
runtime setup, and context digest. Start a separately authorized fresh proof
context only after correcting the mismatch; do not automatically retry, copy
evidence between attempts, relax all-rank validation, or route the proof
through a pipeline sampler.

## 43. A render dies with `takes from N to M positional arguments but M+1 were given`

**Symptom:** after a ComfyUI update, a rewritten forward raises an argument
count `TypeError`, such as
`Krea2Adapter.inject_usp.<locals>.usp_forward() takes from 4 to 6 positional arguments but 7 were given`.
Gate may then abort and stock fallback may report a `LifecycleBusyError`
saying
`1 abandoned sample(s) may still own a result or RDMA descriptor of their setup generation`.
Diagnose the original `TypeError`; the lease refusal is a consequence.

**What to do:** update dgx-monarch to match ComfyUI. Doctor checks
infrastructure and may remain green. Until a compatible release is available,
topology `single` uses no rewritten forward. On a newer ComfyUI commit, run
`python -c "from dgx_monarch.comfy_surface import assert_comfy_surface; assert_comfy_surface()"`
against that installation.

On older builds without Gate-abort cleanup, use `dgxm restart`, then restart
ComfyUI to clear the blocked fleet. Current builds leave the mesh clean when
the fallback push was refused before dispatch; the remaining abandoned lease
can recover through the one-respawn path in
[#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).
The underlying `TypeError` still abandons its lease and needs a compatible
adapter.

**Why:** rewritten forwards require exact stock signatures and must accept
every declared call shape. Inserted or reordered arguments, swallowed new
keywords, and newly populated positional slots can break this contract even
when imports succeed. `comfy_forward_contracts.py`,
`comfy_rebound_signatures.py`, and `comfy_rebound_sites.py` define and check
these bindings; stock outer callers have separate prefix-compatible checks.

**Krea2 reference latents:** concatenated reference tokens with a resolved
`reference_latents_method` refuse on the driver before dispatch, without an
abandoned sample or reset. Older builds refused mid-forward and needed a
Recycle. The worker guard remains a backstop. Requests without a resolved
method, and `single`/`dp2`, are unaffected. This preflight cannot inspect
`post_input`/`attn1_patch`: no shipped node transports those `model_options`
patches into the resident spec, so those rely on the mid-forward guard.

<a id="44-lens-uly2-1-step-fidelity-nrms-013-016-pre-issue-132-historical"></a>

## 44. Historical Lens uly2 one-step fidelity NRMS 0.13 to 0.16

**Symptom (older builds):** Lens `uly2`/`ring2` had one-step NRMS 0.13-0.16
against certified single-GPU/dp2 output, while `cfg2` and `dp2` matched exactly.
The adapter omitted divisibility-pad exclusion, exposing synthetic attention
keys when image or GPT-OSS text tokens did not divide the SP degree.

**What to do:** update dgx-monarch. Lens now uses
`padded_row_indices`/`drop_rows` in image-first joint order. The retest measured
NRMS 0.061 at 1328x1328 and 0.060 across two 1344x1344 prompts, below 0.10,
with cleanup confirmed. Current padding restrictions remain in
[#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid);
see CHANGELOG.md for the fix.

## 45. Anima `dp2` (and `single` at world 2 with an even batch) rendered a different image

**Symptom (older builds):** Anima `dp2`, or `single` at world 2 with an even
batch, produced unrelated images. `uly2` and `ring2` were unaffected.

**What to do:** update dgx-monarch. Until then, use `uly2` (the auto choice)
or `ring2`; avoid both affected presets. The hardware retest measured DP2
against certified Ulysses2 at NRMS 0.024, below 0.10, with cleanup confirmed.

**Why:** the generic DP slicer mistook Anima's unbatched T5 token IDs and
weights for per-image data, sometimes leaving a rank one token. Anima now
marks `t5xxl_ids` and `t5xxl_weights` in `Adapter.dp_cond_exempt_keys`, so every
rank receives the full arrays for `llm_adapter`, as in a single-GPU batch.

## 46. An older build refuses PixelDiT/PiD sequence parallelism

Older builds refuse `uly2` and `ring2` because tests measured NRMS 0.265 and
0.338 against a verified `dp2`/single reference. Update dgx-monarch. The fix keeps
projections and MLPs sequence-sharded but runs attention in gathered stock
stream and head order. World-2 PixelDiT 1024 matched `dp2` exactly on both
`uly2` and `ring2` across 20 steps. PiD 1024-to-4096 matched `dp2` exactly on
`uly2` across its full 4-step schedule with the optional pixel-stage LQ
feature active. `auto` selects `uly2` for this family.

The `sp_unvalidated:pixeldit_comfy` accuracy waiver is retired. Ledger rows
written in its protocol-v9 vocabulary stay readable, but the adapter raises no
PixelDiT waiver card and ignores `DGXM_WAIVE_KNOWN_WRONG=sp_unvalidated`. World
sizes other than two remain untested on hardware.

## 47. Wan SCAIL/SCAIL-2 render refused: activation footprint preflight

**Symptom:** a `WanSCAILToVideo` (or SCAIL-2) render is refused before any
GPU work starts, with a `StockLoadCapacityError` naming "activation footprint
preflight refuses...". A Wan Animate 2 render can meet the same refusal.

**Cause:** SCAIL/SCAIL-2 concatenate reference-image tokens onto the video
latent's temporal axis and append pose tokens onto the flattened token axis
before the DiT blocks. Sequence parallelism shards only the token axis, while
the 14B weight stays fully resident on every rank whatever the topology. On
UMA, `stock_load_preflight` (see [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)) checks only the checkpoint file size at
load time and knows nothing of resolution, frame count or these extra token
streams, so on its own it would let an oversized SCAIL config reach an untyped
kernel or NVRM out-of-memory kill. `activation_footprint_preflight`
(`mesh_safety.py`) estimates weight plus activation bytes from the
checkpoint's file size and the video, reference and pose latent shapes, and
raises the same `StockLoadCapacityError` as [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) when the estimate exceeds
`MemAvailable`. It runs on the driver before any worker RPC, only on UMA, and
only for the families in `mesh_safety.REF_POSE_TOKEN_FAMILIES` (`wan_scail`
and `wan_animate2`); no other family meets it. `minimax_h3` uses the same
guard id from its own module, because a packed audio+video+text sequence needs
its own row algebra and its own calibrated constants: see [#56](#56-minimax-h3-render-refused-activation-footprint-preflight).

**Reading the error:** the message splits the estimate into weights and
activations, gives the token count of each stream (video+ref and pose), and
names a `tokens/rank <= N` bound. The driver prices the render as one rank, so
the bound applies to the render's whole token count.

**Fix:** shrink resolution, frame count or reference-image count until the
token count is under that bound, or free memory (`sync && echo 3 | sudo tee
/proc/sys/vm/drop_caches`, stop co-resident LLM servers, and `dgxm reap`). If
this preflight refuses a config you know fits, set
`DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver process before launch to
bypass it. The same variable also turns off the H3 activation preflight
([#56](#56-minimax-h3-render-refused-activation-footprint-preflight)) and the
LTX one, together with the render-activation charge each adds to the
loader-site capacity check, and the render memory preflight
([#79](#79-a-cross-spark-render-refuses-because-the-ranks-disagree-about-weight-residency)).
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)'s preflight, the driver
footprint preflight
([#52](#52-driver-side-footprint-preflight-refused-a-render)) and every gate
stay on. This estimator's activation-bytes-per-token constant is an
uncalibrated placeholder awaiting hardware measurement, so a refusal at the
margin may be conservative, not a hard ceiling.

**Warm re-renders:** a checkpoint that already rendered in this driver
session, or that a pipeline sampler has submitted, is credited: its weight
bytes already sit inside the lower `MemAvailable` reading, the message says
"weights resident, credited", and only the activation term is estimated.
Every successful Recycle drops that credit, and so does a render that fails
while its result is collected and evicts the fleet. Other evictions do not,
for example a fleet that dies during submit or in a Fleet wave. If the
workers lose the model another way, for example when a different slot
replaced it, the credit is stale and the estimate omits the weight term. That
case can still end in an allocator OOM, as it would with no preflight, but it
never blocks a render that fits.

**Minimal-footprint gate runbook (first SCAIL attempt on new hardware):**
build a `WanSCAILToVideo` graph with a small, 32-divisible `width`/`height`
(for example 384x384), a small `length` (for example 17, giving 5 latent
frames), `batch_size=1`, exactly one `reference_image`, and no `pose_video` or
`pose_video_mask`. On the Init node set `auto_gate=off` for this first
plain-render probe, which skips the multi-leg check, with world=2 and
topology=auto (resolves uly2). There are three possible outcomes. The render
completes: capacity at this minimal config is proven, not the family in
general; re-run with `auto_gate=first_use` next to attempt the real check.
The typed `StockLoadCapacityError` above fires before any GPU work: confirm in
`journalctl`/`dmesg` that no `nv_err_no_memory` or kernel OOM-killer event
occurred; this is the catchable outcome the preflight is for. Or a kernel or
NVRM OOM occurs although the preflight passed: report it as a regression,
because the estimator under-refused.

**No panel card.** The activation estimate is class C like the residency
capacity check in [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), but the
consent subsystem
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)) raises a
card only where a rescue exists, and no residency change makes an over-budget
activation footprint fit: residency changes when weight bytes appear, never
how many activation bytes a shape needs. The escape stays the environment
variable above. The MiniMax H3 refusal in
[#56](#56-minimax-h3-render-refused-activation-footprint-preflight) carries
the same guard id, `activation_footprint_preflight`, also with no card, for
the same reason.

## 48. MiniMax H3 refuses `cfg2`, `dp2`, and `single` at world 2

**Symptom:** an H3 render under the `cfg2` preset is refused with a
`PackedCfgParallelError`. `dp2`, an `auto` fold that derives dp, and the
`single` preset at world size 2 (which derives dp2 without saying so) are
refused with a `PackedDataParallelError`. `auto` never selects any of them for
this family.

**Cause:** stock ComfyUI runs this DiT at batch 1 and raises
`ValueError("MiniMax H3 supports batch size 1")` on anything larger. It also
rebuilds the packed layout object per conditioning entry, and comfy decides
whether cond and uncond may share one batched call by comparing those payloads
by value, which two freshly built layout objects never satisfy. So every H3
step is two separate batch-1 model calls at any CFG value, and cfg-parallel has
no batched call to split: the split would fail on the leading batch dimension
even if the dispatch reached the worker. The driver refuses both on the latent
itself: H3's latent is a `NestedTensor` pair (video plus audio), and the
generic packed-latent guard will not split it across ranks for either cfg or
data parallelism.

**Reading the error:** both refusals fire on the driver before the first-use
gate and before any dispatch; each took 0.0 s in testing (docs/VALIDATION.md,
MiniMax H3 anchored guides). If a cfg or dp topology ever gets past that guard,
a backstop in the submit path raises an `UnsupportedModelError` naming the
family.

**Fix:** use `uly2`, which is what `auto` picks for this family at every
resolution, or `ring2` on an unpadded packed total ([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). For a true
single-GPU reference use `mode=local` with `gpus_per_host=1`; the `single`
preset at world 2 is not that reference. None of this constrains
classifier-free guidance itself: CFG above 1.0 renders normally on every
supported topology, at two sequential model calls per step instead of one
batched call.

## 49. MiniMax H3 `ring2` refused on a padded packed sequence

**Symptom:** `ring2`, or any hybrid preset carrying ring degree, raises a
divisibility-padding refusal (the H3-worded one, same cause as [#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid)) on a graph
that runs fine on `uly2`. Editing one word of the prompt can flip the same
graph between refusing and running.

**Cause:** H3 denoises one packed sequence, and its length is

```
S = text_rows + cond_rows + ref_rows + 2 * audio_t + latent_t * (height // 32) * (width // 32)
```

where `text_rows` is the raw Qwen presentation length of the prompt. The audio
term is always even, but the text term is user-typed, so the parity of `S` can
change with every prompt edit. When `S` does not divide the sequence-parallel
degree the shard pads, and H3's attention is maskless and fully bidirectional,
so a zero pad row is a real key for every real token: the first block's
modulation multiplies the row by its scale, then adds a learned shift, and the
row stops being zero. The adapter therefore passes the pad rows as explicit
drop coordinates, and that exclusion exists on the pure-Ulysses path only, so
ring refuses instead of attending to invented rows.

Observed in hardware testing: on the first use of a model/context
combination, this refusal fires inside the auto-gate check (the check's
own proof render hits it before your render would), the check aborts, and
the fleet then refuses further queues until a Recycle (the sidebar's **Reset
attached mesh**, or a Worker service restart when no driver is up). One Recycle was enough in the test. At world 2
`auto` never routes H3 onto ring, so only an explicit ring preset on an odd
total reaches it there. At an odd world `auto` falls back to pure ring, and at
an even world above 2 it folds the spare data-parallel ranks into ring; both
refuse a padded sequence the same way.

**Fix:** use `uly2`. If you need ring for a capacity or comparison reason,
make the packed total divide the degree: pick a canvas where `(width // 32) *
(height // 32)` is even (1344x768 gives 42 * 24 = 1008), which makes the
video, audio and keyframe terms all even, and then only the prompt's token
count and any reference-image blocks decide the parity. The negative prompt
has its own token count whenever the sampler evaluates the unconditional pass:
comfy skips it at cfg 1.0 only for samplers that leave the cfg-1 optimization
on, and every `_cfg_pp` sampler turns it off, so it runs even at cfg 1.0 (an
empty negative is still a few tokens). H3 builds a separate packed layout per
conditioning, so cond and uncond each pad or do not pad on their own. For a
ring leg, either pin a non-cfg_pp sampler (both committed H3 example graphs
pin `res_multistep`) or land both prompts on an even total. There is no
supported way to read the packed total from the UI, so treat ring on H3 as an
expert path and `uly2` as the default. Do not force ring past the refusal: it
keeps a padded ring render from producing wrong math that nothing flags.

The refusal opens with `[dgxm:K guard=ring_pad:minimax_h3 waivable=1]`
(docs/DESIGN.md section 5.9): output known to be incorrect with an expert waiver behind
it, off by default. The generic backstop in the adapter base carries the
unscoped id `ring_pad`; the two stay separate so the permanent row names which
family's guard was waived. The card states the measured error it asks you
to accept: Krea2 artifacts caused by attention to padding rows. H3 has no
padded-ring measurement of its own and needs none: the mechanism that
corrupted Krea2 is present here in full, so the guard follows from that
mechanism rather than an H3 threshold. A granted waiver stamps the output
rendered-under-waiver and writes a permanent ledger row
([#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means)). When
the refusal fires during the check, recover in the order above: allow it, then
Recycle, then queue again.

## 50. MiniMax H3 capacity: planning a clip length before you start one

**Symptom:** you want to know how long an H3 clip fits before you start one.
The typed refusal and everything it charges are in [#56](#56-minimax-h3-render-refused-activation-footprint-preflight); the driver-side half is
in [#52](#52-driver-side-footprint-preflight-refused-a-render).

**Capacity estimate.** The row count in
[#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence) grows linearly
in frames and in canvas area: at 1344x768 the video term alone is 1008 rows per
latent frame, and a 124-frame clip carries 37 latent frames. Start from the
smallest measured shape, 512x320 at 5 frames and 1 step (the small-shape H3
retest in docs/VALIDATION.md), then grow canvas and length one step at a time.
The loader-site capacity check prices the text encoder and the two VAEs from
the graph a queued render passes; a headless call passes no graph, so budget
them by hand there
([#85](#85-a-script-or-harness-fills-the-box-although-the-loader-wall-exists)).

## 51. A worker actor outlives its client and holds tens of GiB

**Symptom:** an idle host keeps losing free memory while a Python actor holds
tens of GiB. `dgxm status` may be green and doctor may attribute pressure to
page cache. Older deployments and hosts outside cluster.toml may retain the
actor even after `dgxm restart`.

**What to do:** inspect `dgxm doctor`'s `orphaned actor procs` row, then run
`dgxm reap --dry-run` to review the sweep. `dgxm reap` sweeps every configured
host and the local host, even without cluster.toml. `dgxm reap --grace 300`
requires that much orphan dwell time, or process age if dwell was not recorded.
The sweep is safe during rendering: it leaves live owners, children of live
Worker services, and uncertain identities alone.

Only proven strays are signalled, with `(pid, start time)` rechecked in procfs
immediately before SIGTERM, then SIGKILL if needed. Attribution accounts for
reparenting to init/subreapers, dead owners, and parent services born after the
candidate. The driver embeds its identity in spawned-worker environments;
processes cannot expose later environment writes through `/proc/<pid>/environ`.
An inspection error, remote timeout, incomplete sweep, or survivor makes the
host UNKNOWN/failed and the command nonzero. No signal or success line is
proof of cleanup unless the sweep settled; repair the host/transport and retry.

**Automatic cleanup:** the driver renews actor leases. After 300 seconds
without renewal by default, the actor logs `dgxm-reap:` and exits. Inspect
`journalctl --user -u dgxm-worker` for that line.
`dgxm down` sweeps after stopping the service, including its children;
`dgxm up` sweeps only orphans. `dgxm restart` uses both. A host is not reported
successful before its sweep finishes. The Worker service records its actors in
`~/.local/state/dgx-monarch/actor-procs.json`. A writer killed mid-write leaves
an `actor-procs.json.<pid>.<starttime>.<thread id>.tmp` file beside it, and the
next ledger write removes each such file once no process holds that pid with
that start time. Earlier releases wrote `actor-procs.json.<pid>.<thread id>.tmp`.
The sweep removes such a file once no process holds its pid, so a file whose pid
a later process took stays until that process exits. `nostat` in place of the
start time means the writer could not read its own `/proc` entry, and no sweep
removes that file.

Doctor's orphan row covers only its local host. It names PID, size, dwell, and
remedy for marked actors holding at least 4 GiB with no live owner able to stop
them. It does not identify a dead-client actor while its Worker service lives:
an established service socket says nothing about the driver. The actor lease
and the next `down` sweep cover that case. Liveness recovery after a timed-out
stop is covered in
[#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn).

**Lease settings, on the driver:** `DGXM_REAP_GRACE_S` accepts 300-3600
seconds. Other values are replaced by 300, not clamped, with a warning. The
lease minus a 30 s renewal interval must exceed the teardown token's 240 s
lifetime. Use `dgxm reap --dry-run` to inspect cleanup rather than shortening
the grace.

`DGXM_DISABLE_ACTOR_REAPER=1` sends a zero lease and disables both actor
triggers; only the sweep and doctor remain. Both settings arrive with the
first renewal cast. Until then, parent-loss detection remains armed; it covers
the gap before Monarch installs its parent-death SIGKILL and stands down when
a live lease exists.

**Local or unmarked actors:** local (`this_host`) actors are ComfyUI children
and have no configured hosts for `up`/`down` to sweep. Use `dgxm reap` or
restart ComfyUI. Recycle requires a live mesh and working in-process stop; it
helps only while that fleet is attached and rendering has finished.

An older actor without ownership markers is reported but not signalled. On its
host, review `python -m dgx_monarch.cli.actor_reaper --sweep --no-marker`
with `--dry-run` first. That option accepts the launcher marker alone; `dgxm`
never passes it. Alternatively, before signalling an exact PID yourself,
confirm all three: `/proc/<pid>/cmdline` contains
`monarch._src.actor.bootstrap_main`, its UID is yours, and it has no
ESTABLISHED socket. Send SIGTERM first, then SIGKILL only if it does not exit.
Never use `pkill -f`: it can match the issuing shell or unrelated Monarch
applications. A process in uninterruptible kernel sleep that survives SIGKILL
requires a reboot.

**Why this happens:** client death, runtime teardown panic, or a lost stop
outcome can defeat cleanup inside the actor. Older service-only scans missed
these children. Memory returns only when the actor exits; an uncertain stop
must not be retried merely because the memory remains allocated.

## 52. Driver-side footprint preflight refused a render

**Symptom:** `DriverFootprintCapacityError` (a `StockLoadCapacityError`)
refuses before worker RPCs. Render-time messages start
`driver-side footprint preflight:`; loader messages start
`driver-side footprint preflight at the loader node:`. The latter carries
`[dgxm:K]` if a Gate FAIL blocks rescue; ordinary capacity refusals are class C.

**What to do:** follow the reported shortfall and fitting options. For H3,
try a smaller text encoder, a pruned/quantized DiT, fewer or smaller references,
a smaller canvas/clip, or more free host memory. Pruned int8/fp8 reference-video
files are 19.5 GiB versus bf16's 61.7 GiB. For LTX, start with a smaller canvas
or clip and a world-2 `uly*` or `ring*` preset, then the same artifact and memory
options. To free memory: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`,
stop co-resident LLM servers, or use `dgxm reap`.

H3 truncates reference video to the generation length before encoding. A
48-frame reference in a 5-frame generation costs five encoded frames;
shortening only the submitted reference may change nothing. The error reports
encoded geometry, including block count, frames, and pixels.

**Why it refuses:** on UMA, the driver and a colocated worker share memory.
The driver also holds text/vision encoders, VAEs, and reference encodes. A
worker's checkpoint-only estimate cannot protect that combined footprint.
The H3 reference-video failure exhausted memory during driver work after an
admitted DiT load; see [VALIDATION.md](VALIDATION.md).

**Read the two loader windows:**

| Window | Charge and remedy |
|---|---|
| `load` | Stock placement peaks near 2.1x checkpoint size, before the driver stack exists. Slab avoids the host copy and may offer rescue. |
| `resident` | Weights plus the graph's future text encoder, VAEs, reference/guide encodes, and declared render activations. If this does not fit, use smaller artifacts or inputs; changing residency cannot remove the stack. |

Rescue is offered only if slab fits both windows
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)). With
`slab_weights` auto/unset, `capacity_fit.SLAB_VOUCHED_FAMILIES` determines
whether the family has slab identity evidence: verified families omit the host
copy; other families include it. H3 activations are described in
[#56](#56-minimax-h3-render-refused-activation-footprint-preflight).

The loader checks before liveness, topology, or rank effects. It prices the
pending stack from the prompt graph. An unreadable graph leaves that stack
unpriced and warns once per family; other estimator failures also admit the
load. A cleared charge is cached per checkpoint and world for the session.
Direct headless calls need their own budget
([#85](#85-a-script-or-harness-fills-the-box-although-the-loader-wall-exists)).

**Render-time estimate:** both `run_render` and pipelined depth>1 submissions
check before dispatch or mesh effects. Only pending rank-0 weights on the
driver host are charged. `auto` defers loading to this point; explicit presets
load eagerly, so their weights are already in `MemAvailable`. The reserve is
5 GiB or a larger configured `uma_reserve_gb`.

* Pending weights cost the checkpoint's file size, a conservative upper bound.
* A host copy adds 1.1x only when `slab_weights` is explicitly off. Auto,
  unset, and comfy-managed residency omit it because placement is unknown.
* A persisted FAIL can write `slab_weights=False` into graph worker args. This
  estimate runs before the ledger read, so that extra charge affects later
  pending loads, not the first render that reads the FAIL.
* Existing text encoders, VAEs, and encoded references are reported, not added
  again: their memory is already absent from `MemAvailable`.

Explicit eager loading can therefore pass before reference work exhausts the
host. The loader's graph estimate covers that gap; the render check cannot
protect work that ran before the sampler. Predictions that `auto` would have
refused the historical H3 failure are derived from its memory curve, not a
recorded rerun.

**FSDP charges differ.** The loader charges `1.05 * file/world + 256 MiB` and
omits the host copy only when all five conditions hold: an explicit FSDP preset
(not `auto`), world >= 2, one rank per host with rank 0 on the driver, a
bf16/fp16 header, and comfy-managed residency off. Otherwise it charges the
whole file. FP8/int8 direct wrapping still moves full blocks. `slab_weights`
does not affect a sharded rank.

The worker uses `(1/world + 0.08) * file + 5 GiB` for known-world bf16/fp16;
fp8, int8, unreadable precision, or unknown world uses `1.20 * file + 5 GiB`.
Hardware confirmation of the floor and direct-wrap bound is **NOT RUN**.
FP8 direct-wrap and Krea2/Chroma/Ideogram4 INT8 tests proved stored-byte
execution, not those bounds (VALIDATION.md, memory calibration).

The shard estimate assumes the current bounce-buffer build; older workers
can use about twice the shard. It excludes ComfyUI's sample-time reservation,
roughly another shard, so an admitted load can still meet the partial-load
refusal in [#79](#79-a-cross-spark-render-refuses-because-the-ranks-disagree-about-weight-residency).

**Scope and limits:** this is an admission check, not a peak-memory model.
It covers only UMA `minimax_h3` and `ltx`
(`DRIVER_STACK_FAMILIES` in `driver_footprint.py`). It reads safetensors
headers, sizes, `/proc/meminfo`, cached host resolution, and a local-address
route probe; it issues no RPC. Except for `localhost`, `127.0.0.1`, and `::1`,
cluster host resolution uses `getfqdn`, plus `getaddrinfo` for a hostname.
Those calls have no timeout and cache once per configured name per driver.

Only a complete over-budget estimate refuses. Unresolved files, unreadable
headers, non-Linux/discrete hardware, unsupported families, loader dtype casts,
previously cleared loads, or estimator exceptions proceed without this check.
A cast changes resident size, so charging file size could falsely refuse it.
Cleared-load caches avoid charging weights again after a sampling failure;
stale entries can undercount, but cannot create a false refusal.

The error lists projected/usable totals, reserve, artifact, charges and their
provenance, omitted terms, fitting artifact size, and shortfall. If nothing
fits, it says so. Budget figures lead to survive the Gate's 200-character
truncation. Calibration separates measured failed and successful H3 loads;
near-boundary refusals may be conservative. See VALIDATION.md for the
measurements rather than treating these figures as peak-memory predictions.

**Unstable artifact or ledger:** a checkpoint being copied or rewritten has
no stable byte identity. Wait for it to finish, then queue again. This is a
nonwaivable class C refusal: even standing consent cannot authorize unidentified
bytes. Rescue first reads durable ledger denials without a capability context;
only `unknown` is retried within this load's context. Click resolution uses the
same rule. A torn line still refuses because it could contain a relevant
verdict; rerun Identity Gate. FAIL and other definite denials are not bypassed.

**Disabling a known false-positive check:** set
`DGXM_DISABLE_DRIVER_PREFLIGHT=1` before starting the driver. It disables both
loader and render driver-footprint checks and logs one WARNING per process.
It is separate from `DGXM_DISABLE_ACTIVATION_PREFLIGHT`
([#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight));
other capacity checks, including
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), and all gates remain active.

## 53. A render failed and the panel is asking for one click

**Symptom:** Queue fails before a large load. The error names a panel button
and headless environment variable; the sidebar shows a badge, toast, and card
with the model, measurements, risk, and available protection. ComfyUI cannot
pause execution for consent, so authorization happens after the refusal.

**What to do:** read the card, then click its primary button if you accept.
The driver must write the permanent ledger row before saving consent. A failed
write grants nothing. Once the card says "Allowed", use "Queue this graph
again" or queue the intended workflow yourself: the button queues the tab
currently open, which may differ from the failed graph. A capacity-rescued
load uses slab with a byte-verify certificate. Dismiss records no consent;
the card returns if the next queue reaches the same refusal.

Capacity cards offer slab when stock cannot fit. Accuracy cards concern
`ring_pad`, `sol_attn`, or `shard_quant_scale`; read
[#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means) before
accepting. Tags distinguish capacity (`C`), unproven correctness (`U`), known
incorrect output (`K`), and physical constraints with no bypass (`P`);
see DESIGN.md section 5.9.

**No button means no panel remedy.** Class C messages can link here without
offering consent. Follow their "What would fit" clause; all cases below are
`waivable=0`, and `DGXM_ALLOW_SLAB_RESCUE=1` grants nothing:

| Guard/path | Why no rescue is offered |
|---|---|
| `stock_load_preflight` | Slab is off or quarantined; the file is not safetensors; FSDP/`compile_dit` is active; `lora_low_rss` is off; or slab also cannot fit. The message names the blocker. |
| `slab_load_preflight` | Slab itself does not fit. The message includes the larger stock estimate, so disabling slab cannot help. |
| Comfy-managed `stock_load_preflight` | That residency cannot fit and has forced slab off. Turn off `comfy_managed` and follow its reset requirements in [#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses) before using slab. |

The slab and comfy-managed checks retain a 5 GiB floor to prevent partial
loading during sampling. A larger `uma_reserve_gb` wins where applicable,
capped by the arena stock loading would retain. Consent cannot add memory.

**Headless consent:** set the error's variable on the driver before ComfyUI
starts. Capacity rescue uses `DGXM_ALLOW_SLAB_RESCUE=1`. It writes the same
ledger authorization but no persistent memo; authority lasts for the process.

**Scope and revocation:** panel memos bind checkpoint device, inode, size,
mtime, ctime, consent kind, and a narrow context. Replacing or rewriting the
file invalidates the memo and greys its panel entry. Revoke in "Consents and
waivers" takes effect on the next render. Memo schema 2 also retains the legacy
artifact-digest alias. Other schemas are ignored, dropping stored consents and
Auto-allow capacity rescue with a log message; grant them again if intended.

**Automatic capacity rescue:** the panel toggle skips individual capacity
prompts. Changing it requires a standing ledger row; a failed write leaves it
unchanged. Each rescued combination writes a `WAIVER` row on first use and
after each fleet recycle, not on every load. Unlike accepting a new card,
failure to write this per-combination row does not stop an already-authorized
load; the driver logs the missing audit record.

`DGXM_AUTO_RESCUE=1` grants the same process-wide standing authority without a
card or memo. It covers every auto-eligible kind and every checkpoint the
process loads. Slab is currently the only eligible kind, so
`DGXM_ALLOW_SLAB_RESCUE=1` has the same scope. The panel shows an environment-
enabled toggle checked and disabled; unset the variable and restart the driver
to change it. Automatic rescue never grants an accuracy waiver.

**Quarantine always takes precedence.** FAIL applies before residency
selection, and an explicit stock request remains stock regardless of consent.
An effective FAIL consumes its pending capacity card and revokes the class C
memo it disproved. Accuracy waivers survive because they affect math guards,
not residency settings.

Both list and accept recheck the ledger. List hides cards/consents for FAIL,
RETESTING, unknown, or corrupt state without altering stored state; accept
returns a conflict. Without a governing FAIL, INCONCLUSIVE and RETESTING deny
acceptance without consuming the card or revoking the memo. Later INCONCLUSIVE
or RETESTING cannot override FAIL. Only a newer current-source PASS in the
same canonical capability context clears that FAIL and permits new consent.
Consent never turns INCONCLUSIVE into PASS
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

**Hardware scope:** the 61.7 GiB H3 bf16 reference-to-video and first-last-to-
video files cannot stock-load on the measured host, so they require slab
consent. Earlier stock-residency fidelity results do not validate their slab
runs. Pruned and unpruned int8/fp8 artifacts still load stock.

**If the card is missing:** reload a stale browser tab and confirm the sidebar
is installed. Its poller alone fetches cards; the consent route uses driver-
local state and remains available even if the mesh is blocked. Headless
variables are the alternative when the panel is unreachable. If acceptance
reports "the waiver could not be recorded", repair ComfyUI output-directory
write access, then retry the click; no consent was granted.

**Error metadata:** `[dgxm:measured {...}]` contains ledger measurements and
`<<<DGXM-CONSENT-V1 ... DGXM-CONSENT-END>>>` contains the card descriptor.
These survive text-only worker exception transport. Follow the prose before
them; the panel reads the metadata.

## 54. Slab byte-verify certificate mismatch

**Symptom:** a slab load aborts with a typed refusal naming a tensor, a file
offset, a slab offset and the index of the first differing byte inside a
window. Nothing is adopted, no model becomes resident, and the message says
that nothing was quarantined and that this is not a model-correctness finding.

**Cause:** every slab load verifies the bytes it just placed against the
checkpoint they came from, and this one did not match. The check runs per
tensor, immediately after that tensor's read and before anything can cast,
bake or reabsorb it, with the file's inode identity pinned on both sides of the
read. So a mismatch means one of three things: the file changed under the
loader, the storage returned different bytes than it claimed, or the artifact
was staged badly (a truncated copy, an interrupted rsync, a bad download).

**Fix:** treat it as an artifact or storage fault, not a model problem.
Checksum the file against its source, check `dmesg` for IO errors on that
device, and re-stage the artifact. On a cluster, check every host's copy: each
host verifies its own local copy, and only one of them may be bad.

**Why there is no bypass.** This is class P: no consent, no waiver, no
environment variable. A byte mismatch means the slab does not hold the model
you asked for, so proceeding would publish output from incorrect weights. That
is why every slab load is verified, not only loads that skipped a gate.

**The other arm: no certificate at all.** A consented pre-gate slab load whose
worker returned no certificate row refuses the same way, class P, for a
different reason: nothing was measured wrong; the driver asked for a row shape
the worker does not write. That is version skew, so the fix is `dgxm restart`
on the Worker service and one more queue. On the way out the driver asks the
workers to unload all their models, not only that slot, so nothing uncertified
stays resident; if that call fails, the driver log says so and asks for a fleet
recycle before the next render. It is class P because this is missing load
evidence, not a measured image mismatch that could be accepted under an
accuracy waiver.

**What it does not do.** It writes no gate verdict, sets no quarantine lever,
and revokes no consent. A storage fault does not evaluate model correctness;
quarantining it would also block a later run from healthy bytes. The
certificate is evidence about bytes; the identity gate is evidence about
pixels; neither substitutes for the other, and a certificate is never a PASS
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

**Reading the ledger side.** A slab load that completes on a box where stock
provably could not fit records a `CAPACITY_CERTIFIED` row carrying the
certificate summary next to the INCONCLUSIVE verdict, never in place of it.
Those rows and the `WAIVER` rows live in a separate key namespace, so they can
never displace a real verdict. The two audit reads are
`grep '"verdict":"CAPACITY_CERTIFIED"'` and `grep '"verdict":"WAIVER"'` over
`dgxm_gate_ledger.jsonl` in the ComfyUI output directory.

## 55. Rendering under an accuracy waiver, and what the stamp means

**What you see:** a red "accuracy waiver" card names the model, known error,
and measurement. Three class-K guards offer waivers: `ring_pad`
([#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid),
[#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)), `sol_attn`
([#84](#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call)),
and `shard_quant_scale`
([#95](#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales)).
They refuse known incorrect math by default. Waivers are never automatic.

**What to do:** accept only if you intend to run with the stated accuracy loss,
for example a comparison or reproduction. Click "Render under waiver (output
stamped)", then queue again. The permanent ledger row must be written before
the memo; failure to record it grants nothing
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)). If the
refusal occurred during first-use proof, Recycle before requeueing
([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). Proof renders
never use waivers: the check may refuse again, record its result, and yield
without raising another card for the accepted waiver.

**Headless:** set `DGXM_WAIVE_KNOWN_WRONG` before starting the driver. Valid
values name guards (`ring_pad`, `sol_attn`, `shard_quant_scale`) or consent kinds
(`waive-known-wrong:ring-pad`, `waive-known-wrong:sol-attn`,
`waive-known-wrong:shard-quant`). Commas combine names, for example
`ring_pad,sol_attn`. A bare `1`, scoped ledger IDs (`ring_pad:minimax_h3`,
`sol_attn:minimax_h3`, `shard_quant_scale:chroma`), and retired PixelDiT names
(`sp_unvalidated`, `waive-known-wrong:pixeldit-sp`) are invalid. Unknown names
warn once per distinct set and list accepted spellings; valid names do not
warn merely because another guard has no use for them.

Environment grants write ledger rows but no memo. They apply to every render
in that driver process, across checkpoints, topology, and world size, and
survive checkpoint replacement. They have no revocable panel row: unset the
variable and restart the driver. Prefer the panel for narrowly scoped grants.

**Panel scope:** a memo covers the checkpoint, its options and LoRA names,
resolved topology, and world size, bound to file identity. A different degree
asks again. An explicit pad waiver also covers `auto` if it resolves to that
same topology: world 3 can resolve `ring3`; a world-4 batch-1 `uly2` row can
resolve `uly2+ring2`. The panel and ledger name the resolved topology.

The driver resolves consent at every dispatch and puts it on that request;
resident models retain no authorization. Revoke in "Consents and waivers"
writes a revocation row and affects the next render without reloading.
Checkpoint replacement/rewrite invalidates a panel memo too. A Gate FAIL does
not revoke a math waiver, but its residency quarantine still wins.

**What is recorded:** every waived render carries
`rendered-under-waiver: <kind> (<measured wrongness>)` in its mandatory class-K
ledger `stamp`, a permanent `WAIVER` row with `"action":"use"`, its returned
latent, and the driver log. The PNG is not stamped: the stock save node owns
that path. Retain the workflow embedded in the PNG and the matching ledger
for provenance.

Chaining that latent preserves prior entries under `"inherited": true`.
A later leg that triggers no guard writes no new use row and does not claim a
waiver of its own, even though its output descends from waived math. Any waiver
it independently uses gets a separate entry.

Pipeline/Fleet joins concatenate `batch_index` in render order, omitting it
only if every constituent omitted it. Waiver entries retain first-seen order,
deduplicated by `(run_id, guard)`; Fleet records every actual use before the
join. Inherited entries write no new use row. Before gating, mesh/session work,
or submission, inherited metadata must be an exact list/tuple of mappings with
nonempty string `run_id` and `guard`. Mixed or malformed `batch_index` and
malformed provenance refuse
([#71](#71-pipelinefleet-refuses-batch_index-or-a-chained-sampler-refuses-inherited-waiver-provenance)).
Older aggregates may contain only the first constituent's metadata; update
and rerun the batch rather than editing it by hand.

**Proof and memory limits:** a waiver grants no PASS, converts no INCONCLUSIVE,
changes no MODELS.md support row, and never overrides quarantine. Auto-allow
capacity rescue excludes class K. A first-use proof that encounters the waived
guard ends without comparison evidence and quarantines no residency setting
([#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)). With no PASS,
the following ordinary render uses stock residency and may need more memory
or time. If only slab fits, accept its separate capacity card too
([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc)).

FSDP cannot obtain its required clean-reload PASS through a waiver. In
particular, `uly2+fsdp` without LoRAs cannot proceed if a class-K guard such as
`sol_attn` refuses the proof. Use a non-FSDP topology or an input/kernel the
guard accepts. Ring padding never reaches that proof because ring+FSDP is
already refused at topology validation.

**`audit window is full`:** the driver allows at most 512 outstanding
waiver-audit joins and never evicts one that might return waived output. New
requests omit the waiver until work settles, so their guards may refuse.
Let outstanding renders settle or close, then queue again. The memo is still
valid; the limit protects mandatory audit records.

**Audit:** run `grep '"waiver_class":"K"' dgxm_gate_ledger.jsonl` in the
ComfyUI output directory. Rows identify grant/use/revoke, checkpoint, guard,
channel (`panel`/`env`), the card's exact sentence, and stamp. Waiver rows use
a separate audit namespace and cannot replace a trust verdict.

## 56. MiniMax H3 render refused: activation footprint preflight

**Symptom:** an H3 render is refused before any GPU work with a
`StockLoadCapacityError` tagged `[dgxm:C guard=activation_footprint_preflight
waivable=0]`, whose text after the tag starts `minimax_h3 activation
preflight:`. No kernel out-of-memory kill appears in `dmesg`: this refusal
fires in its place.

**Cause:** H3 runs one packed row sequence through its blocks, and every stream
in it costs memory, but the nested AV latent's `shape` reports only the video
tensor. The guard counts every stream, with the row formula [#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence) states:

```
S = text + keyframe + reference + 2*audio_t + latent_t*(height//32)*(width//32)
```

Each positive conditioning entry is its own model call that builds its own
packed sequence, and ComfyUI copies the same keyframe and reference lists onto
every entry of a scheduled prompt. So the guard takes the largest value of each
stream across the entries, not their sum: the peak is one call.

In two tests of a 1344x768x124 bf16 render on one rank, the kernel recorded
103.3 GiB of anonymous RSS for the ComfyUI process (61.7 GiB slab weights, 16.7
GiB resident text encoder, and roughly 25 GiB of activations and runtime) and
killed it. The capacity checks then in place had allowed both runs.

**Reading the error.** After the tag comes the budget: bytes needed per rank,
bytes usable, the MemAvailable reading, and the reserve subtracted from it,
named as the driver floor (5 GiB) or your larger `uma_reserve_gb`. Then the
charged terms: the DiT weights, with a note that says whether they were
charged, credited as resident, or stood down for a `weight_dtype` cast; and the
packed activations, with the total row count and its five-stream split. Then
rows per rank over the sequence-parallel degree, and the split between the part
that shards with the sequence and the replicated staging that no topology
divides. Then what would fit: the largest packed total this topology and budget
admit, beside the render's own total, and the free memory the render is short
by. The offered row bound is reachable: shrinking to exactly that number
passes, and one row more refuses. When the weight charge alone fills the budget
the message offers no row bound and says so. No canvas change fixes that one,
so take a smaller artifact or free memory.

**Fix,** in the order the message prints them:

1. Shrink the canvas or the clip length. Rows grow linearly in latent frames
   and in canvas area, and at 1344x768 every latent frame is 1008 rows.
2. Run it on world 2 with a `uly*` preset. Sequence parallel is the only degree
   that shards the packed row axis for this family, and `cfg` and `dp` are
   refused for it outright ([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)). The 1344x768x124 bf16 shape completed this
   way in 727.6 s.
3. Cut the number, resolution or length of reference blocks and keyframes.
   Every reference row stays in the packed sequence for every sampling step.
4. Use a pruned or quantized DiT artifact. The pruned int8 and fp8
   reference-to-video files are 19.5 GiB each against bf16's 61.7.
5. Free host memory: drop caches, stop co-resident LLM servers, `dgxm reap`.

**Calibration and limits:** the estimate uses 1,032,192 bytes per sharded
row and 73,728 bytes per replicated row. Constants and derivation are in
`src/dgx_monarch/h3_calibration.py` (`H3_SHARDED_ROW_BYTES` and
`H3_REPLICATED_ROW_BYTES`); measured cases are in VALIDATION.md. Calibration
rejects the killed single-rank 1344x768x124 bf16 case while admitting its
world-2 Ulysses and single-rank int8-convrot controls. The current 5 GiB floor
preserves those outcomes. The observed kill slope is only a lower bound and
would admit the killed shape if used directly.

For 1344x768x124 bf16 at world 2 under `auto`, the pending estimate is
82.5 GiB, requiring about 87.5 GiB MemAvailable with the default floor, or
more with a larger `uma_reserve_gb`. The 92.2 GiB pending-memory example is
derived from idle memory minus encoder/VAEs, not directly measured. An explicit
`uly2` preset uses the already-resident calculation; select it or free memory
if the pending estimate refuses. A four-step pair run completed in 245 s with
61.8 GiB MemAvailable during dispatch-time loading. These examples do not
establish a universal peak-memory limit.

**Per-rank division.** The divisor is the mesh world, and for this family that
is exact: the sharded runtime requires world equal to dp x ulysses x ring x cfg,
and H3 refuses `cfg > 1` and `dp > 1` before dispatch ([#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2)), so every H3 render
that dispatches has ulysses x ring equal to world. Weights are never divided,
because Ulysses replicates the checkpoint on every rank, and the replicated
staging is charged for the full packed total on every rank at any degree.

**Warm re-renders and residency.** The weight term is credited, not charged,
when an explicit topology preset means the loader node already performed the
load, or when one of three session memos names this checkpoint: the last
completed render, the last pipelined submission, and the last pending load the
driver footprint check cleared. The memos are sampled before that check runs,
or a render would credit itself for its own charge. A `weight_dtype` cast also
stands the weight term down, because after a cast the file size does not price
the resident weights; the activations are still charged. A successful recycle,
and an eviction that retires the fleet, clear all three memos. A memo that
still goes stale can only drop the weight term: it can admit a render that
should refuse, never refuse one that fits.

**The loader-node half.** The loader node's settled window also charges the
render the graph declares ([#52](#52-driver-side-footprint-preflight-refused-a-render)), so a 61.7 GiB eager load is refused before it
happens when the render behind it cannot then run. The graph shows only video
and audio rows: text, keyframe and reference rows do not exist until their
nodes run. That under-count errs toward admitting, and the render site catches
what the loader site misses one node later.

**Escape hatch.** `DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver. It is the
same variable [#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight) documents, because a kill switch belongs to its guard and this
site raises under the same guard id: a separate variable would leave an
operator who set the documented one still meeting an activation preflight. The
driver logs a WARNING once per process when it is set; grep for that line after
an incident.

| site | `DGXM_DISABLE_ACTIVATION_PREFLIGHT` | `DGXM_DISABLE_DRIVER_PREFLIGHT` |
|---|---|---|
| render-site H3 activation guard | stands down, WARNING logged once | no effect |
| loader-site activation fold | term drops to 0 | term drops to 0 |
| render-site driver footprint ([#52](#52-driver-side-footprint-preflight-refused-a-render)) | no effect | stands down |
| SCAIL activation preflight ([#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight)) | stands down | no effect |

The loader row is the one asymmetry. There the activation bytes are a term
inside the driver footprint estimate, so that site's own switch must remove
them, and this guard's switch must too, or an operator who turned the guard off
would still meet its charge one node earlier.

**No panel card.** No consent can change this verdict: residency decides when
weight bytes appear, not how many activation bytes a shape needs.

**One case is not modelled:** a `_cfg_pp` sampler evaluates two independent
packed totals in sequence ([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)). The peak is still one packed total, so the
estimate stands, but the two calls can carry different row counts and only the
positive one is measured.

## 57. Console repeats "Future.get() called from within an active event loop"

**Symptom:** the ComfyUI console carries `[WARNING] [actor=<root>] Future.get()
called from within an active event loop` through cluster bring-up, topology
selection, the identity gate and the render itself, roughly ten times per
render cycle. Renders complete, latents match across ranks, the gate passes and
`dgxm doctor` is clean. The line does not appear on torchmonarch 0.5.0.

**Cause:** torchmonarch 0.6.0 forwards that WARNING to its tracing chain on
every `Future.get` made while an asyncio loop is running on the calling thread,
and ComfyUI executes nodes on such a loop. The dgx-monarch releases that print
it make that blocking call on the loop at four driver-side sites: the attach
initialization wait, the partial bring-up rollback stop, the setup rollback
group teardown, and the one dispatch function that every load, sigma
computation, gate leg and sample collection passes through.

**Fix:** update dgx-monarch on every host, and restart the ComfyUI driver as
well as the Worker services so it imports the updated source. Current
dgx-monarch runs all four calls through the bounded scratch-thread helper it
also uses for proc stops and RDMA latent reads. A complete render cycle should
then show the line zero times.

**If it persists after an update.** The message is noise, never a failure:
never infer a distributed-math or transport problem from it. All four sites
are in the driver, so confirm the driver restarted, not only the Worker
services: the driver holds the imported module. Do not silence the message with
a warnings filter or a logging level change. It is a true statement about the
calling thread, and hiding it would also hide a new on-loop caller.

## 58. `worker status poll timed out while a render is in flight`

**What it is.** An INFO note, not a fault. The DGX Monarch sidebar polls
`GET /dgxm/telemetry` every 2.5 seconds, and the worker status call inside that
poll has a 10-second bound. A worker answers `status` on the same actor that
runs your render, so while both ranks are saturated the ping can expire. The
panel keeps its last good numbers and the next poll recovers. The render is
unaffected: `status` is not a session-scoped endpoint, so a timed-out ping
cannot dirty the setup generation, and a plain timeout is not a supervision
failure, so it cannot evict the fleet.

**What to do.** No action is needed. The note appears at most once per render,
and only when the panel has a cached snapshot.

**Warnings that need attention.** Three cases log `worker status refresh failed:
...` at WARNING instead: the same timeout with the fleet idle, any non-timeout
failure, and a timeout with no cached snapshot to fall back on. The last is
what a fleet looks like before its first successful poll, and there the panel
shows no workers at all. Check that the Worker services are up (`dgxm status`),
then run `dgxm doctor`.

One WARNING case is expected: a long model load. The note keys on the render
progress tracker, which opens at dispatch, so a ping lost while the workers
still read a checkpoint from disk logs the WARNING for the same reason a busy
render logs the note. Let the load finish before you investigate.

## 59. `board hotspots` says the box has no temperature sensors, or names a hot one

**Symptom A.** `dgxm doctor` prints

```
[ ok ] board hotspots: no temperature sensors on this box: nothing readable
under /sys/class/hwmon and `sensors` added none.
```

This row is information, not a fault: the box publishes no hwmon temperature,
so the thermal tripwire is inert here and cannot warn you. It is `ok` rather
than `WARN` because no action clears it and nothing points to a problem. A
stock Spark publishes its board sensors through sysfs with no package
installed. So if you see this row on a Spark, `/sys` is not readable here:
check that `/sys` is mounted and that the process can read it (a container
started with a restricted `/sys` is the usual cause).

**Symptom B.** `dgxm doctor` prints

```
[WARN] board hotspots: 1 of 16 sensors at or above 90 C: hwmon0/acpitz/temp6
91.4 C. The Spark fan curve tracks CPU load and can miss a board sensor
```

This warning requires attention. The Spark fan curve ramps on CPU load, and at least
one board sensor sits outside it: the community-identified `temp6` can reach
90 C+ with the fans idle. Check case clearance and airflow first, then whether
anything sits on or against the unit. A hot board sensor is one plausible cause
of a silent hard power-off (the Spark failure catalog below: PD-stuck state and
its hard power-off signature), so do not leave it running hot while you
investigate.

**Symptom C.** The row ends with a count it could not read:

```
[ ok ] board hotspots: 15 sensors via sysfs hwmon (1 unreadable), hottest ...
```

One `temp*_input` errored or held a non-number, or a whole chip directory would
not list. The count makes a sensor that goes dark visible instead of only
shrinking the total. It is `ok`, not `WARN`, because the usual causes never
clear: an mlx5 module bay with no transceiver in it errors on every read, and
so does the `mt7925_phy0` wifi sensor on a Spark with the radio down. Find the
culprit by reading each file yourself and watching which one fails:

```bash
for f in /sys/class/hwmon/hwmon*/temp*_input; do echo "$f $(cat $f)"; done
```

Act on it only if the sensor that went dark is one you rely on. A previously
working sensor that becomes unreadable warrants investigation.

**Reading it yourself.** Each name in the row maps to a sysfs file:
`hwmon0/acpitz/temp6` is `/sys/class/hwmon/hwmon0/temp6_input`, in
millidegrees:

```bash
cat /sys/class/hwmon/hwmon0/temp6_input      # 91400 -> 91.4 C
```

A sensor with a `temp*_label` file shows its label instead of the file stem, so
`hwmon1/nvme/Composite` is whichever `tempN_input` in `hwmon1` has
`tempN_label` = `Composite`.

**No package is needed.** The check reads sysfs directly. `lm-sensors` is
optional: doctor runs `sensors -u`, its machine format, only when sysfs
publishes no temperature at all.

## 60. Which teardown absorption still fires: the `dgxm-absorb-fire` soak

**What you see:** INFO lines tagged `dgxm-absorb-fire` record teardown
classification. They do not themselves suppress faults.

**How to use them:** count sites and actual absorptions separately with the
commands below. torchmonarch 0.5.0 produced transport faults on every stop;
0.6.0 did not in a clean loopback test. That did not justify removal: the
two-host soak still exercised the correlated and grace-window paths.
[VALIDATION.md](VALIDATION.md#lifecycle) records the evidence required before
removing a path.

The fault hook and the supervision path share one list of stop reasons, so a
deliberate recycle, a client detach, a failed-setup rollback and a partial
bring-up rollback each write the same `reason-marked` line wherever the fault is
handled. A typed `SupervisionError` that carries a stop reason writes it too:
the reason check runs after the type check, not inside the string fallback.

**The soak grep** counts lines per site (`$DRIVER_LOG` is wherever the driver's
console output is captured):

```bash
grep -o "dgxm-absorb-fire site=[a-z-]*" "$DRIVER_LOG" | sort | uniq -c | sort -rn
```

With the marker or branch that fired, not only the site:

```bash
grep -o "dgxm-absorb-fire site=[a-z-]* detail='[^']*'" "$DRIVER_LOG" | sort | uniq -c | sort -rn
```

| site | what fired | how to read it |
|---|---|---|
| `reason-marked` | a fault carried an explicit stop reason, one line per marker; `detail` is the marker | the unconditional path. The first two-box soak never saw it: no fault carried a stop reason, so an empty count does not mean the counters are dead |
| `fault-marker` | reasonless fault text matched a transport marker, one line per marker | a marker with zero lines across a whole soak is dead vocabulary in this wheel's taxonomy |
| `in-flight` | the classifier matched the fault to a deliberate stop still in flight | |
| `grace-window` | the classifier matched the fault to the post-stop grace window | a straggler arrived after the stop was confirmed |
| `token-authority` | a teardown token was checked against its time bound; `detail='granted'` or `detail='expired'` | `expired` means the bound refused a stale token and the fault stayed visible: the guard working, not failing |
| `string-fallback` | the supervision string net matched where the `SupervisionError` type check did not; `detail` is the string | usually a fault shape the type check misses, which is a reason to widen the typed check, not the strings. Rule out the other cause first: the typed check also yields when the wheel stops exporting `SupervisionError` from the audited import path, and then every line here is a moved import, not a taxonomy gap |

**No line here is an absorption.** Each site reports what a classifier saw, and two
later rules can still discard that answer. Visibility wins when another mesh is live or
a creation is in progress, so a `fault-marker` line fires and the fault still surfaces
as ERROR. `in-flight` and `grace-window` fire inside `is_deliberate_teardown_fault`, and
the fault hook re-reads the teardown state version after classification
(`src/dgx_monarch/mesh_helpers.py`, the condition that requires `absorb` and an
unchanged `teardown_state_version()`): a teardown transition in between discards the
verdict after the line is written. Site counts are an upper bound. Count real
absorptions from the surfaces that perform them:

```bash
grep -c "teardown notice (deliberate)" "$DRIVER_LOG"    # the fault hook
grep -c "stray in-flight call absorbed" "$DRIVER_LOG"   # the supervision path
```

Sites also count lines, not faults. `reason-marked` and `fault-marker` write one line
per matching marker, so one fault that carries two markers writes two.
`token-authority` writes one line each time a present token is checked: the fault hook
checks every mesh that holds a token for each reasonless fault, and checks one again
while its state changes mid-read; the supervision path can check its own mesh several
times for one failure, or not at all. Divide before you compare a site count with a
count of faults.

**An empty log** is a result, not a failure: under this pin, on this fabric, with this
traffic, the correlated path never fired. Record the soak window and the workload
beside the empty count; a later prune rests on that evidence.

## 61. The first render took minutes and the next one took seconds

**Symptom:** a render that should take about 45 s took about 250 s, and the
progress bar gave no reason. The identical render right after was about 5x
faster.

**Cause:** the first render of a new model+stack combination pays three
one-time stages before your render samples: NCCL bring-up, a cold NVMe weight
load, and the identity gate's 2-4 short test renders. The callout at the top of
this file says what makes a combination new. Every release invalidates earlier
PASS results, so the first render after an upgrade repeats the gate.

**What you should see.** The driver reports these stages in the ComfyUI window,
not only in the console. Toasts appear beside the canvas for a deferred NCCL
bring-up, a deferred cold load, the gate start and the gate verdict, and the
gate's toast stays up for the whole check. Every phase, the test renders
included, reaches the DGX Monarch sidebar's event tail as a `notice` row,
`dgxm top`'s event ticker (its graphs mark gate and load events `G` and `L`)
and the driver log. The closing line names the verdict, the seconds the gate
cost and the number of test renders it took:

    auto-gate: identity gate PASS in 61.4s over 3 proof renders. Your render
    starts now and is the only cost left; the next render of this combination
    skips the gate and the cold load

**A verdict other than PASS denies optimized residency.** FAIL, INCONCLUSIVE or
ERROR means the gate could not verify the optimized residency paths (slab
weights, low-RSS LoRAs), so the driver either runs your render on stock
residency or, under FSDP, refuses it. That toast is styled as a warning and
names the verdict; the driver log above it carries the reason. Entry 35 covers
an `ERROR`/`INCONCLUSIVE` verdict after a prior PASS.

**It denies that combination, not your session.** The ERROR line names the
model file and the combination key whose optimizations were disabled. The next
combination you render through the same graph starts from the settings the
graph asked for and pays for its own check, so a sweep whose first cell ends
unverified does not measure stock for every cell after it. Three exceptions.
The combination that failed keeps its residency settings off until its own next
check. A measured FAIL is written to the ledger and carried into every later
session for those bytes
([#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass)), but
only that combination's renders read it: other checkpoints on the same Init
node keep the `lora_low_rss` and `slab_weights` settings the graph asked for.
A FAIL does turn `comfy_managed` off for the whole Init node
([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)). And a check that a measured class-K guard stopped
disables no residency setting;
[#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora) has that line and
what to do about it.

**If you see no toasts:** the notices use the same websocket channel as
cluster-fault toasts, so a browser that never shows either has not loaded
`web/js/dgx_monarch.js` (hard-refresh the tab, and check `WEB_DIRECTORY` is
served). The driver log and the `events` array of `/dgxm/telemetry` carry every
notice whatever the browser does, so a headless driver loses only the toast.

**Do not restart a healthy job during the test renders.** Killing the queue
during the check makes the next attempt pay the same cost again.

## 62. Comfy-managed residency: what the `comfy_managed` widget does, and everything it refuses

`comfy_managed` starts ComfyUI DynamicVRAM (comfy-aimdo) in every actor to
place and page DiT weights. It is opt-in, off by default, has no `auto`, and
is not enabled by `slab_weights`.

It is experimental, LoRA-less, and lacks a sustained multi-prompt soak. Use
slab for LoRAs or LLM co-residency. Hard Clear VRAM emptied model stores and
ComfyUI's cache in two small Chroma FP8-mixed tests, and reloading reproduced
the first output exactly, but actor RSS stayed about 8.6 GiB versus a 1.2 GiB
baseline (stock retained about 8.4 GiB too). Use **Reset attached mesh** to
retire the actors and release retained process memory; persistent Worker
services stay available.
See [Memory calibration](VALIDATION.md#memory-calibration) for scope.

### The two things it forces, and the one it leaves alone

| lever | forced to | why |
|---|---|---|
| `lora_low_rss` | off | The low-RSS lazy un-bake replays comfy's full-load bake protocol, and comfy's `ModelPatcherDynamic` overrides the patch, unpatch, load and backup methods that protocol depends on. dgx-monarch keeps comfy's memory manager out of every LoRA swap path, because on this hardware it once wrote a sentinel over a real weight mid-load (docs/VALIDATION.md, LoRA original-weight backup test), so under this residency the swap path does not run at all. |
| `slab_weights` | off | Slab requires `lora_low_rss`, and a slab tensor is a raw host mapping aimdo's own allocator knows nothing about. |
| `disable_pinned_memory` | nothing, the host's configuration governs | This residency owns placement, not pinning. On unified memory the worker's configuration turns pinning off, and that is what makes DynamicVRAM zero-copy: with no pinned budget ComfyUI builds its staging buffer with no capacity and weights page straight out of the checkpoint mapping. A pinned budget sizes that buffer at twice the model and fills it with a second copy of every weight in the same pool the device computes from, measured at 2.3x the checkpoint. An operator who sets `disable_pinned_memory = false` still gets it, and the capacity check charges the copy. |

The Init node refuses `comfy_managed=on` beside an explicit `slab_weights=on`
or `lora_low_rss=on` rather than silently overruling you, and `cluster.toml`
gets the same two rules from the schema.

### The seven render-time refusals, by first line

Match the error against this list. All seven are class P: they carry
the bare `[dgxm:P]` tag, no guard, no card, and no consent clears them, because
inside a running process there is no bypass. Each names the working
alternative, and none of them loads, bakes or quarantines anything.

1. **`comfy-managed residency is on for this worker and this render carries N LoRA(s)`**
   Turn the widget off for graphs that use LoRAs and **Reset attached mesh**, or
   render the graph without a stack. Nothing was loaded and nothing was
   quarantined: the refusal fires before the load.
2. **`comfy-managed residency is on for this worker and FSDP is active for this render`**
   FSDP reshards weights into DTensors after the load while DynamicVRAM pages
   the same weights independently on every rank. The combination is untested. Turn the widget off for FSDP topologies, or pick a topology
   without FSDP. The driver also raises this
   sentence at the loader node, above the footprint card, whenever the preset
   carries `fsdp` and the widget is on. The worker's copy needs a `load_model`
   call, so without the driver's copy a large artifact would meet a capacity
   card offering slab residency, which this residency cannot use. Both sites raise
   the same string.
3. **`comfy-managed residency and the RDMA latent return cannot run in the same worker`**
   Both register host memory with the CUDA driver, and that collision has never
   been measured. `rdma_latent_return` is off by default (docs/DESIGN.md
   section 5.6), so you only see this if you turned it on.
4. **`Fleet cannot run the identity ceremony itself`**
   A Fleet call has no way to prove a new combination and no way to turn a
   bootstrap policy off for one call, so it refuses. Render the combination
   once through a normal DGX Monarch KSampler so the check can prove it,
   then Fleet takes the exact PASS.
5. **`comfy_managed is a bootstrap policy and this worker process already started with it ...`**
   You changed the widget on a live actor. Use the panel's **Reset attached
   mesh** button and queue the render again. Restart the persistent Worker
   service only if the mesh reset cannot confirm actor teardown.
6. **`comfy-managed residency is on for this worker and the driver could not authorize this render's residency`**
   Two graphs land here. One is a combination with no exact PASS for the
   comfy-managed capability context. The other is a **dual-model** render: a graph
   that dispatches a second, unconditional model is always stamped stock,
   because a check proves exactly one model. Dual-model families are
   therefore out of scope for this residency in this release; run them with the
   widget off.
7. **`comfy-managed residency is on for this worker and the first-use identity ceremony for this combination cannot prove it`**
   The check's own proof render meets a class-K guard, and a check
   renders without accuracy waivers deliberately, so the waiver that clears that
   guard on an ordinary render does not clear it inside the check and the
   check can never reach PASS here. The message names the guard. What
   works: grant that guard's waiver from the panel, or headless from the
   variable the message names, because the card the check raised expires
   after half an hour and this class-P refusal carries no panel sentence of its
   own. Then set the Init node's `auto_gate` widget to off for this graph, and
   every render is stamped rendered-under-waiver; [#67](#67-what-auto_gateoff-costs-and-what-it-does-not) says what
   `auto_gate=off` costs. Or turn `comfy_managed` off and use **Reset attached
   mesh**. If reset cannot confirm actor cleanup, follow its recovery message
   before retrying.

   The first render of such a combination in a ComfyUI session meets the
   class-K refusal itself, with its one-click card, rather than either class-P
   sentence: the check runs, its proof render refuses, and that refusal
   answers the render instead of a forced-stock fallback. The known shape is a
   chroma nvfp4 checkpoint on a sharded topology, whose
   `shard_quant_scale:chroma` guard is [#95](#95-a-sharded-nvfp4-render-refuses-with-a-waiver-card-about-activation-scales).

   A render that dispatches a second, unconditional model reads refusal 6
   instead, even when an earlier check for the same checkpoint stopped on a
   class-K guard: that render is stamped stock because of the second model, and
   `auto_gate` off does not change that, so refusal 6 names the cause and the
   one setting that works.

There is also one capacity refusal on this path, class C under the usual
`stock_load_preflight` guard and
[#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click):
**`comfy-managed residency cannot load <model> on <host>`**. See "The capacity
wall it keeps" below.

### The five bring-up refusals, by first line

These fire while the worker is starting, before anything is loaded, so you
meet them on a first bring-up rather than on a render. They are class P too.
The first four share this remedy: *turn the Init node's `comfy_managed` widget
off (or remove `comfy_managed` from `cluster.toml`) and reset the Attached
mesh, which gives this box its ordinary stock or slab residency back.*

1. **`comfy-managed residency was requested but comfy-aimdo is not installed`**
   The worker's Python environment has no `comfy_aimdo` package. Install it
   into the same environment the workers run from, on every host.
2. **`comfy-managed residency was requested but comfy_aimdo.control.init()`**
   ` could not load aimdo.so on this host.` The package is importable but its
   native library is not loadable here (wrong architecture, missing build).
3. **`comfy-managed residency was requested but this host does not meet`**
   ` ComfyUI's own DynamicVRAM requirements (NVIDIA, not WSL, torch 2.8 or newer).`
   This is stock ComfyUI's own gate, restated, not an extra rule.
4. **`comfy-managed residency was requested but`**
   ` comfy_aimdo.control.init_devices() reported no working install for device N.`
   aimdo loaded but could not install its patches for this worker's device.
5. **`comfy-managed residency was requested and this worker process already tried`**
   ` to bring ComfyUI's DynamicVRAM up and failed; ...` One of the four above
   already fired in this process. It cannot be retried in place, because comfy
   is imported by then and its patcher was never rebound, so every later render
   and every policy push into this actor refuses until you reset the Attached
   mesh. Look further back in the same worker's log for the first refusal: that
   one says what went wrong.

### Why a change needs an Attached mesh reset

Changing this policy requires **Reset attached mesh**. It patches the loaded
CUDA driver and rebinds `CoreModelPatcher` process-wide; neither can be undone
in a live actor. `disable_custom_nodes` has the same bootstrap restriction.
A persisted FAIL that turns off comfy-managed residency on the graph therefore
hits refusal 5 on an existing actor. Reset before retrying.

### What the first render costs

Enabling the widget adds `comfy_managed: true` to the capability context and
requires a new identity check. Turning it off returns to the original context
and its PASS without re-proving it.

In `cluster.toml`, **omit the key when off; never write
`comfy_managed = false`**. The merged worker-args dictionary is part of every
capability context, so either explicit value changes it and requires fresh
proof. Init writes the key only when on; quarantine changes it only on a graph
that already carried it.

### What the ceremony does and does not prove

The two test renders check consistency across ranks within comfy-managed
residency. They do not compare against classic residency: switching managers
requires fresh actors, so a same-seed, same-graph comparison must cross a reset.
VALIDATION.md records byte-equal output for one LTX 2.5 export at world 1;
world 2 is unmeasured.

No slab or cross-residency leg runs. A class-K guard cannot be waived inside
proof, so such a combination cannot pass; see refusal 7 and
[#84](#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call).

### The capacity wall it keeps

The estimate is `file size + 5 GiB`, or `2.3 * file size + 5 GiB` with a
pinned budget. It includes no legacy load arena and never charges less than
the file. The measured 61.7 GiB artifact used an 86.3 GiB working set with
28.4 GiB MemAvailable remaining; lazy placement did not reduce it below file
size. Slab is forced off, so this path cannot offer slab rescue.

On UMA, DynamicVRAM sees host available memory divided by the number of GPU
workers on that host, rather than stock `torch.cuda.mem_get_info`. Placement
can therefore differ from stock. Watch for lowvram, partial-load, or
models-unloaded lines on any rank.

The worker reserves only the 5 GiB floor. A larger `uma_reserve_gb` is enforced
by driver-side checks only for MiniMax H3/LTX on the driver host. Other
families/hosts have just the floor. Watch `dgxm top` and MemAvailable on the
first large load: UMA exhaustion can stall the host without a catchable CUDA
OOM ([#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc), "The Spark failure
catalog" below).

### How to check it took

`dgxm status` shows, per worker:

    comfy_managed: true
    comfy_aimdo: 0.4.13
    aimdo_enabled: true
    weight_residency: comfy_managed
    patcher_class: ModelPatcherDynamic

and the worker journal carries one line per rank:

    comfy-managed residency ACTIVE: comfy-aimdo <version>, device 0,...

Do not grep for stock ComfyUI's own "DynamicVRAM support detected and enabled"
line: that string belongs to `main.py`, which a worker never runs.

The model store's load line ends in a bracketed group that opens with
`residency=comfy_managed rung=comfy_managed`, then `priced=` or `unpriced`,
then the residency's `reason=`. The signed resident-adoption evidence
reports `cudaMalloc` for this residency deliberately: that field records the
storage backing, and this residency changes which manager places the weights, not
what backs them. Read the `weight_residency` field above for the residency,
not the evidence row.

### Related

* [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) covers a load that fails or takes the box down on unified memory, and the
  card that offers slab residency instead. Under this residency no card appears,
  because slab is forced off.
* [#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click) covers class C capacity refusals, with and without a one-click card;
  this residency's capacity refusal carries no card.
* [#52](#52-driver-side-footprint-preflight-refused-a-render) covers the driver-side footprint preflight and `uma_reserve_gb`.

## 63. A toast at queue time says the workflow is missing DGX Monarch nodes

**Symptom:** pressing Queue pops a toast beside the canvas that names node ids
and the DGX Monarch nodes that replace them. The render starts anyway.

**Cause:** the graph is valid and runs on the driver while the cluster stays
idle. At queue time, the driver lists the changes needed for distributed
rendering. The advice neither blocks the queue nor edits the graph.

**The three advisory cases.**

* **No DGX Monarch nodes at all.** An info toast: this render is stock and
  single-box. It lists the swap for each stock node that has a cluster
  counterpart, as `#<node id> <stock class> -> <DGX Monarch node>` (eight at
  most, then a count of the rest), and it names the Init node you have to add.
  A graph with no such node gets no toast. Three nodes have no one-for-one
  replacement and are described instead: `CheckpointLoaderSimple` becomes the
  DGX Monarch diffusion-model loader plus the stock CLIP and VAE loaders;
  `LoraLoader` becomes the DGX Monarch LoRA node, which patches the model only
  and leaves the CLIP half with no cluster path; `SamplerCustom` becomes a DGX
  Monarch guider feeding the DGX Monarch custom sampler.
* **DGX Monarch nodes with no Init node.** A warning toast naming them. Init
  attaches the cluster and hands out the mesh every other node in the pack
  takes, directly or through a loader, so a graph without it is incomplete.
* **Init present, stock nodes left over.** A warning toast listing them. Those
  nodes run in the ComfyUI process while the rest of the render runs on the
  cluster, which is usually a half-finished conversion.

**What it never flags.** The CLIP and VAE loaders, text encode, latent, image
and save nodes. Those run on the driver on both paths by design, so there is
nothing to swap them for. The advisor works from a table of known
replacements, so any node not in that table stays silent.

**One swap carries a condition.** A stock `BasicScheduler` swaps for Basic
Scheduler (DGX Monarch), and that node needs an explicit topology on the Init
node: ring2, uly2 or cfg2, not the `auto` default. It computes the sigmas
against the resident model, and `auto` picks the topology at the first render,
after the sigmas are already needed, so the node refuses with a RuntimeError
instead. Every toast that names that node says so; set the topology while you
make the swap and the first Queue works.

**Skip Layer Guidance.** A stock `SkipLayerGuidanceDiTSimple` swaps for Skip
Layer Guidance (DGX Monarch), which takes the same four widgets in the same
order, so the swap keeps every value, and carries them to every Worker. The
topologies and guiders it refuses are in [#126](#126-skip-layer-guidance-dgx-monarch-refuses-on-cfg2-on-a-sequence-parallel-family-that-does-not-apply-block-replacement-or-with-a-two-checkpoint-guider).

**One toast per graph.** Each advisory appears once per driver process.
Changing the relevant nodes produces new advice; requeueing unchanged nodes
does not. The driver log and sidebar event tail retain advisories even without
a browser.

**Turning it off:** set `DGXM_GRAPH_ADVISOR` to `0`, `false`, `off` or `no`
(case and surrounding spaces do not matter). It is read per queue, so it takes
effect on the next Queue with no restart. Anything else, including an unset
variable, leaves it on.

## 64. A packed audio/video custom render dies on the last step

**Symptom:** a MiniMax H3 or LTX audio/video graph built on `RandomNoise` +
`KSamplerSelect` + Basic Scheduler (DGX Monarch) + Basic Guider (DGX Monarch) +
Sampler Custom (DGX Monarch) samples every step and then raises `custom sampler
x0 must be a Tensor, got NestedTensor`. The sample is abandoned rather than
retired, so the next render meets `LifecycleBusyError` until `dgxm restart`.
The same graph on KSampler (DGX Monarch) renders.

**Cause:** an older dgx-monarch build. ComfyUI runs the sampler on one flat
pack, and since Comfy-Org/ComfyUI#15196 it hands every callback the nested view
of `x0` back whenever the latent carries more than one modality. Older
dgx-monarch builds rebuilt the custom sampler's second output from the flat
pack alone, so the only x0 shape a packed render still gets was the one shape
they refused. The denoised output is rebuilt whether or not a node consumes it,
so leaving that slot unconnected never helped.

**Fix:** update dgx-monarch on every host and restart the driver and Worker
services from the same source. Both x0 shapes rebuild the same shape- and
dtype-matched denoised wrapper, per modality, as stock SamplerCustomAdvanced
does. Nothing about the graph or the family changes.
Related: [#36](#36-packed-audiovideo-sampling-fails-after-denoising) for the packed output contract, and [#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2) for H3's topology
refusals.

## 65. `dgxm status` and `dgxm doctor` are green and every attach still fails

**Symptom:** both diagnostics report a healthy cluster,

    10.0.0.1: worker_service=running listener=open (tcp://10.0.0.1:26600, nohup)
    10.0.0.2: worker_service=running listener=open (tcp://10.0.0.2:26600, nohup)

while every queued render raises `MeshAttachError`, or a `LifecycleBusyError`
counting abandoned samples. The driver log usually shows the identity gate
aborting on that error first and falling back to stock residency.

**Cause:** the Worker service rows are true, but they answer the wrong
question. A running process and an open listener are host facts; what refuses
the render is driver-side state on the cached mesh handle: a sample abandoned
by a render that died mid-sample, or a teardown whose outcome was lost.
Neither touches the Worker services, so no per-host probe can see them.

**Read the mesh row.** Both commands ask the driver over its telemetry route
and print what it says:

    mesh: ok (cached, 0 active leases)
    mesh: DIRTY (1 abandoned sample may still own worker-side latent backing); reset the attached mesh...
    mesh: POISONED (the driver's Monarch transport cannot be reused...); restart ComfyUI...
    mesh: idle (no attached mesh; the next render session creates one)
    mesh: busy (load_model RPC completion in flight); wait for it, nothing to fix
    mesh: no driver reachable (no mesh view)

`doctor` prints the same verdict in its `mesh health` row, and a DIRTY mesh is
a failure there, not a warning: the driver has already decided that the next
attach raises. Active leases are not a fault; they are renders in flight.

**`busy` is not a fault either.** A cold first load runs for minutes and a
gate cycle longer, and the driver holds the same lifecycle latch throughout.
The row names the operation and passes `doctor`, because the reset a DIRTY
row prescribes would kill that load. Wait for it. If the named operation is
still there long past its budget, treat it as the dirty case below.

**Fix:** follow the mesh row's remedy. An abandoned sample clears with **Reset attached
mesh** in the sidebar panel; restart the Worker service only if setup still
fails after that. An unresolved teardown outcome asks for the same reset first;
restart ComfyUI if the reset cannot confirm the stop.
Never restart the Worker service underneath a live driver mesh.

**Two rows the reset button cannot clear.** `POISONED` reports that this
driver's Monarch transport cannot be reused after a partial bring-up or a
creation failure. It is process state, not handle state, so it reads the same
with an empty cache and no reset touches it: restart ComfyUI. A `DIRTY` row
naming a fleet that was created and never published is the other one. The reset
route reads the published cache only, so it answers `no_live_mesh` there and
changes nothing. Start the next render instead: it drops an interrupted fleet
that is already gone and attaches a fresh one. While that fleet is still live,
or while the driver cannot tell whether it is, the render refuses with a typed
error naming the restart, because a second fleet on
one transport is worse than a stopped render.

**Related:** [#1](#1-attach-fails-with-mesh_attach_config_timeout) and [#2](#2-every-attach-times-out-after-one-failed-attach) cover attach timeouts and a wedged Worker service, a
different failure that looks the same at first. Start there when the mesh row
is `ok` and attaches still fail. [#3](#3-next-render-hangs-at-nccl-rendezvous-after-a-crash) covers stranded procs on the NCCL
master port. `no driver reachable` means the CLI found no ComfyUI on this box,
so there is no mesh to report on yet; both commands find your driver on
whatever port it was launched with, not only the default one.

## 66. The gate says INCONCLUSIVE on a graph with no LoRA

**Symptom:** the first render of a LoRA-free graph reaches a verdict of
INCONCLUSIVE whose reason is that there was nothing to compare. It reads as a
failure, and on older builds it was treated as one: the line

    ERROR auto-gate could not complete; lora_low_rss, slab_weights disabled
    for this session

followed, and every later render in that ComfyUI process ran on stock
residency whatever model it loaded, for the rest of the session.

**What you see now.** A check that ran end to end on a graph with no LoRA
stack, with slab residency not engaged, had nothing to compare. That is the
expected outcome for such a graph, not a failure, and it logs one line at INFO:

    INFO identity gate: nothing to gate on this graph (auto_first_use,
    <model>): no LoRA stack and slab residency is not engaged, so there is
    nothing to compare; INCONCLUSIVE recorded, residency levers unchanged

The ledger and the gate report row still record INCONCLUSIVE for that
combination, and both rows carry `"inconclusive_kind": "no_material"` beside
the verdict. Your render still runs on stock residency, because an unproven
combination never gets the optimized paths. The canvas toast still names the
verdict and says optimized residency is off, which is true of this render.
What no longer happens is the session-wide disable: every other model and
every other graph in the process keeps the residency it was configured with.

**A no-material verdict skips later checks.** First use of the same
combination, in this driver or a later one, reuses the ledger verdict. The
render runs on stock residency and logs at INFO:

    INFO normal render: the identity gate found nothing to gate for this
    combination, so it keeps that verdict; this dispatch runs on stock
    residency and the session's levers stay as configured

**Scope and invalidation.** The skip binds the same identity as a PASS: model
bytes, LoRA set, ComfyUI commit, residency/topology policy, gate protocol,
package release, and the canonical source manifest cached on first use. Any
change requires one fresh check, including a restart into changed same-version
source. The cache does not monitor live source edits. Source-independent FAIL
quarantine still takes precedence for the same bytes. Older rows without
`inconclusive_kind` require one new check.

**What still disables both residency settings, and for which combination.**
Every other INCONCLUSIVE, at ERROR. A check that could not finish the
comparison it set out to make leaves the fleet in an unmeasured state, so both
residency settings are disabled. That record names the combination it came from: the model file, its options and its
LoRA set. The next combination you render through the same graph starts from
the settings the graph requested and runs its own check, because a check
that never loaded those bytes proved nothing about them. The combination that
aborted keeps its residency settings off, and its own next check re-decides
them. The one exemption to the write itself is a live capacity-rescue consent
([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)).
`NO_QUARANTINE_VERDICTS` in `src/dgx_monarch/nodes/gate_inconclusive.py` lists
the outcomes that write no residency setting into the graph, and
`src/dgx_monarch/nodes/gate_quarantine_scope.py` decides which combination the
write lands on. A FAIL is on that list because it never writes `lora_low_rss`
or `slab_weights` into the graph: each render of the failed combination turns
both settings off for itself
([#67](#67-what-auto_gateoff-costs-and-what-it-does-not), item 2).

**A check that a class-K guard stopped disables no residency setting.** A check
renders without accuracy waivers deliberately, so a measured guard such as
`shard_quant_scale:chroma` or `sol_attn` refuses inside it whether or not you
hold the waiver. That refusal is an accuracy answer about that combination and
says nothing about slab residency or the low-RSS swap path, so nothing is
quarantined and no worker policy is pushed. The line names the guard:

    ERROR auto-gate could not complete for <model> (combination <key>): the
    class K guard shard_quant_scale:chroma refused inside the ceremony. That
    guard is measured, so it answers this combination's accuracy and takes no
    residency lever from the graph

Render that combination again in the same session and it answers from the
cache, with no second check and no quarantine, and still on stock: the guard's
answer stands until you clear it. Grant the guard's card and set `auto_gate` to
off for that graph ([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)),
or render it on a shape the guard does not refuse.

**The no-material outcome covers the residency you ran under, and nothing
else.** Turn `slab_weights` on afterwards and you are asking for a combination
this check never exercised, so that render runs its own check. The grant is
scoped the same way. The check writes a row for the Fleet scope and for the
explicit-on slab siblings too, and none of those rows carries the
classification, so none of them grants a skip.

**Telling them apart:** INFO with `nothing to gate on this graph` took nothing
from you. ERROR with `auto-gate could not complete` and `disabled for that
combination` took both residency settings from the combination it names; the class K line
above took none. Grep `dgxm_gate_reports.jsonl` for `inconclusive_kind` to get
the same answer per run, or `dgxm_gate_ledger.jsonl` to see which combinations
hold the grant.

Another ERROR-path INCONCLUSIVE says the two repeated optimized-path renders
diverged even though the fresh stock comparison matched one of them. That is
repeat instability, not proof: it publishes no process PASS, persists no PASS
for normal, Fleet, or slab sibling contexts, and disables the risky residency
for that combination. The gate protocol version that added this check
invalidates PASS rows written before it. Preserve the report and investigate
nondeterminism; do not relabel the result as a LoRA failure when the repeat
itself was unstable.

**Not this entry:** [#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass) covers an `ERROR`/`INCONCLUSIVE` verdict on a
combination that previously passed, which is a revoked proof rather than an
absent one.

## 67. What `auto_gate=off` costs, and what it does not

Turning the Init node's `auto_gate` widget off skips identity proof. It keeps
LoRA rendering and the requested optimized residency available, subject to the
independent guards below.

**What off does.** It skips the proof, and nothing else. A single-model render
is stamped `operator_off` and dispatched on the worker policy your graph asked
for, untouched. It does not use the forced-stock pair an unverified combination
gets under `first_use`. So, with `auto_gate=off` on a single-model render:

| | with `auto_gate=off` |
|---|---|
| a LoRA stack | renders, on whichever residency the settings below resolve to |
| `lora_low_rss` | stays as configured (`auto` by default, which is on for unified memory): the lazy un-bake, not baked mode |
| `slab_weights=auto` | still resolves per family on the worker, and a family verified for slab residency still slab-loads |
| `slab_weights=on` | still honored |
| the capacity rescue ([#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click)) | still offered, still consented, still certified |
| render accuracy | unproven; checking it is up to you |

**Four things that stay true with off, and that off does not cause.**

1. **Fleet keeps forcing both residency settings off for an unverified combination.** A Fleet
   call cannot run the check inline, so it reads the ledger and, finding no
   exact PASS, renders that call with `slab_weights` and `lora_low_rss` off.
   `off` records no PASS, ever, so it never fills the ledger in either. The
   Fleet job still runs and still applies your LoRAs; it bakes them, at the
   residency the `lora_low_rss` tooltip describes for its OFF setting. Prove
   the combination once with the Identity Gate node or `dgxm gate` and Fleet
   takes that exact PASS.
2. **A FAILED check still quarantines.** `off` skips a missing proof, never
   a measured one. A FAIL for this exact combination and capability context,
   persisted or measured earlier in the session, turns both residency
   settings off for that combination's renders before setup, whatever the
   widget says. Other checkpoints on the same Init node keep their
   `lora_low_rss` and `slab_weights`; `comfy_managed`, if the graph sets it,
   goes off for the whole Init node (DESIGN.md, "What a FAIL takes, and from
   whom"). [#35](#35-identity-gate-reports-errorinconclusive-after-a-prior-pass) is the way back.
   The FAIL holds within one package version. A release update expires
   context-scoped FAILs, so `first_use` retests the combination and `off`
   treats it as untested.
3. **`comfy_managed=on` refuses LoRA stacks.** That widget sits in the same
   Init advanced block, so its refusal is easy to blame on `auto_gate`.
   Different lever, different reason ([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)), and `auto_gate` neither causes
   that refusal nor clears it.
4. **A dual-model custom render dispatches on stock residency.** When such a
   render dispatches at all, both residency settings are off for it whatever the widget
   says, for the reason the note at the top of this file gives. `off` is not
   what turned them off, and `first_use` would not have turned them back on.

**Not the capacity route.** For a checkpoint only slab residency can fit, use
the capacity rescue: a card, a consent, and a ledger row that records the
combination as INCONCLUSIVE rather than unexamined.
[#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) and
[#53](#53-a-render-failed-and-the-panel-is-asking-for-one-click) carry that
flow.

**When to use it.** A combination you have already proved and do not want
re-proved once per release; a graph whose identity test render triggers a
refusal your own render would have tripped anyway, so the check aborts and
costs you a Recycle on the way
([#49](#49-minimax-h3-ring2-refused-on-a-padded-packed-sequence)); a bisect
where the extra test renders confuse the measurement. In each case `off` saves
test renders; it adds no feature.

**Effect on later renders.** A check that aborts on a measured class-K guard,
the shape the second case above describes, takes no residency setting from the
graph at all, and any other aborted check takes them only from the combination
that aborted ([#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora)).
So `off` here saves you the test renders and the Recycle, not the residency of
later renders.

## 68. MiniMax H3 fails with `cannot import name time_shift_slope` after a ComfyUI update

**Symptom:** the first distributed MiniMax H3 render fails inside the rebound
forward with an import error for `time_shift_slope`, or the Comfy canary names
the `minimax h3 audio carry` seam. A newer dgx-monarch beside an older, partial
or mixed ComfyUI tree instead refuses because `minimax_payload` lacks a valid
`audio_scale` marker, or because the bound stock `MiniMaxH3Model.forward` does
not carry the matching audio conversion. Infrastructure and ordinary actor
checks can stay green.

**Cause:** current ComfyUI removed that helper and moved the dual-schedule
conversion around the inner `_forward`: its stock outer
`MiniMaxH3Model.forward` restores the audio latent before invoking wrappers and
converts the returned raw velocity afterward. An older dgx-monarch adapter
still imports and applies the former inner derivative, so it crashes on the
removed name, and would transform audio twice if the helper were copied back.
The current distributed inner contract under an older outer forward would omit
the conversion. The payload marker comes from a different Comfy module, so it
can sit beside the legacy outer method in a partly updated tree. The adapter
therefore requires both a finite positive `audio_scale` and the fingerprinted
current outer forward before any distributed math runs.

**Fix:** update dgx-monarch and the complete ComfyUI checkout on every host,
check that driver and workers resolve the same source revision, and restart
them so no old payload producer, outer model class or adapter module stays
imported. Do not restore the removed helper or patch the stock outer forward.
Until the versions match, use a true stock single-GPU run with
`mode=local, gpus_per_host=1`; the world-2 `single` preset is no substitute,
because MiniMax H3 refuses it for the data-parallel reason in [#48](#48-minimax-h3-refuses-cfg2-dp2-and-single-at-world-2).

**Lease behavior after the refusal:** the three payload arms (absent payload,
missing `audio_scale`, malformed `audio_scale`) are decided from the request,
fire identically on every rank before any collective, and carry the class P
tag, so their sample lease retires consumed: fix the cause and queue again on
the same fleet, with no recycle. The bound-forward arm ("does not carry
ComfyUI's matching audio unscale") is decided per host: on a mixed fleet the
sibling rank may already sit inside a collective, so that arm is untagged and
retires abandoned. After it, the next dispatch refuses with the
abandoned-sample guard until one Attached mesh reset (hardware test: one reset,
then the next render dispatches normally).

Version alignment proves only that the current outer and inner contract is
present; it does not prove fidelity. The support row in docs/MODELS.md is the
status authority. The carried-audio retest passed (docs/VALIDATION.md), so that
row is marked HW.

## 69. Worker bring-up is interrupted during spawn, rollback, or client-lease startup

**Symptom:** Init reports `worker actor bring-up failed`, says spawned
processes were `rolled back` or `not safely stopped`, or reports a
client-lease renewer startup failure. When rollback is incomplete, the error
also says the Monarch transport or prior fleet is not reusable. A request for
a different mesh configuration can instead report that another mesh creation
is still in progress; retry it after the active attempt settles.

**Cause:** actor creation touched a client-owned ProcMesh and then raised, a
completed actor spawn was interrupted at its return handoff, partial-ProcMesh
rollback raised as well, or the process-wide daemon that renews actor lifetime
leases could not start after the handle was built. These boundaries can raise
native `BaseException` subclasses, not only `Exception`, and cleanup treats
both as failures. The handle becomes ready only after the lease renewer starts,
so a waiting caller cannot reuse a fleet with no renewer. If native ProcMesh
construction aborts before the pinned callback, active registry or return value
exposes its identity, the driver cannot prove stop authority and poisons the
session instead of guessing. Mesh creation runs one at a time per process,
even across cache keys, because the transport, callback registry and
ownership handoff are process-global.

**Fix:** read the original spawn, handoff or lease failure first; rollback
evidence adds to it and never replaces it. Confirmed cleanup after an
already-built handle or client-lease startup failure may evict that handle and
retry the same transport mode. A `WorkerSpawnError` after transport
initialization leaves the process poisoned even when its partial ProcMesh
rollback was confirmed; restart ComfyUI or the driver before retrying. Only a
local failure proven to occur before transport import or initialization is
retryable in-session.
If a completed stop call reports a definite non-timeout ordinary failure,
replacement stays blocked, but a later explicit **Reset attached mesh**,
shutdown or supervision cleanup gets a fresh bounded stop attempt. A second
definite failure stays blocked, and only another explicit cleanup retries it;
the failed action never retries itself. If the stop timed out, was cancelled or
interrupted, its outer caller timed out while the lifecycle thread was still
settling, or an attempt token survives without a published outcome, do not
issue another stop: the first call may already have succeeded. Wait for
settlement, or confirm the client-owned actor processes are gone and restart
ComfyUI. In every case, do not queue a replacement until cleanup is confirmed.
Never call `hosts.shutdown()` or stop the attached Worker services beneath a
live driver; they are owned separately.

## 70. `dgxm update`, `gate`, or `top` misses a ComfyUI driver on a custom port

**Symptom:** ComfyUI was started by `scripts/comfy-driver.sh` on a port other
than 8188 or 8191, but an old CLI falls back to 8188, or `dgxm update` does not
refuse while that driver is live.

**Cause:** the shipped launcher changes into the ComfyUI checkout before it
runs `python main.py --port N`, so process inspection sees the bare relative
script name, not a path containing `ComfyUI`. Older discovery discarded that
process before probing its listening port.

**Fix:** update dgx-monarch. Discovery admits the bare `main.py ... --port N`
shape as a candidate only. It confirms the pack through ComfyUI's stock
`/object_info/DGXMonarchInit` endpoint and requires the exact Init node name,
category and output type, so an unrelated `main.py` listener or any HTTP 200
or 503 is not enough. `gate` and `top` then use `/dgxm/telemetry` only to rank
confirmed instances by live workers; a cold telemetry timeout or a valid 503
does not hide a confirmed driver, and `update` does not wait on that slower
endpoint. If process inspection is unavailable or restricted, pass the address:
`dgxm update --host 127.0.0.1:N`, `dgxm gate --host 127.0.0.1:N`, or
`dgxm top --host 127.0.0.1:N`. Stop ComfyUI before rerunning an update that
refuses the live driver.

## 71. Pipeline/Fleet refuses `batch_index`, or a chained sampler refuses inherited waiver provenance

**Symptom:** a multi-render node raises `inconsistently returned batch_index
metadata`, says the value must be a list, says its entries must be nonnegative
integers, or reports that its length does not match the constituent sample
batch. A chained sampler can instead say `inherited waiver provenance must be
a list or tuple`, that an entry must be a mapping, or that it must carry
nonempty string `run_id` and `guard`.

**Cause:** a Pipeline or Fleet result joins several separately rendered latent
batches. An explicit index list belongs to each constituent batch and must
concatenate in the same order as `samples`; copying only the first list
creates `N x B` samples with `B` indices. The node refuses mixed presence,
non-list values, entries other than exact non-boolean nonnegative integers,
and wrong lengths instead of publishing a misaligned latent. Separately,
`_dgxm_rendered_under_waiver` is trust-bearing input metadata: when present it
must be an exact list or tuple of mappings keyed by nonempty string
`(run_id, guard)`. The Sequential, Pipeline and Fleet entry paths validate it
before any gate, mesh, session or submission side effect. Valid duplicates
keep the first entry; inherited entries never claim or record a second waiver
use. [#55](#55-rendering-under-an-accuracy-waiver-and-what-the-stamp-means) describes the audit contract.

**Fix:** update and restart the driver and Worker services from one source
revision, then rerun the complete Pipeline or Fleet batch. If the refusal
persists, inspect the upstream or custom sampler that produced the named
constituent: every result must either omit `batch_index`, or give a list whose
length equals that result's sample batch and whose entries are exact
nonnegative integers. For an inherited-provenance refusal, do not delete or
hand-edit the stamp: find the custom node that published the malformed
container or identity, update it, and rerun from the last valid latent.
Padding, truncating or cleaning the metadata by hand can change seed and index
meaning, or erase the record of an earlier render under an accuracy waiver.

## 72. A distributed render refuses a Comfy attention patch or NAG-style hook

**Symptom:** a sequence-parallel Flux-family, Chroma, Hunyuan or Wan-Animate 2
render raises `UnsupportedModelError` naming `attn1_patch`,
`attn1_output_patch`, or (for Wan-Animate 2) `attn2_patch`; a Qwen Image 2.1
render refuses a Comfy block or attention hook; or a Qwen Image 2.1,
Wan-Animate 2 or LTX sequence-parallel render refuses a foreign
`optimized_attention_override`. A CFG-parallel render can also refuse an output
hook that needs both condition branches.

**Cause:** each USP rank owns only a token shard, and each CFG rank owns only
one condition branch. Passing the original callable through unchanged does not
keep its tensor domain: a full-stream image slice can address the wrong local
rows under USP, and an NAG-style hook that looks for both positive and negative
branches silently returns unchanged under CFG. Wan-Animate 2's cross-attention
hook also receives only spatially local rows. Qwen Image 2.1's block-causal
prefix must stay replicated and unchanged. A foreign attention override writes
the key the distributed adapter owns, so one of the two writers would be
dropped, and on the native-attention paths (Qwen Image 2.1, Wan-Animate 2) it
would replace the kernel the selector gate checked. Exact math requires the
hook to declare and implement the matching shard-local or condition-local
contract, so unknown hooks fail before distributed attention instead of
producing finite but wrong guidance.

Stock Comfy patch nodes accept `MODEL` and DGX loaders output `DGXM_MODEL`, so
a stock NAG node cannot attach to the shipped DGX graph. This refusal guards
out-of-tree worker-side patch carriers and any later patch transport.

**Fix:** remove the model-option patch or attention-backend override, or use
topology `single` for stock hook behavior. Do not rename or wrap the hook to
get past the guard. Qwen Image 2.1 admits none of these hooks under USP. A
hook opts in through its `_dgxm_attention_patch_capabilities` attribute:
`sequence_shard_local` on the Flux-family, Chroma, Hunyuan and Wan-Animate 2
USP paths, `condition_shard_local` for a CFG output hook. Declare
`sequence_shard_local` only when the hook maps its image slice to local rows in
every affected USP block. Declare `condition_shard_local` only when the hook is
correct on one CFG branch per rank; a hook that needs both branches must
exchange across conditions to be correct. Declare either one only after
hardware fidelity validation has passed.

## 73. Wan SCAIL/SCAIL-2 refuses a multi-row timestep embedding

**Symptom:** a distributed SCAIL or SCAIL-2 forward raises
`UnsupportedModelError` because its timestep embedding has more than one row.
A normal sampler timestep shaped `[batch, 1]` is accepted.

**Cause:** the SCAIL sequence-parallel contract replicates one scalar timestep
row on every rank. Replicating a multi-row `e0` while sharding the combined
reference, video and pose token stream repeats the rows in each local shard
instead of aligning them once over the global sequence. Plain Wan and
WanDancer have different expansion rules; borrowing them without proving
SCAIL's complete combined-stream layout would be another silent math change.

**Fix:** use the stock scalar-timestep SCAIL path or topology `single`. A
custom sampler that emits multiple timestep rows needs a specified
full-sequence expansion and an identity and fidelity campaign before the
distributed guard can be relaxed. This is separate from [#47](#47-wan-scailscail-2-render-refused-activation-footprint-preflight)'s activation
footprint refusal.

## 74. `cluster.toml` is rejected for an unsupported key

**Symptom:** `dgxm` or the Init node raises `ClusterConfigError` naming the top
level, `[cluster]`, an indexed `[[hosts]]` entry or `[worker_args]`, and lists
one or more unsupported keys. A `[fabric.<name>]` profile error instead names
the one key it rejects, such as a name outside the allowed `NCCL_`, `GLOO_` and
`UCX_` tuning variables, and fires even when that profile is not selected.

**Cause:** configuration tables are closed schemas. An unknown root, cluster or
host key is never ignored, because a misspelling such as `client_bnd` or a
misplaced `transport_security` would otherwise select a default in silence and
fail later at attach or security enforcement. Every declared fabric profile is
validated when the file is read, so a typo in an unselected profile cannot
become active at a later profile switch.

**Fix:** move the setting into the table documented in docs/CLUSTER.md and use
the exact listed key. Remove obsolete private metadata instead of renaming it to
look like a supported field. A valid unselected fabric profile stays
unapplied; only its schema is checked.

## 75. Native RDMA fails at QP RTR because the two ends selected different rails

**Symptom:** after an operator enables `rdma_latent_return`, a small latent
returns by actor messaging, but the first threshold-sized or larger native
return fails while bringing an RDMA queue pair to RTR. The two ends report
different selected rails even though normal dual-Spark NCCL renders,
`dgxm doctor`, and passive fabric health can stay green.

**Cause:** without an explicit RDMABuffer rail selector, each transfer end
chooses on its own among NICs with the same best CPU-path score. A
point-to-point dual-rail fabric has no cross-rail path, so a deterministic
address and size choice can put the two QP endpoints on different links. A QP
split of one limits the descriptor to one part; it does not force both hosts to
choose the same rail. The operator-authorized test measured this failure at
exactly 8 MiB; the full gate failed, and the later above-threshold and reattach
cases were not run.

**Fix:** leave `rdma_latent_return=false`, its shipped default. This does not
disable rendering across two Sparks: distributed compute continues over NCCL,
and latent results return through actor messaging. Keep the first native
failure and confirm the run's cleanup and postflight; do not retry that native
registration or read, increase the QP split, or hide, rename, reorder or remove
devices to force a rail. Device hiding is not the supported selector and can
silently make the RDMA, NCCL and diagnostic views disagree. If the failed
attempt reports ownership poison or unconfirmed settlement, follow [#29](#29-large-rdma-latent-return-fails-after-workers-finish-rendering) and do
not reuse that process or numeric handle.

The pinned runtime provides `rdma_ibverbs_target`; the later
[selected-rail qualification](VALIDATION.md#native-rdma) records its limited
production-return integrity, ownership, recycle and reattach evidence. Native
return stays off by default and positive-step qualification is limited to the recorded at-threshold case.
The historical failure still governs uncertain ownership; a dependency update
alone does not qualify another rail, workload or configuration.

## 76. An operator workflow refuses, reports `unknown`, or leaves Worker services stopped

**Symptom:** the panel, TUI or JSON readiness is not green, `dgxm setup` prints
a plan but changes nothing, `dgxm doctor --repair` declines a diagnosis, or
`dgxm update --verify` exits with a receipt and, by design, does not restart
the Worker services.

**Readiness:** inspect the named Worker service, Attached mesh or Render
session layer, and rerun the read after telemetry is back. Do not treat
`unknown` as lifecycle authority. Status and doctor observe different subsets
by design; the meanings and aggregation rules are in
[CONCEPTS.md](CONCEPTS.md#operator-words).

**Guided setup:** fix the named host or prerequisite and rerun the dry run; do
not add `--apply` to make a blocker go away. For an ownership conflict, stop
and inspect the existing unit, enablement and source deployment, then use
verified update where it applies, or retire or move the legacy managed source
by hand, before retrying. `config_lock_unavailable` means another
cooperating config operation is active or the private lock state is unsafe;
let the operation finish or inspect that state, and do not run competing
config writers. `service_activity_unknown` means process enumeration was
incomplete, and `service_ownership_unknown` means setup could not prove the
exact source or unit it would change; neither authorizes config publication.
`verification_unknown` means doctor or smoke transport, evidence or cleanup was
incomplete after apply. In that state setup leaves the config and Worker
services in place, attempts no optimistic compensation, and records a partial
top-level result with `unknown` receipt steps. Inspect the named evidence
before deciding whether to stop, repair or retry. `verification_failed`, by
contrast, is exact negative evidence with settled smoke ownership, and may
compensate the owned services and roll back the config.
After an interruption, keep the private output and inspect the config, unit,
source and receipt state. The complete plan, confirmation, recheck, ownership,
verification, compensation and settlement rules are in
[INSTALL.md](INSTALL.md#guided-multi-node-setup); the profile mapping and Gate
digest effect are in [CLUSTER.md](CLUSTER.md#guided-setup-profiles).

**Safe doctor repair:** if a safety recheck refuses, apply the named non-file
remedy by hand. If receipt publication fails, inspect the config mode first:
permission tightening may already have happened even though doctor did not
rerun and the command returned nonzero. Keep the failure output, repair the
private receipt destination, then run doctor again. The narrow contract is in
[INSTALL.md](INSTALL.md#safe-doctor-repair).

**Verified update:** use the receipt's last stable step to find the uncertain
boundary. After an ambiguous stop, switch, compensation, start or driver
commit, do not start a possibly mixed cohort by hand; inspect per-host release
links, release identities, pin payloads, and all-rank source and status
evidence first. An interrupt may leave no receipt, so keep the private terminal
log. Smoke cleanup owns only its ProcMesh and never calls `hosts.shutdown()`;
attach auto-heal can restart an unhealthy Worker service within the lifecycle
authority already granted. Recognized legacy units are migrated
under an installation journal; an edited unit or override is not adopted.
If ownership or installation readback is unknown, inspect the captured unit,
source link and journal before retrying. Keep the retained controller snapshot,
prior releases and transaction evidence. A definite `failed` step stays
eligible for stop plus guarded compensation and restart; schema-v2 `unknown`
produces top-level `partial` instead and ends mutation. An unknown stop,
activation, target start, compensation or prior restart runs nothing more; an
unknown doctor, cluster smoke, exact readback or driver commit after target
start runs one `fail_closed_stop` and nothing more. Inspect each host before
you start anything. Preconditions and settlement are canonical in
[INSTALL.md](INSTALL.md#updating).

**Receipts:** a disclosure-reduced receipt is not automatically safe to
publish. Review it before sharing, and keep full private logs for omitted
details. The default and custom path rules, how the `setup_compensation`
counts tell an unchanged config from a rolled-back one, and every no-receipt
and partial-publication case are in [INSTALL.md](INSTALL.md#operator-receipts).

The [operator checks](VALIDATION.md#operator-checks) cover readiness, repair
(including refusal during a conflicting operation), guided setup and its
two-rank smoke test, and verified updates on the colocated two-Spark layout. A
stopped Worker correctly blocked status and the Doctor Worker-row adapter, and
the same service was then restored; this was not a full Doctor run. The record
includes update successes, refusals before changes began, and rollback results.
These results do not change
[#75](#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails):
native latent RDMA stays off by default with native cancellation and fabric-failure recovery untested, while ordinary dual-Spark
NCCL rendering and actor-message latent return stay available.

## 77. A Chroma, Omnigen2, or Z-Image USP render refuses a mask or sequence length

**Symptom:** a sequence-parallel render raises `UnsupportedModelError` saying
that an effective attention mask cannot be applied, or that a sequence length
is not divisible by the sequence-parallel degree and the adapter requires
exact divisibility. The refusal occurs before distributed denoising.

**Cause:** the USP kernel takes no arbitrary attention bias, so Chroma and
Z-Image cannot keep a real caption or joint attention mask on a sharded
forward. Separately, an adapter with no named evidence for excluding a
synthetic divisibility-pad row from attention refuses a stream that does not
divide the degree: padding and trimming only at gather would leave the pad row
as a real key in every block, so the runtime refuses instead of returning a
finite output whose accuracy has not been checked. Only Omnigen2 refuses this
way, on its one joined ring stream. Chroma, Flux, Flux2 and LongCat each
declare a pad-exclusion check: they pad the stream and remove those rows before
every kernel call (docs/ADAPTERS.md). The Flux, Flux2 and LongCat tests passed
every check for the tested padded shapes; Flux 1 Schnell has no leg of its own
and stays unmeasured on a padded stream.

**Fix:** for a Chroma effective mask, use `cfg2` where the graph has a real
batched cond/uncond call, or a true one-GPU run with `mode=local` and
`gpus_per_host=1`; stock Chroma attention honors that mask.
Z-Image has no supported topology for an arbitrary mask, because its stock
Lumina forward clears `cap_mask`. Use mask-free conditioning; standard
asymmetric prompts under pure cfg2 take the exact trim path, but do not
treat `single` as proof that an arbitrary mask was kept. For an unmasked USP
render, choose a topology whose SP degree divides every applicable stream. The
error names the family and the affected stream with its length, dimension and
degree: Omnigen2 checks one joined-token stream and is ring-only. Do not drop
a real mask or add prompt tokens to silence the guard. A padded shape needs an
exact pad-row exclusion contract plus a named fidelity campaign, which Chroma
and the three other flux-family members carry. That contract has one limit:
the exclusion applies at a full-sequence point only Ulysses has, so a padded
stream on ring or hybrid raises the waivable class K `ring_pad` card
(entry [#21](#21-sequence-length-needs-divisibility-padding--ulysses-only-on-ringhybrid)) instead of rendering. It fires before any collective, so the
render stops instead of hanging. These divisibility refusals are not
accuracy-waiver cards.

## 78. The dashboard records `UNKNOWN`, or mesh reset says busy or outcome unknown

**Symptom:** after pausing or scrubbing `dgxm top`, a slow live response does not
appear. A malformed poll creates an `UNKNOWN` tick, or only the run block says
`run story unavailable (Type)`. The retained-pool number can hold its previous
value while the driver is down. In the browser, **Reset attached mesh** can say
`attached mesh busy; wait`, `reset already active; wait`, or `mesh outcome
unknown; inspect` instead of quickly returning to its idle label.

**Cause:** each of these fails closed: none publishes what it cannot prove. A
poll started before timeline navigation is stale even if the user returns to
`LIVE` while it is in flight. Normalization and run-narration failures are
isolated, so stale READY data or one broken panel cannot replace current
uncertainty. The local retained-pool scan has a 0.5-second budget and keeps its
last complete value instead of publishing a partial `/proc/*/smaps` sum. Mesh
reset runs one at a time and is rate-limited; a timed-out HTTP request does not
prove that its background lifecycle operation stopped, so the gate and visible
outcome stay until that operation settles.

**Fix:** for the TUI, return to `LIVE` and wait for the next fresh poll. Treat
`UNKNOWN` as missing evidence; inspect `dgxm status --json` and the driver log
instead of assuming the previous green state still applies. A narrator-only
error does not invalidate the other panels. For mesh reset, wait while a render,
lease, or already accepted reset is active. A reset refused for live work stops
nothing, and the `/dgxm/recycle` route answers it with HTTP 409 and a retryable
result naming that work. Do not send repeated resets after a
timeout or unknown outcome. Inspect lifecycle status and retry only after the
accepted operation has settled or a definite non-timeout ordinary failure
grants a later fresh cleanup attempt. A reset skips the workers' model unload,
and the process stop frees the weights instead, so a stop that
times out leaves those processes holding their weights until `dgxm reap` or a
worker service restart removes them. An unexpected reset failure is sanitized
in the browser; use the driver log for private diagnostic detail.

## 79. A cross-Spark render refuses because the ranks disagree about weight residency

**Symptom:** a world-2 render stops with a class C refusal saying the ranks
disagree about weight residency: at least one rank holds the model only
partially. Every rank raises the same message. Just before it, each box's
worker journal carries one line, `partial-load guard: local full_load=...
world=...`, and a box that reported `False` is short. ComfyUI's own line above
it says `loaded partially` on a short box and `loaded completely` on a whole
one.

**Cause:** ComfyUI decides how much of a model it can hold when sampling
starts, not when the file loads, and it decides per box against the unified
memory free at that moment. Two Sparks rarely have the same amount free, so one
can drop into the partial-load regime while the other does not. A partially
held rank streams and casts its offloaded blocks per step, so the ranks compute
different weights for the same step. The render does not crash: its ranks drift
apart. Measured on an LTX 2.5 bf16 transformer with a baked LoRA stack under
forced-stock residency: the cross-rank latent canary read 2.27e-03 relative
drift. The usual source of the pressure is a bf16 checkpoint whose LoRA stack
is baked with ComfyUI's full weight backup still held, which costs about twice
the checkpoint on every box. Under an FSDP preset, an older shard build itself
caused the pressure: it copied this rank's rows to the device straight from the
mmap'd checkpoint, the driver broke the mapping's copy-on-write under its pin,
and a family with any file-backed leftover (a replicated fp32 island, a scalar,
a matching buffer) held its whole shard again as anonymous host pages until the
model was dropped. The current build sends rows through a pinned bounce buffer
and the mapping is released before ComfyUI's load. The worker journal's `FSDP load: checkpoint mapping released` line names the copy-on-write bytes, and a
non-zero figure there means that defect is back, not a capacity fact.

**A stock resident is placed at load time.** ComfyUI sizes a stock model's
device copy at sample time, while the loaded file still sits on the host, so a
load the stock estimate allowed could partially load there. The store now runs ComfyUI's own full load when the stock load
finishes, and the stock memory estimate charges that placement's whole peak (2.1 times
the file plus the floor), so a box without the room refuses at
`cross_rank_capacity` before anything loads. A stock resident with a LoRA stack
baked through ComfyUI's weight backup (hot-swap residency, `lora_low_rss` off)
places at sample time instead and can meet this guard.

**Fix:** reduce memory pressure, then rerun. Load an int8 or fp8 checkpoint
in place of bf16; turn `lora_low_rss` on so the stack bakes and the weight
backup is freed; or run the shape on a single box, where a partial load is
ComfyUI's ordinary correct behavior and nothing can disagree with it. The
refusal is not waivable and has no card: going ahead would authorize ranks that
compute different weights. The mesh survives it. The lease is consumed, not
abandoned, so the next render dispatches on the same fleet with no recycle.

ComfyUI's own sample-time estimate can also exceed the box before any rank
loads, and then ComfyUI offloads every weight on that box (entry 91 shows one
such checkpoint). The render-memory preflight refuses that case before
dispatch, and its refusal points here. It refuses at world 1 as well, because
the estimate is per box, so the single-box remedy above does not answer it:
reduce the resolution or use a smaller or lower-precision checkpoint.
`DGXM_DISABLE_ACTIVATION_PREFLIGHT=1` on the driver is its only bypass.

**When the short rank holds a slab or an FSDP shard,** read the load lines
first. Both builds put every weight on the compute device without ComfyUI's own
load running, so ComfyUI's weight ledger would read zero and it would ask for a
whole model of free memory at sample time for bytes the box already holds.
The fresh load declares the residency to ComfyUI, and the
worker journal carries two lines for it at load time. ComfyUI writes the first
from inside the load the declaration asks it to run: `loaded completely; N MB
loaded, full load: True`. The pack writes the second: `resident ledger:
declared N GiB of slab weights to comfy...` (`fsdp shard weights` for a
shard). Where nothing was declared there is no ComfyUI load line, and the
pack's line reads `resident ledger: ... left undeclared (reason)`, naming why.

A declared rank prints no load line when sampling starts, because ComfyUI takes
its early return for an already-loaded model and that return logs nothing. The
sample-time evidence is the guard's own line, `partial-load guard: local
full_load=True world=2`, on each box.

Three shapes tell the cases apart:

* `loaded partially` at sample time with no declared line above it, and a
  loaded figure well under the model size: the zero-ledger case. The `usable`
  figure on that line is what ComfyUI had left to spend, not the size it asked
  for. Check why the declaration was skipped.
* `Unloaded partially: X MB freed, Y MB remains loaded` at sample time after a
  declared line. ComfyUI has the weights but not the working set, so it gave
  weights back. The fixes above apply: a smaller checkpoint, a lower
  resolution, or more free memory on the short box.
* a declared line, no load line at sample time, and `local full_load=True` on
  both boxes. Nothing is wrong; this is what a declared render looks like.

## Power-user escape hatch

When the node surface is not enough, use what the integration owns and can
stop:

* `dgxm top` ([TUI.md](TUI.md)) and the DGX Monarch sidebar panel;
* `/dgxm/telemetry` and `/dgxm/metrics` on the ComfyUI port, which include
  the live handoffs, leases, poisoned owners and the native-return latch;
* a bounded `dgxm diagnostics` capture
  ([CLUSTER.md](CLUSTER.md#monarchdiagnostics));
* with `[monarch.diagnostics] metrics = true` and an `otlp_endpoint`, the phase
  metrics and the ownership occupancy counters in your own collector
  ([CLUSTER.md](CLUSTER.md#monarchdiagnostics));
* for each completed native return in a normal render, the `native return:`
  log line, which splits the Worker's pack and the driver's read into phases.

Monarch's own services stay blocked: `enable_telemetry`, `Job.profile`,
`monarch query`, the Monarch dashboard and the mesh admin agent that
`monarch tui --addr` reads. All but the admin agent need a Job around the
persistent Workers, which would make a second Monarch client attach to them
and push its config to their host agents. Telemetry also spawns a proc mesh on
every Worker host and stops it only on a best-effort basis, and the dashboard,
when enabled, binds every interface. The admin agent needs no Job and can bind
loopback, but it has no stop handle and, as far as the 0.7.0 Python surface
shows, ends only with the process that started it. The periodic snapshot actor
has no stop handle and runs until its procs stop. So dgx-monarch has no
inspector mode for them, and you should not start them by hand against a live
fleet. [DESIGN.md section 5.8](DESIGN.md#58-progress-logging-observability)
cites each fact in the 0.7.0 wheel.

An inspector session that dgx-monarch can own needs four upstream changes: a
stop on the admin agent's handle, an abort handle for periodic snapshots
(upstream already marks it as a TODO), telemetry that does not spawn procs on
the hosts, and an attach that does not push the client's config. A bind host
for the dashboard would also let that session include it.

## Never disable DynamicVRAM in stock ComfyUI on a Spark

Measured on a single GB10: krea2-class 24.5 GiB bf16 DiT plus 7 LoRAs, 20 steps
at 1024, identical flags, page cache prewarmed.

| | dynamic (default) | --disable-dynamic-vram |
|---|---|---|
| warm render | 76 s | 238 s (3.1x slower) |
| GPU-attributed after load | 40.3 GiB | 8.6 GiB |
| retained allocator pool | 0 | 19.7 GiB |
| anonymous memory | 5.3 GiB | 41.8 GiB |

Without comfy-aimdo's DynamicVRAM, ComfyUI falls back to its legacy VRAM
estimator, which misjudges unified memory and drops to offload streaming
without saying so: only about 8.6 GiB stays resident, and the rest of the model
moves through the offload path on every step. On a Spark the disable flag does
not mean "classic full load"; it means accidental lowvram mode, at 3x the
render time, plus the retained one-model-sized allocator arena that the legacy
load path leaks (the "pool" segment in `dgxm top`). The arena returns to the OS
only when its process exits, which in stock ComfyUI means a restart. For
dgx-monarch workers the ClearVRAM node's `recycle` level and the panel's
**Reset attached mesh** button do that: they stop the actor processes, which
frees the model and the arena together, and the next render attaches a fresh
fleet.

DynamicVRAM can run inside Monarch workers. Its background threads were tested
starting, running beside concurrent endpoint calls and joining cleanly. The
integration depends on funchook patches to the loaded libcuda library. Enable
it with the Init `comfy_managed` widget; it is opt-in, supports no LoRAs, and
has not had a sustained multi-prompt test
([#62](#62-comfy-managed-residency-what-the-comfy_managed-widget-does-and-everything-it-refuses)).

The default worker-side answer is the Init `slab_weights` widget (zero-copy
shared-memory weight residency, docs/VALIDATION.md): with it on, a worker shows
pool 0 like stock DynamicVRAM (0.0 vs 20.7 GiB dirty in the same process),
loads in about 20 s, renders with identical output, and returns the model's
memory to the OS as soon as it unloads.

### Single-box LoRA swaps under stock DynamicVRAM are fast and exact

Measured on the same setup: a strength change re-renders with about 10-14 s of
overhead, and the swapped image is pixel-identical to a fresh load at the same
strengths. Compare pixel arrays, not PNG file hashes: ComfyUI embeds per-run
metadata in the PNG. Single-box users do not need this stack for memory or swap
speed; the README's [What we measured](../README.md#what-we-measured) says what
the pair gains.

## 80. An LTX attenuated guide refuses on ring or hybrid

**Symptom:** an LTX 2.5 image to video or first-and-last-frame render with a
guide strength other than 1.0, or with a connected `attention_mask`, stops with
a class P refusal before any pixels. It names a pure-ulysses topology and
`single` as the alternatives. Pure ulysses renders the same graph.

**Cause:** LTXV Add Guide records one entry per guide. When a guide asks to be
weakened or strengthened, comfy turns those entries into an additive
self-attention bias so the noisy tokens and the guide tokens attend each other
at a chosen weight. Two settings ask for it: a strength other than 1.0, and a
connected `attention_mask`.

Comfy never builds a dense mask for that. It splits the query axis into the
noisy tokens, the tracked guide tokens, and anything after them, then attends
each group against the whole sequence under its own weights. Ulysses
re-expresses that exactly: after the head scatter every rank holds the whole
token axis for a subset of heads, and the weights do not vary by head, so each
rank applies the same groups to its own heads. Ring cannot. Its ranks see the
keys one block at a time, and the kernel that walks those blocks takes no bias
argument, so there is nothing to hand the weights to and nothing a waiver could
grant.

**Fix:** run a pure-ulysses topology. On a two-rank fleet that is `uly2`, which
is also what `auto` picks for LTX at every canvas size. For a stock one-GPU
run, use topology `single` (`mode=local` with `gpus_per_host=1`), where comfy
applies the bias itself; `cfg` topologies also run stock attention per rank, so
they take an attenuated guide unchanged.

Setting strength to 1.0 on every LTXV Add Guide and leaving `attention_mask`
unconnected works everywhere, including ring: comfy then builds no bias.

The concat route has no such limit: LTXV Image To Video writes the frame into
the latent and sets no conditioning keys, so its own `strength` widget is a
noise-mask value, not an attention weight, and every value shards.

The refusal is rank-symmetric and decided from the request, so it retires the
sample lease consumed: fix the graph and queue again on the same fleet, with no
Recycle. This entry is a launch constraint, not a defect, and carries no
accuracy-waiver card.

## 81. Explicit slab residency still reads quarantined after a LoRA-only gate FAIL

**Symptom:** `slab_weights=on` refuses for one model and LoRA stack, and the
log names a persisted quarantine, but no check ever ran slab residency for that
combination. Rescue consent for the same combination refuses too.

**Cause:** older builds wrote an identity-gate FAIL verdict onto the
explicit-slab sibling contexts whether or not the check had exercised slab
residency. A failure that came from the lazy-swap leg alone therefore recorded
a FAIL against a capability nothing had tested. Those rows are sticky for the
same artifact bytes and terminal for the automatic path, so the quarantine
outlives the check that wrote it.

**Fix:** the gate writes a FAIL onto the sibling contexts only when the
cross-residency leg itself diverged, the only leg a check runs under slab
residency. Otherwise the preflight `RETESTING` rows stand: the sibling stays
denied and re-gates on next use.

**Recovery for a row already on disk:** capability-scoped ledger rows bind to
the running gate protocol and package version, so an upgrade that changes
either one retires the stale row. The rescue consent path reads the same
binding, so a retired row stops blocking the capacity ladder when it stops
governing the gate.

Consent ignores the rows this entry is about whatever version wrote them: a
stamped sibling row whose check never diverged on the cross-residency leg
records nothing about slab residency. Older releases read them, which trapped
the operator: the rescue consent is what lets the checkpoint load
slab-resident, so refusing it also blocked the one explicit check that would
overwrite the row.

On a version-pinned install, a stale row the gate still reads clears with that
check: add the Identity Gate node to the graph for that exact model, LoRA
stack, and residency policy, and run it once. An explicit check re-tests and
overwrites the row, where the automatic path treats it as terminal and never
retries.

## 82. A MiniMax H3 guide refuses at the loader node for its encode

**Symptom:** a graph with Add Guide for MiniMax H3 refuses before the model
loads, and the refusal charges a guide encode of 94 GiB at 1344x768, 67 GiB at
672x384, or, at 448x256, between 17 and 48 GiB depending on how many frames the
wall could prove. The image on that input is a single still.

**Cause:** Add Guide resizes its frames to the target canvas and encodes them
with the video VAE before the sampler runs. That encode grows faster than the
canvas area does. Measured on one DGX Spark, with the VAE already resident: one
frame at 1344x768 cost 13.6 GiB, five frames at 448x256 cost 16.2 GiB, five at
672x384 cost 66.9 GiB, and five at 1344x768 never completed, past 94 GiB.
Length was swept at the one canvas where a long clip completes, 448x256: 16.2
GiB at 5 frames, 33.4 at 22 and 47.5 at 39. Roughly a gigabyte a frame, so
length matters as much as a resolution step. The frames arrive on a link, so
the graph does not say how many there are.

**Known guide length.** A Load Image node names its file, so the capacity
check opens that file before the render and counts its frames. One frame takes
the measured still charge and the render proceeds. The check uses file
contents: an animated PNG, WebP or GIF reports its real count and takes the
clip charge, and a video, a missing file or a damaged one does not open at
all, which also takes the clip charge. Put a scaler or any other node between
the loader and the guide and the file is no longer named, so nothing is proved
and the clip charge stands.

**Fix when nothing can be proved:** start ComfyUI with
`DGXM_H3_GUIDE_FRAMES=1`. The capacity check then prices the measured still,
14 GiB at 1344x768 or smaller, and charges every other term as before. Any
value under 5 prices a still, because a batch that short anchors its first
image alone. A value of 5 or more prices a clip. Above 448x256 a clip
declaration buys nothing: the canvas price stands. At or below 448x256 the
declaration also picks a length bucket (5, 22 or 39 frames), so a short
declaration reduces the estimate from the 39-frame worst case used for an
unreadable link. A frame count proved from the file outranks the declaration
either way.

That declaration is a statement about the whole driver process, so a driver
started with `=1` believes it for every guide whose length nothing proves. Set
it for graphs whose sources hide their files, and leave it unset otherwise.

**Fix for a clip:** shrink the canvas *and* let the capacity check see the length. Five
frames at 448x256 fit beside the DiT and the text encoder, but only when a Load
Image node names the file or `DGXM_H3_GUIDE_FRAMES` declares the count. An
unread link at that canvas is charged the longest clip a run measured, 39
frames, at 48 GiB, and can still refuse. At 1344x768 no clip fits on this
hardware, which entry 83 covers.

`DGXM_DISABLE_DRIVER_PREFLIGHT=1` also clears the refusal, but it turns off the
weight, text encoder, VAE, reference and activation charges with it.

## 83. A multi-frame guide clip at 1344x768 takes the whole host down

**Symptom:** a guide clip at the full H3 canvas does not raise, does not
degrade, and does not finish. Load average climbs past 200, MemAvailable sits
near zero for tens of minutes, and the kernel eventually kills the process
with the highest `oom_score`, which is usually not the one that filled the
memory.

**Cause:** two things compound. The encode itself needs more than 94 GiB at
that geometry, which is more than a 121.6 GiB box has beside a DiT and a text
encoder. And ComfyUI's tiled-VAE fallback, built to rescue an encode this size,
never fires: it triggers on a caught CUDA out-of-memory error, and on GB10
unified memory there is no clean CUDA OOM to catch. The allocation succeeds
against host memory, the host exhausts, and the box thrashes instead of
degrading.

**Fix:** respect this measured hardware limit. Anchor a
still at this canvas, or move a real clip to 448x256. Entry 82 covers the
charge that refuses the clip before it starts, on a render queued from the
browser. A script that imports the loader node and calls it directly hands the
wall no graph, so that charge never runs and the host thrashes as above.
Entry 85 covers that caller.

## The Spark failure catalog

This catalog lists recurring GB10 failure modes from the community forums and
this project's own history. `dgxm doctor` checks the safe subset. It runs no
load test. The one GPU kernel it runs is a small Sage probe, which it runs only
when no other process holds the GPU (entry [#89](#89-doctors-sageattention-row-fails-its-kernel-probe-or-counts-fallbacks)). The other checks belong in a
maintenance window the operator controls.

### PD-stuck state

One USB-C power-delivery bug shows three ways: silent hard power-offs under GPU
load, renders suddenly ~3x slower after an OTA update, and "GPU reads 11 W at
44 C while a workload is running." A hard power-off leaves nothing in any log:
the absence of a kernel panic, Xid or OOM record across an unclean boot is the
signature. After a crash the GPU can stay pinned at ~650 MHz and 5-9 W across
normal reboots. `dgxm doctor` warns when the last 30 boot records hold more
than one reboot with no clean shutdown before it (row `unexpected shutdowns`).
It discounts one, because the oldest reboot in the window often lacks its
pair. Doctor reports its `GPU clock under load` row as **NOT RUN** and gives no
PD-state verdict. During a workload you already authorized, compare
its clock and power readings with a healthy host; do not start a diagnostic
load beside a render. The community-confirmed remedy is a full cold
power-cycle: unplug the power brick at both ends for at least 30 s, then boot.
A plain `reboot` does not clear it. Some users also cap clocks
(`nvidia-smi -lgc`) as prevention. A cap costs speed: the head box of this
project's pair runs under a 2100 MHz stability cap, which cost up to about 4
percent of render time when measured
([BENCHMARKS.md](BENCHMARKS.md)).

### ConnectX-7 / cross-Spark fabric

* Dead rail after boot (link trains DOWN, phys LinkUp): reboot the affected
  box. There is no software fix; do not spend time on driver restarts.
* NIC vanished entirely (lspci shows the bridge but no Mellanox device;
  dmesg: "retraining non-functional downstream link"): full power cycle,
  not a reboot.
* Bandwidth halved with both rails UP (nccl-tests busbw ~15-17 GB/s
  instead of ~22-24): a known regression when the SoC firmware moves from
  the 10500 family to 10600. Check `fwupdmgr get-releases`; the community
  remedy is `fwupdmgr downgrade` to the prior SoC firmware.

### Everything else

* CUDA "operation not permitted" after a container idles for hours:
  restart the container; no host reboot needed.
* Sage or Triton silently ~20x slower with correct launch flags: the kernel
  fails on every call and comfy falls back to torch attention, logging
  `Error running sage attention` each time. Two common causes are missing
  Python development headers, which break Triton's JIT (install the matching
  `python3.x-dev`; doctor row `python dev headers`), and a stale-ABI Sage build
  after a torch bump. Doctor runs the kernel and counts the real fallbacks in
  `~/comfyui-monarch.log` (entry [#89](#89-doctors-sageattention-row-fails-its-kernel-probe-or-counts-fallbacks)). For a ComfyUI that logs elsewhere,
  count the lines that carry `Error running sage attention` and not
  `Unsupported head_dim`: a growing count means Sage is not running, whatever
  the flags say.
* Memory exhaustion can hard-lock the whole OS rather than OOM-kill the
  offender (still possible after firmware improvements). Prevention on this
  stack: `lora_low_rss` (on by default on a Spark) replaces comfy's full
  weight backup with references into the checkpoint file, and
  `uma_reserve_gb` warns when a model load or swap leaves less than that many
  GiB available and also refuses a driver-side weight load
  that would eat into it ([#52](#52-driver-side-footprint-preflight-refused-a-render)). For Docker co-tenants, set explicit --memory
  limits.
* One board sensor ("temp6") sits outside the fan curve and can reach 90 C+
  quietly; `dgxm doctor` warns on it, reading `/sys/class/hwmon` directly, so
  the check needs no package and runs on a stock Spark (entry #59).

## 84. A sol-attn render refuses with a waiver card, or stalls on its first call

The `SOL_ATTN_TAU*` attention kernels (tau 0.6, 0.7 and 1.0 ship as named
values; lower tau keeps more key blocks and runs closer to exact attention, and
the three elapsed times sit within about 4 percent of each other) are
approximate by construction and not identity preserving. A persisted panel
consent covers its combination, not one tau: switching the widget between
shipped taus on the same model and topology re-uses the standing grant, and the
ledger's use row records which tau ran. Measured on a two-Spark pair, the same
seed and graph give a coherent, sharp video of a different composition from
exact attention. So it refuses by default; that refusal is intended, not a
fault.

Accept it the way any class-K waiver is accepted. In the browser the sidebar
offers a card; headless, submit once, let it refuse, then read the pending row
and accept it:

```
curl -s localhost:8191/dgxm/consents            # read `pending`
curl -s -X POST localhost:8191/dgxm/consent \
  -H 'X-DGXM-Action: consent' -H 'Content-Type: application/json' \
  -H 'Origin: http://localhost:8191' \
  -d '{"action":"accept","key":"<key>","id":"<id>"}'
```

Then re-queue. Both ranks log KNOWN-WRONG MATH PROCEEDING UNDER WAIVER, the
driver logs RENDERED UNDER WAIVER, the output carries
`_dgxm_rendered_under_waiver`, and the gate ledger gets one `use` row per
guard. `DGXM_WAIVE_KNOWN_WRONG=sol_attn` on the driver is the headless
equivalent of the click.

**The first call of each shape is slow, and that is not a hang.** The kernel
compiles per shape and per configuration on first use, seconds rather than
milliseconds, worst at long sequences. A first render pays it once per shape,
and a resolution or duration change pays it again. Wait it out before
diagnosing a stall.

**Other refusals you may see, and what each means.** A render at
sequence-parallel degree 1 refuses because the model keeps ComfyUI's own
attention there, so the kernel would never run under a name that says it did.
A ring or hybrid topology refuses: the kernel returns no log-sum-exp for ring
ranks to merge. A family other than MiniMax H3 refuses with its own reason. A
box without the package refuses naming that box: the driver checks every rank
before it dispatches, because a rank raising inside a collective abandons the
render and strands its peers. A checkpoint that pins `comfy_kitchen_sol` on
its attention modules raises no card under sequence parallelism, because no
attention call runs the pin there: the sharded attention uses the widget's kernel, and
the stock attention the pack leaves in place (cross-attention, and the
refiners and encoders every rank runs whole) uses ComfyUI's default exact
backend ([ADAPTERS.md](ADAPTERS.md) lists each site). Under sequence
parallelism the card appears only when the widget selects Sol. A topology
without sequence parallelism keeps every stock attention call, so a pin runs
there as it does in stock ComfyUI, and the render refuses with this same card
until the operator grants it
([#117](#117-a-render-refuses-because-the-checkpoint-or-a-lora-pins-an-attention-kernel),
which also covers the FSDP launch check a bf16 or fp16 checkpoint that pins
meets first).

**The kernel is optional and is not a dependency of this project.** The opt-in
installer plans every configured Worker interpreter. First stop ComfyUI and
other GPU clients and retire their Attached meshes; idle persistent Worker
coordinators may remain running. Planning downloads/builds artifacts into
private per-host directories but does not install packages:

```bash
dgxm --config /absolute/cluster.toml sol-install
# If installed compiler dependencies need changing, explicitly name each one:
dgxm --config /absolute/cluster.toml sol-install --allow-change protobuf
# Review the exact per-host old/new versions, then use the printed token/digest:
dgxm --config /absolute/cluster.toml sol-install --plan TOKEN --plan-digest SHA256 --allow-change protobuf --idle-maintenance --apply
```

Use only the `--allow-change` names required by your reviewed plan. Torch,
Triton, TorchMonarch, xFuser, yunchang and NVIDIA CUDA runtime/NCCL library
changes are refused even with that flag. Compiler bindings (`cuda-python`,
`cuda-bindings`, `cuda-pathfinder`) may be added; replacing an installed binding
requires the same explicit permission as protobuf. Apply rechecks idle state,
installed versions, staged wheel hashes and protected runtime file hashes under
the shared lifecycle lock, installs the reviewed wheels with `--no-deps`, and
reads back the result. Existing file-ownership overlaps refuse. Neither mode
restarts services, installs system headers, repairs caches or rolls back an
uncertain/partial result. Keep clients stopped throughout maintenance; the lock
coordinates Monarch commands but cannot prevent a manual process launch.
Failed/unknown outcomes retain their private staging for inspection; do not
blindly retry a partially applied fleet plan.

A normal editable install may expose both installed `.dist-info` and source
`.egg-info`. The installer accepts only a matching version and a source
companion bound to the recorded editable project; file ownership still comes
from the installed distribution. If a duplicate remains unproven, inspect the
named distribution instead of deleting metadata or bypassing ownership checks.

An existing protected library may be a symbolic link to a regular file outside
the environment. Its link identity and target contents are preserved and
checked independently; that target never becomes an allowed installation or
uninstall destination. Directory links, replacement through a link, and
changing or unreadable link evidence still refuse.

The installer uses Sana revision
`71350faed59a6865173ddfd5a7d4471da4bde328`, CUTLASS DSL `4.7.0` and
`apache-tvm-ffi==0.1.14.post1`. For a separately managed environment, the
underlying manual package selection is:

```
pip install 'sol-attn @ git+https://github.com/NVlabs/Sana@71350fae#subdirectory=techniques/sparse_backends' apache-tvm-ffi
```

`apache-tvm-ffi` is required: the fast kernel needs it at run time, and the
package's own requirements do not pull it in. Without it upstream dispatch
selects the fast backend and then fails at call time, so a box that lacks it
refuses as if the package were missing.

**On a DGX Spark the fast kernel needs one dispatch entry.** Upstream matches
compute capability exactly and ships no `(12, 1)` key, so a Spark silently
runs the portable reference; every published Spark figure this project found,
including NVIDIA's own GB10 configuration, ran on that slow path. This project
adds the key when the device really is `(12, 1)` and both the cutlass DSL and
`tvm_ffi` import, and logs which backend it got. The DSL is the CuTe compiler
and no render refuses without it (the `dgxm doctor` row `sol-attn install`
warns), so a worker log reading `backend=triton` on a Spark usually means it
is missing: `pip install nvidia-cutlass-dsl==4.7.0` (the version measured
here) on every box, then read the backend again. Do not "fix" any of this by
forcing the reported capability to `(12, 0)`: the compiler then targets
`sm_120a` and every call dies with "no kernel image is available for execution
on the device". The gap is easy to miss: the two backends differ by about 20
percent end to end on the same graph, and two earlier runs recorded the fast
path while running the portable one. [VALIDATION.md](VALIDATION.md) carries the
timings for the shipped recipe and the earlier backend misidentification. Use
the logged backend to identify which implementation ran.

**Check every configured interpreter before selecting Sol.** `dgxm doctor`
reports detected packages, installed requirement compatibility (including
transitive protobuf constraints), and readable `Python.h` for each configured
worker interpreter. It reads package metadata without importing Sol or compiling
a kernel. A dependency conflict is advisory for this optional backend; it does
not make a stock `TORCH_FLASH` installation fail. Missing or unreadable metadata
is unknown, not proof of compatibility. Inspect requirements using that worker's
Python (`python -m pip check` is a useful broader check); satisfy the installed
packages' constraints rather than copying another machine's protobuf downgrade.
Review the package plan and preserve the working Torch/CUDA build.

A missing header needs the development headers matching that interpreter (for
example, `python3.12-dev` for a system Python 3.12); installing headers for a
different Python does not resolve it. Doctor also inspects the account-home
Triton cache candidate for an ownership/access blocker. This is bounded,
read-only evidence: it does not inspect existing cache children or prove that
compilation can write. It does not attest the running Worker/actor environment.
In particular, `TRITON_CACHE_DIR` exported in an SSH/login shell is filtered by
Monarch's Worker environment boundary and does not reach actors. Service `HOME`
and allowed cache settings may also differ from the diagnostic shell. Inspect
the actual service configuration when resolving cache problems; preserve old
caches and avoid recursive ownership changes or deletion during active work.
Other compiler caches and their runtime access remain untested.

After changing dependencies, wait for rendering and queued work to finish, then
retire the **Attached mesh** through its owning ComfyUI client before attaching
fresh actors. Their imported packages and cached backend choice can otherwise
outlive the installation. This is separate from restarting persistent **Worker
services**; service maintenance needs its own explicit authority and idle checks.
Run Doctor again and inspect the new actor's `backend=... cute=...` kernel log.
Detected packages and observed prerequisites do not prove GPU compilation or
execution, and successful kernel execution does not establish full-render
accuracy. The explicit Sol accuracy waiver still applies.

**An optional compilation check is a separate GPU action.** While the cluster
remains idle, `dgxm --config /absolute/cluster.toml sol-smoke --idle-maintenance`
checks each configured single-GPU host with 4,096-token BF16 tensors. On GB10 it
requires the CuTe backend, checks finite output, and compares a full-KV dense
control with PyTorch SDPA (relative L2 at most 0.01). The reported backend and
measured error describe that kernel control only: no model is loaded, no video
is rendered, and no sparse-attention fidelity or accuracy waiver is granted.
Doctor never invokes this check. A timeout or missing host result is unknown;
inspect retained process state before attempting another GPU action.

**A first-use identity check does not run under this kernel, and that has a
cost.** The check compares residency, not kernels: every leg reuses one frozen
request, so both legs would run sol and the comparison could only measure the
kernel against itself, where a PASS would record known-wrong math as proven. So
the gate logs that it did not run and writes nothing, and no residency setting
is quarantined, because a kernel choice does not implicate slab or lazy-swap
residency. The stated limit that follows: with no PASS for that context, a sol
render on a risky combination (a LoRA stack, or slab residency requested) runs
under stock residency, which means baked LoRA and no slab. Renders with no LoRA
and no slab request are unaffected. That limit stands until a cross-kernel
measurement verifies a tau. That instrument is separate from the check, and
this project does not ship one.

## 85. A script or harness fills the box although the loader wall exists

**Symptom:** a leg driven by a Python script, a benchmark or a test loads a
model that the same graph would have refused in the browser. MemAvailable falls
to near zero during the driver's load and encode stage, the box thrashes, and
the kernel eventually kills some other process. The driver log carries one line:
`loader footprint preflight: no readable graph, headless call`.

**Cause:** the loader-site capacity check prices the text encoder, the VAEs, reference and
guide encodes and the unspent render activations from the graph ComfyUI passes
with a queued render. A caller that imports the node classes and calls `load()`
passes no graph, so the capacity check charges the checkpoint weights and nothing else. On
a family whose driver stack is tens of gigabytes before any media, the term that
would have refused is the missing one. Nothing has failed when the line appears:
it describes the caller, not the checkpoint, and it is printed once per model
family per driver process. The capacity check prices only the families it holds a
driver-stack profile for, so a headless load of any other family prints nothing:
the capacity check never armed for it and had no stack to leave unpriced.

**Fix: gate capacity in the caller.** A direct node call cannot see the graph, so
the harness owns the budget. Run one model-stage configuration per process, wait
for the previous leg's process to be gone and MemAvailable back near idle before
launching the next, and record the reading each leg launched against.

**Or submit the graph.** A harness that has an API-format graph should POST it to
the driver's `/prompt` endpoint instead of importing the node classes. That path
carries the graph, so every term is priced and every refusal is typed and
readable.

Entries 82 and 83 cover the guide encode and the hardware exclusion behind it.

## 86. A render error says the fleet stalled, or a worker refuses with a poisoned CUDA context

**What happened.** A rank died inside the render: a sticky device fault (a
misaligned address or illegal access class CUDA error), an out-of-memory kill,
or any crash that never sends its reply. Its peer then blocks inside the next
collective, so no rank answers and the driver's wait would otherwise sit
silent until the hard timeout while the prompt queue holds.

**The stall error.** The driver compares the fleet's progress stream against a
stall budget (default 600 s; `DGXM_SAMPLE_STALL_TIMEOUT_S` overrides it, and 0
disables the guard). A fleet silent past the budget is declared stalled with a
typed error, the mesh is marked defunct, and the next render respawns workers.
The eviction completes its own teardown, so the session heals in place and no
ComfyUI restart is needed. The respawned workers hold nothing, so the next
render loads the checkpoint again, and the driver stops crediting them with
weights they no longer hold. Where a capacity estimate reads that credit, that
render is charged for the load it now has to pay for and can refuse where the
one before it did not. Raise the budget for graphs whose single step or cold
load takes longer than the budget without stalling. Each worker journal
(`journalctl --user -u dgxm-worker.service`, per box) names what its rank was
doing when the stream stopped.

**The poisoned-context refusal.** A sticky fault outlives its exception: every
later kernel in that worker process fails or corrupts. After any endpoint
failure the worker probes its own context once, and a dead context latches a
refusal on every later GPU endpoint in that process. The refusal is untagged
because the decision is host-local and a mixed fleet can leave the healthy rank
mid-collective, so clearing it costs one Recycle (sidebar button, or
`POST /dgxm/recycle`), which respawns clean workers.

**What to look at next.** The first failure in the poisoned worker's journal is
the real defect; the stall and refusal report its consequences.

## 87. A log line says the abandoned-lease wedge healed and the render paid a respawn

**What you see:** after a worker-side crash without a typed refusal, a later
render logs that it healed an abandoned lease and pays for fresh actors plus a
cold model load.

**What to do:** diagnose the original crash in the failed worker's journal.
The automatic reset addresses abandoned state, not the underlying defect. If
the retry also fails, follow the reported recovery; it does not loop.

**Why the lease stays abandoned:** a sample whose rank raised an error has
stopped. A collective call in Monarch 0.7.0 gathers a reply from every rank
before it returns, so the driver sees an actor error only after every rank has
replied (traced in the 0.7.0 source on 2026-10-14). A rank that raised after
packing its latent can still own the RDMA backing for it, so the driver keeps
the lease abandoned rather than released. The mesh row and
`SampleResultBusyError` say the sample may still own worker-side latent
backing, and `LifecycleBusyError` says it may still own a result or RDMA
descriptor of its setup generation. In three other cases the sample may still
be running: the driver stopped waiting after a collection timeout, an
interrupt abandoned the render, or a worker died. Monarch reports a worker
death at once, while the other ranks may still be inside the sample, for
example blocked in NCCL (DESIGN.md section 5.2).

**Automatic reset is limited:** it runs at most once per submit, only when
abandonment is the sole obstruction, with no active sample lease or pending
RDMA read. A Fleet queue now runs the same reset, once per queue and before its
first dispatch, and its log line names Fleet. The nodes that set the fleet up
before any render run it too: Init with an explicit topology, the UNET Loader
under one, and the BasicScheduler. Their log lines name the node, so changing
the checkpoint, topology or levers after a crash no longer needs a manual
Reset. Every path checks and resets the fleet it is about to use, not the Init
node's first fleet, so a later wedge in the same ComfyUI session heals the same
way. The ProcMesh stop must be
observed before the retry, the Fleet dispatch or the node's setup. When it is
not, every path raises a `LifecycleBusyError`. Active work is left
alone. An unconfirmed earlier stop keeps replacement blocked; supervision text,
a timeout, or a stalled channel is not proof that processes exited. In a
recovery test, a reported-dead worker remained alive for eleven more minutes.

**After worker death plus a timed-out stop:** the driver requests read-only
actor liveness on every configured host, over SSH remotely and locally on the
driver host. It checks actors, zombies, live ledger rows, and unidentified
processes. Unidentified includes empty command line plus environment, and a
process holding >=4 GiB with unreadable environment and no Python module in
its command line. A named actor module is counted as an actor. Zombies count
as gone; stopped processes remain alive.

All hosts must report gone before the fleet is considered gone. Any live
process keeps it alive; incomplete evidence stays unknown. Either blocks
replacement. The check runs at Init attach, render liveness, and Reset:

* A `gone` result younger than 30 seconds can be reused; retirement rechecks
  its age.
* Each handle is probed at most once every two seconds. Alive/unknown results
  are not cached beyond that interval.
* Without configured hosts, one shared thread reads local procfs. A stuck
  read leaves later probes unknown instead of creating more threads.

A fleet-wide gone result retires the cached handle and permits one fresh fleet;
it never issues a second stop. If replacement setup fails, the new handle is
blocked and the error identifies its liveness-proven predecessor. The
replacement attach dials each loop's current id, so a loop that restarted
within the delivery timeout attaches at once
([#2](#2-every-attach-times-out-after-one-failed-attach)). It tries once, plus
once more if a loop moved to a new id during that try, and otherwise names the
restart recovery in [#101](#101-reading-the-attach-trace-after-a-cluster-attach-times-out).

Logs show every result as `liveness probe host=... answer=...` and
`liveness fold answer=...`. Release adds
`replacement latch released on liveness proof`, host names, and abandoned count.

**A timeout without worker death is different.** An operator Reset, clean
detach, or stall-guard eviction whose stop times out cannot be released by this
proof and gets no second stop. Confirm the workers are gone, then restart
ComfyUI. The render, panel, and Reset error distinguish this state from the
worker-death recovery above.

## 88. The first render of an FSDP topology refuses with a capacity figure and no proof

**What you see:** first-use FSDP proof refuses with a capacity estimate before
it can establish PASS. Clean-reload proof unloads/reloads every rank and
compares two renders; the refusal says the next load does not fit beside
other resident memory, often the driver's text encoder. This says nothing
about topology or weight correctness.

**What to do:** free memory outside this render's shards, such as another
model or the driver's text encoder (Clear VRAM, `include_driver=True`). The
estimate already counts this render's own shard pool, so freeing those shards
gains nothing. A smaller bf16/fp16 checkpoint may help; fp8/int8 still pays
the full-file bound below. Reset alone does not help because the estimate runs
after your next initial load. To render without proof, set Init's
`auto_gate=off` and accept its ordinary ungated status.

**Read where the refusal occurred:**

| Stage | Resident state and retry cost |
|---|---|
| Initial preflight | No shards unloaded or proof render run. The ledger records INCONCLUSIVE with `cross_mode: CAPACITY` and measured bytes; no verdict is cached. Free memory and requeue to re-estimate. |
| Reprice after baseline, or capacity refusal during proof | Shards have already been unloaded and a baseline may have run. Requeue repeats proof and its render cost. Only initial preflight preserves the starting residency. The reprice credits the baseline shards the cycle drops before it reloads, up to one share. It does not credit the baseline render's sampling reservations, though the cycle frees them too. A refusal there means the driver's text encoder, another resident model, or those sampling reservations leave too little. |

The least-free rank governs, after a credit for its resident shard pool. Every
load the proof prices follows an unload: the ceremony unloads your shards
before the baseline load, and the cycle unloads the baseline before the
reload. So the driver adds each rank's torch reserved memory, capped at one
share of the file, to its MemAvailable. The refusal and the driver log state
both figures: the GiB available and the GiB of resident shards the proof drops
first. An unknown world gets no credit. The price is the worker launch estimate:
known-world bf16/fp16 needs `(1/world + 0.08) * file + 5 GiB`; other precisions
or unknown world need `1.2 * file + 5 GiB` (DESIGN.md 5.4). The streaming build
commits shards plus one block; its world-2 0.58 factor exceeds the measured
0.547x high-water on a 60 GiB file. The Worker prices the same load again after
the real unload, and its guard stays decisive.

On 2026-10-13 a 60.02 GiB bf16 Flux2 proof refused at the initial preflight
with the driver's encoder resident: rank 0 read 39.1 GiB available, with its
30.0 GiB share resident, against 39.8 GiB. A 24.48 GiB bf16 Krea2 proof passed,
and both of its unloads returned the whole share to the host. With the credit
the Flux2 preflight reads 39.1 plus 29.9 GiB against 39.8 GiB. The Flux2 proof
has not been rerun on hardware with the credit
([VALIDATION.md](VALIDATION.md#fsdp-calibration)).

Proof does not replace the entire fleet to free memory: that would unload
existing residents and break the initial-preflight preservation promise. A
preflight refusal still unloads nothing.

**A worker refusal inside proof** can mean the estimate was unavailable
(discrete GPU, loader dtype cast, unmeasurable checkpoint, or missing rank
memory), that an unload returned less than the credit counted, or that an
initially fitting load was followed by another load while sampling
reservations remained. The driver logs every applied estimate, even
when it fits. Such ledger rows retain `cross_mode: CAPACITY` and classify the
refusal instead of recording driver-measured bytes. The estimate covers one
load, not proof's peak memory.

See [#5](#5-loads-fail--oom-on-uma-boxes-dgx-spark-etc) for ordinary stock-load
capacity and [#86](#86-a-render-error-says-the-fleet-stalled-or-a-worker-refuses-with-a-poisoned-cuda-context)
for a stall rather than a refusal.

## 89. Doctor's sageattention row fails its kernel probe, or counts fallbacks

**What happened.** `dgxm doctor` runs a real SageAttention
kernel on a small synthetic tensor (head_dim 128, then 64) in a bounded
subprocess of the same Python, whenever a CUDA device is present and no other
process holds the GPU. A FAIL on that row means the installed `sageattention`
package cannot run on this torch. The usual causes are a build from before a
torch bump (a stale ABI) and a PyPI wheel that shadowed the local build when a
custom node's requirements pulled it in. ComfyUI does not stop on either: it
logs `Error running sage attention` and samples on torch attention, slower.
That is what the second row counts from the driver log
(`~/comfyui-monarch.log`); lines that carry `Unsupported head_dim` are the
normal per-shape fallback and are not counted.

**What to do.** Rebuild sageattention against the installed torch in the same
venv (`"$COMFY_PYTHON" -m pip install --no-build-isolation --no-deps` on the
sageattention checkout, with `TORCH_CUDA_ARCH_LIST` naming this GPU's
architecture), then run doctor again: the OK row reads `kernel ran`.
Missing Python headers surface as a build failure and have their own row
(`python dev headers`). A `skipped: GPU busy` row is not a verdict: doctor
found a live process on the GPU and left it alone; run it again on an idle
box. A WARN with a fallback count names the log it read, and the count resets
when that log rotates.

## 90. `install-service` or `uninstall` refuses an existing Worker unit

**Symptom:** the command refuses and names the refusal it hit.
`REFUSED_FOREIGN_UNIT` reports that `dgxm-worker.service` is not this
lifecycle's canonical unit. `REFUSED_SETUP_UNIT` reports that guided setup wrote
the unit and points at `dgxm uninstall`. `REFUSED_FOREIGN_DROPIN` names the
files it found in a drop-in directory systemd reads for
`dgxm-worker.service`, each name written from your home directory down, or in
full for one systemd read from the runtime directory. None
of the three makes a source, generation, service, or unit change. Uninstall
runs that ownership proof, widened to accept the setup token, before it stops
a worker or clears a generation record, so a refused uninstall leaves the host
as it found it.

**The drop-in refusal.** systemd reads `dgxm-worker.service.d/*.conf` after
the unit. An override can replace `ExecStart=` while the unit body remains
canonical. `install-service` therefore refuses every file in those directories,
including editor backups, whether or not the unit exists. Remove the files and
rerun. Uninstall instead removes the owned unit and reports the overrides as
`left behind:`; a later install will refuse them.

**Six directories, not one.** The check reads `dgxm-worker.service.d` and
its dash-truncated equivalent, `dgxm-.service.d`, under each of
`~/.config/systemd/user`, `~/.local/share/systemd/user`, and
`$XDG_RUNTIME_DIR/systemd/user` (used by `systemctl --user edit --runtime`).
It excludes general `service.d` overrides, which affect every service, and
root-owned `/etc/systemd/user` and `/usr/lib/systemd/user`, which this user
lifecycle cannot remove. If the effective command differs from the unit body,
inspect `systemctl --user cat dgxm-worker.service` for all applied overrides.

**Cause:** standalone lifecycle commands mutate only an absent unit, a unit
carrying the current dgx-monarch managed marker, or the pre-marker canonical
unit for the current configuration. Two byte shapes count as pre-marker: the
current render, and that render as the previous release's installer wrote it,
one trailing newline longer. A marked unit counts as owned whatever its body,
so a config change can rewrite it. An unmarked manual edit, a different address
or interpreter, a symlink, a non-regular file, or an unrelated unit at that
name is foreign state. So is a unit whose body predates the current canonical
render, such as a hand-hardened copy. Only bytes this lifecycle can reproduce
prove ownership; a running process or listener cannot prove who owns the unit
bytes.

`install-service` also refuses a unit written by `dgxm setup`
(`REFUSED_SETUP_UNIT`): its first line carries the setup token that remote
source sync reads before it writes, and publishing the canonical body would
drop that token. `dgxm uninstall` does remove that unit, because uninstall is
the only teardown guided setup has. Guided setup will not take over a
pre-existing unit either, so removal comes first whichever installer you rerun.
The `~/.local/share/dgx-monarch/src` link guided setup published into its
release slot survives that removal. Source sync refuses a link without its
matching managed unit. Remove the link by hand on every host that
carried the setup unit, in the same pass.

**Fix:** inspect the unit privately with `systemctl --user cat
dgxm-worker.service` and compare it with the intended configuration. Back up
any manual unit. To retire it, stop and remove it yourself, then rerun
`dgxm install-service`; otherwise restore the managed unit or the configuration
that wrote it. Verified update can migrate recognized generated legacy and
hardened units after exact ownership checks. Other hand-edited units still
need manual retirement. Do not force the command past the
ownership refusal.

## 91. A Z-Image render refuses with an upstream gate, or takes the host down

**Symptom:** a Z-Image render stops before any load with a class P refusal
saying the checkpoint is the L2P pixel-space surface and this release carries
no L2P forward. It names no guard and offers no card and no waiver. On an older
build the same artifact does not refuse: the render loads on one box, the
worker journal reads `loaded partially; 0.00 MB usable`, the host runs out of
memory with no OOM kill, comfy's interrupt cannot reach the render, and the
worker has to be killed to keep the box alive.

**Cause:** the checkpoint is the Z-Image L2P pixel-space surface. Its header
carries `local_decoder` and no `dec_net` head. Current ComfyUI has no L2P model
contract, so it reads the header as latent Z-Image and gives that model a
`memory_usage_factor` of 2.8, which prices about 129 GiB of sample-time memory
on a box that holds 121. ComfyUI then offloads every weight, reports zero usable
bytes, and streams and casts every block on every step. Past that point this
release would bind the latent forward onto pixel-space weights, so the image
would be wrong even if the memory held.

**Fix:** use the latent Z-Image checkpoint or the DCT PixelSpace checkpoint;
both are hardware validated and neither is gated. There is no waiver and no env
var: no consent can supply a forward this release does not have. The refusal
retires when ComfyUI carries the L2P model contract (upstream draft
Comfy-Org/ComfyUI#14055) and this release carries the L2P forward. The driver
log carries one INFO line per file, opening `upstream gate:`, that names the
matched key paths, so a record can confirm which spelling the gate read.

The same gate refuses a checkpoint whose weight scales ComfyUI would drop; see
[#120](#120-a-render-refuses-because-comfyui-would-drop-the-checkpoints-weight-scales).

## 92. A render refuses with `FsdpGateProofError`, and the retry says the same thing

**What happened.** `auto_gate=first_use` on an fsdp topology runs a clean-reload
proof before your render. Something stopped that proof before it reached a
verdict. What you see depends on what stopped it:

- `FSDP clean-reload proof aborted before a terminal verdict: <type>: <text>`
  is an abort. Nothing answered. The text after the colon, cut at 200
  characters, names what stopped the proof; the driver log holds the whole
  exception.
- `FSDP clean-reload proof is incomplete: <reasons>` means the reload cycle ran
  but the ranks did not return the evidence a PASS needs; the reasons name the
  missing rank evidence.
- A message that opens with a class tag (`[dgxm:P`, `[dgxm:C` or `[dgxm:K`) is
  not an abort. A guard inside the proof answered your combination, and that
  answer stands: read its class and its remedy, not this entry.
- `[dgxm:K] the class K guard <id> refused inside the FSDP clean-reload proof`
  restates a waivable class K refusal. A check renders without accuracy
  waivers deliberately, so a waiver you already hold cannot be spent inside the
  proof, and granting one would return the same card forever. That refusal
  cannot be waived on this topology, for this shape. It keeps the guard's own
  remedy and entries, so a padding guard still sends you to token counts and
  topology and a sol-attn guard still sends you to an exact kernel.

**What the driver records.** Before any side effect, the check writes a durable
`RETESTING` row. Each ordinary stop closes it with an `INCONCLUSIVE` row
identifying the class and guard for a typed refusal, or the abort. The row
includes neither refusal prose nor remote tracebacks and authorizes nothing.
Both statuses deny execution; closing the transaction records completion
without changing that denial.

**What the retry does.** When a typed guard answered, the process-local
prejudgement the check published is withdrawn, so your next queue meets that
same guard and reads the same class. On an older build that
retry reads `automatic FSDP execution denied by the process-local clean-reload
Gate verdict INCONCLUSIVE` instead. The withdrawal holds for attempts queued one
after another. A second render already waiting on this combination when the
first one refused wakes to an untyped denial instead; queue it again and it
reads the guard.

When nothing answered, the denial stands: an aborted proof proved nothing, and
fail-closed means the combination stays denied until a proof completes. The
retry reads `automatic FSDP execution denied by the process-local clean-reload
Gate verdict INCONCLUSIVE`. Restart ComfyUI to clear it, or run the combination
with `auto_gate=off`.

A capacity stop taken during the proof, and the reprice the check runs between
the baseline render and the reload cycle, both happen after the check unloaded
your shards. The reprice, and any stop after the baseline render's own load,
also come after that render was paid. Re-queueing runs the proof again and pays
that cost again. Only the preflight stop (entry 88) leaves your residency
untouched.

**A check that was killed leaves its row open.** A ComfyUI interrupt or a
process kill writes no terminal row and keeps its denial, by design (entry 35):
the check settles only an ordinary exception, and a process kill leaves it no
chance to write. `dgxm gate --repair` finds those rows, and `--apply` closes
them:

```
dgxm gate --repair                 # list them; exits 1 when it finds any
dgxm gate --repair --apply         # close them; exits 0
```

The dry run prints one line per open row with its time, model, protocol and
package version, then a total, then a count of inert rows on its own line. An
inert row belongs to another gate protocol or another package build; it is
skipped, and a run that finds only inert rows exits 0, so read that line rather
than the exit code alone. `--apply` appends one terminal `INCONCLUSIVE` row per
open capability context and never rewrites, deletes or truncates a line. It
changes nothing about what is refused: an open `RETESTING` row already denies.
Run it at a driver boundary, with no render queued: it reads the open rows
first and appends after, so a check that passes in between would be
superseded by a repair row landing later and would have to prove itself again.

## 93. A multi-rank render refuses before any rank loads, and names a rank

**Symptom:** a world-2 render stops with a class C refusal that names one rank
and its numbers, says a rank did not answer within 30 s, or says a message to
a rank's Worker actor was undeliverable. No worker journal shows a load
starting and the refusal quarantines nothing. Except after an undeliverable
message, the fleet is still up and the next queue dispatches on the same fleet
with no recycle.

**Cause:** every rank prices the load before any rank allocates. The driver
asks each rank one bounded question on a read-only endpoint that takes no GPU
lock and mutates nothing, collects the rows, and decides once. The leanest
answer governs, because the load runs everywhere: a rank that cannot hold the
model refuses the render for the fleet rather than letting its peers load first
and meet the refusal sixty seconds in. Five shapes reach the operator:

* **A rank cannot fit.** The refusal names every refusing rank with its
  numbers, worst headroom first, and the fitting ranks with the residency they selected.
  A rank that answered on slab residency is named as such. When the hosts selected
  different residencies, one cause is one host having verified the family for slab residency
  while the other has not.
* **A rank did not answer.** A timeout at 30 s, a row count short of the world,
  two rows carrying one rank id, or ranks disagreeing about the checkpoint's
  own bytes all refuse the same way. This is not a capacity verdict: a rank
  that cannot answer a memory question in thirty seconds is starved or gone,
  and it has not said it cannot load anything. Nothing is torn down and no rank
  is evicted.
* **A rank's actor is gone.** When Monarch reports a message to a rank's
  Worker actor undeliverable while the price waits, which is what a Worker
  loop restart under the attached mesh produces, the wait ends within half a
  second of the report rather than at 30 s (about 2.5 s in when the first
  request is lost; see #2), and the refusal says the message was
  undeliverable. A sampler render whose setup is unchanged then reads the
  loops' ids and, when every loop restarted, attaches again by itself and
  renders; otherwise the refusal names the remedy
  ([#2](#2-every-attach-times-out-after-one-failed-attach)).
* **A rescue is on offer.** One rank needs slab residency and no rank refuses,
  so the ordinary consent card is raised before any rank loads, carrying the
  leanest rescuing rank's numbers.
* **World 1 is a no op.** The one box prices its own load.

Read the numbers with two limits in mind. Each rank prices against what it
holds at that moment, so a card can quote a figure taken with the previous
model still resident, and it says so. The quote does not price shared memory
either: a rank holding a weight slab in the other slot, or a slab retained by a
load that failed, reads as having more available than it has. The rank's row
records both (`shmem_bytes`, each slot's resident and the retained-slab count)
and draws no conclusion from them.

**Fix:** when the message names what would fit, start there. For a shortfall,
run the checkpoint slab-resident on every rank, use a pruned or quantized
artifact, or free unified memory on the short box. For a silent rank, recycle
the fleet from the DGX Monarch panel, or check that box, then queue again. For
an undeliverable one, follow the remedy the refusal names
([#2](#2-every-attach-times-out-after-one-failed-attach)). For
disagreeing bytes the two boxes are looking at different files: confirm the
checkpoint is the same artifact on both, then queue again.

**Estimate limits.** The agreement exposes every rank's answer before
allocation; it cannot establish that the estimates are correct. A load that
every rank prices as fitting is admitted, even when the price is too generous
and the box dies anyway. It prevents loading when a rank has already reported
insufficient capacity. The post-load checks still run: the partial-load guard,
the readiness exchange (entry 94) and the byte-verify certificate.

## 94. A multi-rank render refuses saying a peer rank could not load

**Symptom:** a world-2 render stops with a class P refusal saying a peer rank
could not load or verify the model, that this rank loaded cleanly, and that the
other box's journal names the cause. This box's journal shows its own load
finishing normally and then the line `rank readiness: this rank loaded`. The
other box's journal carries `load failed on this rank` and the real cause under
it. Nothing was sampled. The refusal arrives in seconds, not minutes, and the
fleet is still up.

**Cause:** every rank prices the load before any rank allocates (entry 93), but
memory moves between that quote and the load. A capacity check can still refuse on one box
afterwards: the allocation seam refuses, ComfyUI's own load fails, a file is
readable on one box and not the other, or the resident identity check does not
match. On every topology that puts the whole model on every rank, uly2, ring2,
dp2 and cfg2, the rank that failed and the rank that loaded cleanly cross one
readiness flag after the load and before the render's first collective. The
message on the healthy box is short on detail: the exchange carries agreement,
not identity, so this box does not know which peer failed or why. The class is
P because no consent on this box could change the answer here.

On an older build only dm-cfg2 crosses that flag; on uly2 and dp2 the healthy
rank walks into the partial-load guard, all-reduces alone, and waits for the
refused peer until the process group's timeout, ten minutes by default.

**Fix:** read the other box's worker journal. It prints the real cause with its
own refusal class, and that is the one to act on: a capacity refusal there names
what would fit. Then run the same graph again, or run it on a single box, where
there is no peer to disagree with. A rank that died rather than refused joins no
exchange, so a render that still ends in a group timeout is the dead-worker
case, not this one: see entries 9 and 87. One other case still ends that way.
The flag itself has to allocate, on the box that has just run out of memory, so
it can fail to cross; that box then prints `readiness flag not crossed` above
its own cause and this box waits out the group timeout. The cause on that box is
still the one to act on.

**Related:** entry 93 (the price every rank answers before allocation), entry 79
(the ranks disagree about weight residency, which is the check one step later,
after ComfyUI has placed the weights).

## 95. A sharded nvfp4 render refuses with a waiver card about activation scales

**Symptom:** a chroma nvfp4 checkpoint on uly2, ring2 or cfg2 refuses before the
first step with a class K message quoting a 1-step NRMS against a bf16 baseline
of 0.032 and a floor of 0.10. The message offers a waiver card and names
`DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`. The same checkpoint renders on one
GPU or on dp2 without complaint. Two texts arrive here. One says the render
already quantizes against one shared scale and still does not match a one-GPU
render. The other counts nvfp4 layers that would quantize against a scale each
rank takes from its own shard.

**Cause:** ComfyUI quantizes each activation with the module's `input_scale`,
and this checkpoint ships none. With no scale, nvfp4 alone falls back to a
statistic over the whole activation tensor, so the answer depends on which rows
the rank holds. dgx-monarch hands every rank one shared scale, which makes each
rank's rows identical to a one-GPU render's rows. That hook is on by
default and it is exact.

**It is not enough on chroma.** With the hook on, both ranks agreeing and 256
of 256 nvfp4 linears wrapped, a chroma nvfp4 uly2 render read 0.120 and cfg2
0.129 against 0.131 and 0.134 with the hook off. The rest of that gap has no
explanation yet, and it is not the kernel: bf16, fp8 and nvfp4 kernels return
identical rows whether a row is computed inside a full call or a half call. So
chroma nvfp4 refuses on every sharded topology, hook on or hook off, and only
the message changes. Krea2 read 0.011 with the hook on, under the floor, and is
not refused.

The second text means the hook declined a layer it would otherwise wrap. Two
ways in:

* `shared_act_scale = false` is set in worker_args, or `DGXM_SHARED_ACT_SCALE=0`
  on the workers, which is the off switch;
* the checkpoint carries `pre_quant_scale` on a layer with no `input_scale`, or
  it carries an expert bank. ComfyUI applies the AWQ smoothing before it takes
  the amax, so on a smoothed layer the statistic belongs to a product the hook
  never sees, and an expert bank quantizes with no scale at all, so one written
  for it would be dropped. The hook declines both.

A third gap prints neither text: a new comfy quantization layout whose
activation scale is a whole-tensor statistic, with no rule in the pack yet. The
hook neither wraps nor counts such a layer. The daily comfy canary turns red on
an unclassified layout before a render meets it.

**Fix, in order:**

1. Render on one GPU (`mode=local` with `gpus_per_host=1`) or a dp topology,
   where each rank sees the whole activation tensor and no scale is shard
   dependent.
2. Use an fp8, mxfp8 or int8 artifact of the same model. fp8 falls back to a
   constant, mxfp8 scales per 32-element block inside a row, and int8 does not
   quantize activations at all, so none of the three depends on the shard.
3. Render under the waiver, knowing the output is stamped. On chroma a waived
   render keeps the gap measured above. The grant is re-read on every render
   and its stamp rides each request, never the fleet's worker args, but the
   memo behind it stays live until you revoke it in the panel, so later renders
   of the same shape take it too (below). The headless spelling is
   `DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale`, and it grants on every render
   until it is unset. The guard is family scoped, so a grant for one family
   never silences another.
4. If the message tells you to turn the shared scale back on, set
   `shared_act_scale = true` in worker_args, or remove the key, and reload the
   model. The log line `shared activation scale: N/M nvfp4 linears wrapped`
   says whether the hook ran. On chroma this changes the message, not the
   outcome, but the hook removes a real term. A bare `DGXM_SHARED_ACT_SCALE=0`
   exported on a worker host is not removed by dropping the key; unset it on
   that host and restart the mesh.

**Setting the off switch on one box only.** `shared_act_scale` is a worker arg
so that one setting reaches every worker; the bare `DGXM_SHARED_ACT_SCALE` is a
single-box aid and an absent worker arg leaves it alone. The switch decides
whether a rank issues the shared-scale collective, so set it on every worker or
on none. With the variable set on one box of a pair, a chroma render with no
waiver refuses on both boxes, each with its own class K text: that box counts
the layers it declined and its peer names the shared scale. Read the journal on
both boxes. Under a waiver, or on a family the bar does not refuse, nothing
refuses here and the two boxes issue different collectives.

**Measuring it.** Set `log_act_scale = true` (or `DGXM_LOG_ACT_SCALE=1`) and read
one step: each rank prints its own local amax, the reduced amax and the scale per
wrapped layer. With the hook on, both ranks print the same reduced amax. With the
hook off the line still prints, with `reducers=none` and the rank's own amax in
both fields, which is how the before and after legs are compared. That trace showed the two ranks disagreeing in 338 of 548 sp-reduced layers, by
up to 6.2x, which is the term the shared scale removes. On chroma both legs need
`DGXM_WAIVE_KNOWN_WRONG=shard_quant_scale` to reach the render at all.

**A later render of the same model may not refuse at all.** A granted card is a
memo scoped to the combination, the topology label and the world, so once one
render at cfg2 has been waived, a later render of that checkpoint at cfg2 on
that world takes the same grant, renders, and is stamped. Read the ledger rather
than the absence of a refusal: a render that wrote a `use` row for
`shard_quant_scale:chroma` met the bar and was waived. A sweep run of four
chroma cfg2 cells saw this: one refusal and three carried grants
(docs/VALIDATION.md).

**Every cfg path meets the bar.** The check is read on the sample path before
the first collective, from the record the install wrote, so a render whose
conditionings run one per rank rather than as one batched call (entry 97) meets
it as a batched call does. The shared scale drops its cfg reducer on that
dispatched path, and on a padded pair whose original lengths one GPU would not
have batched: each rank then runs a whole conditioning of its own, and a max
across the cfg group would merge two amax a one-GPU render never merges. The
trace line names the reducers the call ran, so `reducers=sp` on a uly2+cfg2
render is one of those two paths, not a missing hook.

**Related:** entry 55 (the ring padding waiver, the same class and card shape),
entry 94 (the readiness exchange this bar is read inside).

## 96. A cfg2 render is no faster than the same render on one GPU

**Symptom:** a `cfg2` render completes, the image is right, and the warm
elapsed time matches a single-GPU render of the same graph. Both boxes show
full GPU load throughout. Nothing refuses and no gate complains. The sweep
harness scores such a cell CHECK with the line "cfg2 did not split".

**Cause:** the split was installed on a wrapper seam that family never reaches.
ComfyUI applies a `DIFFUSION_MODEL` wrapper nowhere central; each family's own
forward builds the executor over it, and a family that builds none (boogu,
ernie and omnigen2) leaves the wrapper installed and never called, so every
rank runs the whole cond and uncond batch. The output is correct, which is why
no identity gate and no fidelity floor can see it.

**Fix:** update dgx-monarch. It declares the seam per
family and puts those three families on `APPLY_MODEL`, the seam comfy applies
for every family. To confirm on your own graph, time the same render at `cfg2`
and as a one-GPU render (`mode=local` with `gpus_per_host=1`). In the
sweep on two DGX Sparks (docs/VALIDATION.md), for every cfg2
family whose split landed, the one-GPU reference took 1.7 to 2.4 times as long.
If you have added a family adapter of your own, run
`python tests/canary/comfy_seam_contracts.py`; the "cfg split" seam names any
family whose declaration and whose ComfyUI forward disagree.

**Related:** entry 97 (the same topology refusing instead of rendering, when
the model call reaches the split as a single row).

## 97. A cfg2 render refuses saying the model call does not split

**Symptom:** a `cfg2` render stops with a class P refusal saying cfg-parallel
got a model call of batch 1 which does not split into 2 equal slices. The card
describes the batched call ComfyUI builds and the sampler that decides it
exists.

**Cause:** ComfyUI folds the conditional and unconditional passes into one
model call only when it runs both passes. It runs both above CFG 1.0, and at
CFG 1.0 only under a sampler whose name carries `cfg_pp`, which keeps the
unconditional pass. Every other sampler drops that pass at CFG 1.0 and hands
the split a single row. That sampler case is most of what reaches this
refusal: a pair of conditionings that does not concatenate goes to the per-cond
dispatch one seam further out instead.

**Fix:** the card names it. Raise CFG above 1.0, or pick a `cfg_pp` sampler.
Where neither fits the workflow, use a ulysses or ring topology, as the card
says. Topology `auto` skips cfg-parallel rows at CFG 1.0 only when the sampler
drops the unconditional pass: any sampler without the `cfg_pp` suffix, and
every Sampler Custom (DGX Monarch) render, whose SAMPLER input carries no name.
Under a `cfg_pp` sampler set on a DGX Monarch KSampler node, auto can still
take a cfg row at CFG 1.0, because ComfyUI then folds both passes into one
call of batch 2. If such a render meets this refusal, ComfyUI ran the pair as
separate calls, for example because a conditioning carries an area, a mask, a
hook or a timestep range. Pick an explicit ulysses or ring topology for that
render.

**Unequal prompts render through the dispatch.** The per-cond
dispatch at the sampler's `CALC_COND_BATCH` seam installs for every family on a
cfg topology and decides per render by ComfyUI's own concat rule.
Each rank runs one whole conditioning, so no batched call is needed. Two prompts
of very different length on a family that pads nothing are the common case. A
family that publishes the real caption length as a `CONDConstant` takes the
same path: comfy compares a constant by value, so such a pair never folds, and
padding would add rows the constant does not count. Older builds without per-condition dispatch carried a
second card for that case, telling you to give the two prompts the same real
token count; that card is gone.

What still reaches this card besides the sampler case is a conditioning the
rule declines to read: an area, a mask, a gligen patch, a timestep window, a
hook group, or two conditionings on different control objects. There, use a
ulysses or ring topology, or write two prompts close enough in length that
ComfyUI batches them on its own.

**Related:** entry 96 (the same families before the seam fix, where the render
completed and only elapsed time showed that the topology did nothing).

## 98. A sharded render logs an attention kernel you did not select

**Symptom:** an Ideogram4 render on `auto`, `uly2` or `ring2` runs, and the
worker journal carries a line opening `USP attention kernel: TORCH_FLASH` that
goes on `TORCH_FLASH substituted for` the kernel the Init node's `attention`
widget names, `TORCH_CUDNN` or `SAGE_AUTO`. The render completes normally. On
`cfg2` the same graph logs no substitution.

**Cause:** a sharded topology routes attention through xFuser's ring layer into
yunchang's `pytorch_attn_forward`, and that function restricts torch to one
backend per kernel name. `TORCH_CUDNN` enables cuDNN alone, and cuDNN carries a
head dimension of at most 128. Ideogram4's transformer declares 256. A sage
kernel refuses the same head with `Unsupported head_dim`. So the runtime builds
the sharded attention on `TORCH_FLASH`, which carries up to 256, and says so in
the log.

It substitutes rather than refusing because nothing else runs those kernels on
this family either. ComfyUI orders its own SDPA backends by priority with all of
them enabled, so a cuDNN decline off the sharded path runs flash, and comfy's
sage wrapper catches the kernel's error and calls its pytorch attention. A
`cfg2` or single-box render of this family already computes flash whichever of
the three you select, and the sharded path matches it.

On an older build the render crashes instead: the model loads, the fleet comes
up, and the first attention call raises `RuntimeError: No available kernel. Aborting execution.` about forty seconds in, untyped.

**Fix:** no action is needed. If you want the log line to match the widget, select
`TORCH_FLASH`. If you are comparing kernels, note that this family has no cuDNN
or sage measurement to compare on any topology, sharded or not.

**When it refuses instead.** A head dimension no kernel this build ships can
carry, and a kernel that declines the live probe at load with nothing left to
substitute, both refuse class P at model load, on every rank, before the first
collective. There is no waiver and no consent: the kernel must support the
requested head width.

**Related:** entry 84 (the sol-attn kernel, which refuses its own geometry with
the same class and never substitutes, because no other kernel computes what it
computes).

## 99. A render refuses saying an acceptance fault was injected on that rank

**Symptom:** one box's worker journal carries a class P refusal reading
`acceptance fault injected on this rank`, quoted in the line `load failed on
this rank; refusing on every rank`. The other box refuses with the peer-load
message of entry 94. Nothing was sampled, the fleet is still up, and every
fresh load fails the same way until something changes. Higher up in that same
journal, from when the worker set up, sits a line reading `DEBUG LOAD FAULT
ARMED`.

**Cause:** a debug knob is set on that box's worker.
`DGXM_FAULT_LOAD_RANK=<rank>` makes the rank it names refuse inside the model
load, after every rank has priced the load and before any weight is read.
`DGXM_ACCEPTANCE=1` has to be set on the same worker or the knob is refused and
says in the journal that it stayed off. Both are read once per worker setup and
logged at WARNING then, so the arming line always sits above the refusal it
explains.

The knob tests the readiness exchange between pre-load agreement (entry 93)
and the first render collective (entry 94). External faults such as missing or
truncated files, held pages, or permission changes are normally caught by the
agreement before any checkpoint opens. Timing a permission change against the
load races a roughly one-second window and is not repeatable. This switch
injects the failure inside that window.

**Fix:** clear both variables from that worker's environment, wherever they were
set, and restart that worker. Nothing else on the box is wrong. The refusal
prices nothing, writes no gate verdict and quarantines no residency setting, and the box
renders as it did before once the variables are gone. If you did not set them,
read it as an acceptance run that leaked into a unit and check that unit.

**Related:** entry 94 (what the healthy box refuses with, and the exchange this
knob exercises), entry 93 (the agreement that catches every other injectable
fault first).

## 100. An FSDP render dies with "got mixed torch.Tensor and DTensor"

**Symptom:** the shard build finishes, both boxes report the model loaded
completely, and then every rank fails the render with

```
RuntimeError: aten.addmm.default got mixed torch.Tensor and DTensor, need to
convert all torch.Tensor to DTensor before calling distributed operators!
```

The last frames are `process_conds`, `encode_model_conds`, `extra_conds`, then
a model method with `text_embeds` or `prefix` in its name. Nothing is sampled.
The same render on one box, or under any resident topology, is fine.

**Cause:** ComfyUI runs `extra_conds` once per sampling, before the denoise
loop and outside the diffusion model's forward. Some families reach into the
diffusion model from there to refine the text embeddings, which means those
modules run before any FSDP2 pre-forward hook has gathered anything. If the
wrap sharded them their weights are DTensors while the text states are plain
tensors, and the first matrix multiply stops the render. FSDP replicates them
instead: they ride `ignored_params`, stay plain tensors on every rank, and
never enter a collective. Three families reach one, Anima, MiniMax H3 and
LTXAV, and the pack names all seven modules.

The cost is bytes, not speed. A rank holds those modules whole where a shard
would have been 1/world of them, 0.74 GiB a rank on the 61.7 GiB H3 file at
world 2 and 1.88 GiB on the LTX 2.5 bf16 transformer. The FSDP capacity line
in the worker journal states the count and the GiB as `auxiliary-module
parameters ignored`, so you can read what a rank replicated.

**Fix:** update dgx-monarch. There is no setting or workaround in the graph:
the decision is made once, inside the wrap, before the model is published. On
an older pack the render fails on every attempt of that family under FSDP, and
the way past it is a resident topology (drop `fsdp` from the preset) until the
pack is updated.

**Related:** entry 88 (the FSDP capacity refusal that stops a launch before
the shard build), entry 92 (an FSDP gate proof that refuses).

## 101. Reading the attach trace after a cluster attach times out

**Symptom:** a render refuses with `attach to workers [...] failed
(TimeoutError()); no further in-process retry runs` (older builds: `the
in-process retry already ran`), and every later render in
that ComfyUI process refuses at once with `this ComfyUI session's Monarch
transport is not reusable after a partial bring-up failure`. Both worker loops
are up, `dgxm doctor` reads clean, and a fresh driver attaches on its first
try. Known triggers include a Worker outage longer than the delivery timeout,
`dgxm restart` followed immediately by a render, and dead-rank recovery
releasing its latch. In the third case the attach stops after its first
attempt, and its refusal says that no in-process retry ran; its trace reads
differently (below). A refusal that starts `no attach ran` is the id gate in
[#2](#2-every-attach-times-out-after-one-failed-attach): nothing dialed, the
session stays usable, and its trace ends at `close outcome=refused
stage=proc_id`.

**What the driver writes.** Every cluster attach writes one line per phase
into the driver log, all sharing a sequence number. Grep the ComfyUI log:

```
grep attach_trace comfy.log
```

Every line reads `attach_trace event=<name> seq=<n> key=value ...`. The `seq`
counts attaches for the life of the driver process, so `grep 'seq=3 '` gives
one attach. Values never contain a space, so `cut`, `awk` and a second `grep`
work on them.

Every attach ends in one `close` line carrying an `outcome` and, unless it
attached, the `stage` it stopped at. Grep `event=close` rather than reading
the tail of the sequence, because a line can land after it: when auto-heal
is on and the attach failed for good (after its retry, or after a released
fleet's one attempt), the best-effort Worker restart for the next session
writes `post_fail_restart` after the close, and a render that attached again
after a Worker restart writes `restart_reattach` after it. A sequence with no
`close` is an attach still running, or a driver process that died during one.

**The events.**

| event | says |
|---|---|
| `open` | the loop addresses and their count, `client_bind`, the `attach_config_timeout` in force, whether this process had already bound the transport, whether a liveness proof released the fleet just before (`released_predecessor`), whether auto-heal is on, and `init_wait_s`, the bound on the mesh wait |
| `heal_probe` | which loops the pre-attach probe read as down, and which ones it never observed |
| `heal_restart` | the loops read as down, and the loops the restart touched (every configured loop) |
| `heal_settled` | whether worker health settled after that restart, and how long it took |
| `loop_generation` | one line per loop before every attach attempt: the generation `gen`, the process that holds the loop's LISTEN socket and how long that process has been up, the unit restart count, `proc_id`, the id that process serves and the attempt dials (or why there is none: `bare`, `unpublished`, `malformed`, `unreadable`, `none`, `unknown`), and `changed=true` when that process is not the one this driver read last time. A failed first attempt is followed by one more reading, `phase=before_attach_2`, even when no second attempt runs |
| `proc_id_moved` | the loops whose id changed between the dial of the first attempt and the reading after it failed: they restarted during that attempt, so the second attempt dials their new ids |
| `proc_id_prime` | the loops whose id changed since this driver last dialed them, sent one throwaway message before this `attempt` dials ([#2](#2-every-attach-times-out-after-one-failed-attach)), and any of them the message could not reach (`unprimed`) |
| `phase` | one timed step. `phase=transport_enable`, `phase=auto_heal`, or `phase=attach attempt=1..2`, each with `wall_s` and an `outcome` of `ok` or the exception class |
| `attach_call` | the two halves of one monarch attach: `half=attach_to_workers` is the worker config push, `half=initialized_get` is the wait on the mesh, bounded by its `bound_s` (70 s under the shipped 60 s config-push budget) |
| `close` | the terminal line, one per attach, with the total duration. `outcome=attached` is the good end. `outcome=poisoned` names the stage that poisoned this session: `transport_enable`; `attach_1_released`, a released fleet's one attempt failing; `attach_1_reread_refused`, an ordinary attach whose first attempt failed and whose reading after it found no id, so no second attempt ran (a released fleet in that case closes as `attach_1_released`); `attach_2_failed`, any other attach failing both attempts (older builds named it `listener_wait`); or `attach_1` or `attach_2`, the first or second attempt cut short by an exception that does not subclass `Exception`, such as `KeyboardInterrupt`, after which no retry runs. `outcome=refused` means no attach ran: `stage=transport_poisoned reason=earlier_attempt` for a session an earlier attach already poisoned, `stage=transport_bind reason=client_bind_mismatch` for a `client_bind` this process cannot switch to, and `stage=proc_id` for a loop whose id the first reading did not carry, which leaves the session usable. `outcome=failed stage=auto_heal` is a pre-attach heal that raised, so no attach ran; `stage=unexpected` is any other escape, carrying the exception class as `evidence` |
| `post_fail_restart` | the loops the post-failure Worker restart touched and the failure class that prompted it. It lands after the `close`, as `restart_reattach` does |
| `restart_reattach` | a render, or an explicit loader's eager load, that met Worker loops restarted under its mesh ([#2](#2-every-attach-times-out-after-one-failed-attach)). `outcome=attached` carries the replacement attach's `seq` and lands after its `close`, with `verdict=all`, the loops that `moved`, the `teardown` that retired the stale mesh (`stop_confirmed` or `liveness_release`) and the detour's `wall_s`. `outcome=refused` carries `seq=0`, since no attach ran, and a `reason`: `partial`, `none` or `unknown` for what the id reading found, or `not_retired` for a stale mesh the liveness proof did not release. `outcome=failed` carries the exception class as its `reason`: a replacement attach that raised, whose own `close` names the stage, or an explicit loader's capacity price that refused before that attach ran, which writes no new `close` |

**Read the attach timings first.** Each `phase=attach` line follows the `attach_call`
lines of its attempt, which split its duration between the config push and the
mesh wait. A `half=attach_to_workers` that took the entire duration is
a config push that never landed on a loop. A `half=initialized_get` that ran
to its `bound_s` and raised is a push that landed and a mesh that never came
up. Which one you see decides whether the next question is about the loop or
about the driver's own transport.

**Then read the generations.** Each reading is compared with the last one
this driver process took of that loop, in this attach or an earlier one.
`changed=true` says the process holding the loop's listener is not the one
this driver read last time: the loop restarted. With per-start ids that is
expected after a restart, and the reading's `proc_id` is the new id the
attach dials ([#2](#2-every-attach-times-out-after-one-failed-attach)). `changed=unknown` means there is nothing to compare: this
is the driver's first reading of that loop, or this reading did not come back.
`gen=none` means nothing holds the loop's LISTEN socket; that is a reading, so
a loop that goes from a generation to `none` reads `changed=true`.
`gen=unknown` with an `error=` field means the probe did not answer. That loop
is unread, not healthy and not moved: the line reads `changed=unknown`, and
`previous` keeps the last generation this driver did read, so the next reading
that comes back is still compared against something.

**What each of the three cases should look like.**

*After a Worker outage longer than the delivery timeout.* The restarted loop's
first `loop_generation` line reads `changed=true` with a new `proc_id`, and a
`proc_id_prime` line names it. Attempt 1 spends its whole window in
`half=attach_to_workers`: the config push never lands, because the driver's
channel to that address closed during the outage. The `before_attach_2`
reading shows the same `proc_id`. On a released fleet the `close` line names
`stage=attach_1_released`; on an ordinary attach, attempt 2 fails the same way
and the `close` names `stage=attach_2_failed`. Restart ComfyUI
([#2](#2-every-attach-times-out-after-one-failed-attach)). If the untouched loop also reads
`changed=true`, something moved a loop without a recorded request. The `heal_probe` line names the loops the probe found
down, and the `heal_restart` line names every configured loop, because the
restart is fleet-wide even when the probe found one loop down.

*After `dgxm restart` with a render posted straight away.* Compare `age_s` on
the `loop_generation` lines with the `wall_s` on the failing `attach_call`. A
loop a few seconds old under an attach that spends its window in
`half=attach_to_workers` says the handshake needs the loop to settle, and the
driver's attach window is absorbing a wait the start path should make.

*After the dead-rank recovery released its latch.* The `open` line carries
`released_predecessor=true`. With per-start ids the first attempt normally
attaches. When it fails, there is only one `phase=attach attempt=1` line, the
`before_attach_2` readings show the same `proc_id` per loop, and the `close`
line names `stage=attach_1_released`: no loop moved, so a second attempt with
the same ids cannot win, and the likely cause is an outage longer than the
delivery timeout. A `proc_id_moved` line instead means a loop restarted during
the attempt, and a second attempt runs with its new id.
The recovery is the restart path: both worker loops restarting together, then
a fresh driver process, which attached in 0.75 s in the recovery test
([docs/TROUBLESHOOTING.md #2](#2-every-attach-times-out-after-one-failed-attach)).

**The doctor row.** `dgxm doctor` prints one `listener generation` row per
loop, among that host's rows after its Worker service row:

```
  [ ok ] <host> listener generation: gen=<generation> pid=4242 age_s=612 restarts=0 started=<timestamp> marker=match proc_id=<16 hex digits>
```

`gen` is a short digest of the boot id, the process holding the LISTEN socket,
and that process start time. It changes exactly when that process changes and
carries nothing about the box it came from, so it is safe to paste into an
issue. `age_s` is how long that process has been up, `restarts` is the user
unit's `NRestarts`, and `marker` compares the process with the generation
marker that a start by `dgxm update` or `dgxm setup` writes. `marker=absent` is
the normal reading after a plain `dgxm up` or a hand `systemctl --user
restart`: neither writes the marker, and the unit clears it before every start.
`proc_id` is the id that process serves on this start, which the driver dials;
it normally starts with the 12 digits of `gen`. The row fails when a listening
loop definitely publishes no id (`bare`, `unpublished` or `malformed`),
because the driver refuses to attach to it, and names the remedy from
[#2](#2-every-attach-times-out-after-one-failed-attach). An id the probe could
not read (`unreadable`, `unknown`) warns as unobserved. Nothing else in the row moves doctor's exit code.

A host doctor could not read still gets its row:

```
  [WARN] <host> listener generation: unobserved: ssh failed: OSError; the live listener generation is unknown (docs/TROUBLESHOOTING.md #101)
  [WARN] <host> listener generation: unobserved: remote check exited 255; the live listener generation is unknown (docs/TROUBLESHOOTING.md #101)
```

The first is an ssh that raised, the second a remote check that exited
nonzero; the `<host> env` row just above each carries the fault itself.
Neither moves the exit code.

**Trace limits.** These lines record driver actions and external listener
state, not the worker's protocol health. A wedged loop can retain both its
process and LISTEN socket.

**Related:** entry 1 (the attach timeout ladder and `client_bind`), entry 2 (a
wedged Worker service), entry 87 (the dead-rank recovery whose latch release
leads to case 3).

## 102. A CogVideoX checkpoint reads family `unknown`, or ComfyUI will not load it

**Symptom:** on older builds, a CogVideoX 1.5 file in the
diffusers layout logs `no signature matched this checkpoint header; the
closest is cogvideo (1 of 2 key paths present, missing
time_embedding_linear_1.)`, auto topology gets no table row and no family
preflight, and a sweep marks every cell derived from the template that names
that file `skip:unknown-family`. A CogVideoX template pointed at a file in the
ComfyUI layout runs as usual. On current builds the same file reads
`cogvideo`, and the load can still fail inside ComfyUI with `Could not detect
model type`.

**Cause:** CogVideoX ships under two key spellings. The ComfyUI layout
flattens the embeddings and names its blocks `blocks.`, so a 1.5 image-to-video
file holds `time_embedding_linear_1.`, `ofs_embedding_linear_1.` and
`blocks.0.norm1.linear.weight`. A file straight out of diffusers keeps
`time_embedding.linear_1.`, `ofs_embedding.linear_1.` and
`transformer_blocks.`. Older builds matched only the ComfyUI
spelling. Both spellings now read `cogvideo`, but for the diffusers one that
is a name only: ComfyUI detects CogVideoX from
`blocks.0.norm1.linear.weight` and carries no diffusers conversion for this
family, so no loader can build that file.

**Fix:** read the header keys of the file the template names. Dotted embedding
names mean the diffusers layout. Convert the file by renaming its keys, then
point the template at the converted file. The rename is the whole conversion:
every tensor keeps its dtype, shape and bytes, and the key count stays the
same (1028 keys in the 1.5 image-to-video file, 1024 in text-to-video).
Rename each diffusers key by the first prefix below that matches it, where `N`
is the block index and does not change:

| Diffusers prefix | ComfyUI prefix |
|---|---|
| `transformer_blocks.N.attn1.to_q.` | `blocks.N.q.` |
| `transformer_blocks.N.attn1.to_k.` | `blocks.N.k.` |
| `transformer_blocks.N.attn1.to_v.` | `blocks.N.v.` |
| `transformer_blocks.N.attn1.to_out.0.` | `blocks.N.attn_out.` |
| `transformer_blocks.N.attn1.norm_q.` | `blocks.N.norm_q.` |
| `transformer_blocks.N.attn1.norm_k.` | `blocks.N.norm_k.` |
| `transformer_blocks.N.ff.net.0.proj.` | `blocks.N.ff_proj.` |
| `transformer_blocks.N.ff.net.2.` | `blocks.N.ff_out.` |
| `transformer_blocks.N.` (only the `norm1.` and `norm2.` keys remain) | `blocks.N.` |
| `time_embedding.linear_` | `time_embedding_linear_` |
| `ofs_embedding.linear_` | `ofs_embedding_linear_` |

Keys under `patch_embed.`, `norm_final.`, `norm_out.` and `proj_out.` keep
their names. Rename every key the table covers: a partial rename loads without
an error. ComfyUI detects CogVideoX from `blocks.0.norm1.linear.weight` alone
and loads the state dict with `strict=False`, so a weight left under its
diffusers name is never loaded, and only an `unet missing` warning in the log
says so. A converted file usually carries `converted_from` in its header
metadata, which is a quick way to tell two staged copies apart when both sit
under nearly the same name. A missing OFS pair is not a fault: it says the file
is text-to-video, since only the image-to-video and inpaint surfaces carry the
OFS embedding.

**Related:** entry 91 covers another case where a header identifies a family
but the release cannot render the file.

## 103. A slab-resident LoRA load refuses right before the bake, naming an exact GiB figure

**What happened.** Under `lora_low_rss`, a fresh slab load bakes the stack
through comfy's own full load before the slab reabsorbs the patched keys, and
every key comfy's patcher touches in between is a fresh allocation outside
the slab. `slab_load_fit` prices that transient from the LoRA and checkpoint
headers before either file loads. A target it cannot map onto a single base
key (a diffusers-named LoRA over a checkpoint that fuses q/k/v into one
weight, for instance) makes it charge the checkpoint's whole file size for
that stack, which errs high but is not exact (docs/VALIDATION.md).
This refusal is the exact check behind that estimate: once `ModelStore` has
built the real patch set (`active.patches`), it re-checks MemAvailable against
every patched key's own live byte count immediately before the bake runs, and
refuses typed instead of letting the kernel OOM-kill the worker.

**What the message says.** It names the exact duplicate size, the host floor
charged beside it, MemAvailable and the shortfall, under guard
`slab_lora_bake_preflight`. The base weights were loaded, but the bake did not
run and nothing was quarantined: the refusal lands before `_merge_and_free`
allocates anything.

**What helps.** Free unified memory that is not this render's own model
(another resident model, the driver's own text encoder, `dgxm reap`), use a
smaller or narrower-rank LoRA stack, or set `slab_weights=off`, which bakes the
same stack through the stock residency's load arena instead: larger, but
priced by the stock preflight.

**What it does not mean.** It is not evidence that the LoRA price is wrong in
general. The estimate errs high where it cannot map a target, and this backstop
is a second, exact check, not the primary capacity check.

**Related:** entry 5 covers the stock preflight. Unlike entry
53's rescue offers, this refusal is not waivable and has no panel action: slab
is already the residency, and no consent adds memory.

## 104. xFuser 0.7 CUDA setup imports a missing NPU helper

**Symptoms:** worker setup fails before model loading with an import error for
`yunchang.ring.utils.update_npu_out`. A plain `import xfuser` can still succeed.

**Cause:** the official xFuser 0.7.0 wheel imports an optional NPU ring
backend while it constructs CUDA attention, and released yunchang 0.6.4 lacks
the helper that backend imports.

**Action:** follow [INSTALL.md](INSTALL.md) in the existing ComfyUI environment
on every host. Its builder verifies the official wheel and produces the
temporary `0.7.0+dgxm.npuimport1` compatibility wheel, which makes the NPU
import lazy and leaves the NVIDIA attention math unchanged. A bare pip install
of this local version cannot resolve it from PyPI; pass the builder's output
with `--find-links` as INSTALL.md shows. Keep the selected CUDA Torch build,
and compare dependency versions across hosts before restarting while no render
runs. The temporary wheel stays until an official xFuser
and yunchang pair constructs CUDA attention without the local fix. It adds no
NPU support. The GPU import canary exercises the ring import that exposed this
failure; the two-host constructor and fidelity controls are recorded in
[VALIDATION.md](VALIDATION.md#dependency-and-setup-checks).

## 105. Nodes 2.0 shows nodes and links but no input widgets

**Symptoms:** the Fleet, Identity Gate or dual-Spark split template opens, but
input controls and Note text disappear while nodes and links remain. Opening
a new browser page can appear to help temporarily.

**Cause:** older copies of these templates hold a colon in their top-level
workflow `id`. ComfyUI frontend 1.53.6 keys widgets with colon-separated
fields, so that extra separator stops Nodes 2.0 registering stock and custom
widgets. The failure reproduces with every custom node disabled and one stock
node, without running a model (docs/VALIDATION.md).

**Action:** update dgx-monarch, restart ComfyUI as described in INSTALL, and
reopen the updated template from Browse Templates; the generator drops the
colon. To keep edits in an older saved workflow, remove colons only from its
top-level `id` string and reopen that file. Leave numeric node IDs, links and
widget values unchanged. The repair touches UI metadata only and changes no
sampling or model-residency behavior.

## 106. Sampling finishes but VAE decode runs out of memory

A finished distributed sampler can leave its DiT weights, LoRA backups and
allocator memory resident in the actors while ComfyUI loads the driver's VAE.
On unified memory those allocations compete with the decoder for the same RAM.

For the measured BF16 accelerated Wan Animate 2 recipe, use the BF16 warm
template: it passes the trimmed latent through Clear VRAM at `soft` before
native VAE decode. That frees unused allocator buffers without unloading the
DiT or LoRA backups, so the next prompt or seed reuses them. Its acceptance
runs used 8 GiB driver and Init reserves. A larger reserve can still refuse a
warm render; if you need one, keep it and choose a smaller quantization.

For more capacity at the cost of reloading weights, wire the final sampler or
latent trim into Clear VRAM's optional `samples` input, select `recycle`, set
`include_driver=false`, and wire its `samples` output to VAE Decode. The Wan
Animate 2 memory-saving template uses this order; the standard INT8 template
keeps its model loaded. The node returns the latent unchanged only after the
client-owned actors have retired; persistent Worker services and driver models
stay available. The next distributed render pays a fresh actor setup and model
load.

If the node reports that the recycle did not complete, it blocks decode. Let
active render and result leases finish, or inspect the reported lifecycle
failure before retrying; a busy or unknown outcome does not guarantee that any
memory was released. A Clear VRAM output node with nothing connected sets no
order between sampling and decode. `hard` alone does not guarantee that an
actor's retained allocator memory returns to the OS.

## 107. The same cluster fault repeats on a timer

**Symptoms:** the driver log shows one `cluster fault` ERROR whose body starts
`undeliverable message to cast.`, then lines such as `cluster fault repeated 8
times`. Nobody is rendering. Older builds printed the whole fault
and a browser toast every 30 seconds for as long as ComfyUI stayed up.

**Cause:** the driver renews a client lease against its worker fleet every 30
seconds ([#51](#51-a-worker-actor-outlives-its-client-and-holds-tens-of-gib)). The renewal is a broadcast, so a fleet that cannot receive it
answers later, as an unhandled fault, and the renewer never sees a failure it
could back off from. Two states do it. In one, the fleet stopped receiving
(it died or was restarted, for example) while the driver sat idle, and the
driver still holds its handle as live. In the other, the fleet latched after a
worker death while a failed render's abandoned lease remains; its renewals
fault once the worker services no longer know the fleet, for example after
they restart.

**Action:** the first report indicates an unreachable fleet.
Queue again or reset the Attached mesh: the driver then finds out what happened
to the fleet and either replaces it or prints what blocks it. The repeat lines
count one fault at 2, 4, 8 and every later power of two; they are not new
faults, and a different fault is always printed whole. For a fleet latched
after a worker death the driver stops renewing by itself and says so once:
`client lease renewals stopped for a fleet latched after a worker death`. An
actor process that survived such a latch then exits at the lease grace, which
lets the next attach prove the fleet gone; with actor reaping disabled it
stays until its Worker service restarts. The measured record is in
[VALIDATION.md](VALIDATION.md#lifecycle).

## 108. cuDNN Ring refuses its native normalization check

**Symptoms:** a Ring render with `TORCH_CUDNN` selected refuses before
sampling with a native LSE, output-contract or processor-binding error. A peer
may report that another rank could not load or verify the model (entry 94).

**Cause:** combining Ring key/value blocks needs each block's real
log-sum-exp (LSE) normalization; a normalized attention output with
placeholder zero LSE values cannot reproduce full attention. The cuDNN Ring
processor calls the private operator `aten._scaled_dot_product_cudnn_attention`
with `compute_log_sumexp` set, and checks a small native output and LSE example
before the rank-readiness exchange. The check fails when the installed Torch
lacks that operator or returns another result layout, when the installed
xFuser or yunchang does not keep the custom processor, or when the native
result is wrong on one worker. It runs after the final model and head binding,
including a warm kernel switch. A request the head-dimension rule already
moved to Flash does not run it. On a supported Ring topology a failing check
enters the same readiness exchange as a load failure, so a healthy peer
refuses before its first Ring communication.

**Action:** keep the failing worker's exact error and compare the installed
Torch, CUDA and dependency versions on both hosts. Make any update through the
normal verified workflow while no render runs. Do not replace LSE with zeros,
loosen the comparison tolerance, or treat a passing CPU stub test as native
cuDNN validation. Switching to another supported attention kernel is a
separate configuration and needs its own reference comparison. Ring with FSDP
still refuses.


## 109. pip check reports cuSPARSELt is not supported on this platform

**Symptom:** on Linux ARM64, pip 24.2 and later report:

```text
nvidia-cusparselt-cu13 0.8.1 is not supported on this platform
```

**Cause:** the official wheel's filename says `manylinux2014_aarch64`, but
its internal `WHEEL` tag is `py3-none-manylinux2014_sbsa`. The internal tag is
invalid. Installation selects the wheel by its filename; pip 24.2 and later
check the recorded tag. Pip 24.0 does not perform that check.

The defect affects the cu13 ARM64 wheels for 0.8.0 and 0.8.1. Versions 0.9.0,
0.9.1 and 0.10.0 have the correct tag, and the x86_64 wheels are unaffected.
PyPI files cannot be replaced, so 0.8.1 cannot be corrected in place. PyTorch
still pins 0.8.1 and links against its library. Installing 0.9.x would create
a dependency conflict. The tag error does not affect imports or renders.

**Action:** run the repository's dependency checker using the ComfyUI Python
and repository path selected in [INSTALL](INSTALL.md#install-from-source):

```bash
"$COMFY_PYTHON" "$REPO_DIR/tools/check_dependencies.py"
```

The checker runs `python -m pip check` with that same interpreter. A successful
check with no errors passes silently. It accepts the line above only when it
is the sole error, the host is Linux aarch64, and the installed
`nvidia-cusparselt-cu13` version is 0.8.1 with the invalid internal tag.
It downloads the [official wheel](https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl),
checks SHA-256
`4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`,
and compares every installed wheel member except `RECORD`, which pip rewrites.
This requires network access. No package or metadata is changed.

After verification, the tool prints one line identifying a known upstream
platform-tag failure, its verified wheel hash and [issue #5](https://github.com/Deen-Media/dgx-monarch/issues/5).
Save that line with the installation record. A download failure, mismatched
file or hash, different platform, or any additional dependency error fails
the check and preserves the full pip output. Resolve the failure before
continuing; do not downgrade pip or change the working Torch/NVIDIA packages
to bypass it. Continue with the required imports, Doctor and saved distributed
render after the dependency check passes.

The remaining upstream blocker is PyTorch's dependency pin. Watch the
cuSPARSELt entry in PyTorch's
[`generate_binary_build_matrix.py`](https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py).
Remove the allowlist entry only after the pinned Torch build depends on
`nvidia-cusparselt-cu13>=0.9.0` and a fresh ARM64 environment passes dependency
checks, imports and the ARM64 CI job. A corrected NVIDIA wheel alone does not
meet that condition.


## 110. Verified update refuses color-matcher test files

An older updater can report `the driver's torchmonarch does not match the
current checkout's exact pin` even when the pinned files are intact. The
underlying ownership error distinguishes a real file conflict from an
unsupported package layout.

The official color-matcher 0.6.0 wheel includes three test images both inside
`site-packages/tests/data` and through the wheel's external data directory.
Pip also records compiled bytecode for its installed CLI script. These are
valid [installed-package records](https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file),
but older Monarch verifiers reject their path shapes. The wheel also adds an
empty `tests/__init__.py`, which turns the shared test namespace into a regular
Python package. That affects the bundled test namespace; it is not an overlap
with a pinned TorchMonarch file.

The compatibility check recognizes only the known external records for
color-matcher 0.6.0, binds its three in-tree images to their official hashes, and verifies
its uniquely owned, empty test initializer. It never reads or changes external
record targets. Every pinned TorchMonarch file must still match; changed
payloads, duplicate claims, nonempty initializers and runtime-package shadows
remain errors. A successful ownership check does not prove the rest of an
update or a render will pass.

### Recover an installation with an older verifier

Keep the original checkout, environment and Worker sources unchanged. Save the
failed update receipt and close ComfyUI. Use a separate clean checkout at the
reviewed fix commit to run the same verified update against the original
installation:

```bash
REPO_DIR=/absolute/path/to/the/installed/dgx-monarch
COMFY_PYTHON=/absolute/path/to/the/existing/environment/bin/python
CONFIG_PATH=/absolute/path/to/cluster.toml
FIX_COMMIT=REPLACE_WITH_REVIEWED_FULL_COMMIT
CONTROLLER_DIR="$(mktemp -d /tmp/dgxm-update-controller.XXXXXXXX)"
git clone --no-checkout "$(git -C "$REPO_DIR" remote get-url origin)" "$CONTROLLER_DIR"
git -C "$CONTROLLER_DIR" checkout --detach "$FIX_COMMIT"
"$COMFY_PYTHON" -I -B "$CONTROLLER_DIR/tools/update_from_checkout.py" \
  --repo "$REPO_DIR" --config "$CONFIG_PATH" --target "$FIX_COMMIT"
```

Replace the placeholders with discovered installation paths and the reviewed
commit before running the commands. The recovery tool requires matching Git
origins, a clean controller at the exact selected commit, and an unchanged
original checkout. It preserves the normal activity and ownership checks,
confirmation, release verification, receipts and rollback. The original
checkout advances only after the verified update succeeds. Keep the controller
checkout and recovery records if the outcome is incomplete.

Do not reinstall TorchMonarch repeatedly, delete package test files, edit
`RECORD`, or suppress a distribution to get past this error. If the check
reports another conflict, preserve it for diagnosis. Older Ultralytics wheels
that claim actual TorchMonarch files are a separate collision and remain
rejected.

## 111. dgxm is missing from an ordinary terminal or launcher installation refuses

Installing the package places `dgxm` in the selected Python environment's
`bin` directory. A full-path command working while bare `dgxm` is missing is
a PATH issue. Activate that environment, use the full executable path, or
install the optional managed launcher following [INSTALL](INSTALL.md#launch-and-handoff).
The TUI extra must also be installed through the documented constrained
dependency procedure.

`dgxm launcher --check` plans without writing; `--apply` installs only the
local launcher. An identical rerun preserves it. Unmanaged commands, edited
launchers, symlink targets or parents, and unsafe directory permissions are
refused. Inspect the existing destination and choose another location rather
than replacing an unrelated command. A held launcher lock requires the other
installer to finish; it is not a reason to delete its lock file.

In the ordinary terminal, check `command -v dgxm` and `dgxm version`; use
`hash -r` to clear Bash's cached command lookup. An alias, function or earlier
PATH entry can still select a different command. The installer reports its
own PATH view and does not edit shell startup files. A missing saved config
refuses; reinstall the managed launcher for the new config or pass a leading
`--config` override. For the dashboard, separately verify the ComfyUI endpoint
and pass `top --host HOST:PORT` when discovery does not select the intended
driver. A CLI launcher does not start ComfyUI or Worker services.

## 112. The torchmonarch canary reports semantic drift against a new release

**Symptom:** `torchmonarch-canary` fails its final compatibility gate while the
checked repository pin passes. This is separate from `comfy-canary`, which checks
ComfyUI compatibility. The dependency canary installs the candidate only in a
scratch environment; it does not change the repository pin or running Workers.

Read the Actions summary and the `torchmonarch-surface-<run_id>` artifact.
`checked-pin-report.md` compares the installed pin with the committed baseline;
`latest-vs-pin-report.md` compares the latest stable release with that pin.
The final gate names failed stages, and installation logs remain in the artifact.

The 2026-10-12 alert compared the passing 0.6.0 pin with
[Monarch 0.7.0](https://github.com/meta-pytorch/monarch/releases/tag/v0.7.0), which
removed `Future(coro=...)` construction and the `actor_queue_dispatch` setting.
The former was used only by this canary, so its failure did not demonstrate a
production Future failure. The probe now obtains a Future from the public
channel API before testing `.get()` inside an asyncio loop. Queue dispatch is
now unconditional upstream, but the removed setting still requires review.

On 2026-10-12 the development pin moved to 0.7.0, with a regenerated ARM64 schema-2 baseline.
The canary uses Torch 2.12.0+cu132 and additionally requires a native RDMA import,
an empty native action and a separate `cpu-smoke` run. The smoke checks local
actors, concurrent dispatch, terminal timeout cancellation for decorated
Futures, re-observable endpoint timeouts and lazy ProcMesh shutdown. Since
2026-10-13 it also times out a ProcMesh stop at once and requires that stop to
stay cancelled and its procs to answer until a fresh stop. Neither
stage contacts configured Worker services or exercises a GPU/fabric.

Complete the
[pin-bump procedure](../skills/dgx-monarch-dev/SKILL.md#torchmonarch-bump-procedure-63)
before deployment. Review the candidate's shutdown-timeout cancellation and immediately started
RDMA buffer drops as well as dispatch behavior; unchanged signatures do not
prove those ownership contracts. The review for the current pin, dated
2026-10-13, is in
[VALIDATION](VALIDATION.md#ownership-and-actor-start-review-2026-10-13).
Do not refresh the baseline or suppress drift
just to clear the alert. CPU surface checks do not establish hardware or native
RDMA acceptance.

## 113. Lifecycle commands refuse with POLICY_STALE after a policy edit

**Symptom:** after a change to any `[monarch]` setting other than
`startup_timeout_s`, or to `cluster.rdma_latent_return`, on an installation that
guided setup or a verified update wrote, `dgxm restart`, `dgxm up` and
auto-heal refuse with `POLICY_STALE` and point here. `dgxm doctor` fails the
host's `worker policy` row with `STALE`. The Worker services keep running:
restart checks every host before it stops any of them, unless a host's check
goes unanswered.

**Cause:** each Worker unit carries the policy resolved from cluster.toml as
`--runtime-policy` arguments, plus `--native-rdma` when native return is on.
Only guided setup and a verified update write a managed unit, so editing the
file leaves the installed unit with the earlier policy. A unit whose only
difference from the generated unit is a well-formed policy other than the
current one reads `POLICY_STALE`, and lifecycle commands refuse to start it. A
unit with no policy arguments counts as current only while cluster.toml selects
the default policy with native return off. Any other difference from the
generated unit, including a malformed policy argument, still reads
`FAILED_SOURCE_LAYOUT`.

The same source check names three other failures, none of them a stale
policy. `dgxm restart`, a syncing `dgxm up` and the `dgxm doctor` env row
print them with the host, the exit code and, when there is one, the last
stderr line:
`FAILED_WORKER_INTERPRETER` (exit 126 or 127: the shell could not run
`cluster.python` on that host), `FAILED_LIFECYCLE_LOCK` (the lifecycle lock
wrapper failed before the check gave a verdict, for example on an unsafe
`~/.local/state/dgx-monarch`) and `FAILED_SOURCE_PROBE` (the check exited
without a verdict). Doctor runs the check without the lock, so it never
reports `FAILED_LIFECYCLE_LOCK`; an exit without a verdict reads
`FAILED_SOURCE_PROBE` there. Fix the cause the line names on that
host, then run the command again; the remedy below is for `POLICY_STALE` only.

**Remedy:** stop ComfyUI, then run a verified update to the commit the driver
checkout already has:

```bash
dgxm update --verify --target HEAD
```

The update stages a fresh release of that commit, stops the Workers, writes
units rendered from the current cluster.toml, starts them, and runs Doctor and
the cluster smoke. A check that definitely fails restores the earlier units
and release. The
usual [verified-update conditions](INSTALL.md#updating) apply: a clean
checkout, a reachable `origin`, and no ComfyUI, Render session or actor
running. A transport-policy change needs a fresh driver process in any case.
To back out instead, restore the earlier settings; the installed unit then
matches again. If config loading now refuses the earlier value, as after an
upgrade that tightened `[monarch]` validation, fix the value in cluster.toml,
then run the command above to re-render the unit. The update, like the probe,
checks the unit's policy argument for its keys, value types and canonical
encoding but not its values, so it also re-renders a unit whose policy value
someone edited by hand. A hand edit that breaks that form, or that changes any
other line, reads `FAILED_SOURCE_LAYOUT`, and the update refuses that unit
with `prior_worker_drift`.

A verified update run from a checkout older than Worker policy arguments
writes units without them. Those units are current under the default policy
with native return off. Under any other policy, the new release's Doctor step
fails the `worker policy` row and the update restores the prior release. Run
that update with `rdma_latent_return = false`, then restore the setting and
run the command above.

## 114. Every combination runs its first-use Gate again after an update

**Symptom:** after deploying the 2026-10-12 change that narrowed the Gate's
configuration fingerprint, the first render of each model, residency and LoRA
combination runs the Identity Gate again, although `cluster.toml` is unchanged.

This is expected, once per combination. Deploying any new source already
re-proves every combination (the note at the top of this page), so the
narrowed fingerprint changes nothing you can see at that deploy. Let each
combination prove itself once; nothing needs clearing. The narrowing matters
only for later `cluster.toml` edits: editing a setting that
[DESIGN.md](DESIGN.md) §5.2 lists as outside the fingerprint keeps PASS rows,
and every other edit still needs a fresh proof.

## 115. A render fails with `NativeReturnLatchedError` and later renders return by messaging

**Symptom:** with `rdma_latent_return = true`, a render fails after the workers
finish, and its error names native RDMA latent return and the latch. The driver
log carries one WARNING saying native return is latched off for this ComfyUI
process. Later renders return their latents by actor messaging, so large
latents come back slower but complete. `/dgxm/telemetry` shows
`native_return_latch` beside `return_ownership`, with `latched: true`, the
failed read's exception type as `reason`, `latched_unix_s`, `fault_count`,
`last_fault_unix_s` and `clears_on: driver_restart`.

**Cause:** a native latent read failed on the driver. Any failure of a native
read on the driver trips the latch: a queue-pair initialization timeout, a
registration or read failure, a read timeout, a policy readback refusal, a
failed ACK, or a refusal before the read starts, including a read on a lease
that already retired. Ownership settles as
[#29](#29-large-rdma-latent-return-fails-after-workers-finish-rendering)
describes: nothing is dropped or acknowledged without proof, a started read
keeps its destination, and the failed render's lease stays abandoned. Before
the latch, the recycle and retry built fresh actors that registered natively
again, so the same fault repeated on every render and kept one more latent
each time. Now the driver records the fault, and every later setup asks the
actors for messaging. The return mode is part of each actor's setup identity,
so the fresh fleet sets up in messaging mode. The latch changes only what the
next setup requests; it never frees, drops or acknowledges anything.

**What to do:** the next normal render or Fleet queue resets the attached mesh
once when the abandoned lease is its only obstruction
([#87](#87-a-log-line-says-the-abandoned-lease-wedge-healed-and-the-render-paid-a-respawn)),
so a Fleet re-queue after a native fault now heals on its own. When something
else also blocks, or the reset cannot confirm its ProcMesh stop, Fleet still
refuses with `LifecycleBusyError` (`... reset the Attached mesh first`).
Follow the recovery the refusal names, or run **Reset attached mesh**, then
queue again.
On 2026-10-13, before Fleet ran this reset, a failed Fleet wave on the paired
Sparks left two abandoned samples and one poisoned RDMA owner, and the next
Fleet queue refused this way; after a manual reset the next one completed by
messaging, with the latch and the poisoned owner still in place. Diagnose
the native path from
the original error, which the typed error carries as its cause:
[#75](#75-native-rdma-fails-at-qp-rtr-because-the-two-ends-selected-different-rails)
covers rail selection. A Fleet job fails the same way, with the same typed
error.

**Clearing:** only a ComfyUI restart clears the latch. Neither **Reset attached
mesh** nor the automatic reset clears it: both stop the actors through the same
recycle, the driver still holds any retained destination, and a recycle does
not show that the fault went away. Fix the fault, then restart ComfyUI; the
next setup requests native return again. To stay on messaging across restarts,
set `rdma_latent_return = false`; on a managed installation, follow
[#113](#113-lifecycle-commands-refuse-with-policy_stale-after-a-policy-edit).

**Held latents:** each native read that failed after it started holds its
destination latent in the driver process until ComfyUI exits. The latch changes
only what the next setup requests, so work already dispatched on the native
fleet when the fault hit still reads its latents natively: the rest of a Fleet
wave, or the other renders in flight under `pipeline_depth`. Each of those
reads that fails after it starts holds one more. Fleet keeps at most one job
per GPU in flight and stops after the failed wave, and the pipeline sampler
keeps at most `pipeline_depth` renders in flight and stops after a failure, so
one fault holds at most one wave or one pipeline window of latents. There is
deliberately no per-process cap and no operator release for them. Native
return is opt-in (`rdma_latent_return` defaults to false), the latch sends
every later render to messaging, and the count is bounded, so one ComfyUI
restart frees them all.

## 116. An Anima render refuses because ComfyUI passed an argument to the attention op

**Symptom:** an Anima render on a sequence-parallel topology (ring, uly or
hybrid) refuses with a class P message: `anima self-attention: ComfyUI passed
'<name>' to the attention op dgx-monarch installs, and this release does not
apply that argument under sequence parallelism.`

**Cause:** dgx-monarch replaces the attention op on each Anima self-attention
block, and ComfyUI's cosmos attention calls that op directly. The op accepts
the arguments ComfyUI passes today: `transformer_options`, the per-block
`preferred_attention` and q, k and v in single-use containers. A newer
ComfyUI passed one more. The sharded attention cannot apply an argument it
does not know, and dropping it could change the output, so the render stops
before any attention runs. Nothing is wrong with the checkpoint, the fleet or
the graph.

**What to do:** run the workflow on topology `single` (mode=local with
gpus_per_host=1) for a stock one-GPU render, or run a ComfyUI build this
release supports. Report the argument name: a dgx-monarch release has to
adopt it, either applying it under sequence parallelism or refusing it for a
stated reason. No consent or waiver bypasses this refusal.

## 117. A render refuses because the checkpoint or a LoRA pins an attention kernel

The refusal reads "this model pins an approximate kernel on its attention
modules" and carries the Sol-Attn waiver card. ComfyUI lets a checkpoint, or a
LoRA with `.comfy_attention.config` keys, pin an attention kernel on each
attention module. It ships two such kernels, `comfy_kitchen_sol` (sparse
attention, with a tau) and `comfy_kitchen_int8` (q, k, v and the attention
weights rounded to 8 bits), and both are approximate by construction. On a
topology without sequence parallelism (on a two-rank pair: single, cfg2, dp2,
cfg2+fsdp and dp2+fsdp) ComfyUI runs the pinned kernel as it does on one GPU,
so dgx-monarch refuses before any sampling unless the operator has granted the
waiver. The message names the kernels, the number of attention modules and
the topology; [ADAPTERS.md](ADAPTERS.md) has the contract. On cfg2+fsdp and
dp2+fsdp an fp8 or int8 checkpoint keeps its pins, and they ask too. A bf16 or
fp16 checkpoint whose own header pins never gets that far: its uint8 config
tensors leave the FSDP launch check unable to prove the file's precision, so
every `+fsdp` preset refuses it class P with "The detector could not prove the
checkpoint's precision". Render that file on a topology without fsdp; uly2
drops the pins.

The FLUX.3 Image Klein INT8 ConvRot files are the first shipped checkpoints
that pin a kernel: comfy-kitchen's INT8 attention on 33 attention modules, the
five image blocks and the 28 single blocks. Their testing graphs ask for the
waiver on single, cfg2 and dp2 and render without it on uly2, which drops the
pins and runs exact attention, so a uly2 render differs from stock INT8 on one
GPU by construction. The BF16 and NVFP4 files pin nothing, and the shipped
FLUX.3 templates use BF16.

The MOSS-Image INT8 ConvRot files, the SFT, Base and Kromoss checkpoints, pin
comfy-kitchen's INT8 attention on all 42 blocks and on neither text-embedder
block. Their testing graphs ask for the waiver on single, cfg2 and dp2 and
render without it on uly2, which drops the pins and runs exact attention, so
a uly2 render differs from stock INT8 on one GPU by construction. The BF16
files pin nothing, and the shipped MOSS-Image templates use BF16.

**What to do:** either accept the card and queue the render again, or pick a
sequence-parallel topology such as uly2, where dgx-monarch drops every pin and
the Init widget's kernel runs (exact under TORCH_FLASH; the Sol card appears
there only if the widget selects Sol). The card grants the Sol-Attn waiver
for this checkpoint, topology and world; headless, set
`DGXM_WAIVE_KNOWN_WRONG=sol_attn` on the driver or follow the consent recipe
in [#84](#84-a-sol-attn-render-refuses-with-a-waiver-card-or-stalls-on-its-first-call).
Under the waiver each rank whose pins would run logs KNOWN-WRONG MATH
PROCEEDING UNDER WAIVER, dgx-monarch stamps the output rendered-under-waiver,
and the gate ledger gets a use row under the guard `sol_attn:checkpoint_pin`.
Removing the LoRA that carries the pin also clears the refusal. Under
dual-model cfg2 the card names the conditional checkpoint even when only the
unconditional one pins, because the driver keys every grant of that render on
the conditional checkpoint; the card's detail line names the checkpoint that
pins.

**On cfg2+fsdp and dp2+fsdp under `auto_gate=first_use` the grant is not
enough.** FSDP has no stock residency to fall back to, so such a render needs
a Gate PASS, and the first-use ceremony that would earn one renders without
waivers and ends on this guard. A pin that runs there (a LoRA's, or an fp8 or
int8 checkpoint's) therefore refuses even after the grant. Render it on
uly2+fsdp, where dgx-monarch drops the pin, with `auto_gate=off`
([#67](#67-what-auto_gateoff-costs-and-what-it-does-not)), or on a topology
without fsdp. On a resident topology such as cfg2 the ceremony ends on this
guard too and changes no residency setting. Under `auto_gate=first_use` every
new combination there runs that ceremony before it renders, because the
default `slab_weights=auto` makes it risky, so a checkpoint that pins pays
for it even with no LoRA. After the ceremony the combination renders
under stock residency, and that answer stands for the rest of the ComfyUI
session: the grant does not run the ceremony again, and each new session
runs it once more. To keep the slab and low-RSS levers, grant the card and
set `auto_gate=off` for that graph
([#66](#66-the-gate-says-inconclusive-on-a-graph-with-no-lora),
[#67](#67-what-auto_gateoff-costs-and-what-it-does-not)).

**Reading the record.** The `render.last` block of `/dgxm/telemetry` and
`benchmark/run_matrix.py` result rows carry `attention_pins` (pinned attention
modules per kernel), `attention_pins_ran` and `attention_pins_dropped`
(modules per reason). The sidebar's last-render line, the run_matrix console
and the driver log print one line, for example "render pinned:
comfy_kitchen_sol tau=0.7 x 40 modules, ran". Under dual-model cfg2 each cfg
rank renders its own checkpoint, so the three fields add the two models'
modules. When both models pin, `attention_pins_models` carries each model's
own fields under `cond` and `uncond`, and the line gives one clause per
model, for example "pinned: cond model: comfy_kitchen_sol tau=0.7 x 20
modules, ran; uncond model: comfy_kitchen_sol tau=0.7 x 10 modules, dropped
(kernel unavailable on this host)". The count is per attention
module, so a block whose self- and cross-attention are both pinned counts
twice. "ran" means ComfyUI dispatched each pinned module to its pinned kernel,
as on one GPU; the kernel itself still runs exact attention on a masked call
(Sol) or on a float32 call with `low_precision_attention` off (int8). A pin
does not run for one of three reasons: `sequence parallelism`; `FSDP shard
build`, for a pin the shard build of a bf16 or fp16 checkpoint cleared, though
no checkpoint whose own header pins reaches that build today, because the FSDP
launch check refuses it class P first; and `kernel unavailable on this host`,
where ComfyUI could not bind the kernel and runs its default backend.
A render on more than one rank still refuses for an unbound pin, because every
rank must reach the same verdict and a peer may bind the kernel; a render on
one GPU asks nothing for it. The `attention` fact keeps naming the Init
widget's kernel.

**Probing it on hardware without patching a checkpoint.**
`tools/make_pin_probe_lora.py` reads a checkpoint's safetensors header, finds
one family's attention modules from their query projection keys, and writes a
small LoRA that holds only `diffusion_model.<module>.comfy_attention.config`
tensors:

```bash
python tools/make_pin_probe_lora.py \
  ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors \
  ComfyUI/models/loras/pin_probe_wan21_sol.safetensors \
  --family wan21 --kernel comfy_kitchen_sol --tau 0.7
```

It knows krea2, anima, kandinsky5, wan21, qwen_image21, flux3, flux3_early,
moss_image and moss_image_text, and its docstring lists the key pattern for
each. `flux3` pins every FLUX.3 block; `flux3_early` pins only the five text
and five reference blocks, which every rank runs whole before the sharded
blocks. `moss_image` pins every MOSS-Image block and both text-embedder
blocks; `moss_image_text` pins only the two text-embedder blocks, which every
rank runs whole on ComfyUI's default attention. It pins
self-attention by default. `--cross`
adds cross-attention, for int8 only: Sol needs q, k and v of one shape with a
head dimension of 128, and cross-attention gives its keys and values the
text's length, not the image's. `--blocks N` pins only the first N modules.
Load the file with a LoRA loader at strength 1.0, since strength 0 stages no
pin. On cfg2 under `auto_gate=first_use` the first queue runs the first-use
ceremony above, then shows the card; after the grant expect a stamped render
on stock residency whose record reads "ran". On uly2 expect no card and a
record that reads "dropped (sequence parallelism)"; on cfg2+fsdp or dp2+fsdp
under `auto_gate=first_use` expect the refusal above even after the grant. The file
patches no weight, and slab residency prices it as no bake.

## 118. Workers in release slots still run the previous version after a plain update

**Symptom:** after a plain `dgxm update`, one of these appears:

- `dgxm doctor` fails a host's `worker release version` row: the host "runs
  dgx-monarch X from release slot ..., but the driver runs Y";
- a host's `env` row ends in `version skew vs driver pin`;
- `dgxm update --verify` refuses with `prior_worker_drift` ("worker hosts do
  not exactly match the current driver release");
- the first-use Gate refuses with "Gate pre provenance package source does not
  match driver";
- with a driver on 1.2.0 or later and a slot on a release before 1.2.0, a
  render refuses before dialing with `no attach ran: ... its loop runs an older
  release that serves the bare address` ([#2](#2-every-attach-times-out-after-one-failed-attach)),
  and `dgxm doctor` fails that host's `listener generation` row.

From 1.2.0 on, plain update refuses this case before it fetches, with
`release_slot_worker`, and names the host. It refuses with
`release_slot_unknown` when it cannot read a host's source link: the host gave
no SSH answer, or the Python that cluster.toml names failed there. Nothing was
fetched; fix that host, then rerun the update.

**Cause:** guided setup puts each remote Worker in a release slot, and a
verified update puts every Worker in one: the host's
`~/.local/share/dgx-monarch/src` is then a link into
`~/.local/share/dgx-monarch/releases/`. Plain update pulls the driver's
checkout, installs the new torchmonarch pin, and restarts the Workers, but it
never changes a release slot, so those Workers keep the earlier package. A
slot that a verified update wrote also carries its own torchmonarch, which
the env row then reports as skew. A slot that guided setup wrote holds only
the package, so its Worker runs the earlier package on the new torchmonarch
and only the release version row and the Gate name it. The verified update
then refuses, because it accepts only Workers that match the driver exactly.
A guided-setup install that ever ran plain update has been in this state
since that update.

Check each host first. A link into `releases/` means a slot; `u-` slots come
from a verified update and `s-` slots from guided setup:

```bash
readlink -f ~/.local/share/dgx-monarch/src
```

Both routes below use `REPO_DIR` and `COMFY_PYTHON`, the paths from
[INSTALL.md](INSTALL.md#install-the-node-pack). To see which release the
driver runs:

```bash
"$COMFY_PYTHON" -c 'import dgx_monarch; print(dgx_monarch.__version__)'
```

**Repair forward** (works for every slot): move the Workers onto plain source
directories at the driver's version, then return them to release slots. It
works with a driver on this release or still on 1.1.0; steps 5 and 6 say where
the two differ.

1. Stop ComfyUI.
2. Remove the Worker units and keep the config. Answer `N` when uninstall asks
   whether to remove the config file:

   ```bash
   dgxm uninstall
   ```

   Ignore its closing hint to remove the `custom_nodes/dgx-monarch` checkout;
   the next steps run from it.
3. On every configured host, the driver's own host included, remove the source
   link. Uninstall leaves it in place, and both installers refuse a link
   without its unit. The test leaves a plain directory alone:

   ```bash
   if [ -L ~/.local/share/dgx-monarch/src ]; then rm ~/.local/share/dgx-monarch/src; fi
   ```

4. Install the driver's torchmonarch pin on every configured host. A plain
   Worker reads torchmonarch from the interpreter that cluster.toml names, and
   a plain update that stopped early, as on "unsupported key(s): monarch",
   never moved it. Read the pin from the driver's checkout:

   ```bash
   PIN=$(grep -o 'torchmonarch==[0-9A-Za-z.+-]*' "$REPO_DIR/pyproject.toml")
   echo "$PIN"
   ```

   Then on each host install that pin, with the version `echo` printed after
   `torchmonarch==` and the `python` path from cluster.toml's `[cluster]`
   table. Where the pin is already installed, pip changes nothing:

   ```bash
   <python from cluster.toml> -m pip install --no-deps torchmonarch==<version>
   ```

5. Install plain Worker units. This syncs the driver's package to each remote
   host and starts the services (the
   [Worker services](INSTALL.md#worker-services) section covers lingering and
   the alternative of rerunning guided setup):

   ```bash
   dgxm install-service
   dgxm doctor
   ```

   Every `worker release version` row should now read "plain source directory".
   A 1.1.0 doctor prints neither that row nor a `worker policy` row; check
   instead that no host's `env` row ends in `version skew vs driver pin`.
6. Put the Workers back in release slots. From now on, update this install
   only with `dgxm update --verify`. On a driver that runs this release, verify
   the commit it runs:

   ```bash
   dgxm update --verify --target HEAD
   ```

   On a driver still on 1.1.0, the Workers now match it, so skip
   `--target HEAD`: read
   [Upgrading from 1.1.0](INSTALL.md#upgrading-from-110), then run
   `dgxm update --verify` to update to the new release in one step.

   The usual [verified-update conditions](INSTALL.md#updating) apply: a clean
   checkout, a reachable `origin`, and no ComfyUI, Render session or actor
   running. Where protected process environments are unreadable, start the
   [read-only process inspectors](INSTALL.md#protected-process-inspection) and
   add `--privileged-process-inspection`. The 1.1.0 updater has no such flag;
   where it refuses with `activity_unknown`, run the update with
   [the recovery procedure in #110](#recover-an-installation-with-an-older-verifier)
   and that flag, as step 5 of the rollback below says. If it refuses because
   a candidate dependency such as `xxhash` is missing, install it as INSTALL
   describes, then rerun it.

**Roll back** (for `u-` slots, or an `s-` slot whose commit you know): return
the driver to the commit its Workers run, then make the verified update from
there.

1. Stop ComfyUI. Remove any `[monarch]` table from cluster.toml: the earlier
   release's config loader rejects it.
2. Find the Workers' commit. A `u-` slot name carries its first 12 hex digits
   (`u-<commit>-<random>`); an `s-` slot carries none, so use the commit guided
   setup ran at. Reset the checkout's branch to it, which refuses rather than
   drop local changes:

   ```bash
   git -C "$REPO_DIR" reset --keep <commit>
   ```

3. Reinstall that commit's torchmonarch pin on the driver, without
   dependencies:

   ```bash
   PIN=$(grep -o 'torchmonarch==[0-9A-Za-z.+-]*' "$REPO_DIR/pyproject.toml")
   echo "$PIN"
   "$COMFY_PYTHON" -m pip install --no-deps "$PIN"
   ```

   Run the same install on each Worker host too, with the value `echo`
   printed and the `python` path from cluster.toml's `[cluster]` table in
   place of `COMFY_PYTHON`. The plain update moved that interpreter's
   torchmonarch as well: an `s-` slot runs on it, and the earlier release's
   Doctor reads it even for a `u-` slot, which carries its own copy.
4. Reinstall the package from the reset checkout, restart the Workers and
   check them:

   ```bash
   "$COMFY_PYTHON" -m pip install -e "$REPO_DIR" --no-deps
   dgxm restart
   dgxm doctor
   ```

5. Update with the verified path only, after reading
   [Upgrading from 1.1.0](INSTALL.md#upgrading-from-110) when the Workers run
   that release:

   ```bash
   dgxm update --verify
   ```

   The earlier release's updater runs this step. If it refuses with
   `activity_unknown` because protected process environments are unreadable,
   that updater has no `--privileged-process-inspection`; run the same update
   from a checkout of the target release with
   [the recovery procedure in #110](#recover-an-installation-with-an-older-verifier),
   adding that flag to `tools/update_from_checkout.py`, with the inspectors
   running.

If that update still refuses with `prior_worker_drift`, the Workers run a
different commit from the one you reset to, as after a guided setup followed
by plain updates. Use the forward repair instead.

## 119. A mesh build or Worker start refuses on an ambient environment variable

**Symptom:** a render's mesh build, or a Worker service start, refuses with
one of these messages, each ending in a pointer to this entry:

- "ambient OTLP endpoint conflicts with disabled diagnostic export";
- "unmanaged MONARCH_RDMA_ environment setting", followed by the names it
  found;
- "`NAME` conflicts with the validated Monarch policy", for example
  `DGXM_RDMA_QP_SPLIT`.

A Worker that refuses exits at start, and its systemd unit keeps restarting
it; on that host, `journalctl --user -u dgxm-worker.service` shows the
message.

**Cause:** dgx-monarch sets Monarch's transport and diagnostic settings from
cluster.toml's `[monarch]` table before Monarch loads, in the driver, each
Worker and each actor. A variable already set in the environment would
override those settings without any record, so the process refuses it
instead. The driver checks on every mesh build, in local mode too; with no
cluster.toml the default policy applies, so a single box with no cluster
config can see this. The Worker loop drops
every variable outside a short allowlist before it checks, so of these only
`DGXM_RDMA_QP_SPLIT` can stop a Worker start; the other two come from the
environment that starts ComfyUI.

**Remedy**, per variable:

- `OTEL_EXPORTER_OTLP_ENDPOINT` with export off: unset it in the environment
  that starts ComfyUI (the shell, its profile, or the launcher), then restart
  ComfyUI. Setting `[monarch.diagnostics] otlp_endpoint` to the same URL with
  `otlp_logs = true` also clears it, but that turns on export of Monarch's
  upstream logs as well as metrics; choose it only on purpose
  ([CLUSTER.md](CLUSTER.md#monarchdiagnostics)).
- A `MONARCH_RDMA_*` name the policy does not set: unset it. `[monarch.rdma]`
  sets every RDMA option this release supports
  ([CLUSTER.md](CLUSTER.md#monarchrdma)). A managed `MONARCH_RDMA_*` name set
  to the policy's own value is accepted; any other value refuses with the
  conflict message below.
- `DGXM_RDMA_QP_SPLIT`, or another variable named in the conflict message:
  unset it where the process starts. The policy sets each of them;
  `DGXM_RDMA_QP_SPLIT` follows `[monarch.rdma] application_chunks`. For a
  Worker, look in the systemd user manager's environment
  (`systemctl --user show-environment`), in any hand edit to its unit, and in
  the shell environment `dgxm up` starts a `nohup` Worker from.

## 120. A render refuses because ComfyUI would drop the checkpoint's weight scales

**Symptom:** a render stops before any load with a class P refusal saying the
checkpoint stores quantized layers whose weight scale ComfyUI does not read.
The message gives the number of such layers and the first one. It names no
guard and offers no card and no waiver. The driver log carries one INFO line
per file, opening `upstream gate:`, that names up to four of the layers.
Stock ComfyUI loads an fp8 or int8 file of this kind and logs a
`unet unexpected:` warning that lists the dropped `weight_scale` or
`scale_weight` keys.

**Cause:** ComfyUI learns how a layer is quantized from a
`<layer>.comfy_quant` record. The record is a tensor in the file, an entry in
the `layers` table of the header's `_quantization_metadata`, or one that
ComfyUI's legacy conversion writes: in a file with a `scaled_fp8` tensor and
no `_quantization_metadata`, it renames each `<layer>.scale_weight` to
`weight_scale` and records that layer, and no other. ComfyUI's
diffusion-model load also strips one key prefix, such as
`model.diffusion_model.`, and drops every key outside it. So a `comfy_quant`
tensor must sit under the same prefix as its weight, and a
`_quantization_metadata` entry must name the layer with no prefix or with
exactly that one. A layer that stores an fp8, fp4, int8 or packed 4-bit
(`U8`) weight beside a scale with no record loads into a plain Linear, and
ComfyUI drops the scale as an unexpected key. That includes a `weight_scale`
in a `scaled_fp8` file, and a `scale_weight` in a file that has
`_quantization_metadata`. An fp8 or int8 file then loads without an error and
computes with its scales ignored, so the image is wrong. A packed 4-bit file
fails at load or at the first step. The gate refuses an fp4 layer by the same
header rule. The known cases are in Black Forest Labs' FLUX.3 Image Klein
diffusers repositories: `variants/fp8r/model.safetensors` stores per-row fp32
scales and names its format only in header fields ComfyUI does not read, and
`variants/nvfp4/model.safetensors` stores 198 packed 4-bit layers with scales
and no record, so stock ComfyUI fails at the first step.
The gate reads the header alone, on the driver before capacity or mesh work
and again at every Worker's load entry point, and replays ComfyUI's load
steps on the key names.

**Fix:** use an export of the same model in ComfyUI's quantized layout, whose
quantized layers carry `comfy_quant` keys or a `_quantization_metadata` table,
or the model's bf16 file. For FLUX.3 Image Klein, use the files in the
vendor's ComfyUI-layout repository, `black-forest-labs/flux-3-image-klein-comfy`,
which the artifact manifest pins ([MODELS](MODELS.md#model-flux-3-klein)).
There is no waiver and no env var: no setting can make ComfyUI read these
scales.

## 121. A FLUX.3 checkpoint fails to load on ComfyUI older than 0.41.0

**Symptom:** a FLUX.3 Image Klein template is refused before it runs:
ComfyUI's prompt check reports `Value not in list` for the CLIPLoader type
`flux3`. A graph that loads a `flux3_image_klein_*` file another way stops at
the model load, where ComfyUI's loader raises `Could not detect model type`,
while dgx-monarch's own header read still names the family `flux3`. A ComfyUI
that loads the file as some other model class leaves the render refused with
`has no dgx-monarch adapter`.

**Cause:** ComfyUI added the FLUX.3 model, its detection, the `flux3` text
encoder type and the LoRA key map in one change, first released in
ComfyUI 0.41.0. dgx-monarch reads the family from the checkpoint header
without asking ComfyUI, so it names `flux3` on any ComfyUI, but only a
ComfyUI that defines `comfy.model_base.Flux3` can build the model, and the
adapter binds that exact class. The driver needs it for the `flux3`
CLIPLoader type too.

**Fix:** update ComfyUI to 0.41.0 or newer on the driver and on every Worker,
through the procedure you use for ComfyUI, then restart the driver and the
Workers. `dgxm update` moves dgx-monarch and torchmonarch only, never
ComfyUI. Every other family keeps working on the older ComfyUI meanwhile.
Every Gate PASS goes stale with the ComfyUI change, so each combination runs
its first-use Gate again
([#114](#114-every-combination-runs-its-first-use-gate-again-after-an-update)).

## 122. Studio refuses a stale review or cannot display a complete library catalog

**Symptom:** Studio reports `schema_changed`, `operation_consumed`, or an
oversized library catalog. A cancelled request may remain unresolved after a
connection loss.

**Cause:** generation previews bind the installed driver schema. A management
review authorizes one execution. Cancellation records one durable intent per
job before contacting the driver. Catalog selectors also refuse to load more
than 8 MiB of saved records rather than silently select from an incomplete list.

**Fix:** after `schema_changed`, inspect and review the draft again. For
`operation_consumed`, open the existing operation receipt; obtain a fresh review
only when another execution is intended. Reconnect to reconcile uncertain jobs
or cancellations without changing their request keys. The companion never
repeats a cancellation simply because its acknowledgement was lost. Open the
Project library to browse records in bounded windows when a selector exceeds
its catalog budget. Full job provenance and operation receipts load when opened
or exported; compact list rows do not mean those records were deleted.

If startup reports `driver_identity_mismatch`, use the original driver for that
state directory, or choose a separate `--state-dir` for the new driver. Existing
unbound draft databases bind on their first upgraded start; select the original
origin then. If a preview reports a 4 MiB expansion limit, reduce the template or
workflow payload. Grid expansion also has a combined 32 MiB budget. A rejected
expansion does not partially save its cell previews.

An unreadable saved request blocks its operation, grid or library import
controls. Inspect the durable records and actual outcome before using the
explicit clear action; removing an uncertain identity can otherwise permit a
duplicate request. A recent-history warning does not mean the driver queue is
offline: Studio continues displaying observed queue state and resolving known
prompt identities while reducing oversized history windows.

## 123. Studio refuses to start, or refuses the browser's host, origin or token

**Symptom:** `dgxm studio` or `dgxm-studio` exits at startup, or the page or
an API call returns one of these refusals:

- `Unrecognized companion host.` (HTTP 403, code `host`). Opening
  `http://localhost:8220` shows only this JSON error.
- `Cross-origin access is forbidden.` (HTTP 403, code `origin`)
- `Enter this companion's owner token.` (HTTP 401, code `authentication`)
- `A companion already owns this state directory.`
- `State directories cannot be symlinks.`
- `State directory must belong to this user with mode 0700.`
- `State database must be an owner-private regular file.`
- `Unsupported storage schema; export with the matching companion.`
- `XDG_STATE_HOME must be an absolute path; unset it or pass an absolute --state-dir.`
- `The owner token file must be in an existing directory.`
- `The owner token file directory must belong to this user with mode 0700 and must not be a symlink.`
- `The owner token file cannot replace a companion state file.`
- `The owner token file path holds another file; choose a new path.`
- `Another process already listens on the companion port; stop it, or choose another port: [studio] port in cluster.toml, or --port for dgxm-studio.`
- `The sign-in page in the state directory holds another file; remove it, then start the companion again.`
- `Cursor must be below 2^63.` (HTTP 400, code `cursor`)
- `The companion could not complete this request; its log records the cause.`
  (HTTP 500, code `internal`)

**Cause:** the companion checks every request, and its state directory at
startup, before it does anything else
([STUDIO.md](STUDIO.md#install-and-start),
[threat model](THREAT_MODEL.md#browser-to-studio-companion)).

- Host: the request's `Host` header must be exactly the address and port the
  companion bound and printed: `127.0.0.1:8220` by default, or `[::1]:8220`
  for an IPv6 bind. `localhost:8220` reaches the same socket but is refused,
  and so is a tunnel or proxy that rewrites `Host` or listens on a different
  port. `--host` itself takes only a loopback IP literal, never a name.
- Origin: a request that carries an `Origin` header must name the same
  scheme, host and port as the request itself. A page from any other origin,
  the ComfyUI driver's page included, is refused.
- Token: every `/api/` request needs `Authorization: Bearer <owner token>`.
  The companion makes a new token at every launch, so a token from an earlier
  launch, or from another companion, no longer works.
- Lock: one companion holds an exclusive lock on `companion.lock` in its
  state directory for as long as it runs. A second companion on the same
  directory exits.
- Directory: no part of the state directory path may be a symbolic link, and
  the directory must belong to the user running the companion and grant no
  group or other access (mode `0700`). Its database, `studio.sqlite3`, must be
  a regular file with the same owner and no group or other access.
- Schema: the database records its storage schema. A companion that reads a
  schema other than its own refuses to start rather than change the records.
- State root: an empty `XDG_STATE_HOME` counts as unset. A relative
  `XDG_STATE_HOME` is refused, and so is a relative `--state-dir` or
  `--token-file` (`--state-dir must be an absolute path`), because a relative
  root would open a second database from another launch directory.
- Token file: when standard output is not a terminal, the companion writes
  the token to `owner-token` in its state directory. `--token-file` names
  another file. Its directory must exist, belong to the user and grant no
  group or other access, and the companion replaces only a file that holds an
  earlier token.
- Port: the companion binds its port before it prints anything, so no token
  exists yet when another process holds the port, and it writes no token file
  or sign-in page.
- Sign-in page: the companion writes `sign-in.html` (mode `0600`) in its state
  directory when the token goes to a file or when it sends the page to a
  browser, and replaces only a page it wrote earlier. Any other file, a
  symbolic link or a file with group or other access under that name refuses.
- Driver origin: each state directory is bound to the first driver origin it
  served. A launch that asks for another origin exits with
  `driver_identity_mismatch` and two more lines: the origin the directory is
  bound to and the one this launch asked for, then an `mv` command that moves
  the directory aside under a name that does not exist yet.
- Cursor and errors: an event cursor must be below 2^63, and every other
  failure returns the JSON error envelope with code `internal`; the companion
  log records its cause.

**Fix:**

- Open the exact sign-in link the companion printed, or run
  `dgxm studio open`, and do not swap `127.0.0.1` for `localhost`. Through an
  SSH tunnel, forward the same port on both ends
  (`ssh -L 127.0.0.1:<port>:127.0.0.1:<port>`) and browse `http://127.0.0.1:<port>`;
  `dgxm studio tunnel <host>` prints the SSH line and the address to browse.
- After a token refusal, enter the token this launch made: the one it printed,
  or the contents of the token file it named, or open its sign-in link.
  A restart makes a new token, so reconnect every open tab after one;
  `dgxm studio open` prints the current link.
- For a port refusal, stop the other process, set `[studio] port` in
  cluster.toml, or pass `--port` to `dgxm-studio`;
  `ss -ltnp 'sport = :8220'` names the program that holds it.
- For a sign-in page refusal, look at the file named `sign-in.html` in the
  state directory, remove it if it is not yours to keep, and start the
  companion again.
- For a driver origin refusal, start Studio with the origin the directory is
  bound to, or run the printed `mv` command so the next start makes a fresh
  state directory for the new driver. To bring the old records along, export
  the library from the old directory and import it into the new one.
- For the lock refusal, stop the other companion, or give this one its own
  `--state-dir`. Do not delete `companion.lock`; the running companion still
  owns the directory, and a second one would share its records.
- For a symlink refusal, pass the real directory path with `--state-dir`. For
  a mode refusal, check the owner and run `chmod 700` on the directory, or
  `chmod 600` on its database; never loosen either.
- For a schema refusal, start the companion that wrote the directory, export
  the library, then import the export with the new companion into a fresh
  `--state-dir`.
- For a state root refusal, unset `XDG_STATE_HOME` or give it an absolute
  path, and pass absolute `--state-dir` and `--token-file` paths.
- For a token file refusal, choose a new path in a directory you own with
  `chmod 700`; never point `--token-file` at another file you keep.
- For an `internal` refusal, read the companion's log; the response holds no
  exception text.

## 124. Studio refuses a submission, retry, upload, cancellation, operation or update

**Symptom:** a Studio review, submission, retry, upload, cancellation,
management review or update returns one of these codes, or a job settles with
one of these reasons:

- Submission and retry: `choice_unavailable` (HTTP 409, "<node>.<input>: the
  driver no longer offers the selected option; review a fresh preview."),
  `schema_changed` (409), `preview_expired` (409, "Preview expired; review a
  fresh submission."), `grid_preview` (409), `lora_binding` (400, "LoRA slot
  <n> (<node>.<input>): <reason>") and `workflow` (400, for example "<input>
  uses COMFY_DYNAMICGROUP_V3, which needs an explicit adapter").
- Job outcomes: an error `prompt_rejected` ("Driver refused the prompt before
  queueing it (HTTP <status>)."), with the failing nodes; a settlement reason
  `absent_from_driver` ("The driver no longer holds this prompt in its queue or
  history."); or a `node_errors` warning on a job that stays queued.
- Uploads and media: `image_limit` (413, "Use a PNG, JPEG or WebP image." or
  "Downscale the image to at most 16 megapixels (4096 x 4096 pixels) and try
  again."), `asset_limit` (413, "The normalized image exceeds the 65 MiB
  backstop. ...") and `thumbnail_limit` (413, "Source exceeds the 65 MiB
  thumbnail backstop.").
- Limits: `request_limit` (413, "Request body exceeds the 4 MiB companion
  limit."), `driver_response_limit` (502, "Driver response from <endpoint>
  exceeds its 4 MiB budget.", or 16 MiB for `/object_info`),
  `record_window_limit` (413, "Saving this record would take the <kind> catalog
  past the 8 MiB browser window. ..." or "Importing this library would take the
  <kind> catalog past the 8 MiB browser window, so no records were imported.
  ...") and `library_limit` (413, for an export past 4 MiB or 2,000 rows of one
  collection).
- Cancellation: `cancellation_scope` (409, "This driver has no
  prompt-specific cancellation route; only queued graph tasks can be
  cancelled.").
- Operations: `operation_unavailable` (409, "Restart the companion with
  --allow-management to write a diagnostics file.", or "Studio is stopping or
  restarting for dgxm studio up, down or token --rotate, so it accepted
  nothing; try again once Studio answers."), `operation_fields` (400,
  "Enter both the pending consent key and id, or leave both empty to list
  pending requests."), `operation_path` (400, "Management paths must be
  absolute."), `consent_state` (502), `gate_sampler` and `gate_history` (409),
  and `driver_service_identity` (409, "The selected driver service serves
  <scheme> on <addresses> port <port>, not the --driver origin; select the unit
  that runs that driver.").
- Update: `update_config`, `update_driver_service`, `update_repository`,
  `update_controller_source`, `update_release_slot`,
  `update_release_slot_unknown` and `update_changed` (all 409).
- Driver state: `driver.code` reads `studio_storage`, `driver_data`,
  `studio_internal` or `observer_failed`, or the queue reads `over_budget`.

**Cause:** a review binds to the node definitions its prompt uses, and
submission and retry check each selected file or option against the live
driver again. A 4xx from POST `/prompt` is a definite refusal, so the job fails
with the driver's node, class, input and type for each error. A job is absent
when a successful queue snapshot lacks it and its own history lookup returns no
entry, which follows a driver restart or a history clear. The upload rule
bounds decoded pixels, not the encoded size. Studio update binds the driver's
interpreter and checkout through its systemd user unit, and legacy update
refuses Workers that run from release slots. A driver service must serve the
`--driver` origin ([STUDIO.md](STUDIO.md#install-and-start)).

**Fix:**

- After `choice_unavailable` or `schema_changed`, inspect and review the draft
  again; adding a model, LoRA or input file alone never voids a review. After
  `preview_expired` or `grid_preview`, review again; expired previews are
  deleted, and a submit of one is refused rather than left uncertain.
- For `prompt_rejected`, fix the inputs the job lists, then retry or review
  again. An `absent_from_driver` job may or may not have run; check the
  driver's output folder before you resubmit.
- For upload refusals, downscale the image to 16 megapixels or less, or save it
  as PNG, JPEG or WebP.
- For `record_window_limit`, shorten the record or another record of that kind.
  For `library_limit`, export one project at a time, or move records into
  another project.
- For `cancellation_scope`, wait for the running job to finish, or interrupt it
  from ComfyUI; Studio kept the cancellation intent and its key unspent.
- For `driver_service_identity`, pass the `--driver` origin the unit serves,
  or select the unit that runs that driver.
- When `dgxm studio up`, `down` or `token --rotate` is stopping or restarting
  Studio, nothing was accepted or changed; review the operation again once
  Studio answers.
- For update refusals, start the companion with `--config` and
  `--driver-service`, select the checkout the driver's
  `custom_nodes/dgx-monarch` links to, and start Studio from that checkout.
  After `update_release_slot`, select verified mode. `dgxm update` on the
  command line has none of these Studio requirements.
- For a driver code, read the companion log. The observer and the event
  watcher retry on their own with a pause of up to 30 seconds; free disk space
  for `studio_storage`.

## 125. A MOSS-Image checkpoint fails to load on ComfyUI older than 0.42.0

**Symptom:** a MOSS-Image or Kromoss template is refused before it runs:
ComfyUI's prompt check reports `Value not in list` for the CLIPLoader type
`moss_image`. A graph that loads a `moss_image_*` or `kromoss_*` file another
way stops at the model load, where ComfyUI's loader raises
`Could not detect model type`, while dgx-monarch's own header read still names
the family `moss_image`. A ComfyUI that loads the file as some other model
class leaves the render refused with `has no dgx-monarch adapter`.

**Cause:** ComfyUI added the MOSS-Image model, its detection, the
`moss_image` text encoder type and the LoRA key map in one change, commit
e087d8f6c, first released in ComfyUI 0.42.0. ComfyUI 0.41.3 does not carry
it, although on 2026-10-20 ComfyUI's own MOSS-Image tutorial still named that
release. dgx-monarch reads the family from the checkpoint header without asking
ComfyUI, so it names `moss_image` on any ComfyUI, but only a ComfyUI that
defines `comfy.model_base.MOSSImage` can build the model, and the adapter
binds that exact class. The driver needs it for the `moss_image` CLIPLoader
type too.

**Fix:** update ComfyUI to 0.42.0 or newer on the driver and on every Worker,
through the procedure you use for ComfyUI, then restart the driver and the
Workers. That ComfyUI also needs the comfy-kitchen version its requirements
name, which the INT8 ConvRot files need as well. `dgxm update` moves
dgx-monarch and torchmonarch only, never ComfyUI. Every other family keeps
working on the older ComfyUI meanwhile. Every Gate PASS goes stale with the
ComfyUI change, so each combination runs its first-use Gate again
([#114](#114-every-combination-runs-its-first-use-gate-again-after-an-update)).

## 126. Skip Layer Guidance (DGX Monarch) refuses on cfg2, on a sequence-parallel family that does not apply block replacement, or with a two-checkpoint guider

**Symptom:** a graph whose Skip Layer Guidance (DGX Monarch) node names a
layer in `double_layers` or `single_layers` stops with a class P refusal
before dispatch. The message ends "To keep this topology and guider instead,
clear both layer lists on the node; the render then runs without the skip.
See docs/TROUBLESHOOTING.md #126." and opens with one of three reasons:

* "Skip Layer Guidance (DGX Monarch) runs the conditional and unconditional
  passes as two separate model calls on every step in its window, and
  cfg-parallel can split only a call that carries both; set the Init topology
  to auto, uly2, ring2, dp2 or single to keep the skip." On a family other
  than MOSS-Image the last clause reads "set the Init topology to dp2 or
  single to keep the skip." When the driver cannot read the checkpoint's
  header it gives the first form, and a Worker on another family then refuses
  with the second reason below.
* "Skip Layer Guidance (DGX Monarch) reaches a sequence-parallel render only
  through the family's sharded forward, and dgx-monarch has shown that only
  for moss_image; `<family>`'s sharded forward could render without the skip
  and report nothing; set the Init topology to single or dp2, where ComfyUI's
  own forward runs, to keep the skip." Here `<family>` is the checkpoint's
  family.
* "Skip Layer Guidance (DGX Monarch) with a two-checkpoint guider is not
  carried to the Workers; use a single-model guider to keep the skip."

Each Worker raises the same refusal, with the same text, when a render
reaches it without the driver's check.

A graph that wires Model Sampling SD3 (DGX Monarch) after the skip node, with
a step window other than 0 to 1, stops at the shift node instead: "Model
Sampling SD3 (DGX Monarch) follows Skip Layer Guidance (DGX Monarch) in this
graph, and the skip's step window runs from `<start>` to `<end>`. Stock
ComfyUI turns that window into sigmas when the skip node runs, from the
sampling the model has at that point, while every Worker applies the shift
before the skip, so the skip would fall on other steps than stock's. Put the
shift node before the skip node instead; a window of 0 to 1 is not affected."

**Cause:** stock `SkipLayerGuidanceDiTSimple` cannot connect to a DGX Monarch
model, so the DGX Monarch node carries its four widgets on the model and each
Worker applies ComfyUI's own node to its render. Inside the node's step
window ComfyUI replaces the batched model call with two calls: the
conditional pass alone, then the unconditional pass alone with each listed
layer replaced by an identity. cfg-parallel splits a call that carries both
passes, one per rank. A call that carries one pass would be sliced instead,
which at latent batch 1 refuses with a reason that blames the prompt lengths
and at batch 2 or more computes what one GPU never does, so every cfg
topology, `+fsdp` included, refuses an active skip. Under sequence
parallelism each rank runs the family's sharded forward, which must apply
ComfyUI's block replacement as stock does. dgx-monarch shows that only for
MOSS-Image, with CPU tests that run ComfyUI's own node through the sharded
forward; the other families' sharded forwards never read the replacement and
would render without the skip and say nothing. A two-checkpoint guider sends
a second model that this release does not carry the skip on. dp2 and single
run ComfyUI's own forward and honor the skip on every family.

**Fix:** keep the skip on a topology the refusal names. On MOSS-Image auto
picks uly2. To keep cfg2, a sequence-parallel topology on another family, or a
two-checkpoint guider, clear both layer lists; the node then changes nothing
and the render runs without the skip. As in stock ComfyUI, a node with empty
lists keeps a skip an earlier node in the chain set, so clear or remove every
skip node on the chain. For the shift refusal, wire Model
Sampling SD3 (DGX Monarch) before the skip node, the order every Worker
applies them in. The refusal needs no Recycle: the driver raises it before
dispatch, and a Worker raises it on every rank alike before the first step, so
the next render runs on the same fleet, by design. On 2026-10-21 a hardware
test observed the driver's refusal on MOSS-Image, and the next render ran on
the same fleet with no Recycle; no hardware test has observed the Workers'
refusal yet. Fleet KSampler does not run the driver check; there the
Workers refuse instead.

## 127. A LoRA that changes a weight's shape refuses before any bake

**Symptom:** a render with a LoRA refuses class P while the model loads,
before any weight moves, with a message that opens "a LoRA patch replaces",
names the weight and its two shapes, and goes on "as a control adapter that
widens the model's input does (the MOSS-Image color and depth control LoRA
widens video_embedder from 128 to 384 input channels). dgx-monarch's bake,
un-bake, slab and FSDP paths keep the checkpoint's shapes, so this release
refuses it before any weight moves. Remove that LoRA from the DGX Monarch
LoRA loader, or run it in plain ComfyUI on one GPU instead. See
docs/TROUBLESHOOTING.md #127." On a sequence-parallel MOSS-Image render that
reaches the model with a wider latent by another path, the refusal instead
opens "moss_image: this render feeds the model", gives both channel counts,
and says "A color or depth control adapter concatenates them, and this
release has not validated control under sequence parallelism. Run this
workflow in plain ComfyUI on one GPU instead."

**Cause:** on ComfyUI master since e3bee7ffc, ComfyUI's model patcher can
replace a weight with a tensor of another shape, through a "set" patch or a
padded diff. ComfyUI 0.42.x counts no set patch in its shape rule, so there
the guard finds nothing to refuse and ComfyUI's own merge fails on such a
patch instead. The MOSS-Image color
and depth control (ComfyUI commit e3bee7ffc, the `MOSSImageControl` node, on
ComfyUI master only on 2026-10-20) loads a LoRA whose `video_embedder.weight`
takes 384 input channels where the checkpoint's takes 128, and feeds the
model the latent with the color and depth maps joined to it. dgx-monarch's
bake writes each patched weight into the checkpoint's own tensor, and its
un-bake, slab and FSDP paths keep the file's shapes, so before this check the
bake failed on such a key with an untyped error from
`comfy.utils.copy_to_param`. The check runs on every family and topology
right after the LoRA stack builds, with ComfyUI's own shape rule. Under
sequence parallelism the MOSS-Image forward also refuses a latent wider than
the checkpoint's, because the control channels would shard with the image
rows and no test has shown that exact.

**Fix:** remove the shape-changing LoRA from Load LoRA (DGX Monarch), or run
the control graph in plain ComfyUI on one GPU. An ordinary LoRA keeps every
shape and is not affected. Every rank refuses alike before any bake, so the
next render runs on the same fleet. On 2026-10-21 a hardware test loaded the
MOSS-Image color and depth control LoRA on uly2: both ranks refused class P
at load, and the next render ran on the same fleet with no Recycle.

## 128. Model Sampling SD3 (DGX Monarch) refuses on a family whose ComfyUI sampling multiplier is 1.0

**Symptom:** a graph with Model Sampling SD3 (DGX Monarch) stops before
dispatch with a class P refusal: "Model Sampling SD3 (DGX Monarch) applies
ComfyUI's ModelSamplingSD3 at multiplier 1000, and ComfyUI's `<config>`
config for `<checkpoint>` samples at multiplier 1.0, so this render would be
noise (ComfyUI issue 17119 reports it for MOSS-Image). Remove the node and
render with the checkpoint's own sampling instead; for MOSS-Image that
already applies the vendor's shift of 5. See docs/TROUBLESHOOTING.md #128."
Here `<config>` is ComfyUI's configuration class and `<checkpoint>` the file.

**Cause:** ComfyUI's ModelSamplingSD3 builds a flow sampling object at
multiplier 1000, the timestep scale that Wan and Kandinsky 5 use. A
checkpoint whose ComfyUI configuration samples at multiplier 1.0 then gets
timesteps a thousand times too large, and every step renders noise; stock
ComfyUI does the same with its own node. The driver reads the checkpoint's
ComfyUI configuration from its header, for the model and for the
unconditional model of a two-checkpoint guider, and refuses the node only
when that configuration says multiplier 1.0. Among dgx-monarch's families
that covers FLUX.1 Schnell checkpoints, Chroma and Radiance, Z-Image (latent
and DCT) and Lumina 2, OmniGen2, Boogu, Mage Flow, Qwen-Image, Qwen Image
2.1, Ideogram 4, Krea 2 and Kroma, MOSS-Image and Kromoss, and Anima. Such
graphs rendered noise before this check and now refuse before dispatch. The
shipped graphs wire the node only to Wan and Kandinsky 5, whose
configurations sample at multiplier 1000, so no shipped graph or sweep cell
changes outcome. A configuration with no multiplier, or a header the driver
cannot read, passes unchecked. Fleet KSampler runs the same check before
its first dispatch.

**Fix:** remove the node. MOSS-Image's own configuration already applies the
vendor's shift of 5, and the other families render with their own sampling.
dgx-monarch carries no other shift node, so a different shift on those
families needs stock ComfyUI on one GPU.

## 129. A MOSS-Image INT8 render with a LoRA refuses on every topology

**Symptom:** a render of a `moss_image` INT8 file (MOSS-Image SFT, Base or
Kromoss) with a LoRA in its stack refuses class P before dispatch, or on
every Worker while the model loads, with a message that opens "moss_image:
an INT8 file with a LoRA", names the LoRA, and goes on "rendered far from
stock ComfyUI on the cluster on 2026-10-21. dgx-monarch bakes the LoRA into
the quantized weights as stock's classic loader does, and stock with
DynamicVRAM off rendered the Kromoss INT8 file with its DMD2 LoRA as far from
its own default, so this release refuses the pair on every topology before
any weight moves: each one bakes the LoRA the same way, mode=local on one GPU
included. Render this graph in stock ComfyUI on one GPU instead, with its
default DynamicVRAM loader, or remove the LoRA; a BF16 file with a LoRA is
not refused. See docs/TROUBLESHOOTING.md #129."

**Cause:** on 2026-10-21 the Kromoss INT8 ConvRot file with the DMD2 LoRA at
strength 1.0, rendered on uly2 at 1024 square, read one-step NRMS 0.985
against stock ComfyUI's render of the same file and LoRA on one GPU, far past
the 0.10 floor, and a rerun was bit-identical. A probe found the fault
upstream, in stock ComfyUI's classic INT8 ConvRot bake. With its default
DynamicVRAM loader, stock applies a LoRA at cast time. With DynamicVRAM off
(`--disable-dynamic-vram`), stock ComfyUI alone on one GPU loads the INT8
file in full, bakes the LoRA into the INT8 weights, and renders the same
graph 0.998 NRMS from its own default at one step and 0.989 at 8 steps;
without the LoRA the two loaders agree exactly.
dgx-monarch's store bakes the LoRA the way that classic loader does, and the
cluster render sits with stock's classic render: both move the latent about
2.8 NRMS from the render without the LoRA, where every other LoRA path moves
it about 1.5. Every patched key bakes to within rounding of its exact sum, in
the store and in stock's own bake, so the mechanism inside the classic INT8
render is still open; the LoRA's delta is 13 to 29 times smaller than the
noise each INT8 requantization adds. dgx-monarch bakes a LoRA into an INT8
weight the same way on every topology, `mode=local` included, so the refusal
covers every topology. The driver reads each checkpoint header before
dispatch, for the model and for the unconditional model of a two-checkpoint
guider, and Fleet KSampler runs the same check before its first dispatch. The
model store refuses the same pair on every Worker before any LoRA build or
bake, which covers a header the driver cannot read. On some topologies an
older guard answers first with its own refusal: dp2 at batch 1, cfg2 with the
skip on (#126), and `uly2+fsdp` and `cfg2+fsdp`, through the FSDP LoRA
admission rule, refuse class P, and `ring2+fsdp` refuses class K. The INT8
dispatch fix of the same release made INT8 uly2 without a LoRA bit-identical
to a one-GPU render of the pin-stripped copy and does not change the bake.
Other families' INT8 ConvRot files share the bake. On 2026-10-21, with
comfy-kitchen 0.2.39, none of them had a measured INT8 render with a LoRA, and
they are not refused.

**Fix:** render the graph in stock ComfyUI on one GPU with its default
DynamicVRAM loader; with DynamicVRAM off, stock fails the same way. Or remove
the LoRA from the DGX Monarch LoRA loader, or use the BF16 file with the
LoRA, which is not refused. Every rank refuses alike before any bake, so the
next render runs on the same fleet, by design. On 2026-10-21 a hardware test
observed the driver's refusal before dispatch, and the next renders ran on
the same fleet with no Recycle. The Workers' own check in the model store
has not been observed on hardware yet: every shipped path runs the driver's
check first.

## 130. A MOSS-Image render with reference images refuses on every topology but one GPU

**Symptom:** a `moss_image` render (MOSS-Image SFT, Base or Kromoss) whose
conditioning comes from ComfyUI's Text Encode MOSS-Image Edit node refuses
class P before dispatch, or on every Worker before the first block, with a
message that opens "moss_image: this render carries reference images" and
names topology `single` with `mode=local` and `gpus_per_host=1`.

**Cause:** ComfyUI commit 60aade0bb of 2026-10-22 added edit LoRAs to
MOSS-Image. Its Text Encode MOSS-Image Edit node encodes the reference
images with the VAE and attaches them to the conditioning, and stock's
forward uses them in one of the layouts its `edit_format` widget picks:
either the reference rows join the joint stream at timestep 0 with their own
modulation, or each block's keys and values for them are computed once and
joined to that block's attention. Either way every image and text row
attends to rows that come from the references. The sharded forward cuts the
joint stream into one window per rank and has no rule for those rows, and no
hardware test has compared a uly2, ring2, cfg2, dp2 or FSDP edit with stock,
so this release refuses reference images on each of those topologies rather
than render them unmeasured. The driver reads the references from the
conditioning before dispatch and the family from the checkpoint header, so
the refusal costs no Worker call. If the driver cannot read the header,
every Worker refuses alike while ComfyUI builds the conditioning, before
stock runs a block over the references and before any exchange, on cfg2,
dp2 and FSDP as on the sharded topologies, so by design the next render
runs on the same fleet, which no hardware test has observed yet. Stock
computes the references' keys and values once per conditioning only in
inference mode, which a Worker's sampling thread is not in, so on 2026-10-22
a one-GPU edit recomputed them at every step and ran 1.46 times stock's time
a step, with stock's latent. Since then the adapter runs that step in
inference mode on every Worker, as stock does
([ADAPTERS](ADAPTERS.md#what-an-adapter-provides)).

**Fix:** run the edit on one GPU: set the Init node to `mode=local` with
`gpus_per_host=1` and topology `single`, which runs ComfyUI's own forward,
or render it in plain ComfyUI. Fleet KSampler runs each job on one GPU with
stock's forward, so it is not refused. A
text-to-image render without the edit node is not affected. The refusal
stays until a hardware test compares a sharded edit with stock; as of
2026-10-22 no cluster edit render has run on hardware.

## 131. Studio shows Driver not running

**Symptom:** the connected Studio page shows a **Driver not running** panel,
the footer reads "Driver not running", Review generation is disabled, and the
catalog or a workflow inspection answers:

- `No driver accepted a connection at the driver origin; start the driver.` (HTTP 503, code `driver_not_running`)

**Cause:** nothing accepts a connection at the driver origin Studio was started
with. Either the driver is stopped, or `[studio] driver` (or `--driver`) names
another address or port than ComfyUI's `--listen` and `--port`. Studio uses
this code only for a refused connection; a timeout, a reset, or a DNS or TLS
failure shows the driver as unavailable with `driver_unknown`.

**Fix:**

- Run the command the panel shows on the Studio host. The panel says where the
  command came from: `[studio] driver_service`, `[studio] driver_command` or
  the repository launcher
  ([STUDIO.md](STUDIO.md#when-the-driver-is-not-running)). Studio checks every
  few seconds and connects by itself when the driver answers, with no reload.
- Check what listens on the origin's port with `ss -ltn 'sport = :<port>'`.
- When ComfyUI runs at another address, make `[studio] driver` match it
  exactly and restart Studio (`dgxm studio up` for the service). The state
  directory is bound to the old origin, so that start refuses and prints an
  `mv` command that moves the old records aside; run it, and the next start
  makes a fresh state directory. Import a library export if the old records
  matter.
- To show your own start command in the panel, set `driver_command` in the
  `[studio]` table and restart Studio; Studio reads it when it starts. Edit
  cluster.toml when no render runs.
- A job dispatched while no driver listened settles `unknown`. Check it after
  the driver answers, before you submit again.

## 132. dgxm studio up refuses, or the always-on Studio does not start

**Symptom:** `dgxm studio`, `dgxm studio up`, `down`, `open` or `token`, or
`dgxm uninstall`, prints one of these lines, or the service does not start:

- `Run dgxm studio up as the account that runs ComfyUI, not as root.`
- `No systemd user manager answers here; run dgxm studio in a terminal instead.`
- `Another file already uses the Studio unit name; dgxm studio up writes only a unit it wrote itself.`
- `The Studio unit path is a symbolic link; dgxm never writes through a link.`
- `A drop-in changes the Studio unit; remove it, then run dgxm studio up again.`
- `systemd loads the Studio unit from another file or with drop-ins; remove them, then run dgxm studio up again.`
- `[studio] driver_service names a unit that the user manager does not know; install that unit first or remove the key.`
- `[studio] driver_service listens on another address or port than [studio] driver; make them match.`
- `Note: Studio cannot read the address from this driver unit's command; check that it serves [studio] driver. Studio's own driver controls and Studio update refuse such a unit.`
- `A companion already runs on this state directory; stop it before installing the always-on Studio.`
- `Studio did not start; its journal lines follow.`
- `Studio has not reported ready after 30 seconds; check systemctl --user status dgxm-studio.service.`
- `The Studio unit is still loaded or active after removal; check systemctl --user status dgxm-studio.service.`
- `Studio is still running or still enabled after systemctl --user disable --now, so dgxm studio down kept its unit file; stop it with systemctl --user stop dgxm-studio.service, then run dgxm studio down again.`
- `No always-on Studio to rotate; a Studio started with dgxm studio makes a new token each time it starts.`
- `Removed --allow-management from the service. Run dgxm studio up --allow-management to keep management.`
- `Linger is off for this user, so the service runs only while you are logged in: it starts at your first login and stops when your last session ends. To keep it running from boot, run: sudo loginctl enable-linger $USER`
- `Note: the installed service differs from what dgxm studio up would write now; run dgxm studio up from the environment and config you want it to use. Fields that differ:`
- `Note: the Studio code on disk changed after dgxm studio up started the service; run dgxm studio up to restart Studio on it.`
- `uninstall refused: dgxm-studio.service is still installed; run dgxm studio down first, then dgxm uninstall.`
- `Another dgxm studio up, down or token --rotate is running; wait for it.`
- `Studio is running a management operation; wait for it to finish, or stop the service yourself with systemctl --user stop dgxm-studio.service.`
- `Cannot tell whether Studio is running a management operation; wait, or stop the service yourself with systemctl --user stop dgxm-studio.service.`
- `PYTHONPATH is set in this shell, so the service would import whatever it names; unset it, then run dgxm studio up again.`
- `The service would run code from a path that another account can change; fix the owner or mode of:`
- `~/.config/systemd/user must be a directory you own that no other account can write.`
- `The service cannot hold a path that contains '$', a newline, or a ':' inside a PYTHONPATH entry:`
- `The always-on Studio is running; its settings come from its unit. Run dgxm studio down to use these flags here, or change cluster.toml and run dgxm studio up.`
- `No driver origin is set. Add driver to the [studio] table of cluster.toml, for example driver = "http://127.0.0.1:8188", or pass --driver. Studio keeps its records for the first driver address it uses.`
- `Studio reads cluster.toml only when it is a regular file you own that no other account can write; fix its owner or mode, for example chmod 600.`
- `Note: this cluster.toml has no [[hosts]], so cluster actions stay unavailable in Studio.`
- `dgxm studio up does not write a cluster.toml found in the working directory into a service; pass --config with its path, or keep it at ~/.config/dgx-monarch/cluster.toml.`

A broken `[studio]` table exits 2 with a line that starts with the file's path
and names the key: a table that is not a table, an unsupported key, a
`studio.driver` that is not an `http` or `https` origin with only a host and
port, a listen address such as `0.0.0.0`, a host other than this computer, a
spelling other than the canonical one (the message names it), a `studio.port`
outside 1024 to 65535, a `studio.driver_service` that is not a user service or
names a Worker unit or `dgxm-studio.service`, and a `studio.driver_command`
that is not 1 to 512 printable ASCII characters. A typed `--driver` gets the
same words through a usage error. `up` also refuses, with the interpreter or
the two directories in the message, when the `studio` extra is not installed
for the interpreter `dgxm` runs, or when aiohttp and Pillow sit in different
site-packages directories, and it prints the last line of a failed preflight
run of the service command. When the user manager does not answer, or the
preflight takes more than 60 seconds, `up` exits 75 and installs nothing.

**Cause:** `up` checks everything before it writes anything, and `down`,
`open` and `token --rotate` check the unit before they touch it
([STUDIO.md](STUDIO.md#keep-studio-running)):

- Account and manager: `up` runs as the account that runs ComfyUI, never as
  root, and needs a systemd user manager, which containers and some WSL setups
  lack.
- Config: Studio reads cluster.toml only when it is a regular file you own
  that no other account can write, and `up` writes no file found in the
  working directory into a service. A file without `[[hosts]]` only prints a
  note. Releases before this one refuse a cluster.toml that holds `[studio]`,
  so an older checkout that reads the same file (the driver, a Worker host or
  the TUI) fails until it runs this release.
- Driver origin: a fresh state directory needs `[studio] driver` or
  `--driver`. A state directory bound to another origin refuses with the
  store's message and two more lines, the second of which is the `mv`
  command that gives fresh records.
- Paths: the service imports the source and dependencies of the `dgxm` that
  ran `up`, so `up` refuses while `PYTHONPATH` is set, refuses a path another
  account can change, a `~/.config/systemd/user` that is not yours alone, and
  a path the unit cannot hold.
- Unit ownership: `up` and `down` touch only a unit dgxm wrote, never a
  symbolic link, and refuse while a drop-in sits in `dgxm-studio.service.d` or
  `dgxm-.service.d`, or while systemd loads the unit from another file or
  with a drop-in from a directory this account or the administrator writes.
  A vendor drop-in under `/usr/lib/systemd/user` only prints a note.
- Driver unit: `[studio] driver_service` must name a unit the user manager
  knows that serves `[studio] driver`. A unit whose command runs a wrapper
  script only prints the note above.
- Running companions: a Studio started in a terminal holds the state
  directory's lock, another `up`, `down` or `token --rotate` holds the unit
  lock, and a review or management operation the service accepted (an
  update, even before its controller starts, a management command or a status
  report) keeps the service from being stopped. A control group that cannot
  be read counts as busy.
- Stop: `down` keeps the unit file when `systemctl --user disable --now`
  fails or the service still runs afterwards, and prints the `ActiveState`
  it read and systemctl's last error line, so a later `down` still finds the
  unit.
- Start: a start that fails prints the service's journal lines. Common causes
  are a busy port, a Studio in a terminal holding the state lock, an
  interpreter that moved, and a state directory bound to another origin.
  Five failed starts within 60 seconds trip the start limit, and the unit
  stays installed and enabled.
- Notes: linger off stops the service at logout; the drift note names
  settings that differ from what `up` would write now, such as `interpreter`
  or `driver`; the stale-code note means `dgxm update` replaced the code the
  service imported. A companion crash during a Studio update ends the update
  controller with the unit; systemd restarts Studio and `Wants=` starts the
  driver unit again, possibly on a checkout whose update was cut off.
- Uninstall: `dgxm uninstall` refuses while the unit is installed, because
  `dgxm studio down` would no longer exist after it.

**Fix:**

- Run `up` as the ComfyUI account. Without a user manager, run `dgxm studio`
  in a terminal.
- Fix the table as the message says. Own cluster.toml and remove other
  accounts' write access (`chmod 600`). For a file in the working directory,
  pass `--config /absolute/path/cluster.toml`, or keep it at
  `~/.config/dgx-monarch/cluster.toml`. Update every checkout that reads the
  file and restart the driver before you add `[studio]`.
- Add `driver` to the `[studio]` table. For a state directory bound to
  another origin, set `driver` to that origin, or run the printed `mv`
  command so the next start makes fresh records.
- Run `unset PYTHONPATH`, then `up` again. Fix the owner or mode of the named
  path, and of `~/.config/systemd/user` (`chmod 700`). Install the `studio`
  extra into the interpreter `dgxm` runs ([INSTALL](INSTALL.md)).
- Check what the foreign unit, symbolic link or drop-in is, then remove or
  move it yourself; dgxm never deletes it.
- Install the driver unit and run `systemctl --user daemon-reload`, or remove
  `driver_service`. Make the driver unit's `--listen` and `--port` match
  `[studio] driver`.
- Stop a Studio running in a terminal with Ctrl+C, then run `up`. Wait for
  another `up`, `down` or `--rotate` to end, and for a management operation to
  finish, or stop the service yourself with
  `systemctl --user stop dgxm-studio.service`.
- When `down` kept the unit file, read systemctl's line, run
  `systemctl --user status dgxm-studio.service`, stop the service with
  `systemctl --user stop dgxm-studio.service`, then run `down` again.
- After a failed start, read the journal lines `up` printed, or
  `journalctl --user -u dgxm-studio.service -n 20`, fix the cause and run
  `up` again; `up` resets the start limit. After a timeout, check
  `systemctl --user status dgxm-studio.service`.
- For a rotation without a service, restart the Studio in its terminal; each
  start makes a new token.
- Turn linger on to keep the service running from boot and after logout:
  `sudo loginctl enable-linger $USER`.
- After `dgxm update`, or when the drift note names fields you want, run
  `dgxm studio up` from the environment and with the config you want the
  service to use. Run `dgxm studio up --allow-management` to keep
  management.
- After a companion crash during a Studio update, Studio records the
  operation as unknown; check the checkout and the cluster before you render
  again.
- Run `dgxm studio down` before `dgxm uninstall`.

## 133. The sign-in link does not open Studio

**Symptom:** no browser opens, the link opens a page that asks for the token,
or a `dgxm studio` command prints one of these lines:

- `Studio is not running. Start it with dgxm studio, or keep it running with dgxm studio up.`
- `No sign-in page: a Studio started in a terminal printed its sign-in link or token there.`
- `No token file: a Studio started in a terminal printed its token there.`
- `Refusing to print the token to a pipe or file; run this in a terminal or read the token file.`
- `No browser opened: this desktop opens .html files with another program than your web browser. Open the sign-in link above.`
- `No browser opened: xdg-open is missing or failed. Open the sign-in link above.`
- `No browser opened: no display here, or a tmux or screen session. Open the sign-in link above in a browser on this computer, or run dgxm studio tunnel for another computer.`
- `This Studio runs with --allow-management, so it never signs in by link; paste the owner token into the page's form.`
- `Studio already runs in another terminal on this account and printed its sign-in link or token there. Stop it there with Ctrl+C, or use dgxm studio up for an always-on Studio.`
- `dgxm-studio.service runs but has not written its sign-in page or token file yet; see journalctl --user -u dgxm-studio.service -n 20.`
- `tunnel takes an SSH host name, an IP address or user@host; it refuses spaces, shell characters, an IPv6 zone (%) and a leading '-'.`
- `Name the Studio host the way the browser computer reaches it, for example dgxm studio tunnel studio-host or dgxm studio tunnel user@studio-host.`
- `ssh was not found on PATH.`
- `Keep 127.0.0.1 and the same port on both ends; Studio refuses localhost and any other port.`
- `If ssh says "Address already in use" or "Could not request local forwarding", another program on that computer holds the port: close the session if it opened, free the port, and do not open the sign-in link until the tunnel works, because a page served on that port would read the token from the link.`

The page itself can say "This sign-in link was not accepted: it is from an
earlier start of Studio, or it was cut short when copied. Run dgxm studio open
on the Studio host for a new link."

**Cause:**

- Nothing runs: `open`, `token` and `tunnel` read the files of the running
  Studio, and a Studio started in a terminal prints its link or token there
  and writes no file. A token or link never goes to a pipe or a file.
- Starting: the service runs but has not written its sign-in page or token
  file within 30 seconds; the command exits 75.
- Management: a Studio with `--allow-management` never makes a sign-in link;
  its page signs in only through the form.
- Desktop handler: on GNOME, `xdg-open` opens a `file://` page with the
  program the desktop uses for `.html` files, which can be a chat client or an
  editor rather than the web browser. Studio then opens nothing and prints a
  line such as `.html opens with <entry>; the web browser is <entry>.`
  followed by the `xdg-mime default` command that fixes it.
- Confined browser: a snap or Flatpak browser may be unable to read
  `sign-in.html` under `~/.local/state`, and shows an error. Studio has
  already counted the page as sent, so it printed no link.
- SSH session, tmux or screen: nothing opens by design, because a browser
  started there could draw on the Studio host's own screen.
- Old or short link: a link from history after a restart, or a link cut short
  by a narrow terminal, holds a token the running companion does not accept.
  The page shows the sentence above and the token form; a good token stored
  in the tab is not replaced.
- Tunnel: the printed ssh line names `127.0.0.1` as the address the forward
  listens on and carries `-o ExitOnForwardFailure=yes`, so ssh prints
  `Address already in use` and `Could not request local forwarding` and exits
  when another program on the browser computer holds the port. An ssh command
  or app that forwards without that address listens on `::1` too, opens the
  session after the `Address already in use` line, and a page served on that
  port would read the token from the link.
- `localhost` in the address bar, or a different local port, fails the Host
  check ([#123](#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token)).

**Fix:**

- Start Studio with `dgxm studio`, or `dgxm studio up` for the service, then
  run `dgxm studio open`. For a Studio in another terminal, use the link it
  printed there, or stop it with Ctrl+C. Run `dgxm studio token --value` or
  `open` in a terminal, not through a pipe.
- For a service that is still starting, wait and run `open` again, or read
  `journalctl --user -u dgxm-studio.service -n 20`.
- For a management service, paste the owner token that
  `dgxm studio token --value` prints into the page's form.
- On a desktop that opens `.html` files elsewhere, click the printed link, or
  run the printed `xdg-mime default` command to make the web browser open
  `.html` files too. With a confined browser, run `dgxm studio --no-open` in a
  second terminal, which acts as `dgxm studio open` and prints the link, or
  open `http://127.0.0.1:<port>/` and paste the printed `Owner token:` into its
  form, or use a browser installed another way.
- In an SSH session, tmux or screen, click or paste the printed link. For a
  browser on another computer, run `dgxm studio tunnel <host>` and follow its
  lines ([STUDIO.md](STUDIO.md#reach-studio-from-another-computer)).
- For an old or cut link, run `dgxm studio open` on the Studio host for a new
  one, and copy it whole.
- After `Address already in use` or `Could not request local forwarding`, do
  not open the link. Close the session if it opened, free the port on the
  browser computer, and connect again.
- Give `tunnel` a host as a name, an SSH config alias, an IP address or
  `user@host`, and install `ssh` for `--run`. Keep `127.0.0.1` and the same
  port on both ends.
