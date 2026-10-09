# Threat Model

## Overview

dgx-monarch is a ComfyUI custom-node pack and cluster runtime. A ComfyUI driver
accepts workflows, resolves model files, exposes browser telemetry, consent, and
Attached mesh reset routes, and dispatches each Render session through
torchmonarch actors reached through one or more Worker services. The CLI also
reads cluster configuration and manages local or SSH Worker services.

Assets to protect include:

- control of the driver and worker OS accounts and their GPU workloads;
- model and LoRA files, generated outputs, and render integrity;
- credentials present in the operator session or service environment;
- private host, network, hardware, path, and performance details; and
- the integrity of internal source archives, local build artifacts, workflows,
  reports, and release automation.

The project assumes one trusted operator and one trust domain per ComfyUI and
cluster deployment. It is not a multi-tenant scheduler or a sandbox for hostile
custom nodes, Python packages, administrators, or local OS users.

Local source archives are disclosure boundaries even though the project
publishes no package-registry artifacts. Only the two tracked skills are
included from `skills/` in an archive.
Ignored worktrees and private Markdown are excluded. CI verifies that an
ignored sentinel file does not enter the archive.

## Threat Model, Trust Boundaries, and Assumptions

### Browser to ComfyUI driver

`GET /dgxm/telemetry`, `GET /dgxm/metrics`, and `GET /dgxm/consents` are
read-only but expose runtime and topology information to clients admitted by
the surrounding ComfyUI and proxy controls, or to any client that can reach an
unprotected deployment. The Attached mesh reset POST and the consent POST each
require an action header and same-origin Host/Origin, and reject
`Sec-Fetch-Site: cross-site` when supplied. These are CSRF controls, not user
authentication; a one-reset-at-a-time gate and bounded server deadlines limit the
resource cost of repeated calls ([FAQ](FAQ.md#install-and-first-hour),
[routes.py](../src/dgx_monarch/nodes/routes.py),
[route tests](../tests/test_routes_cache.py)). A network-exposed deployment
inherits the authentication, authorization, and CORS policy protecting ComfyUI
and depends on its reverse proxy and normal network access controls. Reverse
proxies must overwrite, not append or trust, client-supplied forwarding
headers: the action check reads the scheme from the first
`X-Forwarded-Proto` value.

A remote browser user is attacker-controlled unless those deployment controls
authenticate and authorize them. Workflow inputs, prompt text, selection among
operator-installed model choices, and route requests are untrusted at this
boundary; the ComfyUI process and its installed custom nodes remain in the
trusted application domain. Telemetry and websocket output may include host and
GPU details, model and LoRA identifiers, configuration context and paths, ledger
and event data, and error text; these browser-facing outputs
are not scrubbed for secrets.

### Driver to worker fabric

The primary and dynamically allocated secondary actor channels run without
peer authentication ([SECURITY.md](../SECURITY.md#fabric-trust-boundary)).
Attach also sends the driver's effective hyperactor configuration to every
host. This is separate from the Worker service environment allowlist.

Interface isolation provides this boundary's protection.
[SECURITY.md](../SECURITY.md#fabric-trust-boundary) specifies the requirements.
NCCL, Gloo, UCX, RDMA, and upstream torchmonarch are also trusted
dependencies at this boundary.

The setup and update cluster smoke runs as a transient client at this
boundary. It attaches a data-parallel ProcMesh, establishes NCCL, checks
all-rank source and status evidence, and retires only the ProcMesh it owns
through the typed mesh recycle transaction. Cleanup never calls the broad
`hosts.shutdown()` API. Normal attach auto-heal may restart an unhealthy Worker
service as part of an already authorized setup or update. Those
ownership checks limit accidental teardown; they do not authenticate the fabric
peer or sandbox actor code.

### CLI operator workflows

Guided setup takes an explicit host inventory and optional artifact paths, then
probes the named hosts and paths locally or over SSH; it does not scan for
hosts. Its dry-run config diff shows full values, so it can expose the
operator's network, usernames, paths, and config policy. Config application,
service installation and start, and verification each need their own explicit
request ([SECURITY.md](../SECURITY.md#operator-workflow-boundary) states which
step requires which). The trusted-fabric acknowledgement is an assertion about
external isolation, not a probe result.

Cooperating lifecycle commands serialize each target user's Worker service
mutations per host, and setup also binds remote staging to its owned
transaction identities. Legacy init, guided setup, safe repair, and uninstall
serialize config mutation per normalized destination. These locks and the
snapshot and inode checks reduce accidental overlap and reject observed drift.
They are not a cluster-wide lock, an authorization boundary, or a defense
against manual or hostile same-account mutation in the last check-to-rename or
check-to-unlink window. [INSTALL.md](INSTALL.md#guided-multi-node-setup) gives
the operational scope.

An optional process-inspection helper is a separate, explicit administrator
boundary. It executes a reviewed root-owned temporary copy under isolated
system Python, reads only process markers for the invoking UID, and expires within
30 minutes. Its local Unix socket authenticates the caller UID with kernel peer
credentials; setup authenticates root peer credentials, exact helper source,
UID, boot identity and a fresh bounded response. Command lines and environment
values never leave the helper. The fixed request cannot select a command,
path, or PID. The helper cannot mutate processes or services, and setup never elevates
or starts it automatically. A missing helper or incomplete inventory stays
unknown through activation and recovery. This trusts the administrator and
reviewed helper code; it is not a defense against a malicious root account.
The launch and expiry contract is in
[INSTALL.md](INSTALL.md#protected-process-inspection).

Setup profiles produce explicit configuration values. The runtime config
stores those values, not the profile label; the receipt records the selected
profile. Conservative hardware classification
limits unsafe optimistic configuration but does not attest hardware against a
malicious host. [CLUSTER.md](CLUSTER.md#guided-setup-profiles) gives the exact
profile contract, including the native-RDMA HOLD.

Safe doctor repair opens the exact config without following a final symlink,
requires a regular inode, hashes its bytes, confirms the proposed mode-only
change, then rechecks inode, content, and mode before `fchmod(0600)`. Checks
for changes between inspection and mutation, together with the
permission-only operation, limit accidental changes; a hostile
same-account or privileged process remains outside the project boundary.
Repair never automates network, SSH, package, service, process, Attached mesh,
or power changes.

Verified update fetches an operator-selected ref, resolves one commit, checks a
clean fast-forward checkout and unchanged config, rejects active or unknown
work, and stages private no-deps release slots on every remote host before
confirmation. It trusts the Git origin, package indexes and cache, SSH peers,
and target code. Checks of release versions, source, and dependency pins on
every host, plus
source and status checks from every rank, reduce accidental version mixtures;
they are not a signature or protection from a malicious trusted publisher.
Update commits the live driver pin and checkout only after remote
verification; an ambiguous stop, activation, compensation, restart, or driver
commit fails closed instead of running workers whose versions are unverified
or inconsistent.
[INSTALL.md](INSTALL.md#updating) holds the complete transaction contract.

Pinned payload checks permit one recorded color-matcher 0.6.0 compatibility
case: its known external test-data/cache records and a verified empty
`tests/__init__.py`. The initializer makes the ancillary `tests` namespace a
regular Python package. This allowance does not extend to runtime packages,
nonempty initializers, overlapping pinned files or ambiguous ownership. All
pinned wheel files still require their original hashes. External record paths
are classified without reading or changing their targets. Installed metadata
establishes local ownership, not publisher authenticity.

The recovery updater explicitly trusts a separately reviewed controller
checkout at the selected target commit. It binds that controller separately
from the unchanged original checkout and config, then uses the normal update
lock, prior-release checks, confirmation, activation and recovery. It does not
grant the new controller authority to adopt an unknown prior installation.

### GitHub Actions to trusted hardware

The public source repository has no self-hosted or custom-label hardware route.
It must not own or request a self-hosted or JIT runner. Protected hardware
validation uses hand-run scripts from a separately reviewed private control
repository. No CI workflow, Actions runner, or scheduled job may execute on
protected hardware. A GitHub Environment approval or acknowledgement string
cannot authorize hardware access.

The operator reviews the script and selects one exact 40-character public
source commit. Before execution, verify that commit and the source manifest
on every participating host, including the source loaded by persistent Worker
services. Passively establish that the hosts are idle, identify existing
services and work owners, and obtain authorization for the planned operations.
Unknown ownership, active unrelated work, or a source mismatch blocks the run.
A status command alone is not sufficient evidence of idle hardware.

Worker services persist between client sessions. A validation client must not
stop them as cleanup. Service stops or restarts require explicitly authorized
maintenance, with the expected service identity recorded and restoration
checked afterward. A stopped-service readiness test needs that authorization.
Native latent RDMA remains **NOT RUN / HOLD**; actor messaging is the supported
return path. No manual test may silently expand into RDMA qualification.

Repository responsibilities are:

- The public source repository holds reviewable code and one exact public
  target commit. No executable hardware workflow may sit there, and no
  public workflow may call a runner-registration endpoint such as
  `actions/runners/generate-jitconfig`.
- The private control repository holds reviewed hand-run scripts and private
  evidence. It has trusted writers only and no CI hardware execution route.
- The operator owns execution, authorization, source verification, and cleanup.
  Do not expose GitHub, SSH-agent, cloud, deploy, or release credentials to
  test code. If a script needs a cluster-lifecycle SSH key, restrict it to the
  intended hosts, account, and operations. Treat it as exposed to reviewed code.

Hand-run code is not sandboxed. It has the filesystem and network authority of
its OS account, including readable model files and reachable fabric. Prefer a
dedicated account without operator credentials. Use of a shared account trusts
the reviewed code with everything that account can access.

Record the exact source, tested configuration, result, and owned cleanup in
private evidence. Retain the records needed to investigate failures; remove
temporary workspaces only after evidence is preserved, without deleting model
files or unrelated work. A check without a recorded clean run remains
**NOT RUN**. CPU tests and operator checks do not qualify a model family or a
transport. Acceptance applies only to the tested feature and configuration.

Before a visibility change, complete every step:

1. Keep protected hardware disconnected from Actions: no registered hardware
   runners and no runner process awaiting work.
2. Prove the public default branch carries no workflow under
   `.github/workflows/` whose `runs-on` value can select a self-hosted or
   caller-supplied custom label. Verify that the private control repository
   also has no CI hardware execution route.
3. Inventory registered runners, queued and in-progress jobs, workflow runs,
   historical logs, artifacts, caches, repository variables, secrets,
   Environments, deployments, and Pages or release assets that may carry
   private data.
4. Remove sensitive state or prove it holds only reviewed public data. State
   that cannot be proved safe blocks changing that repository in place;
   publish a new sanitized repository instead.
5. Confirm private vulnerability reporting is enabled and the public issue form
   asks for narrow, sanitized excerpts.
6. Read back the final public default branch and repeat the no-route check.

Adopting any hardware runner route requires a reviewed policy, threat-model,
and test change. Manual execution does not waive the visibility checks.

The public conclusion of a protected run may carry only the tested public
source commit, PASS, FAIL, or **NOT RUN**, high-level probe conclusions, and a
sanitized dependency and hardware class. It must not carry private control
references, raw logs, artifacts, credentials, hosts, private addresses,
usernames, absolute paths, interface names, GPU UUIDs, model paths, process
identifiers, listener ports, timings, or rig-specific diagnostics. Review and
sanitize evidence before disclosure; keeping it in a private repository does
not make it safe to publish.

### Worker process environment

Before Monarch import and actor launch, long-lived Worker services apply a `0077`
umask, a private runtime directory, a name-based environment allowlist, and a
managed `PYTHONPATH`. Lifecycle and actor launch also disable bytecode writes
and redirect bytecode-cache lookup outside the source trees, so attested
dgx-monarch and ComfyUI imports cannot reuse tree-local caches.
Credential-shaped and non-allowlisted variable names are removed without
reading their values. The filter does not inspect values under allowed names;
runtime, scratch, managed import-path, and bytecode-policy variables are then
replaced with project-controlled values
([worker_process_env.py](../src/dgx_monarch/cli/worker_process_env.py),
[tests](../tests/test_worker_process_env.py)).

This boundary limits accidental credential propagation and diagnostic exposure;
it does not scrub the driver process or an already-running service until restart,
and it is not an OS sandbox.

For strict acceptance, each worker hashes canonical manifests of every live
`.py` and tracked native-extension import in dgx-monarch and ComfyUI twice per
snapshot, around the artifact identity reads, and rejects any change.
ComfyUI's root `input`, `models`, `output`, `temp`, and `user` data trees are
excluded from that source inventory. The top-level `custom_nodes` directory alone is
omitted from Git status, hidden-index-flag, and source-inventory checks when
the fixed `custom_nodes_disabled` bootstrap policy prevented those packs from
loading; any loaded origin there is still rejected.
Every other checkout path remains strict, as do hidden index flags on included
paths, the tracked source set, and loaded origins. An included direct or
sourceless `.pyc`, symlink, non-regular import file, or loaded origin outside
the exact source trees is rejected. Loaded-origin maps are path-redacted and
may grow monotonically as modules load, but cannot lose or rebind an origin.

A managed secondary root must contain only the `dgx_monarch` package. A
Git-backed root must be clean, at the exact checkout root, free of hidden index
flags on its included provenance surface, and equal to the import-capable files
tracked at HEAD; Git inspection ignores ambient repositories, configuration,
and replacement objects. ComfyUI must always be an exact Git checkout, clean
except for the `custom_nodes` exception above. The Git found on `PATH` is
resolved once to a pinned absolute executable, and each query runs with
sanitized Git configuration and environment. That executable's device, inode,
size, and time identity is checked around every query, so replacement or drift
fails closed. Policy changes require process recycle, and the legacy
`DGXM_NO_CUSTOM_NODES` preload switch does not satisfy the strict
bootstrap-policy check.

Strict snapshots publish hashes identifying each machine and process lifetime,
not hostnames, checkout paths, or raw process IDs. Worker-side artifact binding
hashes an ID plus a stable device/inode/size/time tuple; it is not a content
hash. The external acceptance preflight full-hashes the selected files on both
hosts before hardware evidence can become eligible.

These controls detect changes to runtime inputs on trusted hosts; they are not
authentication, a supply-chain signature, or an OS sandbox, and they do not
defend against a malicious administrator racing or replacing an already-running
process. Code imported by ComfyUI, dgx-monarch, torchmonarch, PyTorch, CUDA,
xfuser, or another installed node retains the worker account's filesystem and
network privileges.

### Checkpoints and model artifacts

Checkpoint bytes may be supplied by an untrusted publisher, but the operator
controls which files are installed and exposed for workflow selection.
dgx-monarch's header-only consumers accept safetensors JSON, cap the encoded
header length at 256 MiB, reject duplicate keys, malformed descriptors, invalid
shape/byte relationships, out-of-file ranges, overlaps, truncation, and mutation
of the opened inode's checked stat fields during the read
([safetensors_header.py](../src/dgx_monarch/safetensors_header.py),
[tests](../tests/test_safetensors_header_properties.py),
[detection tests](../tests/test_detect.py)). The parser uses JSON and positional
reads; it does not invoke pickle or execute checkpoint content. That no-pickle
guarantee covers only this header parser and automatic format sniffing; an
explicitly configured legacy checkpoint can still be delegated to ComfyUI and
PyTorch.

These checks do not authenticate a file or make a model semantically
trustworthy. Valid tensor data can still cause memory exhaustion, non-finite
output, extreme runtime, adversarial model behavior, or exercise a defect in a
trusted loader or kernel. Downstream safetensors, PyTorch, CUDA, and ComfyUI
loaders remain part of the trusted computing base; use safetensors-only inputs
for untrusted publishers. Other ComfyUI nodes or formats may have different
loading rules and are outside this parser's guarantee.

### Diagnostics, reports, and repository artifacts

Benchmark reports written outside the gitignored default are redacted for host
identity, GPU UUID, and private paths before the destination is opened;
`--no-redact` is an explicit full-fidelity override
([run_matrix.py](../benchmark/run_matrix.py),
[tests](../tests/test_run_matrix_sanitize.py)). CI also scans every Git-tracked
working-tree file for structural workstation residue and common credential
shapes. The scanner does not carry exact private usernames, host identities, or
key filenames, and its allowlist covers only explicit fictional fixtures
([leak_check.py](../tools/leak_check.py),
[tests](../tests/test_leak_check.py)).

These controls cover benchmark publication and committed repository content.
Report redaction is not a general secret scrubber: it preserves model and LoRA
filenames, sizes, mtimes, sampled signatures, matrix settings, prompts, and
other free text, and it does not sanitize console errors. The tracked-tree
scanner is regex-based defense in depth, not proof that the repository is
free of secrets; it does not inspect untracked files, history, logs, issues, or
attachments. Operators must review every artifact before sharing.

Operator-workflow receipts apply a narrower schema and redaction boundary than
benchmark reports: atomically published, non-overwriting `0600` JSON stores
only stable operation and step names, outcomes, counts, timestamps, an optional
profile and target, bounded hashes, and sanitized free-text notes. Host and
network identities, commands, environments, credentials, URLs, paths, and raw
exceptions are rejected or redacted. The remaining timing, count, target,
digest, and note content can still be sensitive, and console and full logs
remain outside this boundary. A receipt proves only what the local workflow
recorded; it is not signed machine identity or remote attestation.

## Attack Surface, Mitigations, and Attacker Stories

- **Browser routes.** A hostile web page, open in any browser that can reach the
  deployment, may try to reset the Attached mesh, and an unauthenticated network
  client may query telemetry. The browser CSRF checks reduce cross-site mutation
  triggers; deployment authentication and network policy protect both disclosure
  and authorization, including direct clients.
- **Consent actions.** Such a page or an admitted client may try to accept a
  pending residency-consent card. The same action-header and same-origin CSRF
  boundary covers this POST, with a rate limit, a bounded request body, and a
  server-side action timeout. A consent is an availability grant only: it never
  clears a quarantine and never asserts render accuracy
  ([TRUST.md](TRUST.md)). A consent memo write, for an accept or a revoke,
  reports success only after the rename and a parent-directory fsync, and a
  full waiver-audit buffer leaves the request unwaived instead of discarding
  records needed for an active request's permanent use row. Reading `GET /dgxm/consents`
  discloses pending model and capability-context detail on the same
  unauthenticated terms as telemetry.
- **Cluster configuration and lifecycle.** Operator-controlled host addresses,
  paths, environment values, and SSH lifecycle commands cross process and host
  boundaries. Typed config validation, unicast-address requirements, validated
  fabric variables, stdin-delivered remote scripts, and shell quoting reduce
  injection and misbinding risk. The operator must still protect the config,
  SSH key, worker account, and fabric.
- **Guided setup and safe repair.** Malicious or mistaken explicit host/path
  input may target the wrong SSH peer or disclose a full config diff; setup's
  plan/confirmation/ownership checks limit mutation but do not establish trust.
  Repair's exact-file allowlist prevents diagnostics from turning into broad
  lifecycle automation.
- **Verified update.** A compromised origin, target commit, package source, SSH
  peer, or worker administrator is trusted code at activation. Staging the
  exact version,
  refusing active work, confirming stop and switch outcomes, and verifying the
  result reduce accidental version mismatch and unsafe continuation. They do
  not prevent supply-chain compromise.
- **Actor and data-plane traffic.** A hostile fabric peer may spoof, observe,
  disrupt, or replay unauthenticated control traffic or attack NCCL/RDMA
  services. Interface isolation is the primary mitigation; there is no project
  claim of confidentiality or peer authentication on this path.
- **Models and workflows.** Malformed safetensors headers are rejected before
  tensor-payload mapping or allocation by dgx-monarch's header and slab
  consumers, and auto detection does not deserialize pickle. Valid but
  adversarial tensors or workflows may still exhaust CPU, GPU, memory, storage,
  or operator time; resource exhaustion is treated as a realistic availability
  threat, not code execution by itself.
- **Environment and diagnostics.** Ambient secrets may leak through inherited
  variables, command lines, logs, reports, or committed artifacts. The worker
  allowlist, private runtime paths, report redaction, narrow diagnostics, and CI
  leak scanner reduce this risk, but explicit full-fidelity output remains the
  operator's responsibility.
- **Supply chain.** ComfyUI, torchmonarch, PyTorch/CUDA, xfuser, safetensors,
  optional attention packages, and other custom nodes execute trusted code.
  Dependency pinning and compatibility canaries reduce accidental drift; they
  do not defend against a malicious maintainer or compromised upstream release.

Out of scope are a malicious repository maintainer, malicious root or worker
administrator, physical or firmware compromise, and containment of arbitrary
Python code that the operator installs. Vulnerabilities in those components may
still affect dgx-monarch deployments, but this repository cannot provide an
isolation boundary against them.

## Severity Calibration (Critical, High, Medium, Low)

- **Critical:** an unauthenticated remote path to arbitrary code execution on
  driver or worker accounts; compromise of release credentials that ships
  attacker code; or a boundary break that exposes credentials and yields
  cluster-wide execution without operator action.
- **High:** bypass of deployment authorization for destructive cluster control;
  a reachable fabric flaw that lets an untrusted peer control worker actors;
  checkpoint parsing that produces memory corruption or code execution; or
  broad credential disclosure from default diagnostics or artifacts.
- **Medium:** repeatable denial of service by an authenticated browser user or
  adjacent fabric peer; CSRF that reaches **Reset attached mesh** or a consent
  action; private topology, paths, or hardware identifiers exposed beyond the
  intended deployment; or a malformed artifact that causes bounded process
  failure without code execution.
- **Low:** low-sensitivity information disclosure, noisy logs, or operator-only
  unsafe behavior that requires explicit full-fidelity flags and does not cross
  a trust boundary.

Severity rises with remote reachability, absence of operator action, privilege,
credential impact, and the number of hosts affected. It falls when exploitation
requires a trusted administrator, a deliberately installed custom node, or an
already-compromised fabric or OS account.
