# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through this repository's
GitHub **Security** tab using **Report a vulnerability**. Include the affected
version, impact, reproduction details, and any proposed fix; do not open a
public issue before coordinated disclosure.

Do not include credentials, private model files, private network addresses, or
unsanitized diagnostic bundles. The maintainer will coordinate disclosure and
credit on a best-effort basis.

Public bug reports must contain only narrow, sanitized excerpts. Replace
hostnames, private addresses, usernames, absolute paths, GPU UUIDs, model and
LoRA names, commands, environment values, and tokens with placeholders. If the
evidence cannot be redacted safely, use private vulnerability reporting.

## Supported versions

Security fixes target the current default branch and the latest published
release. Older releases and arbitrary development commits are not supported;
users may be asked to reproduce on a current supported revision.

## Operator-workflow boundary

`dgxm setup` probes only the candidate hosts and artifact paths the operator
supplies; it neither scans the LAN nor secures it.
Hostnames, fabric addresses, SSH users and keys, ComfyUI locations, and model
paths remain sensitive operator inputs. A dry run changes no config, service,
or cluster state. It writes a local audit receipt only when requested.
The terminal plan shows the full config diff and can disclose those values. The
disclosure-reduced JSON plan still includes versions, commits, fingerprints,
counts, and hashes. Review either form before sharing it. Applying the plan,
installing services, starting them, and running verification each require an
explicit request. Starting requires installation in the same transaction;
verification requires startup. Setup writes the
`transport_security = "trusted_fabric"` acknowledgement only when the operator
asserts the [fabric trust boundary](#fabric-trust-boundary) below; setup cannot
create or prove that external isolation.

The per-host lifecycle locks, the setup staging lock, and the shared
per-destination config lock are cooperative concurrency controls, not
authorization. Identity and readback checks reject observed changes, and an
absent config is published atomically, never over a file that appeared
meanwhile. Linux offers no portable "replace or unlink only if this inode is
still current" operation, so a manual or hostile same-account writer can
bypass the locks and change the target between the final check and rename or
unlink. The account must therefore be trusted.
[INSTALL.md](docs/INSTALL.md#guided-multi-node-setup) describes the supported operations and their limits.

The standalone service installer and uninstaller also require the unit path to
be absent or a non-symlink regular file that carries the current managed marker
or matches the exact pre-marker canonical unit for the current configuration.
The marker permits updating an owned unit for a changed config. Symlinks,
modified unmarked files, and foreign units are rejected before source sync,
generation invalidation, stop, replacement, or unlink. The uninstaller also
accepts a unit that guided setup wrote; the installer refuses that unit, and
refuses while any drop-in for the unit exists in the user's three unit paths,
because a drop-in can replace `ExecStart`. Installation writes a temporary
file in the same directory and replaces the unit path. It never follows a
symlink at that path to write its target.

`dgxm doctor --repair` only changes `cluster.toml` permissions. It opens the
exact inode without following a final symlink, requires a regular file, and
sets mode `0600` after operator confirmation and an inode, content and mode
recheck. It does not alter network, SSH, packages, services, processes,
attached meshes, or power. All other recommended fixes require manual operator
action.

Guided setup publishes a changed config only after confirming that the
requested service changes succeeded. Unknown process activity, service
ownership, staging, activation, or final transaction outcomes leave the config
unchanged. Before cleanup dispatches a stop, it durably records that intent.
An ambiguous stop is never retried. If ownership can no longer be confirmed,
cleanup retains its artifacts to avoid affecting a replacement process. A
missing, changed, or replaced-and-restored marker (an ABA mismatch) also
blocks mutation, and a verified update starts an existing
systemd Worker unit only if it carries the canonical generation-invalidating
`ExecStartPre`. After a positively confirmed target start, a verified update
may stop that confirmed Worker-service generation once to contain the updated workers. It never issues a second stop. [INSTALL.md](docs/INSTALL.md#updating) describes the marker, fence,
readback, and settlement mechanics behind those rules.

`dgxm update --verify` fetches and resolves an operator-selected ref to an exact
commit, stages a private no-deps release through the configured SSH accounts,
and runs that target code during doctor and cluster smoke. Those checks reduce
accidental drift; they do not authenticate a malicious origin, package
publisher, Git server, repository maintainer, SSH peer, or worker administrator.
Review and trust the target before confirmation. When stop or activation
evidence is ambiguous, update skips rollback and restart and may leave Worker
services stopped for inspection.

Operator receipts, including the cluster-smoke step that setup and update
record, are disclosure-reduced audit records, not a secret scrubber or a
signature. They reject or redact paths, network identity, commands, environment
assignments, credentials, URLs, and raw exceptions; the default private
directory, non-overwriting publication, and `0600` file protect local storage.
Timestamps, operation names, step outcomes and counts, the selected profile or
commit hash, source and config digests, and sanitized notes remain. Review a
receipt before publishing it, and never substitute it for the full private logs
needed to investigate a failure.

`dgxm doctor --json` returns full diagnostic details, including host, network,
path, version and raw check output. Treat it as a private diagnostic bundle
and redact it before sharing; receipt redaction does not apply.

## Trusted hardware validation boundary

This public source repository must not own or request a self-hosted or JIT
runner. Hardware validation uses hand-run scripts from a separately reviewed
private control repository. No CI workflow, Actions runner, or scheduled job
may execute on protected hardware.

Before a run, the operator authorizes its scope, verifies the exact source on
every host, and passively checks idle state and ownership. Unperformed checks
remain **NOT RUN**; a passing run qualifies only the configuration it tested.
Persistent Worker services may be stopped only for explicitly authorized
maintenance. Native latent RDMA remains **NOT RUN / HOLD**.
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#github-actions-to-trusted-hardware)
defines manual validation, evidence handling, and the checks required
before changing repository visibility.

## Fabric trust boundary

The torchmonarch Worker service listener and attach API this project calls accept
only `ca="trust_all_connections"` and raise `NotImplementedError` for
certificate/CA plumbing, so peer authentication is unavailable through
dgx-monarch. This limitation applies to the APIs dgx-monarch uses.
Torchmonarch also has environment-configured TLS support in its channel layer,
but dgx-monarch does not use or manage it. The two call sites are
[`mesh_runtime.py:144-149`](src/dgx_monarch/mesh_runtime.py#L144-L149 "anchor:attach_once")
and
[`worker_loop.py:15-60`](src/dgx_monarch/cli/worker_loop.py#L15-L60 "anchor:main").
`docs/VALIDATION.md` records the dependency versions tested for this limitation.

A client that can reach those listeners can execute actor code on a worker.
Bind the complete Monarch transport surface to a dedicated fabric and
source-restrict all traffic on that interface to the trusted driver and
cluster peers. A rule covering port 26600 alone is insufficient: 26600 is only
the Worker service listener, and spawned actors allocate secondary dynamic
ports. Do not expose the interface to a shared LAN, Wi-Fi, or the public
Internet.

Only after that boundary exists, set `transport_security = "trusted_fabric"`
under `[cluster]`. It is an operator acknowledgement of an external boundary,
not a security mechanism and not a protocol feature. Without it, Worker service
startup, service installation and cluster attach refuse and `dgxm doctor` fails
its transport row; none of them adds authentication. Networked configs must
also name concrete unicast addresses. Config rejects unspecified (`0.0.0.0`,
`::`) and multicast binds rather than treat them as a trusted-fabric
configuration, and rejects IPv4-mapped IPv6 spellings, so an IPv6 object cannot
bypass the IPv4 classification. The client-owned cluster smoke uses these same
unauthenticated actor APIs. Its source, status, and teardown checks assess
runtime correctness; they do not authenticate peers.

This section is the authoritative fabric-trust reference (docs/DESIGN.md §7).
Other documentation links here.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the repository-wide trust
boundaries and severity model.
