# DGX Monarch Studio (draft)

Studio is an optional Svelte interface served by a separate Python companion.
The companion uses the existing ComfyUI queue and Monarch nodes. It does not
load models, own a mesh, schedule GPU work, or change ComfyUI source. The CLI,
TUI and ComfyUI panel remain available independently.

This implementation is a draft against v1.2.0 source commit
`5604ac77799f1f3e1a077a2f841dfce59fbcb892`. The full agreed product contract is
[STUDIO_SPEC.md](STUDIO_SPEC.md); the [acceptance record](STUDIO_ACCEPTANCE.md)
identifies incomplete features and unperformed checks. An inventory row is
evidence of source coverage, not proof of a working Studio operation.
The [measurement record](STUDIO_PERFORMANCE.md) separates packaged asset sizes
from unperformed host and browser performance acceptance.

The [interface QA record](STUDIO_INTERFACE_QA.md) distinguishes control-source
inspection from browser checks, including phone and tablet viewport layouts.
Physical-device and comprehensive accessibility acceptance remain open.

## Install and start

Follow [the existing installation procedure](INSTALL.md) with the `studio`
extra added to the selected source installation. It adds `aiohttp` and Pillow; the
normal Torch, Monarch and xFuser constraints still apply. Node is needed only
when rebuilding the frontend, not when installing or running a built checkout,
wheel or source archive.

```sh
dgxm-studio --check
```

`dgxm-studio --check` compares every packaged Studio file with its recorded
SHA-256 and reports whether the studio extra is installed; it needs neither the
extra nor a driver.

Tell Studio once where ComfyUI listens, in the `[studio]` table of
cluster.toml, then start it on the computer that runs ComfyUI:

```toml
[studio]
driver = "http://127.0.0.1:8188"
```

```sh
dgxm studio
```

`dgxm studio` reads the `[studio]` table of the cluster.toml that `dgxm` uses,
starts the companion and signs your browser in: when this computer has a
display it sends the browser a sign-in page, and otherwise it prints a sign-in
link. The first start needs the driver address, from `[studio] driver` or
`--driver`; Studio keeps its records for the first driver address it uses, so
use the address your ComfyUI listens on.
[Configure Studio in cluster.toml](#configure-studio-in-clustertoml) lists the
keys. `dgxm studio` runs the same companion as `dgxm-studio`, with explicit
flags. `dgxm-studio` itself stays available for another port, an IPv6 bind or
a `--token-file`, and takes its driver from `--driver`.

When standard output is a terminal, the companion prints the owner token there,
and prints the sign-in link `http://127.0.0.1:<port>/#token=<token>` on a line
of its own only when it did not send the sign-in page to a browser. Otherwise,
for example when output goes to a file, a pipe or a service journal, it writes
the token to `owner-token` and the sign-in page to `sign-in.html` (both mode
0600) in the state directory and prints only their paths. The companion removes
both files when it exits. While it runs, `dgxm studio open` turns them into a
link. `--token-file /absolute/path`, a `dgxm-studio` option, writes the token
to another file instead, whose directory must belong to you with mode 0700.
Treat the token as a local password. It changes on each launch, and it can land
in terminal scrollback, the token file and the browser's session storage. The
sign-in link holds the token, and the browser keeps it in its history; treat it
like a password. A Studio started with `--allow-management` never makes a link:
paste its token into the form.

The companion starts a browser only on a terminal with a display, outside an
SSH session and outside tmux or screen, and only when the desktop opens `.html`
files and `http` links with the same program. It hands that browser the path of
`sign-in.html`, never the token, and prints
`Sent the sign-in page to your web browser.` Inside tmux or screen, or in an
SSH session, nothing opens: click or paste the printed link. On a desktop where
`.html` files open in another program than the web browser, nothing opens
either, and the companion prints the `xdg-mime default` line that fixes it.
`--no-open` prints the link and never starts a browser.

Studio binds only to a loopback IP, verifies Host and Origin, requires bearer
authentication for every API request and disables cross-origin access. It does
not open a public port or configure a reverse proxy. It installs a service only
when you run `dgxm studio up`, and that service is one user unit for the
companion itself ([Keep Studio running](#keep-studio-running)). The Host
check accepts only the literal bind address and port, so `localhost` is refused.
For a remote browser, the one working shape is an SSH tunnel to the same port,
`ssh -L 127.0.0.1:<port>:127.0.0.1:<port> <host>`, browsed at `http://127.0.0.1:<port>`.
`dgxm studio tunnel <host>` prints that line and the address to browse
([Reach Studio from another computer](#reach-studio-from-another-computer)).
TLS termination, Host rewriting and a different local port are not supported;
the companion refuses their Host or Origin. ComfyUI itself must stay on a
trusted local connection or behind HTTPS with a valid server certificate. The
companion sends no credentials to the driver, so a front end that requires
client credentials refuses it.
Each state directory is bound to one canonical driver origin. A different
driver requires a separate `--state-dir`; this prevents old artifact names from
being served by an unrelated driver. Existing unbound draft databases adopt the
selected origin on their first upgraded start, so use their original driver.
`[studio] driver` must already be in that canonical form and must name this
computer, and a typed `dgxm studio --driver` must be canonical too, so the
string you write is the one Studio binds its records to. When a launch asks for
another driver than the one the state directory is bound to, the companion
refuses and prints two more lines: the origin the directory is bound to and the
one the launch asked for, then an `mv` command that moves the old records aside
under a name that does not exist yet, so the next start makes a fresh state
directory. To keep the old records, export the library from the old directory
and import the export into the new one.

Pass an existing `--config /absolute/path/cluster.toml` to use the supported
cluster reporting actions; a relative `--config` resolves against the launch
directory at startup. `dgxm studio` passes the cluster.toml that `dgxm` itself
finds, made absolute, prints its path, and says when it came from the working
directory. `dgxm studio --config PATH` works too. Management actions additionally require
`--allow-management` and an explicit review/apply confirmation in the browser.
Without the flag only these read-only actions run: version, status, doctor, reap
preview, gate inspect, gate repair preview, top replay, diagnostics and
diagnostics capabilities. A diagnostics request that names an output path writes
a file, so it counts as a management action.

Without `--allow-management`, the owner token controls jobs, outputs, saved
records and the read-only actions above. With it, setup and init run a browser-supplied interpreter on every
host and update runs a browser-named checkout, so the token is code execution as
the worker account on every host;
[THREAT_MODEL.md](THREAT_MODEL.md#browser-to-studio-companion) records its scope
and where it can land. Browser JSON never becomes command text. Read the exact
operation description before applying it. Worker lifecycle, client-owned actor
recycle, stock driver `/free`, and ComfyUI driver lifecycle are separate
actions. Backend idle, consent, refusal and recovery rules remain authoritative.

Studio runs each management command with `python -P -s -B`, so a dgx_monarch
directory in the launch directory cannot replace the installed package, and it
records the version the command itself printed. A doctor repair receipt path
must be absolute. An exit code of 75 (an unanswered critical probe) records
unknown, not failed. A 4xx refusal from the driver's consent or recycle route
records failed with the driver's short reason; a 5xx or a lost response records
unknown. Consent apply reads the driver's pending consent requests at review.
Leave key and id empty to list them; with one pending request the review selects
it. Studio offers accept only; dismiss, revoke and the automatic rescue setting
stay in the ComfyUI panel. The driver's waiver ledger records a Studio accept
with `consent_source` set to `studio`, where a panel click records `panel`. A
Sol install review with no plan token reuses the
staging of a live review of the same plan. Each review first removes, on every
host, staging that only expired or replaced Studio reviews used; staging an
apply used and staging for a token you supplied stay.

The optional `--driver-service` setting names an already installed owner-managed
ComfyUI systemd user unit. It neither installs a unit nor grants permissions.
`[studio] driver_service` passes it, and also orders the Studio unit after
that driver unit ([Start Studio with the driver](#start-studio-with-the-driver)).
The companion refuses unrecognized driver ownership and does not interpret
Worker-service controls as driver controls. The selected unit must serve the
`--driver` origin: Studio reads its ComfyUI `--listen`, `--port` and TLS options
(ComfyUI's default is 127.0.0.1 port 8188) and refuses a unit that serves
another address, port or scheme. A wildcard listen address matches only a
loopback or local origin. Management controls expose typed review plans and
retained receipts.

Update needs both `--config` and `--driver-service`, so in Studio it needs
ComfyUI to run from an existing systemd user unit rather than the foreground
launcher; `dgxm update` on the command line has no such requirement. Without
either setting, Studio lists update as unavailable and names the missing one. It
refuses a checkout other than the one the driver unit's ComfyUI imports, the
target of its `custom_nodes/dgx-monarch`, and refuses when the companion's own
dgx_monarch source differs from that checkout. The default verified mode runs
the existing verified transaction. Legacy mode runs the original in-place
update, and its review first refuses a cluster whose Workers run from release
slots. Both modes run in a frozen controller with the driver unit's interpreter.
Studio stops the driver unit first and starts it again only after the update
succeeds or the backend refuses before it changes anything; any other outcome
leaves the driver stopped and keeps the controller and the backend's reason for
recovery.

The generated inventory maps all 126 CLI command and option rows. 101 map to
typed actions, parameters, selectors, launch context or presentation behavior.
The 25 rows of `dgxm studio` start, stop, open or reach the companion itself;
the inventory waives them with that reason.
Diagnostics capture and capabilities, last-gate inspection, reap grace,
Worker synchronization choices and recording/replay controls have explicit
mappings. Inventory validation rejects missing mappings and mismatched
advertised defaults or choices. Panel handlers and TUI bindings meet the same
closed check: each maps to a Studio operation, route or equivalent control, or
carries a written waiver. Node schema and node input rows record source
coverage only, because Studio reads them from the driver's live object_info. Live availability still depends on the
selected configuration, permissions and backend guards.

## Configure Studio in cluster.toml

`dgxm studio` reads an optional `[studio]` table from the cluster.toml that
every `dgxm` command uses: `--config` (before or after the word `studio`),
else `$DGXM_CLUSTER_TOML`, else `./cluster.toml`, else
`~/.config/dgx-monarch/cluster.toml`. It never probes ports or looks for a
driver. Before anything else it prints the file and where the driver origin
came from:

```text
Config: /home/user/.config/dgx-monarch/cluster.toml
Driver: http://127.0.0.1:8191 (from [studio] driver)
```

A file found in the working directory reads
`Config: /home/user/work/cluster.toml (from the working directory)`, and a
typed `--driver` reads `(from --driver)`.

```toml
[studio]
driver = "http://127.0.0.1:8191"
port = 8220
driver_service = "comfyui-driver.service"
driver_command = "bash /home/user/start-comfyui.sh"
```

| Key | Default | Meaning and rule |
|---|---|---|
| `driver` | none | The ComfyUI driver origin on this computer. An `http` or `https` origin with a host and nothing else: no user, password, path (not even `/`), query, fragment or `%`. The host is a `127.0.0.0/8` address, `[::1]` or `localhost`; a listen address such as `0.0.0.0` is refused. It must already be in canonical form (lowercase host, no default port), and a refusal names the canonical spelling. |
| `port` | `8220` | Studio's own loopback port, 1024 to 65535. The SSH tunnel forwards this port. |
| `driver_service` | none | A ComfyUI systemd user unit that the Studio unit starts first ([Start Studio with the driver](#start-studio-with-the-driver)). It must name a `.service` unit, not a Worker unit and not `dgxm-studio.service`. |
| `driver_command` | none | Text the Driver not running panel shows as the command that starts your driver ([When the driver is not running](#when-the-driver-is-not-running)). 1 to 512 printable ASCII characters. Studio never runs it, and every token holder can read it, so it must hold no secret. |

Prefer `127.0.0.1` to `localhost`; they are different origins to Studio. A
driver on another computer needs a typed `dgxm studio --driver <origin>` or
`dgxm-studio --driver`: a cluster.toml can be picked up from the working
directory, so a file may not send Studio's prompts and uploads to another
computer. There is no key for management, the state directory, the bind host or
a token file. Management stays the typed `--allow-management` flag, and the
bind host is always `127.0.0.1`.

Only `dgxm studio` commands check the table. Every other reader, the driver's
nodes and the Workers included, admits it and ignores its contents, so a typo
there never stops a render. A broken table exits 2 with a message that names
the file and the key; [CLUSTER.md](CLUSTER.md#studio-optional) lists the same
rules.

Studio reads cluster.toml only when it is a regular file you own that no other
account can write; a group that holds only you counts as yours.
`dgxm studio up` takes no file from the working directory: a service outlives
the shell that chose it, so pass `--config` with the path to use such a file. A
cluster.toml without `[[hosts]]` still gives Studio its table; cluster actions
then stay unavailable, and Studio says so.

The first start on a fresh state directory needs `[studio] driver` or
`--driver`. Studio refuses to guess, because it keeps its records for the first
driver address it uses. A state directory already bound to
`http://127.0.0.1:8188`, the `dgxm-studio --driver` default, keeps starting
without one.

Adding the table has three consequences:

1. Deploy order. Releases before this one refuse a cluster.toml that holds
   `[studio]`, in every reader: the CLI, the driver's Init node, mesh setup,
   service observations, the TUI and Studio's management actions. Add the
   table only after every checkout that reads the file runs this release and
   the driver has restarted on it, and when no render runs.
2. A byte edit retires the fleet. In cluster mode the Init node fingerprints
   the file's bytes on every Queue, and any byte change, a comment included,
   retires the old fleet before attaching a replacement. Edit `[studio]` when
   no render runs.
3. Only guided setup keeps the table. `dgxm setup --apply` keeps `[studio]`
   when it rewrites the file ([CLUSTER](CLUSTER.md#re-running-setup)).
   `dgxm init` and Studio's init action write a new file without `[studio]`,
   as they drop `[monarch]` and `[fabric.*]`; the reviewed diff shows the
   removal. After either, add the table again. An installed Studio service
   keeps the values it was written with, and `dgxm studio open` names the
   fields that differ.

## Keep Studio running

`dgxm studio up` installs `dgxm-studio.service`, a systemd user unit that runs
the companion with the settings of the `[studio]` table, enables it, and starts
or restarts it. Run it from the environment and with the config you want the
service to use:

```sh
dgxm studio up
```

On a terminal it prints:

```text
Config: /home/user/.config/dgx-monarch/cluster.toml
Driver: http://127.0.0.1:8191 (from [studio] driver)
Interpreter: /home/user/monarch-env/bin/python
Source: /home/user/projects/dgx-monarch/src
Wrote /home/user/.config/systemd/user/dgxm-studio.service (starts with comfyui-driver.service)
Started dgxm-studio.service on http://127.0.0.1:8220.
Also started comfyui-driver.service: the Studio service wants it.
Sign-in link:
  http://127.0.0.1:8220/#token=<token>
Run dgxm studio or dgxm studio open to open it in a browser.
```

`Wrote` reads `Unchanged` when the unit already holds the same text. `up` never
starts a browser, and without a terminal it prints the token file's path in
place of the link. Nothing is written until every check has passed: the config,
the interpreter and source paths, the state directory's driver binding, a
preflight run of the exact service command with `--check`, the driver unit, the
unit path and its drop-ins, a companion already running on the state
directory, and the port. `up` then waits up to 30 seconds for the service to
write its token file and sign-in page. A start that fails prints the service's
journal lines and leaves the unit installed and enabled.

The other actions:

- `dgxm studio` while the service runs acts as `dgxm studio open`. It refuses
  `--driver`, `--state-dir` and `--allow-management` then, because they would
  not reach the service.
- `dgxm studio open` prints the sign-in link of the running Studio and opens it
  when this session has a display. While the service is starting it waits up to
  30 seconds for a sign-in page newer than the service's start.
- `dgxm studio token` prints where the token file is. `--value` also prints the
  token itself, on a terminal only. `--rotate` restarts the service for a new
  token; tabs signed in with the old one need the new link.
- `dgxm studio down` stops, disables and removes the unit and leaves Studio's
  records in the state directory; `dgxm studio up` starts it again. It never
  stops the driver unit.

`up` always restarts the service, so every `up` issues a new token. Run
`dgxm studio up` after `dgxm update`, which does not restart Studio. A
long-running service imports some modules on first use, so after an update
replaces the code on disk it can mix old and new code until it restarts.
`dgxm studio open` says when the code on disk changed after the service
started, and names the settings that differ from what `up` would write now.

The unit text, with `driver_service` set and without management:

```ini
# dgxm-managed-studio-unit-v1
# dgxm-studio-state=/home/user/.local/state/dgx-monarch/studio
# dgxm-studio-port=8220
# dgxm-studio-driver=http://127.0.0.1:8191
# dgxm-studio-config=/home/user/.config/dgx-monarch/cluster.toml
# dgxm-studio-driver-service=comfyui-driver.service
# dgxm-studio-management=no
# dgxm-studio-interpreter=/home/user/monarch-env/bin/python
# dgxm-studio-pythonpath=/home/user/projects/dgx-monarch/src:/home/user/monarch-env/lib/python3.12/site-packages
# dgxm-studio-source=<64 hex characters>
[Unit]
Description=DGX Monarch Studio companion
Documentation=https://github.com/Deen-Media/dgx-monarch/blob/main/docs/STUDIO.md#keep-studio-running
Wants=comfyui-driver.service
After=comfyui-driver.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
UnsetEnvironment=PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS LD_ASSUME_KERNEL LD_AUDIT LD_BIND_NOT LD_BIND_NOW LD_DEBUG LD_DEBUG_OUTPUT LD_DYNAMIC_WEAK LD_HWCAP_MASK LD_LIBRARY_PATH LD_ORIGIN_PATH LD_POINTER_GUARD LD_PREFER_MAP_32BIT_EXEC LD_PRELOAD LD_PROFILE LD_PROFILE_OUTPUT LD_SHOW_AUXV LD_TRACE_LOADED_OBJECTS LD_TRACE_PRELINKING LD_USE_LOAD_BIAS LD_VERBOSE LD_WARN GLIBC_TUNABLES GCONV_PATH LOCPATH NLSPATH
WorkingDirectory=/
ExecStart="/usr/bin/env" "-i" "HOME=%h" "USER=%u" "LOGNAME=%u" "XDG_RUNTIME_DIR=%t" "PATH=/usr/bin:/bin" "LANG=C.UTF-8" "PYTHONPATH=/home/user/projects/dgx-monarch/src:/home/user/monarch-env/lib/python3.12/site-packages" "PYTHONNOUSERSITE=1" "PYTHONSAFEPATH=1" "PYTHONDONTWRITEBYTECODE=1" "PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647" "/home/user/monarch-env/bin/python" "-S" "-P" "-s" "-B" "-m" "dgx_monarch.studio" "--host" "127.0.0.1" "--port" "8220" "--driver" "http://127.0.0.1:8191" "--state-dir" "/home/user/.local/state/dgx-monarch/studio" "--config" "/home/user/.config/dgx-monarch/cluster.toml" "--driver-service" "comfyui-driver.service" "--no-open"
UMask=0077
NoNewPrivileges=yes
Restart=on-failure
RestartSec=3
KillMode=control-group
SendSIGKILL=yes
TimeoutStopSec=75

[Install]
WantedBy=default.target
```

Line by line:

- The first line marks a unit dgxm wrote. `up` and `down` refuse any other file
  at that path, a symbolic link, and any file in a `dgxm-studio.service.d` or
  `dgxm-.service.d` drop-in directory. After the reload, `up` reads back where
  systemd loaded the unit from and refuses a unit that systemd loads from
  another file or with a drop-in from a directory this account or the
  administrator writes; a vendor drop-in under `/usr/lib/systemd/user` only
  prints a note.
- The `# dgxm-studio-` lines record the settings; systemd ignores them. `open`,
  `token` and `tunnel` read the state directory and port from them, and `open`
  compares them with what `up` would write now. `# dgxm-studio-source=` is a
  digest of the dgx_monarch source the service imports.
- `Wants=` and `After=` appear only with `[studio] driver_service`.
- `StartLimitIntervalSec=60` and `StartLimitBurst=5` stop a restart loop after
  five failed starts in a minute; `up` clears a tripped limit with
  `systemctl --user reset-failed` before it restarts the unit.
- `UnsetEnvironment=` and `/usr/bin/env -i` empty the environment, as in the
  hardened Worker unit. The service gets no `DISPLAY`, so it never starts a
  browser, and no `XDG_STATE_HOME`, `DGXM_CLUSTER_TOML` or value that
  `systemctl --user set-environment` placed in the user manager.
  `XDG_RUNTIME_DIR=%t` lets the companion reach the user manager for
  `--driver-service`.
- `PYTHONPATH` names the source and dependency directories of the `dgxm` that
  ran `up`, so the service imports the same checkout as that `dgxm`. On an
  editable install that is the checkout the driver imports, which Studio update
  requires. `up` refuses while `PYTHONPATH` is set in its own environment, so a
  service started at boot never runs a worktree by accident, and it refuses an
  interpreter, source or dependency path that another account can change.
- `WorkingDirectory=/` matches the preflight run of the service command.
  `NoNewPrivileges=yes` and `UMask=0077` need no privilege.
- `KillMode=control-group` ends every process the companion started when the
  service stops. `TimeoutStopSec=75` exceeds aiohttp's 60-second graceful
  shutdown, so a normal stop removes the token file and the sign-in page.
- Every companion flag is explicit and absolute. `--no-open` is always there;
  `--allow-management` appears only after `dgxm studio up --allow-management`.
  The unit holds no token.

Linger. Without linger the service runs only while you are logged in: it starts
at your first login and stops when your last session ends, and `up` prints a
note. To keep it running from boot, run `sudo loginctl enable-linger $USER`.

Journal. The unit's output goes to the user journal: at each start the plain
address, the token file path and the sign-in page path, plus any startup
refusal. The token, the link and the content of `sign-in.html` never reach it.
`journalctl --user -u dgxm-studio.service -n 20` shows the latest lines.

Lock and busy check. `up`, `down` and `token --rotate` take a per-user lock,
`~/.local/state/dgx-monarch/studio-unit.lock`, against each other. Before they
stop or restart the service, they take `operations.lock` in its state
directory, which the companion holds for each review it prepares and each
operation it runs, from the moment it accepts one until it settles. While a
review or operation holds it they refuse; once they hold it, the companion
refuses new reviews and operations with `operation_unavailable` until the
restart or stop has finished. They create that file when it is missing, as
they create `studio-unit.lock`, and a command that ends for any reason releases
both. They also read the service's control group and refuse while any process
other than the companion runs there, such as an update controller, a
management command or a status report, and print one line per process. Wait for
it to finish, or stop the service yourself with
`systemctl --user stop dgxm-studio.service`. `down` removes the unit file only
after systemd reports the service stopped; when the stop fails it keeps the
file and says so.

A stop from outside dgxm, such as `systemctl --user stop`, a logout without
linger or a shutdown, ends any management command the companion started, and
so does a crash of the companion. systemd then stops the unit's whole control
group, which ends a Studio update's controller, restarts Studio three seconds
later, and `Wants=` starts the driver unit again, possibly on a checkout whose
update was cut off. Studio records the interrupted operation as unknown; check
the checkout and the cluster before you render again.

Run `dgxm studio down` before `dgxm uninstall`; uninstall refuses while the
unit is installed. The page's own uninstall operation, under
`--allow-management`, makes no such check and leaves the Studio unit in place.

`dgxm studio up --allow-management` writes `--allow-management` into the unit.
Its token is then code execution as the worker account on every host for the
life of the service, and the token file is a standing credential until the
service restarts. Such a service never signs in by link: `up`, `open` and
`token --value` print the token on a terminal, and the page takes it only
through its form. Running `up` again without the flag removes it, and `up`
says so.

A foreground `dgxm studio --state-dir /absolute/path` is invisible to `open`,
`token` and `tunnel`, which read the unit's state directory or the default
one. On a terminal it prints its own link or token.

## Reach Studio from another computer

The companion listens on `127.0.0.1` only, and its Host check accepts only the
literal `127.0.0.1:<port>` it bound. A browser on another computer reaches it
through an SSH local forward of the same port to the same address. The first
time, from a laptop:

1. On the laptop, open an SSH session with the forward:
   `ssh -o ExitOnForwardFailure=yes -L 127.0.0.1:8220:127.0.0.1:8220 user@studio-host`.
2. In that session, run `dgxm studio`, or nothing new when the service already
   runs: `dgxm studio` then acts as `dgxm studio open`. The session is an SSH
   session, so Studio prints the sign-in link and the line to run on the
   laptop, which is harmless when the forward already exists.
3. Click or paste the link in the laptop's browser. It loads
   `http://127.0.0.1:8220/` through the forward, takes the token from the link
   and connects.

`dgxm studio tunnel [HOST]` prints that line. Run inside the SSH session on the
Studio host, it fills in the address you connected to and the running Studio's
port, and prints the sign-in link:

```text
Port: 8220 (from the running Studio)
On the computer with the browser, run:
  ssh -o ExitOnForwardFailure=yes -L 127.0.0.1:8220:127.0.0.1:8220 user@192.0.2.10
If you reach this host another way, add -L 127.0.0.1:8220:127.0.0.1:8220 to the ssh command you use, or set a local forward in your SSH app that listens on 127.0.0.1 port 8220 and goes to 127.0.0.1:8220.
Then browse: http://127.0.0.1:8220/
Sign in with this link in that browser:
  http://127.0.0.1:8220/#token=<token>
Keep 127.0.0.1 and the same port on both ends; Studio refuses localhost and any other port.
If ssh says "Address already in use" or "Could not request local forwarding", another program on that computer holds the port: close the session if it opened, free the port, and do not open the sign-in link until the tunnel works, because a page served on that port would read the token from the link.
```

With a named host, on another computer, without a terminal, with `--run`, or
when no Studio runs, the two sign-in lines become one that points at
`dgxm studio open` in an SSH session on the Studio host; for a management
service it points at `dgxm studio token --value`. `Then browse:` and the
"another way" line always print: a jump host, a router port forward, a stale
SSH session variable in an old tmux pane or a phone SSH app can make the
derived line wrong.

- `HOST` is a name, an SSH config alias, an IP address or `user@host`. Spaces,
  shell characters, an IPv6 zone (`%`) and a leading `-` are refused, so the
  printed line is safe to paste into a shell. With a named host, `tunnel`
  never reads this computer's Studio state, so it can run on the browser
  computer too.
- The port comes from `--port`, then from the running Studio (only for the host
  read from the SSH session), then from `[studio] port`, then 8220. Each source
  is named on the `Port:` line.
- `--run` runs the ssh command itself, with no shell, after it prints the
  other lines, except `On the computer with the browser, run:`, to standard
  error, and never prints a link. It needs `ssh` on `PATH`
  and dgx-monarch on the browser computer, such as a Linux laptop or a second
  Spark; most laptops paste the printed line. Its exit code is ssh's, and
  Ctrl+C closes the tunnel.
- The printed command has no `-N`, so the same session gives a shell on the
  Studio host, where `dgxm studio open` prints the link.

The printed command names `127.0.0.1` as the address the forward listens on
and carries `-o ExitOnForwardFailure=yes`: when another program on the browser
computer already holds that port, ssh prints `Address already in use` and
`Could not request local forwarding` and exits instead of opening the session
without the forward. A forward written without that address, as in many ssh
commands and SSH apps set up another way, listens on both `127.0.0.1` and
`::1`, opens the session when either one is free, and prints only the
`Address already in use` line; the last line of the output is for that case.
A page served on that port then reads the token from the link, so do not open
the link after either message; free the port and connect again. The token crosses the SSH connection inside the terminal
session and lands in the laptop terminal's scrollback and the laptop browser,
its history included.

Out of scope: a `--trusted-origin` list, Tailscale Serve and a LAN bind. Studio
sets none of them up, and a different local port or `localhost` in the address
bar fails the Host check.

## Start Studio with the driver

With `[studio] driver_service`, the Studio unit declares `Wants=` and `After=`
on that ComfyUI user unit. Starting Studio, at boot with linger or through
`up` or `--rotate`, then starts the driver first, and `http://127.0.0.1:8220`,
opened with `dgxm studio`, is the one address you need. With linger, the
enabled Studio unit pulls the driver in at boot even when the driver unit is
not enabled itself. The driver unit's file is never edited, and there is no
`Requires=`, `BindsTo=` or `PartOf=`: stopping the driver leaves Studio
running with the Driver not running panel, and `dgxm studio down` never stops
the driver. A restart of Studio starts a stopped driver unit again, so a driver
you stopped to free memory comes back; `up` and `--rotate` then print
`Also started <unit>: the Studio service wants it.`

Studio's driver controls and Studio update accept a driver unit that runs an
absolute Python and an absolute ComfyUI `main.py` directly, with no
`ExecStartPre` or other extra commands, no `EnvironmentFile=`,
`KillMode=mixed`, no dependency on another `.service`, and a name without
`worker`. This one copies the exports and the arguments of
`scripts/comfy-driver.sh`. Replace `/home/user` and the interpreter with yours:

```ini
# ~/.config/systemd/user/comfyui-driver.service
[Unit]
Description=ComfyUI driver with DGX Monarch

[Service]
WorkingDirectory=/home/user/ComfyUI
Environment=CUDA_CACHE_MAXSIZE=4294967296 CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=16
Environment=TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas DGXM_CLUSTER_TOML=/home/user/.config/dgx-monarch/cluster.toml
UnsetEnvironment=NCCL_P2P_DISABLE NCCL_PROTO
ExecStart=/home/user/monarch-env/bin/python /home/user/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-pinned-memory --disable-async-offload --reserve-vram 2 --bf16-vae --bf16-text-enc --dont-upcast-attention --disable-auto-launch
KillMode=mixed
Restart=on-failure

[Install]
WantedBy=default.target
```

Then add the table, reload the user manager and install Studio:

```toml
[studio]
driver = "http://127.0.0.1:8188"
driver_service = "comfyui-driver.service"
```

```sh
systemctl --user daemon-reload
dgxm studio up
sudo loginctl enable-linger $USER
```

The last command starts both at boot. Once `driver_service` is set, start
ComfyUI only through that unit; a second launch on the same port makes the
unit fail. `up` checks that the unit serves `[studio] driver` before it writes
the Studio unit, and refuses a unit that the user manager does not know.

A unit whose `ExecStart` runs a wrapper script, such as your own launcher,
works for the ordering (`Wants=` and `After=`), for the command the Driver not
running panel shows and for starting Studio. Studio cannot read the address
such a command serves, so `up` prints a note, and Studio's own driver stop,
start and restart, and Studio update, refuse the unit. A unit in the form above
is needed only for those.

## When the driver is not running

While nothing accepts a connection at the driver origin, the connected page
shows a Driver not running panel in place of the yellow notice, and the footer
and Settings read `not running`. The panel names the origin and shows the
command that starts the driver with a Copy button and the place the command
came from. It also says that if ComfyUI already runs at another address,
`driver` in the `[studio]` table must name that address, and that the
records belong to the current origin, so a new address needs fresh records
([Install and start](#install-and-start)). Generate and Review settings stay
disabled until the driver answers. Studio checks every few seconds and connects by
itself when the driver answers: the workspace reads the catalog and inspects
the restored draft again, with no click.

The command, first match wins:

1. `[studio] driver_service`: `systemctl --user start <unit>`, with the
   journal command to read if the start fails.
2. `[studio] driver_command` in the companion's `--config`, shown verbatim
   with the config path it came from. Studio reads it when it starts, so restart
   Studio after an edit, and edit cluster.toml when no render runs.
3. The repository launcher, with the Python that runs Studio, when the origin
   is on this computer and a ComfyUI checkout is known: dgx-monarch is
   installed under `<comfy>/custom_nodes` as [INSTALL.md](INSTALL.md)
   describes, or `[cluster] comfy_dir` names the checkout. The command is
   `bash <script> --comfy-dir <comfy> --python <python> --listen 127.0.0.1 --port <port> --no-browser`,
   where `<script>` is that checkout's `scripts/comfy-driver.sh`, plus
   `--cluster-config <config>` when the companion has `--config`. Run it in a
   terminal you keep open.
4. Otherwise the panel asks you to start your driver so it listens at the
   origin, and suggests `driver_command` to show your own command there.

When Studio refuses the companion's `--config`, for example because another
account can write it, neither 2 nor 3 applies, since the launcher command
would pass that file on; the panel shows the text of 4.

With `--allow-management` and `--driver-service` the panel adds that
Operations can start the driver too. Studio never runs the command: the page
shows it as text, and only a token holder reads it, with any home paths in it.
The catalog and workflow inspection answer HTTP 503 with the code
`driver_not_running`, which Studio uses only for a refused connection; a
timeout or a DNS or TLS failure stays unavailable with `driver_unknown`. A job
dispatched while no driver listens still settles `unknown`, which is why the
page disables Generate and Review settings.

## Workspace controls and media

Use the navigation rail or Ctrl/Cmd+K to open a view. The rail lists Create,
Gallery, Queue, Utilities, Library and Settings. Utilities holds Grids, Fleet
and Operations as tabs, each with one line on what it is for, and opens on the
tab you used last, Grids at first. The palette lists each tab by name under
Utilities, and a button that opens one, such as Grant and retry, opens that tab.
The command palette takes keyboard focus immediately; Tab chooses a result and
Enter opens it. It also
lists operations to review and, once you type, packaged workflows to load, as
[Working with workflows](#working-with-workflows) describes. Escape
closes dialogs or leaves presentation mode. In Recent Outputs, Left and Right
on a focused output select the previous or next one and show it in the viewer,
as a click does. Arrow keys keep their usual use in text fields, selects, the
viewer frame and video controls. Project selectors include the full
bounded project catalog, including projects outside the recent snapshot.
The Telemetry and Console buttons in the footer show and hide the Spark
telemetry panel and the event console, and Settings has a checkbox for each.
The browser keeps that choice, and a saved session records it.

The telemetry panel lists every host in the driver's latest sample side by
side: the driver's own box and each Worker's, with what runs there, such as
Driver · Worker rank 0 of 2. Each host shows available memory against its
total, GPU memory, GPU utilization, GPU power and GPU temperature, with the
names, units and precision of the shared presentation contract. An unknown
value reads unknown, and memory categories are never added into one total.
Available memory above 20 GiB carries a green mark and 20 GiB or less a yellow
one, the display rule `dgxm top` and the ComfyUI panel use; the mark decides
nothing. A host reading more than 3 seconds older than the sample says Stale
reading. When the companion's telemetry polls fail it keeps the last sample,
and when Studio loses the companion it keeps the last snapshot. The panel and
the Fleet page then say so above the hosts and label each host and readiness
row stale, with its age on the companion's clock when the companion still
answers. Below the hosts, Worker service, Attached mesh and Render session show
the driver's readiness report as the ComfyUI panel's card does, with an active
layer in yellow. Workers report their hosts only while a Monarch render's mesh
is attached, so with no mesh attached, for example before the first Monarch
render after the driver starts, the panel lists the driver's host alone and
says why. Metric history charts one series at a time. Host picks a host or
Render progress, the driver's step count, step time and last render duration;
Metric picks one of that host's metrics, with a rail rate for each rail. A
choice with a single option shows as text, and the chart opens on GPU
utilization.

The telemetry panel keeps the job rows, a Readiness line and both hosts in view.
Readiness, under the driver's state, gives the worst state among the readiness
layers, such as active, and names the layers in it; it never reads better than
the driver's own verdict. Detailed telemetry, under the hosts, holds the
readiness rows with the driver's sentences, the metric history with its Host and
Metric choices, sample inspection, the sample table, export and the raw host
sample. It starts closed, and the browser keeps it open or closed with the other
layout preferences. The Fleet page shows all of it at once.

Phone layouts use a scrolling bottom navigation rail, larger touch controls and
stacked panels. Expanded viewer details remain reachable without clipping the
output area. Dialogs and disclosures use short transitions; generation adds a
subtle activity border around the viewer. Reduced-motion preferences disable
these animations and the activity logo.

The output viewer follows one job: the latest one this tab submitted, or the
newest active job when this tab has submitted none. A status line under the
viewer title names its state: Submitting, Queued, Running, Cancel requested,
Finished, Failed, Cancelled or Outcome unknown. A running job shows the step
count and node from the driver's latest progress report, with a bar that fills
to that count. When ComfyUI starts the next node, the companion clears that
count, so neither the status line nor the Queue row shows one node's steps while
another runs; many nodes, such as a VAE decode, report none. When no step count
applies, the bar sweeps and never fills; with reduced motion it stays still and
dim. Studio never merges the steps of different nodes into one share. Running
time counts on the companion's clock from when the companion first saw the job
run, not from the driver's start time, since the driver may run on another host
with its own clock; a job that ended shows the driver's own time from start to
end. Running time needs the companion's start time and its clock, and a total
needs both driver times; when one is missing no time shows, and Studio never
estimates the time left. A failed job shows the first sentence of its reason and
a link to Queue, and
[Read a failure and grant a consent](#read-a-failure-and-grant-a-consent) says
what else it shows. An unknown outcome links to Queue, where you can reconcile
it. Either link opens Queue on that job's row, expanded and scrolled into view.
Queue loads its newest history first; when that leaves the job out, Queue reads
the one job by its identity and shows its row above the history pages. A link
applies once, so a later visit to Queue opens on no row. Create counts unknown
outcomes with active jobs, as Queue does, until you reconcile them; only jobs in
flight animate the viewer border and the logo.

When the viewer begins to follow a job in flight, such as one you just
submitted, it stops showing the earlier output, its file name and its media
label. The heading names the job's workflow, by its title when it is a packaged
workflow, the status line shows its progress, and the frame says that the new
output appears there when the job finishes. A tab that submitted nothing does
the same for the newest job in flight. An open comparison stays, and so does a
result you choose while the job runs. A job that fails or is cancelled leaves
the frame without an output, and Recent Outputs keeps the earlier outputs.

When the followed job succeeds, the viewer shows its last final image or video,
unless you chose another result or a comparison after the job began, or a
comparison is open. When the snapshot leaves that output out, Studio asks for
the job's outputs, at most three times. The answer lists them most recently
stored first, as the snapshot does, so the viewer shows the same output either
way. This changes only the viewer, never the draft. A job this page submitted
counts as new even when its submit reply was lost. After a reload Studio shows
no result of a job that had already ended, so an earlier result never shows as
new. The recent outputs strip shows previews of job outputs only: uploads, masks
and live previews stay out of it, and an intermediate output carries a Preview
label. The telemetry panel names this tab's job, or the newest active job, with
its state and time, and while it runs the step count and node the status line
shows; it never ties the driver's last render to a job. The raw host sample
stays closed until you open it.

A video's preview in the strip, the Gallery and Queue is its first frame, drawn
in the browser, with its length from the video's own metadata in the corner. To
draw it Studio downloads the video once per page, two videos at a time, within
the viewer's 64 MiB limit. A tile that gets no frame, after a failed download or
for a file with no picture, says No thumbnail, with the length when known. Only
a video the browser cannot decode says Cannot play in this browser.

Over an image in the output viewer, the mouse wheel zooms in and out around the
point under the pointer, between 10% and 800% as the zoom buttons do, and the
first step from Fit starts at the size Fit shows. While the pointer is over the
image the wheel never scrolls the page. Anywhere else, over a video, or for a
sideways scroll, the wheel scrolls as before, and Fit, 1:1, the zoom buttons,
drag to pan and the arrow keys on the focused frame work as before. An image
smaller than the frame stays in its middle at any zoom.

Image previews retain their source proportions across portrait, landscape and
extreme aspect ratios. The mask canvas follows the displayed source bounds, so
painted positions map to the uploaded mask. An uploaded input or mask can be bound to
image upload widgets that read the input store. It joins that widget's choices
for the request that names it, at inspection and review; any other value outside
the driver's list is still refused. Studio refuses an empty mask before upload. Video frame controls wait for valid
metadata; native playback, mute and loop controls remain available. The viewer
reports a decode failure only for the output on screen, so an image chosen
after a video shows at once. The
[interface QA record](STUDIO_INTERFACE_QA.md) describes the tested fixtures and
browser viewports without extending those results to every device or codec.

An output's `created_at` is the time the driver's history reports it finished;
when history carries no completion time it is the time the companion imported
it. `observed_at` is always the import time, and `created_at_source` says which
applied. Width and height filters apply to uploaded inputs and masks, because
ComfyUI does not report the size of generated outputs; the gallery shows those
filters only when a loaded output carries its size. Intermediate previews have
their own status option, and uploads have their own media kinds.

## Working with workflows

Choose one of the 71 packaged workflows or import a ComfyUI UI workflow/API
prompt. Supported inputs appear as typed controls; connected inputs remain
graph connections. The Inputs tab lists the inputs of the workflow's App Mode
layout, in its order, and a workflow without one opens on All inputs. Each
packaged workflow carries a layout, and ComfyUI's App Mode shows the same one.
It lists, where the graph has them: the prompt, the negative prompt, the seed,
the image size, steps, CFG and any Flux guidance, sampler and scheduler, the
denoise or strength of an upscale, the input image, video or audio, the LoRA
strength and the model file. A graph that samples in two stages, or holds the
size on two nodes, lists each copy side by side; change them together. All inputs lists every supported
primitive field, such as the DGX Monarch Init settings, the frame count and the
VAE and text encoder files. The Layout tab changes which inputs are
prominent, their labels and their order without changing binding identity, and
its edits also update the workflow repair text. Validating unchanged source
preserves edits made in typed controls; intentional JSON changes use the edited
workflow's defaults. Changing a dynamic choice, such as the codec or encoding
mode of SaveVideo, inspects the workflow again with the draft's values: the
controls for the new choice's inputs appear, those of the old choice leave the
draft, and review accepts edits to the new inputs. Inside a subgraph, Studio
lists only the inputs of the saved choice. Multiline fields retain line breaks, and import controls
allow selecting the same file again after a success or error.

The prompt boxes read Prompt and Negative prompt. The inputs chosen under
Ordered prompt styles take those roles; otherwise the graph decides: the text
that reaches a sampler's `positive` input is the prompt, and the text that
reaches its `negative` input is the negative prompt. A node that encodes both
texts, such as Qwen Image 2.1's, gives each the role its input name says. When
several boxes share a role, each adds its input name or its node. Any other
multi-line text box, and a text box whose input name another one repeats, reads
its node's title or class and its input name, such as `CLIPTextEncode · text`,
with the node id when two would still read the same. Every other field keeps its
input name, and pointing at a name shows the field's node. The prompt box starts
taller and grows with its text up to a limit, then scrolls. On Inputs, an empty
negative prompt waits behind Add negative prompt, and the raw types, such as
STRING, INT and COMBO, show only on All inputs.

Reuse settings, Send to edit and Upscale record the output they start from as
the draft's input lineage, and each asset you upload on the Assets tab joins it.
Inputs and All inputs list each lineage id in full under Input lineage, wrapped
to fit, with a Copy button that copies the whole id and shows it when you point
at it. Clear lineage empties the list. Lineage records ancestry; the asset
controls set the input the job reads. Create's pane never scrolls sideways, at
any pane width: ids, file names and labels wrap, and fields narrow to fit. Only
the row of tabs scrolls along its own line when the pane is too narrow for all
five. On a short window the field list keeps at least 280 pixels, or 40 percent
of the window's height when that is less, and the pane scrolls instead, with
Generate pinned at its foot. A field brought into view stops above Generate,
and the prompt box grows no taller than 30 percent of the window's height.

The Workflow list names each packaged workflow by the title of its Note node,
in title order, and shows the chosen workflow's file name in small type under
the list. A template without a titled Note appears under its file name, with the
`dgx-monarch-` prefix dropped and dashes shown as spaces. Type in the box above
the list to filter it: a workflow stays listed when each word you type appears
in its title or file name, in upper or lower case and in any order. The loaded
workflow stays listed and selected while the filter hides it, and a line says
when no workflow matches. Enter loads the first match while no packaged
workflow is loaded, and Escape or Clear empties the box. In the command
palette, type part of a title or file name and choose a Load workflow row; it
opens Create and loads that workflow as the list does. Drafts, exports and
records still use the file name.

Under the list, an Output line says what a run saves. Studio reads it from the
graph's save nodes and the draft's format choices, for example "PNG image",
"PNG frames plus separate FLAC audio" or "one MP4 video with sound". Save Image
writes frames when the graph makes video. A saved video has sound when the
Create Video node that feeds it takes an audio input; when another node makes
the video, the line does not say. A preview node saves no file.

The Job category select sets the job's operation. It labels the job, and Queue,
the gallery filter and the job record keep that label. The loaded workflow
decides what runs, as the line under the select says, so Image and Workflow run
the same graph. The select lists only the operations the loaded graph supports,
read from the nodes that run. Muted and bypassed nodes do not count, and neither
does a subgraph that no running instance reaches:

- **Image** needs a node that saves or previews an image, in a graph that makes
  no video.
- **Video** needs a clip frame count, a video input, a Create Video node or a
  video output, and a node that saves or previews frames or a video. A clip
  frame count is a `length` input on a node that also sets a width and a height,
  such as a video latent.
- **Edit** needs an image, video or mask input node, such as Load Image or Load
  Image (as Mask).
- **Upscale** needs an image or video input node and a node that enlarges by a
  factor or a model, such as Upscale Image By or PiD Conditioning. A resize to a
  fixed size does not count, because it also shrinks.
- **Fleet** needs the Fleet KSampler (DGX Monarch) node.
- **Workflow** fits every graph.

Loading a workflow keeps the draft's operation when the graph supports it, and
otherwise picks the first one it supports of Fleet, Upscale, Video, Edit and
Image. A draft can still hold an operation its graph lacks, for example after
Upscale on an output of a text to image workflow. The select then marks it not
supported, and a line under it names the input the operation needs. That line
does not stop Generate. An operation you choose while a workflow loads stays as
you chose it. Until Studio has inspected the graph, it lists every operation.

MiniMax H3 and LTX graphs get a Duration control after their last size field,
or at the end of the list, on Inputs and on All inputs. These families hold one
clip's frame count on more than one node. Duration shows the count in frames
and in seconds, takes a new value in either, and writes one count to every node
that holds it. It rounds the value to the nearest count the model renders
exactly. MiniMax H3 holds the count on its conditioning node and its latent
node, renders 5 plus a multiple of 17 frames and runs at 24 fps, so 124 frames
is 5.17 s. LTX holds it on the video latent and the audio latent, renders 1 plus
a multiple of 8 frames, and takes its frame rate from its `frame_rate` inputs.
When those disagree, or a wire from anything but a primitive node sets one,
Duration shows frames only.

A clip is the set of these nodes that feed the same sampler. A graph with two
clips, or a subgraph used twice, gets one numbered Duration control for each
clip, and each writes only its own nodes. When Studio cannot find the sampler a
count feeds, it shows no control for that count. When the counts of a clip
differ, Duration shows each one, and a new value sets them all. When a wire
feeds one of them, no value can set it, so Studio shows no Duration control.
Under All inputs, each of these fields names its node, its clip when there are
several, and its unit, frames, and says how it relates to the other. Any other
clip frame count also reads frames, with its node when another frame count
shares its name.

Compilation preserves the original workflow and extension metadata. Unknown
widgets, ambiguous bindings and unsupported graph constructs refuse execution
with a repair reason. Repeated nested subgraphs have separate execution and
input-binding identities, so changing one instance leaves the others intact.
Bindings identify the root graph, host-instance path, node and typed input;
legacy definition bindings remain aliases only when they identify one instance.
Nested promoted scalar widgets follow explicit serialized mappings. Ambiguous,
missing or unsupported mappings require repair. Studio does not execute
imported widget JavaScript.

Preview resolves values and seed policy and freezes the submission. Styles
require explicit positive/negative prompt bindings. LoRA stacks fill and extend
a single supported loader chain. For a workflow without slots, select the
loader and exact connected consumers that should receive the new chain.
Core loaders require `MODEL` sockets; `DGXMonarchLoraLoader` requires
`DGXM_MODEL`. A loader with text-encoder support also requires the corresponding
`CLIP` connections. Other branches keep their original inputs. Preview freezes
the inserted node identities, connection boundaries, asset names and strengths.
Multiple ambiguous chains or incompatible typed connections refuse execution.
An invalid LoRA strength, a removed LoRA or an out-of-range inserted loader is
refused with `lora_binding` when Generate checks the draft, naming the slot and
input.
The LoRA list comes from the driver catalog Studio reads at connect. Studio
reads it again when the driver connects again or the companion restarts, and
once more after a failed read while the driver stays connected. Refresh list on
the LoRA tab reads it on demand, for example after adding LoRA files.
File names never establish model compatibility.
Execution-time guards still check the actual model and topology. A schema-only
preview is not a memory-admission, topology or identity-gate PASS.

Prompt styles change the text of the prompt input, and of the optional negative
prompt input, that you choose under Ordered prompt styles on Create. Studio
applies the styles from top to bottom, to the prompt and the negative prompt
separately. Each `{prompt}` in a style's text stands for the text so far. A
style whose text has no `{prompt}` adds that text after the text so far, with a
comma and a space between them, and an empty style text leaves the text as it
was. For example, the prompt "a red fox" with the style "oil painting" and then
the style "(({prompt}))" gives "((a red fox, oil painting))"; in the other order
it gives "((a red fox)), oil painting". Make styles in Library under styles, and
add one to the draft with Append style, which puts it at the bottom. Move up and
Move down on Create change the order. Each move is one draft change, so Undo
puts the style back. Submission review shows the effective text, and the
submission also keeps the text as it was before the styles. Studio skips a draft
entry whose `enabled` field is false; Create has no control that sets it. Style
records saved before Library dropped its Style mode select may carry a `mode`
field, `append` or `template`. Studio ignores it, and only `{prompt}` decides.

Generate checks the draft and starts the job in one action. Studio first checks
each field against the inputs the driver reports. While a field fails, Generate
stays off, and the line under it names the first such field, by the name the
field shows, and the problem, for example "Fix unet_name: Choose an available
value." It adds the field's node, such as "seed on node 10", only when another
field shows the same name. Show field opens the tab that lists the field, Inputs or All inputs, scrolls to it and
focuses it. With more than one issue, a list of them all opens under the line,
each with its own Show field, and these controls stay on while Generate and
Review settings are off. When only the workflow check reports a problem, the
line starts with Check and Generate stays on. Workflow needs attention, under the
fields, lists what the companion found when it inspected the workflow with the
draft's values. After you edit a value, Studio inspects the draft again once
your edits pause, so a problem you fixed, such as a model file the driver lacks,
leaves the list and the line, and one that remains stays. Generate then asks
the companion for the job's immutable submission record. The companion checks
the draft again, applies styles and LoRAs, sets the seeds by the seed policy
and freezes the prompt. When nothing needs a person, Studio submits that record at once, and the
line under the button says the job was submitted. That line clears itself after
five seconds. The viewer follows the job, and Queue lists it with its identifier
and its details. Generate again with unchanged settings starts another job the
same way, with no dialog.

Review settings, beside Generate, makes the same record and shows it in a
dialog before anything starts. Generate opens that dialog by itself only when
something needs a person:

- **A warning.** When the companion reports a warning for the job, such as a
  resource warning, the dialog lists it under Needs your decision with its own
  box. Generate in the dialog stays off until you tick each one, and Adjust the
  draft closes the dialog instead. Today the companion reports no warning when
  it checks a draft, so this case does not arise yet.
- **A refused draft.** When the companion refuses the draft, the dialog says no
  job started and shows the refusal with its error code. Its button opens the
  Create tab that holds the cause: LoRAs for `lora_binding`, Inputs for a
  style, seed or lineage refusal, All inputs for a refused workflow input and
  Layout for any other workflow refusal.
- **An error in the record.** The status line reads Blocked with the error, and
  Generate stays off.

A consent point never opens the dialog. The driver decides consent when it
loads a model for the job, so a job that needs one fails with its typed refusal
and the consent card, as
[Read a failure and grant a consent](#read-a-failure-and-grant-a-consent)
describes, and granting it stays a reviewed operation.

The dialog starts with a status line: ready to generate, a warning that needs
your decision, or what blocks the job. A summary follows with the workflow and
operation, the model files, the seeds, the prompt text the job sends, the
requested topology, one line on checks and the time the review expires. The
checks line says that Studio checked the values against the driver's inputs and
that the backend checks memory, topology, identity and consent when the job
runs. The prompt text lists each multi-line text input and any input a style
rewrote. When a style changed the text, the summary also shows your draft text.
A workflow with no multi-line text input lists every text input. No checkbox
appears unless a warning asks for one.

The full record the job keeps sits closed under Immutable submission record
(JSON). Studio renders it only when you open it, and it scrolls inside a short
box. Once the job is accepted, open its row in Queue: the same record sits
there, also closed, beside the full provenance.

A review expires 10 minutes after the companion made it. The expiry does not
bind the review to the driver's inputs: submit checks the node definitions and
each chosen file against the live driver whatever the review's age. The expiry
bounds how old the driver runtime that the record freezes can be, and it lets
the companion delete reviews that were never submitted. Generate submits within
the same action, so only a review left open in the dialog can expire. The
dialog says when it expires and why. Once it has, Generate in the dialog stays
off: close it and press Generate for a fresh record. The dialog keeps the
companion's clock, starting from the time the companion made the review, and
reads it every minute and at the expiry moment. The companion's own check at
submit decides, so a submit in the last second or two can still be refused as
expired. The dialog closes after every submit attempt, so a refusal or an
unknown outcome shows its banner and the Reconcile request button.

New drafts use the Random seed policy. For each job the companion picks a new
value for each integer `seed` and `noise_seed` input and freezes it in the
record. Your seed fields keep their values, but under Random each one is off and
reads Random each run, with a line that says to choose Fixed to set it. A seed
field with a value Studio cannot submit stays open until you repair it. Fixed
makes the seed fields editable and sends them as they are. A saved draft or record
keeps its own policy, and one that names no policy restores as Fixed, the
companion's default. Review settings shows the seeds the job will use. After the job
runs, the viewer footer shows the seed of the selected output, Queue rows show
the seeds of each job, and Copy seed in the gallery copies the seeds alone: one
seed as a plain number, several as JSON keyed by node and input. Copy named
inputs still copies every input of the recorded prompt. Job summaries in the
snapshot and in history carry the recorded seeds, so these views read them
without the full record. A job imported from another interface has no frozen
seed list, so its summary lists the integer `seed` and `noise_seed` inputs of
the prompt the driver ran. An output whose job the snapshot leaves out reads
that job's summary once, with the `job_id` filter of the job history. ComfyUI
seeds reach 18446744073709551615, but a browser holds whole numbers exactly only
up to 9007199254740991, so a summary sends a larger seed as a string of its
digits. Queue, the viewer, the review and Copy seed show every digit, and the
companion keeps the frozen number exact. Use this seed, in the viewer footer,
puts the output's seed into the matching seed field of the current draft and
sets the policy to Fixed. Undo reverts both. It matches a seed field by node and
input, and a single seed also fits a draft with a single seed field. When no
field matches, it changes nothing and says why; Reuse settings loads the
recorded workflow instead. It also refuses a seed above 9007199254740991, which
a draft field cannot hold exactly, rather than round it. Copy named inputs, Copy
API prompt and Export provenance read the job's full record with every digit of
each whole number, so a seed above 9007199254740991 comes out exact in the
prompt, the frozen seed list and the workflow's widget values.

Submitted jobs keep requested and effective settings, the workflow and its
digest, adapter version and lineage. Missing fingerprints and execution facts
are labelled unavailable. Reproduction reconstructs configuration; it does not
promise deterministic pixels across hardware and software versions.
Reuse loads the recorded executable prompt with its resolved seeds as a fixed
draft. Already expanded styles and inserted LoRAs remain in that prompt and are
not applied twice; original workflow metadata is retained separately. A
recorded prompt or workflow that holds a whole number above 9007199254740991
does not load, since a draft field cannot hold it exactly: Reuse settings,
Upscale and Load workflow name each such input with its digits and leave the
draft unchanged.

Send to edit, in the viewer footer and on a gallery card, puts an image output
into the image input of a workflow that edits images. When the workflow in
Create supports Edit and has an image input that Studio can set, Send to edit
keeps it. Otherwise it looks in the Workflow list for an editing workflow: one
that loads an image from the driver's input store, makes no video, has no
upscale or Fleet node and uses only node classes the driver lists. When it
finds one, it loads it. When it finds several, a dialog lists them, and the
draft stays as it is until you choose one. When it finds none, the dialog says
why and names where editing workflows come from: the Workflow list, Import JSON
and workflows saved in Library. The draft and the selected output stay as they
are. The companion copies the output into the driver's input store as an
upload of its own, and Studio binds the copy to the image input that the
workflow's App Mode layout lists first, never to a mask input. It sets the
Operation to Edit, opens the tab that lists the prompt and the image, and
records the output and its copy as lineage, after any lineage a kept workflow
already has. One Undo restores the earlier draft.
Send to edit takes no video output or uploaded mask. Under Edit, Generate and
Review settings stay off until the image input holds an image the driver
offers, and the line under Generate says why.

A review binds to the definitions of the node classes its prompt uses, without
their file lists, so adding a model, LoRA or input file does not void it.
Submission still checks each selected file or option against the live driver and
refuses a missing one with `choice_unavailable`. A confirmed retry does not need
a fresh review: Studio reads the driver schema again, checks the frozen
submission against it the same way and keeps the frozen request unchanged. It
records the driver version and runtime the retry ran against in `retry_runtime`,
which a grid's full record also shows per cell, and comparisons of a retry read
those recorded versions.

To run a grid, load a workflow in Create and set its inputs there: each cell is
that Create draft with the axis values filled in. On Grids, pick an input for
Axis A and list its values as a JSON array, for example `[20, 30, 40]` for steps
20, 30 and 40. An optional Axis B multiplies the cells, so three values on Axis
A and two on Axis B make six cells. Press Validate every cell, tick the review
box and press Submit reviewed grid. When a cell finishes, Inspect full
submission lists its outputs; View shows one, and Compare sets it as the
comparison for the output in view. The value fields start empty. Validate every
cell stays disabled until the draft has a workflow, Axis A names an input and
each chosen axis holds a JSON array with at least one value; the line beside the
button names what is missing.

Grids opens on what a grid does, a worked example and a first comparison to
try. In the example, steps [20, 30, 40] on Axis A and cfg [3.5, 5] on Axis B
make six cells with the same prompt and seed. Axis B changes first: cell 1 is
20 steps at CFG 3.5, cell 2 is 20 steps at CFG 5 and cell 6 is 40 steps at CFG
5. The first comparison puts steps [20, 40] on Axis A alone with the Fixed
draft seeds policy, so its two cells differ only in steps. View cell 1's output
and Compare cell 2's, and the viewer shows them side by side above their
recorded parameter differences.

Parameter grids use typed workflow inputs or named style stacks as axes. A
style-axis value is an ordered list of saved style IDs; an empty list selects
no styles. Each cell freezes the selected definitions and revisions before
expanding its prompt. Later library edits do not change that submission or its
retry. Fixed seed mode preserves draft seeds; incrementing mode requires an
explicit seed binding and base value. Invalid cells remain visible and block
dispatch until corrected or explicitly excluded. Each accepted cell has its
own immutable submission and outcome, with selective failed/cancelled retries
through the existing ComfyUI queue.
Changing a grid definition invalidates its reviewed preview and confirmation;
validate the edited cells again before submission. Grids keeps an identity of
the draft and grid fields each preview came from. When Grids opens again, or the
page reloads, it restores a saved preview only if the draft and fields still
match; otherwise it drops the preview and asks for validation again. A dispatch
whose outcome is unknown keeps its preview and request ID so it can be
reconciled, and Grids says when that review came from an earlier draft.

## Workflow sources and updates

The Workflow list is the bundled set, read from one file: the workflow catalog
in the installed dgx-monarch package. `tools/gen_studio_inventory.py` builds it
from the checkout's `example_workflows` folder, which ComfyUI lists under
Browse Templates. Studio scans no folder at run time, neither that one nor
ComfyUI's saved workflows.

Any other workflow comes in through Import JSON on Create: a UI workflow that
ComfyUI saves or exports, or an API prompt from its Export (API). The driver's
`/object_info` must list every node class the graph uses, and the graph needs a
save or preview node. Studio edits INT, FLOAT, STRING, BOOLEAN and COMBO
widgets, dynamic choices included. A wired input stays a connection, and an
unknown custom widget refuses with a repair reason. A graph needs no DGX
Monarch node to run, but only DGX Monarch nodes render across the Sparks
([QUICKSTART](QUICKSTART.md#2-only-some-nodes-move-to-the-cluster)). App Mode
is optional. `extra.linearData.inputs` lists the prominent inputs in order,
each as a `[node id, widget name]` pair or a `graphId:nodeId:name` identity,
with an optional configuration object. `extra.linearData.outputs` lists save
node ids. Each input must name an editable widget that no wire feeds;
otherwise Generate's check refuses the draft until you repair it on Layout. A
graph without App Mode opens on All inputs.

An update replaces the bundled templates and the catalog with the new
release's copies. Studio never writes to either, and your edits live in copies
that an update leaves alone. The draft stays in this browser. Presets, sessions
and workflows saved in Library, and each job's submitted workflow, stay in the
companion's [state directory](#state-reconciliation-and-limits). Files you
export stay where you saved them. A copy keeps the version it was made from;
load the template again to get the new one. ComfyUI saves a template you
changed as a new file in its own user workflows folder.

Do not edit or add files in the checkout's `example_workflows`.
`dgxm update --verify` refuses a checkout with local changes (`repo_dirty`).
A plain `dgxm update` stops at its merge when the new release changes the same
file, and otherwise keeps your edit, which the Workflow list still does not
show. To keep a workflow with its Layout, save it in Library under workflows.
To keep its input values too, save a preset. Load into draft brings either
back.

## Run many prompts with Fleet

Fleet, a tab under Utilities, runs a workflow whose `DGXMonarchFleetKSampler`
node turns each line of its prompts input into its own job. The jobs go to the
GPUs in turn, one per GPU at a time, and share the model, size, steps, CFG,
sampler and scheduler. Line 1 uses seed `noise_seed`, line 2 `noise_seed + 1`
and so on, and the outputs come back in prompt order. Use it for more pictures
per hour, such as many prompts on one model. It does not make one picture
faster, and on one GPU it gains nothing; a normal Monarch workflow can split
one render across the GPUs instead.

The page says this and gives the first step. Set up Fleet in Create loads the
packaged workflow "Fleet: many prompts, one per GPU" into Create, unless the
draft already holds a Fleet sampler, sets the draft's Operation to Fleet and
opens Create. A notice there names the next step: put your prompts in the
prompts input, one per line, then generate as usual. When the load replaces a
workflow, Undo in Create brings that workflow back, and when the load fails
the page stays on Fleet with the error and leaves the draft as it was. If you
leave Fleet while the workflow loads, the draft still takes it, the notice
still appears, and Studio stays on the page you chose.
Operation only labels the job, for the
Gallery filter and the Fleet page's list of graph tasks: a workflow without a
Fleet sampler runs as one ordinary job whatever its Operation says. When the
packaged workflow is missing from the catalog, the button sets the label, opens
Create and says to load a workflow with a Fleet sampler; a loaded workflow
without one shows Fleet as not supported.

A run takes up to 1,024 jobs, and the negative prompt is one value for every
job or exactly one per job. The run is one ComfyUI task: no job in it can be
paused, reordered or retried alone, so Queue retries the whole run. The page
also shows the reporting hosts as the telemetry panel does, the metric history,
the runtime evidence of each recorded Fleet graph task, and the driver's last
Fleet queue snapshot and Worker sample, closed until you open them.

## Read a failure and grant a consent

Studio reads a failed job's recorded error the way dgx-monarch writes it and
shows only what the record holds. When the message carries a refusal class tag,
the viewer status line and the Queue row start with the class, such as
"Refused, class C (capacity-protective)", then the first sentence after the tag.
The classes are those of
[DESIGN.md section 5.9](DESIGN.md#59-refusal-taxonomy-what-a-typed-refusal-is-allowed-to-do):
P physics, C capacity-protective, U unproven correctness and K known-wrong math.
Any other message shows its error code and its first sentence. When the message
names the driver's panel and headless remedy, the status line shows it. Expand
the Queue row for the full text, the guard, the error code and a link to the
docs/TROUBLESHOOTING.md entry the message cites. The link opens that page, so
find the entry by its number. When the companion shortens a long message it
keeps the class tag and the text after it, the remedy and the consent block,
even for a worker refusal that arrives after Monarch's remote traceback. Full
provenance in the expanded row still holds the raw record. A Queue row also
shows a 64 pixel thumbnail of the job's newest final output when the snapshot
holds one, and nothing when it does not. The preview downloads once the row
scrolls within 200 pixels of the visible area.

A refusal that a consent can clear carries a consent request. The status line
and the Queue row then show its reason and the label of the ComfyUI panel button
that grants it, with a Grant and retry button. That button opens Operations with
consent apply chosen and its key and id empty, beside the request's model, kind
and consent id. Review it: with one pending request the review selects it, and
with several it lists each one with its consent id. Studio says whether the
chosen request carries the job's consent id, and offers to fill the key and id of
the listed request that does. Studio never grants a consent by itself. You
review the exact plan and confirm it, as for any operation. After the grant
succeeds, retry the job from Queue. The driver keeps a pending request for 30
minutes after the refusal that raised it, and forgets every one when it
restarts, so an older failure can find none pending. Retry the job to ask
again. A companion started without `--allow-management` lists consent apply as
unavailable, and the button reads Review consent. Grant the consent in the
ComfyUI panel instead, or start Studio again with `--allow-management`, which
for the managed service is `dgxm studio up --allow-management`. The management
paragraphs of [Install and start](#install-and-start) say what that flag
authorizes.

## Runtime evidence

Expand a queue entry's Monarch evidence view to inspect per-render outcomes
or exact runtime events. These records use the Comfy prompt ID, executing node
ID and list index, plus a Monarch render ID and optional Fleet item index.
Capture requires ComfyUI's `get_executing_context()` boundary. If that API or an
executing context is absent, Studio leaves the facts unavailable. It never
assigns the global last-render or last-Fleet snapshot to a particular job.

A dispatch observation establishes that the driver's guards and dispatch
returned; it does not establish Worker acceptance or successful output.
Validated Worker results supply their actual setup generation, topology,
software versions and available process-lifetime source manifest. Dispatch
attention is labelled as the driver's selection; Worker results report both
selected and effective attention where available. Model/LoRA identities reuse
the existing parity-validated start/middle/end fingerprints, not full-file
content hashes. Latent statistics retain the runtime's full-byte digest.

Output validation, resource retirement and graph completion are separate facts.
A render or Fleet item is complete only after its output is validated and its
existing resource retirement succeeds. Native owner ACKs are recorded only
after the existing validator checks the response count, generation/token and
exactly one owner; they are not inferred from a returned tensor or graph
success. The journal stores a SHA-256 identity for an owner token, never the
raw capability. The separate opt-in resident-adoption evidence is not copied
into this journal. Unknown outcomes remain unknown. Late verified facts may
enrich an ended job without changing its terminal graph state or submitted
settings.

The companion polls the Monarch-owned `/dgxm/execution-events` route and grants
a 30-second observer lease. Without a lease, hooks skip context lookup,
fingerprinting and event serialization; no driver journal is written to disk.
Requests already marked for observation can still finish collecting their
Worker identity after the companion stops. Ordinary callback and context
handling remain, so this is not a claim of zero runtime overhead.

Driver replay retains at most 8,192 events and 8 MiB of encoded event payload.
Event data is limited to 32 KiB plus sequencing metadata; a page budgets 2 MiB
for event data and at most 128 rows, with additional sequencing and response
metadata. These are payload
bounds, not measurements of Python process memory. Producer restarts, expired
leases, eviction and omitted events create explicit coverage gaps. Capture is
not retroactive, and reconnecting cannot reconstruct a missing native ACK.

The owner-private companion database retains received events and complete
per-render records. Deduplication and producer watermarks survive replay-window
and display-summary eviction. Each job displays at most 32 render summaries,
limited to 4 KiB each, with complete outcome counts and a count of omitted
summaries. Full outcomes and events remain available through the evidence view
and authenticated `/api/v1/jobs/{id}/runtime` pages. Durable history is not
silently pruned; its long-session disk growth remains a measurement gate.
Runtime records can contain Worker hostnames and model filenames. The driver's
existing trusted-network/authentication boundary and the companion's owner
boundary protect those details.

## State, reconciliation and limits

Owner state defaults to `$XDG_STATE_HOME/dgx-monarch/studio`, or
`~/.local/state/dgx-monarch/studio`. An empty `XDG_STATE_HOME` counts as unset.
The companion refuses a relative `XDG_STATE_HOME` or `--state-dir`, so launching
from another directory never opens a second database. Use `--state-dir` to
choose another private directory. Directories must be owner-only (0700); the
SQLite database uses 0600, WAL and full synchronous transactions. A process lock
permits one companion per state directory. One connection serves the event loop
and the worker threads that settle driver history and write live and operation
previews. Every use of it, closing included, holds one lock, and shutdown waits
for those threads before the database closes. A schema mismatch refuses startup;
this first schema has no older migration. A state directory, lock or database
refusal exits with one line. Preserve the directory to preserve shared records.

Projects, presets, styles, sessions, imported workflows and experiments use
optimistic revisions. A stale save returns a conflict, retaining the newer
record. Browser-local drafts and layout survive connection loss. If a saved
draft cannot be restored, Studio pauses automatic saves and offers the stored
text for export before anything replaces it. If the browser refuses a write,
Studio says so and offers an export of the current draft. A second tab raises a
conflict only when it saves a different draft. Exported records retain unknown
metadata. Durable jobs and operation receipts are not silently deleted. Expired
previews can be recreated; review a new preview after capabilities change. The
companion deletes expired job and grid previews at startup and on later preview
writes, at most once every five minutes, and keeps the preview each dispatched
grid was reviewed from. A submit of a deleted preview is refused as expired.
Each preview and each job stores its workflow once, and reads return the same
requested, effective and workflow fields. A stored workflow stays while any
preview or job names it.

Renaming a saved preset, session or workflow, or changing its membership,
preserves its saved content. Save a new record to capture the current workspace.

Saved records support explicit project memberships and project filtering;
artifact memberships have their own revision checks. A record can be shared
across projects without moving its source assets. Portable library export
includes records, extension metadata and source references, for the entire
library or one project. Import first creates a review plan. Preserve-ID mode
requires an explicit keep/replace decision for conflicts; copy mode creates
new record IDs and remaps project/style references. Applying the reviewed
digest checks revisions again and commits atomically with an idempotency key.
Missing media or job references remain on a repair list. Import does not copy
media, dispatch jobs or recreate missing source provenance. Library documents
are limited to 4 MiB. Exports leave out each artifact's search text, which the companion
derives from the job that made it. One export holds at most 2,000 rows per
collection, the most one import accepts, and refuses beyond that or beyond
4 MiB, naming the collection to split. Import keeps a reference only when it
needs repair, once, with its reason.

Save a durable grid as a named experiment to retain its definition, observed
cell outcomes and source references in the project library. Loading the
definition replaces the editable draft and axes and requires a fresh preview;
resuming a local run reads its existing jobs. An imported experiment can retain
its frozen definition while its original run is unavailable. Saving or importing
an experiment never submits those historical jobs again.

A request has a caller idempotency key and a server-issued identity. Repeat
requests must have identical bodies. After a lost response, reconcile by the
key; never invent a new key and resubmit an uncertain job. The companion
reconciles its submissions and other interfaces' jobs from driver queue and
history snapshots. External jobs contain only provenance the driver supplied.
The companion issues each prompt identity (a UUID) before it submits, so a lost
or malformed acknowledgement reconciles through that prompt's exact history.

A job the driver no longer holds is settled once, from exact evidence: it is
missing from a successful queue snapshot and its own history lookup answers with
no entry. A queued job whose cancellation the driver acknowledged then becomes
cancelled; any other such job becomes unknown with the reason
`absent_from_driver` and is no longer looked up. The recent-history window is
never used as proof. The adapter halves that window when a response exceeds the
4 MiB budget and doubles it again after three pages in a row under half that
budget; a doubled window that overflows waits twice as long before the next try.
It also queries missing known prompts individually. A history failure shows a
warning while successful queue observations remain available. When the queue
response passes its budget, the driver stays connected, the queue shows
`over_budget`, and history reconciliation continues without settling absent
jobs. A driver row that cannot be read is skipped and counted in the driver
state, and a non-finite number in a driver response reads as null.

A 4xx refusal of a prompt marks its job failed with the failing nodes (node,
class, input and type only); a 5xx, a lost response or an unreadable
acknowledgement stays unknown. A prompt the driver queued with node errors stays
queued with a warning. The driver state carries `code` and `since` while it is
unavailable, `queue.state`, the current `history.limit`, a count of skipped
driver rows, and the event watcher's state while it waits to reconnect.
Background failures are logged and retried with a growing pause of up to 30
seconds.

Events have durable cursors and server timestamps. A replay gap requires a new
snapshot. The snapshot is authoritative. The event feed is a notification log:
it names created jobs, operations and artifacts and the fields an update
changed. A field too large for an event is listed under `omitted_fields`; read
it from the snapshot or the record. Terminal job state never regresses; a late
submission response cannot replace a running state with queued. A job reads
succeeded only once the companion has stored the outputs that the driver's
history lists for it, so a snapshot never shows a succeeded job without them.
Restarted in-flight operations become unknown and are never automatically re-executed.
Queue deletion records a cancellation request. It settles as cancelled only when
the driver acknowledged it while the job was queued and the prompt has left both
the queue and its history. Running-job cancellation uses the driver's atomic
prompt-specific cancellation route when available. On a driver without that
route, a queued job falls back to queue deletion under the same cancellation
intent; a running job is refused and its intent and key stay unspent. The
adapter never falls back to a global interrupt. Each job retains one durable
cancellation intent, including after a lost acknowledgement or companion
restart. Repeating that intent cannot dispatch it again. A reviewed management
operation is single-use; its original request key reconciles the receipt. Fresh
execution requires a fresh review.

The capabilities response publishes the companion's request, event, console,
chart, thumbnail, page, preview, grid and `/object_info` limits in its `limits`
object. A JSON request and a driver response are limited to 4 MiB, except the
driver's `/object_info` schema, which has its own 16 MiB budget
(`object_info_bytes`); a refusal names the endpoint and its budget. The
companion's browser event feed replays at most 4,096 events and 8 MiB of event
payload (`event_count` and `event_bytes`), and each event is limited to 64 KiB
(`event_max_bytes`). Other limits are 400 browser
console lines/128 KiB, 240 chart samples and one hour, a 32 MiB thumbnail cache
budget, a 64 MiB media display limit, and 1,024 grid combinations. Every
companion error, including the 4 MiB request limit and an unexpected failure,
returns JSON `{"error": {"code", "message"}}`. Workflow and draft import read
files up to twice `request_bytes` and refuse an encoded request above
`request_bytes`.

Snapshots return compact windows of up to 200 rows per kind under a shared byte
budget, prioritizing unresolved jobs and reporting omitted counts. Job history
returns summaries; full job provenance and operation receipts load by identity
when opened or exported. Artifact lists omit only the derived search index.
Record and history pages are limited to 8 MiB; cursors advance by actual
returned rows. Artifact pages list outputs with equal creation times most
recently stored first, as snapshots do. Record storage permits 2,000 editable
records per kind. The browser bounds loaded record windows and catalogs to
8 MiB and 2,000 rows, and the companion refuses a save or a library import
that would take a kind's catalog past that 8 MiB window before it commits. A
new record keeps one identity per form until its save is confirmed, so a
retried save that already committed reads as saved.

Studio decodes at most 16 megapixels (4096 x 4096 pixels) per image, for input
uploads and thumbnails alike, and checks that from the file header before it
decodes. Input uploads accept PNG, JPEG or WebP files of at most 2 MiB, keep
mask alpha, and store an orientation-normalized RGBA PNG; a larger image is
refused with a request to downscale it. Send to edit stores its copy of an
output the same way, read from the driver's file, so the 2 MiB limit on what a
browser sends does not apply to it. Thumbnails return at most 320 pixels per
edge and run at most two decodes at once. A 65 MiB ceiling on the normalized
upload and on a thumbnail's source file is only a backstop that no 8-bit image
within the pixel limit reaches. Live previews retain at most 16 identity-bound
frames. Expanded style text and complete submission previews each have a 4 MiB
JSON budget.

Grid expansion is bounded before copying parameters. All cell previews and the
grid review commit together under a combined 32 MiB budget; failure leaves no
partial previews. Previews and jobs store each workflow once, so the budget
counts each distinct workflow once and each cell for its compiled prompt and
settings. The same budget bounds what submitting the grid writes: a job for each
cell and the grid run. Grids whose compiled prompt passes about 30 KB reach that
byte budget before 1,024 cells. Excluded cells are not expanded, so they have no
preview to inspect. Grid history and polling use compact views
under an 8 MiB budget, with full records fetched only for explicit export.
Random seeds honor the installed input bounds and JavaScript's exact integer
range so the generated seed can be reused without rounding.
Definitive pre-acceptance refusals release pending review controls. Incomplete
or malformed acknowledgements, authentication failures and transport timeouts
retain the exact request identity for reconciliation. Corrupt optional draft or
layout storage cannot clear that protection.
Unreadable saved operation, grid or import identities block new requests until
the user checks durable records and explicitly clears the unreadable entry.
Storage access failures leave authentication available in memory, but requests
requiring a saved recovery identity remain blocked. Malformed imported draft
controls are rejected before replacing the working draft.
A 401 while polling returns to the token form, because a restarted companion
issues a new token. A sign-in link from before a restart stops working the
same way; `dgxm studio open` prints a new one. A workflow inspection or job review is discarded, with a
notice, when the workflow, its values or the connection change while it runs;
edits to other draft controls do not discard an inspection. The inspection Create
runs again after a value edit is dropped without a notice in the same cases, and
the next edit runs it again.

Snapshot refreshes preserve the event replay cursor so bursts spanning several
pages reach the console. Evicted replay gaps are labelled explicitly. Media
decoding failures retain a download action, and frame stepping waits for valid
duration and frame-rate metadata.

## Development

```sh
cd studio
npm ci
npm run check
npm test
npm run build
cd ..
python tools/gen_studio_inventory.py
python tools/gen_studio_inventory.py --check
python -m pytest -q tests/test_studio_inventory.py tests/test_studio_workflows.py \
  tests/test_studio_preview.py tests/test_studio_backend.py
python -m build
```

The checked-in Vite output is under `src/dgx_monarch/studio/static`, outside
ComfyUI's `WEB_DIRECTORY`. Both package formats include the compiled assets,
inventory and workflow catalog. The source archive also includes the complete
frontend source and lockfile. Builds do not download assets at runtime or use a
CDN. Release acceptance requires the full specification, not just these checks.

After each bundle build, regenerate `src/dgx_monarch/studio/resource_hashes.py`
and `web/js/dgx_monarch_presentation.js` with the inventory tool. Its literal SHA-256 table is covered by the Python source
manifest. Setup and update validation check the listed resource bytes;
source-sync cleanup permits those exact files only after verifying the reviewed
source digest. Extra files, changed assets and symlinks remain refusal cases.

## Troubleshooting

If the page is unavailable, start or recover the companion itself. A browser
cannot restart its entirely unavailable host. If only the driver is unavailable,
the page and durable records remain accessible. A stale telemetry timestamp is
not a live observation. If a schema changes, refresh capabilities and re-preview
the job; if a record revision conflicts, reload and explicitly reapply changes.
For startup, Host, Origin and token refusals, see
[TROUBLESHOOTING #123](TROUBLESHOOTING.md#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token).
For the Driver not running panel, see
[#131](TROUBLESHOOTING.md#131-studio-shows-driver-not-running); for
`dgxm studio up` and the always-on Studio, see
[#132](TROUBLESHOOTING.md#132-dgxm-studio-up-refuses-or-the-always-on-studio-does-not-start);
and for a sign-in link or `dgxm studio tunnel`, see
[#133](TROUBLESHOOTING.md#133-the-sign-in-link-does-not-open-studio).

Do not delete state, resubmit uncertain jobs, bypass guards or restart Workers
as a connection-repair shortcut. An operation receipt reports only the boundary
that actually completed. A CLI exit code or a mocked test is not render fidelity
or hardware acceptance.
