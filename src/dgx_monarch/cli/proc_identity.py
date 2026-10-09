"""Read worker actor identities and resource use from procfs without side effects.

Each function reads one process fact through an injectable proc root. The
``(pid, starttime)`` identity detects PID reuse within a boot;
``cli/actor_reaper.py`` applies selection policy.

These helpers never signal, connect, or launch commands. Doctor's orphan row
also uses real procfs during tests, so reads must stay cheap and passive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PROC_ROOT = "/proc"

# Keep the actor marker out of command-line options and process argv. Lifecycle
# scripts pass match patterns on stdin to prevent the sweep from matching its
# own command line; candidate matching happens against procfs here.
ACTOR_MODULE = "monarch._src.actor.bootstrap_main"
LOOP_MODULE = "dgx_monarch.cli.worker_loop"
BOOTSTRAP_ENV_MARKER = "HYPERACTOR_MESH_BOOTSTRAP_MODE="
# Both the worker loop boundary and the driver set this project-specific
# marker. A separate Monarch application may have the bootstrap marker but is
# not a sweep target without this one.
OWNER_ENV_MARKER = "DGXM_PYTHONPATH="
# The driver sets its ``<pid>:<starttime>`` before spawning children. Read the
# marker from a candidate child: ``/proc/<pid>/environ`` reflects exec-time
# state, so a process cannot observe its later ``os.environ`` updates there.
DRIVER_ENV_MARKER = "DGXM_DRIVER="

_ESTABLISHED = "01"
# telemetry.py pool_gib rule: a mapping counts only from 4 GiB up.
FAT_ANON_KB = 4 * 1024 * 1024
_CLK_TCK = float(os.sysconf("SC_CLK_TCK") or 100)
_PAGE_SIZE = float(os.sysconf("SC_PAGE_SIZE") or 4096)


@dataclass(frozen=True)
class ProcInfo:
    """One process as the sweep sees it."""

    pid: int
    ppid: int
    starttime: int
    uid: int | None
    argv: tuple[str, ...]
    actor_argv: bool
    bootstrap_env: bool
    dgxm_owned: bool
    owner: tuple[int, int] | None = None

    @property
    def marked(self) -> bool:
        return self.actor_argv or self.bootstrap_env

    @property
    def argv_tail(self) -> str:
        return " ".join(self.argv[-3:])


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def iter_pids(proc_root: str = PROC_ROOT) -> list[int]:
    """Every numeric entry under the proc root, ascending."""
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    return sorted(int(name) for name in names if name.isdigit())


def read_stat(pid: int, proc_root: str = PROC_ROOT) -> tuple[int, int] | None:
    """``(ppid, starttime)`` from ``/proc/<pid>/stat``.

    ``comm`` is unquoted and may contain spaces and parentheses, so the fields
    are taken from after the final ``)``: ppid is then field 4 (index 1) and
    starttime field 22 (index 19).
    """
    text = _read_text(f"{proc_root}/{pid}/stat")
    if not text:
        return None
    _, _, tail = text.rpartition(")")
    fields = tail.split()
    if len(fields) < 20:
        return None
    try:
        return int(fields[1]), int(fields[19])
    except ValueError:
        return None


def read_session(pid: int, proc_root: str = PROC_ROOT) -> int | None:
    """Session id from ``/proc/<pid>/stat`` field 6, or None.

    A session is inherited at fork and changes only through ``setsid``. So if
    neither a process nor its parent called it after the fork, a session
    unlike the current parent's means a subreaper adopted the process once its
    spawner exited. This tells an actor adopted by systemd from one whose
    spawner remains its parent.
    """
    text = _read_text(f"{proc_root}/{pid}/stat")
    if not text:
        return None
    _, _, tail = text.rpartition(")")
    fields = tail.split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def read_argv(pid: int, proc_root: str = PROC_ROOT) -> tuple[str, ...]:
    raw = _read_bytes(f"{proc_root}/{pid}/cmdline")
    if not raw:
        return ()
    return tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def read_uid(pid: int, proc_root: str = PROC_ROOT) -> int | None:
    """Real uid from ``/proc/<pid>/status``."""
    text = _read_text(f"{proc_root}/{pid}/status")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    return None


def read_env_markers(
    pid: int, proc_root: str = PROC_ROOT
) -> tuple[bool, bool, tuple[int, int] | None]:
    """``(bootstrap_env, dgxm_owned, owner)`` from ``/proc/<pid>/environ``."""
    raw = _read_bytes(f"{proc_root}/{pid}/environ")
    if not raw:
        return False, False, None
    entries = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    bootstrap = any(entry.startswith(BOOTSTRAP_ENV_MARKER) for entry in entries)
    owned = any(entry.startswith(OWNER_ENV_MARKER) for entry in entries)
    owner = None
    for entry in entries:
        if entry.startswith(DRIVER_ENV_MARKER):
            owner = parse_owner(entry[len(DRIVER_ENV_MARKER):])
    return bootstrap, owned, owner


def driver_identity(pid: int | None = None, proc_root: str = PROC_ROOT) -> str:
    """Return ``<pid>:<starttime>``, or "" if identity cannot be read.

    Never substitute start time 0: no live owner would match it, making its actors
    eligible for reaping. An empty answer leaves the ownership marker unset and
    preserves the conservative ``held`` verdict.
    """
    target = os.getpid() if pid is None else int(pid)
    stat = read_stat(target, proc_root)
    return "" if stat is None else f"{target}:{stat[1]}"


def parse_owner(value: str) -> tuple[int, int] | None:
    """The ``(pid, starttime)`` a DGXM_DRIVER value names, or None."""
    pid, _, starttime = value.strip().partition(":")
    try:
        return int(pid), int(starttime)
    except ValueError:
        return None


def has_module_argv(argv: tuple[str, ...], module: str) -> bool:
    """True when argv runs ``-m <module>`` as two adjacent tokens.

    List comparison prevents a process that only mentions the module name from
    becoming a candidate. ``argv[0]`` and ``comm`` are ignored because a
    launcher may rewrite ``argv[0]``.
    """
    return any(
        argv[index] == "-m" and argv[index + 1] == module
        for index in range(1, max(len(argv) - 1, 1))
    )


def loop_argv_address(argv: tuple[str, ...]) -> str | None:
    """The ``--address`` a worker loop argv carries; None if not a loop, "" if it has none.

    Mirrors the anchored loop tail in ``worker_health.loop_regex`` as a list
    comparison, so it cannot match a process that only mentions the module.
    """
    if not has_module_argv(argv, LOOP_MODULE):
        return None
    for index, word in enumerate(argv[:-1]):
        if word == "--address":
            return argv[index + 1]
    return ""


def proc_info(pid: int, proc_root: str = PROC_ROOT) -> ProcInfo | None:
    stat = read_stat(pid, proc_root)
    if stat is None:
        return None
    ppid, starttime = stat
    argv = read_argv(pid, proc_root)
    actor_argv = has_module_argv(argv, ACTOR_MODULE)
    bootstrap_env, dgxm_owned, owner = read_env_markers(pid, proc_root)
    return ProcInfo(
        pid=pid,
        ppid=ppid,
        starttime=starttime,
        uid=read_uid(pid, proc_root),
        argv=argv,
        actor_argv=actor_argv,
        bootstrap_env=bootstrap_env,
        dgxm_owned=dgxm_owned,
        owner=owner,
    )


def alive(pid: int, starttime: int, proc_root: str = PROC_ROOT) -> bool:
    """Identity-checked liveness: a recycled pid reads as dead."""
    stat = read_stat(pid, proc_root)
    return stat is not None and stat[1] == starttime


def read_state(pid: int, proc_root: str = PROC_ROOT) -> str | None:
    """The state letter from ``/proc/<pid>/stat``, or None when it will not read.

    Index 0 of the same tail ``read_stat`` already parses. ``Z`` is a process
    that exited and waits for its parent to reap it: it holds no memory and
    answers nothing. ``T`` is a stopped process that still holds every page it
    had, so it is alive for every question this module answers.
    """
    text = _read_text(f"{proc_root}/{pid}/stat")
    if not text:
        return None
    _, _, tail = text.rpartition(")")
    fields = tail.split()
    return fields[0] if fields else None


def alive_and_not_zombie(pid: int, starttime: int, proc_root: str = PROC_ROOT) -> bool:
    """Identity-checked liveness that a zombie fails.

    ``alive`` keeps its own answer: the kill ladder still has a pid to confirm
    and a signal to send. A liveness question about a worker actor is a
    question about held memory, and a zombie holds none.
    """
    return alive(pid, starttime, proc_root) and read_state(pid, proc_root) != "Z"


def reads_empty(pid: int, proc_root: str = PROC_ROOT) -> bool:
    """True when cmdline and environ both read as empty or unreadable.

    Both, not either: ``ProcInfo.marked`` needs only one of the two reads to
    succeed, so a candidate drops out of a scan only when neither answers; an
    OR rule would count every own-uid process with an unreadable environ.
    """
    raw = _read_bytes(f"{proc_root}/{pid}/environ")
    entries = [part for part in (raw or b"").split(b"\0") if part]
    return not read_argv(pid, proc_root) and not entries


def environ_failed(pid: int, proc_root: str = PROC_ROOT) -> bool:
    """True when the environ read failed, as against coming back empty.

    A zombie's environ is gone and reads empty; a read the kernel refuses
    says nothing about the process at all. ``/proc/<pid>/environ`` needs
    ptrace read access, which an own-uid process can still deny (ssh-agent and
    the systemd user manager do). Yama's ``ptrace_scope`` limits only attach,
    so it does not gate this read.
    """
    return _read_bytes(f"{proc_root}/{pid}/environ") is None


def names_a_module(argv: tuple[str, ...]) -> bool:
    """Return True when argv runs ``-m`` followed by any module.

    Include the actor module: actors already counted by the caller must not also
    count as unidentified processes.
    """
    return any(
        argv[index] == "-m"
        for index in range(1, max(len(argv) - 1, 1))
    )


def statm_rss_gib(pid: int, proc_root: str = PROC_ROOT) -> float:
    """Read resident GiB from ``/proc/<pid>/statm`` field 2.

    Unlike ``smaps_rollup``, this read is not ptrace-gated and avoids a mapping walk.
    """
    text = _read_text(f"{proc_root}/{pid}/statm")
    fields = (text or "").split()
    if len(fields) < 2 or not fields[1].isdigit():
        return 0.0
    return round(int(fields[1]) * _PAGE_SIZE / 2**30, 2)


def unreadable_to_us(pid: int, starttime: int, empty: bool,
                     proc_root: str = PROC_ROOT) -> bool:
    """Return True for a live, memory-holding process whose identity is unreadable.

    Cover both empty cmdline/environ reads and a refused environ read when argv
    names no Python module. The latter requires a size floor so routine protected
    processes, such as ssh-agent, cannot permanently block replacement. Named
    Python modules are excluded because actors are already counted separately.

    Recheck liveness here: a process may exit or its PID may be reused during the
    scan.
    """
    if not empty:
        if not environ_failed(pid, proc_root):
            return False
        if names_a_module(read_argv(pid, proc_root)):
            return False
        if statm_rss_gib(pid, proc_root) < FAT_ANON_KB / 2**20:
            return False
    return alive(pid, starttime, proc_root)


def confirmed(pid: int, starttime: int, uid: int, proc_root: str = PROC_ROOT) -> bool:
    """Confirm the same marked process and uid immediately before signaling.

    Re-reading identity and markers prevents signaling a different process if
    the kernel reused the pid after the initial sweep.
    """
    info = proc_info(pid, proc_root)
    return (info is not None and info.starttime == starttime
            and info.uid == uid and info.marked)


def ancestry(pid: int, proc_root: str = PROC_ROOT, limit: int = 64) -> list[int]:
    """Parent pids from the immediate parent up to pid 1."""
    chain: list[int] = []
    seen = {pid}
    current = pid
    for _ in range(limit):
        stat = read_stat(current, proc_root)
        if stat is None:
            break
        parent = stat[0]
        if parent <= 0 or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return chain


def boot_id(proc_root: str = PROC_ROOT) -> str:
    return (_read_text(f"{proc_root}/sys/kernel/random/boot_id") or "").strip()


def uptime_s(proc_root: str = PROC_ROOT) -> float:
    text = _read_text(f"{proc_root}/uptime") or ""
    fields = text.split()
    if not fields:
        return 0.0
    try:
        return float(fields[0])
    except ValueError:
        return 0.0


def age_s(starttime: int, uptime: float) -> float:
    """Seconds since the process started, from its stat field 22."""
    return max(uptime - starttime / _CLK_TCK, 0.0)


def rollup_rss_gib(pid: int, proc_root: str = PROC_ROOT) -> float:
    """Resident total from ``smaps_rollup``: one read, kernel summed.

    ``Rss:``, not ``Anonymous:``: slab residency keeps weights in a MAP_SHARED
    memfd (actor/slab_arena.py), which reports ~0 anonymous while holding tens
    of GiB resident, and the per-mapping rule below counts those mappings.
    ``Rss`` is a true upper bound on any per-mapping Rss sum, so this
    prefilter can never hide a candidate.
    """
    text = _read_text(f"{proc_root}/{pid}/smaps_rollup")
    if not text:
        return 0.0
    for line in text.splitlines():
        if line.startswith("Rss:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return round(int(fields[1]) / 2**20, 2)
    return 0.0


def pool_gib(pid: int, proc_root: str = PROC_ROOT) -> float:
    """Sum of giant (>= 4 GiB) unbacked mappings held by one process.

    This process-scoped procfs measurement remains available after a process
    releases its CUDA context. ``/memfd:`` mappings also count: slab residency
    holds weights in ``MAP_SHARED`` memory that reports no anonymous pages,
    and nothing closes the slab of an actor with no live owner, so that memory
    returns only when the process exits.
    """
    text = _read_text(f"{proc_root}/{pid}/smaps")
    if not text:
        return 0.0
    total_kb = 0
    anon = False
    for line in text.splitlines():
        if not line:
            continue
        if line[0].isdigit() or line[0] in "abcdef":
            fields = line.split()
            path = fields[5] if len(fields) > 5 else ""
            anon = (not path and fields[4:5] == ["0"]) or path.startswith("/memfd:")
        elif line.startswith("Rss:") and anon:
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                kb = int(fields[1])
                if kb >= FAT_ANON_KB:
                    total_kb += kb
    return round(total_kb / 2**20, 2)


def _socket_inodes(pid: int, proc_root: str) -> set[str]:
    directory = f"{proc_root}/{pid}/fd"
    try:
        names = os.listdir(directory)
    except OSError:
        return set()
    inodes: set[str] = set()
    for name in names:
        try:
            target = os.readlink(f"{directory}/{name}")
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[len("socket:["):-1])
    return inodes


def _established_inodes(proc_root: str) -> set[str]:
    """Inodes of every ESTABLISHED TCP socket in the host's proc tables.

    Passive, like the LISTEN check in worker_health: connecting to a monarch
    port is not a harmless probe. Column 4 is the state and column 10 the
    inode.
    """
    inodes: set[str] = set()
    for table in ("net/tcp", "net/tcp6"):
        text = _read_text(f"{proc_root}/{table}")
        if not text:
            continue
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10 and fields[3] == _ESTABLISHED:
                inodes.add(fields[9])
    return inodes


def established_counts(pids: list[int], proc_root: str = PROC_ROOT) -> dict[int, int]:
    """ESTABLISHED socket count per pid, from one pass over the proc tables."""
    established = _established_inodes(proc_root)
    return {
        pid: len(_socket_inodes(pid, proc_root) & established)
        for pid in pids
    }
