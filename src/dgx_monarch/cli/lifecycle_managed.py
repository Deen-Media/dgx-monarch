"""Read-only checks for immutable Worker release links before lifecycle changes."""
from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
from collections.abc import Callable

from ..config import ClusterConfig, HostConfig
from ..log import get_logger
from .lifecycle_host import LifecycleResult
from .lifecycle_lock import locked_script
from .lifecycle_systemd import MANAGED_SRC_REL, systemd_worker_unit

log = get_logger(__name__)
_SENTINEL = "u-000000000000-0000000000000000"
# A unit that differs from its template only in this ExecStart tail is stale,
# not foreign: cluster.toml changed after the unit was written.
POLICY_SUFFIX = r' "--runtime-policy" "([A-Za-z0-9_=-]{1,16384})"( "--native-rdma")?$'
POLICY_STALE = "POLICY_STALE"
POLICY_STALE_EXIT = 3
POLICY_REMEDY = "dgxm update --verify --target HEAD"
Runner = Callable[[ClusterConfig, HostConfig, str], subprocess.CompletedProcess[str]]


def _shape(value: object) -> object:
    return {key: _shape(item) for key, item in value.items()} if isinstance(value, dict) else type(value).__name__


def _shaped(value: object, shape: object) -> bool:
    """The probe's rule: the schema's keys and leaf types, walked by the schema."""
    if isinstance(shape, dict):
        return (isinstance(value, dict) and set(value) == set(shape)
                and all(_shaped(value[key], item) for key, item in shape.items()))
    return type(value).__name__ == shape


def legacy_policy_is_current(config: ClusterConfig, host: HostConfig) -> bool:
    """A unit without policy arguments runs the default policy with native return off."""
    from ..monarch_launch import worker_policy_args
    from ..monarch_policy import MonarchPolicy, encode_policy

    return worker_policy_args(config, host) == ["--runtime-policy", encode_policy(MonarchPolicy(), host.name)]


def without_policy_suffix(unit: str) -> str | None:
    """Return ``unit`` minus a canonical, well-formed policy tail, or None.

    The token must be the exact encoding of a payload with the policy's keys and
    value types, the isolated probe's rule, so a hand-made payload stays
    foreign. Values are not checked: the update captures a unit only beside
    its running Worker, whose release accepted them, and restores it only
    with that release.
    """
    from ..monarch_policy import MonarchPolicy, policy_dict

    lines = unit.split("\n")
    found = [index for index, line in enumerate(lines) if line.startswith("ExecStart=")]
    match = re.search(POLICY_SUFFIX, lines[found[0]]) if len(found) == 1 else None
    if match is None:
        return None
    try:
        payload = json.loads(base64.b64decode(match[1], altchars=b"-_", validate=True))
    except (ValueError, RecursionError):
        return None
    if not _shaped(payload, _shape(policy_dict(MonarchPolicy()))):
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if base64.urlsafe_b64encode(encoded).decode() != match[1]:
        return None
    lines[found[0]] = lines[found[0]][:match.start()]
    return "\n".join(lines)


def managed_source_probe(config: ClusterConfig, host: HostConfig) -> str:
    """Authenticate a release link and its unit without importing installed code.

    A plain legacy directory can still be synchronized. Any link needs a matching
    setup or update unit, and every parent must remain owned and non-symlinked.
    The rsync receiver repeats its no-symlink check under the lifecycle lock.
    A unit whose only difference is a well-formed earlier policy prints
    POLICY_STALE and exits 3; every other mismatch is FAILED_SOURCE_LAYOUT.
    """
    canonical = systemd_worker_unit(
        config, host, managed_source=True, update_token=_SENTINEL,
    )
    hardened = systemd_worker_unit(
        config, host, managed_source=True, update_token=_SENTINEL,
        site_packages="/DGXM_SITE_PACKAGES",
    )
    from ..monarch_launch import worker_policy_args
    from ..monarch_policy import MonarchPolicy, policy_dict
    from .systemd_unit import quote

    suffix = " " + " ".join(quote(arg) for arg in worker_policy_args(config, host))
    legacy_canonical = canonical.replace(suffix + "\n", "\n")
    legacy_hardened = hardened.replace(suffix + "\n", "\n")
    code = r'''
import base64,grp,hashlib,json,os,pathlib,pwd,re,stat,sys
home=pathlib.Path.home(); base=home/'.local/share/dgx-monarch'
src=base/'src'; unit=home/'.config/systemd/user/dgxm-worker.service'
def safe(path,kind):
 info=path.lstat(); forbidden=0o022
 if kind is stat.S_ISDIR and info.st_mode&0o020 and info.st_gid==os.getegid():
  group=grp.getgrgid(info.st_gid); user=pwd.getpwuid(os.geteuid()).pw_name
  if group.gr_name==user and set(group.gr_mem)<={user} and all(entry.pw_uid==os.geteuid() for entry in pwd.getpwall() if entry.pw_gid==info.st_gid):
   forbidden=0o002
 if info.st_uid!=os.geteuid() or info.st_mode&forbidden: raise ValueError('unsafe ownership')
 if not kind(info.st_mode): raise ValueError('unexpected file type')
def parents(path):
 for part in reversed(path.parents):
  if part==home or home in part.parents: safe(part,stat.S_ISDIR)
def shaped(value,shape):
 if isinstance(shape,dict): return isinstance(value,dict) and set(value)==set(shape) and all(shaped(value[key],shape[key]) for key in shape)
 return type(value).__name__==shape
def policy(text,exact,legacy):
 if text==exact: return 'MANAGED'
 if text==legacy: return 'MANAGED' if LEGACY_IS_CURRENT else 'POLICY_STALE'
 lines=text.split('\n'); found=[i for i,line in enumerate(lines) if line.startswith('ExecStart=')]
 match=re.search(POLICY_SUFFIX,lines[found[0]]) if len(found)==1 else None
 if match is None or '\n'.join(lines[:found[0]]+[lines[found[0]][:match.start()]]+lines[found[0]+1:])!=legacy: return None
 payload=json.loads(base64.b64decode(match[1],altchars=b'-_',validate=True))
 if base64.urlsafe_b64encode(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).decode()!=match[1] or not shaped(payload,SHAPE):
  raise ValueError('policy payload')
 return 'POLICY_STALE'
def setup_unit(text,environment):
 bodies=[]
 for template in (CANONICAL,LEGACY_CANONICAL):
  lines=template.splitlines()[2:]
  invalidator=next(line for line in lines if line.startswith('ExecStartPre='))
  lines.remove(invalidator)
  lines=[environment if line.startswith('Environment="PYTHONPATH=') else line for line in lines]
  index=next(i for i,line in enumerate(lines) if line.startswith('ExecStart='))
  lines.insert(index,invalidator)
  lines.insert(lines.index('RestartSec=3')+1,'TimeoutStopSec=20')
  bodies.append('\n'.join([text.splitlines()[0],*lines])+'\n')
 kind=policy(text,*bodies)
 if kind is None: raise ValueError('setup unit mismatch')
 return kind
def external_setup(text,target,raw):
 match=re.fullmatch(r'# dgxm-setup-token=(s-[0-9a-f]{16}) ordinal=([1-9][0-9]*) source=([0-9a-f]{64})',text.splitlines()[0])
 if match is None or not 1<=int(match[2])<=64 or raw!=str(target): raise ValueError('foreign source link')
 token,ordinal,digest=match[1],int(match[2]),match[3]
 txn=home/'.local/state/dgx-monarch/setup'/f'{token}-{ordinal}'
 parents(txn); safe(txn,stat.S_ISDIR)
 def record(name):
  path=txn/name; safe(path,stat.S_ISREG)
  value=json.loads(path.read_text())
  if not isinstance(value,dict): raise ValueError('setup record type')
  if value.get('token')!=token or value.get('ordinal')!=ordinal or value.get('source_manifest')!=digest:
   raise ValueError('setup record identity')
  info=txn.lstat()
  if value.get('txn_dev')!=info.st_dev or value.get('txn_ino')!=info.st_ino: raise ValueError('setup transaction identity')
  return value
 data=record('transaction.json'); proof=record('verified.json')
 if data.get('managed_source') is not False or proof.get('managed_source') is not False:
  raise ValueError('not external setup')
 if data.get('link_target')!=raw or data.get('unit_sha256')!=hashlib.sha256(text.encode()).hexdigest():
  raise ValueError('setup publication mismatch')
 if target.resolve()!=target: raise ValueError('indirect checkout')
 for path,prefix in ((target,'site'),(target/'dgx_monarch','package')):
  info=path.lstat()
  if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid() or info.st_mode&0o002:
   raise ValueError('unsafe checkout')
  if info.st_mode&0o020:
   import grp,pwd
   group=grp.getgrgid(info.st_gid); current=pwd.getpwuid(os.geteuid()).pw_name
   users={entry.pw_name for entry in pwd.getpwall() if entry.pw_gid==info.st_gid}|set(group.gr_mem)
   if users-{current}: raise ValueError('shared checkout group')
  if proof.get(prefix+'_dev')!=info.st_dev or proof.get(prefix+'_ino')!=info.st_ino:
   raise ValueError('checkout identity changed')
 value=('PYTHONPATH='+raw).replace('%','%%').replace('\\','\\\\').replace('"','\\"')
 return setup_unit(text,'Environment="'+value+'"')
def classify():
 text=''
 if unit.exists() or unit.is_symlink():
  if unit.is_symlink(): raise ValueError('unit symlink')
  text=unit.read_text()
 if not src.is_symlink():
  if text.startswith('# dgxm-update-token='): raise ValueError('update link absent')
  if src.exists() and not src.is_dir(): raise ValueError('source not directory')
  return 'PLAIN'
 parents(src); parents(unit); safe(unit,stat.S_ISREG)
 if src.lstat().st_uid!=os.geteuid(): raise ValueError('link owner')
 raw=os.readlink(src); target=pathlib.Path(raw)
 if not target.is_absolute(): target=base/target
 for root in (home/'.config/systemd/user',home/'.local/share/systemd/user',pathlib.Path(os.environ.get('XDG_RUNTIME_DIR','/nonexistent'))/'systemd/user'):
  for name in ('dgxm-worker.service.d','dgxm-.service.d'):
   directory=root/name
   if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
    raise ValueError('unit override')
 try: relative=target.relative_to(base)
 except ValueError:
  return external_setup(text,target,raw)
 if len(relative.parts)!=3 or relative.parts[0]!='releases' or relative.parts[2]!='site':
  raise ValueError('foreign release path')
 token=relative.parts[1]
 for path in (base/'releases',target.parent,target): safe(path,stat.S_ISDIR)
 if target.resolve()!=target: raise ValueError('indirect release path')
 if re.fullmatch(r'u-[0-9a-f]{12}-[0-9a-f]{16}',token):
  kind=policy(text,CANONICAL.replace(SENTINEL,token),LEGACY_CANONICAL.replace(SENTINEL,token))
  if kind is None:
   match=re.search(r'\"PYTHONPATH=%h/\.local/share/dgx-monarch/src:([^\"\n]+)\"',text)
   if match is None or not match[1].startswith('/') or ':' in match[1]: raise ValueError('update source mismatch')
   kind=policy(text,HARDENED.replace(SENTINEL,token).replace('/DGXM_SITE_PACKAGES',match[1]),
               LEGACY_HARDENED.replace(SENTINEL,token).replace('/DGXM_SITE_PACKAGES',match[1]))
   if kind is None: raise ValueError('update unit mismatch')
  return kind
 if re.fullmatch(r's-[0-9a-f]{16}-[1-9][0-9]*',token):
  setup,ordinal=token.rsplit('-',1)
  if not 1<=int(ordinal)<=64: raise ValueError('setup ordinal')
  if not re.fullmatch(r'# dgxm-setup-token='+setup+r' ordinal='+ordinal+r' source=[0-9a-f]{64}',text.splitlines()[0]):
   raise ValueError('setup token mismatch')
  return setup_unit(text,'Environment="PYTHONPATH=%h/.local/share/dgx-monarch/releases/'+token+'/site"')
 raise ValueError('unknown release token')
def verdict():
 kind=classify(); print(kind)
 if kind=='POLICY_STALE': sys.exit(STALE_EXIT)
try: verdict()
except (OSError,ValueError,IndexError,KeyError,RecursionError): print('FAILED_SOURCE_LAYOUT'); sys.exit(1)
'''
    constants = {
        "LEGACY_CANONICAL": legacy_canonical, "LEGACY_HARDENED": legacy_hardened, "HARDENED": hardened,
        "CANONICAL": canonical, "SENTINEL": _SENTINEL, "POLICY_SUFFIX": POLICY_SUFFIX,
        "SHAPE": _shape(policy_dict(MonarchPolicy())), "STALE_EXIT": POLICY_STALE_EXIT,
        "LEGACY_IS_CURRENT": legacy_policy_is_current(config, host),
    }
    code = "".join(f"{name}={value!r}\n" for name, value in constants.items()) + code
    python = shlex.quote(config.python_bin)
    return (
        f'PYBIN={python}\n'
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac\n'
        f'"$PYBIN" -I -S -B -c {shlex.quote(code)}\n'
    )


def managed_start_guard(
    config: ClusterConfig, host: HostConfig, pythonpath: str, *, allow_stale: bool = False,
) -> str:
    """Choose the same source for fallback starts, before stopping any process.

    Only a verified update's generation-bound start passes ``allow_stale``. That
    transaction wrote the unit itself, and its Doctor step fails a stale policy
    before anything is committed. A refusal there would leave no generation
    record, so the update could not stop and restore its own Workers.
    """
    stale = 'DGXM_START_SOURCE="$HOME/.local/share/dgx-monarch/src"' if allow_stale else f"echo {POLICY_STALE}; exit 1"
    return f'''DGXM_SOURCE_KIND=$(\n{managed_source_probe(config, host)})
case "$?:$DGXM_SOURCE_KIND" in
  0:MANAGED) DGXM_START_SOURCE="$HOME/.local/share/dgx-monarch/src";;
  0:PLAIN) DGXM_START_SOURCE={pythonpath};;
  {POLICY_STALE_EXIT}:{POLICY_STALE}) {stale};;
  *) echo FAILED_SOURCE_LAYOUT; exit 1;;
esac
'''


def probe_kind(result: subprocess.CompletedProcess[str]) -> str | None:
    """MANAGED, PLAIN, POLICY_STALE or a FAILED_* verdict; None when unanswered."""
    if result.returncode == 255:
        return None
    if result.returncode == 0:
        return "MANAGED" if result.stdout.strip() == "MANAGED" else "PLAIN"
    if result.returncode == POLICY_STALE_EXIT and result.stdout.strip() == POLICY_STALE:
        return POLICY_STALE
    return probe_failure(result)


def probe_failure(result: subprocess.CompletedProcess[str], *, locked: bool = True) -> str:
    """Name a probe run that gave no verdict its caller accepts.

    Only the probe prints FAILED_SOURCE_LAYOUT. Exit 126 or 127 means the shell
    could not run the Worker interpreter. Under ``locked_script``, exit 1
    without that marker means the lock wrapper (or the probe's interpreter)
    failed before any verdict. Doctor runs the probe without the lock and
    passes ``locked=False``, so there every other exit is FAILED_SOURCE_PROBE.
    """
    if "FAILED_SOURCE_LAYOUT" in result.stdout.split():
        return "FAILED_SOURCE_LAYOUT"
    if result.returncode in (126, 127):
        return "FAILED_WORKER_INTERPRETER"
    return "FAILED_LIFECYCLE_LOCK" if locked and result.returncode == 1 else "FAILED_SOURCE_PROBE"


_PROBE_FAILURES = {
    "FAILED_SOURCE_LAYOUT": "its managed source link or Worker unit is not one dgx-monarch wrote, "
                            "or a unit override exists",
    "FAILED_WORKER_INTERPRETER": "the shell could not run cluster.python there (missing or not "
                                 "executable), so the source check never ran",
    "FAILED_LIFECYCLE_LOCK": "the lifecycle lock wrapper failed before the source check gave a verdict, "
                             "for example on an unsafe ~/.local/state/dgx-monarch",
    "FAILED_SOURCE_PROBE": "the managed source check exited without a verdict",
}
_HOME_PREFIX = re.compile(r"(?:/home|/Users|/var/home)/[^/\s'\":]+|/root(?=[/\s'\":]|$)")
# A path may hold spaces, commas or colons: a quoted one is redacted whole, and
# an unquoted one, as bash prints it, up to the ": " before the error text.
_QUOTED_PATH = re.compile(r"""(['"])/(?:(?!\1).)*\1""")
_OTHER_PATH = re.compile(r"(?<![\w~.])/.*?(?=: |\Z)")


def stderr_tail(stderr: str | None) -> str:
    """The last stderr line, printable, home-relative, at most 200 characters."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    tail = _HOME_PREFIX.sub("~", lines[-1] if lines else "")
    tail = _OTHER_PATH.sub("<path>", _QUOTED_PATH.sub(r"\1<path>\1", tail))
    return "".join(char if char.isprintable() else "?" for char in tail)[:200]


def policy_stale_detail() -> str:
    """The cause and remedy every stale-policy refusal and Doctor row prints."""
    return (
        "the Worker unit carries the Monarch policy of an earlier cluster.toml ([monarch] or "
        "cluster.rdma_latent_return changed after the unit was written), so dgxm up and restart refuse it. "
        f"Stop ComfyUI, then run `{POLICY_REMEDY}` to render the current policy, or restore the earlier "
        "settings (docs/TROUBLESHOOTING.md #113)"
    )


def probe_cause(kind: str, result: subprocess.CompletedProcess[str] | None = None) -> str:
    """A FAILED_* verdict, what it means and, except for a layout verdict, the exit code."""
    detail = _PROBE_FAILURES.get(kind, _PROBE_FAILURES["FAILED_SOURCE_PROBE"])
    code = "" if result is None or kind == "FAILED_SOURCE_LAYOUT" else f" (exit {result.returncode})"
    return f"{kind}: {detail}{code}"


def probe_refusal(host_name: str, kind: str | None, result: subprocess.CompletedProcess[str] | None = None) -> str:
    """Name the probe's verdict so the operator reads a cause, not a bare code."""
    if kind == POLICY_STALE:
        return f"{host_name}: {POLICY_STALE}: {policy_stale_detail()}. Its Worker service was left as it was"
    if kind is None:
        return f"{host_name}: the managed source probe was not observed (transport exited 255)"
    tail = stderr_tail(result.stderr) if result is not None else ""
    return (f"{host_name}: {probe_cause(kind, result)}; nothing was synced or started there"
            + (f"; stderr: {tail}" if tail else ""))


def restart_preflight(config: ClusterConfig, runner: Runner) -> bool:
    """Probe every host's source before a restart stops any Worker service.

    ``up`` would refuse a stale or foreign source only after the stop and leave
    that Worker down, so refuse here instead. An unanswered probe keeps the old
    order, and ``down`` then reports that host itself.
    """
    refusals = []
    for host in config.hosts:
        try:
            result = runner(config, host, locked_script(config.python_bin, managed_source_probe(config, host)))
        except (OSError, subprocess.TimeoutExpired):
            continue
        kind = probe_kind(result)
        if kind is not None and kind not in ("MANAGED", "PLAIN"):
            refusals.append(probe_refusal(host.name, kind, result))
    for line in refusals:
        print(f"  {line}")
    if refusals:
        print("restart refused before stopping any Worker service")
    return not refusals


def start_failure_detail(host_name: str, result: subprocess.CompletedProcess[str]) -> str:
    """The lines ``up`` prints under a host whose start reported failure."""
    if POLICY_STALE in result.stdout.splitlines():
        return f"    {probe_refusal(host_name, POLICY_STALE)}"
    return f"    stdout: {result.stdout.strip()}\n    stderr: {result.stderr.strip()}"


def prepare_source_sync(
    config: ClusterConfig, host: HostConfig, local: bool, runner: Runner,
) -> tuple[bool, LifecycleResult]:
    """Return whether rsync is needed and the definite preparation outcome."""
    result = runner(config, host, locked_script(config.python_bin, managed_source_probe(config, host)))
    kind = probe_kind(result)
    if kind not in ("MANAGED", "PLAIN"):
        log.warning("%s", probe_refusal(host.name, kind, result))
        return False, None if kind is None else False
    if kind == "MANAGED" or local:
        return False, True
    mkdir = runner(config, host, locked_script(config.python_bin, f'mkdir -p "$HOME/{MANAGED_SRC_REL}"'))
    if mkdir.returncode != 0:
        log.warning("mkdir on %s failed: %s", host.name, mkdir.stderr.strip())
    return True, None if mkdir.returncode == 255 else mkdir.returncode == 0
