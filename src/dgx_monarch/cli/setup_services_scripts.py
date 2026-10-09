"""Setup's ownership probe (read-only), staged-source check (writes verified.json), and builder re-exports."""

from __future__ import annotations

import re
import shlex

from .setup_process_inspection import privileged_process_source
from .setup_services_activation_scripts import (
    build_activate_script,
    build_compensate_script,
    build_probe_activation_script,
)
from .setup_services_manifest_script import trusted_json_source, trusted_manifest_source
from .setup_services_storage_scripts import (
    build_cleanup_script,
    build_reserve_script,
    build_stage_probe_script,
)

__all__ = [
    "build_activate_script",
    "build_cleanup_script",
    "build_compensate_script",
    "build_probe_activation_script",
    "build_reserve_script",
    "build_stage_probe_script",
]

_TOKEN = re.compile(r"s-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
STATE_MARKER = "DGXM_SETUP_SERVICE_STATE="


def validate_identity(token: str, ordinal: int, digest: str) -> None:
    if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
        raise ValueError("invalid setup service token")
    if type(ordinal) is not int or not 1 <= ordinal <= 64:
        raise ValueError("invalid setup service host ordinal")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid setup service source digest")


def release_rel(token: str, ordinal: int) -> str:
    validate_identity(token, ordinal, "0" * 64)
    return f".local/share/dgx-monarch/releases/{token}-{ordinal}"


def remote_site(token: str, ordinal: int) -> str:
    return f"$HOME/{release_rel(token, ordinal)}/site"


def systemd_site(token: str, ordinal: int) -> str:
    return f"%h/{release_rel(token, ordinal)}/site"


def _pybin(python_bin: str) -> str:
    return (
        f"PYBIN={shlex.quote(python_bin)}\n"
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac'
    )


def build_ownership_script(
    python_bin: str, address: str, ordinal: int, *, local_source: bool,
    privileged_process_inspection: bool = False,
) -> str:
    """Return passive tri-state unit/process/listener/source evidence."""
    # local_source is unused: both layouts publish the shared ``base/src`` link.
    # Remote setup copies into a private release slot; local setup later binds
    # that link to the separately verified checkout without copying it.
    del local_source
    if type(ordinal) is not int or not 1 <= ordinal <= 64:
        raise ValueError("invalid setup service host ordinal")
    return f"""set -u
{_pybin(python_bin)}
export DGXM_SETUP_ADDRESS={shlex.quote(address)}
exec "$PYBIN" -I -S -B - <<'PY'
import ipaddress,json,os,pathlib,stat,subprocess
{privileged_process_source(privileged_process_inspection)}
def run(args):
 try: return subprocess.run(args,capture_output=True,text=True,timeout=5)
 except (OSError,subprocess.TimeoutExpired): return None
def endpoint(value):
 try:
  value=value.split('://',1)[-1]
  if value.startswith('['): host,port=value[1:].split(']:',1)
  else: host,port=value.rsplit(':',1)
  return ipaddress.ip_address(host.split('%',1)[0]),int(port)
 except (AttributeError,ValueError): return None
def listener_absent(address):
 expected=endpoint(address)
 if expected is None: return None
 ip,port=expected; wanted_port=f'{{port:04X}}'
 if ip.version==4: matches=[('/proc/net/tcp',{{ip.packed[::-1].hex().upper(),'0'*8}}),('/proc/net/tcp6',{{'0'*32}})]
 else:
  packed=b''.join(ip.packed[n:n+4][::-1] for n in range(0,16,4)); matches=[('/proc/net/tcp6',{{packed.hex().upper(),'0'*32}})]
 try: rows=[(row,addresses) for table,addresses in matches for row in pathlib.Path(table).read_text().splitlines()[1:]]
 except OSError: return None
 return not any(len(parts:=row.split())>3 and parts[3]=='0A' and parts[1].split(':')[0] in addresses
                and parts[1].split(':')[1]==wanted_port for row,addresses in rows)
def processes():
 if {privileged_process_inspection!r}:
  _,actors,workers=_privileged_process_facts(DGXM_INSPECTOR_SHA)
  return (None if workers is None else not workers),(None if actors is None else not actors)
 worker=False; actors=False; unknown=False
 try: entries=tuple(pathlib.Path('/proc').iterdir())
 except OSError: return None,None
 for entry in entries:
  if not entry.name.isdigit(): continue
  try:
   if entry.stat().st_uid!=os.geteuid(): continue
   argv=[part.decode('utf-8','replace') for part in (entry/'cmdline').read_bytes().split(b'\\0') if part]
   env=(entry/'environ').read_bytes().split(b'\\0')
  except FileNotFoundError: continue
  except OSError: unknown=True; continue
  worker|=any(argv[n:n+2]==['-m','dgx_monarch.cli.worker_loop'] for n in range(len(argv)-1))
  actor=any(argv[n:n+2]==['-m','monarch._src.actor.bootstrap_main'] for n in range(len(argv)-1))
  bootstrap=any(part.startswith(b'HYPERACTOR_MESH_BOOTSTRAP_MODE=') for part in env)
  owned=any(part.startswith(b'DGXM_PYTHONPATH=') for part in env)
  actors|=actor or (bootstrap and owned)
 return False if worker else None if unknown else True, False if actors else None if unknown else True
address=os.environ['DGXM_SETUP_ADDRESS']; worker_absent,actors_absent=processes()
unit=pathlib.Path.home()/'.config/systemd/user/dgxm-worker.service'
available=run(['systemctl','--user','show-environment'])
systemd_available=None if available is None else available.returncode==0
enabled=run(['systemctl','--user','is-enabled','dgxm-worker.service'])
enable=pathlib.Path.home()/'.config/systemd/user/default.target.wants/dgxm-worker.service'
if os.path.lexists(enable): enablement_absent=False
elif enabled and ((enabled.returncode==1 and enabled.stdout.strip() in ('disabled','not-found')) or (enabled.returncode==4 and enabled.stdout.strip()=='not-found')): enablement_absent=True
else: enablement_absent=None
shown=run(['systemctl','--user','show','dgxm-worker.service','--property=LoadState','--property=ActiveState','--property=FragmentPath','--no-pager'])
loaded=None; inactive=None
if shown and shown.returncode==0:
 fields=dict(line.partition('=')[::2] for line in shown.stdout.splitlines() if '=' in line)
 load=fields.get('LoadState'); active=fields.get('ActiveState'); fragment=fields.get('FragmentPath')
 if load=='not-found' and not fragment: loaded=False
 elif load=='loaded' and pathlib.Path(fragment or '')==unit: loaded=True
 elif load=='loaded': loaded=True
 if active=='inactive' and loaded is not None: inactive=True
 elif active in ('active','activating','deactivating','reloading','failed'): inactive=False
unit_absent=False if os.path.lexists(unit) else (None if loaded is None else not loaded)
base=pathlib.Path.home()/'.local/share/dgx-monarch'
payload={{'ordinal':{ordinal},'unit_absent':unit_absent,'systemd_inactive':inactive,
 'worker_absent':worker_absent,'listener_absent':listener_absent(address),'actors_absent':actors_absent,
 'source_absent':not os.path.lexists(base/'src'),'systemd_available':systemd_available,
 'enablement_absent':enablement_absent}}
print('{STATE_MARKER}'+json.dumps(payload,sort_keys=True,separators=(',',':')))
PY
"""


def build_verify_stage_script(
    python_bin: str,
    token: str,
    ordinal: int,
    digest: str,
    *,
    source_root: str | None = None,
) -> str:
    validate_identity(token, ordinal, digest)
    if source_root is not None and (not source_root.startswith("/") or any(ord(c) < 32 for c in source_root)):
        raise ValueError("local setup source must be an absolute path")
    verifier = trusted_manifest_source()
    reader = trusted_json_source()
    return f"""set -eu
SITE="{remote_site(token, ordinal)}"
TXN="$HOME/.local/state/dgx-monarch/setup/{token}-{ordinal}"
{_pybin(python_bin)}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647
"$PYBIN" -I -S -B - "$SITE" "$TXN" {token!r} {ordinal} {shlex.quote(digest)} {shlex.quote(source_root or "")} <<'PY'
import fcntl,json,os,pathlib,stat,sys
slot_site=pathlib.Path(sys.argv[1]); txn=pathlib.Path(sys.argv[2]); token=sys.argv[3]
ordinal=int(sys.argv[4]); expected=sys.argv[5]; package=pathlib.Path(sys.argv[6]) if sys.argv[6] else slot_site/'dgx_monarch'
site=package.parent; managed=not bool(sys.argv[6])
{verifier}
{reader}
try:
 si=site.lstat(); ssi=slot_site.lstat(); pi=package.lstat(); ti=txn.lstat()
 marker,mi=trusted_json(slot_site.parent/'setup.json'); reservation,ri=trusted_json(txn/'reservation.json')
 lockfd=os.open(txn/'mutation.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0)); li=os.fstat(lockfd)
 safe=(stat.S_ISDIR(si.st_mode) and not stat.S_ISLNK(si.st_mode) and si.st_uid==os.geteuid()
       and stat.S_ISDIR(pi.st_mode) and not stat.S_ISLNK(pi.st_mode) and pi.st_uid==os.geteuid()
       and stat.S_ISDIR(ssi.st_mode) and not stat.S_ISLNK(ssi.st_mode) and ssi.st_uid==os.geteuid()
       and stat.S_ISDIR(ti.st_mode) and not stat.S_ISLNK(ti.st_mode) and ti.st_uid==os.geteuid()
       and marker.get('site_dev')==ssi.st_dev and marker.get('site_ino')==ssi.st_ino
       and marker.get('token')==token and marker.get('ordinal')==ordinal
       and marker.get('source_manifest')==expected and marker.get('txn_dev')==ti.st_dev
       and marker.get('txn_ino')==ti.st_ino and reservation.get('token')==token
       and reservation.get('ordinal')==ordinal and reservation.get('source_manifest')==expected
       and reservation.get('txn_dev')==ti.st_dev and reservation.get('txn_ino')==ti.st_ino
       and reservation.get('lock_dev')==li.st_dev and reservation.get('lock_ino')==li.st_ino
       and stat.S_ISREG(li.st_mode) and not stat.S_ISLNK(li.st_mode) and li.st_uid==os.geteuid())
 if not safe: raise RuntimeError('unsafe staged source')
 try: fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: print('UNKNOWN'); raise SystemExit(0)
 if os.path.lexists(txn/'closed.json'): print('UNKNOWN'); raise SystemExit(0)
 check,_=trusted_json(txn/'reservation.json')
 if check!=reservation: print('UNKNOWN'); raise SystemExit(0)
 actual=trusted_manifest(package); sf=site.lstat(); ssf=slot_site.lstat(); pf=package.lstat()
 if (sf.st_dev,sf.st_ino,pf.st_dev,pf.st_ino)!=(si.st_dev,si.st_ino,pi.st_dev,pi.st_ino):
  raise RuntimeError('staged source identity changed')
 if (ssf.st_dev,ssf.st_ino)!=(ssi.st_dev,ssi.st_ino): raise RuntimeError('transaction slot changed')
except Exception: print('UNKNOWN')
else:
 if actual!=expected: print('MISMATCH')
 else:
  proof={{'schema':1,'token':token,'ordinal':ordinal,'source_manifest':expected,
         'txn_dev':ti.st_dev,'txn_ino':ti.st_ino,'site_dev':si.st_dev,'site_ino':si.st_ino,
         'slot_site_dev':ssi.st_dev,'slot_site_ino':ssi.st_ino,'package_dev':pi.st_dev,
         'package_ino':pi.st_ino,'managed_source':managed}}
  path=txn/'verified.json'
  try:
   fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
  except FileExistsError:
   prior,_=trusted_json(path)
   if prior!=proof: print('UNKNOWN'); raise SystemExit(0)
  else:
   with os.fdopen(fd,'w',encoding='ascii') as out:
    json.dump(proof,out,sort_keys=True,separators=(',',':')); out.flush(); os.fsync(out.fileno())
   parent=os.open(txn,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
   try: os.fsync(parent)
   finally: os.close(parent)
  print('MATCH')
 os.close(lockfd)
PY
"""
