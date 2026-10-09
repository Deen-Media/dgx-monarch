"""Read-only checks for immutable Worker release links before lifecycle changes."""
from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable

from ..config import ClusterConfig, HostConfig
from ..log import get_logger
from .lifecycle_host import LifecycleResult
from .lifecycle_lock import locked_script
from .lifecycle_systemd import MANAGED_SRC_REL, systemd_worker_unit, unit_exec_args

_SENTINEL = "u-000000000000-0000000000000000"


def managed_source_probe(config: ClusterConfig, host: HostConfig) -> str:
    """Authenticate a release link and its unit without importing installed code.

    A plain legacy directory can still be synchronized. Any link needs a matching
    setup or update unit, and every parent must remain owned and non-symlinked.
    The rsync receiver repeats its no-symlink check under the lifecycle lock.
    """
    canonical = systemd_worker_unit(
        config, host, managed_source=True, update_token=_SENTINEL,
    )
    hardened = systemd_worker_unit(
        config, host, managed_source=True, update_token=_SENTINEL,
        site_packages="/DGXM_SITE_PACKAGES",
    )
    code = r'''
import grp,hashlib,json,os,pathlib,pwd,re,stat,sys
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
def setup_unit(text,environment):
 lines=CANONICAL.splitlines()[2:]
 invalidator=next(line for line in lines if line.startswith('ExecStartPre='))
 lines.remove(invalidator)
 lines=[environment if line.startswith('Environment="PYTHONPATH=') else line for line in lines]
 lines.insert(lines.index('ExecStart='+EXEC_ARGS),invalidator)
 lines.insert(lines.index('RestartSec=3')+1,'TimeoutStopSec=20')
 if text!='\n'.join([text.splitlines()[0],*lines])+'\n': raise ValueError('setup unit mismatch')
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
 setup_unit(text,'Environment="'+value+'"')
def verdict():
 text=''
 if unit.exists() or unit.is_symlink():
  if unit.is_symlink(): raise ValueError('unit symlink')
  text=unit.read_text()
 if not src.is_symlink():
  if text.startswith('# dgxm-update-token='): raise ValueError('update link absent')
  if src.exists() and not src.is_dir(): raise ValueError('source not directory')
  print('PLAIN'); return
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
  external_setup(text,target,raw)
  print('MANAGED'); return
 if len(relative.parts)!=3 or relative.parts[0]!='releases' or relative.parts[2]!='site':
  raise ValueError('foreign release path')
 token=relative.parts[1]
 for path in (base/'releases',target.parent,target): safe(path,stat.S_ISDIR)
 if target.resolve()!=target: raise ValueError('indirect release path')
 if re.fullmatch(r'u-[0-9a-f]{12}-[0-9a-f]{16}',token):
  expected=CANONICAL.replace(SENTINEL,token)
  if text!=expected:
   match=re.search(r'\"PYTHONPATH=%h/\.local/share/dgx-monarch/src:([^\"\n]+)\"',text)
   if match is None or not match[1].startswith('/') or ':' in match[1]: raise ValueError('update source mismatch')
   expected=HARDENED.replace(SENTINEL,token).replace('/DGXM_SITE_PACKAGES',match[1])
   if text!=expected: raise ValueError('update unit mismatch')
 elif re.fullmatch(r's-[0-9a-f]{16}-[1-9][0-9]*',token):
  setup,ordinal=token.rsplit('-',1)
  if not 1<=int(ordinal)<=64: raise ValueError('setup ordinal')
  if not re.fullmatch(r'# dgxm-setup-token='+setup+r' ordinal='+ordinal+r' source=[0-9a-f]{64}',text.splitlines()[0]):
   raise ValueError('setup token mismatch')
  expected='Environment="PYTHONPATH=%h/.local/share/dgx-monarch/releases/'+token+'/site"'
  setup_unit(text,expected)
 else: raise ValueError('unknown release token')
 print('MANAGED')
try: verdict()
except (OSError,ValueError,IndexError,KeyError): print('FAILED_SOURCE_LAYOUT'); sys.exit(1)
'''
    code = f"HARDENED={hardened!r}\nCANONICAL={canonical!r}\nSENTINEL={_SENTINEL!r}\nEXEC_ARGS={unit_exec_args(config, host)!r}\n" + code
    python = shlex.quote(config.python_bin)
    return (
        f'PYBIN={python}\n'
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac\n'
        f'"$PYBIN" -I -S -B -c {shlex.quote(code)}\n'
    )


def managed_start_guard(config: ClusterConfig, host: HostConfig, pythonpath: str) -> str:
    """Choose the same source for fallback starts, before stopping any process."""
    return f'''DGXM_SOURCE_KIND=$(\n{managed_source_probe(config, host)}) || {{ echo FAILED_SOURCE_LAYOUT; exit 1; }}
case "$DGXM_SOURCE_KIND" in
  MANAGED) DGXM_START_SOURCE="$HOME/.local/share/dgx-monarch/src";;
  PLAIN) DGXM_START_SOURCE={pythonpath};;
  *) echo FAILED_SOURCE_LAYOUT; exit 1;;
esac
'''


def prepare_source_sync(
    config: ClusterConfig, host: HostConfig, local: bool,
    runner: Callable[[ClusterConfig, HostConfig, str], subprocess.CompletedProcess[str]],
) -> tuple[bool, LifecycleResult]:
    """Return whether rsync is needed and the definite preparation outcome."""
    managed = runner(config, host, locked_script(config.python_bin, managed_source_probe(config, host)))
    if managed.returncode != 0:
        return False, None if managed.returncode == 255 else False
    if managed.stdout.strip() == "MANAGED" or local:
        return False, True
    mkdir = runner(config, host, locked_script(config.python_bin, f'mkdir -p "$HOME/{MANAGED_SRC_REL}"'))
    if mkdir.returncode != 0:
        get_logger(__name__).warning("mkdir on %s failed: %s", host.name, mkdir.stderr.strip())
    return True, None if mkdir.returncode == 255 else mkdir.returncode == 0
