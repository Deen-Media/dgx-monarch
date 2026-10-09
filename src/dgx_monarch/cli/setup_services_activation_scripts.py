"""Setup's scripts that publish the worker unit and source link, read the publication back, and undo it."""

from __future__ import annotations

import base64
import re
import shlex

from .lifecycle_lock import lifecycle_lock_source
from .setup_process_inspection import privileged_process_source
from .setup_services_compensation_script import build_compensation_script
from .setup_services_generation import generation_reader_source, generation_recorder_source
from .setup_services_manifest_script import trusted_json_source, trusted_manifest_source

_TOKEN = re.compile(r"s-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _validate(token: str, ordinal: int, digest: str) -> None:
    if _TOKEN.fullmatch(token) is None or _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid setup service identity")
    if type(ordinal) is not int or not 1 <= ordinal <= 64:
        raise ValueError("invalid setup service host ordinal")


def _pybin(python_bin: str) -> str:
    return (
        f"PYBIN={shlex.quote(python_bin)}\n"
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac'
    )


def build_activate_script(
    python_bin: str,
    token: str,
    ordinal: int,
    digest: str,
    unit_text: str,
    address: str,
    *,
    start_service: bool,
    source_site: str | None = None,
    privileged_process_inspection: bool = False,
) -> str:
    """Build a script that hard-links only objects recorded in the journal.

    Hold the transaction, lifecycle and generation locks while linking.
    """
    _validate(token, ordinal, digest)
    if source_site is not None and (not source_site.startswith("/") or any(ord(c) < 32 for c in source_site)):
        raise ValueError("local setup source site must be absolute")
    encoded = base64.b64encode(unit_text.encode()).decode("ascii")
    source_encoded = base64.b64encode((source_site or "").encode()).decode("ascii")
    verifier, reader = trusted_manifest_source(), trusted_json_source()
    lifecycle_lock = lifecycle_lock_source()
    generation = generation_recorder_source(token)
    return f"""set -eu
{_pybin(python_bin)}
export DGXM_TOKEN={token} DGXM_ORDINAL={ordinal} DGXM_DIGEST={digest}
export DGXM_ADDRESS={shlex.quote(address)} DGXM_UNIT_B64={encoded} DGXM_START={int(start_service)}
export DGXM_SOURCE_B64={source_encoded} DGXM_MANAGED={int(source_site is None)}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647
"$PYBIN" -I -S -B - <<'PY'
import base64,fcntl,grp,hashlib,ipaddress,json,os,pathlib,pwd,stat,subprocess,time
{verifier}
{reader}
{lifecycle_lock}
home=pathlib.Path.home(); token=os.environ['DGXM_TOKEN']; ordinal=int(os.environ['DGXM_ORDINAL'])
{generation}
{privileged_process_source(privileged_process_inspection)}
digest=os.environ['DGXM_DIGEST']; start=os.environ['DGXM_START']=='1'; address=os.environ['DGXM_ADDRESS']
managed=os.environ['DGXM_MANAGED']=='1'
def unsafe(info):
 if info.st_uid!=os.geteuid() or info.st_mode&0o002: return True
 if not info.st_mode&0o020: return False
 try:
  current=pwd.getpwuid(os.geteuid()).pw_name; group=grp.getgrgid(info.st_gid)
  users={{entry.pw_name for entry in pwd.getpwall() if entry.pw_gid==info.st_gid}}|set(group.gr_mem)
 except KeyError: return True
 return bool(users-{{current}})
def directory(path):
 info=path.lstat()
 if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or unsafe(info): raise SystemExit(30)
 return path,info
base,base_info=directory(home/'.local/share/dgx-monarch'); releases,_=directory(base/'releases')
slot,slot_info=directory(releases/f'{{token}}-{{ordinal}}'); slot_site,slot_site_info=directory(slot/'site')
site=slot_site if managed else pathlib.Path(base64.b64decode(os.environ['DGXM_SOURCE_B64']).decode())
site,site_info=directory(site); package,package_info=directory(site/'dgx_monarch')
units,units_info=directory(home/'.config/systemd/user')
wants,wants_info=directory(units/'default.target.wants'); setup,_=directory(home/'.local/state/dgx-monarch/setup')
txn,txn_info=directory(setup/f'{{token}}-{{ordinal}}')
reservation,_=trusted_json(txn/'reservation.json'); marker,mi=trusted_json(slot/'setup.json')
slot_record,_=trusted_json(txn/'slot.json')
proof,_=trusted_json(txn/'verified.json'); lock_info=(txn/'mutation.lock').lstat()
if (reservation.get('token')!=token or reservation.get('ordinal')!=ordinal
    or reservation.get('source_manifest')!=digest or reservation.get('txn_dev')!=txn_info.st_dev
    or reservation.get('txn_ino')!=txn_info.st_ino or reservation.get('lock_dev')!=lock_info.st_dev
    or reservation.get('lock_ino')!=lock_info.st_ino or not stat.S_ISREG(lock_info.st_mode)
    or stat.S_ISLNK(lock_info.st_mode) or lock_info.st_uid!=os.geteuid()
    or slot_record!=marker or marker.get('slot_dev')!=slot_info.st_dev or marker.get('slot_ino')!=slot_info.st_ino
    or marker.get('site_dev')!=slot_site_info.st_dev or marker.get('site_ino')!=slot_site_info.st_ino
    or marker.get('txn_dev')!=txn_info.st_dev or marker.get('txn_ino')!=txn_info.st_ino
    or marker.get('token')!=token or marker.get('ordinal')!=ordinal or marker.get('source_manifest')!=digest
    or proof.get('token')!=token or proof.get('ordinal')!=ordinal or proof.get('source_manifest')!=digest
    or proof.get('txn_dev')!=txn_info.st_dev or proof.get('txn_ino')!=txn_info.st_ino
    or proof.get('site_dev')!=site_info.st_dev or proof.get('site_ino')!=site_info.st_ino
    or proof.get('slot_site_dev')!=slot_site_info.st_dev or proof.get('slot_site_ino')!=slot_site_info.st_ino
    or proof.get('managed_source') is not managed
    or proof.get('package_dev')!=package_info.st_dev or proof.get('package_ino')!=package_info.st_ino):
 raise SystemExit(31)
lockfd=os.open(txn/'mutation.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0)); li=os.fstat(lockfd)
if (li.st_dev,li.st_ino)!=(lock_info.st_dev,lock_info.st_ino): raise SystemExit(31)
try: fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit(32)
if os.path.lexists(txn/'closed.json'): raise SystemExit(32)
reservation2,_=trusted_json(txn/'reservation.json'); marker2,_=trusted_json(slot/'setup.json')
slot_record2,_=trusted_json(txn/'slot.json'); proof2,_=trusted_json(txn/'verified.json')
if (reservation2!=reservation or marker2!=marker or slot_record2!=slot_record or proof2!=proof
    or (txn.lstat().st_dev,txn.lstat().st_ino)!=(txn_info.st_dev,txn_info.st_ino)
    or (slot.lstat().st_dev,slot.lstat().st_ino)!=(slot_info.st_dev,slot_info.st_ino)
    or (slot_site.lstat().st_dev,slot_site.lstat().st_ino)!=(slot_site_info.st_dev,slot_site_info.st_ino)
    or (site.lstat().st_dev,site.lstat().st_ino)!=(site_info.st_dev,site_info.st_ino)
    or (package.lstat().st_dev,package.lstat().st_ino)!=(package_info.st_dev,package_info.st_ino)):
 raise SystemExit(32)
try: globalfd=acquire_lifecycle_lock(home,True)
except (BlockingIOError,OSError,RuntimeError): raise SystemExit(32)
def run(args,timeout=10):
 try: return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 except (OSError,subprocess.TimeoutExpired): return None
def manager():
 row=run(['systemctl','--user','show','dgxm-worker.service','--property=LoadState','--property=ActiveState','--property=SubState','--property=FragmentPath','--property=MainPID','--property=Job','--no-pager'])
 return None if row is None or row.returncode else dict(line.partition('=')[::2] for line in row.stdout.splitlines() if '=' in line)
def endpoint_listener(pid=0):
 try:
  value=address.split('://',1)[-1]
  host,port=(value[1:].split(']:',1) if value.startswith('[') else value.rsplit(':',1))
  ip=ipaddress.ip_address(host.split('%',1)[0]); wanted=f'{{int(port):04X}}'; matches=[]
  if ip.version==4:
   matches=[('/proc/net/tcp',{{ip.packed[::-1].hex().upper(),'0'*8}}),('/proc/net/tcp6',{{'0'*32}})]
  else:
   packed=b''.join(ip.packed[n:n+4][::-1] for n in range(0,16,4))
   matches=[('/proc/net/tcp6',{{packed.hex().upper(),'0'*32}})]
  found={{p[9] for path,addresses in matches for row in pathlib.Path(path).read_text().splitlines()[1:]
         if len(p:=row.split())>9 and p[3]=='0A' and p[1].split(':')[0] in addresses
         and p[1].split(':')[1]==wanted}}
  if not pid: return bool(found)
  owned={{os.readlink(fd).removeprefix('socket:[').removesuffix(']') for fd in (pathlib.Path('/proc')/str(pid)/'fd').iterdir()
         if os.readlink(fd).startswith('socket:[')}}
  return bool(found & owned)
 except (AttributeError,OSError,ValueError): return None
def runtime_clear():
 if {privileged_process_inspection!r}:
  _,actors,workers=_privileged_process_facts(DGXM_INSPECTOR_SHA)
  if actors is True or workers is True: return False
  busy=endpoint_listener()
  return None if actors is None or workers is None or busy is None else not busy
 unknown=False
 try: entries=tuple(pathlib.Path('/proc').iterdir())
 except OSError: return None
 for entry in entries:
  if not entry.name.isdigit(): continue
  try:
   if entry.stat().st_uid!=os.geteuid(): continue
   argv=[v.decode('utf-8','replace') for v in (entry/'cmdline').read_bytes().split(b'\\0') if v]
   env=(entry/'environ').read_bytes().split(b'\\0')
  except FileNotFoundError: continue
  except OSError: unknown=True; continue
  worker=any(argv[n:n+2]==['-m','dgx_monarch.cli.worker_loop'] for n in range(len(argv)-1))
  actor=any(argv[n:n+2]==['-m','monarch._src.actor.bootstrap_main'] for n in range(len(argv)-1))
  bootstrap=any(v.startswith(b'HYPERACTOR_MESH_BOOTSTRAP_MODE=') for v in env)
  owned=any(v.startswith(b'DGXM_PYTHONPATH=') for v in env)
  if worker or actor or (bootstrap and owned): return False
 busy=endpoint_listener()
 return None if unknown or busy is None else not busy
state=manager(); enabled=run(['systemctl','--user','is-enabled','dgxm-worker.service'])
linger=run(['loginctl','show-user',str(os.geteuid()),'--property=Linger','--value'])
unit=units/'dgxm-worker.service'; live=base/'src'; enable=wants/'dgxm-worker.service'
if (state is None or state.get('LoadState')!='not-found' or state.get('ActiveState')!='inactive'
    or enabled is None or not ((enabled.returncode==1 and enabled.stdout.strip() in ('disabled','not-found')) or (enabled.returncode==4 and enabled.stdout.strip()=='not-found'))
    or linger is None or linger.returncode or linger.stdout.strip()!='yes'
    or any(os.path.lexists(p) for p in (unit,live,enable,txn/'transaction.json'))): raise SystemExit(33)
content=base64.b64decode(os.environ['DGXM_UNIT_B64']); target=f'releases/{{token}}-{{ordinal}}/site' if managed else str(site)
data={{'schema':1,'token':token,'ordinal':ordinal,'source_manifest':digest,'address':address,
 'txn_dev':txn_info.st_dev,'txn_ino':txn_info.st_ino,'unit_sha256':hashlib.sha256(content).hexdigest(),
 'privileged_process_inspection':{privileged_process_inspection!r},
 'start':start,'managed_source':managed,'link_target':target,'unit_tmp':'unit.tmp','live_tmp':'live.tmp',
 'enable_tmp':'enable.tmp','enable_target':'../dgxm-worker.service','created':reservation.get('created',[])}}
fd=os.open(txn/'transaction.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
with os.fdopen(fd,'w',encoding='ascii') as out: json.dump(data,out,sort_keys=True,separators=(',',':')); out.flush(); os.fsync(out.fileno())
fd=os.open(txn/data['unit_tmp'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
with os.fdopen(fd,'wb') as out: out.write(content); out.flush(); os.fsync(out.fileno())
os.symlink(target,txn/data['live_tmp'])
if start: os.symlink(data['enable_target'],txn/data['enable_tmp'])
txnfd=os.open(txn,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(txnfd)
# Bind the candidate source and activity again immediately before publication.
if trusted_manifest(package)!=digest or runtime_clear() is not True: raise SystemExit(34)
sf=site.lstat(); pf=package.lstat()
if (sf.st_dev,sf.st_ino,pf.st_dev,pf.st_ino)!=(site_info.st_dev,site_info.st_ino,package_info.st_dev,package_info.st_ino): raise SystemExit(34)
basefd=os.open(base,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
unitfd=os.open(units,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
wantsfd=os.open(wants,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
if ((os.fstat(txnfd).st_dev,os.fstat(txnfd).st_ino)!=(txn_info.st_dev,txn_info.st_ino)
    or (os.fstat(basefd).st_dev,os.fstat(basefd).st_ino)!=(base_info.st_dev,base_info.st_ino)
    or (os.fstat(unitfd).st_dev,os.fstat(unitfd).st_ino)!=(units_info.st_dev,units_info.st_ino)
    or (os.fstat(wantsfd).st_dev,os.fstat(wantsfd).st_ino)!=(wants_info.st_dev,wants_info.st_ino)):
 raise SystemExit(34)
generationfd=acquire_generation_fence()
if not record_generation(generationfd,invalidate=True): raise SystemExit(38)
os.link(data['live_tmp'],'src',src_dir_fd=txnfd,dst_dir_fd=basefd,follow_symlinks=False); os.fsync(basefd)
os.link(data['unit_tmp'],'dgxm-worker.service',src_dir_fd=txnfd,dst_dir_fd=unitfd,follow_symlinks=False); os.fsync(unitfd)
if start: os.link(data['enable_tmp'],'dgxm-worker.service',src_dir_fd=txnfd,dst_dir_fd=wantsfd,follow_symlinks=False); os.fsync(wantsfd)
for fd in (txnfd,basefd,unitfd,wantsfd): os.close(fd)
row=run(['systemctl','--user','daemon-reload'])
if row is None or row.returncode: raise SystemExit(35)
if start:
 release_generation_fence(generationfd); generationfd=None
 row=run(['systemctl','--user','start','dgxm-worker.service'],30)
 if row is None or row.returncode: raise SystemExit(36)
 def started_identity():
  fields=manager()
  try:
   pid=int(fields.get('MainPID','')) if fields else 0
   proc=pathlib.Path('/proc')/str(pid); argv=[v.decode('utf-8','replace') for v in (proc/'cmdline').read_bytes().split(b'\\0') if v]
   tail=(proc/'stat').read_text().rsplit(')',1)[1].split(); born=tail[19]
  except (IndexError,OSError,ValueError): return None
  worker=any(argv[n:n+2]==['-m','dgx_monarch.cli.worker_loop'] for n in range(len(argv)-1))
  endpoint=any(argv[n:n+2]==['--address',address] for n in range(len(argv)-1))
  if (fields is None or fields.get('LoadState')!='loaded' or fields.get('ActiveState')!='active'
      or pathlib.Path(fields.get('FragmentPath',''))!=unit or pid<=0 or not worker or not endpoint): return None
  return str(pid),born
 deadline=time.monotonic()+20; identity=started_identity()
 while (identity is None or endpoint_listener(int(identity[0])) is not True) and time.monotonic()<deadline:
  time.sleep(.25); identity=started_identity()
 if identity is None: raise SystemExit(38)
 if endpoint_listener(int(identity[0])) is not True: raise SystemExit(41)
 generationfd=acquire_generation_fence(); identity=started_identity()
 if (generationfd is None or identity is None
     or endpoint_listener(int(identity[0])) is not True): raise SystemExit(39)
 if not record_generation(generationfd,*identity): raise SystemExit(39)
 if (started_identity()!=identity or endpoint_listener(int(identity[0])) is not True
     or generation_identity()!=identity): raise SystemExit(40)
if not managed:
 archive=txn/'local-slot'
 if os.path.lexists(archive): raise SystemExit(37)
 current=slot.lstat()
 if (current.st_dev,current.st_ino)!=(slot_info.st_dev,slot_info.st_ino): raise SystemExit(37)
 os.rename(slot,archive); moved=archive.lstat()
 if (moved.st_dev,moved.st_ino)!=(slot_info.st_dev,slot_info.st_ino): raise SystemExit(37)
 for parent_path in (releases,txn):
  parent=os.open(parent_path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
print('PUBLISHED')
PY
"""


def _header(
    token: str, ordinal: int, digest: str, *, lock: bool,
    privileged_process_inspection: bool = False,
) -> str:
    _validate(token, ordinal, digest)
    reader, verifier = trusted_json_source(), trusted_manifest_source()
    lifecycle_lock = lifecycle_lock_source()
    generation = generation_reader_source(token)
    lock_code = (
        """
lockfd=os.open(txn/'mutation.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0)); li=os.fstat(lockfd)
if (li.st_dev,li.st_ino)!=(reservation.get('lock_dev'),reservation.get('lock_ino')):
 print('UNKNOWN'); raise SystemExit(0)
try: fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: print('UNKNOWN'); raise SystemExit(0)
check,_=trusted_json(txn/'reservation.json'); slot_check,_=trusted_json(txn/'slot.json'); tf=txn.lstat()
if check!=reservation or slot_check!=slot_record or (tf.st_dev,tf.st_ino)!=(ti.st_dev,ti.st_ino) or os.path.lexists(txn/'closed.json'):
 print('UNKNOWN'); raise SystemExit(0)
try: globalfd=acquire_lifecycle_lock(home,True)
except (BlockingIOError,OSError,RuntimeError): print('UNKNOWN'); raise SystemExit(0)
generationfd=acquire_generation_fence()
if generationfd is None: print('UNKNOWN'); raise SystemExit(0)
"""
        if lock
        else ""
    )
    return f"""import fcntl,grp,hashlib,ipaddress,json,os,pathlib,pwd,stat,subprocess,time
{reader}
{verifier}
{lifecycle_lock}
{privileged_process_source(privileged_process_inspection)}
def unsafe(info):
 if info.st_uid!=os.geteuid() or info.st_mode&0o002: return True
 if not info.st_mode&0o020: return False
 try:
  current=pwd.getpwuid(os.geteuid()).pw_name; group=grp.getgrgid(info.st_gid)
  users={{entry.pw_name for entry in pwd.getpwall() if entry.pw_gid==info.st_gid}}|set(group.gr_mem)
 except KeyError: return True
 return bool(users-{{current}})
home=pathlib.Path.home(); base=home/'.local/share/dgx-monarch'; unit=home/'.config/systemd/user/dgxm-worker.service'
{generation}
txn=home/'.local/state/dgx-monarch/setup/{token}-{ordinal}'
try: reservation,_=trusted_json(txn/'reservation.json'); slot_record,_=trusted_json(txn/'slot.json'); ti=txn.lstat()
except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
if (reservation.get('token')!={token!r} or reservation.get('ordinal')!={ordinal}
    or reservation.get('source_manifest')!={digest!r} or reservation.get('txn_dev')!=ti.st_dev
    or reservation.get('txn_ino')!=ti.st_ino or slot_record.get('token')!={token!r}
    or slot_record.get('ordinal')!={ordinal} or slot_record.get('source_manifest')!={digest!r}
    or not stat.S_ISDIR(ti.st_mode) or stat.S_ISLNK(ti.st_mode)
    or ti.st_uid!=os.geteuid()): print('UNKNOWN'); raise SystemExit(0)
{lock_code}
try: data,_=trusted_json(txn/'transaction.json')
except (OSError,RuntimeError,ValueError):
 clean=not any(os.path.lexists(path) for path in (home/'.config/systemd/user/dgxm-worker.service',
       home/'.config/systemd/user/default.target.wants/dgxm-worker.service',base/'src'))
 print('COMPENSATED' if {lock!r} and clean else 'UNKNOWN'); raise SystemExit(0)
if (data.get('schema')!=1 or data.get('token')!={token!r} or data.get('ordinal')!={ordinal}
    or data.get('source_manifest')!={digest!r} or data.get('txn_dev')!=ti.st_dev
    or data.get('txn_ino')!=ti.st_ino
    or data.get('privileged_process_inspection',False) is not {privileged_process_inspection!r}): print('UNKNOWN'); raise SystemExit(0)
def run(args,timeout=10):
 try: return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 except (OSError,subprocess.TimeoutExpired): return None
def temp(name,kind):
 path=txn/data[name]
 if not os.path.lexists(path): return None
 try:
  one=path.lstat(); value=path.read_bytes() if kind=='file' else os.readlink(path); two=path.lstat()
 except OSError: return False
 if (one.st_dev,one.st_ino)!=(two.st_dev,two.st_ino) or one.st_uid!=os.geteuid(): return False
 if kind=='file': return one if stat.S_ISREG(one.st_mode) and hashlib.sha256(value).hexdigest()==data.get('unit_sha256') else False
 target=data.get('link_target') if name=='live_tmp' else data.get('enable_target')
 return one if stat.S_ISLNK(one.st_mode) and value==target else False
def shared(path,name,kind):
 if not os.path.lexists(path): return 'absent'
 expected=temp(name,kind)
 if expected is None or expected is False: return 'foreign'
 try: one=path.lstat(); value=path.read_bytes() if kind=='file' else os.readlink(path); two=path.lstat()
 except OSError: return 'unknown'
 target=data.get('unit_sha256') if kind=='file' else data.get('link_target') if name=='live_tmp' else data.get('enable_target')
 actual=hashlib.sha256(value).hexdigest() if kind=='file' else value
 return 'owned' if (one.st_dev,one.st_ino)==(two.st_dev,two.st_ino)==(expected.st_dev,expected.st_ino) and actual==target else 'foreign'
def unit_state(): return shared(unit,'unit_tmp','file')
def live_state(): return shared(base/'src','live_tmp','link')
def enable_state(): return shared(home/'.config/systemd/user/default.target.wants/dgxm-worker.service','enable_tmp','link')
def manager():
 row=run(['systemctl','--user','show','dgxm-worker.service','--property=LoadState','--property=ActiveState','--property=SubState','--property=FragmentPath','--property=MainPID','--property=Job','--no-pager'])
 return None if row is None or row.returncode else dict(line.partition('=')[::2] for line in row.stdout.splitlines() if '=' in line)
def inactive_settled(fields):
 if (fields is None or fields.get('ActiveState')!='inactive' or fields.get('SubState')!='dead'
     or fields.get('MainPID')!='0' or fields.get('Job') not in ('','0')): return False
 load=fields.get('LoadState'); fragment=fields.get('FragmentPath','')
 return ((load=='loaded' and pathlib.Path(fragment)==unit)
     or (load=='not-found' and not fragment))
def loaded(fields):
 if fields is None: return 'unknown'
 if fields.get('LoadState')=='not-found' and not fields.get('FragmentPath'): return 'absent'
 if fields.get('LoadState')=='loaded' and pathlib.Path(fields.get('FragmentPath',''))==unit: return 'owned'
 return 'foreign'
def enabled():
 row=run(['systemctl','--user','is-enabled','dgxm-worker.service'])
 if row and row.returncode==0 and row.stdout.strip() in ('enabled','enabled-runtime'): return True
 if row and ((row.returncode==1 and row.stdout.strip() in ('disabled','not-found')) or (row.returncode==4 and row.stdout.strip()=='not-found')): return False
 return None
def processes(fields):
 if {privileged_process_inspection!r}:
  try: pid=int(fields.get('MainPID','')) if fields else 0
  except (TypeError,ValueError): return None,None,None,0
  worker,actors,any_worker=_privileged_process_facts(DGXM_INSPECTOR_SHA,data.get('address',''),pid)
  return worker,actors,any_worker,pid
 try: pid=int(fields.get('MainPID','')) if fields else 0; entries=tuple(pathlib.Path('/proc').iterdir())
 except (OSError,ValueError): return None,None,None,pid
 worker=False; actors=False; any_worker=False; uncertain=False
 for entry in entries:
  if not entry.name.isdigit(): continue
  try:
   if entry.stat().st_uid!=os.geteuid(): continue
   argv=[v.decode('utf-8','replace') for v in (entry/'cmdline').read_bytes().split(b'\\0') if v]
   env=(entry/'environ').read_bytes().split(b'\\0')
  except FileNotFoundError: continue
  except OSError: uncertain=True; continue
  module=any(argv[n:n+2]==['-m','dgx_monarch.cli.worker_loop'] for n in range(len(argv)-1)); any_worker|=module
  worker|=module and entry.name==str(pid) and any(argv[n:n+2]==['--address',data.get('address')] for n in range(len(argv)-1))
  actor=any(argv[n:n+2]==['-m','monarch._src.actor.bootstrap_main'] for n in range(len(argv)-1))
  actors|=actor or (any(v.startswith(b'HYPERACTOR_MESH_BOOTSTRAP_MODE=') for v in env)
                    and any(v.startswith(b'DGXM_PYTHONPATH=') for v in env))
 return (None if uncertain else worker),(None if uncertain else actors),(None if uncertain else any_worker),pid
def listener(pid=0):
 try:
  value=data.get('address','').split('://',1)[-1]; host,port=(value[1:].split(']:',1) if value.startswith('[') else value.rsplit(':',1))
  ip=ipaddress.ip_address(host.split('%',1)[0]); wanted=f'{{int(port):04X}}'; matches=[]
  if ip.version==4: matches=[('/proc/net/tcp',{{ip.packed[::-1].hex().upper(),'0'*8}}),('/proc/net/tcp6',{{'0'*32}})]
  else:
   packed=b''.join(ip.packed[n:n+4][::-1] for n in range(0,16,4)); matches=[('/proc/net/tcp6',{{packed.hex().upper(),'0'*32}})]
  found={{p[9] for path,addresses in matches for row in pathlib.Path(path).read_text().splitlines()[1:]
         if len(p:=row.split())>9 and p[3]=='0A' and p[1].split(':')[0] in addresses and p[1].split(':')[1]==wanted}}
  if not pid: return bool(found)
  owned={{os.readlink(fd).removeprefix('socket:[').removesuffix(']') for fd in (pathlib.Path('/proc')/str(pid)/'fd').iterdir()
         if os.readlink(fd).startswith('socket:[')}}
  return bool(found & owned)
 except (AttributeError,OSError,ValueError): return None
def birth(pid):
 if not pid: return None
 try: return (pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
 except (IndexError,OSError): return None
"""


def build_probe_activation_script(
    python_bin: str, token: str, ordinal: int, digest: str, *,
    privileged_process_inspection: bool = False,
) -> str:
    header = _header(token, ordinal, digest, lock=False,
                     privileged_process_inspection=privileged_process_inspection)
    return f"""{_pybin(python_bin)}
"$PYBIN" -I -S -B - <<'PY'
{header}
deadline=time.monotonic()+20
def source_current():
 try:
  release=base/'releases'/f'{{reservation.get("token")}}-{{reservation.get("ordinal")}}'
  if data.get('managed_source'):
   source=release/'site'; package=source/'dgx_monarch'; ri=release.lstat()
   marker,_=trusted_json(release/'setup.json')
   location_ok=((ri.st_dev,ri.st_ino)==(slot_record.get('slot_dev'),slot_record.get('slot_ino'))
       and marker==slot_record)
  else:
   source=pathlib.Path(data.get('link_target','')); package=source/'dgx_monarch'
   archive=txn/'local-slot'; ai=archive.lstat(); marker,_=trusted_json(archive/'setup.json')
   archived_site=(archive/'site').lstat()
   location_ok=(not os.path.lexists(release)
       and (ai.st_dev,ai.st_ino)==(slot_record.get('slot_dev'),slot_record.get('slot_ino'))
       and (archived_site.st_dev,archived_site.st_ino)==(slot_record.get('site_dev'),slot_record.get('site_ino'))
       and marker==slot_record)
  si=source.lstat(); pi=package.lstat()
  site_ok=(stat.S_ISDIR(si.st_mode) and not stat.S_ISLNK(si.st_mode) and not unsafe(si))
  if data.get('managed_source'):
   site_ok=site_ok and (si.st_dev,si.st_ino)==(slot_record.get('site_dev'),slot_record.get('site_ino'))
  return (location_ok and site_ok and stat.S_ISDIR(pi.st_mode) and not stat.S_ISLNK(pi.st_mode) and not unsafe(pi)
      and trusted_manifest(package)==data.get('source_manifest'))
 except (OSError,RuntimeError,ValueError): return False
while True:
 u=unit_state(); live=live_state(); e=enable_state(); fields=manager(); load=loaded(fields); on=enabled()
 active=None if fields is None else fields.get('ActiveState')=='active' if fields.get('ActiveState') in ('active','inactive') else None
 worker,actors,any_worker,pid=processes(fields); listening=listener(pid if data.get('start') else 0)
 born=birth(pid) if data.get('start') else 'idle'
 expected_identity=(str(pid),born) if data.get('start') and born is not None else ('0','0') if not data.get('start') else None
 authority=expected_identity is not None and generation_identity()==expected_identity
 run_ok=(worker is True and actors is False and listening is True) if data.get('start') else (any_worker is False and actors is False and listening is False)
 enable_ok=(e=='owned' and on is True) if data.get('start') else (e=='absent' and on is False)
 if (authority is True and source_current() and u=='owned' and live=='owned' and load=='owned'
     and active is data.get('start') and enable_ok and run_ok):
  again=manager(); same=(again is not None and fields is not None and again.get('MainPID')==fields.get('MainPID')
        and again.get('ActiveState')==fields.get('ActiveState') and loaded(again)=='owned'
        and unit_state()=='owned' and live_state()=='owned' and enable_state()==e
        and generation_identity()==expected_identity)
  worker2,actors2,any_worker2,pid2=processes(again)
  stable_run=(pid2==pid and worker2 is True and actors2 is False and any_worker2 is True
              and listener(pid2) is True and born is not None and birth(pid2)==born) if data.get('start') else (
              any_worker2 is False and actors2 is False and listener() is False)
  if same and stable_run and source_current(): print('NEW'); break
 if (authority is not True or not data.get('start') or time.monotonic()>=deadline or u!='owned'
     or live!='owned' or e!='owned' or load!='owned' or on is not True or active is False):
  print('UNKNOWN'); break
 time.sleep(.25)
PY
"""


def build_compensate_script(
    python_bin: str, token: str, ordinal: int, digest: str, *,
    privileged_process_inspection: bool = False,
) -> str:
    header = _header(token, ordinal, digest, lock=True,
                     privileged_process_inspection=privileged_process_inspection)
    return build_compensation_script(_pybin(python_bin), header)
