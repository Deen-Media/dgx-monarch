"""Guided setup's scripts that reserve a private release slot, probe it, and clean it up by recorded inode."""

from __future__ import annotations

import re
import shlex

from .lifecycle_lock import lifecycle_lock_source
from .setup_services_generation import generation_reader_source
from .setup_services_manifest_script import trusted_json_source

_TOKEN = re.compile(r"s-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _validate(python_bin: str, token: str, ordinal: int, digest: str) -> None:
    if not python_bin or any(ord(char) < 32 for char in python_bin):
        raise ValueError("invalid setup service Python")
    if _TOKEN.fullmatch(token) is None:
        raise ValueError("invalid setup service token")
    if type(ordinal) is not int or not 1 <= ordinal <= 64:
        raise ValueError("invalid setup service host ordinal")
    if _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid setup service source digest")


def _pybin(python_bin: str) -> str:
    return (
        f"PYBIN={shlex.quote(python_bin)}\n"
        'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac'
    )


def release_rel(token: str, ordinal: int) -> str:
    _validate("python3", token, ordinal, "0" * 64)
    return f".local/share/dgx-monarch/releases/{token}-{ordinal}"


def build_reserve_script(python_bin: str, token: str, ordinal: int, digest: str) -> str:
    _validate(python_bin, token, ordinal, digest)
    leaf = f"{token}-{ordinal}"
    return f"""set -eu
{_pybin(python_bin)}
"$PYBIN" -I -S -B - {leaf!r} {token!r} {ordinal} {digest!r} <<'PY'
import atexit,fcntl,json,os,pathlib,stat,sys
leaf,token,ordinal,digest=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
home=pathlib.Path.home(); created=[]; owned=[]; committed=False
hi=home.stat()
if home.is_symlink() or not home.is_dir() or hi.st_uid!=os.geteuid() or hi.st_mode&0o022: raise SystemExit(20)
def rollback():
 if committed: return
 for path,dev,ino,kind in reversed(owned):
  try:
   info=path.lstat()
   if (info.st_dev,info.st_ino)!=(dev,ino): continue
   path.rmdir() if kind=='dir' else path.unlink()
  except OSError: pass
atexit.register(rollback)
def ensure(parent,name):
 path=parent/name; fresh=False
 try: path.mkdir(mode=0o700); fresh=True
 except FileExistsError: pass
 info=path.lstat()
 if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid()
     or info.st_mode&0o022): raise SystemExit(21)
 if fresh:
  created.append({{'path':str(path.relative_to(home)),'dev':info.st_dev,'ino':info.st_ino}})
  owned.append((path,info.st_dev,info.st_ino,'dir'))
 return path
local=ensure(home,'.local'); share=ensure(local,'share'); base=ensure(share,'dgx-monarch')
releases=ensure(base,'releases'); config=ensure(home,'.config'); systemd=ensure(config,'systemd')
units=ensure(systemd,'user'); wants=ensure(units,'default.target.wants')
state=ensure(local,'state'); state=ensure(state,'dgx-monarch')
setup=ensure(state,'setup')
txn=setup/leaf
try: txn.mkdir(mode=0o700)
except FileExistsError: raise SystemExit(22)
ti=txn.lstat()
owned.append((txn,ti.st_dev,ti.st_ino,'dir'))
lockfd=os.open(txn/'mutation.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
fcntl.flock(lockfd,fcntl.LOCK_EX)
li=os.fstat(lockfd)
owned.append((txn/'mutation.lock',li.st_dev,li.st_ino,'file'))
reservation={{'schema':1,'token':token,'ordinal':ordinal,'source_manifest':digest,
             'txn_dev':ti.st_dev,'txn_ino':ti.st_ino,'lock_dev':li.st_dev,'lock_ino':li.st_ino,
             'created':created}}
def write(path,payload):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
 info=os.fstat(fd); owned.append((path,info.st_dev,info.st_ino,'file'))
 with os.fdopen(fd,'wb') as out: out.write(payload); out.flush(); os.fsync(out.fileno())
write(txn/'reservation.json',json.dumps(reservation,sort_keys=True,separators=(',',':')).encode('ascii'))
txnfd=os.open(txn,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
try: os.fsync(txnfd)
finally: os.close(txnfd)
slot=releases/leaf
try: slot.mkdir(mode=0o700)
except FileExistsError: raise SystemExit(23)
info=slot.lstat()
owned.append((slot,info.st_dev,info.st_ino,'dir'))
(slot/'site').mkdir(mode=0o700); si=(slot/'site').lstat(); owned.append((slot/'site',si.st_dev,si.st_ino,'dir'))
marker={{'schema':1,'token':token,'ordinal':ordinal,'source_manifest':digest,
        'slot_dev':info.st_dev,'slot_ino':info.st_ino,'txn_dev':ti.st_dev,'txn_ino':ti.st_ino,
        'site_dev':si.st_dev,'site_ino':si.st_ino,'created':created}}
marker_path=slot/'setup.json'
fd=os.open(marker_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
mi=os.fstat(fd); marker['marker_dev']=mi.st_dev; marker['marker_ino']=mi.st_ino
owned.append((marker_path,mi.st_dev,mi.st_ino,'file'))
with os.fdopen(fd,'w',encoding='ascii') as out: json.dump(marker,out,sort_keys=True,separators=(',',':')); out.flush(); os.fsync(out.fileno())
dirfd=os.open(slot,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
try: os.fsync(dirfd)
finally: os.close(dirfd)
write(txn/'slot.json',json.dumps(marker,sort_keys=True,separators=(',',':')).encode('ascii'))
os.close(lockfd)
committed=True
print('RESERVED')
PY
"""


def build_stage_probe_script(python_bin: str, token: str, ordinal: int, digest: str) -> str:
    _validate(python_bin, token, ordinal, digest)
    relative = release_rel(token, ordinal)
    reader = trusted_json_source()
    return f"""{_pybin(python_bin)}
"$PYBIN" -I -S -B - {relative!r} {token!r} {ordinal} {digest!r} <<'PY'
import json,os,pathlib,stat,sys
{reader}
allowed={{'.local','.local/share','.local/share/dgx-monarch','.local/share/dgx-monarch/releases',
 '.config','.config/systemd','.config/systemd/user','.config/systemd/user/default.target.wants','.local/state',
 '.local/state/dgx-monarch','.local/state/dgx-monarch/setup'}}
path=pathlib.Path.home()/sys.argv[1]
if not os.path.lexists(path): print('ABSENT'); raise SystemExit(0)
try:
 info=path.lstat(); marker=path/'setup.json'; data,mi=trusted_json(marker)
 txn=pathlib.Path.home()/f'.local/state/dgx-monarch/setup/{{sys.argv[2]}}-{{sys.argv[3]}}'
 reservation,ri=trusted_json(txn/'reservation.json'); recorded,_=trusted_json(txn/'slot.json'); ti=txn.lstat(); li=(txn/'mutation.lock').lstat()
 exact=(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid==os.geteuid()
        and data.get('schema')==1 and data.get('token')==sys.argv[2]
        and data.get('ordinal')==int(sys.argv[3]) and data.get('source_manifest')==sys.argv[4]
        and data.get('slot_dev')==info.st_dev and data.get('slot_ino')==info.st_ino
        and stat.S_ISREG(mi.st_mode) and not stat.S_ISLNK(mi.st_mode) and mi.st_uid==os.geteuid()
        and data.get('marker_dev')==mi.st_dev and data.get('marker_ino')==mi.st_ino)
 reserved=(stat.S_ISDIR(ti.st_mode) and not stat.S_ISLNK(ti.st_mode) and ti.st_uid==os.geteuid()
           and reservation.get('schema')==1 and reservation.get('token')==sys.argv[2]
           and reservation.get('ordinal')==int(sys.argv[3]) and reservation.get('source_manifest')==sys.argv[4]
           and reservation.get('txn_dev')==ti.st_dev and reservation.get('txn_ino')==ti.st_ino
           and stat.S_ISREG(li.st_mode) and not stat.S_ISLNK(li.st_mode) and li.st_uid==os.geteuid()
           and reservation.get('lock_dev')==li.st_dev and reservation.get('lock_ino')==li.st_ino
           and stat.S_ISREG(ri.st_mode) and not stat.S_ISLNK(ri.st_mode) and ri.st_uid==os.geteuid()
           and data.get('txn_dev')==ti.st_dev and data.get('txn_ino')==ti.st_ino)
 site=(path/'site').lstat(); created=data.get('created')
 complete=(recorded==data and stat.S_ISDIR(site.st_mode) and not stat.S_ISLNK(site.st_mode)
           and site.st_uid==os.geteuid() and data.get('site_dev')==site.st_dev and data.get('site_ino')==site.st_ino
           and isinstance(created,list) and all(isinstance(row,dict) and row.get('path') in allowed
           and type(row.get('dev')) is int and type(row.get('ino')) is int for row in created))
except (OSError,ValueError): print('UNKNOWN'); raise SystemExit(0)
print('OWNED' if exact and reserved and complete else 'FOREIGN')
PY
"""


def build_cleanup_script(python_bin: str, token: str, ordinal: int, digest: str) -> str:
    _validate(python_bin, token, ordinal, digest)
    relative = release_rel(token, ordinal)
    reader = trusted_json_source()
    lifecycle_lock = lifecycle_lock_source()
    generation = generation_reader_source(token)
    return f"""{_pybin(python_bin)}
"$PYBIN" -I -S -B - {relative!r} {token!r} {ordinal} {digest!r} <<'PY'
import fcntl,hashlib,importlib.machinery,json,os,pathlib,shutil,stat,subprocess,sys
{reader}
{lifecycle_lock}
home=pathlib.Path.home(); base=home/'.local/share/dgx-monarch'; unit=home/'.config/systemd/user/dgxm-worker.service'
{generation}
slot=home/sys.argv[1]; txn=home/f'.local/state/dgx-monarch/setup/{{sys.argv[2]}}-{{sys.argv[3]}}'
releases=slot.parent; enable=home/'.config/systemd/user/default.target.wants/dgxm-worker.service'
allowed={{'.local','.local/share','.local/share/dgx-monarch','.local/share/dgx-monarch/releases',
 '.config','.config/systemd','.config/systemd/user','.config/systemd/user/default.target.wants','.local/state',
 '.local/state/dgx-monarch','.local/state/dgx-monarch/setup'}}
def service(prop):
 try: row=subprocess.run(['systemctl','--user','show','dgxm-worker.service',f'--property={{prop}}','--no-pager'],capture_output=True,text=True,timeout=10)
 except (OSError,subprocess.TimeoutExpired): return None
 if row.returncode!=0: return None
 values=dict(line.partition('=')[::2] for line in row.stdout.splitlines() if '=' in line)
 return values.get(prop)
try:
 reservation,ri=trusted_json(txn/'reservation.json'); slot_record,_=trusted_json(txn/'slot.json'); ti=txn.lstat()
 reserved=(stat.S_ISDIR(ti.st_mode) and not stat.S_ISLNK(ti.st_mode) and ti.st_uid==os.geteuid()
           and reservation.get('schema')==1 and reservation.get('token')==sys.argv[2]
           and reservation.get('ordinal')==int(sys.argv[3]) and reservation.get('source_manifest')==sys.argv[4]
           and reservation.get('txn_dev')==ti.st_dev and reservation.get('txn_ino')==ti.st_ino
           and stat.S_ISREG(ri.st_mode) and not stat.S_ISLNK(ri.st_mode) and ri.st_uid==os.geteuid()
           and slot_record.get('token')==sys.argv[2] and slot_record.get('ordinal')==int(sys.argv[3])
           and slot_record.get('source_manifest')==sys.argv[4])
 created=reservation.get('created')
except (OSError,RuntimeError,ValueError): reserved=False; created=None
if not reserved or not isinstance(created,list): print('UNKNOWN'); raise SystemExit(0)
try:
 lockfd=os.open(txn/'mutation.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0)); li=os.fstat(lockfd)
 if (not stat.S_ISREG(li.st_mode) or li.st_uid!=os.geteuid() or li.st_dev!=reservation.get('lock_dev')
     or li.st_ino!=reservation.get('lock_ino')): raise RuntimeError('foreign lock')
 fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check,_=trusted_json(txn/'reservation.json'); tf=txn.lstat()
 if check!=reservation or (tf.st_dev,tf.st_ino)!=(ti.st_dev,ti.st_ino): raise RuntimeError('reservation changed')
except (BlockingIOError,OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
try:
 globalfd=acquire_lifecycle_lock(home,True); generationfd=acquire_generation_fence()
except (BlockingIOError,OSError,RuntimeError): print('UNKNOWN'); raise SystemExit(0)
if generationfd is None: print('UNKNOWN'); raise SystemExit(0)
try: enabled=subprocess.run(['systemctl','--user','is-enabled','dgxm-worker.service'],capture_output=True,text=True,timeout=10)
except (OSError,subprocess.TimeoutExpired): enabled=None
disabled=bool(enabled and ((enabled.returncode==1 and enabled.stdout.strip() in ('disabled','not-found')) or (enabled.returncode==4 and enabled.stdout.strip()=='not-found')))
stop_intent=txn/'stop-intent.json'; intent=None; intent_info=None; intent_identity=None
if os.path.lexists(stop_intent):
 try:
  intent,intent_info=trusted_json(stop_intent); identity=intent.get('generation')
  parts=identity.split(':',1) if isinstance(identity,str) and len(identity)<=64 else []
  if (intent.get('schema')!=1 or intent.get('token')!=sys.argv[2]
      or intent.get('ordinal')!=int(sys.argv[3]) or intent.get('source_manifest')!=sys.argv[4]
      or intent.get('txn_dev')!=ti.st_dev or intent.get('txn_ino')!=ti.st_ino
      or len(parts)!=2 or not all(part.isdigit() for part in parts)):
   raise RuntimeError('foreign stop intent')
  intent_identity=tuple(parts)
 except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
fields={{name:service(name) for name in ('LoadState','ActiveState','SubState','MainPID','Job')}}
fields_again={{name:service(name) for name in ('LoadState','ActiveState','SubState','MainPID','Job')}}
manager_settled=(fields==fields_again and fields.get('LoadState')=='not-found'
                 and fields.get('ActiveState')=='inactive' and fields.get('SubState')=='dead'
                 and fields.get('MainPID')=='0' and fields.get('Job') in ('','0'))
if (os.path.lexists(unit) or os.path.lexists(enable) or os.path.lexists(base/'src') or not disabled
    or not manager_settled):
 print('RETAINED'); raise SystemExit(0)
if os.path.lexists(generation):
 marker_identity=generation_identity()
 if marker_identity is None or (intent_identity is not None and marker_identity!=intent_identity):
  print('UNKNOWN'); raise SystemExit(0)
 if intent_identity is None: print('RETAINED'); raise SystemExit(0)
 try:
  intent_again,intent_again_info=trusted_json(stop_intent)
  if (intent_again!=intent or intent_info is None
      or (intent_again_info.st_dev,intent_again_info.st_ino)!=(intent_info.st_dev,intent_info.st_ino)
      or generation_identity()!=intent_identity): raise RuntimeError('settlement authority changed')
  generation.unlink()
  parent=os.open(generation.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except (OSError,RuntimeError,ValueError): print('RETAINED'); raise SystemExit(0)
else:
 try:
  if intent is None and os.path.lexists(stop_intent): raise RuntimeError('stop intent appeared')
  if intent is not None:
   intent_again,intent_again_info=trusted_json(stop_intent)
   if (intent_again!=intent or intent_info is None
       or (intent_again_info.st_dev,intent_again_info.st_ino)!=(intent_info.st_dev,intent_info.st_ino)):
    raise RuntimeError('settlement authority changed')
  parent=os.open(generation.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
closed={{'schema':1,'token':sys.argv[2],'ordinal':int(sys.argv[3]),'source_manifest':sys.argv[4],
        'txn_dev':ti.st_dev,'txn_ino':ti.st_ino,'lock_dev':li.st_dev,'lock_ino':li.st_ino}}
try:
 prior,_=trusted_json(txn/'closed.json')
 if prior!=closed: print('UNKNOWN'); raise SystemExit(0)
except FileNotFoundError:
 fd=os.open(txn/'closed.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
 with os.fdopen(fd,'w',encoding='ascii') as out: json.dump(closed,out,sort_keys=True,separators=(',',':')); out.flush(); os.fsync(out.fileno())
 parent=os.open(txn,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
archive=txn/'local-slot'; local_quarantine=txn/'.cleanup-local-slot'
release_quarantine=releases/f'.cleanup-{{sys.argv[2]}}-{{sys.argv[3]}}'
present=[path for path in (slot,archive,local_quarantine,release_quarantine) if os.path.lexists(path)]
if len(present)>1: print('UNKNOWN'); raise SystemExit(0)
if present:
 candidate=present[0]; partial=candidate in (local_quarantine,release_quarantine)
 try:
  info=candidate.lstat()
  exact=(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid==os.geteuid()
         and slot_record.get('slot_dev')==info.st_dev and slot_record.get('slot_ino')==info.st_ino
         and slot_record.get('created')==created)
  names={{path.name for path in candidate.iterdir()}}
  if not names<= {{'setup.json','site'}} or (not partial and names!={{'setup.json','site'}}): exact=False
  marker=candidate/'setup.json'
  if os.path.lexists(marker):
   data,mi=trusted_json(marker)
   exact=(exact and data==slot_record and data.get('marker_dev')==mi.st_dev
          and data.get('marker_ino')==mi.st_ino and stat.S_ISREG(mi.st_mode)
          and not stat.S_ISLNK(mi.st_mode) and mi.st_uid==os.geteuid())
  elif not partial: exact=False
  site=candidate/'site'
  if os.path.lexists(site):
   si=site.lstat(); exact=(exact and stat.S_ISDIR(si.st_mode) and not stat.S_ISLNK(si.st_mode)
      and si.st_uid==os.geteuid() and slot_record.get('site_dev')==si.st_dev and slot_record.get('site_ino')==si.st_ino)
  elif not partial: exact=False
 except (OSError,RuntimeError,ValueError): exact=False
 if not exact or not getattr(shutil.rmtree,'avoids_symlink_attacks',False):
  print('UNKNOWN'); raise SystemExit(0)
 # A failed rsync may leave a partial Python tree. Refuse links, foreign owners,
 # and non-source payloads, then quarantine the exact recorded slot inode.
 try:
  suffixes=('.py',*importlib.machinery.EXTENSION_SUFFIXES); package=site/'dgx_monarch'
  if os.path.lexists(site) and os.path.lexists(package):
   for root,dirs,files in os.walk(package,followlinks=False):
    current=pathlib.Path(root); ci=current.lstat()
    if not stat.S_ISDIR(ci.st_mode) or stat.S_ISLNK(ci.st_mode) or ci.st_uid!=os.geteuid(): raise RuntimeError('unsafe source')
    for name in (*dirs,*files):
     child=current/name; child_info=child.lstat()
     if stat.S_ISLNK(child_info.st_mode) or child_info.st_uid!=os.geteuid(): raise RuntimeError('unsafe source')
    if any(not name.endswith(suffixes) for name in files): raise RuntimeError('foreign source entry')
  elif os.path.lexists(site) and any(site.iterdir()): raise RuntimeError('foreign site entry')
  quarantine=local_quarantine if candidate in (archive,local_quarantine) else release_quarantine
  if candidate!=quarantine:
   if os.path.lexists(quarantine): raise RuntimeError('cleanup collision')
   os.rename(candidate,quarantine)
  qi=quarantine.lstat()
  if (qi.st_dev,qi.st_ino)!=(info.st_dev,info.st_ino):
   if candidate!=quarantine and not os.path.lexists(candidate): os.rename(quarantine,candidate)
   raise RuntimeError('slot changed')
  shutil.rmtree(quarantine); parent=os.open(quarantine.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except (OSError,RuntimeError): print('UNKNOWN'); raise SystemExit(0)
# Validate activation-private artifacts before removing the exact transaction.
allowed_names={{'reservation.json','slot.json','mutation.lock','closed.json','verified.json','transaction.json',
               'stop-intent.json','unit.tmp','live.tmp','enable.tmp','removed-unit_tmp','removed-live_tmp','removed-enable_tmp',
               'objects.json'}}
try:
 names={{p.name for p in txn.iterdir()}}
 if not names<=allowed_names: raise RuntimeError('foreign transaction entry')
 journal=None
 if 'transaction.json' in names:
  journal,_=trusted_json(txn/'transaction.json')
  if (journal.get('token')!=sys.argv[2] or journal.get('ordinal')!=int(sys.argv[3])
      or journal.get('source_manifest')!=sys.argv[4] or journal.get('txn_dev')!=ti.st_dev
      or journal.get('txn_ino')!=ti.st_ino): raise RuntimeError('foreign transaction')
 if 'verified.json' in names:
  proof,_=trusted_json(txn/'verified.json')
  if proof.get('token')!=sys.argv[2] or proof.get('source_manifest')!=sys.argv[4]: raise RuntimeError('foreign proof')
 if 'stop-intent.json' in names:
  intent_final,intent_final_info=trusted_json(txn/'stop-intent.json')
  if (intent is None or intent_info is None or intent_final!=intent
      or (intent_final_info.st_dev,intent_final_info.st_ino)!=(intent_info.st_dev,intent_info.st_ino)):
   raise RuntimeError('stop intent changed')
 elif intent is not None: raise RuntimeError('stop intent disappeared')
 if 'objects.json' in names: trusted_json(txn/'objects.json')
 for name,kind,target in (('unit.tmp','file',None),('live.tmp','link',journal.get('link_target') if journal else None),
                          ('enable.tmp','link',journal.get('enable_target') if journal else None)):
  path=txn/name
  if not os.path.lexists(path): continue
  info=path.lstat()
  if journal is None or info.st_uid!=os.geteuid(): raise RuntimeError('unbound private object')
  if kind=='file':
   if not stat.S_ISREG(info.st_mode) or hashlib.sha256(path.read_bytes()).hexdigest()!=journal.get('unit_sha256'): raise RuntimeError('foreign unit temp')
  elif not stat.S_ISLNK(info.st_mode) or os.readlink(path)!=target: raise RuntimeError('foreign link temp')
 for removed,original in (('removed-unit_tmp','unit.tmp'),('removed-live_tmp','live.tmp'),('removed-enable_tmp','enable.tmp')):
  path=txn/removed
  if os.path.lexists(path):
   if not os.path.lexists(txn/original): raise RuntimeError('orphaned removed object')
   one=path.lstat(); two=(txn/original).lstat()
   if (one.st_dev,one.st_ino)!=(two.st_dev,two.st_ino): raise RuntimeError('foreign removed object')
except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
for name in ('objects.json','removed-enable_tmp','removed-live_tmp','removed-unit_tmp','enable.tmp','live.tmp','unit.tmp',
             'stop-intent.json',
             'transaction.json','verified.json','closed.json','slot.json','reservation.json','mutation.lock'):
 try: (txn/name).unlink()
 except FileNotFoundError: pass
try: txn.rmdir()
except OSError: print('RETAINED'); raise SystemExit(0)
state='CLEANED'
for row in reversed(created):
 try:
  if row.get('path') not in allowed: state='UNKNOWN'; break
  if row.get('path') in {{'.local','.local/state','.local/state/dgx-monarch'}}:
   shared=home/'.local/state/dgx-monarch/lifecycle.lock'
   if os.path.lexists(shared):
    lock_info=shared.lstat()
    if (not stat.S_ISREG(lock_info.st_mode) or stat.S_ISLNK(lock_info.st_mode)
        or lock_info.st_uid!=os.geteuid() or lock_info.st_mode&0o077): state='UNKNOWN'; break
    continue
  path=home/row['path']; info=path.lstat()
  if info.st_dev!=row['dev'] or info.st_ino!=row['ino']: state='UNKNOWN'; break
  path.rmdir()
 except FileNotFoundError: continue
 except (KeyError,TypeError): state='UNKNOWN'; break
 except OSError: state='RETAINED'; break
print(state)
PY
"""
