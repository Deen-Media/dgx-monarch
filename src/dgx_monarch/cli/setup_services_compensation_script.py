"""Guided setup's compensation program: stop the worker, then remove what the transaction published."""

from __future__ import annotations


def build_compensation_script(pybin: str, header: str) -> str:
    """Return the standalone compensation script; ``header`` is the activation module's ``_header(lock=True)``."""
    return f"""{pybin}
"$PYBIN" -I -S -B - <<'PY'
{header}
u=unit_state(); live=live_state(); e=enable_state(); fields=manager(); load=loaded(fields); on=enabled()
state=fields.get('ActiveState') if fields is not None else None
if (any(v in ('foreign','unknown') for v in (u,live,e)) or load not in ('owned','absent')
    or state not in ('active','inactive','activating','deactivating','failed') or on is None):
 print('UNKNOWN'); raise SystemExit(0)
stop_intent=txn/'stop-intent.json'
try:
 prior_intent,_=trusted_json(stop_intent)
 identity=prior_intent.get('generation'); parts=identity.split(':',1) if isinstance(identity,str) and len(identity)<=64 else []
 if (prior_intent.get('schema')!=1 or prior_intent.get('token')!=expected_generation
     or prior_intent.get('ordinal')!=reservation.get('ordinal')
     or prior_intent.get('source_manifest')!=reservation.get('source_manifest')
     or prior_intent.get('txn_dev')!=ti.st_dev or prior_intent.get('txn_ino')!=ti.st_ino
     or len(parts)!=2 or not all(part.isdigit() for part in parts)):
  print('UNKNOWN'); raise SystemExit(0)
 intent_identity=tuple(parts)
except FileNotFoundError: prior_intent=None
except (OSError,RuntimeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
marker_present=os.path.lexists(generation); authority=generation_identity() if marker_present else None
if marker_present and authority is None: print('UNKNOWN'); raise SystemExit(0)
if prior_intent is not None:
 if marker_present and authority!=intent_identity: print('UNKNOWN'); raise SystemExit(0)
 authority=intent_identity
elif authority is None: print('UNKNOWN'); raise SystemExit(0)
expected_intent={{'schema':1,'token':expected_generation,'ordinal':reservation.get('ordinal'),
 'source_manifest':reservation.get('source_manifest'),'txn_dev':ti.st_dev,'txn_ino':ti.st_ino,
 'generation':':'.join(authority)}}
if not data.get('start'):
 if state=='active' or fields.get('MainPID')!='0' or authority!=('0','0'):
  print('UNKNOWN'); raise SystemExit(0)
elif state=='active':
 if u!='owned' or load!='owned' or prior_intent is not None: print('UNKNOWN'); raise SystemExit(0)
 try: active_pid=int(fields.get('MainPID','')) if fields else 0; active_birth=birth(active_pid)
 except (TypeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
 if active_birth is None or authority!=(str(active_pid),active_birth): print('UNKNOWN'); raise SystemExit(0)
if prior_intent is None:
 try:
  fd=os.open(stop_intent,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
  with os.fdopen(fd,'w',encoding='ascii') as out:
   json.dump(expected_intent,out,sort_keys=True,separators=(',',':')); out.flush(); os.fsync(out.fileno())
  parent=os.open(txn,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except OSError: print('UNKNOWN'); raise SystemExit(0)
 fields=manager()
 fresh_state=fields.get('ActiveState') if fields is not None else None
 if (fresh_state not in ('active','inactive','activating','deactivating','failed')
     or unit_state()!=u or live_state()!=live or enable_state()!=e or loaded(fields)!=load
     or enabled()!=on or generation_identity()!=authority): print('UNKNOWN'); raise SystemExit(0)
 if fresh_state=='active':
  if not data.get('start'): print('UNKNOWN'); raise SystemExit(0)
  try: active_pid=int(fields.get('MainPID','')); active_birth=birth(active_pid)
  except (TypeError,ValueError): print('UNKNOWN'); raise SystemExit(0)
  if active_birth is None or authority!=(str(active_pid),active_birth): print('UNKNOWN'); raise SystemExit(0)
 run(['systemctl','--user','stop','dgxm-worker.service'],30); deadline=time.monotonic()+25; fields=manager()
 while fields is not None and fields.get('ActiveState') in ('active','activating','deactivating') and time.monotonic()<deadline:
  time.sleep(.25); fields=manager()
 if not inactive_settled(fields): print('UNKNOWN'); raise SystemExit(0)
worker,actors,any_worker,pid=processes(fields)
deadline=time.monotonic()+10
while any_worker is True or actors is True or listener() is True:
 if time.monotonic()>=deadline: print('UNKNOWN'); raise SystemExit(0)
 time.sleep(.25); fields=manager(); worker,actors,any_worker,pid=processes(fields)
if any_worker is not False or actors is not False or listener() is not False: print('UNKNOWN'); raise SystemExit(0)
settled=manager(); settled_again=manager()
if not inactive_settled(settled) or settled_again!=settled: print('UNKNOWN'); raise SystemExit(0)
def remove(path,name,kind):
 if shared(path,name,kind)!='owned': print('UNKNOWN'); raise SystemExit(0)
 quarantine=txn/f'removed-{{name}}'
 if os.path.lexists(quarantine): print('UNKNOWN'); raise SystemExit(0)
 os.rename(path,quarantine); expected=temp(name,kind); observed=quarantine.lstat()
 if expected is None or expected is False or (observed.st_dev,observed.st_ino)!=(expected.st_dev,expected.st_ino):
  if not os.path.lexists(path): os.rename(quarantine,path)
  print('UNKNOWN'); raise SystemExit(0)
 quarantine.unlink(); parent=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
for path,state,name,kind in ((home/'.config/systemd/user/default.target.wants/dgxm-worker.service',e,'enable_tmp','link'),
                             (unit,u,'unit_tmp','file')):
 if state=='owned': remove(path,name,kind)
 elif state!='absent': print('UNKNOWN'); raise SystemExit(0)
run(['systemctl','--user','daemon-reload'])
if loaded(manager())!='absent' or enabled() is not False: print('UNKNOWN'); raise SystemExit(0)
if live=='owned': remove(base/'src','live_tmp','link')
elif live!='absent': print('UNKNOWN'); raise SystemExit(0)
if os.path.lexists(generation):
 if generation_identity()!=authority: print('UNKNOWN'); raise SystemExit(0)
 try: generation.unlink()
 except OSError: print('UNKNOWN'); raise SystemExit(0)
elif prior_intent is None: print('UNKNOWN'); raise SystemExit(0)
try:
 parent=os.open(generation.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
except OSError: print('UNKNOWN'); raise SystemExit(0)
print('COMPENSATED')
PY
"""
