"""Worker generation markers and the lock-protected commands that manage them."""
from __future__ import annotations

import base64
import re
import shlex

from .lifecycle_lock import lifecycle_lock_source
from .systemd_unit import quote as systemd_quote
from .worker_health import _proc_listener_target

_GENERATION = re.compile(r"(?:u-[0-9a-f]{12}-[0-9a-f]{16}|s-[0-9a-f]{16})")
GENERATION_MARKER_REL = ".local/state/dgx-monarch/worker-generation"
GENERATION_LOCK_REL = ".local/state/dgx-monarch/worker-generation.lock"
GENERATION_FORMAT = "DGXM_WORKER_GENERATION_V1"


def validate_generation(value: str) -> str:
    """Accept only an update (``u-``) or setup (``s-``) release-slot token: unguessable and bounded."""
    if not isinstance(value, str) or _GENERATION.fullmatch(value) is None:
        raise ValueError("worker generation is invalid")
    return value


_GENERATION_HELPER = lifecycle_lock_source() + f'''\
import os,pathlib,re,stat,subprocess,sys,time
home=pathlib.Path.home(); marker=home/{GENERATION_MARKER_REL!r}; marker_format={GENERATION_FORMAT!r}
try: fencefd=acquire_generation_lock(home,True)
except (BlockingIOError,OSError,RuntimeError): print('UNKNOWN_GENERATION_FENCE'); raise SystemExit(75)
def remove_marker():
 changed=True
 try: marker.unlink()
 except FileNotFoundError: changed=False
 except OSError: return False
 if not changed: return True
 try:
  parent=os.open(marker.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except FileNotFoundError: pass
 except OSError: return False
 return True
def boot_id():
 try:
  value=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii')
  return value.strip() if value.endswith('\\n') and 1<len(value)<=65 else None
 except (OSError,UnicodeError): return None
def matching(pattern):
 try: entries=tuple(pathlib.Path('/proc').iterdir()); compiled=re.compile(pattern)
 except (OSError,re.error): return None
 found=[]
 for entry in entries:
  if not entry.name.isdigit(): continue
  try:
   if entry.stat().st_uid!=os.geteuid(): continue
   command=b' '.join(value for value in (entry/'cmdline').read_bytes().split(b'\\0') if value).decode('utf-8','replace')
  except FileNotFoundError: continue
  except OSError: return None
  if compiled.search(command): found.append(entry.name)
 return found
def birth(pid):
 try:
  tail=(pathlib.Path('/proc')/pid/'stat').read_text(encoding='ascii').rsplit(') ',1)[1].split()
  return tail[19]
 except (IndexError,OSError,UnicodeError): return None
def owns_listener(pid,proc_table,endpoint):
 try:
  rows=pathlib.Path(proc_table).read_text(encoding='ascii').splitlines()[1:]
  inodes={{parts[9] for row in rows if len(parts:=row.split())>9 and parts[1]==endpoint and parts[3]=='0A'}}
  if not inodes: return False
  owned={{value.removeprefix('socket:[').removesuffix(']') for item in (pathlib.Path('/proc')/pid/'fd').iterdir()
         if (value:=os.readlink(item)).startswith('socket:[')}}
  return bool(inodes&owned)
 except (OSError,UnicodeError): return False
def active_identity(pattern,proc_table,endpoint,expected_pid,expected_birth):
 pids=matching(pattern)
 if pids is None or pids!=[expected_pid] or birth(expected_pid)!=expected_birth: return None
 if not owns_listener(expected_pid,proc_table,endpoint) or birth(expected_pid)!=expected_birth: return None
 return expected_pid,expected_birth
def marker_identity(expected):
 try:
  one=marker.lstat(); value=marker.read_text(encoding='ascii'); two=marker.lstat(); lines=value.splitlines()
 except (OSError,UnicodeError): return None
 if (not stat.S_ISREG(one.st_mode) or stat.S_ISLNK(one.st_mode) or one.st_uid!=os.geteuid()
     or one.st_mode&0o077 or (one.st_dev,one.st_ino)!=(two.st_dev,two.st_ino)
     or not value.endswith('\\n') or len(lines)!=4 or lines[0]!=marker_format
     or lines[1]!=expected or lines[2]!=boot_id()): return None
 identity=lines[3].split(':',1)
 return tuple(identity) if len(identity)==2 and all(part.isdigit() for part in identity) else None
def write_marker(expected,pid,born):
 boot=boot_id()
 if boot is None: return False
 temporary=marker.parent/f'.worker-generation-{{os.getpid()}}-{{time.time_ns()}}'
 try:
  fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
  with os.fdopen(fd,'w',encoding='ascii') as out:
   out.write(f'{{marker_format}}\\n{{expected}}\\n{{boot}}\\n{{pid}}:{{born}}\\n'); out.flush(); os.fsync(out.fileno())
  os.replace(temporary,marker)
  parent=os.open(marker.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)); os.fsync(parent); os.close(parent)
 except OSError:
  try: temporary.unlink()
  except OSError: pass
  return False
 return marker_identity(expected)==(pid,born)
def inactive_unit(unit):
 try:
  row=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--property=SubState','--property=MainPID','--property=Job','--no-pager'],capture_output=True,text=True,timeout=5)
 except (OSError,subprocess.TimeoutExpired): return False
 if row.returncode: return False
 fields={{key:value for line in row.stdout.splitlines() for key,sep,value in (line.partition('='),) if sep}}
 return (fields.get('ActiveState')=='inactive' and fields.get('SubState')=='dead'
         and fields.get('MainPID')=='0' and fields.get('Job') in ('','0'))
def quiesce(mode,pattern,unit,expected):
 try:
  if mode=='systemd':
   stopped=subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
   if stopped.returncode: return False
  killed=subprocess.run(['pkill','-f','--',pattern],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
  if killed.returncode not in (0,1): return False
 except (OSError,subprocess.TimeoutExpired): return False
 time.sleep(.5); pids=matching(pattern)
 if pids!=[]: return False
 if mode=='systemd' and not inactive_unit(unit): return False
 if expected!='-' and not write_marker(expected,'0','0'): return False
 if expected=='-' and not remove_marker(): return False
 time.sleep(.1)
 return matching(pattern)==[] and (mode!='systemd' or inactive_unit(unit))
operation=sys.argv[1]
if operation in ('invalidate','prestart'):
 if not remove_marker(): print('FAILED_GENERATION_INVALIDATION'); raise SystemExit(1)
 raise SystemExit(0)
expected=sys.argv[2]
if operation=='inactive':
 pattern=sys.argv[3]
 if matching(pattern)!=[] or not write_marker(expected,'0','0') or matching(pattern)!=[]:
  remove_marker(); raise SystemExit(1)
 raise SystemExit(0)
if operation=='quarantine':
 expected,pattern,mode,unit=sys.argv[2:6]
 if quiesce(mode,pattern,unit,expected): raise SystemExit(0)
 print('UNKNOWN_START_SETTLEMENT'); raise SystemExit(75)
if operation=='active':
 pattern,proc_table,endpoint,pid,born,mode,unit=sys.argv[3:10]
 identity=active_identity(pattern,proc_table,endpoint,pid,born)
 if identity is None: print('UNKNOWN_START_SETTLEMENT'); raise SystemExit(75)
 if not write_marker(expected,*identity) or active_identity(pattern,proc_table,endpoint,pid,born)!=identity:
  remove_marker()
  if quiesce(mode,pattern,unit,expected): print('FAILED_GENERATION_RECORD'); raise SystemExit(10)
  print('UNKNOWN_START_SETTLEMENT'); raise SystemExit(75)
 raise SystemExit(0)
raise SystemExit(64)
'''
_GENERATION_RUNNER = (
    "import base64;exec(compile(base64.b64decode("
    + repr(base64.b64encode(_GENERATION_HELPER.encode("utf-8")).decode("ascii"))
    + "),'<dgxm-generation>','exec'))"
)
_DURABLE_DIR_RUNNER = (
    "import os,pathlib;path=pathlib.Path.home()/'.local/state/dgx-monarch';"
    "fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0));"
    "os.fsync(fd);os.close(fd)"
)


def _helper_command(*args: str) -> str:
    values = ("/usr/bin/python3", "-I", "-S", "-B", "-c", _GENERATION_RUNNER, *args)
    return " ".join(shlex.quote(value) for value in values)


def generation_prestart_argv() -> tuple[str, ...]:
    """Return an isolated command that clears the prestart marker without waiting for a lock."""
    return ("/usr/bin/python3", "-I", "-S", "-B", "-c", _GENERATION_RUNNER, "prestart")


def generation_quarantine_argv(
    generation: str | None, pattern: str, mode: str, unit_name: str
) -> tuple[str, ...]:
    """Return the failed-start cleanup command as argv; the start script embeds the same call."""
    if mode not in {"systemd", "nohup"}:
        raise ValueError("worker start mode is invalid")
    token = "-" if generation is None else validate_generation(generation)
    return (
        "/usr/bin/python3", "-I", "-S", "-B", "-c", _GENERATION_RUNNER,
        "quarantine", token, pattern, mode, unit_name,
    )


def systemd_generation_invalidator() -> str:
    """Return a shell-free prestart command that clears the marker without waiting for a lock."""
    return "ExecStartPre=" + " ".join(
        systemd_quote(value) for value in generation_prestart_argv()
    )


def invalidate_generation_command_shell() -> str:
    """Clear the generation marker after acquiring its lock without waiting."""
    return _helper_command("invalidate")


def invalidate_generation_shell() -> str:
    """Clear the generation marker while the caller holds its lock."""
    return f'''\
DGXM_GENERATION_FILE="$HOME/{GENERATION_MARKER_REL}"
if ! rm -f -- "$DGXM_GENERATION_FILE"; then
  echo FAILED_GENERATION_INVALIDATION
  exit 1
fi
/usr/bin/python3 -I -S -B -c {shlex.quote(_DURABLE_DIR_RUNNER)} || {{
  echo FAILED_GENERATION_INVALIDATION
  exit 1
}}
'''


def start_generation_shell(
    generation: str | None, pattern: str, address: str, unit_name: str
) -> str:
    """Clear the old marker and define lock-protected recording after readiness."""
    rendered = "" if generation is None else validate_generation(generation)
    proc_table, endpoint = _proc_listener_target(address)
    invalidate = invalidate_generation_command_shell()
    quarantine_generation = rendered or "-"
    if not rendered:
        active = "true"
    else:
        prefix = _helper_command("active", rendered, pattern, proc_table, endpoint)
        active = (
            f'{prefix} "$DGXM_OWNED_LISTENER_PID" "$DGXM_OWNED_LISTENER_START" '
            f'"$DGXM_START_MODE" {shlex.quote(unit_name)}'
        )
    quarantine = _helper_command(
        "quarantine", quarantine_generation, pattern
    ) + f' "$1" {shlex.quote(unit_name)}'
    return f'''\
{invalidate} || {{ DGXM_GENERATION_STATUS=$?; exit "$DGXM_GENERATION_STATUS"; }}
DGXM_GENERATION_FILE="$HOME/{GENERATION_MARKER_REL}"
dgxm_record_generation() {{ {active}; }}
dgxm_quarantine_failed_start() {{ {quarantine}; }}
'''


def require_generation_shell(generation: str, pattern: str) -> str:
    """Refuse changes unless the recorded Worker generation matches under its lock."""
    rendered = shlex.quote(validate_generation(generation))
    quoted_pattern = shlex.quote(pattern)
    return f'''\
DGXM_GENERATION_FILE="$HOME/{GENERATION_MARKER_REL}"
DGXM_EXPECTED_GENERATION={rendered}
DGXM_GENERATION_MATCHED=false
if [ -f "$DGXM_GENERATION_FILE" ] && [ ! -L "$DGXM_GENERATION_FILE" ]; then
  mapfile -t DGXM_GENERATION_LINES < "$DGXM_GENERATION_FILE"
  DGXM_BOOT_ID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)
  DGXM_LOOP_PIDS=$(pgrep -f -- {quoted_pattern} 2>/dev/null || true)
  set -- $DGXM_LOOP_PIDS
  if [ "${{#DGXM_GENERATION_LINES[@]}}" -eq 4 ] && \
      [ "${{DGXM_GENERATION_LINES[0]}}" = {GENERATION_FORMAT!r} ] && \
      [ "${{DGXM_GENERATION_LINES[1]}}" = "$DGXM_EXPECTED_GENERATION" ] && \
      [ -n "$DGXM_BOOT_ID" ] && [ "${{DGXM_GENERATION_LINES[2]}}" = "$DGXM_BOOT_ID" ] && \
      [[ "${{DGXM_GENERATION_LINES[3]}}" =~ ^[0-9]+:[0-9]+$ ]]; then
    DGXM_GENERATION_PID=${{DGXM_GENERATION_LINES[3]%%:*}}
    DGXM_GENERATION_START=${{DGXM_GENERATION_LINES[3]#*:}}
    if [ "$DGXM_GENERATION_PID:$DGXM_GENERATION_START" = 0:0 ] && [ "$#" -eq 0 ]; then
      DGXM_GENERATION_MATCHED=true
    elif [ "$#" -eq 1 ] && [ "$DGXM_GENERATION_PID" = "$DGXM_LOOP_PIDS" ]; then
      DGXM_LOOP_STAT=$(cat "/proc/$DGXM_GENERATION_PID/stat" 2>/dev/null || true)
      DGXM_LOOP_TAIL=${{DGXM_LOOP_STAT##*) }}; set -- $DGXM_LOOP_TAIL
      [ "$#" -ge 20 ] && [ "${{20}}" = "$DGXM_GENERATION_START" ] && DGXM_GENERATION_MATCHED=true
    fi
  fi
fi
if [ "$DGXM_GENERATION_MATCHED" != true ]; then
  echo REFUSED_GENERATION
  exit 75
fi
'''


def settle_generation_shell() -> str:
    """Clear a matched generation after its Worker loop is confirmed stopped."""
    return invalidate_generation_shell()
