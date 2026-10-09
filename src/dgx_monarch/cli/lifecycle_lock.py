"""Per-user locks for Worker lifecycle changes and generation-marker access."""

from __future__ import annotations

import shlex


def lifecycle_lock_source() -> str:
    """Return the shared nofollow implementation of both locks."""
    return r"""
def acquire_dgxm_lock(home,lock_name,nonblocking=False):
 import fcntl,os,pathlib,stat
 if lock_name not in ('lifecycle.lock','worker-generation.lock'): raise RuntimeError('unsafe lock name')
 parent=pathlib.Path(home)
 flags=os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)
 parentfd=os.open(parent,flags); info=os.fstat(parentfd)
 if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid(): os.close(parentfd); raise RuntimeError('unsafe home')
 for name in ('.local','state','dgx-monarch'):
  try: os.mkdir(name,mode=0o700,dir_fd=parentfd)
  except FileExistsError: pass
  childfd=os.open(name,flags,dir_fd=parentfd); info=os.fstat(childfd); os.close(parentfd)
  if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.geteuid() or info.st_mode&0o022: os.close(childfd); raise RuntimeError('unsafe lock parent')
  parentfd=childfd
 try: fd=os.open(lock_name,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600,dir_fd=parentfd)
 except BaseException: os.close(parentfd); raise
 info=os.fstat(fd)
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)&0o077: os.close(fd); os.close(parentfd); raise RuntimeError('unsafe worker lock')
 flags=fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0); fcntl.flock(fd,flags)
 final=os.stat(lock_name,dir_fd=parentfd,follow_symlinks=False); os.close(parentfd)
 if (info.st_dev,info.st_ino)!=(final.st_dev,final.st_ino): os.close(fd); raise RuntimeError('worker lock changed')
 return fd
def acquire_lifecycle_lock(home,nonblocking=False): return acquire_dgxm_lock(home,'lifecycle.lock',nonblocking)
def acquire_generation_lock(home,nonblocking=False): return acquire_dgxm_lock(home,'worker-generation.lock',nonblocking)
"""


_LOCK_CODE = (
    lifecycle_lock_source()
    + r"""
import pathlib,subprocess,sys
home=pathlib.Path.home(); fd=acquire_lifecycle_lock(home)
raise SystemExit(subprocess.run(['/bin/bash','-s'],input=sys.stdin.buffer.read()).returncode)
"""
)

_GENERATION_LOCK_CODE = (
    lifecycle_lock_source()
    + r"""
import pathlib,subprocess,sys
home=pathlib.Path.home(); fd=acquire_lifecycle_lock(home)
try: generation_fd=acquire_generation_lock(home,True)
except (BlockingIOError,OSError,RuntimeError): print('UNKNOWN_GENERATION_FENCE'); raise SystemExit(75)
raise SystemExit(subprocess.run(['/bin/bash','-s'],input=sys.stdin.buffer.read()).returncode)
"""
)

_EXEC_CODE = (
    lifecycle_lock_source()
    + r"""
import os,pathlib,sys
home=pathlib.Path.home(); fd=acquire_lifecycle_lock(home); guard=home/sys.argv[1]
if os.path.lexists(guard) and guard.is_symlink(): raise SystemExit(73)
os.set_inheritable(fd,True); os.execvp(sys.argv[2],sys.argv[2:])
"""
)


def _python(python_bin: str) -> str:
    value = shlex.quote(python_bin)
    return f'PYBIN={value}; case "$PYBIN" in "~/"*) PYBIN="$HOME/${{PYBIN#\\~/}}";; "~") PYBIN="$HOME";; esac'


def locked_script(
    python_bin: str, script: str, *, generation_fence: bool = False
) -> str:
    marker = "DGXM_LIFECYCLE_LOCKED_PAYLOAD"
    if f"\n{marker}\n" in script:
        raise ValueError("invalid lifecycle script payload")
    code = _GENERATION_LOCK_CODE if generation_fence else _LOCK_CODE
    return (
        f"set -eu\n{_python(python_bin)}\n"
        f"\"$PYBIN\" -I -S -B -c {shlex.quote(code)} <<'{marker}'\n{script}\n{marker}"
    )


def locked_rsync_path(python_bin: str, guard_rel: str) -> str:
    if not guard_rel or guard_rel.startswith("/") or any(ord(char) < 32 for char in guard_rel):
        raise ValueError("invalid lifecycle rsync guard")
    fixed = " ".join(shlex.quote(value) for value in (guard_rel, "rsync"))
    return f'{_python(python_bin)}; exec "$PYBIN" -I -S -B -c {shlex.quote(_EXEC_CODE)} {fixed}'
