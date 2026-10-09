"""Python fragments that setup service scripts embed to read and write the worker generation marker."""

from __future__ import annotations

from .lifecycle_generation import GENERATION_FORMAT, GENERATION_MARKER_REL


def generation_reader_source(token: str) -> str:
    """Return marker and lock helpers; the host program first defines ``home`` and ``acquire_generation_lock``."""
    return f"""generation=home/{GENERATION_MARKER_REL!r}; expected_generation={token!r}
generation_format={GENERATION_FORMAT!r}
def acquire_generation_fence():
 try:
  return acquire_generation_lock(home,True)
 except (BlockingIOError,OSError,RuntimeError): return None
def release_generation_fence(fd):
 if fd is None: return
 try: fcntl.flock(fd,fcntl.LOCK_UN)
 finally: os.close(fd)
def current_boot():
 try:
  value=(pathlib.Path('/proc')/'sys/kernel/random/boot_id').read_text(encoding='ascii')
  return value.strip() if value.endswith('\\n') and 1<len(value)<=65 else None
 except (OSError,UnicodeError): return None
def generation_identity():
 try:
  one=generation.lstat()
  if (not stat.S_ISREG(one.st_mode) or stat.S_ISLNK(one.st_mode) or one.st_uid!=os.geteuid()
      or one.st_mode&0o077 or not 1<=one.st_size<=256): return None
  value=generation.read_text(encoding='ascii'); two=generation.lstat(); lines=value.splitlines()
 except (OSError,UnicodeError): return None
 if ((one.st_dev,one.st_ino)!=(two.st_dev,two.st_ino) or not value.endswith('\\n') or len(lines)!=4
     or lines[0]!=generation_format or lines[1]!=expected_generation or lines[2]!=current_boot()): return None
 identity=lines[3].split(':',1)
 return tuple(identity) if len(identity)==2 and all(part.isdigit() for part in identity) else None
"""


def generation_recorder_source(token: str) -> str:
    """Return the reader helpers plus ``record_generation``, which renames a new V1 marker into place."""
    return generation_reader_source(token) + """
def record_generation(fencefd,pid='0',birth='0',invalidate=False):
 boot=current_boot()
 if fencefd is None or boot is None or not pid.isdigit() or not birth.isdigit(): return False
 temporary=generation.parent/f'.worker-generation-{expected_generation}-{os.getpid()}'
 if invalidate:
  try: generation.unlink()
  except FileNotFoundError: pass
  except OSError: return False
 created=False
 try:
  fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600); created=True
  with os.fdopen(fd,'w',encoding='ascii') as out:
   out.write(f'{generation_format}\\n{expected_generation}\\n{boot}\\n{pid}:{birth}\\n'); out.flush(); os.fsync(out.fileno())
  os.replace(temporary,generation)
  parent=os.open(generation.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
  os.fsync(parent); os.close(parent)
 except OSError:
  if created:
   try: temporary.unlink()
   except OSError: pass
  return False
 return generation_identity()==(pid,birth)
"""
