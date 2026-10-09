"""Stdlib-only helpers setup scripts embed; ``trusted_manifest`` must equal ``dgx_source_manifest_sha256``."""

from __future__ import annotations


def trusted_manifest_source() -> str:
    """Return code defining ``trusted_manifest`` without importing candidate code."""
    return r"""
def trusted_manifest(root):
 import grp,hashlib,importlib.machinery,os,pathlib,pwd,stat
 def unsafe(info):
  if info.st_uid!=os.geteuid() or info.st_mode&0o002: return True
  if not info.st_mode&0o020: return False
  try:
   current=pwd.getpwuid(os.geteuid()).pw_name; group=grp.getgrgid(info.st_gid)
   users={entry.pw_name for entry in pwd.getpwall() if entry.pw_gid==info.st_gid}|set(group.gr_mem)
  except KeyError: return True
  return bool(users-{current})
 raw=pathlib.Path(root); raw_info=raw.lstat()
 if stat.S_ISLNK(raw_info.st_mode): raise RuntimeError('source root is a symlink')
 root=raw.resolve(strict=True)
 if not root.is_dir() or unsafe(raw_info): raise RuntimeError('unsafe source root')
 excluded={'.git','.venv','__pycache__','venv'}
 suffixes=tuple(importlib.machinery.EXTENSION_SUFFIXES); paths=[]
 def inventory():
  found=[]
  for current,dirs,files in os.walk(root,followlinks=False):
   base=pathlib.Path(current); relative=base.relative_to(root)
   base_info=base.lstat()
   if (not stat.S_ISDIR(base_info.st_mode) or stat.S_ISLNK(base_info.st_mode)
       or unsafe(base_info)): raise RuntimeError('unsafe source directory')
   retained=[]
   for name in sorted(dirs):
    if name in excluded: continue
    candidate=base/name
    if candidate.is_symlink(): raise RuntimeError('included symlink directory')
    retained.append(name)
   dirs[:]=retained
   for name in sorted(files):
    parts=(*relative.parts,name)
    if any(part in excluded for part in parts[:-1]): continue
    if name.endswith('.pyc'): raise RuntimeError('direct pyc')
    if not (name.endswith('.py') or name.endswith(suffixes)): continue
    candidate=base/name; info=candidate.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
        or unsafe(info)): raise RuntimeError('unsafe source file')
    found.append(candidate)
  return sorted(found,key=lambda path:path.relative_to(root).as_posix())
 paths=inventory()
 if not paths: raise RuntimeError('empty source manifest')
 digest=hashlib.sha256(); digest.update(b'DGXM_RUNTIME_SOURCE_MANIFEST_V1\0')
 scope=b'dgx_monarch'; digest.update(len(scope).to_bytes(2,'big')); digest.update(scope)
 digest.update(len(paths).to_bytes(8,'big'))
 for path in paths:
  relative=path.relative_to(root).as_posix().encode(); digest.update(len(relative).to_bytes(4,'big')); digest.update(relative)
  fd=os.open(path,os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0))
  try:
   before=os.fstat(fd)
   if not stat.S_ISREG(before.st_mode) or unsafe(before): raise RuntimeError('unsafe source file')
   digest.update(int(before.st_size).to_bytes(8,'big'))
   while True:
    chunk=os.read(fd,1024*1024)
    if not chunk: break
    digest.update(chunk)
   after=os.fstat(fd)
  finally: os.close(fd)
  final=path.lstat()
  identity=lambda item:(int(item.st_dev),int(item.st_ino),int(item.st_size),int(item.st_mtime_ns),int(item.st_ctime_ns))
  if identity(before)!=identity(after) or identity(after)!=identity(final): raise RuntimeError('source changed')
 if paths!=inventory(): raise RuntimeError('source set changed')
 return digest.hexdigest()
"""


def trusted_json_source() -> str:
    """Return code defining a nofollow, stable, owner-only JSON reader."""
    return r"""
def trusted_json(path):
 import json,os,stat
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0))
 try:
  before=os.fstat(fd); chunks=[]
  if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.geteuid(): raise RuntimeError('unsafe journal')
  while True:
   chunk=os.read(fd,65536)
   if not chunk: break
   chunks.append(chunk)
  after=os.fstat(fd)
 finally: os.close(fd)
 final=path.lstat(); identity=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
 if identity(before)!=identity(after) or identity(after)!=identity(final): raise RuntimeError('journal changed')
 return json.loads(b''.join(chunks).decode('ascii')),before
"""
