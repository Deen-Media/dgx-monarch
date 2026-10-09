"""The stdlib-only program the remote rsync starts under, so rsync runs holding setup's transaction lock."""

from __future__ import annotations


def locked_exec_source() -> str:
    """Return the program: check identities, lock, check again, then exec argv[6:] with the lock held."""
    return """import fcntl,json,os,pathlib,stat,sys
home=pathlib.Path.home(); txn=home/sys.argv[1]; slot=home/sys.argv[2]
def read(path):
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)); before=os.fstat(fd)
 try: raw=os.read(fd,16385)
 finally: os.close(fd)
 after=path.lstat()
 if (len(raw)>16384 or not stat.S_ISREG(before.st_mode) or before.st_uid!=os.geteuid()
     or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)): raise SystemExit(91)
 return json.loads(raw.decode('ascii'))
ti=txn.lstat(); si=slot.lstat(); site=slot/'site'; sitei=site.lstat(); data=read(txn/'reservation.json')
fd=os.open(txn/'mutation.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0)); li=os.fstat(fd)
if (not stat.S_ISDIR(ti.st_mode) or stat.S_ISLNK(ti.st_mode) or ti.st_uid!=os.geteuid()
    or data.get('token')!=sys.argv[3] or data.get('ordinal')!=int(sys.argv[4])
    or data.get('source_manifest')!=sys.argv[5] or data.get('txn_dev')!=ti.st_dev
    or data.get('txn_ino')!=ti.st_ino or data.get('lock_dev')!=li.st_dev or data.get('lock_ino')!=li.st_ino): raise SystemExit(92)
fcntl.flock(fd,fcntl.LOCK_EX)
ti2=txn.lstat(); again=read(txn/'reservation.json'); sf=slot.lstat(); sitef=site.lstat(); marker=read(slot/'setup.json')
if (again!=data or os.path.lexists(txn/'closed.json') or (ti2.st_dev,ti2.st_ino)!=(ti.st_dev,ti.st_ino)
    or not stat.S_ISDIR(sf.st_mode) or stat.S_ISLNK(sf.st_mode) or sf.st_uid!=os.geteuid()
    or not stat.S_ISDIR(sitef.st_mode) or stat.S_ISLNK(sitef.st_mode) or sitef.st_uid!=os.geteuid()
    or (sf.st_dev,sf.st_ino,sitef.st_dev,sitef.st_ino)!=(si.st_dev,si.st_ino,sitei.st_dev,sitei.st_ino)
    or marker.get('schema')!=1 or marker.get('token')!=sys.argv[3] or marker.get('ordinal')!=int(sys.argv[4])
    or marker.get('source_manifest')!=sys.argv[5] or marker.get('txn_dev')!=ti.st_dev
    or marker.get('txn_ino')!=ti.st_ino or marker.get('slot_dev')!=si.st_dev
    or marker.get('slot_ino')!=si.st_ino or marker.get('site_dev')!=sitei.st_dev
    or marker.get('site_ino')!=sitei.st_ino): raise SystemExit(93)
os.set_inheritable(fd,True); os.execvp(sys.argv[6],sys.argv[6:])
"""
