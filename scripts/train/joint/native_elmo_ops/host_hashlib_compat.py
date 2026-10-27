"""Narrow Python3.10 host adapter for maintained SHA256 file_digest calls.

Only regular blocking binary files and the literal 'sha256' contract are
supported. Existing native hashlib.file_digest is never replaced. Scientific
source bytes remain untouched; this is installed only in the host process.
"""
import hashlib,os,stat

def file_digest(fileobj,digest,/,*,_bufsize=1<<18):
    if digest!='sha256':raise ValueError('Host compatibility supports only the maintained sha256 contract')
    if type(_bufsize) is not int or not 0<_bufsize<=8<<20:raise ValueError('Invalid bounded digest buffer')
    if not callable(getattr(fileobj,'readinto',None)) or not fileobj.readable():
        raise ValueError('Readable regular binary file required')
    try:fd=fileobj.fileno();before=os.fstat(fd)
    except (OSError,ValueError,AttributeError) as error:raise ValueError('Real regular file descriptor required') from error
    if not stat.S_ISREG(before.st_mode) or not os.get_blocking(fd):raise ValueError('Regular blocking file required')
    value=hashlib.sha256();buffer=bytearray(_bufsize);view=memoryview(buffer)
    while True:
        count=fileobj.readinto(buffer)
        if count is None:raise BlockingIOError('Digest read returned no data without EOF')
        if type(count) is not int or not 0<=count<=len(buffer):raise ValueError('Invalid readinto count')
        if count==0:break
        value.update(view[:count])
    after=os.fstat(fd)
    signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if signature(before)!=signature(after):raise ValueError('Regular file changed during digest')
    return value

def install():
    if hasattr(hashlib,'file_digest'):
        return {'status':'NATIVE_FILE_DIGEST_PRESERVED','sha256_only_fallback_installed':False}
    hashlib.file_digest=file_digest
    return {'status':'PYTHON310_REGULAR_BLOCKING_SHA256_ADAPTER_INSTALLED','sha256_only_fallback_installed':True}
