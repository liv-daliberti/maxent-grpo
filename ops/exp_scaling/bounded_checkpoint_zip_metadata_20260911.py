#!/usr/bin/env python3
"""Process-local, read-only ZIP metadata guard; never deserialize tensor payloads."""
from __future__ import annotations
import os
import zipfile

MAX_READ_BYTES=8*1024**2
MAX_TOTAL_BYTES=16*1024**2
MAX_MEMBERS=65536
OriginalZipFile=zipfile.ZipFile


def identity(stat):return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)


class MetadataReader:
    def __init__(self,path):
        self.path=path;self.stream=open(path,'rb')
        try:self.before=identity(os.fstat(self.stream.fileno()))
        except BaseException:self.stream.close();raise
        self.size=self.before[2];self.total=0
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.tell()+offset if whence==1 else self.size+offset if whence==2 else -1
        if not 0<=target<=self.size:raise zipfile.BadZipFile('metadata seek outside captured file')
        return self.stream.seek(target,0)
    def tell(self):return self.stream.tell()
    def read(self,size=-1):
        remaining=self.size-self.tell()
        # Bound no-size reads against the captured EOF, never a growing EOF.
        requested=remaining if size is None or size<0 else size
        if requested<0 or requested>MAX_READ_BYTES or self.total+requested>MAX_TOTAL_BYTES:
            raise zipfile.BadZipFile('checkpoint ZIP metadata read exceeds bounded allowance')
        if requested>remaining:raise zipfile.BadZipFile('metadata read extends beyond captured EOF')
        data=self.stream.read(requested);self.total+=len(data)
        if len(data)!=requested:raise zipfile.BadZipFile('checkpoint changed during metadata read')
        return data
    def seekable(self):return True
    def stable(self):return identity(os.fstat(self.stream.fileno()))==self.before
    def close(self):self.stream.close()


class BoundedZipFile(OriginalZipFile):
    def __init__(self,file,mode='r',*args,**kwargs):
        self.fp=None
        if mode!='r' or not isinstance(file,(str,bytes,os.PathLike)):
            raise zipfile.BadZipFile('observer supports only path-based read-only ZIP metadata')
        self._validate_on_close=False
        self.metadata_reader=MetadataReader(file)
        try:
            super().__init__(self.metadata_reader,mode,*args,**kwargs)
            if len(self.filelist)>MAX_MEMBERS or not self.metadata_reader.stable():
                raise zipfile.BadZipFile('checkpoint ZIP metadata changed or member count exceeds bound')
            self._validate_on_close=True
        except BaseException:
            self.metadata_reader.close();raise
    def _stable(self):
        if not self.metadata_reader.stable():raise zipfile.BadZipFile('checkpoint changed before metadata credit')
    def infolist(self):self._stable();return super().infolist()
    def namelist(self):self._stable();return super().namelist()
    def open(self,*args,**kwargs):raise zipfile.BadZipFile('tensor/member reads forbidden in metadata observer')
    def close(self):
        reader=getattr(self,'metadata_reader',None)
        check=getattr(self,'_validate_on_close',False) and reader is not None and not reader.stream.closed
        self._validate_on_close=False;error=None
        if check:
            try:
                if not reader.stable():error=zipfile.BadZipFile('checkpoint grew before metadata close')
            except OSError as exc:error=exc
        try:super().close()
        finally:
            if reader is not None:reader.close()
            if check:
                try:
                    if identity(os.stat(reader.path))!=reader.before:error=zipfile.BadZipFile('checkpoint path changed before metadata credit')
                except OSError as exc:error=exc
        if error is not None:raise error


def install():
    # Only this observer process changes; shared on-disk helpers remain immutable.
    zipfile.ZipFile=BoundedZipFile
