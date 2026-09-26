from pathlib import Path
from zipfile import ZipFile
import hashlib,json
p=Path('/tmp/paper-appendix-eighth10-20260921')
stage=p/'verified-source'
stage.mkdir(exist_ok=True)
with ZipFile(p/'before.zip') as z:
    for name in z.namelist():
        if name.startswith('reference/') or name=='PACKAGE_MANIFEST.json':
            continue
        f=stage/name
        f.parent.mkdir(parents=True,exist_ok=True)
        if f.exists(): f.chmod(0o644)
        f.write_bytes(z.read(name))
(stage/'main.tex').write_text(Path('/n/fs/similarity/maxent-grpo/paper/main.tex').read_text())
for f in stage.rglob('*'):
    if f.is_file() and f.suffix not in {'.bbl','.aux','.out','.log','.toc','.blg'}:
        f.chmod(0o444)
(p/'verified-source-manifest.json').write_text(json.dumps({str(f.relative_to(stage)):hashlib.sha256(f.read_bytes()).hexdigest() for f in stage.rglob('*') if f.is_file()},indent=2)+'\n')
print('Staged an immutable incoming archive with the revised manuscript.')
