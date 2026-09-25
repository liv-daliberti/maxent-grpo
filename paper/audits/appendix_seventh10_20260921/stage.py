from pathlib import Path
from zipfile import ZipFile
import shutil,json,hashlib
p=Path('/tmp/paper-appendix-seventh10-20260921');stage=p/'release';stage.mkdir(exist_ok=True)
with ZipFile(p/'before.zip') as z:
 for name in z.namelist():
  if name.startswith('reference/') or name=='PACKAGE_MANIFEST.json':continue
  f=stage/name;f.parent.mkdir(parents=True,exist_ok=True);live=Path('paper')/name
  if live.is_file() and name not in {'README.md','latexmkrc'}:shutil.copy2(live,f)
  else:f.write_bytes(z.read(name))
(p/'release-source-manifest.json').write_text(json.dumps({str(f.relative_to(stage)):hashlib.sha256(f.read_bytes()).hexdigest() for f in stage.rglob('*') if f.is_file()},indent=2)+'\n')
print('Staged current source, fragments, and figures.')
