from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile

base=Path(__file__).parent
root=base/'submission'
repo=Path('/n/fs/similarity/maxent-grpo/paper/mathai2026')
for name in ['Makefile','build_submission.py','check_submission.py','package_source.py']:
    shutil.copy2(repo/name,root/name)

def run(name,args,expected=0,cwd=root):
    with (base/(name+'.log')).open('w') as log:
        result=subprocess.run(args,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    assert (result.returncode==0)==(expected==0), (name,result.returncode)
    print(name+': PASS',flush=True)

def hashes():
    names=['main.pdf','main.aux','main.bbl','main.log','main.fls','main.out','main.blg','build_receipt.json']
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}

def bind():
    data=json.loads((root/'snapshot.json').read_text())
    data['workshop_sources']['main.tex']=hashlib.sha256((root/'main.tex').read_bytes()).hexdigest()
    (root/'snapshot.json').write_text(json.dumps(data,indent=2)+'\n')

run('success',['make'])
original={name:(root/name).read_bytes() for name in ['main.tex','snapshot.json']}
validated=hashes()

def restore():
    for name,content in original.items(): (root/name).write_bytes(content)

def preserved():
    assert hashes()==validated,'previous validated artifacts changed after failed build'

(root/'main.tex').write_text(original['main.tex'].decode().replace('\\begin{document}','\\begin{document}\n\\InjectedUndefinedCommand'))
bind()
run('compile-failure',['make'],expected=1)
preserved()
assert (root/'main.failed.log').is_file()
run('stale-receipt-after-failure',['python3','check_submission.py'],expected=1)
restore()

(root/'main.tex').write_text(original['main.tex'].decode().replace('\\label{main:lastpage}','\\clearpage\\mbox{}\\label{main:lastpage}'))
bind()
run('page-limit-failure',['make'],expected=1)
assert 'main text must end on page 4' in (base/'page-limit-failure.log').read_text()
preserved()
restore()

(root/'main.tex').write_bytes(original['main.tex']+b'\n% unbound edit\n')
run('snapshot-failure',['make'],expected=1)
assert 'snapshot source changed: main.tex' in (base/'snapshot-failure.log').read_text()
assert '+ pdflatex' not in (base/'snapshot-failure.log').read_text()
preserved()
restore()

run('restore-success',['make','bundle'])
assert not (root/'main.failed.log').exists()
archive_hash=hashlib.sha256((root/'mathai2026-source.zip').read_bytes()).hexdigest()
(root/'main.tex').write_bytes(original['main.tex']+b'\n% unbound edit\n')
run('stale-bundle-rejection',['python3','package_source.py'],expected=1)
assert hashlib.sha256((root/'mathai2026-source.zip').read_bytes()).hexdigest()==archive_hash
restore()
# Restoring content does not invalidate a hash-bound compilation receipt.
run('restored-content-validation',['python3','check_submission.py'])
(root/'build_receipt.json').unlink()
run('missing-receipt-rebuild',['make'])
run('package-current',['python3','package_source.py'])
extracted=base/'extracted'
if extracted.exists(): shutil.rmtree(extracted)
with zipfile.ZipFile(root/'mathai2026-source.zip') as archive:
    assert 'build_submission.py' in archive.namelist()
    assert 'main.pdf' not in archive.namelist()
    archive.extractall(extracted)
run('standalone-archive-rebuild',['make'],cwd=extracted)
print('All staged-build and standalone-bundle regressions passed.',flush=True)
