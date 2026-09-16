#!/usr/bin/env python3
"""Create a standalone source bundle from the validated submission."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from check_submission import ROOT, check_sources, require


def main():
    subprocess.run([sys.executable, str(ROOT / 'check_submission.py')], check=True)
    sources, _ = check_sources(json.loads((ROOT / 'snapshot.json').read_text()))
    receipt = json.loads((ROOT / 'build_receipt.json').read_text())
    require(receipt['source_sha256'] == sources, 'sources changed during packaging')
    expected = dict(sources, **{'main.bbl': receipt['artifact_sha256']['main.bbl']})
    names = set(sources) | {'README.md', 'template_example.tex', 'main.bbl'}
    output = ROOT / 'mathai2026-source.zip'
    with tempfile.TemporaryDirectory(prefix='.mathai-bundle-', dir=ROOT) as temporary:
        staging = Path(temporary) / output.name
        with ZipFile(staging, 'w', ZIP_DEFLATED) as archive:
            for name in sorted(names):
                content = (ROOT / name).read_bytes()
                if name in expected:
                    require(hashlib.sha256(content).hexdigest() == expected[name],
                            'source changed during packaging: ' + name)
                archive.writestr(name, content)
        # Catch a source/artifact update while the archive was being written.
        subprocess.run([sys.executable, str(ROOT / 'check_submission.py')], check=True)
        require(json.loads((ROOT / 'build_receipt.json').read_text()) == receipt,
                'compiled submission changed during packaging; rebuild ZIP')
        staging.replace(output)
    print(f'Created {output.name}: {len(names)} files, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
