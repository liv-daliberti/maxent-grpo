#!/usr/bin/env python3
"""Compile and validate in isolation before replacing the submission PDF."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from check_submission import BUILD_ARTIFACTS, ROOT, check_sources, require


def build(latex):
    # Reject stale snapshots before running TeX or touching the last good build.
    sources, _ = check_sources(json.loads((ROOT / 'snapshot.json').read_text()))
    with tempfile.TemporaryDirectory(prefix='.mathai-build-', dir=ROOT) as temporary:
        staging = Path(temporary)
        for name in sources:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        try:
            commands = [latex + ['main.tex'], ['bibtex', 'main'],
                        latex + ['main.tex'], latex + ['main.tex'],
                        [sys.executable, 'check_submission.py', '--record-build']]
            for command in commands:
                print('+ ' + shlex.join(command), flush=True)
                subprocess.run(command, cwd=staging, check=True)
            current, _ = check_sources(json.loads((ROOT / 'snapshot.json').read_text()))
            receipt = json.loads((staging / 'build_receipt.json').read_text())
            require(current == sources == receipt['source_sha256'],
                    'sources changed during compilation; rebuild with make')
        except (subprocess.CalledProcessError, OSError, SystemExit):
            failed_log = staging / 'main.log'
            if failed_log.is_file():
                shutil.copy2(failed_log, ROOT / 'main.failed.log')
                print('Build rejected; compiler log saved to main.failed.log.', file=sys.stderr)
            print('Previous submission PDF and compilation receipt preserved.', file=sys.stderr)
            raise
        # Each rename is atomic. Publish the PDF after its auxiliaries and the
        # receipt last; interruption during promotion is detected by the checker.
        outputs = [name for name in BUILD_ARTIFACTS if name != 'main.pdf']
        outputs += ['main.out', 'main.blg', 'main.pdf', 'build_receipt.json']
        for name in outputs:
            (staging / name).replace(ROOT / name)
    (ROOT / 'main.failed.log').unlink(missing_ok=True)
    print('Published validated main.pdf and compilation receipt.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latex', default='pdflatex -interaction=nonstopmode -halt-on-error -recorder',
                        help='LaTeX command and options (default: pdfLaTeX)')
    args = parser.parse_args()
    latex = shlex.split(args.latex)
    require(bool(latex), 'empty LaTeX command')
    try:
        build(latex)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except OSError as error:
        raise SystemExit('FAIL: ' + str(error)) from None


if __name__ == '__main__':
    main()
