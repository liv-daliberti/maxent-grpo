#!/usr/bin/env python3
"""Seeded-launcher v2 controls, including standard site-builtin compatibility.

These CPU controls and historical source regression checks are diagnostic only;
they do not replace the prospective complete reference audit or model evaluation.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import probe_constructive_code_seeded_20260922 as common

SOURCE = common.ROOT / "ops/constructive_code_sandbox_seeded_20260922_v2.c"
FAILED_SOURCE_HASHES = {
    "04a811bb6e8c656897e258f292312ddd9f4a2723b90f2a3c34cab6a8a4460f10",
    "1268d64870b2d0f7ece14ba1595e330e872e3331ea1e7b49da8543dc0ae168fb",
    "1e8de77c8c97a7a1121fe83215623c9c49ffceba544297f49806b0bcc15753fa",
    "214c77ec7d75062a18dc10bef29789a1aabf2ecedaf38d39887b24953f7385c1",
}


def run_compatibility_controls(probe: common.SandboxProbe) -> dict[str, Any]:
    observations = {}
    exit_rows = []
    for expression in ("exit()", "exit(0)", "quit()", "quit(0)", "sys.exit()", "sys.exit(0)", "exit(7)", "quit(7)", "sys.exit(7)"):
        row = probe.run("import sys\nprint('before')\n" + expression + "\nprint('after')\n", hostile=True)
        expected = 7 if "7" in expression else 0
        assert row["returncode"] == expected and row["stdout"] == "before\n" and row["stderr"] == "", row
        exit_rows.append({"expression": expression, **row})
    observations["standard_exit_semantics"] = exit_rows
    row = probe.run("import sys\ntry: exit()\nexcept SystemExit as e: print(repr(e.code), sys.stdin.closed)\n")
    assert row["returncode"] == 0 and row["stdout"] == "None True\n", row
    observations["quitter_closes_stdin"] = row
    value, _ = probe.value("import sys,site,builtins,json\nprint(json.dumps({'types':{name:type(getattr(builtins,name)).__module__+'.'+type(getattr(builtins,name)).__name__ for name in ['quit','exit','copyright','credits','license','help']},'site':site.__file__,'user_site':site.USER_SITE,'user_base':site.USER_BASE,'enable_user_site':site.ENABLE_USER_SITE,'customization_modules':[name for name in ['sitecustomize','usercustomize'] if name in sys.modules]}))\n", hostile=True)
    assert value["types"] == {"quit":"_sitebuiltins.Quitter", "exit":"_sitebuiltins.Quitter", "copyright":"_sitebuiltins._Printer", "credits":"_sitebuiltins._Printer", "license":"_sitebuiltins._Printer", "help":"_sitebuiltins._Helper"}, value
    assert value["site"] == str(probe.runtime_root / "usr/local/lib/python3.10/site.py"), value
    assert all(value[key] is None for key in ("user_site", "user_base", "enable_user_site")), value
    assert value["customization_modules"] == [], value
    observations["standard_builtins_without_site_main"] = value
    with tempfile.TemporaryDirectory(prefix="seeded-site-shadow-") as raw:
        work = Path(raw)
        for name in ("site.py", "_sitebuiltins.py", "sitecustomize.py", "usercustomize.py"):
            (work / name).write_text("raise RuntimeError('UNTRUSTED STARTUP MODULE')\n")
        (work / "probe.pth").write_text("import sys; raise RuntimeError('UNTRUSTED PTH')\n")
        (work / "candidate.py").write_text("print('isolated'); exit(0)\n")
        done = subprocess.run([str(probe.launcher), str(probe.runtime_root), str(work), "candidate.py", "3", "536870912", "1048576", "64"], text=True, capture_output=True, timeout=10, env={**os.environ, "PYTHONPATH":str(work), "PYTHONUSERBASE":str(work)})
        assert done.returncode == 0 and done.stdout == "isolated\n" and not done.stderr, done
        observations["site_helpers_cannot_load_cwd_shadow"] = {"returncode":done.returncode,"stdout":done.stdout,"stderr":done.stderr}
    return observations


def source_regression(probe: common.SandboxProbe, original_probe: common.SandboxProbe, task_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (task_dir / "py3_replays.jsonl").read_text().splitlines()]
    selected = [r for r in rows if r["submission_sha256"] in FAILED_SOURCE_HASHES]
    assert len(selected) == 4 and all(r["known_label"] == "correct" for r in selected)
    with gzip.open(task_dir / "hardened_inputs.jsonl.gz", "rt") as handle:
        inputs = [json.loads(line) for line in handle]
    assert len(inputs) == 20
    results = []
    for row in selected:
        import hashlib
        assert hashlib.sha256(row["code"].encode()).hexdigest() == row["submission_sha256"]
        checks = []
        for case in inputs:
            # Dataset serialization preserves escaped newlines in the stdin field.
            stdin = case["stdin"].replace("\\n", "\n")
            old = original_probe.run(row["code"], stdin=stdin)
            new = probe.run(row["code"], stdin=stdin)
            assert old["returncode"] == new["returncode"] == 0, {"source": row["submission_sha256"], "test": case["test_index"], "old": old, "new": new}
            assert old["stdout"] == new["stdout"] and old["stderr"] == new["stderr"] == "", (old, new)
            checks.append({"test_index":case["test_index"], "original_returncode":old["returncode"], "v2_returncode":new["returncode"], "stdout":new["stdout"]})
        results.append({"submission_sha256":row["submission_sha256"],"tests":checks})
    return {"purpose":"exact original/v2 execution compatibility; not checker admission evidence", "task":"1016_D", "source_file_sha256":common.sha256(task_dir / "py3_replays.jsonl"), "input_file_sha256":common.sha256(task_dir / "hardened_inputs.jsonl.gz"), "programs":results}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--regression-task-dir',type=Path,required=True)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--original-source',type=Path,default=common.ORIGINAL)
    args=p.parse_args()
    common.check_kernel_source_unchanged(args.source,args.original_source)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    binary=args.output_dir/'seeded_v2'
    original_binary=args.output_dir/'original'
    for source,target in ((args.source,binary),(args.original_source,original_binary)):
        subprocess.run(['cc','-O2','-Wall','-Wextra','-Werror','-o',str(target),str(source)],check=True,capture_output=True,text=True)
    receipt={"schema":"constructive-code-seeded-launcher-controls-20260922-v2", "purpose":"CPU controls only; no pilot results", "source_sha256":common.sha256(args.source), "original_source_sha256":common.sha256(args.original_source), "probe_sha256":common.sha256(Path(__file__)), "common_probe_sha256":common.sha256(Path(common.__file__)), "runtime_root":str(args.runtime_root.resolve()), "binary_sha256":common.sha256(binary)}
    try:
        probe=common.SandboxProbe(binary,args.runtime_root)
        receipt['base_controls']=common.run_controls(probe)
        receipt['compatibility_controls']=run_compatibility_controls(probe)
        receipt['source_regression']=source_regression(probe,common.SandboxProbe(original_binary,args.runtime_root),args.regression_task_dir)
        receipt['status']='pass'
    except Exception as exc:
        receipt['status']='fail';receipt['error']=f'{type(exc).__name__}: {exc}'
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':receipt['status'],'receipt':str(args.output_dir/'receipt.json')}))
    return 0 if receipt['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
