#!/usr/bin/env python3
"""Wait for independent grading sidecars and confirm each correction once."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'artifacts/modebench_discovery_curves_20260911/local'
OUT=LOCAL/'measurement_integrity'
while True:
    available={p.parent.name for p in (LOCAL/'results').glob('*/discovery_grading_audit.json')}
    confirmed={p.stem for p in OUT.glob('qwen*.json')}
    if available-confirmed:
        subprocess.run([sys.executable,'-B',str(ROOT/'ops/confirm_modebench_discovery_grade_corrections.py')],check=True,timeout=180)
        confirmed={p.stem for p in OUT.glob('qwen*.json')}
    if len(confirmed)==25:
        records={p.stem:json.loads(p.read_text()) for p in OUT.glob('qwen*.json')}
        assert all(r['status']=='confirmed' for r in records.values())
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('qwen*.json')}
        summary={'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),'cohorts':25,
            'corrected_targets_independently_confirmed':sum(r['corrections_confirmed'] for r in records.values()),
            'confirmation_receipt_sha256':hashes,'generation_calls':0,'original_data_modified':False,
            'verifier_or_timeouts_modified':False,'cause_attribution':None}
        with (OUT/'COMPLETE.json').open('x') as handle:json.dump(summary,handle,sort_keys=True,indent=2);handle.write('\n')
        print(json.dumps({'event':'all_correction_confirmations_complete','cohorts':25,
            'corrections':summary['corrected_targets_independently_confirmed']}),flush=True)
        break
    time.sleep(45)
