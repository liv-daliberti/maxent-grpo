#!/usr/bin/env python3
"""Retired preparation entry point; use the registered shared controller.

The two construction-only trials are preserved under
artifacts/modebench_level3_neutral_calibration_20260911{,_r2}.
No model sampling was submitted by this entry point. The active controller is
ops/exp_scaling/calibrate_modebench_level3_neutral.py and its registration is
var/artifacts/modebench_level3_neutral_v1/registration.json.
"""
if __name__ == '__main__':
    raise SystemExit('Superseded: use the existing registered neutral calibration controller; do not submit duplicate development.')
