#!/usr/bin/env python3
"""Unarmed compatibility entrypoint for the versioned E123 runtime validator.

This wrapper changes only which validator the existing controller imports.
It is not authorized for activation until shared-storage integration is reviewed.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e123_level3_qwen3b_factorial_runtime_repaired_20260911 as launch
sys.modules['launch_e123_level3_qwen3b_factorial'] = launch
import control_e123_level3_release as controller

if __name__ == '__main__':
    raise SystemExit(controller.main())
