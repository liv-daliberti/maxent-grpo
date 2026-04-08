"""Runtime-only monkey patches for fragile third-party startup probes.

This directory is only prepended for targeted launchers. Keep patches narrow,
silent, and side-effect free so they do not disturb shell wrapper probes that
capture Python stdout.
"""

from __future__ import annotations


def _patch_cpuinfo_for_deepspeed_cpu_adam() -> None:
    try:
        import cpuinfo  # type: ignore
    except Exception:
        return

    original_get_cpu_info = getattr(cpuinfo, "get_cpu_info", None)
    if original_get_cpu_info is None:
        return

    def _safe_get_cpu_info():
        try:
            info = original_get_cpu_info()
            if isinstance(info, dict):
                return info
        except Exception:
            pass
        return {"vendor_id_raw": "unknown"}

    cpuinfo.get_cpu_info = _safe_get_cpu_info

_patch_cpuinfo_for_deepspeed_cpu_adam()
