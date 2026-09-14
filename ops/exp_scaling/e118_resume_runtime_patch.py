#!/usr/bin/env python3
"""Minimal, auditable source repair for E118's missing checkpoint step helper."""
from __future__ import annotations

import ast
import textwrap


METHOD_SOURCE = textwrap.dedent('''\
    def _infer_resume_step(self, resume_states: dict[str, Any] | None) -> int:
        """Recover the loaded checkpoint's step without restarting its counters."""
        saved_step = None
        if isinstance(resume_states, dict) and "steps" in resume_states:
            raw_step = resume_states["steps"]
            try:
                saved_step = int(raw_step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Checkpoint has invalid client-state steps") from exc
            if isinstance(raw_step, bool) or saved_step < 0 or str(raw_step) not in (
                str(saved_step), str(float(saved_step))
            ):
                raise RuntimeError("Checkpoint has invalid client-state steps")

        tag = getattr(self.args, "resume_tag", None)
        if not tag and getattr(self.args, "resume_dir", None):
            latest = Path(self.args.resume_dir) / "latest"
            if latest.is_symlink():
                tag = latest.resolve().name
            elif latest.is_file():
                tag = latest.read_text(encoding="utf-8").strip()
        tag_step = None
        if tag:
            tag_name = Path(str(tag)).name
            if tag_name.startswith("step_") and tag_name[5:].isdigit():
                tag_step = int(tag_name[5:])
        if saved_step is not None:
            if tag_step is not None and tag_step != saved_step:
                raise RuntimeError(
                    f"Checkpoint step mismatch: client state={saved_step}, tag={tag_step}"
                )
            return saved_step
        if tag_step is not None:
            return tag_step
        raise RuntimeError("Cannot infer resumed checkpoint step from client state or tag")

''')


def patched_source(source: str) -> str:
    """Insert only the missing method; fail if surrounding source has drifted."""
    tree = ast.parse(source)
    mixin = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "ZeroMathRunMixin")
    methods = {node.name for node in mixin.body if isinstance(node, ast.FunctionDef)}
    if "_infer_resume_step" in methods:
        raise ValueError("Source already defines _infer_resume_step")
    anchor = "    def _restore_prompt_progress(\n"
    if source.count(anchor) != 1 or "_restore_prompt_progress" not in methods:
        raise ValueError("Unexpected checkpoint resume source layout")
    result = source.replace(anchor, textwrap.indent(METHOD_SOURCE, "    ") + anchor, 1)
    ast.parse(result)
    return result
