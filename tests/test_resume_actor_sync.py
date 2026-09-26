import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_SOURCE = ROOT / "src/oat_drgrpo/learner/run.py"


def _run_method() -> ast.FunctionDef:
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    run_mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ZeroMathRunMixin"
    )
    return next(
        node
        for node in run_mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )


def _call_lines(node: ast.AST, attribute: str) -> list[int]:
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attribute
    ]


def test_resume_syncs_restored_weights_before_initial_evaluation_and_rollouts():
    run = _run_method()
    load_line = min(_call_lines(run, "load_ckpt"))
    sync_lines = _call_lines(run, "sync_params_to_actors")
    initial_eval_line = min(_call_lines(run, "eval_and_log"))
    rollout_loop_line = next(
        statement.lineno
        for statement in run.body
        if isinstance(statement, ast.For)
    )

    resume_sync_guards = [
        statement
        for statement in run.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "self.args.resume_dir"
        and _call_lines(statement, "sync_params_to_actors")
    ]

    assert len(resume_sync_guards) == 1
    resume_sync_line = min(_call_lines(resume_sync_guards[0], "sync_params_to_actors"))
    assert sync_lines.count(resume_sync_line) == 1
    assert load_line < resume_sync_line < initial_eval_line < rollout_loop_line
