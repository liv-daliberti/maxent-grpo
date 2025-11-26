"""
Compare math accuracies between two models using the shared inference pipeline.

Defaults to a stub runner + tiny built-in dataset so CI can exercise the wiring
without network/model downloads. When `--runner transformers` is used, the
standard TransformersPromptRunner from `pipelines.inference.inference` is used.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from maxent_grpo.pipelines.inference.inference import (
    InferenceModelSpec,
    MathEvalConfig,
    run_math_inference,
)


# Fallback stub problems so CI stays offline.
_STUB_PROBLEMS = [
    {"problem": "1+1", "answer": "2"},
    {"problem": "2+2", "answer": "4"},
    {"problem": "3*3", "answer": "9"},
]


@dataclass
class DeltaResult:
    baseline_acc: float
    candidate_acc: float
    delta: float


def _load_dataset(
    path: str | None, prompt_column: str, solution_column: str
) -> List[dict[str, Any]]:
    if path is None:
        return list(_STUB_PROBLEMS)
    dataset_path = Path(path)
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    prompt_column: row.get(prompt_column) or row.get("problem", ""),
                    solution_column: row.get(solution_column) or row.get("answer", ""),
                }
            )
    return rows


def _stub_runner_factory(dataset: Sequence[dict[str, Any]], solution_column: str):
    """Return a runner that echoes gold answers so accuracy is deterministic."""

    class _StubRunner:
        def __init__(self, spec: InferenceModelSpec):
            self.idx = 0
            self.solution_column = solution_column

        def generate(self, problems: List[str]) -> List[str]:
            start = self.idx
            self.idx += len(problems)
            outputs: List[str] = []
            for i in range(start, start + len(problems)):
                gold = str(dataset[i % len(dataset)][self.solution_column])
                outputs.append(f"<think></think><answer>{gold}</answer>")
            return outputs

        def close(self) -> None:
            return None

    return _StubRunner


def evaluate_delta(
    baseline: InferenceModelSpec,
    candidate: InferenceModelSpec,
    *,
    dataset: Iterable[dict[str, Any]] | None,
    eval_cfg: MathEvalConfig,
    runner: str = "stub",
) -> DeltaResult:
    """Run math inference for two specs and return the accuracy delta."""

    rows = list(dataset) if dataset is not None else _STUB_PROBLEMS
    examples = rows[: eval_cfg.limit] if eval_cfg.limit else rows
    runner_factory = (
        _stub_runner_factory(examples, eval_cfg.solution_column)
        if runner == "stub"
        else None
    )
    results = run_math_inference(
        [baseline, candidate],
        eval_cfg=eval_cfg,
        dataset=examples,
        runner_factory=runner_factory,
    )
    baseline_acc = results[0].accuracy
    candidate_acc = results[1].accuracy
    return DeltaResult(
        baseline_acc=baseline_acc,
        candidate_acc=candidate_acc,
        delta=candidate_acc - baseline_acc,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare math accuracies between two models."
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline model name or path."
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate model name or path."
    )
    parser.add_argument(
        "--runner",
        choices=["stub", "transformers"],
        default="stub",
        help="Runner backend.",
    )
    parser.add_argument(
        "--dataset", help="Optional JSONL dataset with problem/answer columns."
    )
    parser.add_argument(
        "--limit", type=int, default=8, help="Limit number of examples evaluated."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for deterministic ordering."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    cfg = MathEvalConfig(limit=args.limit)
    dataset = _load_dataset(args.dataset, cfg.prompt_column, cfg.solution_column)
    specs = [
        InferenceModelSpec(model_name_or_path=args.baseline, label="baseline"),
        InferenceModelSpec(model_name_or_path=args.candidate, label="candidate"),
    ]
    delta = evaluate_delta(
        specs[0],
        specs[1],
        dataset=dataset,
        eval_cfg=cfg,
        runner=args.runner,
    )
    print(
        f"baseline_acc={delta.baseline_acc:.3f} "
        f"candidate_acc={delta.candidate_acc:.3f} "
        f"delta={delta.delta:+.3f}"
    )
    # Exit non-zero if candidate regresses by >0.01 to catch CI drifts.
    if delta.delta < -0.01:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
