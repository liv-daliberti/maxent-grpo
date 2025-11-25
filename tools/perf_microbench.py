"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tiny CPU-only microbenchmark to catch obvious performance regressions.

Current coverage:
- logging_metrics_ops_per_sec: iterations of build_training_metrics_dict
  using a representative TrainingMetricsPayload.

Baseline values live in tools/perf_baseline.json. The CLI compares the current
measurement to the baseline and fails if it regresses beyond the tolerance.
"""

from __future__ import annotations

import argparse
import json
import time
import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple


DEFAULT_BASELINE = Path(__file__).parent / "perf_baseline.json"
REPO_ROOT = DEFAULT_BASELINE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_lightweight_stubs() -> None:
    """Stub heavy optional deps so the microbench can run on bare runners."""

    accel_loaded = "accelerate" in sys.modules
    accel_spec = None
    if not accel_loaded:
        try:
            accel_spec = importlib.util.find_spec("accelerate")
        except ValueError:
            accel_spec = None
    if not accel_loaded and accel_spec is None:
        accel_mod = ModuleType("accelerate")

        class _AccelState(ModuleType):
            DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")

        class _Accelerator:
            def __init__(self, *args, **kwargs):
                self.device = "cpu"
                self.is_main_process = True
                self.num_processes = 1
                self.process_index = 0
                self.gradient_accumulation_steps = 1
                self.sync_gradients = True

            def gather(self, obj):
                return obj

            def gather_object(self, obj):
                return [obj]

            def log(self, metrics, step=None):
                return None

            def wait_for_everyone(self):
                return None

            @contextmanager
            def accumulate(self, _model):
                yield

            def backward(self, _loss):
                return None

            def clip_grad_norm_(self, *_args, **_kwargs):
                return 0.0

            def unwrap_model(self, model):
                return model

            def save_state(self, _path):
                return None

            def load_state(self, _path):
                return None

        accel_mod.Accelerator = _Accelerator
        accel_state = _AccelState("accelerate.state")
        accel_mod.state = accel_state
        sys.modules["accelerate"] = accel_mod
        sys.modules["accelerate.state"] = accel_state

    tf_loaded = "transformers" in sys.modules
    tf_spec = None
    if not tf_loaded:
        try:
            tf_spec = importlib.util.find_spec("transformers")
        except ValueError:
            tf_spec = None
    if not tf_loaded and tf_spec is None:
        tf_mod = ModuleType("transformers")
        tf_mod.__spec__ = SimpleNamespace()
        tf_mod.PreTrainedModel = type("PreTrainedModel", (), {})
        tf_mod.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
        sys.modules["transformers"] = tf_mod


def _build_logging_payload():
    """Construct a representative TrainingMetricsPayload for benchmarking."""
    _install_lightweight_stubs()
    from maxent_grpo.training.types import (
        BatchDiagnostics,
        LengthStats,
        LoggingConfigView,
        RewardComponentStats,
        RewardLoggingView,
        TrainingMetricsPayload,
        TrainingScalarStats,
        TokenUsageStats,
        WeightLoggingView,
    )

    reward_stats = RewardLoggingView(
        reward_mean=1.0,
        reward_std=0.5,
        frac_zero_std=0.0,
        advantage_mean=0.1,
        advantage_std=0.01,
        advantage_count=8,
        per_reward={
            "accuracy": RewardComponentStats(mean=0.2, std=0.05),
            "format": RewardComponentStats(mean=-0.1, std=0.02),
        },
    )
    weight_stats = WeightLoggingView(
        entropy=0.3,
        entropy_min=0.1,
        entropy_max=0.5,
        advantage_entropy_mean=0.04,
        advantage_entropy_std=0.01,
    )
    payload = TrainingMetricsPayload(
        reward_stats=reward_stats,
        weight_stats=weight_stats,
        loss_outputs=SimpleNamespace(
            total_loss_scalar=0.4,
            kl_loss_scalar=0.2,
            weighted_kl_loss_scalar=0.25,
            clip_loss_scalar=None,
        ),
        diagnostics=BatchDiagnostics(
            kl_value=0.2,
            clip_ratio=0.1,
            clip_ratio_low_mean=0.02,
            clip_ratio_low_min=0.01,
            clip_ratio_high_mean=0.15,
            clip_ratio_high_max=0.2,
            clip_ratio_region_mean=0.05,
        ),
        length_stats=LengthStats(
            min_length=5.0,
            mean_length=10.0,
            max_length=20.0,
            clipped_ratio=0.0,
            min_terminated=5.0,
            mean_terminated=10.0,
            max_terminated=20.0,
        ),
        config=LoggingConfigView(
            weighting=SimpleNamespace(beta=0.5, tau=0.2),
            clipping=SimpleNamespace(
                clip_range=0.1,
                clip_adv_baseline=None,
                clip_objective_coef=1.0,
            ),
            schedule=SimpleNamespace(num_generations=4),
        ),
        scalars=TrainingScalarStats(
            ref_logp_mean=-1.0,
            tokens=TokenUsageStats(
                avg_completion_tokens=12.0,
                num_completion_tokens=96.0,
                num_input_tokens=128.0,
            ),
            current_lr=1e-4,
            grad_norm_scalar=0.5,
            epoch_progress=1.5,
            vllm_latency_ms=25.0,
        ),
    )
    return payload


def run_benchmarks(iterations: int) -> Dict[str, float]:
    """Run microbenchmarks and return metrics."""
    _install_lightweight_stubs()
    from maxent_grpo.training.metrics import build_training_metrics_dict

    payload = _build_logging_payload()
    start = time.perf_counter()
    for _ in range(iterations):
        build_training_metrics_dict(payload, global_step=10)
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / max(elapsed, 1e-9)
    return {"logging_metrics_ops_per_sec": ops_per_sec}


def compare_to_baseline(
    metrics: Dict[str, float],
    baseline: Dict[str, float],
    tolerance: float,
) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    """Compare current metrics to baseline, allowing a fractional regression."""
    regressions = {}
    for key, current in metrics.items():
        expected = baseline.get(key)
        if expected is None:
            continue
        if current < expected * (1 - tolerance):
            regressions[key] = (current, expected)
    return (len(regressions) == 0, regressions)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Iterations per benchmark."
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to baseline JSON file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.35,
        help="Allowed fractional regression (0.35 = allow 35%% slower).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current metrics to the baseline file and exit.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write measured metrics as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    metrics = run_benchmarks(args.iterations)
    if args.json_output:
        args.json_output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    if args.update_baseline:
        args.baseline_file.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Baseline updated at {args.baseline_file}")
        return 0

    if not args.baseline_file.exists():
        print(f"Baseline file missing: {args.baseline_file}")
        return 1

    baseline = json.loads(args.baseline_file.read_text())
    ok, regressions = compare_to_baseline(metrics, baseline, args.tolerance)
    print(f"Measured metrics: {json.dumps(metrics, indent=2)}")
    if ok:
        print("Performance within tolerance.")
        return 0
    print("Performance regression detected:")
    for key, (current, expected) in regressions.items():
        drop = (expected - current) / expected if expected else 1.0
        print(
            f"- {key}: current={current:.2f}, expected={expected:.2f}, drop={drop:.2%}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
