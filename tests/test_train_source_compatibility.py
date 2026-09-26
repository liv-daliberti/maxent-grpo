"""Compatibility guards between frozen Python and newer shell entrypoints."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "ops/train.sh"


def test_maxent_objective_flag_is_capability_gated():
    text = TRAIN.read_text(encoding="utf-8")

    assert "ARG_SOURCE_ROOT=" in text
    assert "grep -q 'maxent_objective'" in text
    assert 'cmd+=(--maxent-objective "$MAXENT_OBJECTIVE")' in text
    assert "frozen source predates maxent_objective; omitting inert flag" in text
    assert "Frozen source lacks maxent_objective for an active MaxEnt run" in text
