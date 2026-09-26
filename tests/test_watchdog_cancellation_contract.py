from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_administrative_stop_cannot_trigger_watchdog_requeue():
    trainer = (ROOT / "ops/train.sh").read_text(encoding="utf-8")

    assert "shutdown_requested=0" in trainer
    assert "request_shutdown()" in trainer
    assert "shutdown_requested=1" in trainer
    assert "trap request_shutdown TERM INT" in trainer
    assert "&& (( shutdown_requested == 0 ))" in trainer
    assert "administrative shutdown requested; requeue suppressed" in trainer
