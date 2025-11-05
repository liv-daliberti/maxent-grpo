from src.utils.replay_dataset import ReplayMixDataset, replay_collate


def test_replay_collate_marks_not_replay_and_keeps_accuracy():
    batch = [
        {"prompt": [], "answer": "", "accuracy": 1.0, "is_replay": 1},
        {"prompt": [], "answer": "", "accuracy": 0.0},
    ]
    out = replay_collate(batch, replay_buffer=None, replay_prob=0.0)
    assert all(ex["is_replay"] == 0 for ex in out)
    assert "accuracy" in out[0] and "accuracy" in out[1]


def test_replay_mix_dataset_sets_is_replay_zero():
    base = [{"prompt": [], "answer": ""}]
    ds = ReplayMixDataset(base, tok=type("T", (), {"decode": lambda self, x: ""})())
    item = ds[0]
    assert item["is_replay"] == 0

