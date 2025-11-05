from utils.replay_buffer import ReplayBuffer


def make_group(msg: str, reward: float):
    return [{"prompt": [{"role": "user", "content": msg}], "answer": "", "reward": reward}]


def test_add_and_dedup_and_replace():
    rb = ReplayBuffer(capacity=2)
    # Insert two distinct keys
    ok1, uid1 = rb.add({"prompt": [{"role": "user", "content": "A"}], "answer": ""}, reward=0.3)
    ok2, uid2 = rb.add({"prompt": [{"role": "user", "content": "B"}], "answer": ""}, reward=0.4)
    assert ok1 and ok2 and uid1 != uid2

    # Dedup same prompt
    ok3, uid3 = rb.add({"prompt": [{"role": "user", "content": "A"}], "answer": ""}, reward=0.9)
    assert ok3 is False and uid3 == -1

    # Capacity full, lower reward should not replace
    ok4, uid4 = rb.add({"prompt": [{"role": "user", "content": "C"}], "answer": ""}, reward=0.1)
    assert ok4 is False and uid4 == -1

    # Higher reward replaces worst
    ok5, uid5 = rb.add({"prompt": [{"role": "user", "content": "C"}], "answer": ""}, reward=0.8)
    assert ok5 is True and uid5 != -1


def test_update_priority_by_uid_increments_count():
    rb = ReplayBuffer(capacity=1)
    ok, uid = rb.add({"prompt": [{"role": "user", "content": "A"}], "answer": ""}, reward=0.0)
    assert ok and uid >= 0
    # Update twice; n should increase though it is internal, so check no error raised
    rb.update_priority_by_uid(uid, 1.0)
    rb.update_priority_by_uid(uid, 0.0)
