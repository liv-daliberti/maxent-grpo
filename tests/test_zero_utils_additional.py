import maxent_grpo.training.zero_utils as zu


def test_gather_callable_none_when_ds_missing(monkeypatch):
    # When deepspeed helpers are absent, _gather_callable should safely return None.
    monkeypatch.setattr(zu, "ds_zero", None)
    assert zu._gather_callable() is None


def test_call_gather_fn_modifier_rank_type_error():
    calls = {}

    def _gather(params):
        calls["params"] = params
        return "ok"

    # Passing a modifier_rank should fall back to calling without keyword
    # when the gather_fn signature does not accept it.
    result = zu._call_gather_fn(_gather, params=[1, 2, 3], modifier_rank=0)
    assert result == "ok"
    assert calls["params"] == [1, 2, 3]
