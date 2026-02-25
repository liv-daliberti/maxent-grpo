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


def test_maybe_zero_gather_params_skips_available_or_active(monkeypatch):
    gathered: list[list[object]] = []

    class _GatherCtx:
        def __init__(self, params):
            gathered.append(list(params))

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _gather_fn(params, *args, **kwargs):
        del args, kwargs
        return _GatherCtx(params)

    class _Param:
        def __init__(self, *, ds_id, ds_status=None, active=None):
            self.ds_id = ds_id
            self.ds_status = ds_status
            self.ds_active_sub_modules = set() if active is None else set(active)

    active_param = _Param(ds_id=1, ds_status="NOT_AVAILABLE", active={101})
    available_param = _Param(ds_id=2, ds_status="AVAILABLE", active=None)
    pending_param = _Param(ds_id=3, ds_status="NOT_AVAILABLE", active=None)

    class _Model:
        def parameters(self):
            return [active_param, available_param, pending_param]

    monkeypatch.setattr(zu, "_ensure_deepspeed_ready", lambda: True)
    monkeypatch.setattr(zu, "ds_zero", object())
    monkeypatch.setattr(zu, "_is_deepspeed_engine", lambda _m: False)
    monkeypatch.setattr(zu, "_zero_stage", lambda _m: 0)
    monkeypatch.setattr(zu, "_gather_callable", lambda: _gather_fn)

    with zu._maybe_zero_gather_params(_Model(), enabled=True):
        pass

    assert len(gathered) == 1
    assert gathered[0] == [pending_param]


def test_embedding_weights_needing_gather_skips_active_or_available(monkeypatch):
    class _Param:
        def __init__(self, *, ds_status=None, active=None, ndim=1):
            self.ds_status = ds_status
            self.ds_active_sub_modules = set() if active is None else set(active)
            self.ndim = ndim

    input_weight = _Param(ds_status="NOT_AVAILABLE", active={7}, ndim=1)
    output_weight = _Param(ds_status="AVAILABLE", active=None, ndim=1)

    class _Model:
        module = None

        def __init__(self):
            self.module = self
            self.embed_tokens = None
            self.lm_head = type("Head", (), {"weight": output_weight})()

        def get_input_embeddings(self):
            return type("Emb", (), {"weight": input_weight})()

        def get_output_embeddings(self):
            return None

    monkeypatch.setattr(zu, "_is_deepspeed_engine", lambda _m: False)

    assert zu._embedding_weights_needing_gather(_Model()) == []


def test_zero_gather_lock_prevents_nested_active_submodule_assert(monkeypatch):
    class _UncoercibleActive:
        def __bool__(self):
            raise TypeError("cannot coerce active set")

    class _Param:
        def __init__(self):
            self.ds_id = 11
            self.ds_status = "NOT_AVAILABLE"
            self.ds_active_sub_modules = _UncoercibleActive()

    active_by_param: dict[int, set[int]] = {}
    gather_calls: list[list[object]] = []
    token_counter = {"value": 0}

    class _GatherCtx:
        def __init__(self, params):
            self.params = list(params)
            token_counter["value"] += 1
            self.token = token_counter["value"]

        def __enter__(self):
            for param in self.params:
                key = id(param)
                active_by_param.setdefault(key, set()).add(self.token)
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            for param in self.params:
                key = id(param)
                active = active_by_param.get(key, set())
                if self.token in active:
                    active.remove(self.token)
                # Mirror DeepSpeed's free_param assertion: exit must not happen
                # while another gather context still owns the same parameter.
                if active:
                    raise AssertionError({"active_sub_modules": set(active)})
                active_by_param[key] = active
            return False

    def _gather_fn(params, *args, **kwargs):
        del args, kwargs
        gather_calls.append(list(params))
        return _GatherCtx(params)

    class _Model:
        def __init__(self, param):
            self._param = param

        def parameters(self):
            return [self._param]

    param = _Param()
    model = _Model(param)

    monkeypatch.setattr(zu, "_ensure_deepspeed_ready", lambda: True)
    monkeypatch.setattr(zu, "ds_zero", object())
    monkeypatch.setattr(zu, "_is_deepspeed_engine", lambda _m: False)
    monkeypatch.setattr(zu, "_zero_stage", lambda _m: 0)
    monkeypatch.setattr(zu, "_gather_callable", lambda: _gather_fn)

    with zu._maybe_zero_gather_params(model, enabled=True):
        # Nested gather on the same params used to trigger a DeepSpeed-style
        # active_sub_modules assertion at context exit.
        with zu._maybe_zero_gather_params(model, enabled=True):
            pass

    # With the gather lock in place, the nested call is skipped.
    assert len(gather_calls) == 1
