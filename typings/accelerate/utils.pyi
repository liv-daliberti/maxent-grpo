from typing import Any

class DeepSpeedPlugin:
    zero_stage: int | None

    def __init__(self, **kwargs: Any) -> None: ...


def is_peft_model(model: Any) -> bool: ...
