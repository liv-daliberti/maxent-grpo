from enum import Enum
from typing import Any

class DistributedType(str, Enum):
    DEEPSPEED = "deepspeed"


class AcceleratorState:
    distributed_type: Any
    deepspeed_plugin: Any

    @classmethod
    def _reset_state(cls, reset_partial_state: bool = ...) -> None: ...
