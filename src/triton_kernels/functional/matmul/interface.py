from .block import matmul as block_matmul
from .persistent import matmul as persistent_matmul

from torch import Tensor
from typing import Dict, Callable, Any

MATMUL_INTERFACE: Dict[str, Callable[[Any], Tensor]] = {
    "block": block_matmul,
    "persistent": persistent_matmul,
}


def matmul(kind: str = "persistent", **kwargs):
    return MATMUL_INTERFACE[kind](**kwargs)
