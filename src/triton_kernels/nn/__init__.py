from typing import Dict, List
from torch import nn

from .activation.gelu import GELU
from .activation.silu import SiLU

from .softmax import Softmax

KERNELS: List[nn.Module] = [
    GELU,
    Softmax,
    SiLU,
]

__all__ = [x.__name__ for x in KERNELS]
