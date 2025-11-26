from torch import Tensor
import triton
import triton.language as tl

from .utils import validate_inputs_matmul


def matmul_persistent(
    a: Tensor,
    b: Tensor,
) -> Tensor:

    validate_inputs_matmul(a, b)
