

import triton
from torch import Tensor
import torch



@triton.jit()
def _rmsprop_kernel():
    ...

def rmsprop(
        params: list[Tensor],
        grads: list[Tensor],
        square_avgs: list[Tensor],
        state_steps: list[Tensor], 
        lr: float, 
        eps: float, 
        alpha: float, 
    ):

    for i, param in enumerate(params):
        step = state_steps[i]
        grad = grads[i]
        square_avg = square_avgs[i]

        grid = ..., 
        _rmsprop_kernel[grid]()