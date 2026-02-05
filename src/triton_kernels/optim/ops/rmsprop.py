

import triton
import triton.language as tl
from torch import Tensor
import torch



@triton.jit()
def _rmsprop_kernel(params_ptr, v_ptr, grad_ptr, lr, alpha, eps):
    """ 
    Triton kernel for RMSprop optimizeer. Implements the following logit:
    
    v = alpha * v + (1 - alpha) * grad ** 2
    params -= lr * grad / (v ** 0.5 + eps)
    """

    pid = tl.program_id(axis=0)
    # these are all element-wise operations, no need to do tiled stuff

    params = tl.load(..., )
    v = tl.load(..., )
    grad = tl.load(...)

    v = alpha * v + (1 - alpha) * grad * grad
    params -= lr * grad / (tl.sqrt(v) + eps)


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

        step += 1

        grid = ..., 
        with torch.cuda.device(grad.devide):
            _rmsprop_kernel[grid]()