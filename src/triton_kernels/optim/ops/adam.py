import triton
import triton.language as tl
from torch import Tensor
import torch
import math
import os


def adam_interface_torch(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: float,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    implace: bool = True,
):
    if not implace:
        # update running averages
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

        # rescaling
        exp_avg_hat = exp_avg / (1 - beta1**step)
        exp_avg_sq_hat = exp_avg_sq / (1 - beta2**step)

        # update model param
        param -= lr * exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + eps)
    else:
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)


def cdiv(a, b):
    return math.ceil(a / b)


def get_autotune_config():
    block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return [
        triton.Config(
            {
                "BLOCK_SIZE": x,
            }
        )
        for x in block_sizes
    ]


autotune_decorator = triton.autotune(
    configs=get_autotune_config(),
    key=["num_el"],
    restore_value=["param_ptr", "exp_avg_ptr", "exp_avg_sq_ptr"],
)

@autotune_decorator
@triton.jit()
def _adam_kernel(
    param_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    grad_ptr,
    lr,
    beta1,
    beta2,
    bc1_ptr, 
    bc2_ptr,
    eps,
    num_el,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for adam optimizeer. Implements the following logit:

    # update running averages
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

    # rescaling
    exp_avg_hat = exp_avg / (1 - beta1**step)
    exp_avg_sq_hat = exp_avg_sq / (1 - beta2**step)

    # update model param
    param -= lr * exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + eps)
    """

    # these are all element-wise operations, no need to do tiled stuff
    pid = tl.program_id(axis=0)
    # print(f"{pid=}")

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_el

    ### load the tiles for computaiton
    param = tl.load(
        param_ptr + offset,
        mask=mask,
        other=0.0,
    )
    exp_avg = tl.load(
        exp_avg_ptr + offset,
        mask=mask,
        other=0.0,
    )
    exp_avg_sq = tl.load(
        exp_avg_sq_ptr + offset,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(
        grad_ptr + offset,
        mask=mask,
        other=0.0,
    )
    bc1 = tl.load(bc1_ptr) 
    bc2 = tl.load(bc2_ptr) 
    
    ### do actual computation
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

    exp_avg_hat = exp_avg / bc1
    exp_avg_sq_hat = exp_avg_sq / bc2

    param -= lr * exp_avg_hat / (tl.sqrt(exp_avg_sq_hat) + eps)

    ### store back the results in memory
    tl.store(
        exp_avg_ptr + offset,
        exp_avg,
        mask=mask,
    )
    tl.store(
        exp_avg_sq_ptr + offset,
        exp_avg_sq,
        mask=mask,
    )
    tl.store(
        param_ptr + offset,
        param,
        mask=mask,
    )


# @torch.no_grad()
def adam_interface(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: float,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Everything is done in-place, no need to return enything."""

    n_rows, n_cols = param.shape
    num_el = n_rows * n_cols

    grid = lambda META: (triton.cdiv(num_el, META["BLOCK_SIZE"]),)

    # BLOCK_SIZE = 16
    # grid = cdiv(num_el, BLOCK_SIZE) ,

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)

    _adam_kernel[grid](
        param.view(-1),
        exp_avg.view(-1),
        exp_avg_sq.view(-1),
        grad.view(-1),
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
        num_el,
        # BLOCK_SIZE,
    )


@torch.no_grad()
def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Everything is done in-place, no need to return enything."""

    for i, param in enumerate(params):
        step = state_steps[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        step += 1

        with torch.cuda.device(grad.device):
            # adam_interface(
            adam_interface_torch(
                param=param,
                grad=grad,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                step=step,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )
