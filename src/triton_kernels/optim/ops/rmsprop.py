import triton
import triton.language as tl
from torch import Tensor
import torch
import math
import os


def cdiv(a, b):
    return math.ceil(a / b)


def get_autotune_config(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_ROW": x,
            }
        )  #
        for x in [16]  # [16, 32, 64, 128, 256]
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=[
        "n_rows",
    ],
)
@triton.jit()
def _block_rmsprop_kernel(
    param_ptr,
    square_avg_ptr,
    grad_ptr,
    param_stride_row,
    param_stride_col,
    square_avg_stride_row,
    square_avg_stride_col,
    grad_stride_row,
    grad_stride_col,
    lr,
    alpha,
    eps,
    n_rows,
    n_cols,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    """
    Triton kernel for rmsprop optimizeer. Implements the following logit:

    v = alpha * v + (1 - alpha) * grad ** 2
    params -= lr * grad / (v ** 0.5 + eps)
    """

    pid = tl.program_id(axis=0)
    # these are all element-wise operations, no need to do tiled stuff

    row_offset = pid + tl.arange(0, BLOCK_SIZE_ROW)
    col_offset = tl.arange(0, BLOCK_SIZE_COL)
    mask = (row_offset < n_rows)[:, None] & (col_offset < n_cols)[None, :]

    ### load the tiles for computaiton
    param = tl.load(
        param_ptr
        + row_offset[:, None] * param_stride_row
        + col_offset[None, :] * param_stride_col,
        mask=mask,
        other=0.0,
    )
    square_avg = tl.load(
        square_avg_ptr
        + row_offset[:, None] * square_avg_stride_row
        + col_offset[None, :] * square_avg_stride_col,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(
        grad_ptr
        + row_offset[:, None] * grad_stride_row
        + col_offset[None, :] * grad_stride_col,
        mask=mask,
        other=0.0,
    )

    ### do actual computation
    square_avg = alpha * square_avg + (1 - alpha) * grad * grad
    param -= lr * grad / (tl.sqrt(square_avg) + eps)

    ### store back the results in memory
    tl.store(
        square_avg_ptr
        + row_offset[:, None] * square_avg_stride_row
        + col_offset[None, :] * square_avg_stride_col,
        square_avg,
        mask=mask,
    )
    tl.store(
        param_ptr
        + row_offset[:, None] * param_stride_row
        + col_offset[None, :] * param_stride_col,
        param,
        mask=mask,
    )


# @torch.no_grad()
def block_rmsprop_interface(
    param: Tensor,
    grad: Tensor,
    square_avg: Tensor,
    lr: float = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
) -> None:
    """Everything is done in-place, no need to return enything."""

    n_rows, n_cols = param.shape

    grid = lambda META: (triton.cdiv(n_rows, META["BLOCK_SIZE_ROW"]),)
    # BLOCK_SIZE_ROW = lambda META: META["BLOCK_SIZE_ROW"]
    BLOCK_SIZE_COL = triton.next_power_of_2(n_cols)

    _block_rmsprop_kernel[grid](
        param,
        square_avg,
        grad,
        param.stride(0),
        param.stride(1),
        square_avg.stride(0),
        square_avg.stride(1),
        grad.stride(0),
        grad.stride(1),
        lr,
        alpha,
        eps,
        n_rows,
        n_cols,
        BLOCK_SIZE_COL,
        # BLOCK_SIZE_ROW,
    )


def rmsprop_interface_torch(
    param: Tensor,
    grad: Tensor,
    square_avg: Tensor,
    lr: float = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    implace: bool = True,
):
    if not implace:
        square_avg *= alpha
        square_avg += (1 - alpha) * grad.pow(2)  # <- optimizer params, shape (N, )
        param -= lr * grad / (torch.sqrt(square_avg) + eps)
    else:
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        avg = square_avg.sqrt()
        avg = avg.add_(eps)
        param.addcdiv_(grad, avg, value=-lr)


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
    restore_value=["param_ptr", "square_avg_ptr"],
)


@autotune_decorator
@triton.jit()
def _rmsprop_kernel(
    param_ptr,
    square_avg_ptr,
    grad_ptr,
    lr,
    alpha,
    eps,
    num_el,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for rmsprop optimizeer. Implements the following logit:

    v = alpha * v + (1 - alpha) * grad ** 2
    params -= lr * grad / (v ** 0.5 + eps)
    """

    pid = tl.program_id(axis=0)
    # these are all element-wise operations, no need to do tiled stuff
    # print(f"{pid=}")

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_el

    ### load the tiles for computaiton
    param = tl.load(
        param_ptr + offset,
        mask=mask,
        other=0.0,
    )
    square_avg = tl.load(
        square_avg_ptr + offset,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(
        grad_ptr + offset,
        mask=mask,
        other=0.0,
    )

    ### do actual computation
    square_avg = alpha * square_avg + (1 - alpha) * grad * grad
    param -= lr * grad / (tl.sqrt(square_avg) + eps)

    ### store back the results in memory
    tl.store(
        square_avg_ptr + offset,
        square_avg,
        mask=mask,
    )
    tl.store(
        param_ptr + offset,
        param,
        mask=mask,
    )


# @torch.no_grad()
def rmsprop_interface(
    param: Tensor,
    grad: Tensor,
    square_avg: Tensor,
    lr: float = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
) -> None:
    """Everything is done in-place, no need to return enything."""

    n_rows, n_cols = param.shape
    num_el = n_rows * n_cols

    grid = lambda META: (triton.cdiv(num_el, META["BLOCK_SIZE"]),)

    # BLOCK_SIZE = 16
    # grid = cdiv(num_el, BLOCK_SIZE) ,

    _rmsprop_kernel[grid](
        param.view(-1),
        square_avg.view(-1),
        grad.view(-1),
        lr,
        alpha,
        eps,
        num_el,
        # BLOCK_SIZE,
    )


@torch.no_grad()
def rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    state_steps: list[Tensor],
    lr: float = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
) -> None:
    """Everything is done in-place, no need to return enything."""

    for i, param in enumerate(params):
        step = state_steps[i]
        grad = grads[i]
        square_avg = square_avgs[i]

        step += 1

        with torch.cuda.device(grad.device):
            rmsprop_interface(
                param=param,
                square_avg=square_avg,
                grad=grad,
                lr=lr,
                alpha=alpha,
                eps=eps,
            )
