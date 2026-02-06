import triton
import triton.language as tl
from torch import Tensor
import torch


@triton.jit()
def _rmsprop_kernel(
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
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Triton kernel for RMSprop optimizeer. Implements the following logit:

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
    """ Everything is done in-place, no need to return enything. """

    for i, param in enumerate(params):
        step = state_steps[i]
        grad = grads[i]
        square_avg = square_avgs[i]

        step += 1

        n_rows, n_cols = param.shape

        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = triton.next_power_of_2(n_cols)

        grid = (tl.cdiv(n_rows, BLOCK_SIZE_ROW),)
        with torch.cuda.device(grad.device):
            _rmsprop_kernel[grid](
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
                BLOCK_SIZE_ROW,
                BLOCK_SIZE_COL,
            )
