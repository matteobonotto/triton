from ...utils import validate_contiguous

from ..activation.gelu import _gelu_op_fwd, _gelu_op_bwd
from ..activation.silu import _silu_op_fwd, _silu_op_bwd

from ...autotune import block_matmul_autotune_base_config

from math import ceil
import triton
import triton.language as tl
import torch
from torch import nn, Tensor
from torch.autograd import Function


@triton.jit()
def _activation_fwd(x, activation: tl.constexpr):
    if activation == "silu":
        return _silu_op_fwd(x)
    elif activation == "gelu":
        return _gelu_op_fwd(x)
    elif activation == "no_activation":
        return x
    else:
        raise NotImplementedError
    # return MAP_ACTIVATIONS_FWD[activation_name](x)


@triton.jit()
def _activation_bwd(x, grad_output, activation: tl.constexpr):
    if activation == "silu":
        return _silu_op_bwd(x, grad_output)
    elif activation == "gelu":
        return _gelu_op_bwd(x, grad_output)
    elif activation == "no_activation":
        return x * grad_output
    else:
        raise NotImplementedError


@triton.autotune(
    configs=block_matmul_autotune_base_config(),
    key=["M", "N", "K"],
)
@triton.jit()
def _linear_act_fwd_triton(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_bn,
    stride_om,
    stride_on,
    activation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    ### map pid to (pid_m, pid_n)
    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n

    ### get offsets along M, N, K directions
    offset_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offset_m[:, None] < M
    mask_n = offset_n[None, :] < N

    ### get starting pointers for x, w, b
    tile_x_ptr = x_ptr + (offset_m[:, None] * stride_xm + offset_k[None, :] * stride_xk)
    tile_w_ptr = w_ptr + (offset_k[:, None] * stride_wk + offset_n[None, :] * stride_wn)
    tile_b_ptr = b_ptr + offset_n[None, :] * stride_bn

    tile_b = tl.load(tile_b_ptr, mask_n, other=0.0)

    ### frist compute z = x @ W.T
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # mask values depending on k: note that we are not loading all the row
        # up to k * BLOCK_SIZE_K. Since we are advancing the pointers
        # (see tile_x_ptr += ..., tile_w_ptr += ...) we are considering only
        # the portion [k*BLOCK_SIZE_K:(k+1)*BLOCK_SIZE_K].
        mask_k = offset_k < K - k * BLOCK_SIZE_K

        tile_x = tl.load(tile_x_ptr, mask_k[None, :], other=0.0)
        tile_w = tl.load(tile_w_ptr, mask_k[:, None], other=0.0)

        z = tl.dot(tile_x, tile_w, z)

        # advance the pointers along k directions to the next block
        tile_x_ptr += BLOCK_SIZE_K * stride_xk
        tile_w_ptr += BLOCK_SIZE_K * stride_wk

    ### then compute +b and activation + downcast to bfloat16
    z += tile_b
    # o = z
    o = _activation_fwd(z, activation)
    # o = o.to(tl.bfloat16)

    tile_o_ptr = o_ptr + offset_m[:, None] * stride_om + offset_n[None, :] * stride_on
    mask_o = mask_m & mask_n
    tl.store(tile_o_ptr, o, mask_o)


def _linear_act_fwd(
    x: Tensor,
    W: Tensor,
    b: Tensor,
    transpose_x: bool = False,
    transpose_W: bool = False,
    activation: str = "gelu",
):
    """
    Triton implementation of the following kernel
        z = x @ W.T + b
        out =  f(z)

    IMPORTANT NOTE: the .T operation is done inside tge triton kernel by swapping the
    stride values of W, therefore pass W and not W.T.

    """
    assert x.ndim == 2, "expected A to have ndim=2"
    assert W.ndim == 2, "expected B to have ndim=2"

    if transpose_x:
        M, K = x.shape
        raise NotImplementedError
    else:
        K, M = x.shape

    if transpose_W:
        N = W.shape[0]
        assert K == W.shape[1], "dimension mismatch"
        stride_W = (W.stride(1), W.stride(0))
    else:
        N = W.shape[1]
        stride_W = (W.stride(0), W.stride(1))
        assert K == W.shape[0], "dimension mismatch"

    assert b.ndim <= 1, f"b is expected to be either 0D or 1D tensor, got {b.ndim}D"
    # assert b.shape[0] == W.shape[1], "dimension mismatch"

    if b.ndim == 0:
        b = b.view(1, -1)

    # using naive block matmul for now, implement more efficient algo later

    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (64, 64, 64)  # hard-coded for now
    # GROUP_SIZE_M = 8
    # grid = ((ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)),)

    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _linear_act_fwd_triton[grid](
        x,
        W,
        b,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        stride_W[0],
        stride_W[1],
        b.stride(0),
        out.stride(0),
        out.stride(1),
        activation,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_N,
        # BLOCK_SIZE_K,
        # GROUP_SIZE_M,
    )
    ...

    return out.to(x.dtype)


@triton.jit()
def _linear_act_bwd_triton(): ...


def _linear_act_bwd(x, W, b, grad_output): ...


class LinearActivationFunction(Function):
    def forward(ctx, x: Tensor, W: Tensor, b: Tensor):
        ctx.save_for_backward(x, W, b)
        return _linear_act_fwd(x, W, b)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor,
    ):
        x, W, b = ctx.saved_tensors
        grad_x, grad_W, grad_b = _linear_act_bwd(x, W, b, grad_output)
        return grad_x, grad_W, grad_b


class Linear(nn.Linear):
    def __init__(self, act: str = "silu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_kind = act

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 2:
            x = validate_contiguous(x)
            new_shape = (-1, x.shape[-1])
            return LinearActivationFunction.apply(
                x.view(new_shape),
                self.weight,
                self.bias if self.bias is not None else None,
            ).view(x.shape)
        return LinearActivationFunction.apply(
            x,
            self.weight,
            self.bias if self.bias is not None else None,
        )
