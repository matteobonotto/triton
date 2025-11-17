from ...utils import validate_contiguous
from ..activation.gelu import _gelu_op_fwd, _gelu_op_bwd

from math import ceil
import triton
import triton.language as tl
import torch
from torch import nn, Tensor
from torch.autograd import Function

MAP_ACTIVATIONS_FWD = {"gelu": _gelu_op_fwd}
MAP_ACTIVATIONS_BWD = {"gelu": _gelu_op_bwd}


@triton.jit()
def _activation_fwd(x, activation_name: tl.constexpr):
    return MAP_ACTIVATIONS_FWD[activation_name](x)


@triton.jit()
def _activation_bwd(x, grad_output, activation_name: tl.constexpr):
    return MAP_ACTIVATIONS_BWD[activation_name](x, grad_output)


@triton.jit()
def _linear_act_fwd_triton(
    x_ptr,
    W_ptr,
    b_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    activation_name,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    GROUP_SIZE_M : tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    # map pid to (pid_m, pid_n)
    pid_m = pid // num_programs_n
    pid_n = pid % num_programs_n
    
    


def _linear_act_fwd(x, W, b, activation_name: str = "silu"):
    """
    Triton implementation of the following kernel
        z = x @ W.T + b
        out =  f(z)

    """

    M, K = x.shape
    _, N = W.shape

    # using naive block matmul for now, implement more efficient algo later

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (64, 64, 64)  # hard-coded for now
    GROUP_SIZE_M = 8

    grid = ((ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)),)

    out = torch.zeros((M, N)).to(x.device)

    out = _linear_act_fwd_triton[grid](
        x,
        W,
        b,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        W.stride(0),
        W.stride(1),
        b.stride(0),
        out.stride(0),
        out.stried(1),
        activation_name,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )
    ...


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
