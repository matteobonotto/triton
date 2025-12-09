import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple
import math

from .utils import get_num_streaming_multiprocessors


@triton.jit()
def _compute_abc(): ...


@triton.jit()
def _compute_dx(): ...


def mlp_hidden_states_bwd(
    x: Tensor, W_up: Tensor, W_gp: Tensor, grad_output: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    B, M, K = x.shape
    N, _ = W_gp.shape

    kwargs = {"device": x.device, "dtype": x.dtype}

    dx = torch.zeros(x.shape, **kwargs).view(-1, x.shape[-1])
    a = torch.zeros(dx.shape, **kwargs)  # check dtype
    b = torch.zeros(dx.shape, **kwargs)  # check dtype
    c = torch.zeros(dx.shape, **kwargs)  # check dtype

    dW_up = torch.zeros(W_up.shape, **kwargs)  # check dtype
    dW_gp = torch.zeros(W_gp.shape, **kwargs)  # check dtype

    NUM_SMS = get_num_streaming_multiprocessors()
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)
    grid = (
        min(NUM_SMS, math.ceil(B * M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),
    )

    _compute_abc[grid](...)

    _compute_dx[grid](...)

    # _compute_dW_up[grid](...)
    # _compute_dW_gp[grid](...)

    return dx.view(B, M, K), dW_up, dW_gp
