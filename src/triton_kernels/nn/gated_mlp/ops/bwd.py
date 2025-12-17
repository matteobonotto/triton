import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple
import math

from .utils import get_num_streaming_multiprocessors


@triton.jit()
def _compute_fwd_bwd_quantities(): 
    """
    Kernel for computing:

        a = x @ W_up.T
        b = x @ W_gp.T
        c = act_fwd(b)

        sigma = torch.sigmoid(b)
        act_prime = sigma * (1 + b * (1 - sigma))
        
        grad_output_a_act_prime = (grad_output * a) * act_prime
        grad_output_c = grad_output * c
    """



@triton.jit()
def _compute_dx(): ...


def mlp_hidden_states_bwd(
    x: Tensor, W_up: Tensor, W_gp: Tensor, grad_output: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    act_fun = "silu" # hard-coded for now, parametrize later
    
    B, M, K = x.shape
    N, _ = W_gp.shape

    kwargs = {"device": x.device, "dtype": x.dtype}

    dx = torch.zeros(x.shape, **kwargs).view(-1, x.shape[-1])
    a = torch.zeros(dx.shape, **kwargs)  # check dtype
    b = torch.zeros(dx.shape, **kwargs)  # check dtype
    c = torch.zeros(dx.shape, **kwargs)  # check dtype
    act_prime = torch.zeros(dx.shape, **kwargs)  # check dtype
    grad_output_a_act_prime = torch.zeros(dx.shape, **kwargs)  # check dtype
    grad_output_c = torch.zeros(dx.shape, **kwargs)  # check dtype

    dW_up = torch.zeros(W_up.shape, **kwargs)  # check dtype
    dW_gp = torch.zeros(W_gp.shape, **kwargs)  # check dtype

    NUM_SMS = get_num_streaming_multiprocessors()
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)


    ### this kernel is doing only element-wise computations 
    _compute_fwd_bwd_quantities[grid](
        x, 
        W_up,
        W_gp,
        act_fun, 
        M, N, K, 
    )

    grid = (
        min(NUM_SMS, math.ceil(B * M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),
    )
    _compute_dx[grid](...)

    # _compute_dW_up[grid](...)
    # _compute_dW_gp[grid](...)

    return dx.view(B, M, K), dW_up, dW_gp
