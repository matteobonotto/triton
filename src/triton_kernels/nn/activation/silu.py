

from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl
from math import ceil

from ...utils import validate_tensor_device, validate_contiguous

import math

@triton.jit()
def _compute_sigma(x):
    return 1 / (1 + tl.exp(-x))


@triton.jit
def _silu_fwd_triton(x_poiter, act_pointer, num_elements, block_size: tl.constexpr):

    pid = tl.program_id(axis=0)
    pointers = pid*block_size + tl.arange(0, block_size)
    mask = pointers < num_elements

    x = tl.load(x_poiter + pointers, mask)
    # sigma = 1 / (1 + tl.exp(-x))
    # chunk_res = x * sigma
    # chunk_res = x / (1 + tl.exp(-x))
    chunk_res = x * _compute_sigma(x)

    tl.store(act_pointer + pointers, chunk_res, mask)
    # tl.store(sigma_pointer+pointers, sigma, mask)


def _silu_fwd(x:Tensor, block_size:int=1024) -> Tensor:

    num_elements = x.numel()
    grid = ceil(num_elements / block_size),

    act = torch.empty_like(x).to(x.device)
    _silu_fwd_triton[grid](x, act, num_elements, block_size)
    return act#, sigma


@triton.jit()
def _silu_triton_bwd(x_ptr, grad_output_ptr, grad_input_ptr, num_elements, block_size: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    
    offset = tl.arange(0, block_size)
    ptr_offset = pid * block_size + offset
    mask = ptr_offset <  num_elements
    # offset_ptr = tl.arange(0, )
    
    # for row_idx in tl.range(0, num_elements, block_size):
    x = tl.load(x_ptr + ptr_offset, mask)
    grad_output = tl.load(grad_output_ptr + ptr_offset, mask)
    
    sigma = _compute_sigma(x)
    act_prime = sigma * (1 + x * (1 - sigma))
    grad_input = grad_output * act_prime
    
    tl.store(grad_input_ptr + ptr_offset, grad_input, mask)
        
            
    ...
    
def _silu_bwd(x: Tensor, grad_output: Tensor, block_size: int = 2048) -> Tensor:
    
    validate_tensor_device(x)
    
    assert x.numel() == grad_output.numel(), "Expected x.numel() = grad_output.numel()"
    assert x.is_contiguous(), f"x not contiguous, {x.stride()=}"
    assert (
        grad_output.is_contiguous()
    ), f"grad_output not contiguous, {grad_output.stride()=}"
    
    num_elements = x.numel()
    grid = math.ceil(num_elements / block_size),
    
    grad_input = torch.empty_like(x).to(x.device)
    
    _silu_triton_bwd[grid](x, grad_output, grad_input, num_elements, block_size)

    return grad_input



class SiluFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        act = _silu_fwd(x)
        ctx.save_for_backward(x)
        return act

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        # factor = 1 / (2 * math.pi) ** 0.5
        # gx = (2 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)
        # tanh = nn.functional.tanh(gx)
        # phi_prime = factor * (1 - tanh**2) * (1 + 3 * 0.044715 * x ** 2)
        # phi = 0.5 * (1 + tanh)

        # derivative = phi + x * phi_prime
        grad_output = validate_contiguous(grad_output)
        grad_input = _silu_bwd(x, grad_output)
        return grad_input
    

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 1:
            x = validate_contiguous(x)
            return SiluFunction.apply(x.view(-1)).view(x.shape)
        return SiluFunction.apply(x)


base_benchmark_kwargs = {
    "x_names": ["N"],  # argument names to use as an x-axis for the plot
    "x_vals": [
        128 * i for i in range(2, 100, 10)
    ],  # different possible values for `x_name`
    "line_arg": "provider",  # argument name whose value corresponds to a different line in the plot
    "line_vals": ["triton", "torch"],  # possible values for `line_arg``
    "line_names": ["Triton", "Torch"],  # label name for the lines
    "plot_name": "gelu",  # name for the plot. Used also as a file name for saving the plot.
    "args": {"M": 4096},  # values for function arguments not in `x_names` and `y_name`
}

_fwd = SiLU()


def fwd(x, provider):
    ### for benchmark olny!
    if provider == "torch":
        return torch.nn.functional.silu(x)
    elif provider == "triton":
        return _fwd(x)
    else:
        raise ValueError