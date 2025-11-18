from torch import nn, Tensor
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl
from math import ceil

from ...utils import validate_tensor_device, validate_contiguous
import math


@triton.jit
def _compute_tanh_phi(x):
    angle = tl.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * angle)
    tanh = (exp - 1) / (exp + 1)
    phi = 0.5 * (1 + tanh)
    return tanh, phi


@triton.jit
def _compute_tanh_phi(x):
    # angle = tl.sqrt(2 / math.pi)  * (x + 0.044715 * x * x * x)
    angle = 0.7978845608028654  * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * angle)
    tanh = (exp - 1) / (exp + 1)
    phi = 0.5 * (1 + tanh)
    return tanh, phi


@triton.jit()
def _gelu_op_fwd(x):
    _, phi = _compute_tanh_phi(x)
    return x * phi


@triton.jit
def _gelu_fwd_triton(x_ptr, act_ptr, num_elements, block_size: tl.constexpr):

    pid = tl.program_id(axis=0)
    ptr_offset = pid * block_size + tl.arange(0, block_size)
    mask = ptr_offset < num_elements
    x = tl.load(x_ptr + ptr_offset, mask)

    chunk_act = _gelu_op_fwd(x)
    tl.store(act_ptr + ptr_offset, chunk_act, mask)


def _gelu_fwd(x: Tensor, block_size: int = 2048) -> Tensor:
    validate_tensor_device(x)

    num_elements = x.numel()
    grid = (ceil(num_elements / block_size),)
    act = torch.empty_like(x).to(x.device)

    _gelu_fwd_triton[grid](x, act, num_elements, block_size)
    return act  # , tanh, phi


@triton.jit()
def _gelu_op_bwd(x, grad_output):
    tanh, phi = _compute_tanh_phi(x)

    factor = 1 / tl.sqrt(2 * math.pi)
    phi_prime = factor * (1 - tanh * tanh) * (1 + 3 * 0.044715 * x * x)

    derivative = x * phi_prime + phi
    return derivative * grad_output


@triton.jit
def _gelu_bwd_triton(
    x_ptr, grad_input_ptr, grad_output_ptr, num_elements, block_size: tl.constexpr
):

    pid = tl.program_id(axis=0)
    ptr_offset = pid * block_size + tl.arange(0, block_size)
    mask = ptr_offset < num_elements

    x = tl.load(x_ptr + ptr_offset, mask)
    grad_output = tl.load(grad_output_ptr + ptr_offset, mask)

    grad_input = _gelu_op_bwd(x, grad_output)
    tl.store(grad_input_ptr + ptr_offset, grad_input, mask)


def _gelu_bwd(x: Tensor, grad_output: Tensor, block_size: int = 512) -> Tensor:
    # Aggiungi questi controlli
    validate_tensor_device(x)

    assert x.numel() == grad_output.numel(), "Expected x.numel() = grad_output.numel()"
    assert x.is_contiguous(), f"x not contiguous, {x.stride()=}"
    assert (
        grad_output.is_contiguous()
    ), f"grad_output not contiguous, {grad_output.stride()=}"

    num_elements = x.numel()
    grid = (ceil(num_elements / block_size),)

    grad_input = torch.empty_like(x).to(x.device)

    _gelu_bwd_triton[grid](x, grad_output, grad_input, num_elements, block_size)


class GeluFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        act = _gelu_fwd(x)
        ctx.save_for_backward(x)
        # return _gelu(x)
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
        grad_input = _gelu_bwd(x, grad_output)
        return grad_input


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 1:
            x = validate_contiguous(x)
            return GeluFunction.apply(x.view(-1)).view(x.shape)
        return GeluFunction.apply(x)


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

_fwd = GELU()


def fwd(x, provider):
    ### for benchmark olny!
    if provider == "torch":
        return torch.nn.functional.gelu(x)
    elif provider == "triton":
        return _fwd(x)
    else:
        raise ValueError
