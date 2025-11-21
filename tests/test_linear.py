import pytest
import torch
from torch import Tensor, autograd, nn
import triton

from triton_kernels.utils import get_device, copy_weights
from triton_kernels.nn.linear import Linear

DEVICE = get_device()

test_tensors = [
    torch.rand(10),
    torch.rand(10, 20),
    torch.rand(10, 20, 30),
    torch.rand(10, 20, 30, 40),
]


M = 100
N = 200
x = torch.randn(
    M, N, device=DEVICE, dtype=torch.float32
)  # Changed dtype to float32 for debugging
_fwd = Linear(N, N, bias=False).to(DEVICE)
_fwd_ref = nn.Linear(N, N, bias=False).to(DEVICE)


def fwd(x, provider):
    ### for benchmark olny!
    if provider == "torch":
        return _fwd_ref(x)
    elif provider == "triton":
        return _fwd(x)
    else:
        raise ValueError


from torch import autograd


def bwd(x, provider):
    x.requires_grad = True
    out = fwd(x, provider)

    loss = out.sum()
    _fwd_provider = _fwd if provider == "triton" else _fwd_ref
    if _fwd_provider.bias is not None:
        inputs = (x, _fwd_provider.weight, _fwd_provider.bias)
    else:
        inputs = (x, _fwd_provider.weight)
    grads_ref = autograd.grad(loss, inputs)


# def bwd(x, provider):
#     x.requires_grad = True
#     out = fwd(x, provider)
#     loss = out.sum()
#     loss.backward()


MAP_FWD_BKW = {
    "fwd": fwd,
    "bwd": bwd,
}

for mode in ["fwd", "bwd"]:
    for provider in ["triton", "torch"]:
        out = MAP_FWD_BKW[mode](x, provider)


@pytest.mark.parametrize("x", test_tensors)
def test_linear(x: Tensor):
    x = x.to(DEVICE)
    x.requires_grad = True

    for has_bias in [True, False]:
        linear = Linear(x.shape[-1], x.shape[-1], bias=has_bias)
        linear_ref = nn.Linear(x.shape[-1], x.shape[-1], bias=has_bias)
        copy_weights(
            linear_ref,
            linear,
        )

        out = linear(x)
        out_ref = linear_ref(x)
        triton.testing.assert_close(out, out_ref, atol=1e-6)

        # loss_ref = out_ref.sum()
        # loss_ref.backward()
        # print(linear_ref.weight.grad[0,0])

        # loss = out.sum()
        # loss.backward()
        # print(linear.weight.grad[0,0])

        grad_outputs = torch.rand_like(x).to(DEVICE)
        inputs = (
            (x, linear_ref.weight, linear_ref.bias)
            if has_bias
            else (
                x,
                linear_ref.weight,
            )
        )
        grads_ref = autograd.grad(
            out_ref,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )
        inputs = (
            (x, linear.weight, linear.bias)
            if has_bias
            else (
                x,
                linear.weight,
            )
        )
        grads = autograd.grad(
            out,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )
        for g_r, g in zip(grads_ref, grads):
            triton.testing.assert_close(g_r, g, atol=1e-6)
        print("Done!!")


# test_linear(test_tensors[-1])
