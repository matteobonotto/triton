import pytest
import os
import torch
from torch import Tensor, nn

import triton
import triton.language as tl
import math

from torch import autograd
from triton_kernels.utils import get_device
from triton_kernels.nn import SiLU

DEVICE = get_device()

test_tensors = [torch.rand(100), torch.rand(100, 100)]


@pytest.mark.parametrize("x", test_tensors)
def test_gelu(x: Tensor):
    x = x.to(DEVICE)
    x.requires_grad = True

    out = SiLU()(x)
    out_ref = nn.SiLU()(x)
    triton.testing.assert_close(out, out_ref, atol=1e-6)

    grad_outputs = torch.rand_like(x).to(DEVICE)
    grads_ref = autograd.grad(
        out_ref, (x,), grad_outputs=grad_outputs, retain_graph=True
    )
    grads = autograd.grad(out, (x,), grad_outputs=grad_outputs, retain_graph=True)
    for g_r, g in zip(grads_ref, grads):
        triton.testing.assert_close(g_r, g, atol=1e-6)
    print("Done!!")


test_gelu(test_tensors[1])
