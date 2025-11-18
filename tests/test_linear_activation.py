
from triton_kernels.nn.fused.linear_activation import _linear_act_fwd
from triton_kernels.functional.matmul import matmul
from triton_kernels.utils import get_device

import torch

DEVICE = get_device()

def test_linear_activation():
    M, N, K = (128, 128, 128)
    # M, N, K = (4, 4, 4)
    x = torch.rand(M, K).to(DEVICE) / M / K
    W = torch.rand(K, N).to(DEVICE) / K / N
    b = torch.rand(N).to(DEVICE) / N

    out_ref = torch.nn.functional.gelu(x @ W + b)
    out = _linear_act_fwd(x, W, b)
    # out_2 = matmul(x, W)
    print((out - out_ref).norm()/out_ref.norm())
    # print((out_2 - out_ref).norm()/out_ref.norm())
    ...
    
test_linear_activation()