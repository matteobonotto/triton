
from triton_kernels.nn.fused.linear_activation import _linear_act_fwd
from triton_kernels.functional.matmul import matmul, matmul_swizzling
from triton_kernels.utils import get_device

import torch

DEVICE = get_device()

def test_linear_activation():
    M, N, K = (128, 256 + 10, 128)
    dtype = torch.bfloat16
    # M, N, K = (4, 4, 4)
    x = torch.rand(M, K, device=DEVICE, dtype=dtype) / M / K
    W = torch.rand(K, N, device=DEVICE, dtype=dtype) / K / N
    b = 0 * torch.rand(N, device=DEVICE, dtype=dtype) / K / N

    out_ref = x @ W + b
    out = _linear_act_fwd(x, W, b, activation='no_activation')
    # out_2 = matmul_swizzling(x, W)
    print((out - out_ref).norm()/out_ref.norm())
    # print((out_2 - out_ref).norm()/out_ref.norm())
    
    out_ref = torch.nn.functional.silu(x @ W + b)
    out = _linear_act_fwd(x, W, b, activation='silu')
    # out_2 = matmul(x, W)
    print((out - out_ref).norm()/out_ref.norm())
    # print((out_2 - out_ref).norm()/out_ref.norm())
    ...
    
    M, N, K = (128, 256, 128)
    # M, N, K = (4, 8, 4)
    x = torch.rand(M, K, device=DEVICE, dtype=dtype) / M / K
    W_transpose = torch.rand(N, K, device=DEVICE, dtype=dtype) / K / N
    b = torch.rand(N, device=DEVICE, dtype=dtype) / N / N

    out_ref = x @ W_transpose.T
    out_1 = _linear_act_fwd(x, W_transpose, 0 * b, transpose_W=True, activation='no_activation')
    print((out_1 - out_ref).norm()/out_ref.norm())
    
    out_ref = x @ W_transpose.T + b
    out_1 = _linear_act_fwd(x, W_transpose, b, transpose_W=True, activation='no_activation')
    print((out_1 - out_ref).norm()/out_ref.norm())
    
    out_ref = torch.nn.functional.silu(x @ W_transpose.T + b)
    out_2 = _linear_act_fwd(x, W_transpose, b, transpose_W=True, activation='silu')
    print((out_2 - out_ref).norm()/out_ref.norm())
    ...
    
test_linear_activation()