
from triton_kernels.utils import get_device, is_cuda
from triton_kernels.functional.matmul import block_matmul as matmul
from triton_kernels.functional.matmul.ref import matmul as matmul_ref

import torch
import pytest
import triton

# from triton_kernels.functional.matmul import persistent_matmul as matmul

# from triton_kernels.functional.matmul import eager_matmul as matmul


def test_matmul_1():
    DEVICE = get_device()
    torch.manual_seed(0)
    M, N, K = (128, 234, 128)
    a = torch.rand((M,K), device=DEVICE, dtype=torch.float16) 
    b = torch.rand((K, N), device=DEVICE, dtype=torch.float16) 
    triton_output = matmul(a, b, optimize_L2=True)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    
    print((torch_output - triton_output).norm() / torch_output.norm())

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


        
# test_matmul_1()

def test_matmul():
    M, N, K = (128, 234, 128)
    # M, N, K = (64, 75, 64)
    # M, N, K = (9, 8, 9)
    M, N, K = (4, 4, 4)
    # M, N, K = (64, 64, 64)
    # M, N, K = (16, 19, 16)

    DTYPE = torch.float32

    a = torch.rand(M, K).to(DTYPE) / M / K
    b = torch.rand(K, N).to(DTYPE) / K / N
    # A = torch.tensor([[1.,2.,3.],[1.,2.,3.]])
    # B = A.T

    out_ref = a @ b
    out = matmul(a, b, optimize_L2=True)
    # out = matmul_ref(A, B)

    print((out_ref - out).norm() / out_ref.norm())

    ...


test_matmul()
