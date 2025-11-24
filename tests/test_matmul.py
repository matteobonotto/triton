import torch
import pytest

from triton_kernels.functional.matmul import swizzle_matmul as matmul

# from triton_kernels.functional.matmul import eager_matmul as matmul


def test_matmul():
    M, N, K = (128, 128, 128)
    M, N, K = (9, 8, 9)
    M, N, K = (4, 3, 4)
    # M, N, K = (64, 64, 64)
    # M, N, K = (4, 4, 4)

    DTYPE = torch.float32

    A = torch.rand(M, K).to(DTYPE)
    B = torch.rand(K, N).to(DTYPE)
    # A = torch.tensor([[1.,2.,3.],[1.,2.,3.]])
    # B = A.T

    out_ref = A @ B
    out = matmul(A, B)

    print((out_ref - out).norm() / out_ref.norm())

    ...


test_matmul()
