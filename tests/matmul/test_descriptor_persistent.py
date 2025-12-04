from triton_kernels.utils import get_device, is_cuda
from triton_kernels.functional.matmul.descriptor_persistent import matmul as dp_matmul
from triton_kernels.functional.matmul.block import matmul as b_matmul
import torch
import math
import random
import pytest
import triton
import triton.language as tl

# from triton_kernels.functional.matmul import persistent_matmul as matmul

# from triton_kernels.functional.matmul import eager_matmul as matmul


# test_map_pid_L2_optimization()


def test_simple_matmul():
    DEVICE = get_device()
    torch.manual_seed(0)

    M, N, K = (4, 4, 5)
    M, N, K = (64, 75, 66)
    # M, N, K = (64, 64, 64)

    a = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.rand((K, N), device=DEVICE, dtype=torch.float32)

    triton_output_p = dp_matmul(a, b, DEBUG=True)
    triton_output_b = b_matmul(a, b, optimize_L2=False)
    torch_output = torch.matmul(a, b)

    print((torch_output - triton_output_p).norm() / torch_output.norm())
    print((torch_output - triton_output_b).norm() / torch_output.norm())

    print(f"Persistent triton output: {triton_output_p}")
    print(f"Block triton output:      {triton_output_b}")
    print(f"Torch:                    {torch_output}")
    assert torch.allclose(triton_output_p, torch_output, atol=1e-2, rtol=0)


test_simple_matmul()

# test_matmul_1()

dtypes = [torch.float32, torch.bfloat16, torch.float16]
pytest.mark.parametrize("dtype", dtypes, "optimize_L2", [True, False])


def test_matmul_block(dtype: torch.dtype, optimize_L2: bool):
    print(f"Running test_matmul, {dtype=}, {optimize_L2=}")
    # M, N, K = (128, 234, 128)
    # M, N, K = (64, 75, 64)
    # M, N, K = (9, 8, 9)
    # M, N, K = (14, 14, 14)
    # M, N, K = (2, 2, 2)
    # M, N, K = (64, 64, 64)
    # M, N, K = (16, 19, 16)
    DEVICE = get_device()
    if DEVICE == torch.device("cpu") and dtype == torch.bfloat16:
        print(f"Skipping test, bfloat16 not available on cpu")
    else:
        for _ in range(10):
            M, N, K = tuple(random.randint(1, 2048) for _ in range(3))
            # print(f"{M=}, {N=}, {K=}, ")

            a = torch.rand(M, K).to(dtype)
            b = torch.rand(K, N).to(dtype)

            out_ref = a @ b
            out = dp_matmul(a, b, optimize_L2=optimize_L2)

            assert (out_ref - out).norm() / out_ref.norm() < 1e-6
            print((out_ref - out).norm() / out_ref.norm())

        ...


# test_matmul(dtype=torch.float32, optimize_L2=True)
