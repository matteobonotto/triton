from triton_kernels.utils import get_device, is_cuda
from triton_kernels.functional.matmul import block_matmul as matmul
from triton_kernels.functional.matmul.block import (
    map_pid_m_n_L2_optim,
    map_pid_m_n_L2_optim_ref,
)
from triton_kernels.functional.matmul.ref import matmul as matmul_ref

import torch
import math
import random
import pytest
import triton
import triton.language as tl

# from triton_kernels.functional.matmul import persistent_matmul as matmul

# from triton_kernels.functional.matmul import eager_matmul as matmul


def test_map_pid_L2_optimization():
    for i in range(100):
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M = tuple(
            random.randint(1, 512) for _ in range(5)
        )
        print(
            f"Testing map pid case {i}, {M=}, {N=}, {BLOCK_SIZE_M=}, {BLOCK_SIZE_N=}, {GROUP_SIZE_M=}"
        )
        # M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M = (10, 10, 2, 2, 2)
        num_pids = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
        for pid in range(num_pids):
            assert map_pid_m_n_L2_optim(
                pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
            ) == map_pid_m_n_L2_optim_ref(
                pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
            )



# test_map_pid_L2_optimization()


def test_matmul_1():
    DEVICE = get_device()
    torch.manual_seed(0)
    M, N, K = (128, 234, 128)
    a = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
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

dtypes = [torch.float32, torch.bfloat16, torch.float16]
pytest.mark.parametrize('dtype', dtypes, 'optimize_L2', [True, False])
def test_matmul(dtype: torch.dtype, optimize_L2: bool):
    print(f'Running test_matmul, {dtype=}, {optimize_L2=}')
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
            out = matmul(a, b, optimize_L2=optimize_L2)

            assert (out_ref - out).norm() / out_ref.norm() < 1e-6
            print((out_ref - out).norm() / out_ref.norm())

        ...


# test_matmul(dtype=torch.float32, optimize_L2=True)
