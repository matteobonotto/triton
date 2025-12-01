from triton_kernels.functional.matmul.utils import (
    map_pid_m_n_L2_optim,
    map_pid_m_n_L2_optim_ref,
)
import math
import random
import pytest
import triton.language as tl


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
