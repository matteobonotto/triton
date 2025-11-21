import torch
from torch import tensor
import triton
import triton.language as tl

from math import ceil


@triton.jit()
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    ### map the global pid following the super-grouping with value GROUP_SIZE_M
    num_groups_m = tl.cdiv(M, GROUP_SIZE_M)
    num_groups_n = tl.cdiv(N, GROUP_SIZE_M)

    offset_m = (pid // (GROUP_SIZE_M * GROUP_SIZE_M)) * GROUP_SIZE_M
    offset_n = (pid // (GROUP_SIZE_M * GROUP_SIZE_M * num_groups_m)) * GROUP_SIZE_M
    pid_m = (pid % GROUP_SIZE_M) + offset_m % M
    pid_n = ((pid // GROUP_SIZE_M) - offset_m) + offset_n

    print(f"{int(pid)=}, {int(pid_m)=}, {int(pid_n)=}")
    ...

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    print(f"{int(pid)=}, {int(pid_m)=}, {int(pid_n)=}")
    print(" ")
    ...
    ### some assumption to help the compiler
    # tl.assume(pid_m >= 0)
    # tl.assume(pid_n >= 0)
    # tl.assume(stride_am > 0)
    # tl.assume(stride_ak > 0)
    # tl.assume(stride_bn > 0)
    # tl.assume(stride_bk > 0)
    # tl.assume(stride_cm > 0)
    # tl.assume(stride_cn > 0)

    # tile_c = tl.zeros((M,N), dtype=tl.float32)


def matmul(a, b):

    M, K = a.shape
    _, N = b.shape

    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (1, 1, 1, 2)
    # grid = lambda META: (ceil(M, META['BLOCK_SIZE_M']) * ceil(N, META['BLOCK_SIZE_N']), )
    grid = (ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )

    return c
