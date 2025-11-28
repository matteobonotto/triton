from torch import nn, Tensor
import torch
import triton
import triton.language as tl
from math import ceil

from ...utils import get_device
from .autotune import get_cuda_autotune_config
from .utils import validate_inputs_matmul

DEVICE = get_device()


@triton.git()
def map_pid_m_n_L2_optim(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):

    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_in_group = GROUP_SIZE_M * num_blocks_n

    group_size_m = GROUP_SIZE_M
    offset_m = pid // num_programs_in_group
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - offset_m * GROUP_SIZE_M)

    pid_m = ((pid % num_programs_in_group) % group_size_m) + offset_m * GROUP_SIZE_M
    pid_n = (pid % num_programs_in_group) // group_size_m

    return (pid_m, pid_n)


def map_pid_m_n_L2_optim_ref(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # # print(f"{int(pid)=}, {int(pid_m)=}, {int(pid_n)=}")
    ...

    return (pid_m, pid_n)


@triton.jit()
def map_pid_m_n(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2):
    if optimize_L2:
        pid_m, pid_n = map_pid_m_n_L2_optim(
            pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )
    else:
        n_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // n_programs_n
        pid_n = pid % n_programs_n
    return (pid_m, pid_n)


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["M", "N", "K"],
# )
@triton.jit()
def _block_matmul_triton(
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
    optimize_L2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    pid_m, pid_n = map_pid_m_n(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
    )

    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # offsets along dimensions m, n, k
    offset_ptr_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_ptr_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_ptr_k = tl.arange(0, BLOCK_SIZE_K)

    # get pointers for first tiles tile_a, tile_b
    tile_a_ptr = a_ptr + (
        offset_ptr_m[:, None] * stride_am + offset_ptr_k[None, :] * stride_ak
    )
    tile_b_ptr = b_ptr + (
        offset_ptr_k[:, None] * stride_bk + offset_ptr_n[None, :] * stride_bn
    )

    tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # loop over dimension k
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # if K % k != 0 -> mask out the items after K
        items_after_K = K - k * BLOCK_SIZE_K

        # Create masks that check BOTH dimensions
        # For A: Check M (rows) AND K (cols)
        mask_ak = offset_ptr_k[None, :] < items_after_K

        # # For B: Check K (rows) AND N (cols)
        mask_bk = offset_ptr_k[:, None] < items_after_K
        # mask_bk = (offset_ptr_n[None, :] < N)

        tile_a = tl.load(tile_a_ptr, mask_ak, other=0.0)
        tile_b = tl.load(tile_b_ptr, mask_bk, other=0.0)

        tile_c = tl.dot(tile_a, tile_b, acc=tile_c)

        # now slide the tile_a, tile_b pointers along Z direction of BLOCK_SIZE_K steps
        tile_a_ptr += BLOCK_SIZE_K * stride_ak
        tile_b_ptr += BLOCK_SIZE_K * stride_bk

    # tile_c = tile_c.to(tl.bfloat16)

    # get c global pointers and store the output
    tile_c_ptr = (
        c_ptr + offset_ptr_m[:, None] * stride_cm + offset_ptr_n[None, :] * stride_cn
    )
    mask_c = (offset_ptr_m[:, None] < M) & (offset_ptr_n[None, :] < N)
    tl.store(tile_c_ptr, tile_c, mask=mask_c)


def matmul(
    a: Tensor,
    b: Tensor,
    optimize_L2: bool = True,
):

    validate_inputs_matmul(a, b)

    M, K = a.shape
    N = b.shape[1]

    c = torch.zeros(M, N, device=DEVICE, dtype=a.dtype)

    # grid = lambda META: (ceil(M / META['BLOCK_SIZE_M']) * ceil(N / META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 64, 64, 64, 4
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 2, 2, 2, 4
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 1, 1, 1, 2
    grid = (ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N),)

    _block_matmul_triton[grid](
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
        optimize_L2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )

    return c if c.dtype == a.dtype else c.to(dtype=a.dtype)
