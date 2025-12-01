import torch
from torch import Tensor
import triton
import triton.language as tl
import math
from .utils import validate_inputs_matmul, map_pid_m_n


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


@triton.jit()
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


@triton.jit()
def _matmul_persistent_triton(
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
    NUM_SMS: tl.constexpr,
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

    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n

    # this program will process more than a single tile of c
    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
        )

        start_m = BLOCK_SIZE_M * pid_m
        start_n = BLOCK_SIZE_N * pid_n

        offset_m = (start_m + tl.arange(0, BLOCK_SIZE_M)) % M
        offset_n = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
        offset_k = tl.arange(0, BLOCK_SIZE_K)

        ptr_tile_a = (
            a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        )
        ptr_tile_b = (
            b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        )

        tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # compute the tile related to this tile_id
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Example: BLOCK_SIZE_K = 4, offset_k = [0,1,2,3]. If K = 6, the
            # first block is loaded entirely, so the mask will be 4*[True], while
            # for the second k block, only the first 2 indices must be loaded:
            # this way k_reminder = K = 1 * BLOCK_SIZE_K = 2, and so
            # mask = [0,1,2 3] < 2 = [True, True, False, False], only the first 2
            # elements are loaded
            k_remaining = K - k * BLOCK_SIZE_K
            mask = offset_k < k_remaining

            tile_a = tl.load(ptr_tile_a, mask=mask[None, :], other=0.0)
            tile_b = tl.load(ptr_tile_b, mask=mask[:, None], other=0.0)
            tile_c = tl.dot(tile_a, tile_b, acc=tile_c)

            # advance the k pointers along k directions of BLOCK_SIZE_K steps
            ptr_tile_a += BLOCK_SIZE_K * stride_ak  # )[None, :]
            ptr_tile_b += BLOCK_SIZE_K * stride_bk  # )[:, None]

        # store tile_c
        tile_c = tile_c.to(tl.bfloat16)

        # get the pointers of tile_c to store it in the global c tensor
        ptr_tile_c = (
            c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
        )
        mask_c = (offset_m < M)[:, None] & (offset_n < N)[None, :]
        tl.store(ptr_tile_c, tile_c, mask=mask_c)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    optimize_L2,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    total_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    for tile_id in tl.range(pid, total_tiles, step=NUM_SMS):

        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
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
            c_ptr
            + offset_ptr_m[:, None] * stride_cm
            + offset_ptr_n[None, :] * stride_cn
        )
        mask_c = (offset_ptr_m[:, None] < M) & (offset_ptr_n[None, :] < N)
        tl.store(tile_c_ptr, tile_c, mask=mask_c)


def matmul(
    a: Tensor, b: Tensor, optimize_L2: bool = True, DEBUG: bool = False
) -> Tensor:

    validate_inputs_matmul(a, b)
    M, K = a.shape
    _, N = b.shape

    # number of streaming multiprocessors of the gpu
    if DEBUG:
        # for dev with "TRITON_INTERPRET" = "1" and no gpu
        NUM_SMS = 4000
    else:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 64, 64, 64, 4

    product_tiles = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    grid = (min(product_tiles, NUM_SMS),)

    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    _matmul_persistent_triton[grid](
        # matmul_kernel_persistent[grid](
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
        NUM_SMS,
    )

    return c
