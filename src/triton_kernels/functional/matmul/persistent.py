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

    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation

    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n

    # this program will process more than a single tile of c
    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
        )
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

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
        # tile_c = tile_c.to(tl.bfloat16)

        # get the pointers of tile_c to store it in the global c tensor
        ptr_tile_c = (
            c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
        )
        mask_c = (offset_m < M)[:, None] & (offset_n < N)[None, :]
        tl.store(ptr_tile_c, tile_c, mask=mask_c)



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

    return c if a.dtype == c.dtype else c.to(a.dtype)
