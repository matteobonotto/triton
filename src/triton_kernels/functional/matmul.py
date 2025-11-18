from torch import nn, Tensor
import torch
import triton
import triton.language as tl
from math import ceil
from ..utils import get_device

DEVICE = get_device()


# @triton.jit
# def naive_matmul_kernel(
#         # Pointers to matrices
#         a_ptr, b_ptr, c_ptr,
#         # Matrix dimensions
#         M, N, K,
#         # The stride variables
#         stride_am, stride_ak,
#         stride_bk, stride_bn,
#         stride_cm, stride_cn,
#         # Meta-parameters
#         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#         # GROUP_SIZE_M parameter is removed because we aren't grouping
# ):
#     """
#     Naive Blocked Matrix Multiplication Kernel.
#     Removes the 'super-grouping' (L2 Cache Optimization) logic.
#     """
    
#     # -----------------------------------------------------------
#     # 1. Program ID and Grid Mapping (The Logic Change)
#     # -----------------------------------------------------------
    
#     # Get the unique program ID (1D)
#     pid = tl.program_id(axis=0)
    
#     # Calculate number of blocks along the N dimension
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     # Naive Row-Major Mapping:
#     # We strictly fill one row of blocks (N dimension) before moving to the next row (M dimension).
#     # This removes the L2 cache locality optimization (swizzling).
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     # -----------------------------------------------------------
#     # 2. Pointer Arithmetic (Same as original)
#     # -----------------------------------------------------------
    
#     # Create pointers for the first blocks of A and B.
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
    
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     # -----------------------------------------------------------
#     # 3. Computation Loop (Same as original)
#     # -----------------------------------------------------------
    
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Load blocks of A and B. 
#         # We apply masking on the K dimension to handle cases where K is not a multiple of BLOCK_SIZE_K
#         a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
#         # Accumulate
#         accumulator = tl.dot(a, b, accumulator)
        
#         # Advance pointers
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     c = accumulator.to(tl.float16)

#     # -----------------------------------------------------------
#     # 4. Store Result (Same as original)
#     # -----------------------------------------------------------
    
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)
    
    
    
    
@triton.jit()
def _eager_matmul_triton(
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
    # group_size_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    #
    n_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    # map global pid to (pid_m, pid_n)
    pid_m = pid // n_programs_n
    pid_n = pid % n_programs_n

    # offsets along dimensions m, n, k
    offset_ptr_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_ptr_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_ptr_k = tl.arange(0, BLOCK_SIZE_K)

    # get pointers for first tiles tile_a, tile_b
    tile_a_ptr = (
        a_ptr + (offset_ptr_m[:, None] * stride_am + offset_ptr_k[None, :] * stride_ak)
    )
    tile_b_ptr = (
        b_ptr + (offset_ptr_k[:, None] * stride_bk + offset_ptr_n[None, :] * stride_bn)
    )

    tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # loop over dimension k
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # if K % k != 0 -> mask out the items after K
        items_after_K = K - k * BLOCK_SIZE_K

        # materialize the masks along the Z dimenison for tile_a, tile_b
        # mask_ak = offset_ptr_k[None, :] < items_after_K # mask along 1st dim
        # mask_bk = offset_ptr_k[:, None] < items_after_K # mask along 2nd dim

        # Create masks that check BOTH dimensions
        # For A: Check M (rows) AND K (cols)
        mask_ak = (offset_ptr_k[None, :] < items_after_K)

        # # For B: Check K (rows) AND N (cols)
        mask_bk = (offset_ptr_k[:, None] < items_after_K)
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


def matmul(a: Tensor, b: Tensor):
    assert a.ndim == 2, "expected A to have ndim=2"
    assert b.ndim == 2, "expected B to have ndim=2"
    assert a.shape[1] == b.shape[0], "dimension mismatch"

    M, K = a.shape
    N = b.shape[1]

    c = torch.zeros(M, N).to(DEVICE)

    # these will be tuned via autotune and provided as metaparameters
    block_size_M, block_size_N, block_size_K = (128, 128, 64)
    # block_size_M, block_size_N, block_size_K = (8, 8, 4)
    group_size_M = 8
    # block_size_M, block_size_N, block_size_K = (2, 2, 2)

    grid = (ceil(M / block_size_M) * ceil(N / block_size_N),)

    _eager_matmul_triton[grid](
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
        block_size_M,
        block_size_N,
        block_size_K,
        # group_size_M,
    )

    return c
