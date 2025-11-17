from torch import nn, Tensor
import torch
import triton
import triton.language as tl
from math import ceil
from ..utils import get_device

DEVICE = get_device()


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
        stride_cn,
        stride_cm,
        block_size_M: tl.constexpr,
        block_size_N: tl.constexpr,
        block_size_K: tl.constexpr,
        # group_size_M: tl.constexpr,
    ): 
    
    pid = tl.program_id(axis=0)
    
    #
    n_programs_n = tl.cdiv(N, block_size_N)
    
    # map global pid to (pid_m, pid_n)
    pid_m = pid // n_programs_n
    pid_n = pid % n_programs_n
    
    # offsets along dimensions m, n, k
    offset_ptr_m = pid_m * block_size_M + tl.arange(0, block_size_M)
    offset_ptr_n = pid_n * block_size_N + tl.arange(0, block_size_N)
    offset_ptr_k = tl.arange(0, block_size_K)
    
    # get pointers for first tiles tile_a, tile_b
    tile_a_ptr = a_ptr + offset_ptr_m[:, None] * stride_am + offset_ptr_k[None, :] * stride_ak
    tile_b_ptr = b_ptr + offset_ptr_k[:, None] * stride_bk + offset_ptr_n[None, :] * stride_bn
    
    tile_c = tl.zeros((block_size_M, block_size_N), dtype=tl.float32)
    
    # loop over dimension k
    for k in tl.range(0, tl.cdiv(K, block_size_K)):
        
        # if K % k != 0 -> mask out the items after K
        items_after_K = K - k * block_size_K
        
        # materialize the masks along the Z dimenison for tile_a, tile_b
        # mask_ak = offset_ptr_k[None, :] < items_after_K # mask along 1st dim
        # mask_bk = offset_ptr_k[:, None] < items_after_K # mask along 2nd dim
        
        # Create masks that check BOTH dimensions
        # For A: Check M (rows) AND K (cols)
        mask_ak = (offset_ptr_m[:, None] < M) & (offset_ptr_k[None, :] < items_after_K)

        # # For B: Check K (rows) AND N (cols)
        mask_bk = (offset_ptr_k[:, None] < items_after_K) & (offset_ptr_n[None, :] < N)
        
        tile_a = tl.load(tile_a_ptr, mask_ak, other=0.0)
        tile_b = tl.load(tile_b_ptr, mask_bk, other=0.0)

        tile_c = tl.dot(tile_a, tile_b, acc=tile_c)
        
        # now slide the tile_a, tile_b pointers along Z direction of k steps
        tile_a_ptr += block_size_K * stride_ak
        tile_b_ptr += block_size_K * stride_bk
        
    # tile_c = tile_c.to(tl.bfloat16)
    
    # get c global pointers and store the output
    tile_c_ptr = c_ptr + offset_ptr_m[:, None] * stride_cm + offset_ptr_n[None, :] * stride_cn
    mask_c = (offset_ptr_m[:, None] < M) & (offset_ptr_n[None, :] < N)
    tl.store(tile_c_ptr, tile_c, mask=mask_c)
    
    
# @triton.jit()
# def _eager_matmul_triton_2(
#         a_ptr,
#         b_ptr,
#         c_ptr,
#         M,
#         N,
#         K,
#         stride_am,
#         stride_ak,
#         stride_bk,
#         stride_bn,
#         stride_cn,
#         stride_cm,
#         block_size_M: tl.constexpr,
#         block_size_N: tl.constexpr,
#         block_size_K: tl.constexpr,
#     ): 
    
#     pid = tl.program_id(axis=0)
    
#     n_programs_n = tl.cdiv(N, block_size_N)
    
#     pid_m = pid // n_programs_n
#     pid_n = pid % n_programs_n
    
#     # offsets along dimensions m, n, k
#     offset_ptr_m = pid_m * block_size_M + tl.arange(0, block_size_M)
#     offset_ptr_n = pid_n * block_size_N + tl.arange(0, block_size_N)
#     offset_ptr_k = tl.arange(0, block_size_K)
    
#     tile_a_ptr = a_ptr + offset_ptr_m[:, None] * stride_am + offset_ptr_k[None, :] * stride_ak
#     tile_b_ptr = b_ptr + offset_ptr_k[:, None] * stride_bk + offset_ptr_n[None, :] * stride_bn
    
#     tile_c = tl.zeros((block_size_M, block_size_N), dtype=tl.float32)
    
#     # loop over dimension k
#     for k in tl.range(0, tl.cdiv(K, block_size_K)):
        
#         items_after_K = K - k * block_size_K
        
#         # --- FIX STARTS HERE ---
#         # Masking needs to handle M and N boundaries as well as K
#         mask_ak = (offset_ptr_m[:, None] < M) & (offset_ptr_k[None, :] < items_after_K)
#         mask_bk = (offset_ptr_k[:, None] < items_after_K) & (offset_ptr_n[None, :] < N)
#         # --- FIX ENDS HERE ---
        
#         tile_a = tl.load(tile_a_ptr, mask=mask_ak, other=0.0)
#         tile_b = tl.load(tile_b_ptr, mask=mask_bk, other=0.0)

#         tile_c = tl.dot(tile_a, tile_b, acc=tile_c)
        
#         tile_a_ptr += block_size_K * stride_ak
#         tile_b_ptr += block_size_K * stride_bk
        
#     # Careful with this hardcoded cast
#     # tile_c = tile_c.to(tl.bfloat16) 
    
#     tile_c_ptr = c_ptr + offset_ptr_m[:, None] * stride_cm + offset_ptr_n[None, :] * stride_cn
#     mask_c = (offset_ptr_m[:, None] < M) & (offset_ptr_n[None, :] < N)
#     tl.store(tile_c_ptr, tile_c, mask=mask_c)
    
    
# @triton.jit
# def matmul_kernel_naive(
#     # Pointers to matrices
#     a_ptr, b_ptr, c_ptr,
#     # Matrix dimensions
#     M, N, K,
#     # Strides (in elements)
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     # Meta-parameters
#     block_size_M: tl.constexpr,
#     block_size_N: tl.constexpr,
#     block_size_K: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)

#     n_programs_m = tl.cdiv(M, block_size_M)
#     n_programs_n = tl.cdiv(N, block_size_N)

#     # correct mapping
#     pid_m = pid // n_programs_n
#     pid_n = pid %  n_programs_n

#     offs_m = pid_m * block_size_M + tl.arange(0, block_size_M)
#     offs_n = pid_n * block_size_N + tl.arange(0, block_size_N)
#     offs_k = tl.arange(0, block_size_K)

#     a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
#     b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

#     acc = tl.zeros((block_size_M, block_size_N), dtype=tl.float32)

#     # correct outer K-loop
#     for k in range(0, tl.cdiv(K, block_size_K)):
#         k_global = k * block_size_K + offs_k

#         mask_ak = k_global[None, :] < K
#         mask_bk = k_global[:, None] < K

#         a = tl.load(a_ptrs, mask=mask_ak, other=0.)
#         b = tl.load(b_ptrs, mask=mask_bk, other=0.)

#         acc = tl.dot(a, b, acc)

#         a_ptrs += block_size_K * stride_ak
#         b_ptrs += block_size_K * stride_bk

#     acc = acc.to(tl.bfloat16)

#     c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#     mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
#     tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        # ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    

def matmul(a: Tensor, b: Tensor):
    assert a.ndim == 2, "expected A to have ndim=2"
    assert b.ndim == 2, "expected B to have ndim=2"
    assert a.shape[1] == b.shape[0], "dimension mismatch"

    M, K = a.shape
    N = b.shape[1]

    c = torch.zeros(M, N).to(DEVICE)

    # these will be tuned via autotune and provided as metaparameters
    block_size_M, block_size_N, block_size_K = (64, 64, 64)
    # block_size_M, block_size_N, block_size_K = (2, 2, 2)
    group_size_M = 8

    
    grid = (ceil(M / block_size_M) * ceil(N / block_size_N), )

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
