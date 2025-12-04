from torch import nn, Tensor
import torch
import triton
import triton.language as tl
from math import ceil

from ...utils import get_device
from .autotune import get_cuda_autotune_config
from .utils import validate_inputs_matmul, map_pid_m_n

DEVICE = get_device()


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
    # stride_am,
    # stride_ak,
    # stride_bk,
    # stride_bn,
    # stride_cm,
    # stride_cn,
    optimize_L2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    ### create tensor descriptors for a, b, c
    a_desc = tl.make_tensor_descriptor(
        a_ptr, 
        shape=[M, K], 
        strides=[K, 1], 
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )

    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N]
    )

    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    pid = tl.program_id(axis=0)
    pid_m, pid_n = map_pid_m_n(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
    )

    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # tl.assume(stride_am > 0)
    # tl.assume(stride_ak > 0)
    # tl.assume(stride_bn > 0)
    # tl.assume(stride_bk > 0)
    # tl.assume(stride_cm > 0)
    # tl.assume(stride_cn > 0)

    # offsets along dimensions m, n, k
    offset_ptr_m = pid_m * BLOCK_SIZE_M
    offset_ptr_n = pid_n * BLOCK_SIZE_N

    # get pointers for first tiles tile_a, tile_b
    # tile_a_ptr = a_ptr + (
    #     offset_ptr_m[:, None] * stride_am + offset_ptr_k[None, :] * stride_ak
    # )
    # tile_b_ptr = b_ptr + (
    #     offset_ptr_k[:, None] * stride_bk + offset_ptr_n[None, :] * stride_bn
    # )

    tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # loop over dimension k
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        offset_k = k * BLOCK_SIZE_K
        tile_a = a_desc.load([offset_ptr_m, offset_k])
        tile_b = b_desc.load([offset_k, offset_ptr_n])

        tile_c = tl.dot(tile_a, tile_b, acc=tile_c)

    # get c global pointers and store the output
    c_desc.store([offset_ptr_m, offset_ptr_n], tile_c)


def pad_tensor_16_byte_aligned(t: Tensor, axis: int) -> Tensor:
    assert t.ndim == 2, f"expected tensor to have exactly 2 dimensions, got {t.ndims}"
    old_dims = t.shape
    dim = old_dims[axis]
    padded_dim = dim + 16 - dim % 16
    new_dims = (padded_dim, t.shape[1]) if axis == 0 else (t.shape[0], padded_dim)
    new_t = torch.zeros(new_dims, dtype=t.dtype, device=t.device)
    new_t[:old_dims[0], :old_dims[1]] = t
    return new_t


def matmul(
    a: Tensor,
    b: Tensor,
    optimize_L2: bool = True,
):

    validate_inputs_matmul(a, b)

    M, K = a.shape
    _, N = b.shape
    old_N = N
    
    ### pad if needed (tensor_descriptor expecting stride(0) to 
    # be multiple of 16 -> K, N must be so)
    if K % 16 != 0:
        a = pad_tensor_16_byte_aligned(a, axis=1)
        b = pad_tensor_16_byte_aligned(b, axis=0)

    if N % 16 != 0:
        b = pad_tensor_16_byte_aligned(b, axis=1)
        old_N = N
        
    M, K = a.shape
    _, N = b.shape
    
    c = torch.zeros(M, N, device=DEVICE, dtype=a.dtype)

    # grid = lambda META: (ceil(M / META['BLOCK_SIZE_M']) * ceil(N / META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 64, 64, 64, 4
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 2, 2, 2, 4
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 2, 2, 2, 1
    grid = (ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N),)

    _block_matmul_triton[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        optimize_L2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )
    
    c = c[:, :old_N]
    return c if c.dtype == a.dtype else c.to(dtype=a.dtype)
