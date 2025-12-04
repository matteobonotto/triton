from typing import Optional
import torch
from torch import Tensor
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor
from .utils import validate_inputs_matmul, map_pid_m_n


@triton.jit()
def maybe_pad_16_byte_aligned(n):
    if n % 16 != 0:
        return n + 16 - n % 16
    return n

@triton.jit()
def _matmul_persistent_descriptor_triton(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
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

    ### run mamtul loop on multiple tiles
    pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n
    

    for tile_id in tl.range(pid, num_tiles, NUM_SMS):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2=True
        )
        
        ### with triton tensor_descriptor the offsets are scalar and not 2D like  
        # when using tl.load
        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

        tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            
            offset_k = k * BLOCK_SIZE_K

            tile_a = a_desc.load([offset_m, offset_k])
            tile_b = b_desc.load([offset_k, offset_n])

            tile_c = tl.dot(tile_a, tile_b, acc=tile_c)

        ### store tile_c back im mamory
        c_desc.store([offset_m, offset_n], tile_c)

def pad_tensor_16_byte_aligned(t: Tensor, axis: int) -> Tensor:
    assert t.ndim == 2, f"expected tensor to have exactly 2 dimensions, got {t.ndims}"
    old_dims = t.shape
    dim = old_dims[axis]
    padded_dim = dim + 16 - dim % 16
    new_dims = (padded_dim, t.shape[1]) if axis == 0 else (t.shape[0], padded_dim)
    new_t = torch.zeros(new_dims, dtype=t.dtype, device=t.device)
    new_t[:old_dims[0], :old_dims[1]] = t
    return new_t


def matmul(a: Tensor, b: Tensor, DEBUG: bool = False) -> Tensor:
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
    
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    NUM_SMS = (
        10
        if DEBUG
        else torch.cuda.get_device_properties("cuda:0").multi_processor_count
    )
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)
    grid = (min(NUM_SMS, math.ceil(N / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),)

    # tma descriptors need a global allocation function
    def allocation_fun(size: int, allignment: int, stream: Optional[int]):
        return torch.empty(size, device=a.device, dtype=torch.float32)
    triton.set_allocator(allocation_fun)

    # dummy_block = [1, 1]
    # a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    # b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    # c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    _matmul_persistent_descriptor_triton[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        NUM_SMS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )

    c = c[:, :old_N]
    return c if c.dtype == a.dtype else c.to(dtype=a.dtype)
