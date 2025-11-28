import torch
from torch import Tensor
import triton
import triton.language as tl
import math
from .utils import validate_inputs_matmul, map_pid_m_n_L2_optim


@triton.jit()
def _matmul_persistent_triton(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
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
    NUM_SMS: tl.constexpr,
): 
    pid = tl.program_id(axis=0)
    
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    
    # this program will process more than a single tile of c
    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        pid_m, pid_n = map_pid_m_n_L2_optim(tile_id)
        
        offset_m = ...
        offset_n = ...
        
        tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # compute the tile related to this tile_id 
        for k  in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            tile_a = tl.load(...)
            tile_b = tl.load(...)
            mask = ...
            tile_c = tl.dot(..., acc=tile_c)
            
        # store tile_c


def matmul_persistent(a: Tensor, b: Tensor, DEBUG: bool = False) -> Tensor:

    validate_inputs_matmul(a, b)
    M, K = a.shape
    _, N = b.shape

    # number of streaming multiprocessors of the gpu
    if DEBUG:
        # for dev with "TRITON_INTERPRET" = "1" and no gpu
        NUM_SMS = 40
    else:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 64, 64, 64, 32

    product_tiles = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    grid = (min(product_tiles, NUM_SMS),)

    c = torch.zeros_like(a, device=a.device, dtype=a.dtype)

    _matmul_persistent_triton[grid](
        a,
        b,
        c,
        M, N, K, 
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
        NUM_SMS,
    )

    return c
