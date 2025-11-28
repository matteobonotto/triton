import torch
from torch import Tensor
import triton
import triton.language as tl
import math
from .utils import validate_inputs_matmul


@triton.jit()
def _matmul_persistent_triton():
    ...

def matmul_persistent(
    a: Tensor,
    b: Tensor,
    DEBUG: bool = False
) -> Tensor:

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
    grid = min(product_tiles, NUM_SMS), 
    
    c = torch.zeros_like(a, device=a.device, dtype=a.dtype)
    
    _matmul_persistent_triton[grid]()
    
    return c
