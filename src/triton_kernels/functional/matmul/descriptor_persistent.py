from typing import Optional
import torch
from torch import Tensor
import triton
import triton.language as tl
import math
from .utils import validate_inputs_matmul, map_pid_m_n


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
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
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

        tile_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.arange(0, tl.cdiv(K, BLOCK_SIZE_K)):

            tile_a = a_desc.load([offset_m, offset_k])
            tile_b = b_desc.load([offset_k, offset_n])

            tile_c = tl.dot(tile_a, tile_b, acc=tile_c)

        ### store tile_c back im mamory
        c_desc.store([offset_m, offset_n], tile_c)


def mamtul(a: Tensor, b: Tensor, DEBUG: bool = False) -> Tensor:
    validate_inputs_matmul(a, b)

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
        return torch.empty(size, device=a.device, dtyp=torch.int8)

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

    return c
