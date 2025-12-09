from typing import Optional
import triton
import triton.language as tl
import torch
from torch import Tensor
import math

from .act import _act_fwd, _act_bwd
from .utils import map_pid_m_n


@triton.jit()
def _fwd_kernel(
    x_ptr,
    W_up_ptr,
    W_gp_ptr,
    out_ptr,
    #
    x_stride_m,
    x_stride_k,
    W_up_stride_k,
    W_up_stride_n,
    W_gp_stride_k,
    W_gp_stride_n,
    out_stride_m,
    out_stride_n,
    #
    act_fn: tl.constexpr,
    dropout_p,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    pid_m, pid_n = map_pid_m_n(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, True
    )
    # pid = tl.program_id(axis=0)
    # num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_programs_k = tl.cdiv(K, BLOCK_SIZE_K)
    # total_programs = num_programs_m * num_programs_n
    # tile_id = pid
    # pid_m, pid_n = map_pid_m_n(
    #     tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, True
    # )

    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(x_stride_m > 0)
    tl.assume(x_stride_k > 0)
    tl.assume(W_up_stride_k > 0)
    tl.assume(W_up_stride_n > 0)
    tl.assume(W_gp_stride_k > 0)
    tl.assume(W_gp_stride_n > 0)
    tl.assume(out_stride_m > 0)
    tl.assume(out_stride_n > 0)

    # offsets are 2d
    offset_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    # 2d pointers
    tile_x_ptr = x_ptr + offset_m[:, None] * x_stride_m + offset_k[None, :] * x_stride_k
    tile_W_up_ptr = (
        W_up_ptr + offset_k[:, None] * W_up_stride_k + offset_n[None, :] * W_up_stride_n
    )
    tile_W_gp_ptr = (
        W_gp_ptr + offset_k[:, None] * W_gp_stride_k + offset_n[None, :] * W_gp_stride_n
    )

    tile_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    tile_gp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        remaining = K - k * BLOCK_SIZE_K
        mask_k = offset_k < remaining

        tile_x = tl.load(tile_x_ptr, mask=mask_k[None, :], other=0.0)
        tile_W_up = tl.load(tile_W_up_ptr, mask=mask_k[:, None], other=0.0)

        tile_up = tl.dot(tile_x, tile_W_up, acc=tile_up)
        # if has_bias_up:
        #     tile_up += ...

        tile_W_gp = tl.load(tile_W_gp_ptr, mask=mask_k[:, None], other=0.0)
        tile_gp = tl.dot(tile_x, tile_W_gp, acc=tile_gp)
        # if has_bias_gp:
        #     tile_gp += ...

        ### advance pointers along k direction
        tile_x_ptr += BLOCK_SIZE_K * x_stride_k
        tile_W_up_ptr += BLOCK_SIZE_K * W_up_stride_k
        tile_W_gp_ptr += BLOCK_SIZE_K * W_gp_stride_k

    ### compute act(tile_gp) * tile_up (skip dropout for now)
    tile_out = _act_fwd(tile_gp, act_fn) * tile_up

    ### store back on memory
    mask_out = (offset_m < M)[:, None] & (offset_n < N)[None, :]
    tile_out_ptr = (
        out_ptr + offset_m[:, None] * out_stride_m + offset_n[None, :] * out_stride_n
    )
    tl.store(tile_out_ptr, tile_out, mask=mask_out)


def validate_dimensions(
    x: Tensor,
    WT_up: Tensor,  # this one is transposed
    b_up: Tensor | None,
    WT_gp: Tensor,  # this one is transposed
    b_gp: Tensor | None,
) -> None:
    assert x.ndim <= 2, f"input tensor must have ndims <=2, got {x.ndim}"
    assert WT_up.shape[1] == x.shape[1], "dimension mismatch in WT_up or x"
    assert WT_gp.shape == WT_up.shape, "dimension mismatch in WT_up or WT_gp"

    if b_up is not None:
        assert b_up.shape[0] == WT_up.shape[0], "dimension mismatch in b_up"

    if b_gp is not None:
        assert b_gp.shape[0] == WT_up.shape[0], "dimension mismatch in b_gp"


def pad_tensor_16_byte_aligned(t: Tensor, axis: int) -> Tensor:
    assert t.ndim == 2, f"expected tensor to have exactly 2 dimensions, got {t.ndims}"
    old_dims = t.shape
    dim = old_dims[axis]
    padded_dim = dim + 16 - dim % 16
    new_dims = (padded_dim, t.shape[1]) if axis == 0 else (t.shape[0], padded_dim)
    new_t = torch.zeros(new_dims, dtype=t.dtype, device=t.device)
    new_t[: old_dims[0], : old_dims[1]] = t
    return new_t


def get_num_streaming_multiprocessors() -> int:
    return (
        10  # dummy value for dev/debugging
        if not torch.cuda.is_available()
        else torch.cuda.get_device_properties("cuda:0").multi_processor_count
    )


def mlp_hidden_states_fwd(
    x: Tensor,
    WT_up: Tensor,  # this one is transposed
    b_up: Tensor | None,
    WT_gp: Tensor,  # this one is transposed
    b_gp: Tensor | None,
    act_fn: str,
    dropout_p: float,
) -> Tensor:
    """
    This function computes the follwing operations in a fused fashion:

        self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    Wieghts WT_up and WT_gp are kept transposed, and they are transposed back inside the
    triton kernel when doing tl.dot(x, W.T, ...)
    """

    ### validate input dimension
    validate_dimensions(x, WT_up, b_up, WT_gp, b_gp)

    M, K = x.shape
    N, _ = WT_up.shape

    stride_W_up = (WT_up.stride(1), WT_up.stride(0))
    stride_W_gp = (WT_gp.stride(1), WT_gp.stride(0))

    ### triton tensor_descriptor needs tensors to have stride(0) that
    # is a multiple of 16. Check this and pad if needed.
    if K % 16 != 0:
        raise NotImplementedError()
        # a = pad_tensor_16_byte_aligned(a, axis=1)
        # b = pad_tensor_16_byte_aligned(b, axis=0)

    if N % 16 != 0:
        raise NotImplementedError()
        # b = pad_tensor_16_byte_aligned(b, axis=1)
        # old_N = N

    ### Create the grid
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (8, 8, 8, 8)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)
    grid = (math.ceil(N / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N),)

    ### allocate output and run the kernel
    out = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    has_b_gp = b_gp is not None
    has_b_up = b_up is not None
    _fwd_kernel[grid](
        x,
        WT_up,
        # b_up,
        WT_gp,
        # b_gp,
        out,
        x.stride(0),
        x.stride(1),
        WT_up.stride(1),  # this will transpose W_up
        WT_up.stride(0),  # this will transpose W_up
        # b_up.stride(0),
        # b_up.stride(1),
        WT_gp.stride(1),  # this will transpose W_gp
        WT_gp.stride(0),  # this will transpose W_gp
        # b_gp.stride(0),
        # b_gp.stride(1),
        out.stride(0),
        out.stride(1),
        act_fn,
        dropout_p,
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )

    ###
    return out.to(x.dtype) if x.dtype != out.dtype else out
