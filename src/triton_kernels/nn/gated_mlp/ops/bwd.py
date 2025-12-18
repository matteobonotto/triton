import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple, Optional
import math

from .utils import get_num_streaming_multiprocessors, map_pid_m_n
from .act import _act_bwd, _act_fwd


@triton.jit()
def _compute_quantities_for_bwd(
    x_ptr,
    W_up_ptr,
    W_gp_ptr,
    grad_output_ptr,
    act_prime_ptr,
    grad_output_a_act_prime_ptr,
    grad_output_c_ptr,
    act_fun: tl.constexpr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing:

        a = x @ W_up.T
        b = x @ W_gp.T
        c = act_fwd(b)

        sigma = torch.sigmoid(b)
        act_prime = sigma * (1 + b * (1 - sigma))

        grad_output_a_act_prime = (grad_output * a) * act_prime
        grad_output_c = grad_output * c
    """
    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_total_programs = num_programs_m * num_programs_n

    ### create tensor desctiptors (inputs)
    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    W_up_desc = tl.make_tensor_descriptor(
        W_up_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    W_gp_desc = tl.make_tensor_descriptor(
        W_gp_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    grad_output_desc = tl.make_tensor_descriptor(
        grad_output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    ### create tensor desctiptors (outputs)
    act_prime_desc = tl.make_tensor_descriptor(
        act_prime_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_output_a_act_prime_desc = tl.make_tensor_descriptor(
        grad_output_a_act_prime_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_output_c_desc = tl.make_tensor_descriptor(
        grad_output_c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    ### loop along M
    for tile_id in tl.range(pid, num_total_programs, NUM_SMS):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, True
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

        tile_a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_b = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        ### loop along K
        for offset_k in tl.range(0, K, BLOCK_SIZE_K):
            # NOTE: masking along k is done under the hood by tensor descriptors
            # offset_k = k * BLOCK_SIZE_K

            tile_x = x_desc.load([offset_m, offset_k])
            tile_W_up = W_up_desc.load([offset_n, offset_k])
            tile_W_gp = W_gp_desc.load([offset_n, offset_k])

            tile_a = tl.dot(tile_x, tile_W_up.T, acc=tile_a)
            tile_b = tl.dot(tile_x, tile_W_gp.T, acc=tile_b)

        tile_c = _act_fwd(tile_b, act_fun)
        tile_act_prime = _act_bwd(tile_b, act_fun)

        # grad_output_a_act_prime = (grad_output * a) * act_prime
        tile_grad_output = grad_output_desc.load([offset_m, offset_n])
        tile_grad_output_a_act_prime = (tile_grad_output * tile_a) * tile_act_prime

        # grad_output_c = grad_output * c
        tile_grad_output_c = tile_grad_output * tile_c

        ### store results back in memory
        offsets = [offset_m, offset_n]
        act_prime_desc.store(offsets=offsets, value=tile_act_prime)
        grad_output_a_act_prime_desc.store(
            offsets=offsets, value=tile_grad_output_a_act_prime
        )
        grad_output_c_desc.store(offsets=offsets, value=tile_grad_output_c)


@triton.jit()
def _compute_dx(
    grad_output_c_ptr,
    W_up_ptr,
    grad_output_a_act_prime_ptr,
    W_gp_ptr,
    dx_ptr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    ### tensor descriptors of inputs
    grad_output_c_desc = tl.make_tensor_descriptor(
        grad_output_c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    W_up_desc = tl.make_tensor_descriptor(
        W_up_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )
    grad_output_a_act_prime_desc = tl.make_tensor_descriptor(
        grad_output_a_act_prime_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    W_gp_desc = tl.make_tensor_descriptor(
        W_gp_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )

    ### tensor descriptor of output
    dx_desc = tl.make_tensor_descriptor(
        dx_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )

    for tile_id in tl.range(...):
        ...

        for offset_k in tl.range(0, K, BLOCK_SIZE_K):

            ...


def mlp_hidden_states_bwd(
    x: Tensor, W_up: Tensor, W_gp: Tensor, grad_output: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    act_fun = "silu"  # hard-coded for now, parametrize later

    M, K = x.shape
    N, _ = W_gp.shape
    # M = B * M # aready reshaped (B, M, K) -> (B * M, K)

    kwargs = {"device": x.device, "dtype": x.dtype}

    act_prime = torch.zeros((M, N), **kwargs)  # check dtype
    grad_output_a_act_prime = torch.zeros((M, N), **kwargs)  # check dtype
    grad_output_c = torch.zeros((M, N), **kwargs)  # check dtype

    NUM_SMS = get_num_streaming_multiprocessors()
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)

    ### provide the allocation function
    def allocation_fn(shape: int, stream: int, alligmnet: Optional[int]):
        return torch.empty(shape, device=x.device, dtype=torch.int8)

    triton.set_allocator(allocation_fn)

    grid = (min(NUM_SMS, math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),)

    ### compute quantities for bwd
    _compute_quantities_for_bwd[grid](
        x,
        W_up,
        W_gp,
        grad_output,
        act_prime,
        grad_output_a_act_prime,
        grad_output_c,
        act_fun,
        M,
        N,
        K,
        NUM_SMS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )

    """
    # steps to check the results
    a = x @ W_up.T
    b = x @ W_gp.T
    c = ACT_FWD[act_fn](b)
    sigma = torch.sigmoid(b)

    act_prime_ref = sigma * (1 + b * (1 - sigma))
    grad_output_a_act_prime_ref = (grad_output * a) * act_prime
    grad_output_c_ref = grad_output * c

    atol = 1e-3 if "TRITON_INTERPRET" in os.environ.keys() else 1e-7
    triton.testing.assert_close(act_prime, act_prime_ref, atol)
    triton.testing.assert_close(grad_output_a_act_prime, grad_output_a_act_prime_ref, atol)
    triton.testing.assert_close(grad_output_c, grad_output_c_ref, atol)
    """

    ### compute derivatives
    """
    ### dx
    dx_1 = grad_output_c @ W_up
    dx_2 = grad_output_a_act_prime @ W_gp
    dx = dx_1 + dx_2

    ### dW_up
    dW_up = grad_output_c.T @ x

    ### dW_gp
    dW_gp = grad_output_a_act_prime.T @ x
    
    """
    dx = torch.zeros(x.shape, **kwargs)  # .view() already done at higher level
    dW_up = torch.zeros(W_up.shape, **kwargs)  # check dtype
    dW_gp = torch.zeros(W_gp.shape, **kwargs)  # check dtype

    _compute_dx[grid](
        grad_output_c,
        W_up,
        grad_output_a_act_prime,
        W_gp,
        dx,
        M,
        N,
        K,
        NUM_SMS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )
    # # _compute_dW[grid](...)
    # # _compute_dW[grid](...)

    return dx, dW_up, dW_gp
