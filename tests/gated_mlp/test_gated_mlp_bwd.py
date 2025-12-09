from triton_kernels.nn.gated_mlp.gated_mlp import (
    NaiveGatedMLP,
    FusedGatedMLP,
    eager_fwd,
    mlp_hidden_states_fwd,
)
from triton_kernels.utils import get_device

from torch.autograd import grad
import triton
from torch import nn
import torch


def copy_weights(module_src: nn.Module, module_tgt: nn.Module) -> None:

    for (n1, p1), (n2, p2) in zip(
        module_src.named_parameters(), module_tgt.named_parameters()
    ):
        assert n1 == n2, "name mismatch"
        p2.data = p1.data.clone()


def test_gated_mlp_bwd():

    DEVICE = get_device()
    DTYPE = torch.float32

    gmlp_1 = NaiveGatedMLP(dropout_p=0.0, bias=False).to(DEVICE).to(DTYPE)
    gmlp_2 = FusedGatedMLP(dropout_p=0.0, bias=False).to(DEVICE).to(DTYPE)

    copy_weights(gmlp_1, gmlp_2)

    M = 128
    K = gmlp_1.hidden_size
    B = 16

    x = torch.rand((B, M, K), device=DEVICE, dtype=DTYPE)
    x.requires_grad = True

    ### test fwd pass first
    out_1 = gmlp_1(x)
    out_2 = gmlp_2(x)
    triton.testing.assert_close(out_1, out_2, atol=1e-6)

    ### then test bwd pass
    grad_outputs = torch.rand_like(x).to(DEVICE)
    inputs = (
        x,
        gmlp_1.up_proj.weight,
        # gmlp_1.up_proj.bias,
        gmlp_1.gate_proj.weight,
        # gmlp_1.gate_proj.bias,
        # gmlp_1.act_fn,
        # gmlp_1.dropout.p,
    )
    grads_1 = grad(out_1, inputs, grad_outputs=grad_outputs, retain_graph=True)
    inputs = (
        x,
        gmlp_2.up_proj.weight,
        # gmlp_2.up_proj.bias,
        gmlp_2.gate_proj.weight,
        # gmlp_2.gate_proj.bias,
        # gmlp_2.act_fn,
        # gmlp_2.dropout.p,
    )
    grads_2 = grad(
        out_2, inputs, grad_outputs=grad_outputs, retain_graph=True, allow_unused=True
    )
    for g1, g2 in zip(grads_1, grads_2):
        print((g1 - g2).norm() / g2.norm())
    print("Done!!")

    triton.testing.assert_close(gmlp_1(x), gmlp_2(x), rtol=1e-6)


test_gated_mlp_bwd()


def test_bwd_op_triton():

    DEVICE = get_device()
    DTYPE = torch.float32

    init_args = {
        "hidden_act": "no_act",
        "dropout_p": 0.0,
        "bias": False,
    }

    gmlp = NaiveGatedMLP(**init_args).to(DEVICE).to(DTYPE)

    M = 128
    K = gmlp.hidden_size

    x = torch.rand((M, K), device=DEVICE, dtype=DTYPE)
    out_ref = eager_forward(
        x,
        gmlp.up_proj.weight,
        gmlp.up_proj.bias,
        gmlp.gate_proj.weight,
        gmlp.gate_proj.bias,
        gmlp.act_fn,
        gmlp.dropout.p,
    )
    print(out_ref)

    out_triton = mlp_hidden_states_fwd(
        x,
        gmlp.up_proj.weight,
        gmlp.up_proj.bias,
        gmlp.gate_proj.weight,
        gmlp.gate_proj.bias,
        gmlp.act_fn,
        gmlp.dropout.p,
    )
    print(out_triton)

    print((out_ref - out_triton).norm() / out_ref.norm())

    triton.testing.assert_close(out_ref, out_triton, rtol=1e-6)

    ...

    # triton.testing.assert_close(gmlp_1(x), gmlp_2(x), rtol=1e-6)


test_fwd_op_triton()
