import os

os.environ["TRITON_INTERPRET"] = "1"

import copy
from torch.optim import Adam
from torch import nn, Tensor
import torch

from torch.optim.adam import adam as adam_ref

from triton_kernels.optim.adam import Adam as FusedAdam

from triton_kernels.optim.ops.adam import (
    adam_interface as adam_op,
    adam_interface_torch as adam_eager,
)

DEVICE = torch.device("cuda:0")


def get_tween_models(shape):
    model_1 = nn.Linear(*shape).to(DEVICE)
    model_2 = nn.Linear(*shape).to(DEVICE)
    model_2.load_state_dict(copy.deepcopy(model_1.state_dict()))
    return model_1, model_2


def normalized_diff(t1: Tensor, t2: Tensor) -> Tensor:
    return (t1 - t2).norm() / t2.norm()


def test_eager_adam():

    torch.manual_seed(42)
    shape = (3, 3)
    model_1, model_2 = get_tween_models(shape)

    x = torch.rand(shape[0], device=DEVICE)
    optim = Adam(params=model_1.parameters(), foreach=False, lr=0.02)
    optim_fused = FusedAdam(params=model_2.parameters(), lr=0.02)

    print(list(model_1.parameters()))
    y = model_1(x)
    loss = y.sum()
    loss.backward()
    optim.step()
    print(list(model_1.parameters()))

    print(list(model_2.parameters()))
    y = model_2(x)
    loss = y.sum()
    loss.backward()
    optim_fused.step()
    print(list(model_2.parameters()))

    assert normalized_diff(model_1.weight, model_2.weight) < 1e-6
    assert normalized_diff(model_1.bias, model_2.bias) < 1e-6


test_eager_adam()


def test_adam_op():

    D = 5

    param = torch.rand(D, D).to(DEVICE)
    grad = torch.rand(D, D).to(DEVICE)
    exp_avg = torch.rand(D, D).to(DEVICE)
    exp_avg_sq = torch.rand(D, D).to(DEVICE)

    param_ref = param.clone()
    param_ops = param.clone()

    exp_avg_ref = exp_avg.clone()
    exp_avg_ops = exp_avg.clone()

    exp_avg_sq_ref = exp_avg_sq.clone()
    exp_avg_sq_ops = exp_avg_sq.clone()

    print(f"Triton pre: \n{param_ref}")
    print(f"Ref pre: \n{param_ops}")

    kwargs = {
        "lr": 1e-1,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-5,
        "step": torch.tensor(1.0),
    }
    adam_eager(
        param=param_ref,
        grad=grad,
        exp_avg=exp_avg_ref,
        exp_avg_sq=exp_avg_sq_ref,
        **kwargs,
    )
    print(f"Ref post: \n{param_ref}")
    adam_op(
        param=param_ops,
        grad=grad,
        exp_avg=exp_avg_ops,
        exp_avg_sq=exp_avg_sq_ops,
        **kwargs,
    )

    print(f"Ref post: \n{param_ref - param_ops}")

    diff = normalized_diff(exp_avg_ref, exp_avg_ops)
    assert diff < 1e-6
    print(diff)

    diff = normalized_diff(exp_avg_sq_ref, exp_avg_sq_ops)
    assert diff < 1e-6
    print(diff)

    diff = normalized_diff(param_ref, param_ops)
    print(diff)
    assert diff < 1e-6
    ...


test_adam_op()
