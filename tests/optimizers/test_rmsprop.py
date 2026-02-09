import os

os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_SKIP_AUTOTUNING"] = "1"

import copy
from torch.optim import Adam as Adam, RMSprop
from torch import nn, Tensor
import torch

from triton_kernels.optim.rmsprop import RMSprop as FusedRMSprop
from triton_kernels.optim.adam import Adam as FusedAdam

from triton_kernels.optim.ops.rmsprop import (
    rmsprop_interface as rmsprop_op,
    rmsprop_interface_torch as rmsprop_ref,
)


def get_tween_models(shape):
    model_1 = nn.Linear(*shape)
    model_2 = nn.Linear(*shape)
    model_2.load_state_dict(copy.deepcopy(model_1.state_dict()))
    return model_1, model_2


def normalized_diff(t1: Tensor, t2: Tensor) -> Tensor:
    return (t1 - t2).norm() / t2.norm()


def test_rmsprop_op():

    DEVICE = torch.device("cuda:0")

    D = 5

    param = torch.rand(D, D).to(DEVICE)
    grad = torch.rand(D, D).to(DEVICE)
    square_avg = torch.rand(D, D).to(DEVICE)

    param_ref = param.clone()
    param_ops = param.clone()

    square_avg_ref = square_avg.clone()
    square_avg_ops = square_avg.clone()

    print(f"Triton pre: \n{param_ref}")
    print(f"Ref pre: \n{param_ops}")

    kwargs = {"lr": 1e-1, "alpha": 0.5, "eps": 1e-5}
    rmsprop_ref(
        param=param_ref,
        grad=grad,
        square_avg=square_avg_ref,
        **kwargs,
    )
    rmsprop_op(param=param_ops, grad=grad, square_avg=square_avg_ops, **kwargs)

    print(f"Ref post: \n{param_ref - param_ops}")
    # print(f"Triton post: \n{param_ref}")
    # print(f"Ref post: \n{param_ops}")

    diff = normalized_diff(square_avg_ref, square_avg_ops)
    assert diff < 1e-6
    print(diff)

    diff = normalized_diff(param_ref, param_ops)
    print(diff)
    assert diff < 1e-6
    ...


test_rmsprop_op()


def test_rmsprop():

    torch.manual_seed(42)
    shape = (3, 3)
    model_1, model_2 = get_tween_models(shape)

    x = torch.rand(
        shape[0],
    )
    optim = RMSprop(params=model_1.parameters(), foreach=False)
    optim_fused = FusedRMSprop(params=model_2.parameters())

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
    ...
