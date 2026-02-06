import os

os.environ["TRITON_INTERPRET"] = "1"

import copy
from torch.optim import Adam as Adam, RMSprop
from torch import nn, Tensor
import torch

from triton_kernels.optim.rmsprop import RMSprop as FusedRMSprop
from triton_kernels.optim.adam import Adam as FusedAdam

from triton_kernels.optim.ops.rmsprop import rmsprop as rmsprop_op
from triton_kernels.optim.rmsprop import rmsprop as rmsprop_ref


def get_tween_models(shape):
    model_1 = nn.Linear(*shape)
    model_2 = nn.Linear(*shape)
    model_2.load_state_dict(copy.deepcopy(model_1.state_dict()))
    return model_1, model_2

def normalized_diff(t1: Tensor, t2: Tensor) -> Tensor:
    return (t1 - t2).norm() / t2.norm()


def test_rmsprop_op():

    DEVICE = torch.device("cuda:0")

    n_params = 3
    D = 8
    params = [torch.rand(D, D).to(DEVICE) for _ in range(n_params)]
    grads = [torch.rand(D, D).to(DEVICE) for _ in range(n_params)]
    square_avgs = [torch.rand(D, D).to(DEVICE) for _ in range(n_params)]
    state_steps = [0 for _ in range(n_params)]

    params_ref = [x.clone() for x in params]
    params_ops = [x.clone() for x in params]

    square_avgs_ref = [x.clone() for x in square_avgs]
    square_avgs_ops = [x.clone() for x in square_avgs]

    print(f"Triton pre: \n{params_ref[0]}")
    print(f"Ref pre: \n{params_ref[0]}")

    kwargs = {"lr": 1e-1, "alpha": .5, "eps": 1e-5}
    rmsprop_op(
        params=params_ops, 
        grads=grads, 
        square_avgs=square_avgs_ops, 
        state_steps=state_steps,
        **kwargs
    )

    rmsprop_ref(
        params=params_ref, 
        grads=grads, 
        square_avgs=square_avgs_ref, 
        state_steps=state_steps,
        **kwargs
    )

    print(f"Triton post: \n{params_ref[0]}")
    print(f"Ref post: \n{params_ref[0]}")
    
    for t1, t2 in zip(params_ref, params_ops):
        diff = normalized_diff(t1, t2)
        assert diff < 1e-6
        print(diff)

    for t1, t2 in zip(square_avgs_ref, square_avgs_ops):
        diff = normalized_diff(t1, t2)
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
