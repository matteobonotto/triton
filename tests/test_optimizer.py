
import copy
from torch.optim import Adam as Adam, RMSprop
from torch import nn
import torch

from triton_kernels.optim.rmsprop import RMSprop as FusedRMSprop
from triton_kernels.optim.adam import Adam as FusedAdam

def get_tween_models(shape):
    model_1 = nn.Linear(*shape)
    model_2 = nn.Linear(*shape)
    model_2.load_state_dict(copy.deepcopy(model_1.state_dict()))
    return model_1, model_2

def test_rmsprop():
    
    torch.manual_seed(42)
    shape = (3, 3)    
    model_1, model_2 = get_tween_models(shape)

    x = torch.rand(shape[0], )
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


def test_adam():
    
    torch.manual_seed(42)
    shape = (3, 3)    
    model_1, model_2 = get_tween_models(shape)

    x = torch.rand(shape[0], )
    optim = Adam(params=model_1.parameters(), foreach=False)
    optim_fused = FusedAdam(params=model_2.parameters())

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


test_adam()