
from torch.optim import Adam
from torch import nn


def test_optimizer():
    
    optim = Adam(params=nn.Linear(2, 2).parameters(), foreach=False)

    for _ in range(10):
        optim.step()


test_optimizer()