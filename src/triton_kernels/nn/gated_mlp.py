from typing import Dict
from torch import nn
from collections import OrderedDict


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

# def mlp_fwd(x, W1, W2, W3):
#     return (torch.nn.functional.silu(x @ W2.T) * (x @ W1.T)) @ W3.T

# class LlamaMLP(nn.Module):
#     def __init__(self, hidden_size: int = 64, intermediate_size: int = 256):
#         super().__init__()
#         self.W1 = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.W2 = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.W3 = nn.Linear(intermediate_size, hidden_size, bias=False)
#         self.act = nn.SiLU()

#     def forward(self, x: Tensor) -> Tensor:
#         return mlp_fwd(x, self.W1.weight, self.W2.weight, self.W3.weight)


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(self.dropout(hidden_states))
        return down_proj
