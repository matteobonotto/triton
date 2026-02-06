import triton
from torch import Tensor
import torch


@triton.jit()
def adam_kernel(): ...


def adam(): ...
