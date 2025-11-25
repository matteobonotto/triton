# from .eager import matmul as eager_matmul
from .persistent import matmul as persistent_matmul
from .block import matmul as block_matmul

__all__ = [
    "persistent_matmul",
    "block_matmul",
]
