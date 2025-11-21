from .eager import matmul as eager_matmul
from .persistent import matmul as persistent_matmul
from .swizzle import matmul as swizzle_matmul

__all__ = [
    "eager_matmul",
    "persistent_matmul",
    "swizzle_matmul",
]
