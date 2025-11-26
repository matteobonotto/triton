from torch import Tensor


def validate_inputs_matmul(a: Tensor, b: Tensor) -> None:
    assert a.ndim == 2, "expected A to have ndim=2"
    assert b.ndim == 2, "expected B to have ndim=2"
    assert a.shape[1] == b.shape[0], "dimension mismatch"
