import os
from .utils import is_cuda_available

if not is_cuda_available() or "TRITON_IS_DEBUGGING" in os.environ.keys():
    os.environ["TRITON_INTERPRET"] = "1"
