import torch, os

try:
    import psutil
except ImportError:
    psutil = None

class CUDAMemTracker:
    def __init__(self, device=None):
        self.device = torch.device(device or "cuda")

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_alloc = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        self.start_rss = psutil.Process(os.getpid()).memory_info().rss if psutil else None
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize(self.device)
        self.end_alloc = torch.cuda.memory_allocated(self.device)
        self.end_reserved = torch.cuda.memory_reserved(self.device)
        self.peak_alloc = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device)

    def report(self, unit=1024**2):
        to_mb = lambda b: None if b is None else b / unit
        return {
            "alloc_delta_MB": to_mb(self.end_alloc - self.start_alloc),
            "reserved_delta_MB": to_mb(self.end_reserved - self.start_reserved),
            "peak_alloc_MB": to_mb(self.peak_alloc),
            "peak_reserved_MB": to_mb(self.peak_reserved),
            "cpu_rss_MB": to_mb(self.start_rss) if self.start_rss is not None else None,
        }


