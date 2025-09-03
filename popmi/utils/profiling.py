from __future__ import annotations
import time
import torch
from torch import nn
from typing import Dict, Any, Tuple

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:  # pragma: no cover
    FlopCountAnalysis = None

__all__ = ["profile_model"]

def profile_model(model: nn.Module, input_size: Tuple[int,int,int] = (3,224,224), device: str = 'cuda', steps: int = 30, warmup: int = 5) -> Dict[str, Any]:
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    x = torch.randn(1, *input_size, device=dev)
    # FLOPs
    flops = None
    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, x).total()
        except Exception:
            flops = None
    # Latency + torch.profiler (CPU time / CUDA time)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == 'cuda' else None
        t0 = time.time()
        for _ in range(steps):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == 'cuda' else None
        avg_latency = (time.time() - t0) / steps
    # Memory peak (CUDA)
    peak_mem = None
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            _ = model(x)
        peak_mem = torch.cuda.max_memory_allocated(dev)
    return {
        'flops': flops,
        'avg_latency_s': avg_latency,
        'peak_mem_bytes': peak_mem,
        'params': sum(p.numel() for p in model.parameters())
    }
