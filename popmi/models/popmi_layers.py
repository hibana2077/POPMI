from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple, Optional

# -----------------------------
# D4 Group utilities
# -----------------------------

def _rot90(x: Tensor, k: int) -> Tensor:
    k = int(k) % 4
    if k == 0:
        return x
    return torch.rot90(x, k, dims=(-2, -1))

def _hflip(x: Tensor) -> Tensor:
    return torch.flip(x, dims=(-1,))

class D4Group:
    def __init__(self):
        self.elements: List[Tuple[int, bool]] = [(k, f) for f in (False, True) for k in (0,1,2,3)]
        self.order = len(self.elements)
    def act(self, x: Tensor, element: Tuple[int, bool]) -> Tensor:
        k, flip = element
        y = x
        if flip:
            y = _hflip(y)
        y = _rot90(y, k)
        return y

class POPMIProjection(nn.Module):
    def __init__(self, group: str = 'D4', learnable_weights: bool = False, temperature: float = 1.0,
                 residual: bool = True, beta: float = 0.5, sample_k: Optional[int] = None):
        super().__init__()
        assert group == 'D4', 'Only D4 supported in this minimal release.'
        self.G = D4Group()
        self.learnable_weights = learnable_weights
        self.temperature = float(temperature)
        self.residual = residual
        self.beta = float(beta)
        self.sample_k = sample_k
        if learnable_weights:
            self.theta = nn.Parameter(torch.zeros(self.G.order))
        else:
            self.register_parameter('theta', None)
    def _weights(self, device: torch.device) -> Tensor:
        if self.learnable_weights:
            return F.softmax(self.theta / self.temperature, dim=0)
        return torch.ones(self.G.order, device=device) / self.G.order
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4
        device = x.device
        w_full = self._weights(device)
        if self.training and self.sample_k is not None and 1 <= self.sample_k < self.G.order:
            idx = torch.randperm(self.G.order, device=device)[: self.sample_k]
        else:
            idx = torch.arange(self.G.order, device=device)
        y = torch.zeros_like(x)
        w_sum = 0.0
        for i in idx.tolist():
            elem = self.G.elements[i]
            yi = self.G.act(x, elem)
            wi = w_full[i]
            y = y + wi * yi
            w_sum += float(wi.detach())
        y = y / max(w_sum, 1e-12)
        if self.residual:
            y = (1.0 - self.beta) * x + self.beta * y
        return y

class CE_PnP(nn.Module):
    def __init__(self, projector: nn.Module, rho: float = 1.0, iters: int = 3):
        super().__init__()
        self.projector = projector
        self.rho = float(rho)
        self.iters = int(iters)
        assert self.iters >= 1
    def forward(self, f: Tensor) -> Tensor:
        z = self.projector(f)
        v = f
        u = torch.zeros_like(f)
        rho = self.rho
        for _ in range(self.iters):
            z = self.projector(v - u)
            v = (f + rho * (z + u)) / (1.0 + rho)
            u = u + z - v
        return z
