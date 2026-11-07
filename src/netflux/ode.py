"""Normalized-Hill logic ODEs in PyTorch; everything broadcasts over leading batch dims.

Reaction rate = weight * product of hill(x) (or 1 - hill(x) for ``!x``) over its inputs.
Species input = OR over the reactions producing it, ``1 - prod(1 - rate)``.
"""

from __future__ import annotations

import torch

from .rules import Topology

__all__ = ["DTYPE", "hill", "NetfluxODE"]

DTYPE = torch.float64


def hill(x: torch.Tensor, n, ec50) -> torch.Tensor:
    """Normalized Hill function: 0 at 0, 1/2 at EC50, 1 at 1; ``x`` is clamped at 0."""
    x = torch.clamp(x, min=0)
    en = ec50**n
    beta = (en - 1) / (2 * en - 1)
    xn = x**n
    return torch.clamp(beta * xn / (beta - 1 + xn), max=1)


def _padded(groups, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(g) for g in groups), default=0)
    idx = torch.zeros((len(groups), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(groups), width), dtype=torch.bool, device=device)
    for row, g in enumerate(groups):
        idx[row, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        mask[row, : len(g)] = True
    return idx, mask


class NetfluxODE(torch.nn.Module):
    """Right-hand side ``f(t, y)`` for torchdiffeq."""

    def __init__(self, topology: Topology, weight, n, ec50, ymax, tau) -> None:
        super().__init__()
        device = weight.device
        self.weight, self.n, self.ec50, self.ymax, self.tau = weight, n, ec50, ymax, tau
        self.in_idx, self.in_mask = _padded([[i for i, _ in r.inputs] for r in topology.rules], device)
        self.in_neg, _ = _padded([[int(neg) for _, neg in r.inputs] for r in topology.rules], device)
        self.in_neg = self.in_neg.bool()
        self.prod_idx, self.prod_mask = _padded(topology.producers, device)

    def reaction_rates(self, y: torch.Tensor) -> torch.Tensor:
        h = hill(y[..., self.in_idx], self.n[..., None], self.ec50[..., None])  # (..., R, K)
        terms = torch.where(self.in_neg, 1 - h, h)
        return self.weight * torch.where(self.in_mask, terms, 1.0).prod(-1)

    def activation(self, y: torch.Tensor) -> torch.Tensor:
        off = 1 - self.reaction_rates(y)[..., self.prod_idx]  # (..., S, P)
        return 1 - torch.where(self.prod_mask, off, 1.0).prod(-1)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return (self.activation(y) * self.ymax - y) / self.tau
