"""Port of ODE.m / Netflux2ODE.m: normalized-Hill logic ODEs in PyTorch (batch-broadcasting)."""

from __future__ import annotations

import torch

from .rules import Topology

__all__ = ["DTYPE", "act", "inhib", "OR", "AND", "NetfluxODE"]

DTYPE = torch.float64


def act(x: torch.Tensor, w, n, ec50) -> torch.Tensor:
    """Normalized Hill activation; ``x`` is clamped at 0 (MATLAB goes complex below 0)."""
    x = torch.clamp(x, min=0)
    en = ec50**n
    beta = (en - 1) / (2 * en - 1)
    xn = x**n
    # K**n == beta - 1 (MATLAB computes K first)
    f = w * beta * xn / ((beta - 1) + xn)
    return torch.minimum(f, torch.as_tensor(w, dtype=f.dtype, device=f.device))


def inhib(x: torch.Tensor, w, n, ec50) -> torch.Tensor:
    return w - act(x, w, n, ec50)


def OR(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: N802 - Netflux name
    return x + y - x * y


def AND(w, *terms: torch.Tensor) -> torch.Tensor:  # noqa: N802 - Netflux name
    w = torch.as_tensor(w, dtype=DTYPE) if not isinstance(w, torch.Tensor) else w
    prod = torch.stack(torch.broadcast_tensors(*terms)).prod(0)
    safe_w = torch.where(w == 0, torch.ones_like(w), w)
    return torch.where(w == 0, torch.zeros_like(prod), prod / safe_w ** (len(terms) - 1))


def _padded(groups, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(g) for g in groups), default=0)
    idx = torch.zeros((len(groups), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(groups), width), dtype=torch.bool, device=device)
    for row, g in enumerate(groups):
        if g:
            idx[row, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            mask[row, : len(g)] = True
    return idx, mask


class NetfluxODE(torch.nn.Module):
    """Right-hand side ``f(t, y)`` for torchdiffeq; parameters may have leading batch dims."""

    def __init__(
        self,
        topology: Topology,
        weight: torch.Tensor,
        n: torch.Tensor,
        ec50: torch.Tensor,
        ymax: torch.Tensor,
        tau: torch.Tensor,
    ) -> None:
        super().__init__()
        device = weight.device
        self.topology = topology
        self.weight, self.n, self.ec50, self.ymax, self.tau = weight, n, ec50, ymax, tau

        rules = topology.rules
        self.r_idx, self.r_mask = _padded([r.reactants for r in rules], device=device)
        neg_groups = [r.negated for r in rules]
        self.r_neg = torch.zeros_like(self.r_mask)
        for row, g in enumerate(neg_groups):
            if g:
                self.r_neg[row, : len(g)] = torch.tensor(g, dtype=torch.bool, device=device)
        self.k = torch.tensor([len(r.reactants) for r in rules], dtype=torch.long, device=device)
        self.p_idx, self.p_mask = _padded(topology.producers, device=device)

    def reaction_rates(self, y: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x = y[..., self.r_idx]  # (..., R, K)
        wk, nk, ek = w[..., None], self.n[..., None], self.ec50[..., None]
        a = act(x, wk, nk, ek)
        term = torch.where(self.r_neg, wk - a, a)
        term = torch.where(self.r_mask, term, torch.ones_like(term))
        prod = term.prod(-1)  # (..., R)
        safe_w = torch.where(w == 0, torch.ones_like(w), w)
        and_val = prod / safe_w ** (self.k - 1).clamp(min=0)
        and_val = torch.where((self.k >= 2) & (w == 0), torch.zeros_like(and_val), and_val)
        return torch.where(self.k == 0, w.expand_as(and_val), and_val)

    def activation(self, y: torch.Tensor) -> torch.Tensor:
        f = self.reaction_rates(y)
        shape = f.shape[:-1] + (self.topology.n_species,)
        total = torch.zeros(shape, dtype=f.dtype, device=f.device)
        started = torch.zeros(self.topology.n_species, dtype=torch.bool, device=f.device)
        # right fold reproduces MATLAB's OR(r1, OR(r2, OR(r3, r4)))
        for col in reversed(range(self.p_idx.shape[1])):
            fj = f[..., self.p_idx[:, col]]
            m = self.p_mask[:, col]
            combined = torch.where(started, OR(fj, total), fj)
            total = torch.where(m, combined, total)
            started = started | m
        return total

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return (self.activation(y) * self.ymax - y) / self.tau
