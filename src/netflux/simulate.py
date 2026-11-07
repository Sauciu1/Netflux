"""Simulation with torchdiffeq; runs continue from the last state, as in the Netflux GUI.

Default solver is ``dopri5``: torchdiffeq's ``bosh3`` (ode23's scheme) interpolates poorly.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torchdiffeq import odeint

from .model import UNDEFINED_HILL, Model, NetfluxWarning
from .ode import DTYPE, NetfluxODE
from .paths import output_path

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Simulator", "SimulationResult", "simulate"]


@dataclass
class SimulationResult:
    """``t``: ``(T,)``; ``y``: ``(T, ..., n_species)``."""

    t: torch.Tensor
    y: torch.Tensor
    species_ids: list[str]
    time_unit: str = "a.u."
    name: str = "netflux"

    def __getitem__(self, species_id: str) -> torch.Tensor:
        return self.y[..., self.species_ids.index(species_id)]

    def __len__(self) -> int:
        return self.t.shape[0]

    def _unbatched(self, what: str) -> None:
        if self.y.dim() != 2:
            raise ValueError(f"{what}() needs an unbatched result")

    def final(self) -> dict[str, float]:
        self._unbatched("final")
        return dict(zip(self.species_ids, self.y[-1].tolist()))

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        self._unbatched("to_dataframe")
        index = pd.Index(self.t.tolist(), name="t")
        return pd.DataFrame(self.y.detach().cpu().tolist(), columns=self.species_ids, index=index)

    def write_data(self, path: str | Path | None = None) -> Path:
        """The GUI "Export Data" table (species x time, space-delimited); default ``<name>_data.txt``."""
        self._unbatched("write_data")
        path = output_path(path, f"{self.name}_data.txt")
        lines = [" ".join(["Species_Name"] + [f"T{t:.10g}" for t in self.t.tolist()])]
        for sid, row in zip(self.species_ids, self.y.detach().cpu().T.tolist()):
            lines.append(" ".join([sid] + [f"{v:.4f}" for v in row]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", newline="\n")
        return path

    def plot(self, species: Iterable[str] | None = None, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("plotting needs matplotlib: pip install 'netflux[plot]'") from exc
        self._unbatched("plot")
        if ax is None:
            _, ax = plt.subplots()
        t = self.t.detach().cpu()
        for sid in self.species_ids if species is None else species:
            ax.plot(t, self[sid].detach().cpu(), linewidth=1.7, label=sid)
        ax.set_xlabel(f"Time ({self.time_unit})")
        ax.set_ylabel("Activity")
        ax.legend()
        return ax


class Simulator:
    """Stateful simulator mirroring the Netflux GUI; parameter tensors are public attributes."""

    def __init__(
        self,
        model: Model,
        *,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
        time_unit: str = "a.u.",
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.device = torch.device(device or "cpu")
        self.time_unit = time_unit
        self._has_inputs = torch.tensor([bool(r.inputs) for r in model.topology.rules], device=self.device)
        self.reset_parameters()
        self.reset()

    # --- helpers
    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(list(values), dtype=self.dtype, device=self.device)

    def _set(self, name: str, index: int, value) -> None:
        tensor = getattr(self, name).clone()  # out of place, so gradients still work
        tensor[..., index] = value
        setattr(self, name, tensor)

    @property
    def species_ids(self) -> list[str]:
        return self.model.species_ids

    @property
    def reaction_ids(self) -> list[str]:
        return self.model.reaction_ids

    def ode(self) -> NetfluxODE:
        return NetfluxODE(self.model.topology, self.weight, self.n, self.ec50, self.ymax, self.tau)

    # --- parameters
    def reset_parameters(self) -> None:
        m = self.model
        self.weight = self._tensor(r.weight for r in m.reactions)
        self.n = self._tensor(r.n for r in m.reactions)
        self.ec50 = self._tensor(r.ec50 for r in m.reactions)
        self.ymax = self._tensor(s.ymax for s in m.species)
        self.tau = self._tensor(s.tau for s in m.species)

    def set_reaction(self, reaction_id: str, *, weight=None, n=None, ec50=None) -> None:
        j = self.model.reaction_index(reaction_id)
        for name, value in (("weight", weight), ("n", n), ("ec50", ec50)):
            if value is not None:
                self._set(name, j, value)

    def set_species(self, species_id: str, *, ymax=None, tau=None, y0=None) -> None:
        """``y0`` sets the current state, like the GUI's Yinit slider."""
        i = self.model.species_index(species_id)
        for name, value in (("ymax", ymax), ("tau", tau), ("y", y0)):
            if value is not None:
                self._set(name, i, value)

    def set_state(self, species_id: str, value) -> None:
        self._set("y", self.model.species_index(species_id), value)

    # --- simulation
    def reset(self) -> None:
        """"Reset Simulation": t=0, model y0, empty history; parameters are kept."""
        self.t = 0.0
        self.y = self._tensor(s.y0 for s in self.model.species)
        self._t: list[torch.Tensor] = []
        self._y: list[torch.Tensor] = []

    def run(
        self,
        duration: float = 10.0,
        dt: float = 0.1,
        *,
        method: str = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        options: dict | None = None,
    ) -> SimulationResult:
        """Simulate ``duration`` more time units, sampled about every ``dt``; returns all runs so far."""
        if dt <= 0 or duration < 0:
            raise ValueError("dt must be positive and duration non-negative")
        undefined = self.ec50**self.n >= 0.5
        if undefined.dim() > 1:  # batched parameters: warn if any batch element is affected
            undefined = undefined.flatten(0, -2).any(0)
        for j in (undefined & self._has_inputs).nonzero().flatten().tolist():
            warnings.warn(f"Reaction {self.reaction_ids[j]}: {UNDEFINED_HILL}", NetfluxWarning, stacklevel=2)

        steps = round(duration / dt)
        grid = torch.linspace(self.t, self.t + duration, steps + 1, dtype=self.dtype, device=self.device)
        if steps == 0:
            ys = self.y.unsqueeze(0)
        else:
            ys = odeint(self.ode(), self.y, grid, method=method, rtol=rtol, atol=atol, options=options)
        if self._t:  # the first point repeats the previous run's last point
            grid, ys = grid[1:], ys[1:]
        if len(grid):
            self._t.append(grid)
            self._y.append(ys)
            self.t, self.y = float(grid[-1]), ys[-1]
        return self.result

    @property
    def result(self) -> SimulationResult:
        if self._t:
            t, y = torch.cat(self._t), torch.cat(self._y)
        else:
            t = torch.empty(0, dtype=self.dtype, device=self.device)
            y = torch.empty((0, *self.y.shape), dtype=self.dtype, device=self.device)
        return SimulationResult(t, y, self.species_ids, self.time_unit, self.model.name)


def simulate(
    model: Model,
    t_end: float = 10.0,
    dt: float = 0.1,
    *,
    reactions: dict[str, dict[str, float]] | None = None,
    species: dict[str, dict[str, float]] | None = None,
    **solver_kwargs,
) -> SimulationResult:
    """One-shot run from t=0, e.g. ``simulate(m, reactions={"r1": {"weight": 1}})``."""
    sim = Simulator(model)
    for rid, params in (reactions or {}).items():
        sim.set_reaction(rid, **params)
    for sid, params in (species or {}).items():
        sim.set_species(sid, **params)
    return sim.run(t_end, dt, **solver_kwargs)
