"""Simulation with torchdiffeq; runs continue from the last state, as in the Netflux GUI.

Default solver is ``dopri5``: torchdiffeq's ``bosh3`` (ode23's scheme) interpolates poorly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torchdiffeq import odeint

from .model import Model, NetfluxWarning
from .ode import DTYPE, NetfluxODE
from .paths import output_path

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Simulator", "SimulationResult", "simulate", "matlab_num2str"]


def matlab_num2str(x: float) -> str:
    """Format like MATLAB ``num2str``."""
    x = float(x)
    if math.isfinite(x) and x == int(x):
        return str(int(x))
    if not math.isfinite(x):
        return {math.inf: "Inf", -math.inf: "-Inf"}.get(x, "NaN")
    digits = max(math.floor(math.log10(abs(x))) + 5, 5)
    return f"{x:.{digits}g}"


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

    def final(self) -> dict[str, float]:
        last = self.y[-1]
        if last.dim() != 1:
            raise ValueError("final() needs an unbatched result")
        return {sid: float(v) for sid, v in zip(self.species_ids, last)}

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        if self.y.dim() != 2:
            raise ValueError("to_dataframe() needs an unbatched result")
        frame = pd.DataFrame(
            self.y.detach().cpu().tolist(), columns=self.species_ids,
            index=pd.Index(self.t.detach().cpu().tolist(), name="t"),
        )
        return frame

    def write_data(self, path: str | Path | None = None) -> Path:
        """The GUI "Export Data" text format; default ``<name>_data.txt``."""
        if self.y.dim() != 2:
            raise ValueError("write_data() needs an unbatched result")
        path = output_path(path, f"{self.name}_data.txt")
        times = self.t.detach().cpu().tolist()
        values = self.y.detach().cpu().T.tolist()
        lines = ["".join(f"{c} " for c in ["Species_Name"] + [f"T{matlab_num2str(t)}" for t in times])]
        for sid, row in zip(self.species_ids, values):
            lines.append("".join(f"{c} " for c in [sid] + [f"{v:0.4f}" for v in row]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", newline="\n")
        return path

    def plot(self, species: Iterable[str] | None = None, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("plotting needs matplotlib: pip install 'netflux[plot]'") from exc
        if self.y.dim() != 2:
            raise ValueError("plot() needs an unbatched result")
        species = list(self.species_ids if species is None else species)
        if ax is None:
            _, ax = plt.subplots()
        t = self.t.detach().cpu().tolist()
        for sid in species:
            ax.plot(t, self[sid].detach().cpu().tolist(), linewidth=1.7, label=sid)
        ax.set_xlabel(f"Time ({self.time_unit})")
        ax.set_ylabel("Activity")
        ax.legend()
        return ax


@dataclass
class _History:
    t: list[torch.Tensor] = field(default_factory=list)
    y: list[torch.Tensor] = field(default_factory=list)


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
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.time_unit = time_unit
        self.reset_parameters()
        self.reset()

    # --- helpers
    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(list(values), dtype=self.dtype, device=self.device)

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
                tensor = getattr(self, name).clone()
                tensor[..., j] = value
                setattr(self, name, tensor)

    def set_species(self, species_id: str, *, ymax=None, tau=None, y0=None) -> None:
        """``y0`` sets the current state, like the GUI's Yinit slider."""
        i = self.model.species_index(species_id)
        for name, value in (("ymax", ymax), ("tau", tau)):
            if value is not None:
                tensor = getattr(self, name).clone()
                tensor[..., i] = value
                setattr(self, name, tensor)
        if y0 is not None:
            self.set_state(species_id, y0)

    def set_state(self, species_id: str, value) -> None:
        i = self.model.species_index(species_id)
        y = self.y.clone()
        y[..., i] = value
        self.y = y

    # --- simulation
    def reset(self) -> None:
        """"Reset Simulation": t=0, model y0, empty history; parameters are kept."""
        self.t = 0.0
        self.y = self._tensor(s.y0 for s in self.model.species)
        self._history = _History()

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
        """Simulate ``duration`` more time units from the current state; returns the cumulative result."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        if duration < 0:
            raise ValueError("duration must be non-negative")
        for msg in self.model.parameter_warnings(
            n=self.n.reshape(-1, self.n.shape[-1])[0].tolist(),
            ec50=self.ec50.reshape(-1, self.ec50.shape[-1])[0].tolist(),
        ):
            warnings.warn(msg, NetfluxWarning, stacklevel=2)

        steps = math.floor(duration / dt + 1e-9)
        grid = self.t + dt * torch.arange(steps + 1, dtype=self.dtype, device=self.device)
        if steps == 0:
            ys = self.y.unsqueeze(0)
        else:
            ys = odeint(
                self.ode(), self.y, grid, method=method, rtol=rtol, atol=atol, options=options
            )

        if self._history.t:  # the first point repeats the previous run's last point
            grid, ys = grid[1:], ys[1:]
        self._history.t.append(grid)
        self._history.y.append(ys)
        self.t = float(grid[-1]) if grid.numel() else self.t
        self.y = ys[-1] if ys.shape[0] else self.y
        return self.result

    @property
    def result(self) -> SimulationResult:
        if self._history.t:
            t = torch.cat(self._history.t)
            y = torch.cat(self._history.y)
        else:
            t = torch.empty(0, dtype=self.dtype, device=self.device)
            y = torch.empty((0,) + tuple(self.y.shape), dtype=self.dtype, device=self.device)
        return SimulationResult(
            t=t, y=y, species_ids=self.species_ids, time_unit=self.time_unit, name=self.model.name
        )


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
