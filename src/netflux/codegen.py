"""Export a model as one standalone PyTorch module (parameters, ODE function and a run block)."""

from __future__ import annotations

import datetime as _dt
import keyword
from pathlib import Path

from .model import Model
from .paths import output_path

__all__ = ["python_ode_source", "export_python_ode", "python_variable_names"]

RESERVED_NAMES = frozenset({"t", "y", "tau", "ymax", "w", "n", "ec50", "dydt", "torch", "math", "hill", "OR"})


def python_variable_names(species_ids: list[str]) -> list[str]:
    """Safe, unique Python names for species IDs (unchanged where possible)."""

    def ok(name: str) -> bool:
        return name.isidentifier() and not keyword.iskeyword(name) and name not in RESERVED_NAMES

    taken = {s for s in species_ids if ok(s)}
    names: list[str] = []
    for i, sid in enumerate(species_ids):
        if ok(sid) and sid not in names:
            names.append(sid)
            continue
        name = f"{sid}_" if sid.isidentifier() else f"s{i}"
        while name in names or name in taken or not ok(name):
            name += "_"
        names.append(name)
    return names


def _rate(j: int, inputs: tuple[tuple[int, bool], ...], var: list[str]) -> str:
    factors = [f"w[{j}]"]
    for i, neg in inputs:
        h = f"hill(y[{var[i]}], n[{j}], ec50[{j}])"
        factors.append(f"(1 - {h})" if neg else h)
    return " * ".join(factors)


def _floats(values) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


_HELPERS = '''

def hill(x, n, ec50):
    """Normalized Hill function: 0 at 0, 1/2 at EC50, 1 at 1."""
    x = torch.clamp(x, min=0)
    beta = (ec50**n - 1) / (2 * ec50**n - 1)
    return torch.clamp(beta * x**n / (beta - 1 + x**n), max=1)


def OR(*terms):
    return 1 - math.prod(1 - term for term in terms)
'''

_MAIN = '''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchdiffeq import odeint

    tau, ymax, y0, w, n, ec50 = load_params()
    t = torch.linspace(0.0, 10.0, 101, dtype=torch.float64)
    y = odeint(lambda t, y: ode(t, y, tau, ymax, w, n, ec50), y0, t, rtol=1e-6, atol=1e-8)
    plt.plot(t, y)
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.legend(SPECIES)
    plt.show()
'''


def python_ode_source(model: Model) -> str:
    var = python_variable_names(model.species_ids)
    rates = [_rate(j, rule.inputs, var) for j, rule in enumerate(model.topology.rules)]
    lines = [
        f'"""Netflux model {model.name!r}, generated on {_dt.date.today().isoformat()}."""',
        "import math",
        "",
        "import torch",
        "",
        f"SPECIES = {model.species_ids!r}",
        f"REACTIONS = {model.reaction_ids!r}",
        "",
        "",
        "def load_params():",
        f"    tau = torch.tensor({_floats(s.tau for s in model.species)}, dtype=torch.float64)",
        f"    ymax = torch.tensor({_floats(s.ymax for s in model.species)}, dtype=torch.float64)",
        f"    y0 = torch.tensor({_floats(s.y0 for s in model.species)}, dtype=torch.float64)",
        f"    w = torch.tensor({_floats(r.weight for r in model.reactions)}, dtype=torch.float64)",
        f"    n = torch.tensor({_floats(r.n for r in model.reactions)}, dtype=torch.float64)",
        f"    ec50 = torch.tensor({_floats(r.ec50 for r in model.reactions)}, dtype=torch.float64)",
        "    return tau, ymax, y0, w, n, ec50",
        _HELPERS,
        "",
        "def ode(t, y, tau, ymax, w, n, ec50):",
        *(f"    {v} = {i}" for i, v in enumerate(var)),
        "    dydt = torch.zeros_like(y)",
    ]
    for v, producers in zip(var, model.topology.producers):
        if not producers:
            total = "0"
        elif len(producers) == 1:
            total = rates[producers[0]]
        else:
            total = "OR(" + ", ".join(rates[j] for j in producers) + ")"
        lines.append(f"    dydt[{v}] = (({total}) * ymax[{v}] - y[{v}]) / tau[{v}]")
    lines.append("    return dydt")
    return "\n".join(lines) + "\n" + _MAIN


def export_python_ode(model: Model, path: str | Path | None = None) -> Path:
    path = output_path(path, "NetfluxODE.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(python_ode_source(model), newline="\n")
    return path
