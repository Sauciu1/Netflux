"""SIF networks: export (byte-identical to MATLAB Netflux, plus node attributes) and import."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..model import Model, ModelError, Reaction, Species
from ..output import output_path

__all__ = ["Edge", "network_edges", "to_sif", "to_node_attributes", "write_sif", "read_sif"]

_CONNECTOR = re.compile(r"rcn\d+")


@dataclass(frozen=True)
class Edge:
    source: str
    sign: int  # 1 activating, -1 inhibiting
    target: str


def network_edges(model: Model) -> tuple[list[Edge], list[str]]:
    """Edges of reactions with one product; multi-input reactions go through an ``rcn<k>`` AND node."""
    ids = model.species_ids
    edges: list[Edge] = []
    connectors: list[str] = []
    for k, rule in enumerate(model.topology.rules, start=1):
        if len(rule.products) != 1 or not rule.inputs:
            continue
        target = ids[rule.products[0]]
        if len(rule.inputs) > 1:
            connectors.append(f"rcn{k}")
            edges.append(Edge(connectors[-1], 1, target))
            target = connectors[-1]
        edges += [Edge(ids[i], -1 if neg else 1, target) for i, neg in rule.inputs]
    return edges, connectors


def to_sif(model: Model) -> str:
    edges, _ = network_edges(model)
    return "".join(f"{e.source}\t{e.sign}\t{e.target}\n" for e in edges)


def to_node_attributes(model: Model) -> str:
    _, connectors = network_edges(model)
    lines = ["Type (class=String)"]
    lines += [f"{s.id} = {s.type}" for s in model.species]
    lines += [f"{c} = connector" for c in connectors]
    return "\n".join(lines) + "\n"


def write_sif(model: Model, path: str | Path | None = None) -> Path:
    """Write ``<name>.sif`` and, next to it, ``<name>_nodeAttributes.txt``."""
    path = output_path(path, f"{model.name}.sif")
    path.write_text(to_sif(model), newline="")
    path.with_name(f"{path.stem}_nodeAttributes.txt").write_text(to_node_attributes(model), newline="")
    return path


def _input(name: str, sign: str) -> tuple[str, bool]:
    return name, sign.strip() == "-1"


def read_sif(path: str | Path) -> Model:
    """``rcn<k>`` connector nodes become AND rules, as Netflux's SIF export writes them."""
    path = Path(path)
    lines: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t") if raw.count("\t") == 2 else raw.split()
        if len(parts) != 3:
            raise ModelError(f"{path.name}:{lineno}: expected 'source<TAB>sign<TAB>target', got {raw!r}")
        lines.append(tuple(p.strip() for p in parts))  # type: ignore[arg-type]

    def is_connector(node: str) -> bool:
        return _CONNECTOR.fullmatch(node) is not None

    species_ids = sorted({n for src, _, tgt in lines for n in (src, tgt) if not is_connector(n)})
    gates: dict[str, list[tuple[str, bool]]] = {}
    for src, sign, tgt in lines:
        if is_connector(tgt):
            gates.setdefault(tgt, []).append(_input(src, sign))

    wiring: list[tuple[list[tuple[str, bool]], str]] = []
    warnings: list[str] = []
    for src, sign, tgt in lines:
        if is_connector(tgt):
            continue  # consumed by its connector's output line
        if not is_connector(src):
            wiring.append(([_input(src, sign)], tgt))
        elif src in gates:
            wiring.append((gates[src], tgt))
        else:
            warnings.append(f"Connector {src} has no inputs; skipped")

    species = [Species(s, name=s) for s in species_ids]
    reactions = [Reaction(f"r{k}", inputs, (tgt,)) for k, (inputs, tgt) in enumerate(wiring, start=1)]
    model = Model(path.stem, species, reactions)
    model.warnings = warnings + model.diagnose()
    return model
