"""Netflux model: species, and reactions whose inputs are ANDed (reactions producing a species are ORed).

A reaction is stored as inputs and outputs, as in SBML-qual. Rule strings such as ``"A & !B => C"``
(also ``->``, ``+``/``AND`` and ``NOT``) are only a text form, used by the Excel sheet and
``Reaction.from_rule``.
"""

from __future__ import annotations

import copy as _copy
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, NamedTuple

__all__ = [
    "Species",
    "Reaction",
    "Model",
    "ModelError",
    "NetfluxWarning",
    "DEFAULT_SPECIES_PARAMS",
    "DEFAULT_REACTION_PARAMS",
    "UNDEFINED_HILL",
    "Wiring",
    "Topology",
    "parse_rule",
]

DEFAULT_SPECIES_PARAMS = {"y0": 0.0, "ymax": 1.0, "tau": 1.0}
DEFAULT_REACTION_PARAMS = {"weight": 1.0, "n": 1.4, "ec50": 0.5}
UNDEFINED_HILL = "EC50^n >= 0.5, so the normalized Hill function is undefined"

Inputs = tuple[tuple[str, bool], ...]  # (species id, negated)


class ModelError(ValueError):
    """Raised when a model file cannot be read at all."""


class NetfluxWarning(UserWarning):
    """Recoverable model problems."""


# --- rule strings

_ARROW = re.compile(r"=>|->")
_SEPARATOR = re.compile(r"&|\+|\bAND\b")
_NEGATION = re.compile(r"^(?:!|NOT\s)\s*")


def _terms(text: str) -> list[str]:
    return [t.strip() for t in _SEPARATOR.split(text) if t.strip()]


def parse_rule(rule: str) -> tuple[Inputs, tuple[str, ...]]:
    """``"A & !B => C"`` -> ``((("A", False), ("B", True)), ("C",))``."""
    arrow = _ARROW.search(rule)
    if arrow is None:
        raise ModelError(f"no reaction arrow ('=>') could be identified in {rule!r}")
    inputs = []
    for term in _terms(rule[: arrow.start()]):
        negation = _NEGATION.match(term)
        inputs.append((term[negation.end():], True) if negation else (term, False))
    return tuple(inputs), tuple(_terms(rule[arrow.end():]))


# --- model


def number(value: Any) -> float | None:
    """``float(value)``, or ``None`` for missing, non-numeric, NaN or boolean values."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def with_defaults(
    values: dict[str, float | None], defaults: dict[str, float], label: str, warnings: list[str]
) -> dict[str, float]:
    """Replace ``None`` values with defaults, recording a warning for ``label``."""
    missing = [k for k, v in values.items() if v is None]
    if missing:
        warnings.append(f"{label}: missing parameter(s) {', '.join(missing)}; default parameters assigned")
    return {k: defaults[k] if v is None else v for k, v in values.items()}


@dataclass
class Species:
    """``y0`` is MATLAB's ``Yinit``."""

    id: str
    name: str = ""
    y0: float = DEFAULT_SPECIES_PARAMS["y0"]
    ymax: float = DEFAULT_SPECIES_PARAMS["ymax"]
    tau: float = DEFAULT_SPECIES_PARAMS["tau"]
    type: str = ""
    location: str = ""
    module: str = ""
    ref: str = ""


@dataclass
class Reaction:
    """Rate = weight x the product of each input's Hill term; no inputs means rate = weight."""

    id: str
    inputs: Inputs = ()
    outputs: tuple[str, ...] = ()
    weight: float = DEFAULT_REACTION_PARAMS["weight"]
    n: float = DEFAULT_REACTION_PARAMS["n"]
    ec50: float = DEFAULT_REACTION_PARAMS["ec50"]
    module: str = ""

    def __post_init__(self) -> None:
        self.inputs = tuple((str(s), bool(negated)) for s, negated in self.inputs)
        self.outputs = tuple(str(s) for s in self.outputs)

    @classmethod
    def from_rule(cls, id: str, rule: str, **params) -> Reaction:  # noqa: A002 - mirrors the field
        inputs, outputs = parse_rule(rule)
        return cls(id, inputs, outputs, **params)

    @property
    def rule(self) -> str:
        lhs = " & ".join(f"!{s}" if negated else s for s, negated in self.inputs)
        return f"{lhs} => {' & '.join(self.outputs)}".strip()


class Wiring(NamedTuple):
    inputs: tuple[tuple[int, bool], ...]  # (species index, negated), in species order
    products: tuple[int, ...]


@dataclass(frozen=True)
class Topology:
    """Reactions resolved to species indices; names that are not species are left out."""

    n_species: int
    rules: tuple[Wiring, ...]

    @property
    def producers(self) -> tuple[tuple[int, ...], ...]:
        """Reactions producing each species, in reaction order."""
        out: list[list[int]] = [[] for _ in range(self.n_species)]
        for j, rule in enumerate(self.rules):
            for i in rule.products:
                out[i].append(j)
        return tuple(tuple(p) for p in out)


_VALID_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_RESERVED_IDS = {"AND", "NOT"}


@dataclass
class Model:
    name: str
    species: list[Species]
    reactions: list[Reaction]
    warnings: list[str] = field(default_factory=list)

    # --- access
    @property
    def species_ids(self) -> list[str]:
        return [s.id for s in self.species]

    @property
    def reaction_ids(self) -> list[str]:
        return [r.id for r in self.reactions]

    def species_index(self, species_id: str) -> int:
        for i, s in enumerate(self.species):
            if s.id == species_id:
                return i
        raise KeyError(f"unknown species {species_id!r}")

    def reaction_index(self, reaction_id: str) -> int:
        for j, r in enumerate(self.reactions):
            if r.id == reaction_id:
                return j
        raise KeyError(f"unknown reaction {reaction_id!r}")

    def get_species(self, species_id: str) -> Species:
        return self.species[self.species_index(species_id)]

    def get_reaction(self, reaction_id: str) -> Reaction:
        return self.reactions[self.reaction_index(reaction_id)]

    @property
    def topology(self) -> Topology:
        index: dict[str, int] = {}
        for i, s in enumerate(self.species):
            index.setdefault(s.id, i)
        rules = []
        for r in self.reactions:
            negated: dict[int, bool] = {}
            for name, neg in r.inputs:
                if name in index:
                    negated[index[name]] = negated.get(index[name], False) or neg
            products = tuple(sorted({index[o] for o in r.outputs if o in index}))
            rules.append(Wiring(tuple(sorted(negated.items())), products))
        return Topology(len(self.species), tuple(rules))

    def copy(self) -> Model:
        return _copy.deepcopy(self)

    # --- checking
    def diagnose(self) -> list[str]:
        out: list[str] = []
        ids = self.species_ids
        invalid = [s for s in ids if not _VALID_ID.fullmatch(s)]
        if invalid:
            out.append(
                "Invalid species name(s) (must begin with a letter and contain only "
                "letters, numbers, and underscores): " + ", ".join(invalid)
            )
        reserved = [s for s in ids if s in _RESERVED_IDS]
        if reserved:
            out.append("Species ID(s) " + ", ".join(reserved) + " clash with rule keywords AND/NOT")
        dup_species = [s for s, c in Counter(ids).items() if c > 1]
        if dup_species:
            out.append("Duplicate species detected: " + ", ".join(dup_species))

        def key(r: Reaction):
            return tuple(sorted(r.inputs)), tuple(sorted(r.outputs))

        counts = Counter(key(r) for r in self.reactions)
        reported = set()
        for r in self.reactions:
            if counts[key(r)] > 1 and key(r) not in reported:
                reported.add(key(r))
                rids = [q.id for q in self.reactions if key(q) == key(r)]
                out.append(f"Duplicate reactions detected: {'/'.join(rids)}: {r.rule}")

        known = set(ids)
        for r, wiring in zip(self.reactions, self.topology.rules):
            unknown = [s for s, _ in r.inputs if s not in known]
            if not wiring.products:
                out.append(f"Reaction {r.id}: {r.rule}: product not recognized")
            elif unknown:
                out.append(f"Reaction {r.id}: {r.rule}: reactant(s) not recognized: {', '.join(unknown)}")
            if wiring.inputs and r.ec50**r.n >= 0.5:
                out.append(f"Reaction {r.id}: {UNDEFINED_HILL}")
        return out
