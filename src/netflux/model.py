"""Netflux model data structures."""

from __future__ import annotations

import copy as _copy
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .rules import RULE_ERRORS, Topology, compile_rules

__all__ = [
    "Species",
    "Reaction",
    "Model",
    "ModelError",
    "NetfluxWarning",
    "DEFAULT_SPECIES_PARAMS",
    "DEFAULT_REACTION_PARAMS",
]

DEFAULT_SPECIES_PARAMS = {"y0": 0.0, "ymax": 1.0, "tau": 1.0}
DEFAULT_REACTION_PARAMS = {"weight": 1.0, "n": 1.4, "ec50": 0.5}

_VALID_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_RESERVED_IDS = {"AND", "NOT"}


class ModelError(ValueError):
    """Raised when a model file cannot be read at all."""


class NetfluxWarning(UserWarning):
    """Recoverable model problems."""


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
    id: str
    rule: str
    weight: float = DEFAULT_REACTION_PARAMS["weight"]
    n: float = DEFAULT_REACTION_PARAMS["n"]
    ec50: float = DEFAULT_REACTION_PARAMS["ec50"]
    module: str = ""


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
        return compile_rules(self.species_ids, [r.rule for r in self.reactions])

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
            out.append(
                "Species ID(s) " + ", ".join(reserved) + " clash with rule keywords AND/NOT"
            )
        dup_species = [s for s, c in Counter(ids).items() if c > 1]
        if dup_species:
            out.append("Duplicate species detected: " + ", ".join(dup_species))
        rules = [r.rule for r in self.reactions]
        dup_rules = [r for r, c in Counter(rules).items() if c > 1]
        for rule in dup_rules:
            rids = [r.id for r in self.reactions if r.rule == rule]
            out.append(f"Duplicate reactions detected: {'/'.join(rids)}: {rule}")
        for rxn, parsed in zip(self.reactions, self.topology.rules):
            if parsed.error is not None:
                out.append(f"Reaction {rxn.id}: {rxn.rule}: {RULE_ERRORS[parsed.error]}")
        out.extend(self.parameter_warnings())
        return out

    def parameter_warnings(
        self, weight=None, n=None, ec50=None
    ) -> list[str]:
        """Warn where ``EC50**n >= 0.5`` (undefined Hill curve); ``n``/``ec50`` override the model's."""
        n = [r.n for r in self.reactions] if n is None else list(n)
        ec50 = [r.ec50 for r in self.reactions] if ec50 is None else list(ec50)
        out = []
        for rxn, parsed, nj, ej in zip(self.reactions, self.topology.rules, n, ec50):
            if parsed.reactants and float(ej) ** float(nj) >= 0.5:
                out.append(
                    f"Reaction {rxn.id}: EC50={float(ej):g} and n={float(nj):g} give "
                    "EC50^n >= 0.5, so the normalized Hill function is undefined"
                )
        return out

    # --- io
    @classmethod
    def from_file(cls, path: str | Path) -> Model:
        from .excel import read_model

        return read_model(path)

    def to_excel(self, path: str | Path | None = None) -> Path:
        from .excel import write_model

        return write_model(self, path)
