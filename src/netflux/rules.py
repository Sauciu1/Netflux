"""Reaction rules such as ``"A & !B => C"`` (also ``->``, ``+``/``AND`` and ``NOT``)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

__all__ = ["ParsedRule", "Topology", "parse_rule", "compile_rules"]

_ARROW = re.compile(r"=>|->")
_SEPARATOR = re.compile(r"&|\+|\bAND\b")
_NEGATION = re.compile(r"^(?:!|NOT\s)\s*")


@dataclass(frozen=True)
class ParsedRule:
    inputs: tuple[tuple[int, bool], ...]  # (species index, negated), in species order
    products: tuple[int, ...]
    error: str | None = None


@dataclass(frozen=True)
class Topology:
    n_species: int
    rules: tuple[ParsedRule, ...]

    @property
    def producers(self) -> tuple[tuple[int, ...], ...]:
        """Reactions producing each species, in reaction order."""
        out: list[list[int]] = [[] for _ in range(self.n_species)]
        for j, rule in enumerate(self.rules):
            for i in rule.products:
                out[i].append(j)
        return tuple(tuple(p) for p in out)


def _terms(text: str) -> list[str]:
    return [t.strip() for t in _SEPARATOR.split(text) if t.strip()]


def _parse(rule: str, index: dict[str, int]) -> ParsedRule:
    arrow = _ARROW.search(rule)
    if arrow is None:
        return ParsedRule((), (), "no reaction arrow ('=>') could be identified")

    negated: dict[int, bool] = {}
    unknown = []
    for term in _terms(rule[: arrow.start()]):
        neg = _NEGATION.match(term)
        name = term[neg.end():] if neg else term
        if name in index:
            negated[index[name]] = negated.get(index[name], False) or neg is not None
        else:
            unknown.append(name)
    products = tuple(sorted({index[t] for t in _terms(rule[arrow.end():]) if t in index}))

    error = None
    if not products:
        error = "product not recognized"
    elif unknown:
        error = f"reactant(s) not recognized: {', '.join(unknown)}"
    return ParsedRule(tuple(sorted(negated.items())), products, error)


def _index(species_ids: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, sid in enumerate(species_ids):
        index.setdefault(sid, i)
    return index


def parse_rule(rule: str, species_ids: Sequence[str]) -> ParsedRule:
    return _parse(rule, _index(species_ids))


@lru_cache(maxsize=256)
def _compile_cached(species_ids: tuple[str, ...], rules: tuple[str, ...]) -> Topology:
    index = _index(species_ids)
    return Topology(len(species_ids), tuple(_parse(r, index) for r in rules))


def compile_rules(species_ids: Sequence[str], rules: Sequence[str]) -> Topology:
    """Cached on the species IDs and rules."""
    return _compile_cached(tuple(species_ids), tuple(rules))
