"""Port of the rule parser in xls2Netflux.m: ``"A & !B => C"`` (also ``->``, ``AND``, ``NOT``).

Unlike MATLAB, ``!AB`` does not negate a species ``A``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "ParsedRule",
    "Topology",
    "parse_rule",
    "compile_rules",
    "RULE_ERRORS",
]

_WORD = "A-Za-z0-9_"
_TOKEN = re.compile(rf"[{_WORD}]+")
_ARROW = re.compile(r"=>|->")
_SEPARATOR = re.compile(rf"&|\+|(?<![{_WORD}])AND(?![{_WORD}])")
_NEGATION_BEFORE = re.compile(rf"(?:!|(?<![{_WORD}])NOT\s)\s*$")
_WORD_ONLY = re.compile(rf"[{_WORD}]+")

RULE_ERRORS = {
    "no_arrow": "no reaction arrow ('=>') could be identified",
    "no_products": "product not recognized",
    "reactants_not_recognized": (
        "one or more reactants not recognized; make sure all species appear "
        "exactly as they are in the species sheet"
    ),
}


@dataclass(frozen=True)
class ParsedRule:
    """Species indices in species order; ``negated`` parallels ``reactants``."""

    reactants: tuple[int, ...]
    negated: tuple[bool, ...]
    products: tuple[int, ...]
    error: str | None = None


@dataclass(frozen=True)
class Topology:
    n_species: int
    rules: tuple[ParsedRule, ...]

    @property
    def n_reactions(self) -> int:
        return len(self.rules)

    @property
    def producers(self) -> tuple[tuple[int, ...], ...]:
        """Reactions producing each species, in reaction order."""
        out: list[list[int]] = [[] for _ in range(self.n_species)]
        for j, rule in enumerate(self.rules):
            for i in rule.products:
                out[i].append(j)
        return tuple(tuple(p) for p in out)


def _is_negated(lhs: str, start: int) -> bool:
    return _NEGATION_BEFORE.search(lhs, 0, start) is not None


def _occurrences(
    rule: str,
    word_index: dict[str, list[int]],
    other_ids: Sequence[tuple[int, str]],
) -> list[tuple[int, int, int]]:
    hits: list[tuple[int, int, int]] = []
    for m in _TOKEN.finditer(rule):
        for i in word_index.get(m.group(), ()):
            hits.append((i, m.start(), m.end()))
    # invalid IDs (non-word characters) need a regex for MATLAB-style whole-word matching
    for i, sid in other_ids:
        pat = re.compile(rf"(?<![{_WORD}]){re.escape(sid)}(?![{_WORD}])")
        hits.extend((i, m.start(), m.end()) for m in pat.finditer(rule))
    return hits


def _parse(
    rule: str,
    word_index: dict[str, list[int]],
    other_ids: Sequence[tuple[int, str]],
) -> ParsedRule:
    arrow = _ARROW.search(rule)
    if arrow is None:
        return ParsedRule((), (), (), "no_arrow")
    eq = arrow.start()
    lhs = rule[:eq]

    reactant_neg: dict[int, bool] = {}
    products: set[int] = set()
    masked = list(lhs)  # LHS with species names blanked out, for counting separators
    for i, start, end in _occurrences(rule, word_index, other_ids):
        if start < eq:
            reactant_neg[i] = reactant_neg.get(i, False) or _is_negated(lhs, start)
            masked[start:min(end, eq)] = " " * (min(end, eq) - start)
        elif start > eq:
            products.add(i)

    reactants = tuple(sorted(reactant_neg))
    negated = tuple(reactant_neg[i] for i in reactants)
    prods = tuple(sorted(products))

    error = None
    if not prods:
        error = "no_products"
    elif lhs.strip():
        expected = len(_SEPARATOR.findall("".join(masked))) + 1
        if not reactants or expected > len(reactants):
            error = "reactants_not_recognized"
    return ParsedRule(reactants, negated, prods, error)


def _index(species_ids: Sequence[str]):
    word_index: dict[str, list[int]] = {}
    other_ids: list[tuple[int, str]] = []
    for i, sid in enumerate(species_ids):
        if not sid:
            continue
        if _WORD_ONLY.fullmatch(sid):
            word_index.setdefault(sid, []).append(i)
        else:
            other_ids.append((i, sid))
    return word_index, other_ids


def parse_rule(rule: str, species_ids: Sequence[str]) -> ParsedRule:
    word_index, other_ids = _index(species_ids)
    return _parse(rule, word_index, other_ids)


@lru_cache(maxsize=256)
def _compile_cached(species_ids: tuple[str, ...], rules: tuple[str, ...]) -> Topology:
    word_index, other_ids = _index(species_ids)
    return Topology(
        n_species=len(species_ids),
        rules=tuple(_parse(r, word_index, other_ids) for r in rules),
    )


def compile_rules(species_ids: Sequence[str], rules: Sequence[str]) -> Topology:
    """Cached on the species IDs and rules."""
    return _compile_cached(tuple(species_ids), tuple(rules))
