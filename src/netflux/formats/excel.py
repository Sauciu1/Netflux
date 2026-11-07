"""Port of xls2Netflux.m / getNetfluxParams.m / writeXLSX.m.

Sheets (data from row 3): species = A module, B ID, C name, D Yinit, E Ymax, F tau,
G type, H location, I ref; reactions = A module, B ID, C rule, D weight, E n, F EC50.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from ..model import (
    DEFAULT_REACTION_PARAMS,
    DEFAULT_SPECIES_PARAMS,
    Model,
    ModelError,
    Reaction,
    Species,
    number,
    parse_rule,
    with_defaults,
)
from ..output import output_path

__all__ = ["read_excel", "write_excel"]

_FIRST_DATA_ROW = 2  # zero-based: Excel row 3

SPECIES_HEADERS = ["Module", "ID", "name", "Yinit", "Ymax", "tau", "type", "location", "ref", "notes"]
REACTION_HEADERS = [
    "module", "ID", "Rule", "Weight", "n", "EC50", "location",
    "confidence 1", "species 1", "system 1", "database 1", "refID 1", "link 1",
    "confidence 2", "species 2", "system 2", "database 2", "refID 2", "link 2",
]


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _params(row, columns: dict[str, int], defaults: dict[str, float], label: str, warnings: list[str]):
    return with_defaults({k: number(_get(row, c)) for k, c in columns.items()}, defaults, label, warnings)


def _find_sheet(sheets: dict[str, pd.DataFrame], name: str, path: Path) -> pd.DataFrame:
    for key, frame in sheets.items():
        if str(key).strip().lower() == name:
            return frame
    raise ModelError(
        f"{path.name}: no {name!r} sheet found (sheets: {', '.join(map(str, sheets))})"
    )


def _rows(frame: pd.DataFrame) -> list[list[Any]]:
    return frame.iloc[_FIRST_DATA_ROW:].astype(object).values.tolist()


def _get(row: list[Any], col: int) -> Any:
    return row[col] if col < len(row) else None


def read_excel(path: str | Path) -> Model:
    """Missing parameters get MATLAB defaults; problems are listed in ``Model.warnings``."""
    path = Path(path)
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="calamine")
    except FileNotFoundError:
        raise
    except Exception as exc:  # pandas raises a variety of engine errors
        raise ModelError(f"could not read {path}: {exc}") from exc

    warnings: list[str] = []

    species: list[Species] = []
    for row in _rows(_find_sheet(sheets, "species", path)):
        sid = _text(_get(row, 1))
        if not sid:
            continue
        params = _params(row, {"y0": 3, "ymax": 4, "tau": 5}, DEFAULT_SPECIES_PARAMS, f"Species {sid}", warnings)
        species.append(
            Species(
                id=sid,
                name=_text(_get(row, 2)),
                type=_text(_get(row, 6)),
                location=_text(_get(row, 7)),
                module=_text(_get(row, 0)),
                ref=_text(_get(row, 8)),
                **params,
            )
        )
    if not species:
        raise ModelError(
            f"{path.name}: number of species is zero; check the 'species' sheet "
            "(IDs in column B, starting on row 3)"
        )

    reactions: list[Reaction] = []
    for row in _rows(_find_sheet(sheets, "reactions", path)):
        rule = _text(_get(row, 2))
        if not rule:
            continue
        rid = _text(_get(row, 1))
        try:
            inputs, outputs = parse_rule(rule)
        except ModelError as exc:
            warnings.append(f"Reaction {rid}: {exc}; reaction skipped")
            continue
        params = _params(
            row, {"weight": 3, "n": 4, "ec50": 5}, DEFAULT_REACTION_PARAMS, f"Reaction {rid}: {rule}", warnings
        )
        reactions.append(Reaction(rid, inputs, outputs, module=_text(_get(row, 0)), **params))

    model = Model(name=path.stem, species=species, reactions=reactions)
    model.warnings = warnings + model.diagnose()
    return model


def write_excel(model: Model, path: str | Path | None = None) -> Path:
    path = output_path(path, f"{model.name}.xlsx")

    species_rows = [["Species"], SPECIES_HEADERS]
    species_rows += [[s.module, s.id, s.name or s.id, s.y0, s.ymax, s.tau, s.type, s.location, s.ref]
                     for s in model.species]
    reaction_rows = [["Reactions"], REACTION_HEADERS]
    reaction_rows += [[r.module, r.id, r.rule, r.weight, r.n, r.ec50] for r in model.reactions]

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for title, rows in (("species", species_rows), ("reactions", reaction_rows)):
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append([None if v == "" else v for v in row])
        for row in sheet.iter_rows():
            for cell in row:
                # keep rules like "=> A" as text, not formulas
                if isinstance(cell.value, str) and cell.data_type == "f":
                    cell.data_type = "s"
                    cell.quotePrefix = True
    workbook.save(path)
    return path
