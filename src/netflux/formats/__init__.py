"""Read and write models in any supported format, chosen by file suffix."""

from __future__ import annotations

from pathlib import Path

from ..model import Model
from ..output import output_path
from .excel import read_excel, write_excel
from .python import write_python
from .sbml import read_sbml, write_sbml
from .sif import read_sif, write_sif
from .xgmml import write_xgmml

__all__ = ["read_model", "write_model", "READERS", "WRITERS"]

READERS = {".xlsx": read_excel, ".xls": read_excel, ".xml": read_sbml, ".sbml": read_sbml, ".sif": read_sif}
WRITERS = {
    ".xlsx": write_excel, ".xml": write_sbml, ".sbml": write_sbml,
    ".sif": write_sif, ".xgmml": write_xgmml, ".py": write_python,
}


def read_model(path: str | Path) -> Model:
    """Excel (``.xlsx``/``.xls``), SBML-qual (``.xml``/``.sbml``) or SIF."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"cannot read {path.name!r}; use one of {', '.join(READERS)}")
    return reader(path)


def write_model(model: Model, path: str | Path | None = None, **options) -> Path:
    """Write by suffix (default ``<name>.xlsx``); ``options`` go to the writer, e.g. ``reference=`` for XGMML."""
    path = output_path(path, f"{model.name}.xlsx")
    writer = WRITERS.get(path.suffix.lower())
    if writer is None:
        raise ValueError(f"cannot write {path.name!r}; use one of {', '.join(WRITERS)}")
    return writer(model, path, **options)
