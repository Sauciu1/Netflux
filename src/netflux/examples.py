"""Example models bundled with the package."""

from __future__ import annotations

from pathlib import Path

from ._data import MODELS_DIR
from .excel import read_model
from .model import Model

__all__ = ["list_examples", "example_path", "load_example"]

_SUFFIXES = (".xlsx", ".xls")


def list_examples() -> list[str]:
    return sorted(
        Path(entry.name).stem
        for entry in MODELS_DIR.iterdir()
        if entry.name.lower().endswith(_SUFFIXES)
    )


def example_path(name: str) -> Path:
    wanted = Path(name).stem.lower()
    for entry in MODELS_DIR.iterdir():
        if entry.name.lower().endswith(_SUFFIXES) and Path(entry.name).stem.lower() == wanted:
            return entry
    raise KeyError(f"no bundled example {name!r}; available: {', '.join(list_examples())}")


def load_example(name: str) -> Model:
    return read_model(example_path(name))
