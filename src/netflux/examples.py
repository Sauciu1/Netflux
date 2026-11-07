"""Bundled data: example models (``Netflux/models``) and the Cytoscape SBGN style.

Wheels package them as ``netflux/data/`` (see ``pyproject.toml``); a source checkout reads them in place.
"""

from __future__ import annotations

from pathlib import Path

from .model import Model

__all__ = ["list_examples", "example_path", "load_example", "MODELS_DIR", "SBGN_STYLE"]

_PACKAGED = Path(__file__).parent / "data"
DATA_ROOT = _PACKAGED if _PACKAGED.is_dir() else Path(__file__).resolve().parents[2] / "Netflux"
MODELS_DIR = DATA_ROOT / "models"
SBGN_STYLE = DATA_ROOT / "Netflux SBGN style.xml"


def _examples() -> dict[str, Path]:
    return {p.stem: p for p in sorted(MODELS_DIR.iterdir()) if p.suffix.lower() in (".xlsx", ".xls")}


def list_examples() -> list[str]:
    return sorted(_examples())


def example_path(name: str) -> Path:
    """Bundled example by name, ignoring case and any extension."""
    by_lower = {stem.lower(): path for stem, path in _examples().items()}
    try:
        return by_lower[Path(name).stem.lower()]
    except KeyError:
        raise KeyError(f"no bundled example {name!r}; available: {', '.join(list_examples())}") from None


def load_example(name: str) -> Model:
    from .formats import read_model  # lazy: formats -> xgmml imports this module

    return read_model(example_path(name))
