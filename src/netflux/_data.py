"""Bundled data: ``Netflux/models`` and ``Netflux/Netflux SBGN style.xml`` from the repository.

Wheels package them as ``netflux/data/`` (see ``pyproject.toml``); a source checkout reads them in place.
"""

from pathlib import Path

_PACKAGED = Path(__file__).parent / "data"
DATA_ROOT = _PACKAGED if _PACKAGED.is_dir() else Path(__file__).resolve().parents[2] / "Netflux"
MODELS_DIR = DATA_ROOT / "models"
SBGN_STYLE = DATA_ROOT / "Netflux SBGN style.xml"
