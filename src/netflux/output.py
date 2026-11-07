"""Output folder: set_output_dir()/using_output_dir() > $NETFLUX_OUTPUT_DIR > ./output.

Relative output paths go inside it; absolute paths are used as given.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "ENV_VAR",
    "get_output_dir",
    "set_output_dir",
    "using_output_dir",
    "output_path",
]

DEFAULT_OUTPUT_DIR = Path("output")
ENV_VAR = "NETFLUX_OUTPUT_DIR"

_override: Path | None = None


def get_output_dir() -> Path:
    if _override is not None:
        return _override
    env = os.environ.get(ENV_VAR)
    return Path(env) if env else DEFAULT_OUTPUT_DIR


def set_output_dir(path: str | Path | None) -> None:
    """``None`` restores the default."""
    global _override
    _override = None if path is None else Path(path)


@contextmanager
def using_output_dir(path: str | Path | None) -> Iterator[Path]:
    global _override
    previous = _override
    if path is not None:
        _override = Path(path)
    try:
        yield get_output_dir()
    finally:
        _override = previous


def output_path(path: str | Path | None = None, default: str | Path | None = None) -> Path:
    """Absolute path to write to; relative paths (or ``default``) go inside the output folder.

    The parent folder is created.
    """
    chosen = path if path is not None else default
    base = get_output_dir()
    if chosen is None:
        resolved = base
    else:
        chosen = Path(chosen)
        resolved = chosen if chosen.is_absolute() else base / chosen
    resolved = resolved.absolute()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
