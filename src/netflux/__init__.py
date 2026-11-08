"""Netflux: logic-based ODE models of biological networks (Python/PyTorch port)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netflux")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from .examples import list_examples, load_example
from .formats import read_model, write_model
from .formats.excel import read_excel_frames
from .formats.sbml import from_sbml, to_sbml
from .formats.sif import to_node_attributes, to_sif
from .formats.xgmml import to_xgmml, write_sbgn_style, write_xgmml
from .model import (
    Model,
    ModelError,
    NetfluxWarning,
    Reaction,
    Species,
    Topology,
    parse_rule,
)
from .output import (
    DEFAULT_OUTPUT_DIR,
    get_output_dir,
    output_path,
    set_output_dir,
    using_output_dir,
)
from .simulate import NetfluxODE, SimulationResult, Simulator, hill, simulate

__all__ = [
    "__version__",
    "DEFAULT_OUTPUT_DIR",
    "Model",
    "ModelError",
    "NetfluxODE",
    "NetfluxWarning",
    "Reaction",
    "SimulationResult",
    "Simulator",
    "Species",
    "Topology",
    "from_sbml",
    "get_output_dir",
    "hill",
    "list_examples",
    "load_example",
    "output_path",
    "parse_rule",
    "read_excel_frames",
    "read_model",
    "set_output_dir",
    "simulate",
    "to_node_attributes",
    "to_sif",
    "to_xgmml",
    "to_sbml",
    "using_output_dir",
    "write_model",
    "write_sbgn_style",
    "write_xgmml",
]
