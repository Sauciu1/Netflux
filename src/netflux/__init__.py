"""Netflux: logic-based ODE models of biological networks (Python/PyTorch port)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netflux")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from .codegen import export_python_ode, python_ode_source
from .convert import convert_sbml_qual, import_sbml_qual, import_sif
from .cytoscape import to_node_attributes, to_sif, to_xgmml, write_sbgn_style, write_sif, write_xgmml
from .examples import list_examples, load_example
from .excel import read_model, write_model
from .model import Model, ModelError, NetfluxWarning, Reaction, Species
from .ode import NetfluxODE, hill
from .paths import DEFAULT_OUTPUT_DIR, get_output_dir, output_path, set_output_dir, using_output_dir
from .rules import ParsedRule, Topology, parse_rule
from .simulate import SimulationResult, Simulator, simulate


def main() -> int:
    from .cli import main as _main

    return _main()


__all__ = [
    "__version__",
    "DEFAULT_OUTPUT_DIR",
    "Model",
    "ModelError",
    "NetfluxODE",
    "NetfluxWarning",
    "ParsedRule",
    "Reaction",
    "SimulationResult",
    "Simulator",
    "Species",
    "Topology",
    "convert_sbml_qual",
    "export_python_ode",
    "hill",
    "get_output_dir",
    "import_sbml_qual",
    "import_sif",
    "list_examples",
    "load_example",
    "main",
    "output_path",
    "parse_rule",
    "python_ode_source",
    "read_model",
    "set_output_dir",
    "simulate",
    "to_node_attributes",
    "to_sif",
    "to_xgmml",
    "using_output_dir",
    "write_model",
    "write_sbgn_style",
    "write_sif",
    "write_xgmml",
]
