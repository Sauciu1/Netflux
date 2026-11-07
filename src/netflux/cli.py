"""Command-line interface (``netflux ...`` or ``python -m netflux``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .codegen import export_python_ode
from .convert import convert_sbml_qual, import_sif
from .cytoscape import write_sbgn_style, write_sif, write_xgmml
from .examples import example_path, list_examples
from .excel import read_model, write_model
from .model import Model, ModelError
from .paths import ENV_VAR, output_path, using_output_dir
from .simulate import Simulator

_REACTION_PARAMS = {"weight": "weight", "w": "weight", "n": "n", "ec50": "ec50"}
_SPECIES_PARAMS = {"ymax": "ymax", "tau": "tau", "y0": "y0", "yinit": "y0"}


def _load(spec: str) -> Model:
    path = Path(spec)
    if path.exists():
        return read_model(path)
    try:
        return read_model(example_path(spec))
    except KeyError:
        raise SystemExit(
            f"netflux: {spec!r} is neither a model file nor a bundled example "
            f"({', '.join(list_examples())})"
        ) from None


def _print_warnings(model: Model) -> None:
    for w in model.warnings:
        print(f"warning: {w}", file=sys.stderr)


def _apply_settings(sim: Simulator, settings: Sequence[str]) -> None:
    model = sim.model
    for item in settings:
        try:
            target, value = item.split("=", 1)
            obj_id, param = target.rsplit(".", 1)
            number = float(value)
        except ValueError:
            raise SystemExit(f"netflux: --set expects ID.param=value, got {item!r}") from None
        key = param.strip().lower()
        if key in _REACTION_PARAMS and obj_id in model.reaction_ids:
            sim.set_reaction(obj_id, **{_REACTION_PARAMS[key]: number})
        elif key in _SPECIES_PARAMS and obj_id in model.species_ids:
            sim.set_species(obj_id, **{_SPECIES_PARAMS[key]: number})
        else:
            raise SystemExit(
                f"netflux: cannot set {item!r}: use a reaction ID with weight/n/ec50 "
                "or a species ID with y0/ymax/tau"
            )


# --- commands
def cmd_examples(args) -> int:
    for name in list_examples():
        print(name)
    return 0


def cmd_info(args) -> int:
    model = _load(args.model)
    print(f"Model: {model.name}")
    print(f"Species: {len(model.species)}  Reactions: {len(model.reactions)}")
    if args.verbose:
        print("\nspecies (ID: y0 ymax tau type)")
        for s in model.species:
            print(f"  {s.id}: {s.y0:g} {s.ymax:g} {s.tau:g} {s.type}")
        print("\nreactions (ID: rule  weight n EC50)")
        for r in model.reactions:
            print(f"  {r.id}: {r.rule}  {r.weight:g} {r.n:g} {r.ec50:g}")
    if model.warnings:
        print(f"\n{len(model.warnings)} warning(s):")
        for w in model.warnings:
            print(f"  - {w}")
    return 0


def cmd_simulate(args) -> int:
    model = _load(args.model)
    _print_warnings(model)
    sim = Simulator(model, time_unit=args.time_unit)
    _apply_settings(sim, args.set or [])
    result = sim.run(args.t_end, args.dt, method=args.method, rtol=args.rtol, atol=args.atol)
    species = args.species.split(",") if args.species else model.species_ids
    unknown = [s for s in species if s not in model.species_ids]
    if unknown:
        raise SystemExit(f"netflux: unknown species: {', '.join(unknown)}")

    if args.output:
        if Path(args.output).suffix.lower() == ".csv":
            out = output_path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_dataframe().to_csv(out)
        else:
            out = result.write_data(args.output)
        print(f"Data written to {_show(out)}")
    else:
        final = result.final()
        print(f"t = {float(result.t[-1]):g}")
        for sid in species:
            print(f"{sid}\t{final[sid]:.4f}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        plot = output_path(args.plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        ax = result.plot(species)
        ax.figure.savefig(plot, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {_show(plot)}")
    return 0


def cmd_export_sif(args) -> int:
    model = _load(args.model)
    sif, attrs = write_sif(model, args.output, args.name)
    print(f"Wrote {_show(sif)}\nWrote {_show(attrs)}")
    return 0


def cmd_export_xgmml(args) -> int:
    model = _load(args.model)
    path = write_xgmml(model, args.output, args.reference)
    print(f"Model exported to: {_show(path)}")
    if args.style:
        print(f"Style written to: {_show(write_sbgn_style(args.style))}")
    return 0


def cmd_export_python(args) -> int:
    model = _load(args.model)
    print(f"Wrote {_show(export_python_ode(model, args.output))}")
    return 0


def cmd_convert_sif(args) -> int:
    source = Path(args.sif)
    model = import_sif(source)
    _print_warnings(model)
    output = write_model(model, args.output)
    print(f"Model saved as: {_show(output)}")
    return 0


def cmd_convert_sbml(args) -> int:
    folder = convert_sbml_qual(args.sbml, args.output)
    print(f"Conversion complete. Saved to: {_show(folder)}")
    return 0


def _show(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# --- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netflux",
        description="Logic-based ODE modeling of biological networks (Python port of Netflux).",
    )
    parser.add_argument("--version", action="version", version=f"netflux {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    model_help = "Netflux .xlsx/.xls file or bundled example name (see 'netflux examples')"
    writes = argparse.ArgumentParser(add_help=False)
    writes.add_argument(
        "--output-dir", metavar="DIR",
        help=f"folder for all written files; relative -o paths go inside it "
        f"(default: ${ENV_VAR}, else ./output)",
    )

    p = sub.add_parser("examples", help="list bundled example models")
    p.set_defaults(func=cmd_examples)

    p = sub.add_parser("info", help="summarize a model and show its warnings")
    p.add_argument("model", help=model_help)
    p.add_argument("-v", "--verbose", action="store_true", help="list species and reactions")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("simulate", help="simulate a model from t=0", parents=[writes])
    p.add_argument("model", help=model_help)
    p.add_argument("--t-end", type=float, default=10.0, help="simulation time (default 10)")
    p.add_argument("--dt", type=float, default=0.1, help="output time step (default 0.1)")
    p.add_argument(
        "--set", action="append", metavar="ID.PARAM=VALUE",
        help="override a parameter, e.g. r1.weight=1 or A.tau=2 (repeatable)",
    )
    p.add_argument("--species", help="comma-separated species to print/plot (default: all)")
    p.add_argument("--method", default="dopri5", help="torchdiffeq method (default dopri5)")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--time-unit", default="a.u.", help="label for the time axis")
    p.add_argument("-o", "--output", help="write data: .csv, or Netflux 'Export Data' text otherwise")
    p.add_argument("--plot", metavar="FILE", help="save a plot (needs netflux[plot])")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser(
        "export-sif", help="export Cytoscape .sif + _nodeAttributes.txt", parents=[writes]
    )
    p.add_argument("model", help=model_help)
    p.add_argument("-o", "--output", help="subfolder of the output dir (default: the output dir)")
    p.add_argument("--name", help="base file name (default: model name)")
    p.set_defaults(func=cmd_export_sif)

    p = sub.add_parser("export-xgmml", help="export Cytoscape XGMML", parents=[writes])
    p.add_argument("model", help=model_help)
    p.add_argument("-o", "--output", help="output .xgmml file (default: <model name>.xgmml)")
    p.add_argument("--reference", help="XGMML exported from Cytoscape whose layout should be kept")
    p.add_argument(
        "--style", nargs="?", const="netflux_sbgn_style.xml", metavar="FILE",
        help="also write the Netflux SBGN style XML (default name netflux_sbgn_style.xml)",
    )
    p.set_defaults(func=cmd_export_xgmml)

    p = sub.add_parser("export-python", help="export standalone PyTorch ODE code", parents=[writes])
    p.add_argument("model", help=model_help)
    p.add_argument("-o", "--output", help="output .py file (default: NetfluxODE.py)")
    p.set_defaults(func=cmd_export_python)

    p = sub.add_parser(
        "convert-sif", help="convert a SIF network to a Netflux .xlsx model", parents=[writes]
    )
    p.add_argument("sif", help="input .sif file")
    p.add_argument("-o", "--output", help="output .xlsx (default: <sif file name>.xlsx)")
    p.set_defaults(func=cmd_convert_sif)

    p = sub.add_parser(
        "convert-sbml", help="convert SBML-qual to .xlsx + .sif (needs netflux[sbml])",
        parents=[writes],
    )
    p.add_argument("sbml", help="input SBML-qual .xml file")
    p.add_argument("-o", "--output", help="output folder (default: <file stem>)")
    p.set_defaults(func=cmd_convert_sbml)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with using_output_dir(getattr(args, "output_dir", None)):
            return args.func(args)
    except (ModelError, ImportError, FileNotFoundError, ValueError) as exc:
        print(f"netflux: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
