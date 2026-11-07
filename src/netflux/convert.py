"""Port of importsif.m / sbml2sif.m: SIF and SBML-qual import with writeXLSX.m defaults."""

from __future__ import annotations

import re
from pathlib import Path

from .cytoscape import write_sif
from .excel import write_model
from .model import Model, ModelError, Reaction, Species
from .paths import output_path

__all__ = ["import_sif", "import_sbml_qual", "convert_sbml_qual", "model_from_rules"]

_CONNECTOR = re.compile(r"rcn\d+")


def model_from_rules(name: str, species_ids: list[str], rules: list[str]) -> Model:
    species = [Species(id=s, name=s) for s in species_ids]
    reactions = [Reaction(id=f"r{k}", rule=rule) for k, rule in enumerate(rules, start=1)]
    model = Model(name=name, species=species, reactions=reactions)
    model.warnings = model.diagnose()
    return model


def _signed(name: str, sign: str) -> str:
    return f"!{name}" if sign.strip() == "-1" else name


def import_sif(path: str | Path, name: str | None = None) -> Model:
    """``rcn<k>`` connector nodes become AND rules, as Netflux's SIF export writes them."""
    path = Path(path)
    lines: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != 3:
            parts = raw.split()
        if len(parts) != 3:
            raise ModelError(f"{path.name}:{lineno}: expected 'source<TAB>sign<TAB>target', got {raw!r}")
        lines.append(tuple(p.strip() for p in parts))  # type: ignore[arg-type]

    def is_connector(node: str) -> bool:
        return _CONNECTOR.fullmatch(node) is not None

    nodes = {n for src, _, tgt in lines for n in (src, tgt)}
    species_ids = sorted(n for n in nodes if not is_connector(n))

    inputs: dict[str, list[str]] = {}
    for src, sign, tgt in lines:
        if is_connector(tgt):
            inputs.setdefault(tgt, []).append(_signed(src, sign))

    rules: list[str] = []
    extra_warnings: list[str] = []
    for src, sign, tgt in lines:
        if is_connector(tgt):
            continue  # consumed by its connector's output line
        if is_connector(src):
            gate_inputs = inputs.get(src)
            if not gate_inputs:
                extra_warnings.append(f"Connector {src} has no inputs; skipped")
                continue
            rules.append(f"{' & '.join(gate_inputs)} => {tgt}")
        else:
            rules.append(f"{_signed(src, sign)} => {tgt}")

    model = model_from_rules(name or path.stem, species_ids, rules)
    model.warnings = extra_warnings + model.warnings
    return model


def _libsbml():
    try:
        import libsbml
    except ImportError as exc:
        raise ImportError(
            "SBML-qual import needs python-libsbml: pip install 'netflux[sbml]'"
        ) from exc
    return libsbml


def import_sbml_qual(path: str | Path, name: str | None = None) -> Model:
    """One reaction per transition input; ``qual_`` is stripped from IDs, as in MATLAB."""
    libsbml = _libsbml()
    path = Path(path)
    doc = libsbml.readSBMLFromFile(str(path))
    if doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL) or doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR):
        log = doc.getErrorLog().toString().strip()
        raise ModelError(f"could not read SBML file {path}: {log}")
    sbml_model = doc.getModel()
    qual = sbml_model.getPlugin("qual") if sbml_model is not None else None
    if qual is None:
        raise ModelError(f"{path.name} is not an SBML-qual model (no qual plugin)")

    def clean(species_ref: str) -> str:
        return species_ref.replace("qual_", "")

    rules: list[str] = []
    used: set[str] = set()
    extra_warnings: list[str] = []
    for transition in qual.getListOfTransitions():
        ins = list(transition.getListOfInputs())
        outs = list(transition.getListOfOutputs())
        if len(ins) > 1:
            extra_warnings.append(
                f"Transition {transition.getId() or '?'} has {len(ins)} inputs; "
                "imported as one reaction per input"
            )
        for out in outs:
            target = clean(out.getQualitativeSpecies())
            used.add(target)
            for inp in ins:
                source = clean(inp.getQualitativeSpecies())
                used.add(source)
                negative = inp.isSetSign() and inp.getSign() == libsbml.INPUT_SIGN_NEGATIVE
                rules.append(f"{'!' if negative else ''}{source} => {target}")

    model = model_from_rules(name or path.stem, sorted(used), rules)
    model.warnings = extra_warnings + model.warnings
    return model


def convert_sbml_qual(path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Write ``<stem> Netflux.xlsx``, ``.sif`` and ``_nodeAttributes.txt`` into ``output_dir`` (default ``<stem>``)."""
    path = Path(path)
    folder = output_path(output_dir, path.stem)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{path.stem} Netflux"
    model = import_sbml_qual(path, name=name)
    write_model(model, folder / f"{name}.xlsx")
    write_sif(model, folder, name)
    return folder
