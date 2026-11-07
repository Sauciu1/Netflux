"""SBML-qual model files (the standard for logical models), read and written with python-libsbml.

Netflux maps onto SBML-qual as follows:

* a species is a ``qualitativeSpecies`` (Boolean, ``maxLevel=1``);
* each species produced by a reaction gets one ``transition``; every Netflux reaction producing it
  is one ``functionTerm`` (several terms with ``resultLevel=1`` mean OR), whose math is the AND of
  ``X == 1`` (or ``X == 0`` for a negated input);
* parameters SBML-qual has no place for (y0, ymax, tau, weight, n, EC50, ...) are stored in
  ``<annotation>`` elements in the Netflux namespace, so other tools simply ignore them.

Files from other tools are read too: function-term logic is converted to OR-of-AND form, and
each AND clause becomes one Netflux reaction with default parameters.
"""

from __future__ import annotations

import operator
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

import libsbml
from sympy import And, Not, Or, Symbol, Xor, false, true
from sympy.logic.boolalg import to_dnf

from ..model import (
    DEFAULT_REACTION_PARAMS,
    DEFAULT_SPECIES_PARAMS,
    Model,
    ModelError,
    Reaction,
    Species,
    number,
    with_defaults,
)
from ..output import output_path

__all__ = ["read_sbml", "write_sbml", "from_sbml", "to_sbml", "NAMESPACE"]

NAMESPACE = "https://github.com/Sauciu1/Netflux/sbml-annotation"

_SPECIES_TEXT = ("type", "location", "module", "ref")
_COMPARISONS = {
    libsbml.AST_RELATIONAL_EQ: operator.eq,
    libsbml.AST_RELATIONAL_NEQ: operator.ne,
    libsbml.AST_RELATIONAL_GEQ: operator.ge,
    libsbml.AST_RELATIONAL_GT: operator.gt,
    libsbml.AST_RELATIONAL_LEQ: operator.le,
    libsbml.AST_RELATIONAL_LT: operator.lt,
}
_CONNECTIVES = {
    libsbml.AST_LOGICAL_AND: And,
    libsbml.AST_LOGICAL_OR: Or,
    libsbml.AST_LOGICAL_XOR: Xor,
}


def _problems(doc) -> tuple[list[str], list[str]]:
    """libsbml's read and consistency findings as (errors, warnings), with line numbers.

    Unit checks are skipped: qualitative models have no units.
    """
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    doc.checkConsistency()
    errors, warnings = [], []
    for i in range(doc.getNumErrors()):
        problem = doc.getError(i)
        full = " ".join(problem.getMessage().split())
        # libsbml messages are "<rule text> Reference: <spec section> <specific detail>"
        detail = re.split(r"Section [\d.]*\d", full.split("Reference:", 1)[1])[-1].strip() if "Reference:" in full else ""
        text = f"{problem.getShortMessage().strip()} {detail}".strip() if detail else full
        (errors if problem.getSeverity() >= libsbml.LIBSBML_SEV_ERROR else warnings).append(
            f"line {problem.getLine()}: {text}")
    return errors, warnings


def _error_summary(errors: list[str], limit: int = 5) -> str:
    shown = "\n  ".join(errors[:limit])
    more = f"\n  ... and {len(errors) - limit} more" if len(errors) > limit else ""
    return f"{len(errors)} error(s):\n  {shown}{more}"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


# --- writing


def _ast(kind: int, *children: libsbml.ASTNode) -> libsbml.ASTNode:
    node = libsbml.ASTNode(kind)
    for child in children:
        node.addChild(child)
    return node


def _level_is(species_id: str, level: int) -> libsbml.ASTNode:
    name = libsbml.ASTNode(libsbml.AST_NAME)
    name.setName(species_id)
    value = libsbml.ASTNode(libsbml.AST_INTEGER)
    value.setValue(level)
    return _ast(libsbml.AST_RELATIONAL_EQ, name, value)


def _annotation(tag: str, attributes: dict[str, str]) -> str:
    attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in attributes.items() if v != "")
    return f'<netflux:{tag} xmlns:netflux="{NAMESPACE}" {attrs}/>'


def _model_id(name: str) -> str:
    sid = re.sub(r"\W", "_", name) or "netflux"
    return sid if libsbml.SyntaxChecker.isValidSBMLSId(sid) else f"m_{sid}"


def to_sbml(model: Model) -> str:
    """The model as an SBML Level 3 qual document."""
    ids = model.species_ids
    bad = [s for s in ids if not libsbml.SyntaxChecker.isValidSBMLSId(s)]
    bad += sorted({s for s in ids if ids.count(s) > 1})
    if bad:
        raise ModelError(f"species IDs must be unique, valid SBML identifiers: {', '.join(bad)}")
    known = set(ids)
    for r in model.reactions:
        unknown = [s for s, _ in r.inputs if s not in known] + [s for s in r.outputs if s not in known]
        if unknown:
            raise ModelError(f"cannot write reaction {r.id} ({r.rule!r}) as SBML: unknown species {', '.join(unknown)}")

    doc = libsbml.SBMLDocument(libsbml.SBMLNamespaces(3, 1, "qual", 1))
    doc.setPackageRequired("qual", True)
    sbml_model = doc.createModel()
    sbml_model.setId(_model_id(model.name))
    sbml_model.setName(model.name)
    compartment = sbml_model.createCompartment()
    compartment.setId("cell")
    compartment.setConstant(True)
    qual = sbml_model.getPlugin("qual")

    for s in model.species:
        qs = qual.createQualitativeSpecies()
        qs.setId(s.id)
        if s.name:
            qs.setName(s.name)
        qs.setCompartment("cell")
        qs.setConstant(False)
        qs.setMaxLevel(1)
        qs.setInitialLevel(1 if s.y0 >= 0.5 else 0)
        qs.setAnnotation(_annotation("species", {
            "y0": _num(s.y0), "ymax": _num(s.ymax), "tau": _num(s.tau),
            **{k: getattr(s, k) for k in _SPECIES_TEXT},
        }))

    rules = model.topology.rules
    for i, producers in enumerate(model.topology.producers):
        if not producers:
            continue
        target = ids[i]
        transition = qual.createTransition()
        transition.setId(f"tr_{target}")
        signs: dict[int, set[bool]] = {}
        for j in producers:
            for k, negated in rules[j].inputs:
                signs.setdefault(k, set()).add(negated)
        for k in sorted(signs):
            inp = transition.createInput()
            inp.setId(f"tr_{target}_{ids[k]}")
            inp.setQualitativeSpecies(ids[k])
            inp.setTransitionEffect(libsbml.INPUT_TRANSITION_EFFECT_NONE)
            sign = signs[k]
            inp.setSign(libsbml.INPUT_SIGN_DUAL if len(sign) == 2
                        else libsbml.INPUT_SIGN_NEGATIVE if True in sign else libsbml.INPUT_SIGN_POSITIVE)
        out = transition.createOutput()
        out.setQualitativeSpecies(target)
        out.setTransitionEffect(libsbml.OUTPUT_TRANSITION_EFFECT_ASSIGNMENT_LEVEL)
        transition.createDefaultTerm().setResultLevel(0)
        for j in producers:
            r = model.reactions[j]
            term = transition.createFunctionTerm()
            term.setResultLevel(1)
            literals = [_level_is(ids[k], 0 if negated else 1) for k, negated in rules[j].inputs]
            if not literals:
                term.setMath(libsbml.ASTNode(libsbml.AST_CONSTANT_TRUE))
            else:
                term.setMath(literals[0] if len(literals) == 1 else _ast(libsbml.AST_LOGICAL_AND, *literals))
            term.setAnnotation(_annotation("reaction", {
                "id": r.id, "order": str(j), "weight": _num(r.weight), "n": _num(r.n),
                "ec50": _num(r.ec50), "module": r.module,
            }))
    errors, _ = _problems(doc)
    if errors:  # a bug in this module, never the user's fault
        raise ModelError(f"netflux produced invalid SBML for {model.name!r}; {_error_summary(errors)}")
    return libsbml.writeSBMLToString(doc)


def write_sbml(model: Model, path: str | Path | None = None) -> Path:
    path = output_path(path, f"{model.name}.xml")
    path.write_text(to_sbml(model), encoding="utf-8")
    return path


# --- reading


def _netflux_annotation(element, tag: str) -> dict[str, str] | None:
    annotation = element.getAnnotation()
    if annotation is None:
        return None
    for i in range(annotation.getNumChildren()):
        child = annotation.getChild(i)
        if child.getName() == tag and child.getURI() == NAMESPACE:
            attributes = child.getAttributes()
            return {attributes.getName(k): attributes.getValue(k) for k in range(attributes.getLength())}
    return None


def _expression(node, symbols: dict[str, Symbol], thresholds: dict[str, int], warnings: list[str]):
    """libsbml math -> sympy boolean, with every species treated as Boolean (level 0 or 1)."""
    kind = node.getType()
    children = [node.getChild(i) for i in range(node.getNumChildren())]
    if kind == libsbml.AST_CONSTANT_TRUE:
        return true
    if kind == libsbml.AST_CONSTANT_FALSE:
        return false
    if kind == libsbml.AST_LOGICAL_NOT:
        return Not(_expression(children[0], symbols, thresholds, warnings))
    if kind in _CONNECTIVES:
        return _CONNECTIVES[kind](*(_expression(c, symbols, thresholds, warnings) for c in children))
    if kind in _COMPARISONS and len(children) == 2:
        operands = []  # species id (str) or number
        for child in children:
            if child.getType() == libsbml.AST_NAME and child.getName() in symbols:
                operands.append(child.getName())
            elif child.getType() == libsbml.AST_NAME and child.getName() in thresholds:
                operands.append(thresholds[child.getName()])
            elif child.isNumber():
                operands.append(child.getValue())
        names = [o for o in operands if isinstance(o, str)]
        if len(operands) == 2 and len(names) == 1:
            if any(not isinstance(o, str) and o > 1 for o in operands):
                warnings.append(f"threshold above 1 on {names[0]} treated as 1 (Boolean)")
            compare = _COMPARISONS[kind]

            def holds(level: int) -> bool:  # the comparison with the species at this level
                a, b = (level if isinstance(o, str) else min(o, 1) for o in operands)
                return compare(a, b)

            symbol = symbols[names[0]]
            on, off = holds(1), holds(0)
            return true if on and off else false if not (on or off) else symbol if on else Not(symbol)
    raise ModelError(f"unsupported math in SBML function term: {libsbml.formulaToL3String(node)}")


def _clauses(expr) -> list[list[tuple[str, bool]]]:
    """OR-of-AND form as clauses of (species id, negated)."""
    dnf = to_dnf(expr)
    if dnf == true:
        return [[]]
    if dnf == false:
        return []
    clauses = []
    for term in dnf.args if isinstance(dnf, Or) else (dnf,):
        literals = term.args if isinstance(term, And) else (term,)
        clauses.append(sorted((str(lit.args[0]), True) if isinstance(lit, Not) else (str(lit), False)
                              for lit in literals))
    return clauses


def _from_document(doc, name: str) -> Model:
    errors, sbml_warnings = _problems(doc)
    if errors:
        raise ModelError(f"{name}: invalid SBML-qual, {_error_summary(errors)}")
    sbml_model = doc.getModel()
    qual = sbml_model.getPlugin("qual") if sbml_model is not None else None
    if qual is None:
        raise ModelError(f"{name} is not an SBML-qual model (no qual package)")

    warnings: list[str] = [f"SBML {w}" for w in sbml_warnings]
    species: list[Species] = []
    rename: dict[str, str] = {}
    for qs in qual.getListOfQualitativeSpecies():
        ann = _netflux_annotation(qs, "species")
        sid = qs.getId() if ann is not None else qs.getId().removeprefix("qual_")
        rename[qs.getId()] = sid
        if qs.isSetMaxLevel() and qs.getMaxLevel() > 1:
            warnings.append(f"Species {sid}: maxLevel {qs.getMaxLevel()} treated as Boolean")
        if ann is None:
            params = dict(DEFAULT_SPECIES_PARAMS, y0=float(min(qs.getInitialLevel(), 1)) if qs.isSetInitialLevel() else 0.0)
            species.append(Species(id=sid, name=qs.getName(), **params))
        else:
            values = {k: number(ann.get(k)) for k in DEFAULT_SPECIES_PARAMS if k in ann}
            params = DEFAULT_SPECIES_PARAMS | with_defaults(values, DEFAULT_SPECIES_PARAMS, f"Species {sid}", warnings)
            text = {k: ann.get(k, "") for k in _SPECIES_TEXT}
            species.append(Species(id=sid, name=qs.getName(), **params, **text))

    found: dict[str, dict] = {}  # reaction id -> fields; reactions with several outputs are merged
    for transition in qual.getListOfTransitions():
        tid = transition.getId() or f"tr{len(found) + 1}"
        inputs = list(transition.getListOfInputs())
        outputs = [rename.get(o.getQualitativeSpecies(), o.getQualitativeSpecies()) for o in transition.getListOfOutputs()]
        symbols = {original: Symbol(new) for original, new in rename.items()}
        thresholds = {i.getId(): i.getThresholdLevel() for i in inputs if i.isSetId() and i.isSetThresholdLevel()}
        terms = [transition.getFunctionTerm(k) for k in range(transition.getNumFunctionTerms())]
        default_term = transition.getDefaultTerm()
        default = default_term.getResultLevel() if default_term is not None else 0
        if not outputs:
            warnings.append(f"Transition {tid} has no outputs; ignored")
            continue
        if any(t.getResultLevel() > 1 for t in terms) or default > 1:
            warnings.append(f"Transition {tid}: result levels above 1 treated as 1 (Boolean)")

        def decode(term, tid=tid, symbols=symbols, thresholds=thresholds):
            try:
                return _expression(term.getMath(), symbols, thresholds, warnings)
            except ModelError as exc:
                raise ModelError(f"{name}: transition {tid}: {exc}") from None

        candidates: list[tuple[list, dict | None]] = []  # (clauses, netflux annotation)
        if not terms:  # no logic given: each input activates (or inhibits) the outputs on its own
            if len(inputs) > 1:
                warnings.append(f"Transition {tid} has {len(inputs)} inputs but no function terms; "
                                "imported as one reaction per input")
            for i in inputs:
                negated = i.isSetSign() and i.getSign() == libsbml.INPUT_SIGN_NEGATIVE
                candidates.append(([[(rename.get(i.getQualitativeSpecies(), i.getQualitativeSpecies()), negated)]], None))
        elif default == 0:
            for term in terms:
                if term.getResultLevel() >= 1:
                    candidates.append((_clauses(decode(term)), _netflux_annotation(term, "reaction")))
        else:
            off = [decode(t) for t in terms if t.getResultLevel() == 0]
            candidates.append((_clauses(Not(Or(*off))), None))

        count = 0
        for clauses, ann in candidates:
            for clause in clauses:
                count += 1
                if ann is not None and len(clauses) == 1:  # one of our own reactions
                    rid, order = ann.get("id") or f"{tid}_{count}", number(ann.get("order"))
                    values = {p: number(ann.get(p)) for p in DEFAULT_REACTION_PARAMS if p in ann}
                    params = DEFAULT_REACTION_PARAMS | with_defaults(
                        values, DEFAULT_REACTION_PARAMS, f"Reaction {rid}", warnings)
                    module = ann.get("module", "")
                else:
                    rid, order, params, module = f"{tid}_{count}", None, dict(DEFAULT_REACTION_PARAMS), ""
                if rid in found and found[rid]["clause"] == clause:  # same reaction, another output
                    found[rid]["outputs"] += [o for o in outputs if o not in found[rid]["outputs"]]
                    continue
                while rid in found:
                    rid += "_"
                found[rid] = {"clause": clause, "outputs": list(outputs), "order": order,
                              "params": params, "module": module, "seen": len(found)}

    ordered = sorted(found.items(), key=lambda item: (item[1]["order"] is None, item[1]["order"] or 0, item[1]["seen"]))
    reactions = [Reaction(rid, f["clause"], f["outputs"], module=f["module"], **f["params"]) for rid, f in ordered]
    if not species:
        raise ModelError(f"{name}: the model has no species")
    model = Model(name=sbml_model.getName() or sbml_model.getId() or name, species=species, reactions=reactions)
    model.warnings = warnings + model.diagnose()
    return model


def from_sbml(text: str, name: str = "netflux") -> Model:
    return _from_document(libsbml.readSBMLFromString(text), name)


def read_sbml(path: str | Path) -> Model:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return _from_document(libsbml.readSBMLFromFile(str(path)), path.stem)
