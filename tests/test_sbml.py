import libsbml
import pytest
import torch

import netflux
from netflux import Model, ModelError, Reaction, Species, from_sbml, read_model, to_sbml, write_model
from netflux.cli import main

from .conftest import DATA


def structure(model):
    ids = model.species_ids
    return sorted(
        (tuple((ids[i], neg) for i, neg in r.inputs), tuple(ids[p] for p in r.products))
        for r in model.topology.rules
    )


def sbml_errors(text):
    doc = libsbml.readSBMLFromString(text)
    doc.checkConsistency()
    return [doc.getError(i).getMessage() for i in range(doc.getNumErrors())
            if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]


# --- our own files


@pytest.mark.parametrize("name", netflux.list_examples())
def test_round_trip_of_bundled_models(examples, name, tmp_path):
    original = examples[name]
    for again in (from_sbml(to_sbml(original)), read_model(write_model(original, tmp_path / "m.xml"))):
        assert again.name == original.name
        assert again.species == original.species
        assert again.topology == original.topology
        assert [(r.id, r.weight, r.n, r.ec50, r.module) for r in again.reactions] == [
            (r.id, r.weight, r.n, r.ec50, r.module) for r in original.reactions
        ]
        assert again.warnings == []


@pytest.mark.parametrize("name", netflux.list_examples())
def test_written_files_are_valid_sbml(examples, name):
    assert sbml_errors(to_sbml(examples[name])) == []


def test_sbml_and_excel_simulate_identically(examples, tmp_path):
    original = examples["exampleNet"]
    via_sbml = read_model(write_model(original, tmp_path / "net.xml"))
    a = netflux.simulate(original, 5, reactions={"r1": {"weight": 1}})
    b = netflux.simulate(via_sbml, 5, reactions={"r1": {"weight": 1}})
    torch.testing.assert_close(a.y, b.y, rtol=0, atol=0)


def test_logic_is_standard_sbml_qual(examples):
    text = to_sbml(examples["exampleNet"])
    doc = libsbml.readSBMLFromString(text)  # keep the document alive while using its children
    qual = doc.getModel().getPlugin("qual")
    by_output = {qual.getTransition(i).getOutput(0).getQualitativeSpecies(): qual.getTransition(i)
                 for i in range(qual.getNumTransitions())}
    e = by_output["E"]  # r5: C & !D => E
    assert libsbml.formulaToL3String(e.getFunctionTerm(0).getMath()) == "(C == 1) && (D == 0)"
    c = by_output["C"]  # A => C and E => C: two terms, i.e. OR
    assert c.getNumFunctionTerms() == 2 and c.getDefaultTerm().getResultLevel() == 0


def test_reaction_with_two_outputs_round_trips():
    model = Model("m", [Species("A"), Species("B"), Species("C")], [Reaction.from_rule("r1", "A => B & C", weight=0.7)])
    again = from_sbml(to_sbml(model))
    assert structure(again) == structure(model)
    assert [(r.id, r.weight) for r in again.reactions] == [("r1", 0.7)]


def test_writing_requires_valid_ids_and_known_species():
    with pytest.raises(ModelError, match="1bad"):
        to_sbml(Model("m", [Species("1bad")], []))
    with pytest.raises(ModelError, match="r1.*unknown species X"):
        to_sbml(Model("m", [Species("A")], [Reaction.from_rule("r1", "X => A")]))


# --- files from other tools

def _qual(species: str, transitions: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
      xmlns:qual="http://www.sbml.org/sbml/level3/version1/qual/version1" qual:required="true">
  <model id="foreign">
    <listOfCompartments><compartment id="cell" constant="true"/></listOfCompartments>
    <qual:listOfQualitativeSpecies>{species}</qual:listOfQualitativeSpecies>
    <qual:listOfTransitions>{transitions}</qual:listOfTransitions>
  </model>
</sbml>'''


def _species(*ids, max_level=1):
    return "".join(f'<qual:qualitativeSpecies qual:id="{i}" qual:compartment="cell" qual:constant="false" '
                   f'qual:maxLevel="{max_level}"/>' for i in ids)


def _transition(output, inputs, terms, default=0, thresholds=None):
    thresholds = thresholds or {}
    ins = "".join(
        f'<qual:input qual:id="in_{output}_{i}" qual:qualitativeSpecies="{i}" qual:transitionEffect="none"'
        + (f' qual:thresholdLevel="{thresholds[i]}"' if i in thresholds else "") + "/>"
        for i in inputs
    )
    fts = "".join(f'<qual:functionTerm qual:resultLevel="{level}"><math xmlns="http://www.w3.org/1998/Math/MathML">'
                  f"{math}</math></qual:functionTerm>" for level, math in terms)
    return (f'<qual:transition qual:id="tr_{output}"><qual:listOfInputs>{ins}</qual:listOfInputs>'
            f'<qual:listOfOutputs><qual:output qual:qualitativeSpecies="{output}" '
            f'qual:transitionEffect="assignmentLevel"/></qual:listOfOutputs><qual:listOfFunctionTerms>'
            f'<qual:defaultTerm qual:resultLevel="{default}"/>{fts}</qual:listOfFunctionTerms></qual:transition>')


def _cmp(op, a, b):
    def side(x):
        return f'<cn type="integer">{x}</cn>' if isinstance(x, int) else f"<ci>{x}</ci>"
    return f"<apply><{op}/>{side(a)}{side(b)}</apply>"


def test_or_of_and_becomes_one_reaction_per_clause():
    math = f'<apply><or/><apply><and/>{_cmp("eq", "A", 1)}{_cmp("eq", "B", 0)}</apply>{_cmp("eq", "C", 1)}</apply>'
    m = from_sbml(_qual(_species("A", "B", "C", "X"), _transition("X", "ABC", [(1, math)])))
    assert structure(m) == [((("A", False), ("B", True)), ("X",)), ((("C", False),), ("X",))]
    assert all(r.weight == 1.0 and r.n == 1.4 for r in m.reactions)
    assert m.reaction_ids == ["tr_X_1", "tr_X_2"]


def test_default_level_one_is_negated():
    m = from_sbml(_qual(_species("A", "X"), _transition("X", "A", [(0, _cmp("eq", "A", 1))], default=1)))
    assert structure(m) == [((("A", True),), ("X",))]


@pytest.mark.parametrize(
    "math,negated",
    [(_cmp("geq", "A", 1), False), (_cmp("gt", "A", 0), False), (_cmp("neq", "A", 0), False),
     (_cmp("eq", 1, "A"), False), (_cmp("leq", "A", 0), True), (_cmp("lt", "A", 1), True),
     (f"<apply><not/>{_cmp('eq', 'A', 1)}</apply>", True)],
)
def test_comparisons(math, negated):
    m = from_sbml(_qual(_species("A", "X"), _transition("X", "A", [(1, math)])))
    assert structure(m) == [((("A", negated),), ("X",))]


def test_input_threshold_reference_and_true():
    trs = _transition("X", "A", [(1, _cmp("geq", "A", "in_X_A"))], thresholds={"A": 1})
    trs += _transition("Y", "", [(1, "<true/>")]).replace("<qual:listOfInputs></qual:listOfInputs>", "")
    m = from_sbml(_qual(_species("A", "X", "Y"), trs))
    assert structure(m) == [((), ("Y",)), ((("A", False),), ("X",))]


def test_multivalued_species_warn_and_qual_prefix_is_stripped():
    m = from_sbml(_qual(_species("qual_A", "qual_X", max_level=2),
                        _transition("qual_X", ["qual_A"], [(1, _cmp("geq", "qual_A", 2))])))
    assert m.species_ids == ["A", "X"]
    assert structure(m) == [((("A", False),), ("X",))]
    text = "\n".join(m.warnings)
    assert "maxLevel 2" in text and "threshold above 1" in text


def test_transition_without_function_terms_falls_back_to_one_reaction_per_input():
    model = read_model(DATA / "toy_qual.xml")
    assert model.species_ids == ["A", "B", "C", "D", "E", "Unused"]
    assert [r.rule for r in model.reactions] == ["A => B", "!B => C", "C => D", "A => E", "!D => E"]
    assert any("tr4" in w and "2 inputs" in w for w in model.warnings)


def test_not_sbml_qual(tmp_path):
    old = tmp_path / "old.xml"
    old.write_text('<netflux name="x"><species id="A"/></netflux>')
    with pytest.raises(ModelError):
        read_model(old)
    plain = tmp_path / "plain.xml"
    plain.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">'
                     '<model id="m"/></sbml>')
    with pytest.raises(ModelError, match="qual"):
        read_model(plain)


def test_invalid_sbml_reports_errors_with_line_numbers(examples):
    text = to_sbml(examples["exampleNet"]).replace('qual:transitionEffect="assignmentLevel"', "", 1)
    with pytest.raises(ModelError, match=r"invalid SBML-qual, 1 error\(s\):\n  line \d+: .*'transitionEffect' is missing"):
        from_sbml(text)


def test_unsupported_math_names_the_transition():
    math = '<apply><plus/><ci>A</ci><cn type="integer">1</cn></apply>'
    with pytest.raises(ModelError, match="transition tr_X: unsupported math"):
        from_sbml(_qual(_species("A", "X"), _transition("X", "A", [(1, math)])))


def test_odd_transitions_are_reported():
    no_output = _transition("X", "A", [(1, _cmp("eq", "A", 1))]).replace(
        '<qual:listOfOutputs><qual:output qual:qualitativeSpecies="X" qual:transitionEffect="assignmentLevel"/>'
        "</qual:listOfOutputs>", "")
    level_two = _transition("Y", "A", [(2, _cmp("eq", "A", 1))]).replace("tr_Y", "tr_Y2")
    m = from_sbml(_qual(_species("A", "X") + _species("Y", max_level=2), no_output + level_two))
    text = "\n".join(m.warnings)
    assert "tr_X has no outputs" in text and "tr_Y2: result levels above 1" in text
    assert structure(m) == [((("A", False),), ("Y",))]


def test_cli_reports_bad_sbml(tmp_path, capsys):
    bad = tmp_path / "bad.xml"
    bad.write_text("<sbml>not really</sbml>")
    assert main(["info", str(bad)]) == 1
    assert "invalid SBML-qual" in capsys.readouterr().err


def test_cli_convert_and_use(tmp_path):
    assert main(["convert", "exampleNet", "-o", "ex.xml"]) == 0
    path = tmp_path / "output" / "ex.xml"
    assert sbml_errors(path.read_text()) == []
    assert main(["info", str(path)]) == 0
    assert main(["simulate", str(path), "--set", "r1.weight=1", "-o", "data.txt"]) == 0
    assert main(["convert", str(path), "-o", "back.xlsx"]) == 0
    assert read_model(tmp_path / "output" / "back.xlsx").topology == netflux.load_example("exampleNet").topology
