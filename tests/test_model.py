import pytest

from netflux import Model, ModelError, Reaction, Species, parse_rule


def test_parse_activation_and_inhibition():
    assert parse_rule("A => B") == ((("A", False),), ("B",))
    assert parse_rule("!A => B") == ((("A", True),), ("B",))


def test_parse_and_with_negation():
    assert parse_rule("A & !B => C") == ((("A", False), ("B", True)), ("C",))


def test_parse_input_reaction():
    assert parse_rule("=> A") == ((), ("A",))


def test_parse_plus_separator():
    inputs, outputs = parse_rule("NE + !B1ARPG + !B1ARPA => B1AR")
    assert inputs == (("NE", False), ("B1ARPG", True), ("B1ARPA", True)) and outputs == ("B1AR",)


def test_parse_netflux2_syntax():
    assert parse_rule("A AND NOT B -> C") == ((("A", False), ("B", True)), ("C",))


def test_parse_keyword_inside_name_is_not_a_separator():
    assert parse_rule("BAND & NOTCH => C") == ((("BAND", False), ("NOTCH", False)), ("C",))


def test_parse_negation_with_space():
    assert parse_rule("! A => B")[0] == (("A", True),)


def test_parse_several_outputs():
    assert parse_rule("A => B & C")[1] == ("B", "C")


def test_parse_without_arrow_raises():
    with pytest.raises(ModelError, match="arrow"):
        parse_rule("A B")


def test_reaction_rule_round_trip():
    r = Reaction.from_rule("r5", "C & !D => E", weight=0.5)
    assert (r.inputs, r.outputs, r.weight) == ((("C", False), ("D", True)), ("E",), 0.5)
    assert r.rule == "C & !D => E"
    assert Reaction.from_rule("r1", "=> A").rule == "=> A"
    assert Reaction("r", [("A", 0)], ["B"]) == Reaction.from_rule("r", "A => B")  # lists normalised


def _model(*rules):
    return Model("m", [Species("A"), Species("B"), Species("C")],
                 [Reaction.from_rule(f"r{k}", rule) for k, rule in enumerate(rules, 1)])


def test_topology_resolves_names_in_species_order():
    topo = _model("C & A => B", "!A => A", "AA => A").topology
    assert topo.rules[0].inputs == ((0, False), (2, False))  # species order, not rule order
    assert topo.rules[1] == (((0, True),), (0,))
    assert topo.rules[2] == ((), (0,))  # unknown name AA is left out
    assert topo.producers == ((1, 2), (0,), ())


def test_diagnose_reports_problems():
    model = Model(
        "m",
        [Species("A"), Species("A"), Species("1bad"), Species("C")],
        [Reaction.from_rule("r1", "A => C"), Reaction.from_rule("r2", "A => C"),
         Reaction.from_rule("r3", "A & X => C"), Reaction.from_rule("r4", "A => Z"),
         Reaction.from_rule("r5", "A => C", ec50=0.9, n=1.0)],
    )
    text = "\n".join(model.diagnose())
    assert "Invalid species name(s)" in text and "1bad" in text
    assert "Duplicate species detected: A" in text
    assert "Duplicate reactions detected: r1/r2/r5" in text
    assert "Reaction r3: A & X => C: reactant(s) not recognized: X" in text
    assert "Reaction r4: A => Z: product not recognized" in text
    assert "r5" in text and "EC50^n >= 0.5" in text
