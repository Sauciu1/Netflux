from netflux import Model, Reaction, Species, parse_rule
from netflux.rules import compile_rules

IDS = ["A", "B", "C"]


def test_activation():
    r = parse_rule("A => B", IDS)
    assert (r.inputs, r.products, r.error) == (((0, False),), (1,), None)


def test_inhibition():
    assert parse_rule("!A => B", IDS).inputs == ((0, True),)


def test_and_with_negation():
    r = parse_rule("A & !B => C", IDS)
    assert r.inputs == ((0, False), (1, True)) and r.products == (2,)


def test_input_reaction():
    r = parse_rule("=> A", IDS)
    assert r.inputs == () and r.products == (0,) and r.error is None


def test_plus_separator():
    r = parse_rule("NE + !B1ARPG + !B1ARPA => B1AR", ["NE", "B1ARPG", "B1ARPA", "B1AR"])
    assert r.inputs == ((0, False), (1, True), (2, True)) and r.error is None


def test_netflux2_syntax():
    assert parse_rule("A AND NOT B -> C", IDS).inputs == ((0, False), (1, True))


def test_self_loop():
    r = parse_rule("!A => A", IDS)
    assert r.inputs == ((0, True),) and r.products == (0,)


def test_inputs_in_species_order():
    assert parse_rule("C & A => B", IDS).inputs == ((0, False), (2, False))


def test_whole_names_only():
    r = parse_rule("AA & !AB => A", ["A", "AA", "AB"])
    assert r.inputs == ((1, False), (2, True)) and r.products == (0,)


def test_keyword_inside_name_is_not_a_separator():
    r = parse_rule("BAND & NOTCH => C", ["BAND", "NOTCH", "C"])
    assert r.inputs == ((0, False), (1, False)) and r.error is None


def test_negation_with_space():
    assert parse_rule("! A => B", IDS).inputs == ((0, True),)


def test_error_no_arrow_is_inert():
    r = parse_rule("A B", IDS)
    assert "arrow" in r.error and r.inputs == () and r.products == ()


def test_error_no_products():
    assert parse_rule("A =>", IDS).error == "product not recognized"


def test_error_unknown_reactant_keeps_known_ones():
    r = parse_rule("A & X => C", IDS)
    assert r.error == "reactant(s) not recognized: X" and r.inputs == ((0, False),)


def test_compile_rules_producers():
    topo = compile_rules(IDS, ["=> A", "A => C", "B => C"])
    assert topo.producers == ((0,), (), (1, 2))


def test_diagnose_reports_problems():
    model = Model(
        "m",
        [Species("A"), Species("A"), Species("1bad"), Species("C")],
        [Reaction("r1", "A => C"), Reaction("r2", "A => C"), Reaction("r3", "A C"),
         Reaction("r4", "A => C", ec50=0.9, n=1.0)],
    )
    text = "\n".join(model.diagnose())
    assert "Invalid species name(s)" in text and "1bad" in text
    assert "Duplicate species detected: A" in text
    assert "Duplicate reactions detected: r1/r2/r4" in text
    assert "r3" in text and "no reaction arrow" in text
    assert "r4" in text and "EC50^n >= 0.5" in text
