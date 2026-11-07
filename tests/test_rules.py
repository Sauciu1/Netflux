from netflux import Model, Reaction, Species, parse_rule
from netflux.rules import compile_rules

IDS = ["A", "B", "C"]


def test_activation():
    r = parse_rule("A => B", IDS)
    assert (r.reactants, r.negated, r.products, r.error) == ((0,), (False,), (1,), None)


def test_inhibition():
    r = parse_rule("!A => B", IDS)
    assert r.reactants == (0,) and r.negated == (True,) and r.products == (1,)


def test_and_with_negation():
    r = parse_rule("A & !B => C", IDS)
    assert r.reactants == (0, 1) and r.negated == (False, True) and r.products == (2,)


def test_input_reaction():
    r = parse_rule("=> A", IDS)
    assert r.reactants == () and r.products == (0,) and r.error is None


def test_plus_separator_is_and_like_matlab():
    ids = ["NE", "B1ARPG", "B1ARPA", "B1AR"]
    r = parse_rule("NE + !B1ARPG + !B1ARPA => B1AR", ids)
    assert r.reactants == (0, 1, 2) and r.negated == (False, True, True) and r.error is None


def test_netflux2_syntax():
    r = parse_rule("A AND NOT B -> C", IDS)
    assert r.reactants == (0, 1) and r.negated == (False, True) and r.products == (2,)


def test_self_loop():
    r = parse_rule("!A => A", IDS)
    assert r.reactants == (0,) and r.negated == (True,) and r.products == (0,)


def test_reactants_in_species_order():
    r = parse_rule("C & A => B", IDS)
    assert r.reactants == (0, 2)


def test_whole_word_matching():
    r = parse_rule("AA => A", ["A", "AA"])
    assert r.reactants == (1,) and r.products == (0,)


def test_negation_needs_word_boundary():
    # MATLAB's regexp('!'+ID) would also mark A as negated here.
    r = parse_rule("A & !AB => C", ["A", "AB", "C"])
    assert r.negated == (False, True)


def test_negation_with_space():
    assert parse_rule("! A => B", IDS).negated == (True,)


def test_invalid_id_with_symbol_is_not_a_separator():
    r = parse_rule("IP3 & Ca2+ => X", ["IP3", "Ca2+", "X"])
    assert r.reactants == (0, 1) and r.error is None


def test_error_no_arrow_is_inert():
    r = parse_rule("A B", IDS)
    assert r.error == "no_arrow" and r.reactants == () and r.products == ()


def test_error_no_products():
    assert parse_rule("A =>", IDS).error == "no_products"


def test_error_unknown_reactant_keeps_known_ones():
    r = parse_rule("A & X => C", IDS)
    assert r.error == "reactants_not_recognized" and r.reactants == (0,)


def test_error_no_recognized_reactant():
    r = parse_rule("X => C", IDS)
    assert r.error == "reactants_not_recognized" and r.reactants == ()


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
