import math
import warnings

import pytest
import torch

import netflux
from netflux import NetfluxWarning, Simulator, simulate


def test_input_species_matches_analytic_solution(examples):
    # r1: "=> A" with weight 1, tau = ymax = 1, so dA/dt = 1 - A and A(t) = 1 - exp(-t)
    sim = Simulator(examples["exampleNet"])
    sim.set_reaction("r1", weight=1)
    result = sim.run(10, 0.1, rtol=1e-10, atol=1e-12)
    expected = 1 - torch.exp(-result.t)
    torch.testing.assert_close(result["A"], expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("method,tol", [("dopri5", 1e-6), ("bosh3", 1e-2)])
def test_default_tolerance_accuracy(examples, method, tol):
    result = simulate(examples["exampleNet"], 10, 0.1, reactions={"r1": {"weight": 1}}, method=method)
    error = (result["A"] - (1 - torch.exp(-result.t))).abs().max()
    assert float(error) < tol


def test_time_grid_and_continuation(examples):
    sim = Simulator(examples["exampleNet"])
    first = sim.run(10, 0.1)
    assert len(first) == 101 and float(first.t[0]) == 0.0
    second = sim.run(10, 0.1)
    assert len(second) == 201
    assert float(second.t[-1]) == pytest.approx(20.0)
    assert torch.all(torch.diff(second.t) > 0)


def test_split_run_equals_single_run(examples):
    model = examples["kraeutler_betaAdrenergic_bmcsysbio2010"]
    kw = dict(rtol=1e-10, atol=1e-12)
    a = Simulator(model)
    a.set_reaction(model.reactions[0].id, weight=1)
    a.run(5, 0.5, **kw)
    split = a.run(5, 0.5, **kw)
    b = Simulator(model)
    b.set_reaction(model.reactions[0].id, weight=1)
    whole = b.run(10, 0.5, **kw)
    torch.testing.assert_close(split.t, whole.t)
    torch.testing.assert_close(split.y, whole.y, rtol=0, atol=1e-8)


def test_docs_tutorial_sequence(examples):
    sim = Simulator(examples["exampleNet"])
    for r1, r2 in [(1, 0), (0, 0), (0, 1), (0, 0), (1, 1), (1, 1)]:
        sim.set_reaction("r1", weight=r1)
        sim.set_reaction("r2", weight=r2)
        result = sim.run(10)
    assert len(result) == 6 * 100 + 1
    assert torch.all(torch.diff(result.t) > 0)
    at = {round(float(t), 6): i for i, t in enumerate(result.t)}
    a_on = result.y[at[10.0]]
    assert float(a_on[0]) > 0.99
    assert float(a_on[2]) > 0.9 and float(a_on[4]) > 0.9
    washout = result.y[at[20.0]]
    assert float(washout[0]) < 0.01
    assert float(washout[2]) > 0.9  # C stays on through the C <-> E positive loop


def test_steady_state_satisfies_fixed_point(examples):
    model = examples["kraeutler_betaAdrenergic_bmcsysbio2010"]
    sim = Simulator(model)
    for r in model.reactions:
        if not model.topology.rules[model.reaction_index(r.id)].inputs:
            sim.set_reaction(r.id, weight=0.5)
    sim.run(300, 1.0)
    dydt = sim.ode()(torch.tensor(0.0), sim.y)
    assert float(dydt.abs().max()) < 1e-4


def test_reset_and_reset_parameters(examples):
    sim = Simulator(examples["exampleNet"])
    sim.set_reaction("r1", weight=1)
    sim.set_species("C", tau=3)
    sim.run(5)
    sim.reset()
    assert sim.t == 0.0 and len(sim.result) == 0
    assert sim.y.tolist() == [0.0] * 5
    assert float(sim.weight[0]) == 1.0  # parameters survive "Reset Simulation"
    sim.reset_parameters()
    assert float(sim.weight[0]) == 0.0 and float(sim.tau[2]) == 1.0


def test_set_state(examples):
    sim = Simulator(examples["exampleNet"])
    sim.set_state("B", 0.7)
    result = sim.run(0.1, 0.1)
    assert float(result["B"][0]) == 0.7


def test_zero_duration_returns_current_point(examples):
    sim = Simulator(examples["exampleNet"])
    assert len(sim.run(0)) == 1


def test_invalid_ec50_warns(examples):
    sim = Simulator(examples["exampleNet"])
    sim.set_reaction("r3", ec50=0.9, n=1.0)
    with pytest.warns(NetfluxWarning, match="EC50"):
        sim.run(1)


def test_valid_model_does_not_warn(examples):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(examples["ryall_cardiomyocyte_hypertrophy_jbc2012"]).run(1)


def test_simulate_overrides(examples):
    r = simulate(examples["exampleNet"], 5, 0.5, reactions={"r2": {"weight": 1}}, species={"B": {"tau": 0.5}})
    assert r.final()["B"] == pytest.approx(1 - math.exp(-10), abs=1e-3)


def test_to_dataframe(examples):
    frame = simulate(examples["exampleNet"], 1, 0.5).to_dataframe()
    assert list(frame.columns) == ["A", "B", "C", "D", "E"]
    assert frame.index.tolist() == [0.0, 0.5, 1.0]


def test_write_data_format(tmp_path, examples):
    result = simulate(examples["exampleNet"], 0.3, 0.1, reactions={"r1": {"weight": 1}})
    path = result.write_data(tmp_path / "data.txt")
    lines = path.read_text().split("\n")
    assert lines[0] == "Species_Name T0 T0.1 T0.2 T0.3"
    assert lines[1].startswith("A 0.0000 0.09") and not lines[1].endswith(" ")
    assert len(lines[1].split()) == 5
    assert lines[5].startswith("E ")
    assert lines[-1] == ""


def test_runs_end_exactly_at_duration(examples):
    sim = Simulator(examples["exampleNet"])
    sim.run(1.0, 0.3)
    result = sim.run(1.0, 0.3)
    assert float(result.t[-1]) == pytest.approx(2.0)
    assert torch.all(torch.diff(result.t) > 0)


def test_plot(examples):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    ax = simulate(examples["exampleNet"], 1).plot(["A", "C"])
    assert [line.get_label() for line in ax.get_lines()] == ["A", "C"]


def test_public_api_quickstart():
    model = netflux.load_example("exampleNet")
    sim = netflux.Simulator(model)
    sim.set_reaction("r1", weight=1)
    sim.run(10)
    sim.set_reaction("r1", weight=0)
    result = sim.run(10)
    assert float(result.t[-1]) == pytest.approx(20)
