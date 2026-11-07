import importlib.util
import os
import random
import subprocess
import sys

import pytest
import torch

import netflux
from netflux import Model, Reaction, Species, write_model
from netflux.formats.python import python_variable_names

from .conftest import native_rhs


def _check_generated_matches_native(model, path):
    assert write_model(model, path) == path
    spec = importlib.util.spec_from_file_location(f"gen_{id(model)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SPECIES == model.species_ids and module.REACTIONS == model.reaction_ids
    tau, ymax, y0, w, n, ec50 = module.load_params()
    assert y0.tolist() == [s.y0 for s in model.species]
    rng = random.Random(1)
    for _ in range(3):
        y = [rng.uniform(0, 1.2) for _ in model.species]
        generated = module.ode(torch.tensor(0.0), torch.tensor(y, dtype=torch.float64), tau, ymax, w, n, ec50)
        assert generated.tolist() == pytest.approx(native_rhs(model, y), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", netflux.list_examples())
def test_generated_code_matches_native_rhs(examples, name, tmp_path):
    _check_generated_matches_native(examples[name], tmp_path / "ode.py")


def test_name_collisions_are_renamed(tmp_path):
    ids = ["n", "w", "class", "A", "n_", "my-species"]
    model = Model(
        "collide",
        [Species(s) for s in ids],
        [Reaction.from_rule(f"r{k}", rule, weight=0.8 if k == 1 else 1.0) for k, rule in enumerate(
            ["=> A", "A => n", "n & !w => class", "class => n_", "n_ => my-species"], start=1)],
    )
    names = python_variable_names(ids)
    assert len(set(names)) == len(names)
    assert names[3] == "A" and names[4] == "n_" and names[0] not in {"n", "n_"}
    assert names[5] == "s5"
    _check_generated_matches_native(model, tmp_path / "collide.py")


def test_runs_as_script(examples, tmp_path):
    pytest.importorskip("matplotlib")
    path = write_model(examples["exampleNet"], tmp_path / "NetfluxODE.py")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(path)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
