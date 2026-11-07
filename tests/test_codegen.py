import importlib.util
import os
import random
import subprocess
import sys

import pytest
import torch

import netflux
from netflux import Model, Reaction, Species, export_python_ode, python_ode_sources
from netflux.codegen import python_variable_names

from .conftest import native_rhs


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_generated_matches_native(model, tmp_path, name="NetfluxODE"):
    files = export_python_ode(model, tmp_path, name)
    assert [f.name for f in files] == [f"{name}.py", f"{name}_params.py", f"{name}_run.py"]
    ode = _load_module(tmp_path / f"{name}.py", f"gen_{name}_{id(model)}")
    params = _load_module(tmp_path / f"{name}_params.py", f"gen_{name}_params_{id(model)}")
    species_names, tau, ymax, y0, w, n, ec50 = params.loadParams()
    assert species_names == model.species_ids
    assert y0.tolist() == [s.y0 for s in model.species]
    rng = random.Random(1)
    for _ in range(3):
        y = [rng.uniform(0, 1.2) for _ in model.species]
        generated = ode.ODEfunc(torch.tensor(0.0), torch.tensor(y, dtype=torch.float64), tau, ymax, w, n, ec50)
        assert generated.tolist() == pytest.approx(native_rhs(model, y), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", netflux.list_examples())
def test_generated_code_matches_native_rhs(examples, name, tmp_path):
    _check_generated_matches_native(examples[name], tmp_path)


def test_name_collisions_are_renamed(tmp_path):
    ids = ["n", "w", "class", "A", "n_", "Ca2+"]
    model = Model(
        "collide",
        [Species(s) for s in ids],
        [Reaction("r1", "=> A", weight=0.8), Reaction("r2", "A => n"), Reaction("r3", "n & !w => class"),
         Reaction("r4", "class => n_"), Reaction("r5", "n_ => Ca2+")],
    )
    names = python_variable_names(ids)
    assert len(set(names)) == len(names)
    assert names[3] == "A" and names[4] == "n_" and names[0] not in {"n", "n_"}
    assert names[5] == "s5"
    _check_generated_matches_native(model, tmp_path, name="collide_ode")


def test_invalid_module_name(examples):
    with pytest.raises(ValueError):
        python_ode_sources(examples["exampleNet"], "not-valid")


def test_run_script_executes(examples, tmp_path):
    pytest.importorskip("matplotlib")
    export_python_ode(examples["exampleNet"], tmp_path)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "NetfluxODE_run.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
