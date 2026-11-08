from pathlib import Path

import pytest

import netflux
from netflux import get_output_dir, output_path, set_output_dir, using_output_dir
from netflux.cli import main
from netflux.output import ENV_VAR



def test_default_is_output_folder(tmp_path):
    assert get_output_dir() == Path("output")
    assert output_path() == tmp_path / "output"
    assert output_path("a/b.txt") == tmp_path / "output" / "a" / "b.txt"
    assert output_path(None, "default.txt") == tmp_path / "output" / "default.txt"


def test_absolute_paths_are_kept(tmp_path):
    target = tmp_path / "elsewhere" / "x.txt"
    assert output_path(target) == target


def test_resolving_twice_is_stable():
    once = output_path("x.txt")
    assert output_path(once) == once


def test_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "from_env")
    assert output_path("x") == tmp_path / "from_env" / "x"


def test_set_output_dir_overrides_env_and_resets(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "from_env")
    set_output_dir("chosen")
    assert output_path("x") == tmp_path / "chosen" / "x"
    set_output_dir(None)
    assert output_path("x") == tmp_path / "from_env" / "x"


def test_using_output_dir_is_temporary(tmp_path):
    with using_output_dir("tmp_out") as folder:
        assert folder == Path("tmp_out")
        assert output_path("x") == tmp_path / "tmp_out" / "x"
    assert output_path("x") == tmp_path / "output" / "x"
    with pytest.raises(RuntimeError), using_output_dir("other"):
        raise RuntimeError
    assert get_output_dir() == Path("output")


def test_current_directory_behaviour(tmp_path, examples):
    set_output_dir(".")
    assert netflux.write_model(examples["exampleNet"], "m.xlsx") == tmp_path / "m.xlsx"


def test_every_writer_defaults_to_output_folder(tmp_path, examples):
    model = examples["exampleNet"]
    out = tmp_path / "output"
    assert netflux.write_model(model) == out / "exampleNet.xlsx"
    assert netflux.write_model(model, "sub/copy.xlsx") == out / "sub" / "copy.xlsx"
    assert netflux.write_model(model, "exampleNet.sif") == out / "exampleNet.sif"
    assert (out / "exampleNet_nodeAttributes.txt").exists()
    assert netflux.write_xgmml(model) == out / "exampleNet.xgmml"
    assert netflux.write_sbgn_style() == out / "netflux_sbgn_style.xml"
    assert netflux.write_model(model, "models/net.xml") == out / "models" / "net.xml"
    assert netflux.simulate(model, 1).write_data() == out / "exampleNet_data.txt"
    assert all(p.exists() for p in out.rglob("*"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output"]


def test_cli_uses_env_and_option(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "env_out")
    assert main(["convert", "exampleNet", "-o", "net.sif"]) == 0
    assert (tmp_path / "env_out" / "net.sif").exists()
    assert main(["convert", "exampleNet", "-o", "net.sif", "--output-dir", "opt_out"]) == 0
    assert (tmp_path / "opt_out" / "net.sif").exists()
    assert get_output_dir() == Path("env_out")
