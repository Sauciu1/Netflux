import pytest

import netflux
from netflux.cli import main

from .conftest import DATA


def test_examples(capsys):
    assert main(["examples"]) == 0
    assert capsys.readouterr().out.split() == netflux.list_examples()


def test_info(capsys):
    assert main(["info", "exampleNet", "-v"]) == 0
    out = capsys.readouterr().out
    assert "Species: 5  Reactions: 6" in out and "r5: C & !D => E" in out


def test_simulate_to_stdout(capsys):
    assert main(["simulate", "exampleNet", "--set", "r1.weight=1", "--species", "A,E"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "t = 10" and out[1].startswith("A\t1.00") and out[2].startswith("E\t")


def test_simulate_outputs(tmp_path, capsys):
    data = tmp_path / "data.txt"
    csv = tmp_path / "data.csv"
    assert main(["simulate", "exampleNet", "--set", "r1.weight=1", "--set", "C.tau=2", "-o", str(data)]) == 0
    assert data.read_text().startswith("Species_Name T0 T0.1")
    assert main(["simulate", "exampleNet", "--t-end", "1", "-o", str(csv)]) == 0
    assert csv.read_text().splitlines()[0] == "t,A,B,C,D,E"


def test_simulate_plot(tmp_path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "plot.png"
    assert main(["simulate", "exampleNet", "--t-end", "1", "--plot", str(png)]) == 0
    assert png.stat().st_size > 0


def test_simulate_bad_setting():
    with pytest.raises(SystemExit):
        main(["simulate", "exampleNet", "--set", "nope.weight=1"])


def test_unknown_model():
    with pytest.raises(SystemExit):
        main(["info", "does-not-exist"])


def test_exports(tmp_path):
    assert main(["export-sif", "exampleNet", "-o", str(tmp_path), "--name", "net"]) == 0
    assert (tmp_path / "net.sif").exists() and (tmp_path / "net_nodeAttributes.txt").exists()
    xgmml = tmp_path / "net.xgmml"
    style = tmp_path / "style.xml"
    assert main(["export-xgmml", "exampleNet", "-o", str(xgmml), "--style", str(style)]) == 0
    assert xgmml.exists() and style.exists()
    assert main(["export-xgmml", "exampleNet", "-o", str(tmp_path / "again.xgmml"), "--reference", str(xgmml)]) == 0
    assert main(["export-python", "exampleNet", "-o", str(tmp_path / "ex.py")]) == 0
    assert (tmp_path / "ex.py").exists()


def test_model_file_path(tmp_path):
    path = netflux.load_example("exampleNet").to_excel(tmp_path / "mine.xlsx")
    assert main(["info", str(path)]) == 0


def test_convert_sif(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sif = tmp_path / "net.sif"
    sif.write_text("A\t1\tB\nB\t-1\tC\n")
    assert main(["convert-sif", str(sif)]) == 0
    model = netflux.read_model(tmp_path / "output" / "net.xlsx")
    assert [r.rule for r in model.reactions] == ["A => B", "!B => C"]


def test_everything_defaults_to_output_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["simulate", "exampleNet", "--t-end", "1", "-o", "data.csv"]) == 0
    assert main(["export-sif", "exampleNet"]) == 0
    assert main(["export-xgmml", "exampleNet", "--style", "style.xml"]) == 0
    assert main(["export-python", "exampleNet", "-o", "code/NetfluxODE.py"]) == 0
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "output/code/NetfluxODE.py",
        "output/data.csv",
        "output/exampleNet.sif",
        "output/exampleNet.xgmml",
        "output/exampleNet_nodeAttributes.txt",
        "output/style.xml",
    ]


def test_output_dir_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["export-sif", "exampleNet", "--output-dir", "results"]) == 0
    assert (tmp_path / "results" / "exampleNet.sif").exists()
    assert not (tmp_path / "output").exists()


def test_library_defaults_to_output_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = netflux.load_example("exampleNet")
    netflux.write_sif(model)
    netflux.export_python_ode(model)
    assert (tmp_path / "output" / "exampleNet.sif").exists()
    assert (tmp_path / "output" / "NetfluxODE.py").exists()


def test_convert_sbml(tmp_path, monkeypatch):
    pytest.importorskip("libsbml")
    monkeypatch.chdir(tmp_path)
    assert main(["convert-sbml", str(DATA / "toy_qual.xml")]) == 0
    assert (tmp_path / "output" / "toy_qual" / "toy_qual Netflux.xlsx").exists()
    assert main(["convert-sbml", str(DATA / "toy_qual.xml"), "-o", str(tmp_path / "abs")]) == 0
    assert (tmp_path / "abs" / "toy_qual Netflux.xlsx").exists()


def test_error_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.sif"
    bad.write_text("only two\n")
    assert main(["convert-sif", str(bad)]) == 1
    assert "error" in capsys.readouterr().err
