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


def test_convert_to_every_format(tmp_path):
    out = tmp_path / "out"
    for suffix in (".sif", ".xgmml", ".py", ".xml", ".xlsx"):
        assert main(["convert", "exampleNet", "-o", str(out / f"net{suffix}")]) == 0
        assert (out / f"net{suffix}").exists()
    assert (out / "net_nodeAttributes.txt").exists()


def test_convert_xgmml_reference_and_style(tmp_path):
    first, style = tmp_path / "a.xgmml", tmp_path / "style.xml"
    assert main(["convert", "exampleNet", "-o", str(first), "--style", str(style)]) == 0
    assert style.exists()
    assert main(["convert", "exampleNet", "-o", str(tmp_path / "b.xgmml"), "--reference", str(first)]) == 0
    with pytest.raises(SystemExit):
        main(["convert", "exampleNet", "-o", str(tmp_path / "c.xlsx"), "--reference", str(first)])


def test_convert_from_sif(tmp_path):
    sif = tmp_path / "net.sif"
    sif.write_text("A\t1\tB\nB\t-1\tC\n")
    assert main(["convert", str(sif), "-o", "net.xlsx"]) == 0
    model = netflux.read_model(tmp_path / "output" / "net.xlsx")
    assert [r.rule for r in model.reactions] == ["A => B", "!B => C"]
    assert main(["simulate", str(sif)]) == 0


def test_convert_from_sbml_qual(tmp_path):
    assert main(["convert", str(DATA / "toy_qual.xml"), "-o", "toy.xlsx"]) == 0
    assert netflux.read_model(tmp_path / "output" / "toy.xlsx").species_ids == ["A", "B", "C", "D", "E", "Unused"]


def test_model_file_path(tmp_path):
    path = netflux.write_model(netflux.load_example("exampleNet"), tmp_path / "mine.xlsx")
    assert main(["info", str(path)]) == 0


def test_everything_defaults_to_output_folder(tmp_path):
    assert main(["simulate", "exampleNet", "--t-end", "1", "-o", "data.csv"]) == 0
    assert main(["convert", "exampleNet", "-o", "exampleNet.sif"]) == 0
    assert main(["convert", "exampleNet", "-o", "exampleNet.xgmml", "--style", "style.xml"]) == 0
    assert main(["convert", "exampleNet", "-o", "code/NetfluxODE.py"]) == 0
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "output/code/NetfluxODE.py",
        "output/data.csv",
        "output/exampleNet.sif",
        "output/exampleNet.xgmml",
        "output/exampleNet_nodeAttributes.txt",
        "output/style.xml",
    ]


def test_output_dir_option(tmp_path):
    assert main(["convert", "exampleNet", "-o", "net.sif", "--output-dir", "results"]) == 0
    assert (tmp_path / "results" / "net.sif").exists()
    assert not (tmp_path / "output").exists()


def test_error_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.sif"
    bad.write_text("only two\n")
    assert main(["convert", str(bad), "-o", "x.xlsx"]) == 1
    assert "error" in capsys.readouterr().err
