import openpyxl
import pytest

import netflux
from netflux import Model, ModelError, Reaction, Species, read_model, write_model

from .conftest import EXAMPLE_COUNTS


def test_bundled_examples_listed():
    assert netflux.list_examples() == sorted(EXAMPLE_COUNTS)


@pytest.mark.parametrize("name,counts", sorted(EXAMPLE_COUNTS.items()))
def test_example_counts_and_no_warnings(examples, name, counts):
    model = examples[name]
    assert (len(model.species), len(model.reactions)) == counts
    assert model.warnings == []
    assert model.name == name


def test_example_net_contents(examples):
    m = examples["exampleNet"]
    assert m.species_ids == ["A", "B", "C", "D", "E"]
    assert [r.rule for r in m.reactions] == ["=> A", "=> B", "A => C", "B => D", "C & !D => E", "E => C"]
    assert [r.weight for r in m.reactions] == [0, 0, 1, 1, 1, 1]
    assert m.get_species("B").type == "mRNA"
    assert m.get_reaction("r5").module == "middle"


def test_round_trip(tmp_path, examples):
    original = examples["kraeutler_betaAdrenergic_bmcsysbio2010"]
    path = write_model(original, tmp_path / "copy.xlsx")
    again = read_model(path)
    assert again.species == [
        Species(**{**s.__dict__, "name": s.name or s.id}) for s in original.species
    ]
    assert again.reactions == original.reactions


def _write_raw(path, species_rows, reaction_rows, sheets=("species", "reactions")):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    data = {"species": species_rows, "reactions": reaction_rows}
    for name in sheets:
        ws = wb.create_sheet(name)
        ws.append([name.title()])
        ws.append(["header"])
        for row in data[name]:
            ws.append(row)
        for row in ws.iter_rows():
            for cell in row:
                if cell.data_type == "f":  # keep "=> A" as text
                    cell.data_type = "s"
    wb.save(path)
    return path


def test_missing_parameters_get_defaults(tmp_path):
    path = _write_raw(
        tmp_path / "m.xlsx",
        [[None, "A", "a", 0, 1, 1, "protein"], [None, "B", "b", None, "x", 2]],
        [[None, "r1", "=> A", 1, 1.4, 0.5], [None, "r2", "A => B", None, 2, None]],
    )
    m = read_model(path)
    b = m.get_species("B")
    assert (b.y0, b.ymax, b.tau) == (0.0, 1.0, 2.0)
    r2 = m.get_reaction("r2")
    assert (r2.weight, r2.n, r2.ec50) == (1.0, 2.0, 0.5)
    assert any("Species B" in w for w in m.warnings)
    assert any("Reaction r2" in w for w in m.warnings)


def test_blank_rows_are_skipped_and_columns_stay_aligned(tmp_path):
    path = _write_raw(
        tmp_path / "m.xlsx",
        [[None, "A", "a", 0, 1, 1, "protein"], [None, None, None, None, None, None, "junk"],
         [None, "B", "b", 0, 1, 1, "mRNA"]],
        [[None, "r1", "=> A", 1, 1.4, 0.5], [None, "r2", None, 1, 1, 1], [None, "r3", "A => B", 1, 1.4, 0.5]],
    )
    m = read_model(path)
    assert m.species_ids == ["A", "B"]
    assert [s.type for s in m.species] == ["protein", "mRNA"]
    assert m.reaction_ids == ["r1", "r3"]


def test_missing_sheet(tmp_path):
    path = _write_raw(tmp_path / "m.xlsx", [[None, "A"]], [], sheets=("species",))
    with pytest.raises(ModelError, match="reactions"):
        read_model(path)


def test_no_species(tmp_path):
    path = _write_raw(tmp_path / "m.xlsx", [], [[None, "r1", "=> A", 1, 1.4, 0.5]])
    with pytest.raises(ModelError, match="zero"):
        read_model(path)


def test_write_requires_xlsx(tmp_path):
    with pytest.raises(ValueError):
        write_model(Model("m", [Species("A")], [Reaction("r1", "=> A")]), tmp_path / "m.xls")
