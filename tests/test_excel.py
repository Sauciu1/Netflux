import math

import openpyxl
import pytest

import netflux
from netflux import Model, ModelError, Reaction, Species, parse_rule, read_excel_frames, read_model, write_model
from netflux.examples import example_path

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
        write_model(Model("m", [Species("A")], [Reaction.from_rule("r1", "=> A")]), tmp_path / "m.xls")


def test_rule_without_arrow_is_skipped_with_warning(tmp_path):
    path = _write_raw(
        tmp_path / "m.xlsx",
        [[None, "A", "a", 0, 1, 1], [None, "B", "b", 0, 1, 1]],
        [[None, "r1", "=> A", 1, 1.4, 0.5], [None, "r2", "A B", 1, 1.4, 0.5]],
    )
    m = read_model(path)
    assert m.reaction_ids == ["r1"]
    assert any("Reaction r2" in w and "arrow" in w and "skipped" in w for w in m.warnings)


def test_rules_are_written_in_normal_form(tmp_path, examples):
    original = examples["kraeutler_betaAdrenergic_bmcsysbio2010"]
    again = read_model(write_model(original, tmp_path / "copy.xlsx"))
    assert again.get_reaction("r6").rule == "NE & !B1ARPG & !B1ARPA => B1AR"  # was written with '+'
    assert again.topology == original.topology


@pytest.mark.parametrize("name", sorted(EXAMPLE_COUNTS))
def test_frames_agree_with_model(examples, name):
    model = examples[name]
    species, reactions = read_excel_frames(example_path(name))
    assert species["id"].tolist() == model.species_ids
    assert species["module"].tolist() == [s.module for s in model.species]
    for column in ("y0", "ymax", "tau"):
        assert species[column].tolist() == [getattr(s, column) for s in model.species]
    assert reactions["id"].tolist() == model.reaction_ids
    assert [parse_rule(rule) for rule in reactions["rule"]] == [(r.inputs, r.outputs) for r in model.reactions]
    for column in ("weight", "n", "ec50"):
        assert reactions[column].tolist() == [getattr(r, column) for r in model.reactions]


def test_frames_name_fixed_columns_and_keep_file_headers():
    species, reactions = read_excel_frames(example_path("zeigler_fibroblast_jmcc2016"))
    assert list(species.columns) == ["module", "id", "name", "y0", "ymax", "tau", "type", "gene name", "notes"]
    assert list(reactions.columns[:6]) == ["module", "id", "rule", "weight", "n", "ec50"]
    assert list(reactions.columns[6:]) == [
        "source", "notes", "PMID", "Secondary Reference", "Unnamed: 10", "Tertiary Reference", "AND references",
    ]
    assert species.loc[species["id"] == "AngII", "gene name"].item() == "AGT"


def test_frames_number_repeated_headers():
    _, reactions = read_excel_frames(example_path("tan_mechanosignaling_ploscb2017"))
    assert reactions.columns.is_unique
    assert {"PMID", "PMID.1", "PMID.4"} <= set(reactions.columns)


def test_frames_leave_missing_parameters_blank_and_drop_empty_rows(tmp_path):
    path = _write_raw(
        tmp_path / "m.xlsx",
        [[None, "A", "a", 0, 1, 1, "protein"], [None, None, None, 5, 5, 5, "junk"], [None, "B", None, None, "x", 2]],
        [[None, "r1", "=> A", 1, 1.4, 0.5], [None, "r2", None, 1, 1, 1], [None, "r3", "A B", None, 2, None]],
    )
    species, reactions = read_excel_frames(path)
    assert species["id"].tolist() == ["A", "B"]
    assert list(species.index) == [0, 1]
    assert species["name"].tolist() == ["a", ""]
    assert species["tau"].tolist() == [1.0, 2.0]
    assert math.isnan(species.loc[1, "y0"]) and math.isnan(species.loc[1, "ymax"])
    assert reactions["id"].tolist() == ["r1", "r3"]
    assert reactions.loc[1, "rule"] == "A B"  # not parsed, so kept
    assert math.isnan(reactions.loc[1, "weight"]) and reactions.loc[1, "n"] == 2.0


def test_frames_of_short_sheets_still_have_named_columns(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.title = "species"
    wb["species"].append(["Species"])  # title row only, no header row
    wb.create_sheet("reactions")
    wb["reactions"].append(["Reactions"])
    wb["reactions"].append(["header"])
    wb["reactions"].append([None, "r1"])
    wb.save(tmp_path / "m.xlsx")

    species, reactions = read_excel_frames(tmp_path / "m.xlsx")
    assert species.empty
    assert list(species.columns) == ["module", "id", "name", "y0", "ymax", "tau", "type"]
    assert reactions.empty  # r1 has no rule
    assert list(reactions.columns) == ["module", "id", "rule", "weight", "n", "ec50"]


def test_frames_missing_sheet(tmp_path):
    path = _write_raw(tmp_path / "m.xlsx", [[None, "A"]], [], sheets=("species",))
    with pytest.raises(ModelError, match="reactions"):
        read_excel_frames(path)
