"""Port of Netflux2sif.m / Netflux2xgmml.m: Cytoscape SIF and XGMML export."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from ._data import SBGN_STYLE
from .model import Model
from .paths import output_path

__all__ = [
    "Edge",
    "network_edges",
    "to_sif",
    "to_node_attributes",
    "write_sif",
    "to_xgmml",
    "write_xgmml",
    "write_sbgn_style",
]


@dataclass(frozen=True)
class Edge:
    source: str
    sign: int  # 1 activating, -1 inhibiting
    target: str


def network_edges(model: Model) -> tuple[list[Edge], list[str]]:
    """SIF edges in MATLAB's order, plus the ``rcn<k>`` connector node names."""
    ids = model.species_ids
    edges: list[Edge] = []
    connectors: list[str] = []
    for k, rule in enumerate(model.topology.rules, start=1):
        if len(rule.products) != 1 or not rule.reactants:
            continue
        product = ids[rule.products[0]]
        if len(rule.reactants) == 1:
            edges.append(Edge(ids[rule.reactants[0]], -1 if rule.negated[0] else 1, product))
        else:
            connector = f"rcn{k}"
            connectors.append(connector)
            edges.append(Edge(connector, 1, product))
            for i, neg in zip(rule.reactants, rule.negated):
                edges.append(Edge(ids[i], -1 if neg else 1, connector))
    return edges, connectors


def to_sif(model: Model) -> str:
    """Byte-identical to MATLAB's Netflux2sif output."""
    edges, _ = network_edges(model)
    return "".join(f"{e.source}\t{e.sign}\t{e.target}\n" for e in edges)


def to_node_attributes(model: Model) -> str:
    _, connectors = network_edges(model)
    lines = ["Type (class=String)\n"]
    lines += [f"{s.id} = {s.type}\n" for s in model.species]
    lines += [f"{c} = connector\n" for c in connectors]
    return "".join(lines)


def write_sif(
    model: Model, directory: str | Path | None = None, name: str | None = None
) -> tuple[Path, Path]:
    """Writes ``<name>.sif`` and ``<name>_nodeAttributes.txt``."""
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or model.name
    sif = directory / f"{name}.sif"
    attrs = directory / f"{name}_nodeAttributes.txt"
    sif.write_text(to_sif(model), newline="")
    attrs.write_text(to_node_attributes(model), newline="")
    return sif, attrs


# --- XGMML

_GRAPH_HEADER = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<graph label="Cytoscape" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:cy="http://www.cytoscape.org" xmlns="http://www.cs.rpi.edu/XGMML"  directed="1">\n'
    '<att name="documentVersion" value="1.1"/>\n'
)

# species type -> (shape, height, width)
NODE_GRAPHICS = {
    "": ("RECTANGLE", "30.0", "70.0"),
    "connector": ("ELLIPSE", "28.0", "28.0"),
    "mRNA": ("HEXAGON", "40.0", "85.0"),
    "phenotype": ("HEXAGON", "40.0", "85.0"),
    "protein": ("RECTANGLE", "30.0", "70.0"),
    "smallMolecule": ("ELLIPSE", "40.0", "40.0"),
    "intermediate": ("ELLIPSE", "20.0", "20.0"),
}

_COPIED_NODE_ATTRIBUTES = {"shared name", "canonicalName", "name", "Type", "HyperEdge.EntityType"}
_GRAPHICS_FIELDS = ("type", "h", "w", "x", "y", "z", "fill", "width", "outline")
_EMPTY_VALUE = {"string": " ", "real": "0.0", "integer": "0", "boolean": "0"}
_CY_TYPE = {"string": "String", "real": "Double", "integer": "Integer", "boolean": "Boolean"}


def _q(value: str) -> str:
    """Escape an XML attribute value, keeping tabs as MATLAB does."""
    return escape(str(value), {'"': "&quot;"})


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attr(element: ET.Element, name: str, default: str | None = None) -> str | None:
    return element.attrib.get(name, default)


def _cy_type(att: ET.Element) -> str | None:
    for key, value in att.attrib.items():
        if key == "{http://www.cytoscape.org}type" or key == "cy:type":
            return value
    return None


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in element if _local(c.tag) == name]


@dataclass
class _RefNode:
    atts: list[ET.Element]
    graphics: ET.Element | None


@dataclass
class _RefEdge:
    atts: list[ET.Element]
    graphics: ET.Element | None


def _read_reference(path: str | Path) -> tuple[dict[str, _RefNode], dict[str, _RefEdge]]:
    root = ET.parse(path).getroot()
    nodes: dict[str, _RefNode] = {}
    for node in _children(root, "node"):
        label = _attr(node, "label")
        if label is None or label in nodes:
            continue
        graphics = next(iter(_children(node, "graphics")), None)
        nodes[label] = _RefNode(_children(node, "att"), graphics)
    edges: dict[str, _RefEdge] = {}
    for edge in _children(root, "edge"):
        label = _attr(edge, "label")
        if label is None or label in edges:
            continue
        graphics = next(iter(_children(edge, "graphics")), None)
        edges[label] = _RefEdge(_children(edge, "att"), graphics)
    return nodes, edges


def _att_line(att: ET.Element, empty_values: dict[str, str]) -> str:
    att_type = _attr(att, "type", "string")
    name = _attr(att, "name", "")
    value = _attr(att, "value")
    if value is None:
        value = empty_values.get(att_type, "")
    cy_type = _cy_type(att) or _CY_TYPE.get(att_type, "String")
    return f'<att type="{_q(att_type)}" name="{_q(name)}" value="{_q(value)}" cy:type="{_q(cy_type)}"/>\n'


def to_xgmml(model: Model, reference: str | Path | None = None) -> str:
    """Nodes and edges matching a Cytoscape ``reference`` keep its layout and graphics."""
    ref_nodes, ref_edges = _read_reference(reference) if reference is not None else ({}, {})
    edges, connectors = network_edges(model)
    node_names = model.species_ids + connectors
    node_labels = model.species_ids + ["AND"] * len(connectors)
    node_types = [s.type for s in model.species] + ["connector"] * len(connectors)
    node_id = {}
    for i, name in enumerate(node_names, start=1):
        node_id.setdefault(name, i)

    out = [_GRAPH_HEADER]
    for i, (name, label, ntype) in enumerate(zip(node_names, node_labels, node_types), start=1):
        out.append(f'<node label="{_q(name)}" id="{i}">\n')
        ref = ref_nodes.get(name)
        if ref is not None:
            for att in ref.atts:
                if len(ref.atts) == 1 or _attr(att, "name") in _COPIED_NODE_ATTRIBUTES:
                    out.append(_att_line(att, _EMPTY_VALUE))
            g = ref.graphics
            fields = (
                "".join(
                    f' {f}="{_q(_attr(g, f))}"' for f in _GRAPHICS_FIELDS if _attr(g, f) is not None
                )
                if g is not None
                else ""
            )
            out.append(f"<graphics{fields}>\n")
        else:
            for att_name, value in (
                ("shared name", name),
                ("canonicalName", label),
                ("name", name),
                ("Type", ntype),
                ("HyperEdge.EntityType", "RegularNode"),
            ):
                out.append(
                    f'<att type="string" name="{att_name}" value="{_q(value)}" cy:type="String"/>\n'
                )
            shape, h, w = NODE_GRAPHICS.get(ntype, NODE_GRAPHICS[""])
            out.append(
                f'<graphics type="{shape}" h="{h}" w="{w}" fill="#FFFFFF" width="3.0" outline="#000000">\n'
            )
        out.append("</graphics>\n</node>\n")

    for e in edges:
        label = f"{e.source}\t({e.sign})\t{e.target}"
        key = f"{e.source} ({e.sign}) {e.target}"
        source, target = node_id[e.source], node_id[e.target]
        arrow = "ARROW" if e.sign == 1 else "T"
        ref = ref_edges.get(key)
        if ref is not None:
            out.append(f'<edge label="{_q(label)}" source="{source}" target="{target}" cy:directed="1">\n')
            out.extend(_att_line(att, {}) for att in ref.atts)
            g = ref.graphics
            width = _attr(g, "width", "2.0") if g is not None else "2.0"
            fill = _attr(g, "fill", "#000000") if g is not None else "#000000"
            out.append(f'<graphics width="{_q(width)}" fill="{_q(fill)}">\n')
            out.append(f'<att type="string" name="canonicalName" value="{_q(label)}"/>\n')
            out.append(f'<att type="string" name="interaction" value="{e.sign}" cy:editable="false"/>\n')
        else:
            out.append(f'<edge label="{_q(label)}" source="{source}" target="{target}">\n')
            out.append(f'<att type="string" name="canonicalName" value="{_q(label)}"/>\n')
            out.append(f'<att type="string" name="interaction" value="{e.sign}" cy:editable="false"/>\n')
            out.append('<graphics width="2.0" fill="#000000">\n')
        out.append(
            f'<att name="EDGE_TARGET_ARROW_SHAPE" value="{arrow}" type="string" cy:type="String"/>\n'
        )
        out.append("</graphics>\n</edge>\n")
    out.append("</graph>")
    return "".join(out)


def write_xgmml(
    model: Model, path: str | Path | None = None, reference: str | Path | None = None
) -> Path:
    path = output_path(path, f"{model.name}.xgmml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_xgmml(model, reference), encoding="utf-8", newline="")
    return path


def write_sbgn_style(path: str | Path | None = None) -> Path:
    path = output_path(path, "netflux_sbgn_style.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SBGN_STYLE, path)
    return path
