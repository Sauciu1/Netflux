"""Cytoscape XGMML export and the Netflux SBGN style file."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ..examples import SBGN_STYLE
from ..model import Model
from ..output import output_path
from .sif import network_edges

__all__ = ["to_xgmml", "write_xgmml", "write_sbgn_style"]


_XGMML = "http://www.cs.rpi.edu/XGMML"
_CY = "http://www.cytoscape.org"
ET.register_namespace("", _XGMML)
ET.register_namespace("cy", _CY)

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


def _el(parent: ET.Element | None, tag: str, **attrib: str) -> ET.Element:
    tag = f"{{{_XGMML}}}{tag}"
    return ET.Element(tag, attrib) if parent is None else ET.SubElement(parent, tag, attrib)


def _att(parent: ET.Element, name: str, value: object) -> None:
    _el(parent, "att", type="string", name=name, value=str(value), **{f"{{{_CY}}}type": "String"})


def _reference_graphics(path: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """``{label: <graphics> attributes}`` for the nodes and edges of a Cytoscape XGMML file."""
    found: dict[str, dict[str, dict]] = {"node": {}, "edge": {}}
    for element in ET.parse(path).getroot():
        kind = element.tag.rsplit("}", 1)[-1]
        if kind in found and "label" in element.attrib:
            graphics = next((c for c in element if c.tag.rsplit("}", 1)[-1] == "graphics"), None)
            if graphics is not None:
                plain = {k: v for k, v in graphics.attrib.items() if not k.startswith("{")}
                found[kind].setdefault(element.attrib["label"], plain)
    return found["node"], found["edge"]


def to_xgmml(model: Model, reference: str | Path | None = None) -> str:
    """Nodes and edges matching a Cytoscape ``reference`` keep its layout and graphics."""
    ref_nodes, ref_edges = _reference_graphics(reference) if reference is not None else ({}, {})
    edges, connectors = network_edges(model)
    graph = _el(None, "graph", label="Cytoscape", directed="1")
    _el(graph, "att", name="documentVersion", value="1.1")

    nodes = [(s.id, s.id, s.type) for s in model.species] + [(c, "AND", "connector") for c in connectors]
    node_ids: dict[str, int] = {}
    for i, (name, label, kind) in enumerate(nodes, start=1):
        node_ids.setdefault(name, i)
        node = _el(graph, "node", label=name, id=str(i))
        for att_name, value in (("shared name", name), ("canonicalName", label), ("name", name),
                                ("Type", kind), ("HyperEdge.EntityType", "RegularNode")):
            _att(node, att_name, value)
        shape, h, w = NODE_GRAPHICS.get(kind, NODE_GRAPHICS[""])
        defaults = {"type": shape, "h": h, "w": w, "fill": "#FFFFFF", "width": "3.0", "outline": "#000000"}
        _el(node, "graphics", **(defaults | ref_nodes.get(name, {})))

    for e in edges:
        label = f"{e.source} ({e.sign}) {e.target}"
        edge = _el(graph, "edge", label=label, source=str(node_ids[e.source]), target=str(node_ids[e.target]))
        _att(edge, "canonicalName", label)
        _att(edge, "interaction", e.sign)
        graphics = _el(edge, "graphics", **({"width": "2.0", "fill": "#000000"} | ref_edges.get(label, {})))
        _att(graphics, "EDGE_TARGET_ARROW_SHAPE", "ARROW" if e.sign == 1 else "T")

    ET.indent(graph)
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(graph, encoding="unicode")


def write_xgmml(
    model: Model, path: str | Path | None = None, reference: str | Path | None = None
) -> Path:
    path = output_path(path, f"{model.name}.xgmml")
    path.write_text(to_xgmml(model, reference), encoding="utf-8")
    return path


def write_sbgn_style(path: str | Path | None = None) -> Path:
    path = output_path(path, "netflux_sbgn_style.xml")
    shutil.copyfile(SBGN_STYLE, path)
    return path
