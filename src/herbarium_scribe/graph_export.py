from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .metadata import clean_str


def node_id(kind: str, value: str) -> str:
    key = f"{kind}:{clean_str(value).lower()}"
    return f"{kind.lower()}_{hashlib.md5(key.encode('utf-8')).hexdigest()[:12]}"


def add_node(nodes: dict[str, dict], kind: str, label: str, **attrs) -> str:
    label = clean_str(label)
    nid = node_id(kind, label)
    if nid not in nodes:
        nodes[nid] = {"node_id": nid, "node_type": kind, "label": label, **attrs}
    return nid


def add_edge(edges: list[dict], source: str, target: str, edge_type: str) -> None:
    if source and target:
        edges.append({"source": source, "target": target, "edge_type": edge_type})


def export_graph(records: pd.DataFrame, paths: dict[str, Path] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for _, row in records.iterrows():
        occ = clean_str(row.get("occurrenceID", ""))
        specimen = add_node(nodes, "Specimen", occ, catalogNumber=clean_str(row.get("catalogNumber", "")))
        sci = clean_str(row.get("scientificName_canonical", "")) or clean_str(row.get("scientificName", ""))
        if sci:
            add_edge(edges, specimen, add_node(nodes, "Taxon", sci), "HAS_TAXON")
        collector = clean_str(row.get("recordedBy", ""))
        if collector:
            add_edge(edges, specimen, add_node(nodes, "Collector", collector), "COLLECTED_BY")
        place_parts = [clean_str(row.get("stateProvince_normalised", "")) or clean_str(row.get("stateProvince", "")), clean_str(row.get("country_normalised", "")) or clean_str(row.get("country", ""))]
        place = ", ".join([p for p in place_parts if p])
        if place:
            add_edge(edges, specimen, add_node(nodes, "Place", place), "COLLECTED_IN")
        inst = clean_str(row.get("institutionCode", ""))
        if inst:
            add_edge(edges, specimen, add_node(nodes, "Institution", inst), "HELD_BY")
    nodes_df = pd.DataFrame(nodes.values())
    edges_df = pd.DataFrame(edges).drop_duplicates() if edges else pd.DataFrame(columns=["source", "target", "edge_type"])
    if paths:
        nodes_df.to_csv(paths["processed"] / "graph_nodes.csv", index=False)
        edges_df.to_csv(paths["processed"] / "graph_edges.csv", index=False)
    return nodes_df, edges_df
