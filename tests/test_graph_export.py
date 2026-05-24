import pandas as pd
from herbarium_scribe.graph_export import export_graph


def test_graph_node_edge_creation():
    df = pd.DataFrame([{"occurrenceID": "id1", "catalogNumber": "E1", "scientificName": "Rosa canina", "recordedBy": "A. Smith", "country": "France", "stateProvince": "Normandie", "institutionCode": "E"}])
    nodes, edges = export_graph(df)
    assert set(nodes["node_type"]) >= {"Specimen", "Taxon", "Collector", "Place", "Institution"}
    assert set(edges["edge_type"]) >= {"HAS_TAXON", "COLLECTED_BY", "COLLECTED_IN", "HELD_BY"}
