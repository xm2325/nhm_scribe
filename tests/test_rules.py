from herbarium_scribe.extract_rules import extract_barcode, extract_rule_based


def test_rule_based_barcode_extraction():
    assert extract_barcode("barcode E00633257 label") == "E00633257"


def test_rule_based_coordinates():
    rec = extract_rule_based("E00633257 Rosa canina France 1902 lat 48.86 lon 2.35")
    assert rec["catalogNumber"]["value"] == "E00633257"
    assert rec["decimalLatitude"]["value"] == "48.86"
    assert rec["decimalLongitude"]["value"] == "2.35"
