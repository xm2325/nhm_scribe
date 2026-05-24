from herbarium_scribe.evaluate import exact_match, token_f1, evidence_proxy


def test_exact_match_normalises_case():
    assert exact_match("Rosa canina", "rosa canina") == 1


def test_token_f1_partial_overlap():
    assert 0 < token_f1("Rosa", "Rosa canina") < 1


def test_evidence_proxy_is_not_cer():
    assert evidence_proxy("Rosa canina", "label text Rosa canina here") == 1.0
