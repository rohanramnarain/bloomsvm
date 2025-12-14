from src.acquisition import synth, scrape


def test_synth_min_items():
    rows = synth.synthesize(n=5, model="template", multilabel_p=0.0, domains=["stem"], fast=True)
    assert len(rows) == 5
    assert all(len(r.labels) == 6 for r in rows)


def test_scrape_safe():
    # robots check should return bool even for fake url
    assert scrape.can_fetch("https://example.com") in [True, False]
