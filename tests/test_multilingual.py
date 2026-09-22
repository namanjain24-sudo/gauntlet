import pytest

from gauntlet import multilingual
from gauntlet.corpus import build_dataset, make_benign
import random


def test_looks_german():
    assert multilingual.looks_german("Wie lösche ich alle Zeilen in SQL?")
    assert multilingual.looks_german("Das ist nicht das, was ich meinte, bitte wiederhole es")
    assert not multilingual.looks_german("How do I delete all rows in SQL?")
    assert not multilingual.looks_german("What is the capital of France?")


def test_multilingual_frac_changes_language_mix():
    rng = random.Random(0)
    mono = make_benign(300, rng, multilingual_frac=0.0)
    rng2 = random.Random(0)
    mixed = make_benign(300, rng2, multilingual_frac=0.5)
    assert sum(multilingual.looks_german(t) for t in mono) < sum(multilingual.looks_german(t) for t in mixed)


def test_build_dataset_multilingual_frac_only_touches_benign():
    plain = build_dataset(0, n_intents=30, n_benign=200)
    mixed = build_dataset(0, n_intents=30, n_benign=200, multilingual_frac=0.6)
    assert (plain[plain.label == 1].text == mixed[mixed.label == 1].text).all()   # attacks unaffected
    assert sum(multilingual.looks_german(t) for t in mixed[mixed.label == 0].text) > \
           sum(multilingual.looks_german(t) for t in plain[plain.label == 0].text)


def test_run_raises_when_real_data_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(multilingual.external, "fetch", lambda cache: None)
    with pytest.raises(RuntimeError):
        multilingual.run(external_cache=tmp_path / "nope.json")
