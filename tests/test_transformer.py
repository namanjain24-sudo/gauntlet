import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from gauntlet.defender import make_defender  # noqa: E402
from gauntlet.transformer import TransformerDefender  # noqa: E402


@pytest.fixture(scope="module")
def parts(small_df):
    tr = small_df[small_df.split == "train"].sample(400, random_state=0)
    val = small_df[(small_df.split == "val") & (small_df.label == 0)].text.head(150)
    test = small_df[small_df.split == "test"]
    return tr, val, test.groupby("label", group_keys=False).head(20)      # 20 attacks + 20 benign


@pytest.fixture(scope="module", params=["transformer", "transformer_windowed"])
def fitted(request, parts):
    tr, val, _ = parts
    d = TransformerDefender(request.param, fpr=0.05, steps=12, batch=16)
    return d.fit(tr.text, tr.label).calibrate(val)


def test_scores_are_probabilities_and_threshold_controls_fpr(fitted, parts):
    _, val, test = parts
    s = fitted.score(test.text)
    assert s.shape == (len(test),) and np.all((0 <= s) & (s <= 1))
    assert (fitted.score(val) > fitted.threshold).mean() <= 0.05 + 1 / len(val)   # quantile tolerance: one sample


def test_scoring_is_deterministic_and_order_independent(fitted, parts):
    _, _, test = parts
    texts = list(test.text)
    a = fitted.score(texts)
    b = fitted.score(texts[::-1])[::-1]
    assert np.allclose(a, b, atol=1e-5)


def test_training_actually_learns(parts):
    tr, val, test = parts
    d = TransformerDefender("transformer", fpr=0.05, steps=60, batch=16, lr=1e-4).fit(tr.text, tr.label).calibrate(val)
    s, y = d.score(test.text), test.label.to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean() + 0.3
    assert d.detect(test.text)[y == 1].mean() > 0.6


def test_fit_is_reproducible(parts):
    tr, _, test = parts
    runs = [TransformerDefender("transformer", steps=4, batch=16, seed=3).fit(tr.text, tr.label).score(test.text)
            for _ in range(2)]
    assert np.allclose(runs[0], runs[1], atol=1e-5)


def test_factory_and_validation():
    assert isinstance(make_defender("transformer_windowed"), TransformerDefender)
    assert make_defender("naive").kind == "naive"
    with pytest.raises(ValueError):
        TransformerDefender("nope")
