import pytest

pytest.importorskip("torch")

from gauntlet import scale


def test_run_and_render_shapes(small_df):
    configs = [{"label": "tiny-fast", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "steps": 4}]
    rows = scale.run(configs, df=small_df, family="indirect")
    assert len(rows) == 1 and {"id_recall", "ood_recall", "fpr", "seconds"} <= rows[0].keys()
    assert 0 <= rows[0]["ood_recall"] <= 1
    out = scale.render(rows)
    assert "tiny-fast" in out and "unseen indirect" in out
