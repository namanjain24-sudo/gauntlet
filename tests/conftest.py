import pytest

from gauntlet.corpus import build_dataset


@pytest.fixture(scope="session")
def small_df():
    return build_dataset(seed=0, n_intents=160, n_benign=1200)
