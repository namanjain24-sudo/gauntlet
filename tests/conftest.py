import os
import tempfile

import pytest

# keep the model cache out of the developer's ~/.cache during tests
os.environ.setdefault("GAUNTLET_CACHE", tempfile.mkdtemp(prefix="gauntlet-cache-"))

from gauntlet.corpus import build_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return build_dataset(seed=0, n_intents=160, n_benign=1200)
