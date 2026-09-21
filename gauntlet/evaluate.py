"""Leave-one-family-out (LOFO) generalisation evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .defender import make_defender


def bootstrap_ci(hits: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    hits = np.asarray(hits, dtype=float)
    means = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fit(df: pd.DataFrame, kind: str, fpr: float, exclude: str | None = None, seed: int = 0):
    tr = df[df.split == "train"]
    if exclude:
        tr = tr[tr.family != exclude]
    d = make_defender(kind, fpr=fpr, seed=seed).fit(tr.text, tr.label)
    return d.calibrate(df[(df.split == "val") & (df.label == 0)].text)


def lofo(df: pd.DataFrame, kind: str, fpr: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """For each attack family: recall when it *was* seen in training vs. when withheld."""
    test = df[df.split == "test"]
    benign_test = test[test.label == 0].text
    full = _fit(df, kind, fpr, seed=seed)
    rows = []
    for fam in sorted(df[df.label == 1].family.unique()):
        attacks = test[test.family == fam].text
        held_out = _fit(df, kind, fpr, exclude=fam, seed=seed)
        id_hits, ood_hits = full.detect(attacks), held_out.detect(attacks)
        rows.append({
            "defender": kind, "family": fam,
            "id_recall": id_hits.mean(), "id_lo": bootstrap_ci(id_hits)[0], "id_hi": bootstrap_ci(id_hits)[1],
            "ood_recall": ood_hits.mean(), "ood_lo": bootstrap_ci(ood_hits)[0], "ood_hi": bootstrap_ci(ood_hits)[1],
            "fpr_test": float(full.detect(benign_test).mean()),
        })
    out = pd.DataFrame(rows)
    out["gap"] = out.id_recall - out.ood_recall
    return out
