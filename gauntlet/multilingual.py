"""Does non-English training data narrow the real-data AUC gap, or is it purely an attack-style gap?

The real human-written injection set (``deepset/prompt-injections``) is English + German, while
Gauntlet's own training data is English only. This mixes German/French/Spanish benign text into
training (:func:`gauntlet.corpus.build_dataset`'s ``multilingual_frac``) and re-measures AUC on the
whole real-data set and on a German-looking subset of it, so the two possible explanations --
"the model has never seen this language" vs. "the model has never seen this attack style" -- can be
told apart. Language is a heuristic (diacritics or common German stopwords), not ground truth,
since the dataset carries no language column.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import external
from .corpus import build_dataset
from .defender import make_defender

DEFAULT_CACHE = Path("data/external/deepset.json")

_DE_CHARS = re.compile(r"[äöüßÄÖÜ]")
_DE_WORDS = re.compile(r"\b(und|ist|nicht|für|mit|das|die|der|ich|du|sie|wie|was|ein|eine|kannst|bitte|auf|nach|zu)\b", re.I)


def looks_german(text: str) -> bool:
    return bool(_DE_CHARS.search(text)) or len(_DE_WORDS.findall(text)) >= 2


def _auc_on(defender, ext: pd.DataFrame) -> dict:
    scores = defender.score(ext.text)
    y = ext.label.to_numpy()
    return {"n": int(len(ext)), "auc": float(roc_auc_score(y, scores)) if y.sum() and (~y.astype(bool)).sum() else float("nan"),
            "recall": float((scores[y == 1] > defender.threshold).mean()) if y.sum() else float("nan")}


def run(kind: str = "hardened", seed: int = 0, multilingual_frac: float = 0.25,
        external_cache: Path = DEFAULT_CACHE, log=print) -> dict:
    ext = external.fetch(external_cache)
    if ext is None:
        raise RuntimeError(f"real-data set not cached at {external_cache} and could not be downloaded")
    is_de = np.array([looks_german(t) for t in ext.text])
    log(f"real-data set: {len(ext)} rows, {is_de.sum()} look German (heuristic), {len(ext) - is_de.sum()} look English")

    rows = {}
    for tag, frac in [("english_only", 0.0), ("multilingual", multilingual_frac)]:
        df = build_dataset(seed, multilingual_frac=frac)
        tr = df[df.split == "train"]
        val = df[(df.split == "val") & (df.label == 0)].text
        d = make_defender(kind, seed=seed).fit(tr.text, tr.label).calibrate(val)
        rows[tag] = {
            "overall": _auc_on(d, ext), "german_subset": _auc_on(d, ext[is_de]), "english_subset": _auc_on(d, ext[~is_de]),
            "synthetic_test_fpr": float(d.detect(df[(df.split == "test") & (df.label == 0)].text).mean()),
        }
        log(f"{tag:<13} overall AUC {rows[tag]['overall']['auc']:.3f}   "
            f"german-subset AUC {rows[tag]['german_subset']['auc']:.3f}   "
            f"english-subset AUC {rows[tag]['english_subset']['auc']:.3f}")
    return {"kind": kind, "multilingual_frac": multilingual_frac, "n_german": int(is_de.sum()),
            "n_english": int(len(ext) - is_de.sum()), "results": rows}


def render(r: dict) -> str:
    head = ["training data", "overall AUC", f"German subset (n={r['n_german']})", f"English subset (n={r['n_english']})"]
    body = [[tag, f"{v['overall']['auc']:.3f}", f"{v['german_subset']['auc']:.3f}", f"{v['english_subset']['auc']:.3f}"]
            for tag, v in r["results"].items()]
    w = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    line = lambda cells: "  ".join(str(c).ljust(n) for c, n in zip(cells, w))
    return "\n".join([f"defender: {r['kind']}  (heuristic language split, not ground truth)",
                      line(head), line(["-" * n for n in w])] + [line(row) for row in body])
