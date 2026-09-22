"""Does a bigger transformer / longer schedule close the gap on the family it fails hardest on?

The headline result found the fine-tuned MiniLM-L6 defender's weakest spot was ``indirect``
(hidden-in-a-document) injections at 13% recall when that family is held out of training,
worse than an 86M-parameter-cheaper TF-IDF model. This module isolates that one question:
hold ``indirect`` out of training, fit each (model, steps) config, and measure recall on it.

Only the ``indirect`` fold is computed (not all 8 families), and only the un-windowed
architecture (raw text in, same as the failing ``transformer`` row), so a bigger model's
effect on capacity/training length is not entangled with the windowing fix that already
works. Runtime is steps x model-size, not dataset size (each fit uses a fixed step budget).
"""
from __future__ import annotations

import time

import pandas as pd

from .corpus import build_dataset
from .transformer import TransformerDefender

CONFIGS = [
    {"label": "MiniLM-L6 (22M), 160 steps [headline config]", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "steps": 160},
    {"label": "MiniLM-L6 (22M), 640 steps [4x longer]", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "steps": 640},
    {"label": "MiniLM-L12 (33M), 160 steps [1.5x deeper]", "model_name": "sentence-transformers/all-MiniLM-L12-v2", "steps": 160},
    {"label": "MiniLM-L12 (33M), 640 steps [deeper + longer]", "model_name": "sentence-transformers/all-MiniLM-L12-v2", "steps": 640},
]


def run(configs=CONFIGS, seed: int = 0, family: str = "indirect", df: pd.DataFrame | None = None, log=print) -> list[dict]:
    df = build_dataset(seed) if df is None else df
    train = df[df.split == "train"]
    val = df[(df.split == "val") & (df.label == 0)].text
    test = df[df.split == "test"]
    attacks, benign = test[test.family == family].text, test[test.label == 0].text
    train_excl = train[train.family != family]

    rows = []
    for cfg in configs:
        t0 = time.time()
        full = TransformerDefender("transformer", seed=seed, steps=cfg["steps"],
                                   model_name=cfg["model_name"]).fit(train.text, train.label).calibrate(val)
        excl = TransformerDefender("transformer", seed=seed, steps=cfg["steps"],
                                   model_name=cfg["model_name"]).fit(train_excl.text, train_excl.label).calibrate(val)
        rows.append({
            **cfg, "family": family, "seconds": round(time.time() - t0),
            "id_recall": float(full.detect(attacks).mean()),      # family was in training
            "ood_recall": float(excl.detect(attacks).mean()),     # family held out
            "fpr": float(excl.detect(benign).mean()),
        })
        log(f"{cfg['label']:<45} ood_recall={rows[-1]['ood_recall']:.1%}  "
            f"id_recall={rows[-1]['id_recall']:.1%}  fpr={rows[-1]['fpr']:.1%}  ({rows[-1]['seconds']}s)")
    return rows


def render(rows: list[dict]) -> str:
    head = ["config", f"recall on unseen {rows[0]['family']}", "recall when seen", "benign FPR", "fit time"]
    body = [[r["label"], f"{r['ood_recall']:.0%}", f"{r['id_recall']:.0%}", f"{r['fpr']:.1%}", f"{r['seconds']}s"] for r in rows]
    w = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    line = lambda cells: "  ".join(str(c).ljust(n) for c, n in zip(cells, w))
    return "\n".join([line(head), line(["-" * n for n in w])] + [line(r) for r in body])
