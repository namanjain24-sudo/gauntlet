"""Sim-to-real check on a public, human-written prompt-injection dataset.

Downloads ``deepset/prompt-injections`` via the HuggingFace datasets-server (no extra
dependencies) and caches it locally. Fails soft: if offline, the report just omits it.
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .defender import Defender

URL = ("https://datasets-server.huggingface.co/rows?dataset=deepset%2Fprompt-injections"
       "&config=default&split={split}&offset={off}&length=100")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(cache: Path) -> pd.DataFrame | None:
    if cache.exists():
        return pd.DataFrame(json.loads(cache.read_text()))
    rows = []
    try:
        for split in ("train", "test"):
            off = 0
            while True:
                with urllib.request.urlopen(URL.format(split=split, off=off), timeout=20, context=_ssl_context()) as r:
                    batch = json.load(r)["rows"]
                if not batch:
                    break
                rows += [b["row"] for b in batch]
                off += 100
    except Exception as exc:  # noqa: BLE001 - any network failure just disables this section
        print(f"[external] skipped ({exc})")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


def evaluate(defender: Defender, ext: pd.DataFrame) -> dict:
    scores = defender.score(ext.text)
    y = ext.label.to_numpy()
    return {
        "n": int(len(ext)), "n_attacks": int(y.sum()),
        "auc": float(roc_auc_score(y, scores)),
        "recall": float((scores[y == 1] > defender.threshold).mean()),
        "fpr": float((scores[y == 0] > defender.threshold).mean()),
    }
