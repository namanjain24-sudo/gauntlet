"""Prompt-injection defenders.

Two variants with identical classifiers but different inputs:

* ``naive``    - raw text -> char/word TF-IDF -> logistic regression
* ``hardened`` - canonicalised text (+ obfuscation-flag tokens) -> the same classifier
* ``windowed`` - hardened, but scores every sentence-level window and takes the maximum,
                 so an injection cannot hide by being diluted inside a long document

The decision threshold is always calibrated on held-out *benign* prompts to a target
false-positive rate, so a defender can never look good by simply flagging everything.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .canonical import canonicalize, transform_flags


_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+|\n+|<!--|-->|\[|\]|<[^>]{1,40}>|\(|\)")


def windows(text: str) -> list[str]:
    """The whole text plus every 1- and 2-sentence window of it."""
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and len(p.strip()) >= 12]
    out = [text] + parts + [f"{a} {b}" for a, b in zip(parts, parts[1:])]
    return out


class Defender:
    def __init__(self, kind: str = "naive", fpr: float = 0.01, C: float = 20.0, seed: int = 0):
        if kind not in ("naive", "hardened", "windowed"):
            raise ValueError(f"unknown defender kind: {kind}")
        self.kind, self.fpr = kind, fpr
        self.threshold = 0.5
        self.model = Pipeline([
            ("features", FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=80_000)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
            ])),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)),
        ])

    @staticmethod
    def _harden(t: str) -> str:
        return canonicalize(t) + "".join(f" xflag{f}" for f in transform_flags(t))

    def _prep(self, texts) -> list[str]:
        if self.kind == "naive":
            return list(texts)
        return [self._harden(t) for t in texts]

    def fit(self, texts, labels) -> Defender:
        self.model.fit(self._prep(texts), np.asarray(labels))
        return self

    def score(self, texts) -> np.ndarray:
        """Probability that each text is an injection."""
        if self.kind != "windowed":
            return self.model.predict_proba(self._prep(texts))[:, 1]
        chunks, owner = [], []
        for k, t in enumerate(texts):
            ws = windows(canonicalize(t))
            flags = "".join(f" xflag{f}" for f in transform_flags(t))
            chunks += [w + flags for w in ws]
            owner += [k] * len(ws)
        probs = self.model.predict_proba(chunks)[:, 1]
        out = np.zeros(len(texts))
        np.maximum.at(out, np.asarray(owner), probs)
        return out

    def explain(self, text: str) -> dict | None:
        """For the windowed defender: the window that produced the text's score (where the injection sits).

        Returns ``{"window": str, "score": float}`` or None for non-windowed kinds. Scores are unchanged.
        """
        if self.kind != "windowed":
            return None
        ws = windows(canonicalize(text))
        flags = "".join(f" xflag{f}" for f in transform_flags(text))
        probs = self.model.predict_proba([w + flags for w in ws])[:, 1]
        k = int(np.argmax(probs))
        return {"window": ws[k], "score": float(probs[k])}

    def calibrate(self, benign_val) -> Defender:
        scores = self.score(benign_val)
        self.threshold = float(np.quantile(scores, 1.0 - self.fpr))
        return self

    def detect(self, texts) -> np.ndarray:
        return self.score(texts) > self.threshold


LINEAR_KINDS = ("naive", "hardened", "windowed")
TRANSFORMER_KINDS = ("transformer", "transformer_windowed")


def make_defender(kind: str, fpr: float = 0.01, seed: int = 0):
    """Build any defender by name. Transformer kinds import torch lazily so it stays optional."""
    if kind in TRANSFORMER_KINDS:
        try:
            from .transformer import TransformerDefender
        except ImportError as exc:
            raise SystemExit(f"'{kind}' needs torch + transformers: pip install -r requirements-transformer.txt ({exc})") from exc
        return TransformerDefender(kind, fpr=fpr, seed=seed)
    return Defender(kind, fpr=fpr, seed=seed)
