"""Fine-tuned transformer defenders (optional: needs ``pip install -r requirements-transformer.txt``).

Same interface as :class:`gauntlet.defender.Defender`, so every experiment runs unchanged.

* ``transformer``          - raw text -> fine-tuned MiniLM encoder -> mean-pool -> linear head
* ``transformer_windowed`` - canonicalised text (+ flag tokens), scored per sentence window, max over windows
                             (the same architecture tricks as the linear ``windowed`` defender)

The whole encoder is fine-tuned (not a frozen-embedding probe). To keep every fit comparable and
CPU-feasible, training runs a *fixed budget of optimiser steps* (default 160 x 32 examples) sampled
from whatever data it is given, so adversarial examples keep their proportional share.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .canonical import canonicalize, transform_flags
from .defender import Defender, windows

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KINDS = ("transformer", "transformer_windowed")
_CACHE: dict[str, tuple] = {}


def _backbone(name: str):
    """Load the pretrained weights once per process; every fit starts from a fresh copy."""
    if name not in _CACHE:
        _CACHE[name] = (AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name))
    tok, base = _CACHE[name]
    return tok, copy.deepcopy(base)


class _Classifier(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, 2)

    def forward(self, **enc) -> torch.Tensor:
        h = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


class TransformerDefender:
    def __init__(self, kind: str = "transformer", fpr: float = 0.01, seed: int = 0, steps: int = 160,
                 batch: int = 32, lr: float = 5e-5, max_len: int = 256, model_name: str = MODEL_NAME):
        if kind not in KINDS:
            raise ValueError(f"unknown transformer defender kind: {kind}")
        self.kind, self.fpr, self.seed = kind, fpr, seed
        self.steps, self.batch, self.lr, self.max_len, self.model_name = steps, batch, lr, max_len, model_name
        self.threshold = 0.5
        self.tok, self.model = None, None
        torch.set_num_threads(int(os.environ.get("GAUNTLET_THREADS", min(10, os.cpu_count() or 1))))

    def _prep(self, texts) -> list[str]:
        if self.kind == "transformer":
            return list(texts)
        return [Defender._harden(t) for t in texts]

    def _encode(self, texts: list[str]):
        return self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

    def fit(self, texts, labels) -> TransformerDefender:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.tok, encoder = _backbone(self.model_name)
        self.model = _Classifier(encoder)
        X, y = self._prep(texts), np.asarray(labels, dtype=int)
        counts = np.bincount(y, minlength=2).astype(float)
        weight = torch.tensor(len(y) / (2 * np.maximum(counts, 1)), dtype=torch.float32)   # 'balanced'
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / self.steps))          # warmup, linear decay
        self.model.train()
        order = rng.permutation(len(X))
        pos = 0
        for _ in range(self.steps):
            if pos + self.batch > len(order):
                order, pos = rng.permutation(len(X)), 0
            idx = order[pos:pos + self.batch]
            pos += self.batch
            logits = self.model(**self._encode([X[i] for i in idx]))
            loss = loss_fn(logits, torch.from_numpy(y[idx]))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        self.model.eval()
        return self

    @torch.no_grad()
    def _proba(self, texts: list[str], bs: int = 128) -> np.ndarray:
        """P(injection) per text; sorted by length so padding stays small."""
        out = np.zeros(len(texts))
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), bs):
            idx = order[s:s + bs]
            logits = self.model(**self._encode([texts[i] for i in idx]))
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].numpy()
        return out

    def score(self, texts) -> np.ndarray:
        """Probability that each text is an injection."""
        texts = list(texts)
        if self.kind == "transformer":
            return self._proba(texts)
        chunks, owner = [], []
        for k, t in enumerate(texts):
            flags = "".join(f" xflag{f}" for f in transform_flags(t))
            ws = windows(canonicalize(t))
            chunks += [w + flags for w in ws]
            owner += [k] * len(ws)
        out = np.zeros(len(texts))
        np.maximum.at(out, np.asarray(owner, dtype=int), self._proba(chunks))
        return out

    def calibrate(self, benign_val) -> TransformerDefender:
        scores = self.score(benign_val)
        self.threshold = float(np.quantile(scores, 1.0 - self.fpr))
        return self

    def detect(self, texts) -> np.ndarray:
        return self.score(texts) > self.threshold
