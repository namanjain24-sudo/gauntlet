"""A second victim oracle, using a small pretrained NLI model instead of keyword recovery.

``canonical.victim_understands`` (the oracle used everywhere else in Gauntlet) asks a cheap
question: after canonicalising, are most of the intent's content words still present? That is
fast enough to run millions of times in the evolutionary attacker's inner loop, but it is a proxy
-- it can accept text a real reader would find garbled, and reject a paraphrase that has none of
the original words.

``ModelJudge`` asks a different, more expensive question with a small (66M parameter) NLI model:
does the canonicalised text *entail* the claim "this message asks the reader to {intent}"? It is
still not a full LLM, but it does not depend on shared vocabulary. :func:`compare` runs both
oracles over a sample of real attacks and evolved evasions and reports where they disagree, which
is the honest way to describe how much the choice of oracle drives the arena's numbers.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .canonical import canonicalize, victim_understands
from .evolve import _seeds, evolve
from .families import FAMILIES

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"


class ModelJudge:
    def __init__(self, model_name: str = MODEL_NAME, threshold: float = 0.5):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.threshold = threshold
        self._entail_idx = [k for k, v in self.model.config.id2label.items() if v.upper() == "ENTAILMENT"][0]

    @torch.no_grad()
    def entailment(self, texts: list[str], intents: list[str], batch_size: int = 16) -> np.ndarray:
        premises = [canonicalize(t) for t in texts]
        hyps = [f"This message asks the reader to {i}." for i in intents]
        out = np.zeros(len(texts))
        for s in range(0, len(texts), batch_size):
            enc = self.tok(premises[s:s + batch_size], hyps[s:s + batch_size],
                           padding=True, truncation=True, max_length=256, return_tensors="pt")
            out[s:s + batch_size] = torch.softmax(self.model(**enc).logits, dim=-1)[:, self._entail_idx].numpy()
        return out

    def understands(self, texts: list[str], intents: list[str]) -> np.ndarray:
        return self.entailment(texts, intents) > self.threshold


def sample_attacks(df, n: int = 150, seed: int = 0) -> tuple[list[str], list[str]]:
    """``n`` plain family-rendered attacks, evenly spread over families, from the test split."""
    rng = random.Random(seed)
    test = df[(df.split == "test") & (df.label == 1)]
    per_family = max(1, n // test.family.nunique())
    rows = test.groupby("family", group_keys=False).apply(lambda g: g.sample(min(per_family, len(g)), random_state=seed))
    idx = list(rows.index)
    rng.shuffle(idx)
    rows = rows.loc[idx[:n]]
    return list(rows.text), list(rows.intent)


def sample_evasions(df, defender, n: int = 60, generations: int = 6, pop_size: int = 100, seed: int = 0):
    """Evolve evasions against ``defender`` (using the keyword oracle, as the real arena does)."""
    rng = random.Random(seed)
    intents = sorted(df[df.split == "arena"].intent.unique())
    rng.shuffle(intents)
    res = evolve(defender, _seeds(df, intents[:40]), rng, generations=generations, pop_size=pop_size)
    evolved = [e for e in res.evasions if e.lineage]
    rng.shuffle(evolved)
    picked = evolved[:n]
    return [e.text for e in picked], [e.intent for e in picked]


def compare(texts: list[str], intents: list[str], judge: ModelJudge) -> dict:
    keyword = np.array([victim_understands(t, i) for t, i in zip(texts, intents)])
    model = judge.understands(texts, intents)
    agree = keyword == model
    disagreements = [{"text": " ".join(t.split())[:200], "intent": i, "keyword_oracle": bool(k), "model_oracle": bool(m)}
                     for t, i, k, m in zip(texts, intents, keyword, model) if k != m]
    return {
        "n": len(texts), "agreement_rate": float(agree.mean()),
        "keyword_accepts": float(keyword.mean()), "model_accepts": float(model.mean()),
        "keyword_only": int(((keyword) & (~model)).sum()), "model_only": int(((~keyword) & (model)).sum()),
        "examples": disagreements[:8],
    }


def render(plain: dict, evasions: dict) -> str:
    def block(name, r):
        lines = [f"{name}: {r['n']} texts, oracles agree {r['agreement_rate']:.0%} of the time "
                 f"(keyword accepts {r['keyword_accepts']:.0%}, model accepts {r['model_accepts']:.0%})",
                 f"  keyword says readable but model disagrees: {r['keyword_only']}   "
                 f"model says readable but keyword disagrees: {r['model_only']}"]
        for e in r["examples"][:4]:
            lines.append(f"  [keyword={e['keyword_oracle']} model={e['model_oracle']}] intent: {e['intent']}\n"
                        f"    {e['text']}")
        return "\n".join(lines)
    return block("plain attacks", plain) + "\n\n" + block("evolved evasions", evasions)
