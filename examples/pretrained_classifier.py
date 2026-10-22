"""Wrap a real pretrained prompt-injection classifier for `gauntlet bench` (needs torch + transformers).

    python -m gauntlet bench --defender examples/pretrained_classifier.py:score --baselines

Uses protectai/deberta-v3-base-prompt-injection-v2 (184M params), a DeBERTa-v3-base fine-tuned
specifically for this task and independent of Gauntlet's synthetic training data -- unlike the
built-in defenders, it never saw any of Gauntlet's attack families during its own training, so
its numbers are a genuine outside opinion, not just an out-of-fold one.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
_inj_idx = [k for k, v in _model.config.id2label.items() if v.upper() == "INJECTION"][0]


@torch.no_grad()
def score(texts, batch_size: int = 16) -> list[float]:
    """P(injection) per text, batched for CPU throughput."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
        probs = torch.softmax(_model(**enc).logits, dim=-1)
        out += probs[:, _inj_idx].tolist()
    return out
