"""A deliberately simple detector to try `gauntlet bench` with: counts injection-sounding phrases.

    python -m gauntlet bench --defender examples/keyword_detector.py:score
"""
import re

PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above|earlier)", r"disregard", r"system prompt",
    r"reveal|leak|exfiltrate", r"you are now", r"developer mode", r"no (rules|restrictions)",
    r"forget (what|everything)", r"override", r"jailbreak",
]


def score(texts):
    """One float per text, higher = more suspicious."""
    return [min(1.0, sum(bool(re.search(p, t, re.I)) for p in PATTERNS) / 3) for t in texts]
