"""The same keyword list as `keyword_detector.py`, but run over the canonical view of the text.

    python -m gauntlet bench --defender examples/canonical_keyword_detector.py:score --baselines

An ablation, not a serious detector. `keyword_detector.py` is a hand-written regex list applied to
raw text and it is the floor of the leaderboard. This file changes exactly one thing: the same
regexes run against `canonical.canonicalize(text)`, which undoes leetspeak, decodes base64, strips
invisible characters and normalises homoglyphs first.

Holding the detector constant, that one change moves mean recall from 8.1% to 10.5%, and the
movement is entirely where you would expect it: obfuscation 3% -> 18% and encoding 0% -> 5%, with
all six other families unchanged to the point. So canonicalisation does exactly the job claimed for
it -- and it still cannot rescue a detector that is bad for unrelated reasons. Both rows are evaded
100% of the time by the evolutionary attacker and both score AUC 0.56 on real injections.

The useful reading: canonicalisation is a prerequisite, not a defense. Compare with the `hardened`
reference row, which is `naive` plus the same canonicalisation step and gains far more from it,
because there is a trained model underneath to take advantage of the cleaner input.
"""
import re

from gauntlet.canonical import canonicalize

PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above|earlier)", r"disregard", r"system prompt",
    r"reveal|leak|exfiltrate", r"you are now", r"developer mode", r"no (rules|restrictions)",
    r"forget (what|everything)", r"override", r"jailbreak",
]


def score(texts):
    """One float per text, higher = more suspicious."""
    return [min(1.0, sum(bool(re.search(p, canonicalize(t), re.I)) for p in PATTERNS) / 3)
            for t in texts]
