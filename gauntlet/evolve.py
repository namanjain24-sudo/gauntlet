"""Evolutionary attacker and the attacker/defender arms race.

The attacker mutates injection payloads to lower the defender's score. A mutant counts
only if the victim oracle can still recover the malicious intent, so the search cannot
"win" by destroying the payload.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import victim_understands
from .defender import make_defender
from .transforms import MUTATORS

MAX_LEN = 1500


@dataclass
class Individual:
    text: str
    intent: str
    lineage: tuple[str, ...] = ()


@dataclass
class EvolutionResult:
    asr_curve: list[float]              # attack success rate per generation
    evasions: list[Individual] = field(default_factory=list)
    mutator_counts: Counter = field(default_factory=Counter)


def _mutate(ind: Individual, rng: random.Random) -> Individual | None:
    names = rng.sample(list(MUTATORS), rng.choice([1, 1, 2, 3]))
    text = ind.text
    for name in names:
        text = MUTATORS[name](text, rng)
    if text == ind.text or len(text) > MAX_LEN or not victim_understands(text, ind.intent):
        return None
    return Individual(text, ind.intent, ind.lineage + tuple(names))


def evolve(
    defender, seeds: list[Individual], rng: random.Random,
    generations: int = 10, pop_size: int = 300, elite_frac: float = 0.2, parent_frac: float = 0.4,
    on_generation: Callable[[int, float], None] | None = None,
) -> EvolutionResult:
    """Evolve ``seeds`` against ``defender``. ``on_generation(gen, asr)`` is called after each generation is scored."""
    pop = [rng.choice(seeds) for _ in range(pop_size)]
    result = EvolutionResult(asr_curve=[])
    found: dict[str, Individual] = {}

    for gen in range(generations + 1):
        scores = defender.score([i.text for i in pop])
        evades = scores <= defender.threshold          # every member is already victim-valid
        result.asr_curve.append(float(evades.mean()))
        if on_generation is not None:
            on_generation(gen, result.asr_curve[-1])
        for ind, ok in zip(pop, evades):
            if ok:
                found.setdefault(ind.text, ind)
        if gen == generations:
            break
        order = np.argsort(scores)                     # lowest score = best evader
        parents = [pop[k] for k in order[: max(2, int(parent_frac * pop_size))]]
        nxt = [pop[k] for k in order[: int(elite_frac * pop_size)]]
        tries = 0
        while len(nxt) < pop_size and tries < pop_size * 20:
            tries += 1
            child = _mutate(rng.choice(parents), rng)
            if child is not None:
                nxt.append(child)
        while len(nxt) < pop_size:                     # fall back to fresh seeds if mutation stalls
            nxt.append(rng.choice(seeds))
        pop = nxt

    result.evasions = list(found.values())
    for ind in result.evasions:
        result.mutator_counts.update(set(ind.lineage))
    return result


def _seeds(df: pd.DataFrame, intents: list[str]) -> list[Individual]:
    arena = df[(df.split == "arena") & df.intent.isin(intents)]
    return [Individual(t, i) for t, i in zip(arena.text, arena.intent)]


def arms_race(df: pd.DataFrame, kind: str, rounds: int = 5, generations: int = 10,
              pop_size: int = 300, fpr: float = 0.01, seed: int = 0, adv_cap: int = 400) -> dict:
    """Alternate: attacker evolves against the defender, defender retrains on what got through.

    Each round uses *fresh* arena intents, so ASR at round r is an out-of-sample measure of
    how well adversarial training on rounds < r generalises to unseen attacks.
    """
    rng = random.Random(seed)
    train = df[df.split == "train"]
    benign_val = df[(df.split == "val") & (df.label == 0)].text
    test_attacks = df[(df.split == "test") & (df.label == 1)].text
    test_benign = df[(df.split == "test") & (df.label == 0)].text
    arena_intents = sorted(df[df.split == "arena"].intent.unique())
    rng.shuffle(arena_intents)
    chunks = [arena_intents[k::rounds] for k in range(rounds)]

    texts, labels = list(train.text), list(train.label)
    history, gallery, mutator_totals = [], [], Counter()
    initial = None
    for r in range(rounds):
        defender = make_defender(kind, fpr=fpr, seed=seed).fit(texts, labels).calibrate(benign_val)
        initial = initial or defender
        res = evolve(defender, _seeds(df, chunks[r]), rng, generations=generations, pop_size=pop_size)
        history.append({
            "round": r, "asr_start": res.asr_curve[0], "asr_end": res.asr_curve[-1],
            "asr_peak": max(res.asr_curve), "curve": res.asr_curve, "n_evasions": len(res.evasions),
            "recall_test": float(defender.detect(test_attacks).mean()),
            "fpr_test": float(defender.detect(test_benign).mean()),
        })
        mutator_totals.update(res.mutator_counts)
        if r == 0:
            evolved = [e for e in res.evasions if e.lineage]
            gallery = [{"text": e.text[:240], "lineage": list(e.lineage)} for e in
                       sorted(evolved, key=lambda e: -len(e.lineage))[:6]]
        chosen = rng.sample(res.evasions, min(adv_cap, len(res.evasions)))
        texts += [e.text for e in chosen]
        labels += [1] * len(chosen)

    final = make_defender(kind, fpr=fpr, seed=seed).fit(texts, labels).calibrate(benign_val)
    return {"kind": kind, "history": history, "mutators": dict(mutator_totals.most_common()),
            "gallery": gallery, "initial_defender": initial, "final_defender": final}
