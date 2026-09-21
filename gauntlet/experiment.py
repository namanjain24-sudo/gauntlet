"""Full Gauntlet experiment: LOFO + arms race + real-data check, over several seeds."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import external
from .corpus import build_dataset
from .evaluate import lofo
from .evolve import arms_race

KINDS = ("naive", "hardened", "windowed")


def run(seeds: int = 3, rounds: int = 5, generations: int = 12, pop_size: int = 200,
        cache: Path = Path("data/external/deepset.json"), log=print) -> dict:
    ext = external.fetch(cache)
    lofo_rows, arms, ext_rows, gallery, mutators = [], [], [], {}, {k: Counter() for k in KINDS}
    for seed in range(seeds):
        df = build_dataset(seed)
        for kind in KINDS:
            t0 = time.time()
            r = lofo(df, kind, seed=seed)
            r["seed"] = seed
            lofo_rows.append(r)
            a = arms_race(df, kind, rounds=rounds, generations=generations, pop_size=pop_size, seed=seed)
            for h in a["history"]:
                arms.append({"defender": kind, "seed": seed, **h})
            mutators[kind].update(a["mutators"])
            if seed == 0:
                gallery[kind] = a["gallery"]
            if ext is not None:
                for stage in ("initial", "final"):
                    ext_rows.append({"defender": kind, "seed": seed, "stage": stage,
                                     **external.evaluate(a[f"{stage}_defender"], ext)})
            log(f"seed {seed} | {kind:<8} done in {time.time() - t0:4.0f}s")
    return {
        "config": {"seeds": seeds, "rounds": rounds, "generations": generations, "pop_size": pop_size,
                   "external_n": None if ext is None else int(len(ext))},
        "lofo": pd.concat(lofo_rows, ignore_index=True).to_dict("records"),
        "arms": arms, "external": ext_rows, "gallery": gallery,
        "mutators": {k: dict(v.most_common()) for k, v in mutators.items()},
    }


def save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, default=float))
