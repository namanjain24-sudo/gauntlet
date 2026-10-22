"""On-disk cache of trained linear defenders, so ``scan`` / ``attack`` / ``serve`` start in under a second.

Training the three linear defenders on the synthetic corpus takes ~15 s. The result depends only on
(kind, seed, fpr), the corpus/transform/defender source code and the scikit-learn version, so the cache
key is a hash of exactly those. Any code change or library upgrade silently invalidates old entries.

Location: ``$GAUNTLET_CACHE`` or ``~/.cache/gauntlet``. Set ``GAUNTLET_CACHE=0`` to disable.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import sklearn

from . import canonical, corpus, defender, families, transforms

_SOURCES = (transforms, families, corpus, canonical, defender)


def enabled() -> bool:
    return os.environ.get("GAUNTLET_CACHE", "") != "0"


def cache_dir() -> Path:
    env = os.environ.get("GAUNTLET_CACHE", "")
    return Path(env) if env and env != "0" else Path.home() / ".cache" / "gauntlet"


def fingerprint() -> str:
    """Hash of everything a trained linear defender depends on."""
    h = hashlib.sha256(sklearn.__version__.encode())
    for m in _SOURCES:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()[:16]


def path_for(kind: str, seed: int, fpr: float) -> Path:
    return cache_dir() / f"{kind}-seed{seed}-fpr{fpr:g}-{fingerprint()}.pkl"


def load(kind: str, seed: int, fpr: float):
    """A cached defender, or None if missing/unreadable (callers then retrain)."""
    if not enabled():
        return None
    p = path_for(kind, seed, fpr)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001 - a corrupt or incompatible pickle just means "retrain"
        return None


def save(d, kind: str, seed: int, fpr: float) -> Path | None:
    if not enabled():
        return None
    p = path_for(kind, seed, fpr)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)                                     # atomic: readers never see a half-written file
    return p


def clear() -> int:
    """Delete every cache entry (any fingerprint). Returns the number removed."""
    d = cache_dir()
    if not d.is_dir():
        return 0
    n = 0
    for p in d.glob("*.pkl"):
        p.unlink()
        n += 1
    return n
