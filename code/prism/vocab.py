"""Official vocabulary (supported_classes.json) and the Omnivore EPIC action list."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUPPORTED = HERE / "supported_classes.json"
ACTIONS_CSV = HERE / "epic_action_classes.csv"

FPV_LABELS = [
    "bike", "bikepolo", "boxing", "horse", "jumping", "longboarding", "motor",
    "parkour", "plane", "rockclimbing", "scuba", "skate", "ski", "surfing",
]


@lru_cache(maxsize=1)
def supported() -> dict:
    with open(SUPPORTED) as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def verbs() -> list[str]:
    return supported()["verbs"]  # EPIC verb class index i -> verbs()[i]; last is "other"


@lru_cache(maxsize=1)
def nouns() -> list[str]:
    return supported()["nouns"]


@lru_cache(maxsize=1)
def labels() -> list[str]:
    return supported()["labels"]


@lru_cache(maxsize=1)
def actions() -> list[tuple[str, str]]:
    """Omnivore head index -> (verb, noun). 3806 EPIC-100 action classes."""
    with open(ACTIONS_CSV) as fh:
        return [(r[0], r[1]) for r in csv.reader(fh)]


@lru_cache(maxsize=1)
def action_verb_noun_idx() -> tuple[np.ndarray, np.ndarray]:
    vi = {v: i for i, v in enumerate(verbs())}
    ni = {n: i for i, n in enumerate(nouns())}
    a = actions()
    return (np.array([vi[v] for v, _ in a], dtype=np.int64),
            np.array([ni[n] for _, n in a], dtype=np.int64))


@lru_cache(maxsize=1)
def pair_to_action() -> dict[tuple[int, int], int]:
    v, n = action_verb_noun_idx()
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(v, n))}


N_VERBS_EPIC = 97  # without "other"
N_NOUNS_EPIC = 300
