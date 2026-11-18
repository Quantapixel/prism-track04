"""Output schema checks, mirroring TEAM_README section 3/4 (standard library only).

The organisers' validate.py was missing from the student package, so this
re-implements every rule the README lists. ``code/tools/validate.py`` is the CLI.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REQUIRED = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        with open(Path(__file__).with_name("supported_classes.json")) as fh:
            d = json.load(fh)
        _VOCAB = (set(d["labels"]), set(d["verbs"]), set(d["nouns"]))
    return _VOCAB


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _seg_ok(seg, dur, where, errors) -> bool:
    if not (isinstance(seg, list) and len(seg) == 2 and all(_num(v) for v in seg)):
        errors.append(f"{where}: segment must be [start, end] numbers, got {seg!r}")
        return False
    s, e = seg
    if not (0 <= s < e <= dur + 0.5):
        errors.append(f"{where}: need 0 <= start < end <= duration+0.5, got {seg}")
        return False
    return True


def validate_obj(obj, filename: str | None = None, manifest: dict | None = None,
                 strict_vocab: bool = True) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Any error => the file scores 0."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(obj, dict):
        return ["top level must be a JSON object"], warnings
    for k in REQUIRED:
        if k not in obj:
            errors.append(f"missing field {k!r}")
    if errors:
        return errors, warnings
    labels, verbs, nouns = _vocab()

    if not (isinstance(obj["team_id"], str) and obj["team_id"].strip()):
        errors.append("team_id must be a non-empty string")
    vid = obj["video_id"]
    if not (isinstance(vid, str) and vid.strip()):
        errors.append("video_id must be a non-empty string")
    elif filename is not None and Path(filename).stem != vid:
        errors.append(f"video_id {vid!r} does not match filename {Path(filename).name!r}")
    dur = obj["duration_sec"]
    if not (_num(dur) and dur > 0):
        errors.append("duration_sec must be a positive number")
        return errors, warnings
    if manifest is not None and isinstance(vid, str):
        if vid not in manifest:
            errors.append(f"video_id {vid!r} not in manifest")
        elif abs(dur - manifest[vid]) > 0.5:
            errors.append(f"duration_sec {dur} differs from manifest {manifest[vid]} by > 0.5 s")

    lab = obj["label"]
    if not (isinstance(lab, dict) and isinstance(lab.get("name"), str) and _num(lab.get("score"))):
        errors.append('label must be {"name": str, "score": number}')
    else:
        if not 0 <= lab["score"] <= 1:
            errors.append("label.score must be in [0, 1]")
        if lab["name"] not in labels:
            (errors if strict_vocab else warnings).append(f"label.name {lab['name']!r} not in supported labels")

    acts = obj["actions"]
    if not isinstance(acts, list):
        errors.append("actions must be an array")
    else:
        if len(acts) > 2000:
            errors.append(f"actions has {len(acts)} items (max 2000)")
        for i, a in enumerate(acts):
            w = f"actions[{i}]"
            if not isinstance(a, dict):
                errors.append(f"{w} must be an object")
                continue
            for k in ("segment", "verb", "noun", "score"):
                if k not in a:
                    errors.append(f"{w} missing {k!r}")
            if any(k not in a for k in ("segment", "verb", "noun", "score")):
                continue
            _seg_ok(a["segment"], dur, w, errors)
            if not isinstance(a["verb"], str) or not isinstance(a["noun"], str):
                errors.append(f"{w}: verb/noun must be strings")
            else:
                if a["verb"] not in verbs:
                    (errors if strict_vocab else warnings).append(f"{w}: verb {a['verb']!r} not in vocab")
                if a["noun"] not in nouns:
                    (errors if strict_vocab else warnings).append(f"{w}: noun {a['noun']!r} not in vocab")
            if not (_num(a["score"]) and 0 <= a["score"] <= 1):
                errors.append(f"{w}: score must be a number in [0, 1]")

    hl = obj["highlights"]
    if not isinstance(hl, list) or len(hl) == 0:
        errors.append("highlights must be a non-empty array")
    else:
        prev_end = -math.inf
        total = 0.0
        long_ok = False
        for i, h in enumerate(hl):
            w = f"highlights[{i}]"
            if not (isinstance(h, dict) and "segment" in h):
                errors.append(f"{w} must be {{\"segment\": [start, end]}}")
                continue
            if not _seg_ok(h["segment"], dur, w, errors):
                continue
            s, e = h["segment"]
            if s < prev_end:
                errors.append(f"{w}: highlights must be sorted and strictly non-overlapping")
            prev_end = max(prev_end, e)
            total += e - s
            if e - s >= 4.0:
                long_ok = True
            else:
                warnings.append(f"{w}: {e - s:.2f} s < 4.0 s (ignored when scoring)")
        if not long_ok:
            errors.append("highlights need at least one segment >= 4.0 s")
        if total > 0.15 * dur + 1e-9:
            warnings.append(f"highlights total {total:.2f} s exceeds 15 % budget ({0.15 * dur:.2f} s)")
    return errors, warnings


def validate_file(path: str | Path, manifest: dict | None = None, strict_vocab: bool = True):
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return [f"cannot parse JSON: {e}"], []
    return validate_obj(obj, filename=str(path), manifest=manifest, strict_vocab=strict_vocab)
