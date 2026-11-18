#!/usr/bin/env python3
"""Validate output JSONs against the TEAM_README rules (standard library only).

  python code/tools/validate.py --file outputs/P01_11.json -v
  python code/tools/validate.py --dir outputs/            # also checks all 187 are present
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from prism.schema import validate_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--manifest", default=str(ROOT / "package" / "manifest.csv"))
    ap.add_argument("-v", "--verbose", action="store_true", help="print warnings")
    a = ap.parse_args()

    manifest = None
    if a.manifest and Path(a.manifest).exists():
        with open(a.manifest) as fh:
            manifest = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(fh)}
    files = [Path(a.file)] if a.file else sorted(Path(a.dir).glob("*.json"))
    bad, teams = 0, set()
    for f in files:
        errors, warnings = validate_file(f, manifest=manifest)
        if a.file and manifest and f.stem not in manifest:  # allow the package examples
            errors = [e for e in errors if "not in manifest" not in e]
        try:
            teams.add(json.loads(f.read_text()).get("team_id"))
        except Exception:  # noqa: BLE001
            pass
        if errors:
            bad += 1
            print(f"FAIL {f.name}")
            for e in errors:
                print(f"   x {e}")
        elif a.verbose or a.file:
            print(f"ok   {f.name}")
        if a.verbose:
            for w in warnings:
                print(f"   ! {w}")
    if a.dir:
        if len(teams) > 1:
            bad += 1
            print(f"FAIL team_id differs across files: {sorted(map(str, teams))}")
        if manifest:
            missing = sorted(set(manifest) - {f.stem for f in files})
            if missing:
                print(f"FAIL {len(missing)} manifest videos have no JSON, e.g. {missing[:5]}")
                bad += 1
    print(f"{len(files)} file(s) checked, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
