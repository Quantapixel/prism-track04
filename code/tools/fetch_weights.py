#!/usr/bin/env python3
"""Download the frozen backbones into weights/ (run once, needs internet).

  * weights/omnivore_swinB_epic.torch  (~466 MB, Meta, CC BY-NC 4.0, EPIC-100 train fine-tune)
  * weights/clip-vit-large-patch14/    (~1.7 GB, OpenAI CLIP via Hugging Face)

After this, inference is fully offline. Use --clip openai/clip-vit-base-patch16 for a
smaller/faster CLIP (then set "clip_model": "clip-vit-base-patch16" in your config).
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# the Xet transfer backend hung at 0 bytes in testing; plain HTTPS is reliable
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(ROOT / "code"))

OMNI_URL = "https://dl.fbaipublicfiles.com/omnivore/models/swinB_epic_checkpoint.torch"
OMNI_BYTES = 488373503


def download(url: str, dst: Path, expect: int | None = None) -> None:
    if dst.exists() and (expect is None or dst.stat().st_size == expect):
        print(f"ok   {dst.name} already present")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"get  {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0)) or expect or 0
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r     {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if expect and tmp.stat().st_size != expect:
        raise RuntimeError(f"size mismatch for {dst.name}: {tmp.stat().st_size} != {expect}")
    tmp.replace(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14", help="HF model id")
    ap.add_argument("--skip-omnivore", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    a = ap.parse_args()
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    if not a.skip_omnivore:
        download(OMNI_URL, wdir / "omnivore_swinB_epic.torch", OMNI_BYTES)
    if not a.skip_clip:
        from transformers import CLIPModel, CLIPTokenizer

        out = wdir / a.clip.split("/")[-1]
        if (out / "config.json").exists():
            print(f"ok   {out.name} already present")
        else:
            print(f"get  {a.clip} -> {out}")
            CLIPModel.from_pretrained(a.clip).save_pretrained(out)
            CLIPTokenizer.from_pretrained(a.clip).save_pretrained(out)
    print("done")


if __name__ == "__main__":
    main()
