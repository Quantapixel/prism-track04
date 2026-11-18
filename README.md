# PRISM 2026 · Track 04: Action-Aware Video Summarization

One pipeline for both datasets. `./run.sh <video.mp4> <out.json>` writes the official JSON
(label, verb-noun action detections, highlights) for one egocentric video.

```
video ──ffmpeg (15 fps, 224² centre crop, one decode pass)──┬─► Omnivore Swin-B (EPIC-100 fine-tuned)
                                                            │     32-frame windows, 0.5 s stride
                                                            │     → 1024-d features + 3806 action logits
                                                            ├─► CLIP image embeddings @ 1 fps
                                                            └─► frame-difference motion @ 15 fps
label    : zero-shot CLIP over all frames (16 labels; never looks at the file name)
cooking  → actions   : learned temporal head (weights/head.pt) or the training-free detector
           highlights: budgeted greedy that maximises the expected number of DISTINCT
                       (verb, noun) pairs covered (the 40 % Action Coverage metric)
other    → actions   : []
           highlights: CLIP "exciting vs boring" prompts + motion + novelty, budget filled
                       with the highest-importance time (the 25 % Highlight F1 metric)
```

Every output respects the rules by construction: sorted, strictly non-overlapping highlights
of at least 4.02 s (so still >= 4.0 s after rounding), a total of at most 15 % of the duration
(start times rounded up and end times rounded down, so rounding never pushes the total over),
at most 2000 actions, and only vocabulary from `supported_classes.json`. For clips shorter
than 26.8 s, 15 % is under 4 s, so one 4.02 s segment is emitted and the soft penalty applies.

## Layout

| path | what |
|---|---|
| `run.sh` | official single-video entry point |
| `code/prism/` | the system: `video_io`, `backbones`, `features`, `label`, `actions` (training-free detector), `head` (learned head), `highlights`, `schema`, `pipeline` |
| `code/prism/third_party/omnivore_swin.py` | vendored Swin3D from Meta's Omnivore (CC BY-NC 4.0), so nothing is fetched at run time |
| `code/tools/` | downloads, batch runner, validator, local scorer, training |
| `configs/submission.json` | overrides for the submission (at least `team_id`) |
| `package/` | the organisers' student package (manifest, vocabulary, examples) |
| `tests/` | `python -m pytest -q tests/`: 19 tests, no weights or data needed |

## 1. Set up (RTX 3060 PC)

```bash
git clone <this repo> prism-track04 && cd prism-track04
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt            # ffmpeg >= 4 must be on PATH (sudo apt install ffmpeg)
python code/tools/fetch_weights.py         # Omnivore (466 MB) + CLIP ViT-L/14 (1.7 GB) -> weights/
bash   code/tools/fetch_annotations.sh     # EPIC + FPVSum annotation files -> data/ (git-ignored)
python -m pytest -q tests/                 # sanity check
```

Set your team id once: edit `configs/submission.json` → `{"team_id": "T07"}` (or `export PRISM_TEAM_ID=T07`).

## 2. Get the test videos

```bash
bash   code/tools/download_epic.sh         # 138 EPIC val videos, ~184 GB, several hours, resumable
python code/tools/download_fpvsum.py       # 49 FPVSum videos from YouTube (<=720p, a few GB)
python code/tools/check_videos.py          # all present and within ±0.5 s of the manifest?
```

`OUT=/mnt/bigdisk bash code/tools/download_epic.sh` puts EPIC elsewhere. If so, pass
`--videos /mnt/bigdisk data/fpvsum` to `run_all.py`. If YouTube asks you to sign in, use
`download_fpvsum.py --cookies-from-browser firefox`.

## 3. Produce the 187 outputs

```bash
python code/tools/run_all.py --videos data --workers 2   # 2 processes share the GPU; features cached
python code/tools/validate.py --dir outputs/             # must print "0 problem(s)"
python code/tools/evaluate.py --pred outputs/            # local estimate of all four metrics
```

`run_all.py` uses the same `Pipeline` as `run.sh` but loads the models only once per worker.
It is resumable and processes the longest videos first.
After you change thresholds or train a head, `run_all.py --from-cache --overwrite` rebuilds every
JSON in a few minutes without decoding again.

### Speed and keeping the PC usable

| knob | default | effect |
|---|---|---|
| `--workers N` (run_all) | 1 | N processes sharing the GPU. Decode on one overlaps inference on another. Use 2 on a 12 GB card. |
| `hwaccel` (config) | `auto` | ffmpeg decodes on the GPU (NVDEC) when the build supports it, otherwise falls back to CPU per file. |
| `batch_size` (config) | `0` = auto | Omnivore windows per pass, sized from VRAM ÷ workers (~1.2 GB per window in fp16). CUDA out-of-memory halves the batch and retries. |
| `JOBS` (download_epic.sh) | 4 | parallel EPIC downloads |
| `--jobs` (download_fpvsum.py) | 4 | parallel YouTube downloads |
| `--download-jobs` (build_train_features.py) | 3 | parallel train downloads; `--prefetch` caps how many videos sit on disk |
| `code/tools/gentle.sh <cmd>` | | runs any command at lowest CPU/IO priority with a RAM cap (`MEM=5G`) and a thread cap (`THREADS=4`). Only that command can be OOM-killed, never your browser. |

Memory facts (measured): Omnivore needs ~2.6 GB of RAM for one window in fp32 on CPU, plus ~1.45 GB
for each extra window in a batch. CLIP-L adds ~1.7 GB. On CPU, always run through `gentle.sh`.

**Do not tune on `evaluate.py`.** The EPIC validation split and the FPVSum annotations are the
hidden test set. Use it as a sanity check only, and tune on held-out *train* videos
(`dev_eval.py`, below).

## 4. Train the action head (optional, recommended)

Without `weights/head.pt`, detection is training-free: Omnivore window scores are grouped over
time. The learned head (ActionFormer-style transformer + FPN, verb/noun focal loss, GIoU
boundary regression) learns action boundaries from EPIC-100 **train** videos:

```bash
python code/tools/build_train_features.py --hours 25  # download -> features -> delete, one video at a time
python code/tools/train_head.py                       # writes weights/head.pt only if it beats the baseline on dev
python code/tools/dev_eval.py                         # training-free vs head on held-out train videos
python code/tools/run_all.py --from-cache --overwrite # rebuild outputs with the head
```

* Disk: only about 2 videos at a time (~2-6 GB) plus ~15 MB of features per video. RAM: fine at 16 GB.
* Time: download speed sets the pace (EPIC train is ~14 GB per hour of video). `--hours 75` uses the whole train split.
* `train_head.py` refuses any manifest (test) video, and reports dev mAP against the training-free detector.
  If the head does not win, delete `weights/head.pt`.

## 5. Package the submission

```bash
code/tools/package_submission.sh T07        # -> submission/T07/{run.sh,code,weights,requirements.txt,outputs,README.md}
```

The script checks that the outputs validate and refuses to include ground-truth files.

## Integrity notes

* The label comes from pixels only. FPVSum file names contain the category, so the classifier never sees them.
* No video ids, ground truth or manifest durations are used at inference. `duration_sec` comes from `ffprobe`.
* Annotation files live in git-ignored `data/` and are only used by the training and scoring tools.
* The Omnivore checkpoint was fine-tuned on the EPIC-100 train split (not validation), so it does not leak the test set.

## Known gaps in the organisers' package

* `validate.py` and `examples/` were missing. `code/tools/validate.py` re-implements every rule in the README.
* `examples/skiing_03.json` uses the label `"skiing"`, but the vocabulary says `"ski"`. This repo follows the vocabulary.
* The README says the videos are 2-10 min long, but EPIC ones range from 11 s to 33 min.
* The exact scorer is not public. "Captured" in Action Coverage is ambiguous, so `evaluate.py`
  reports several readings and the selector optimises "midpoint inside OR at least 50 % overlap".
