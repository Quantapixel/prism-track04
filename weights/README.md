Model weights (not in git; ~0.5-2 GB). Fetch with:

    python code/tools/fetch_weights.py

| file | source | licence |
|---|---|---|
| `omnivore_swinB_epic.torch` | Meta Omnivore Swin-B fine-tuned on EPIC-KITCHENS-100 train | CC BY-NC 4.0 |
| `clip-vit-large-patch14/` | OpenAI CLIP (Hugging Face) | MIT |
| `head.pt` (optional) | trained by `code/tools/train_head.py` on EPIC-100 train only | ours |

For the final submission, copy this whole folder into `TEAM_ID/weights/`.
