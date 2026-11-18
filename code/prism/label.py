"""Video-level label by zero-shot CLIP over 1 fps frames.

Only pixels are used: the classifier never sees the file name (FPVSum file names
contain the category, and using them would count as hardcoding).
"""
from __future__ import annotations

import numpy as np

from .config import Config

TEMPLATES = [
    "a first-person POV video of {}.",
    "a head-mounted GoPro camera view of {}.",
    "egocentric footage of {}.",
    "a photo of {}.",
]

# Several descriptions per label; the label embedding is the mean over (description x template).
DESCRIPTIONS: dict[str, list[str]] = {
    "cooking": ["cooking in a kitchen", "preparing food on a kitchen counter",
                "washing dishes in a kitchen sink", "chopping vegetables on a cutting board",
                "opening a fridge in a kitchen", "stirring a pan on a stove"],
    "bike": ["riding a mountain bike on a trail", "cycling on a road bike",
             "downhill mountain biking through a forest", "riding a bicycle with handlebars in view"],
    "bikepolo": ["playing bike polo", "playing hardcourt bike polo with a mallet and ball",
                 "a bicycle polo match on a court"],
    "boxing": ["boxing sparring in a ring", "boxing training hitting focus pads",
               "punching an opponent wearing boxing gloves"],
    "horse": ["horseback riding", "riding a horse with its head and mane in view",
              "riding a horse along a trail"],
    "jumping": ["skydiving in freefall", "parachuting under a canopy", "bungee jumping",
                "base jumping off a cliff", "cliff jumping into water"],
    "longboarding": ["longboarding downhill on a road", "riding a longboard skateboard down a hill"],
    "motor": ["riding a motorcycle", "a motorcycle ride on a road", "riding a dirt bike motocross"],
    "parkour": ["doing parkour on rooftops", "freerunning and jumping over walls in a city"],
    "plane": ["flying an airplane from the cockpit", "piloting a small plane",
              "the view from an aircraft cockpit"],
    "rockclimbing": ["rock climbing a cliff", "climbing a rock wall with ropes", "bouldering on rocks"],
    "scuba": ["scuba diving underwater", "diving near a coral reef with fish"],
    "skate": ["skateboarding in a skatepark", "doing skateboard tricks on a ramp"],
    "ski": ["skiing down a snowy slope", "downhill skiing on a piste", "skiing in fresh powder snow"],
    "surfing": ["surfing a wave", "riding a surfboard in the ocean"],
    "other": ["walking around a city street", "driving a car", "sitting at a desk using a computer",
              "a person doing everyday activities at home"],
}

# Prompts used for FPV importance (excitement) scoring, per label + generic.
HIGHLIGHT_POS = {
    "_generic": ["an exciting action moment", "a thrilling moment of extreme sport",
                 "a spectacular moment worth watching", "an impressive trick"],
    "bike": ["riding fast down a steep mountain bike trail", "a mountain bike jump"],
    "bikepolo": ["hitting the ball with a mallet in bike polo", "scoring a goal in bike polo"],
    "boxing": ["throwing a punch at an opponent", "landing a punch in a boxing fight"],
    "horse": ["galloping fast on a horse", "jumping a fence on a horse"],
    "jumping": ["freefalling through the sky", "jumping off into the air", "the parachute opening"],
    "longboarding": ["carving fast downhill on a longboard", "sliding a longboard at high speed"],
    "motor": ["riding a motorcycle fast around a curve", "overtaking on a motorcycle"],
    "parkour": ["jumping between rooftops", "vaulting over a wall"],
    "plane": ["taking off from the runway", "landing an airplane on a runway", "a steep banking turn"],
    "rockclimbing": ["reaching the top of a climb", "a difficult climbing move high above the ground"],
    "scuba": ["swimming next to a big fish or turtle underwater", "a colourful coral reef"],
    "skate": ["landing a skateboard trick", "skateboarding down a ramp"],
    "ski": ["jumping off a ski jump", "skiing fast through deep powder"],
    "surfing": ["riding inside a breaking wave", "catching a big wave on a surfboard"],
}
HIGHLIGHT_NEG = ["a boring moment where nothing happens", "standing still and waiting",
                 "a blurry dark screen", "the camera pointing at the ground",
                 "adjusting the camera", "a static uneventful view"]


class LabelClassifier:
    def __init__(self, cfg: Config, clip):
        self.cfg, self.clip = cfg, clip
        self.names = list(DESCRIPTIONS)
        embs = []
        for name in self.names:
            texts = [t.format(d) for d in DESCRIPTIONS[name] for t in TEMPLATES]
            e = clip.encode_texts(texts).mean(0)
            embs.append(e / np.linalg.norm(e))
        self.text = np.stack(embs)  # L, D
        self.bias = np.array([cfg.other_logit_bias if n == "other" else 0.0 for n in self.names])

    def predict(self, clip_emb: np.ndarray) -> tuple[str, float, dict[str, float]]:
        if clip_emb.size == 0 or clip_emb.shape[0] == 0:
            return "other", 0.0, {}
        e = clip_emb.astype(np.float32)
        logits = self.cfg.label_temperature * e @ self.text.T + self.bias  # F, L
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        probs = p.mean(0)  # average per-frame probabilities: robust to a few odd frames
        i = int(probs.argmax())
        return self.names[i], float(np.clip(probs[i], 0.0, 1.0)), dict(zip(self.names, probs.tolist()))

    def highlight_text(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        pos = HIGHLIGHT_POS["_generic"] + HIGHLIGHT_POS.get(label, [])
        pos_e = self.clip.encode_texts([t.format(p) for p in pos for t in TEMPLATES[:2]])
        neg_e = self.clip.encode_texts([t.format(p) for p in HIGHLIGHT_NEG for t in TEMPLATES[:2]])
        return pos_e, neg_e
