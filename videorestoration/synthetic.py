"""Deterministic procedural *test* video with hidden clean frames.

The generator supplies gt ONLY to benchmark scoring. Restoration receives
observed pixels and an explicit boolean indicating genuine sharp observations.
Camera motion is a crop of the same synthetic world; corruption is applied
AFTER frame construction. This is a debugging gate, not a real-video result.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SyntheticFrame:
    clean: np.ndarray
    observed: np.ndarray
    sharp: bool
    scene: int
    displacement: tuple[float, float]
    index: int


def make_world(height: int, width: int, seed: int, scene: int) -> np.ndarray:
    rng = np.random.default_rng(seed + scene * 7919)
    H, W = height + 100, width + 100
    # Low frequency visible structure provides a tractable optical-flow signal,
    # while crisp stripes/dots provide genuinely lost high-frequency detail.
    n = rng.normal(0, 1, (H, W, 3)).astype(np.float32)
    broad = cv2.GaussianBlur(n, (0, 0), 13)
    mid = cv2.GaussianBlur(n, (0, 0), 3)
    broad = (broad - broad.min()) / (broad.max() - broad.min() + 1e-6)
    mid = (mid - mid.min()) / (mid.max() - mid.min() + 1e-6)
    base = (0.15 + 0.5 * broad + 0.20 * mid)
    if scene % 2:
        base = np.clip(base * np.array([0.65, 0.7, 1.35], np.float32) + 0.08, 0, 1)
    else:
        base = np.clip(base * np.array([1.25, 1.0, 0.65], np.float32), 0, 1)
    world = (base * 255).astype(np.uint8)
    for _ in range(90):
        x = int(rng.integers(W))
        y = int(rng.integers(H))
        color = tuple(int(x) for x in rng.integers(15, 255, 3))
        radius = int(rng.integers(3, 21))
        kind = int(rng.integers(3))
        if kind == 0:
            cv2.circle(world, (x, y), radius, color, 1, cv2.LINE_AA)
        elif kind == 1:
            cv2.line(world, (x, y),
                     (int(np.clip(x + rng.integers(-35,36), 0, W-1)),
                      int(np.clip(y + rng.integers(-35,36), 0, H-1))),
                     color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(world, (x,y),
                          (min(W-1,x+radius),min(H-1,y+radius)),color,-1)
    # Real repeating fine texture on a smooth background, not white sensor noise.
    yy, xx = np.mgrid[:H,:W]
    stripe = ((np.sin((xx + 0.43 * yy) * 1.95 + scene) > 0) & 
              (np.sin((xx - yy) * 0.22) > 0))
    world[stripe] = np.clip(world[stripe].astype(np.int16) + 45,0,255).astype(np.uint8)
    return world


def make_sequence(seed: int, frames: int = 28, height: int = 128,
                  width: int = 160, anchor_every: int = 7,
                  scene_at: int | None = 17, blur: float = 1.4,
                  noise: float = 2.0) -> list[SyntheticFrame]:
    if frames < 2 or height < 32 or width < 32 or anchor_every < 2:
        raise ValueError("Need >=2 frames, >=32x32 images and >=2 anchor period")
    rng = np.random.default_rng(seed + 111)
    worlds = [make_world(height,width,seed,0),
              make_world(height,width,seed,1)]
    seq = []
    for i in range(frames):
        scene = int(scene_at is not None and i >= scene_at)
        elapsed = i if scene == 0 else i - scene_at
        dx, dy = 0.66 * elapsed, 0.35 * elapsed
        yy, xx = np.mgrid[:height,:width].astype(np.float32)
        mx, my = xx + 38 + dx, yy + 40 + dy
        clean = cv2.remap(worlds[scene], mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        # New moving object deliberately breaks static-memory assumptions.
        if i % 13 in (8, 9, 10):
            x0 = 34 + (i * 5) % max(35,width-70)
            y0 = 30 + (i * 3) % max(35,height-60)
            cv2.rectangle(clean,(x0,y0),(x0+24,y0+18),
                          (50, 235, 70) if scene else (230, 60, 50),-1)
        sharp = i % anchor_every == 0
        if sharp:
            observed = clean.copy()
        else:
            observed = cv2.GaussianBlur(clean,(0,0),blur)
            if noise:
                observed = np.clip(observed.astype(np.float32) +
                                   rng.normal(0,noise,observed.shape),0,255).astype(np.uint8)
        seq.append(SyntheticFrame(clean,observed,sharp,scene,(dx,dy),i))
    return seq
