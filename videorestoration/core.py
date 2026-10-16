"""Forward-only video detail memory: no learned video model, no future-frame access.

All images are BGR uint8 at API boundaries, BGR float32 in [0,1] internally.
The oracle clean target is never passed to this module. A memory write occurs
ONLY on an explicitly declared sharp observation. Degraded frames update
tracking state, never the stored detail. No hallucinated output is reinserted.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


DEPTHS = (0, 1, 4, 16)   # 3 residue bands, plus core; avoids 64 diffusion steps


def _float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 BGR image")
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    arr = frame.astype(np.float32)
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError("Floating images must be finite and in [0,1]")
    return arr


def _uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame * 255), 0, 255).astype(np.uint8)


def heat_step(x: np.ndarray, dt: float = 0.20) -> np.ndarray:
    """Reflecting/replicated-edge 4-neighbor lattice diffusion; stable dt <= 1/4."""
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return x + np.float32(dt) * (p[:-2, 1:-1] + p[2:, 1:-1] +
                                  p[1:-1, :-2] + p[1:-1, 2:] - 4 * x)


def sihti_decompose(frame: np.ndarray, depths=DEPTHS) -> tuple[list[np.ndarray], np.ndarray]:
    """Lossless Sihti lattice decomposition: frame = sum(residues) + core.

    Not a learned representation. Bands and core have full spatial resolution.
    """
    x = _float(frame)
    if not depths or depths[0] != 0 or any(b <= a for a, b in zip(depths, depths[1:])):
        raise ValueError("depths must start at zero and be strictly increasing")
    states = [x]
    cur = x
    for a, b in zip(depths, depths[1:]):
        for _ in range(b - a):
            cur = heat_step(cur)
        states.append(cur)
    return [states[i] - states[i+1] for i in range(len(states)-1)], states[-1]


def sihti_reconstruct(bands: list[np.ndarray], core: np.ndarray) -> np.ndarray:
    return core + sum(bands, np.zeros_like(core))


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def grid(h: int, w: int) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return np.dstack((xx, yy))


def sample(x: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.remap(x, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_coordinates(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    return ((xy[..., 0] >= 1) & (xy[..., 0] <= width - 2) &
            (xy[..., 1] >= 1) & (xy[..., 1] <= height - 2))


def motion_to_previous(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Farneback flow at current pixel pointing to previous frame's pixel."""
    def gray(x):
        u = _uint8(x)
        return cv2.GaussianBlur(cv2.cvtColor(u, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0.8)
    return cv2.calcOpticalFlowFarneback(gray(curr), gray(prev), None,
                                        0.5, 3, 19, 3, 5, 1.2, 0)


def unsharp(frame: np.ndarray, strength: float = 0.9) -> np.ndarray:
    return np.clip(frame + strength * (frame - smooth(frame, 1.2)), 0, 1)


@dataclass
class FrameResult:
    images: dict[str, np.ndarray]
    sharp_observation: bool
    memory_valid_fraction: float
    cut_detected: bool
    memory_age: int


class ObservedDetailRestorer:
    """Tracks a single last-sharp anchor through *observed* frames.

    Arms:
      raw          -- degraded input (no restoration)
      independent  -- frame-only unsharp baseline
      flow_cache   -- warp one original anchor high-pass residue
      sihti_null   -- warp and add individually stored Sihti bands;
                      algebraically the SAME signal as flow_cache
      sihti_select -- candidate: coarse-to-fine band-dependent detail gating

    The null arm is intentionally included to catch false claims that changing
    residue coordinates creates information.
    """

    def __init__(self, strength: float = 0.90, scene_cut_threshold: float = 0.23,
                 disable_confidence: bool = False, use_oracle_flow: bool = False):
        if not 0 <= strength <= 2:
            raise ValueError("strength must be in [0,2]")
        self.strength = strength
        self.scene_cut_threshold = scene_cut_threshold
        self.disable_confidence = disable_confidence
        self.use_oracle_flow = use_oracle_flow
        self.reset()

    def reset(self):
        self.anchor = None
        self.anchor_bands = None
        self.anchor_core = None
        self.anchor_residue = None
        self.map_to_anchor = None
        self.confidence = None
        self.previous_observed = None
        self.age = 0
        self.shape = None

    @property
    def bytes_used(self) -> int:
        parts = [self.anchor, self.anchor_residue, self.anchor_core,
                 self.map_to_anchor, self.confidence, self.previous_observed]
        if self.anchor_bands is not None:
            parts.extend(self.anchor_bands)
        return sum(p.nbytes for p in parts if p is not None)

    def _store_sharp(self, x: np.ndarray):
        bands, core = sihti_decompose(x)
        self.anchor = x.copy()
        self.anchor_bands = bands
        self.anchor_core = core
        self.anchor_residue = sum(bands, np.zeros_like(x))
        self.map_to_anchor = grid(*x.shape[:2])
        self.confidence = np.ones(x.shape[:2], np.float32)
        self.age = 0

    def process(self, frame: np.ndarray, sharp: bool = False,
                override_flow: np.ndarray | None = None) -> FrameResult:
        """Streaming API. frame is the actually observed frame.

        override_flow (current -> previous, HxWx2) is *only* for labeled
        synthetic/oracle diagnostics, never an input to practical inference.
        """
        x = _float(frame)
        h, w = x.shape[:2]
        if self.shape is not None and (h, w) != self.shape:
            raise ValueError("Frames must have identical dimensions")
        self.shape = (h, w)
        # All arms must pass through already-sharp observations unchanged;
        # otherwise the frame-only temporal comparison is unfair at anchors.
        ind = x if sharp else unsharp(x, self.strength)
        outputs = {name: _uint8(z) for name, z in
                   [("raw", x), ("independent", ind)]}
        if sharp:
            self._store_sharp(x)
            self.previous_observed = x.copy()
            for name in ("flow_cache", "sihti_null", "sihti_select"):
                outputs[name] = _uint8(x)
            return FrameResult(outputs, True, 1.0, False, 0)

        # No sharp frame yet: prediction is disabled, not invented.
        if self.anchor is None or self.previous_observed is None:
            for name in ("flow_cache", "sihti_null", "sihti_select"):
                outputs[name] = _uint8(ind)
            self.previous_observed = x.copy()
            return FrameResult(outputs, False, 0.0, False, 0)

        previous = self.previous_observed
        # Conservative hard cut: if the scene jumps, invalidate old memory.
        cut_score = float(np.mean(np.abs(smooth(x, 4.0) -
                                         smooth(previous, 4.0))))
        if cut_score > self.scene_cut_threshold:
            self.anchor = None
            self.anchor_bands = None
            self.anchor_core = None
            self.anchor_residue = None
            self.map_to_anchor = None
            self.confidence = None
            self.previous_observed = x.copy()
            self.age = 0
            for name in ("flow_cache", "sihti_null", "sihti_select"):
                outputs[name] = _uint8(ind)
            return FrameResult(outputs, False, 0.0, True, 0)

        if override_flow is not None:
            flow = np.asarray(override_flow, dtype=np.float32)
            if flow.shape != (h, w, 2):
                raise ValueError("override_flow must be HxWx2")
        else:
            flow = motion_to_previous(x, previous)
        xy_prev = grid(h, w) + flow
        new_anchor_map = sample(self.map_to_anchor, xy_prev)
        old_conf = sample(self.confidence, xy_prev)
        good = valid_coordinates(xy_prev, w, h) & valid_coordinates(new_anchor_map, w, h)
        if self.disable_confidence:
            confidence = good.astype(np.float32)
        else:
            # Both tests compare observed images only. No clean hidden targets.
            warped_previous = sample(smooth(previous, 3.0), xy_prev)
            err_step = np.mean(np.abs(smooth(x, 3.0) - warped_previous), axis=2)
            warped_core = sample(self.anchor_core, new_anchor_map)
            err_anchor = np.mean(np.abs(smooth(x, 4.0) -
                                        smooth(warped_core, 2.0)), axis=2)
            confidence = (old_conf * np.exp(-err_step / 0.25) *
                          np.exp(-err_anchor / 0.22) * 0.995)
            if override_flow is None:
                opposite = motion_to_previous(previous, x)
                inverse_in_prev = sample(opposite, xy_prev)
                fb_error = np.linalg.norm(flow + inverse_in_prev, axis=2)
                confidence *= np.exp(-fb_error / 2.5)
            confidence *= good
            confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

        self.map_to_anchor = new_anchor_map
        self.confidence = confidence
        self.age += 1
        self.previous_observed = x.copy()

        # Strong simple comparator: warp the *same* actual observed high-pass.
        one_residue = sample(self.anchor_residue, new_anchor_map)
        flow_result = np.clip(x + self.strength * confidence[..., None] *
                              one_residue, 0, 1)
        outputs["flow_cache"] = _uint8(flow_result)

        # Null test: linearity implies sum of warped bands = warped sum.
        warped_bands = [sample(r, new_anchor_map) for r in self.anchor_bands]
        null_residue = sum(warped_bands, np.zeros_like(x))
        outputs["sihti_null"] = _uint8(np.clip(
            x + self.strength * confidence[..., None] * null_residue, 0, 1))

        # Candidate: address each scale according to observed current-band energy.
        # Gate never examines an unobserved GT pixel; confidence discourages ghosts.
        current_bands, _ = sihti_decompose(x)
        chosen = np.zeros_like(x)
        for ref_band, current_band in zip(warped_bands, current_bands):
            ref_energy = smooth(np.mean(np.abs(ref_band), axis=2), 1.5)
            current_energy = smooth(np.mean(np.abs(current_band), axis=2), 1.5)
            need = np.clip((ref_energy - current_energy) /
                           (ref_energy + 0.025), 0, 1)
            # Keep a little observed residue when the measured need is uncertain.
            chosen += ref_band * (0.35 + 0.65 * need[..., None])
        candidate = np.clip(x + self.strength * confidence[..., None] * chosen, 0, 1)
        outputs["sihti_select"] = _uint8(candidate)
        return FrameResult(outputs, False, float(np.mean(confidence > 0.25)),
                           False, self.age)
