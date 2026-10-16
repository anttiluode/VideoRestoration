"""Gate 2: selective single-mode streaming inference and falsifiable diagnostics.

Gate 1's ObservedDetailRestorer is kept unchanged for the frozen benchmark.
The selected mode here reuses the SAME motion/confidence equations; it avoids
constructing *unrequested* candidate arms and retains only its required state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np
from .core import (_float, _uint8, grid, heat_step, motion_to_previous,
                   sample, sihti_decompose, smooth, unsharp, valid_coordinates)


MODES = ("raw", "independent", "flow_cache", "sihti_null", "sihti_select")
_MEMORY_MODES = ("flow_cache", "sihti_null", "sihti_select")


def focus_score(frame: np.ndarray) -> float:
    """Heuristic sharpness of OBSERVED frame only; not proof of detail quality.

    Blur before Sobel to suppress some noise. Downsample for consistent cost;
    score is mean gradient magnitude (8-bit grayscale units). Scenes/lighting
    differ, so pair with a rolling comparison and allow a user threshold.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected 3-channel observed frame")
    h, w = frame.shape[:2]
    scale = min(1., 384. / max(h, w))
    if scale < 1:
        frame = cv2.resize(frame, (max(2,int(w*scale)), max(2,int(h*scale))),
                           interpolation=cv2.INTER_AREA)
    u = _uint8(_float(frame))
    gray = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0),1.1)
    fx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(fx,fy)))


class AutoAnchor:
    """A conservative but fallible sharpness heuristic; never sees restored pixels.

    This gate is not a trained detector. Frame zero may be selected if above
    min_score; subsequent candidates must exceed recent median and obey gap.
    """
    def __init__(self, min_score: float = 10., ratio: float = 1.20,
                 min_gap: int = 20, window: int = 21):
        if min_score < 0 or ratio < 1.0 or min_gap < 1 or window < 3:
            raise ValueError("Invalid automatic-anchor threshold")
        self.min_score, self.ratio = min_score, ratio
        self.min_gap = min_gap
        self.history = deque(maxlen=window)
        self.last_anchor = -min_gap

    def consider(self, observed: np.ndarray, index: int,
                 manual: bool = False) -> tuple[bool,float,str]:
        score = focus_score(observed)
        earlier = list(self.history)
        relative = not earlier or score >= np.median(earlier) * self.ratio
        gap = index - self.last_anchor >= self.min_gap
        qualified = score >= self.min_score and gap and relative
        selected = manual or qualified
        reason = "manual" if manual else ("auto-heuristic" if qualified else "rejected")
        # The history is based on observations, never restored outputs or GT.
        self.history.append(score)
        if selected:
            self.last_anchor = index
        return selected, score, reason


@dataclass
class SelectedFrame:
    image: np.ndarray
    sharp_observation: bool
    memory_valid_fraction: float
    cut_detected: bool
    memory_age: int
    mean_abs_correction: float
    mean_confidence: float
    cut_score: float
    diagnostics: dict[str,np.ndarray]


class SelectiveRestorer:
    """Same restoration recipe as Gate 1, but retains and computes only one arm."""

    def __init__(self, mode: str = "flow_cache", strength: float = 0.90,
                 scene_cut_threshold: float = 0.23,
                 disable_confidence: bool = False):
        if mode not in MODES:
            raise ValueError("Unsupported mode: " + mode)
        if not 0 <= strength <= 2:
            raise ValueError("strength must be in [0,2]")
        self.mode = mode
        self.strength = strength
        self.scene_cut_threshold = scene_cut_threshold
        self.disable_confidence = disable_confidence
        self.reset()

    def reset(self):
        self.anchor_uint8 = None
        self.anchor_core = None
        self.anchor_bands = None
        self.anchor_residue = None
        self.map_to_anchor = None
        self.confidence = None
        self.previous_uint8 = None
        self.shape = None
        self.age = 0

    def invalidate(self):
        self.anchor_uint8 = None
        self.anchor_core = None
        self.anchor_bands = None
        self.anchor_residue = None
        self.map_to_anchor = None
        self.confidence = None
        self.age = 0

    @property
    def bytes_used(self) -> int:
        arrays = (self.anchor_uint8,self.anchor_core,self.anchor_residue,
                  self.map_to_anchor,self.confidence,self.previous_uint8)
        if self.anchor_bands is not None:
            arrays += tuple(self.anchor_bands)
        return sum(a.nbytes for a in arrays if a is not None)

    def _sharp(self, x: np.ndarray):
        h,w = x.shape[:2]
        self.anchor_uint8 = _uint8(x)
        self.anchor_bands = None
        self.anchor_residue = None
        if self.mode == "sihti_select":
            self.anchor_bands,self.anchor_core = sihti_decompose(x)
        else:
            # Exactly the same slowest depth (16) as the Gate 1 sieve,
            # without retaining three full-size intermediate residue arrays.
            cur = x
            for _ in range(16):
                cur = heat_step(cur)
            self.anchor_core = cur
            self.anchor_residue = x - cur
        self.map_to_anchor = grid(h,w)
        self.confidence = np.ones((h,w),dtype=np.float32)
        self.age = 0

    def process(self, frame: np.ndarray, sharp: bool = False,
                debug: bool = False,
                override_flow: np.ndarray | None = None) -> SelectedFrame:
        x = _float(frame)
        h,w = x.shape[:2]
        if self.shape is not None and self.shape != (h,w):
            raise ValueError("Frame dimensions changed; open a separate stream")
        self.shape = (h,w)
        observed = _uint8(x)
        debug_maps: dict[str,np.ndarray] = {}
        if debug:
            debug_maps["confidence"] = np.zeros((h,w),np.float32)
        zero_result = observed if sharp or self.mode=="raw" else _uint8(unsharp(x,self.strength))
        if self.mode not in _MEMORY_MODES:
            return SelectedFrame(zero_result,sharp,0.,False,0,
                                 float(np.mean(np.abs(zero_result.astype(np.int16) -
                                                   observed.astype(np.int16)))),
                                 0.,0.,debug_maps)
        if sharp:
            self._sharp(x)
            self.previous_uint8=observed
            if debug:
                debug_maps["confidence"] = np.ones((h,w),np.float32)
                debug_maps["aligned"] = observed
            return SelectedFrame(observed,True,1.,False,0,0.,1.,0.,debug_maps)
        if self.anchor_core is None or self.previous_uint8 is None:
            self.previous_uint8=observed
            return SelectedFrame(zero_result,False,0.,False,0,
                                 float(np.mean(np.abs(zero_result.astype(np.int16) -
                                                   observed.astype(np.int16)))),
                                 0.,0.,debug_maps)

        prev=_float(self.previous_uint8)
        cut_score=float(np.mean(np.abs(smooth(x,4.)-smooth(prev,4.))))
        if cut_score>self.scene_cut_threshold:
            self.invalidate()
            self.previous_uint8=observed
            return SelectedFrame(zero_result,False,0.,True,0,
                                 float(np.mean(np.abs(zero_result.astype(np.int16) -
                                                   observed.astype(np.int16)))),
                                 0.,cut_score,debug_maps)

        if override_flow is None:
            flow=motion_to_previous(x,prev)
        else:
            flow=np.asarray(override_flow,dtype=np.float32)
            if flow.shape!=(h,w,2):
                raise ValueError("Expected current-to-previous flow HxWx2")
        xy_prev=grid(h,w)+flow
        map_to_anchor=sample(self.map_to_anchor,xy_prev)
        old_conf=sample(self.confidence,xy_prev)
        good=valid_coordinates(xy_prev,w,h)&valid_coordinates(map_to_anchor,w,h)
        if self.disable_confidence:
            conf=good.astype(np.float32)
        else:
            warped_previous=sample(smooth(prev,3.),xy_prev)
            err_step=np.mean(np.abs(smooth(x,3.)-warped_previous),axis=2)
            warped_core=sample(self.anchor_core,map_to_anchor)
            err_anchor=np.mean(np.abs(smooth(x,4.)-smooth(warped_core,2.)),axis=2)
            conf=old_conf*np.exp(-err_step/.25)*np.exp(-err_anchor/.22)*.995
            if override_flow is None:
                opposite=motion_to_previous(prev,x)
                inverse=sample(opposite,xy_prev)
                fb_error=np.linalg.norm(flow+inverse,axis=2)
                conf*=np.exp(-fb_error/2.5)
            conf*=good
            conf=np.clip(conf,0.,1.).astype(np.float32)

        self.map_to_anchor=map_to_anchor
        self.confidence=conf
        self.previous_uint8=observed
        self.age+=1
        if self.mode=="sihti_select":
            warped_bands=[sample(b,map_to_anchor) for b in self.anchor_bands]
            curr_bands,_=sihti_decompose(x)
            correction=np.zeros_like(x)
            for ref_band,curr_band in zip(warped_bands,curr_bands):
                ref_energy=smooth(np.mean(np.abs(ref_band),axis=2),1.5)
                curr_energy=smooth(np.mean(np.abs(curr_band),axis=2),1.5)
                need=np.clip((ref_energy-curr_energy)/(ref_energy+.025),0,1)
                correction+=ref_band*(.35+.65*need[...,None])
        else:
            correction=sample(self.anchor_residue,map_to_anchor)

        out=_uint8(np.clip(x+self.strength*conf[...,None]*correction,0.,1.))
        delta=np.abs(out.astype(np.int16)-observed.astype(np.int16))
        if debug:
            debug_maps["confidence"]=conf.copy()
            debug_maps["aligned"]=sample(self.anchor_uint8,map_to_anchor)
            debug_maps["motion"]=flow
            debug_maps["contribution"]=(self.strength*conf[...,None]*correction).copy()
        return SelectedFrame(out,False,float(np.mean(conf>.25)),False,self.age,
                             float(np.mean(delta)),float(np.mean(conf)),
                             cut_score,debug_maps)
