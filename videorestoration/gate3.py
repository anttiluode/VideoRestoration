"""Gate 3: bounded observed-frame multi-frame denoiser, with direct optical flow.

All references are actual noisy camera observations. Never writes generated
outputs, accesses future video or clean benchmark targets. Direct-to-current
registration and confidence computed anew for every candidate reference.
"""
from __future__ import annotations

from collections import deque
import cv2
import numpy as np
from .core import _float, _uint8, grid, sample, smooth, valid_coordinates
from .gate2 import SelectedFrame

GATE3_MODES=("spatial_bilateral","fusion_unaligned","fusion_pair","temporal_fuse")


def spatial_denoise(frame: np.ndarray) -> np.ndarray:
    """Single-frame CPU denoiser; very simple, not state of art."""
    return cv2.bilateralFilter(frame,5,28,2.0)


def _gray_for_flow(image: np.ndarray) -> np.ndarray:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Extra smoothing reduces high-frequency ISO noise in Farneback.
    return cv2.GaussianBlur(gray,(5,5),1.1)


def direct_flow(current: np.ndarray, reference: np.ndarray,
                scale: float=.5) -> np.ndarray:
    """Dense map current -> chosen past observation; never composes old maps."""
    if not 0<scale<=1:
        raise ValueError("flow scale must be in (0,1]")
    h,w=current.shape[:2]
    a,b=_gray_for_flow(current),_gray_for_flow(reference)
    if scale<1:
        sw,sh=max(16,round(w*scale)),max(16,round(h*scale))
        a=cv2.resize(a,(sw,sh),interpolation=cv2.INTER_AREA)
        b=cv2.resize(b,(sw,sh),interpolation=cv2.INTER_AREA)
    else:
        sw,sh=w,h
    flow=cv2.calcOpticalFlowFarneback(
        a,b,None,.5,3,17,3,5,1.2,0)
    if (sw,sh)!=(w,h):
        flow=cv2.resize(flow,(w,h),interpolation=cv2.INTER_LINEAR)
        flow[...,0]*=w/sw
        flow[...,1]*=h/sh
    return flow.astype(np.float32)


class TemporalFusionRestorer:
    """Independent, non-recursive causal short-term fusion of observed frames.

    Each pixel: (observed + sum(w_i * warped(observed_i))) / (1 + sum w_i).
    w_i uses ONLY current/reference observed images, direct flow, validation
    of mappings, and time decay. No infinite-horizon confidence product.
    """

    def __init__(self, mode: str="temporal_fuse", history: int=4,
                 scene_cut_threshold: float=.23,
                 flow_scale: float=.5,
                 color_threshold: float=.075,
                 fb_threshold: float=2.0):
        if mode not in GATE3_MODES:
            raise ValueError("Unknown Gate 3 mode: "+mode)
        if not 1<=history<=12:
            raise ValueError("history must be 1..12")
        if not 0<flow_scale<=1:
            raise ValueError("flow_scale must be in (0,1]")
        if color_threshold<=0 or fb_threshold<=0:
            raise ValueError("confidence thresholds must be positive")
        self.mode=mode
        self.history=1 if mode=="fusion_pair" else history
        self.flow_scale=flow_scale
        self.scene_cut_threshold=scene_cut_threshold
        self.color_threshold=color_threshold
        self.fb_threshold=fb_threshold
        self.frames=deque(maxlen=self.history)
        self.shape=None
        self._age=0
        self.last_debug={}
        self._grid=None

    @property
    def bytes_used(self) -> int:
        return sum(im.nbytes for im in self.frames) # persistent frame cache only

    def reset(self):
        self.frames.clear()
        self._age=0

    def process(self, frame: np.ndarray, sharp: bool=False,
                debug: bool=False) -> SelectedFrame:
        observed=_uint8(_float(frame))
        h,w=observed.shape[:2]
        if self.shape is not None and self.shape!=(h,w):
            raise ValueError("Video frame shape changed")
        self.shape=(h,w)
        self.last_debug={}
        if self._grid is None:
            self._grid=grid(h,w)
        x=observed.astype(np.float32)/255.0
        empty_debug={"confidence":np.zeros((h,w),np.float32)} if debug else {}
        if self.mode=="spatial_bilateral":
            out=observed if sharp else spatial_denoise(observed)
            change=float(np.mean(np.abs(out.astype(np.int16)-
                                        observed.astype(np.int16))))
            return SelectedFrame(out,sharp,0.,False,0,change,0.,0.,empty_debug)

        cut_score=0.
        cut=False
        if self.frames:
            prev=self.frames[-1].astype(np.float32)/255.
            cut_score=float(np.mean(np.abs(smooth(x,4.)-smooth(prev,4.))))
            if cut_score>self.scene_cut_threshold:
                self.reset()
                cut=True
        # A user-declared genuinely sharp observation passes through
        # unchanged, but is still stored as an observed reference.
        if sharp or not self.frames:
            self.frames.append(observed.copy())
            self._age=0
            if debug:
                empty_debug["confidence"]=np.ones((h,w),np.float32)
                empty_debug["aligned"]=observed.copy()
            return SelectedFrame(observed,sharp,1. if sharp else 0.,cut,0,
                                 0.,1. if sharp else 0.,cut_score,empty_debug)

        weighted=x.copy()
        total_weight=np.ones((h,w),np.float32)
        last_aligned=None
        last_flow=None
        good_total=np.zeros((h,w),np.float32)
        current_low=smooth(x,1.4)

        for age,reference in enumerate(reversed(self.frames),start=1):
            if self.mode=="fusion_unaligned":
                flow=np.zeros((h,w,2),np.float32)
                fb_conf=np.ones((h,w),np.float32)
            else:
                flow=direct_flow(observed,reference,self.flow_scale)
                # Expensive but important: reject inconsistent correspondences.
                opposite=direct_flow(reference,observed,self.flow_scale)
                inverse=sample(opposite,self._grid+flow)
                cycle=np.linalg.norm(flow+inverse,axis=2)
                fb_conf=np.exp(-.5*(cycle/self.fb_threshold)**2).astype(np.float32)
            coords=self._grid+flow
            valid=valid_coordinates(coords,w,h).astype(np.float32)
            warped=sample(reference,coords).astype(np.float32)/255.0
            # Smoothing only for match/occlusion confidence: output uses full
            # observed texture, no model-synthesized reference values.
            low_ref=sample(smooth(reference.astype(np.float32)/255.,1.4),
                           coords)
            err=np.mean(np.abs(current_low-low_ref),axis=2)
            # Robust photometric disagreement; avoids leaking into new objects.
            agreement=np.exp(-.5*(err/self.color_threshold)**2).astype(np.float32)
            weight=valid*agreement*fb_conf*(.88**(age-1))*.85
            weighted+=warped*weight[...,None]
            total_weight+=weight
            good_total+=weight
            if last_aligned is None:
                last_aligned=_uint8(warped)
                last_flow=flow
        fused=np.clip(weighted/total_weight[...,None],0,1)
        out=_uint8(fused)
        change=float(np.mean(np.abs(out.astype(np.int16)-
                                    observed.astype(np.int16))))
        useful=np.clip(good_total/(1.+good_total),0,1).astype(np.float32)
        if debug:
            empty_debug.update({
                "aligned":last_aligned,
                "confidence":useful.copy(),
                "motion":last_flow,
                "contribution":fused-x
            })
        self.frames.append(observed.copy())
        self._age=min(self._age+1,self.history)
        return SelectedFrame(out,False,float(np.mean(useful>.25)),cut,
                             self._age,change,float(np.mean(useful)),
                             cut_score,empty_debug)
