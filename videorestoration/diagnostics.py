"""What did the video memory really contribute? Deterministic diagnostic panels."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .gate2 import SelectedFrame


def bgr_motion(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Hue=flow direction, value=relative speed; zero confidence dark."""
    mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1],angleInDegrees=True)
    limit=max(1.,float(np.percentile(mag,95)))
    hsv=np.zeros((*mag.shape,3),np.uint8)
    hsv[...,0]=np.clip(ang/2.,0,179).astype(np.uint8)
    hsv[...,1]=255
    hsv[...,2]=np.clip(mag/limit*240.,0,240).astype(np.uint8)
    colored=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return (colored*(.25+.75*valid[...,None])).astype(np.uint8)


def labeled(image: np.ndarray, title: str, tile_width: int) -> np.ndarray:
    h,w=image.shape[:2]
    new_w=min(w,tile_width)
    new_h=max(1,round(h*new_w/w))
    img=cv2.resize(image,(new_w,new_h),
                   interpolation=cv2.INTER_AREA if new_w<w else cv2.INTER_LINEAR)
    bar=np.full((28,new_w,3),23,np.uint8)
    cv2.putText(bar,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,
                (255,255,255),1,cv2.LINE_AA)
    return np.vstack((bar,img))


def diagnostic_panel(current: np.ndarray, result: SelectedFrame,
                     tile_width: int = 360) -> np.ndarray:
    """Show observed, restored, reference, confidence, transfer and motion.

    Last tile is *actual* 8x amplified absolute output difference, so any
    clipped/cancelled correction is visible rather than merely a prediction.
    """
    if tile_width < 100:
        raise ValueError("tile_width must be >=100")
    dbg=result.diagnostics
    h,w=current.shape[:2]
    conf=dbg.get("confidence",np.zeros((h,w),np.float32))
    cm=cv2.applyColorMap(np.clip(conf*255,0,255).astype(np.uint8),
                         cv2.COLORMAP_VIRIDIS)
    aligned=dbg.get("aligned",np.zeros_like(current))
    flow=dbg.get("motion",np.zeros((h,w,2),np.float32))
    magnitude=np.abs(current.astype(np.int16)-
                     result.image.astype(np.int16))
    change=np.clip(magnitude*8,0,255).astype(np.uint8)
    maps=[
        (current,"observed (not GT)"),
        (result.image,"restored"),
        (aligned,"past reference aligned"),
        (cm,"memory confidence (0..1)"),
        (bgr_motion(flow,conf),"current -> previous flow"),
        (change,"actual pixel change x8")
    ]
    tiles=[labeled(im,name,tile_width) for im,name in maps]
    # identical tile dimensions, independent of source dimensions
    return np.vstack((np.hstack(tiles[:3]),np.hstack(tiles[3:])))


def save_panel(folder: Path, frame_index: int, current: np.ndarray,
               result: SelectedFrame) -> Path:
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"frame_{frame_index:06d}.jpg"
    if not cv2.imwrite(str(path),diagnostic_panel(current,result),
                       [cv2.IMWRITE_JPEG_QUALITY,90]):
        raise IOError(f"Could not write diagnostic panel: {path}")
    return path
