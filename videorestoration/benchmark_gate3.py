"""Frozen Gate 3 CPU study: all-noisy video, independent clean-target evaluation.

Unlike Gate 1, no frame in this fixture is guaranteed sharp. The candidate
cannot read clean frames, true camera motion, future inputs or scene labels.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

from .benchmark import psnr, ssim, temporal_delta_mae
from .gate3 import TemporalFusionRestorer, GATE3_MODES
from .synthetic import make_world

SEEDS=(401,502,603)
ARMS=("raw",)+GATE3_MODES


def sequence(seed: int, frames: int=22, height: int=80, width: int=96,
             sigma: float=18., blur: float=.5,
             scene_at: int=13):
    """Controlled camera translation, occluder, cut, independent sensor noise."""
    if frames<4 or min(height,width)<32 or sigma<0 or blur<0:
        raise ValueError("Invalid fixture shape/noise")
    rng=np.random.default_rng(seed+8006)
    worlds=(make_world(height,width,seed,0),make_world(height,width,seed,1))
    out=[]
    for i in range(frames):
        scene=int(i>=scene_at)
        local=i if not scene else i-scene_at
        dx,dy=.61*local,.31*local
        mx,my=np.meshgrid(np.arange(width,dtype=np.float32)+37+dx,
                          np.arange(height,dtype=np.float32)+42+dy)
        clean=cv2.remap(worlds[scene],mx,my,cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
        # Foreground object is newly visible and moves independently.
        moving= i%10 in (5,6,7,8)
        if moving:
            x0=12+(i*4)%max(14,width-30)
            y0=22+(i*3)%max(14,height-27)
            cv2.rectangle(clean,(x0,y0),(x0+15,y0+11),
                          (25,200,65) if scene else (230,30,40),-1)
        obs=cv2.GaussianBlur(clean,(0,0),blur) if blur else clean.copy()
        obs=np.clip(obs.astype(np.float32)+
                    rng.normal(0,sigma,obs.shape),0,255).astype(np.uint8)
        out.append({"index":i,"observed":obs,"clean":clean,
                    "scene":scene,"moving":moving,"dx":dx,"dy":dy})
    return out


def tile(image: np.ndarray, label: str):
    h,w=image.shape[:2]
    head=np.full((22,w,3),24,dtype=np.uint8)
    cv2.putText(head,label,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.31,
                (250,250,250),1,cv2.LINE_AA)
    return np.vstack((head,image))


def benchmark_gate3(out: str | Path="runs/gate3",
                    seeds: tuple[int,...]=SEEDS,
                    frames: int=22,height: int=80,width: int=96)->dict:
    cv2.setNumThreads(1)
    target=Path(out)
    target.mkdir(parents=True,exist_ok=True)
    metrics=[]
    times={m:[] for m in GATE3_MODES}
    peaks={m:0 for m in GATE3_MODES}
    panels=[]
    for seed in seeds:
        # This simulator's clean and scene metadata are NEVER passed to models.
        seq=sequence(seed,frames,height,width)
        models={m:TemporalFusionRestorer(mode=m,history=4,flow_scale=.5)
                for m in GATE3_MODES}
        old_results=None
        prev=None
        for f in seq:
            output={"raw":f["observed"]}
            for m,model in models.items():
                tick=perf_counter()
                out_frame=model.process(f["observed"],sharp=False)
                times[m].append((perf_counter()-tick)*1000)
                peaks[m]=max(peaks[m],model.bytes_used)
                output[m]=out_frame.image
            for arm in ARMS:
                score={
                    "seed":seed,"frame":f["index"],"scene":f["scene"],
                    "new_foreground":f["moving"],"method":arm,
                    "psnr_db":psnr(output[arm],f["clean"]),
                    "ssim":ssim(output[arm],f["clean"]),
                }
                if prev is not None and f["scene"]==prev["scene"]:
                    score["temporal_delta_mae"]=temporal_delta_mae(
                        output[arm],old_results[arm],f["clean"],prev["clean"],
                        f["dx"]-prev["dx"], f["dy"]-prev["dy"])
                else:
                    score["temporal_delta_mae"]=None
                metrics.append(score)
            if seed==seeds[0] and f["index"] in (4,7,11,14,18):
                names=("raw","spatial_bilateral","fusion_pair","temporal_fuse")
                panels.append(np.hstack([tile(output[m],m) for m in names]+
                                        [tile(f["clean"],"hidden clean GT")]))
            prev=f
            old_results=output
    with (target/"per_frame.csv").open("w",newline="",encoding="utf8") as fp:
        wr=csv.DictWriter(fp,fieldnames=list(metrics[0]))
        wr.writeheader()
        wr.writerows(metrics)
    if panels:
        cv2.imwrite(str(target/"comparison.png"),np.vstack(panels))
    scores={}
    per_seed={}
    for arm in ARMS:
        records=[r for r in metrics if r["method"]==arm]
        pairs=[r["temporal_delta_mae"] for r in records if r["temporal_delta_mae"] is not None]
        scores[arm]={
            "psnr_db":float(np.mean([r["psnr_db"] for r in records])),
            "ssim":float(np.mean([r["ssim"] for r in records])),
            "temporal_delta_mae":float(np.mean(pairs)),
            "mean_runtime_ms":float(np.mean(times[arm])) if arm in times else None,
            "peak_persistent_bytes":int(peaks[arm]) if arm in peaks else 0,
            "frames":len(records)
        }
        per_seed[arm]={str(s):float(np.mean([
            r["psnr_db"] for r in records if r["seed"]==s]))
            for s in seeds}
    candidate=scores["temporal_fuse"]
    spatial=scores["spatial_bilateral"]
    pair=scores["fusion_pair"]
    wins_spatial=sum(per_seed["temporal_fuse"][str(s)]>
                     per_seed["spatial_bilateral"][str(s)] for s in seeds)
    # Scene boundary and moving-object protection relative to raw input.
    hard=[r for r in metrics if r["method"]=="temporal_fuse"
          and (r["new_foreground"] or r["frame"] in (13,14))]
    bad_by_frame=[]
    for r in hard:
        raw=next(q for q in metrics if q["method"]=="raw"
                 and q["frame"]==r["frame"] and q["seed"]==r["seed"])
        if raw["psnr_db"]-r["psnr_db"]>3.0:
            bad_by_frame.append([r["seed"],r["frame"],
                                 round(raw["psnr_db"]-r["psnr_db"],3)])
    gates={
        "fuse_vs_spatial_bilateral":bool(
            candidate["psnr_db"]-spatial["psnr_db"]>=1.
            and candidate["temporal_delta_mae"]<=.9*spatial["temporal_delta_mae"]
            and wins_spatial>=2 and not bad_by_frame),
        "fuse_vs_direct_pair":bool(
            candidate["psnr_db"]-pair["psnr_db"]>=.30
            and candidate["temporal_delta_mae"]<=.95*pair["temporal_delta_mae"]),
        "per_seed_psnr_wins_vs_spatial":int(wins_spatial),
        "large_losses_on_foreground_or_scene_cut":bad_by_frame
    }
    report={
        "protocol":"gate3-frozen-lowlight-v1",
        "data":"deterministic procedural low-light pan, independent Gaussian noise, "
               "moving foreground and scene replacement; NO natural-video proof",
        "input_to_model":"observed frames only; no clean targets, true motion or scene labels",
        "seeds":list(seeds),"frames_per_seed":frames,
        "geometry":[height,width],"noise_sigma_8bit":18.,
        "blur_sigma":.5,"scene_cut_at":13,
        "arms":scores,"per_seed_psnr":per_seed,"registered_gates":gates,
        "limitations":[
            "Three closely related procedural worlds, not real clips or statistical trials",
            "Farneback flow can distort parallax, rolling-shutter or low-texture regions",
            "Source pixels may themselves be noisy or blurred; weighted averaging reduces noise but loses texture",
            "Bilateral single-frame and direct two-frame fusion are meaningful but not best-known video denoisers",
            "Peak bytes count persistent ring buffers only; OpenCV optical flow and temporary float arrays also consume memory",
            "No equal-FLOP comparison, offline optical-flow solver, audio restoration or high-resolution timing claim"
        ]
    }
    (target/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    return report
