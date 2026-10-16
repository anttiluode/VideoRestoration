"""Synthetic held-out gate: true clean images are for metrics only."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

from .core import ObservedDetailRestorer, grid, sample, valid_coordinates
from .synthetic import make_sequence

ARMS = ("raw","independent","flow_cache","sihti_null","sihti_select")
FROZEN_SEEDS = (101, 202, 303)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) / 255
    b = b.astype(np.float32) / 255
    mse = float(np.mean((a - b) ** 2))
    return float(-10 * np.log10(max(mse, 1e-12)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Per-channel Gaussian SSIM, no third-party metrics package."""
    x = a.astype(np.float32) / 255
    y = b.astype(np.float32) / 255
    ux, uy = cv2.GaussianBlur(x,(11,11),1.5), cv2.GaussianBlur(y,(11,11),1.5)
    vx = cv2.GaussianBlur(x*x,(11,11),1.5)-ux*ux
    vy = cv2.GaussianBlur(y*y,(11,11),1.5)-uy*uy
    cov = cv2.GaussianBlur(x*y,(11,11),1.5)-ux*uy
    return float(np.mean(((2*ux*uy+0.01**2)*(2*cov+0.03**2)) /
             ((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2))))


def temporal_delta_mae(current: np.ndarray, previous: np.ndarray,
                       clean_current: np.ndarray, clean_previous: np.ndarray,
                       dx: float, dy: float) -> float:
    """Motion-compensated temporal derivative error relative to clean video.

    Zero for a perfectly reconstructed clip (including legitimate texture
    changes), not a metric rewarding frozen/smeared frames.
    """
    h,w = current.shape[:2]
    mp = grid(h,w)
    mp[...,0] += dx
    mp[...,1] += dy
    good = valid_coordinates(mp,w,h)
    def fl(x): return x.astype(np.float32) / 255.0
    odelta = fl(current) - sample(fl(previous),mp)
    tdelta = fl(clean_current) - sample(fl(clean_previous),mp)
    return float(np.mean(np.abs((odelta-tdelta)[good])))


def montage(frame, outputs):
    # Raw / independent / flow-cache / Sihti-select / hidden truth
    ims = [outputs[k] for k in ("raw","independent","flow_cache","sihti_select")]
    ims += [frame.clean]
    header = 25
    titled=[]
    for label,im in zip(("observed","frame-only","flow cache","Sihti scale gate","ground truth"),ims):
        top=np.full((header,im.shape[1],3),30,np.uint8)
        cv2.putText(top,label,(5,17),cv2.FONT_HERSHEY_SIMPLEX,0.40,(245,245,245),1,cv2.LINE_AA)
        titled.append(np.vstack([top,im]))
    return np.hstack(titled)


def run_benchmark(out: str | Path, frames: int = 28,
                  height: int = 128, width: int = 160,
                  seeds: tuple[int,...] = FROZEN_SEEDS,
                  oracle_flow: bool = False) -> dict:
    """Run post-hoc scoring against sealed ground truth; no training on test."""
    cv2.setNumThreads(1)
    dest=Path(out)
    dest.mkdir(parents=True,exist_ok=True)
    rows, samples=[],[]
    times=[]
    peak_bytes=0
    for seed in seeds:
        sequence=make_sequence(seed,frames=frames,height=height,width=width)
        model=ObservedDetailRestorer()
        prev_output=None
        prev_clean=None
        prev_frame=None
        for f in sequence:
            override=None
            if oracle_flow and prev_frame is not None and f.scene==prev_frame.scene:
                offx=f.displacement[0]-prev_frame.displacement[0]
                offy=f.displacement[1]-prev_frame.displacement[1]
                override=np.empty((height,width,2),np.float32)
                override[...,0]=offx
                override[...,1]=offy
            start=perf_counter()
            result=model.process(f.observed,sharp=f.sharp,override_flow=override)
            milliseconds=1000*(perf_counter()-start)
            times.append(milliseconds)
            peak_bytes=max(peak_bytes,model.bytes_used)
            for arm in ARMS:
                row={"seed":seed,"frame":f.index,"scene":f.scene,
                     "sharp":f.sharp,"arm":arm,
                     "psnr_db":psnr(result.images[arm],f.clean),
                     "ssim":ssim(result.images[arm],f.clean),
                     "valid_memory_fraction":result.memory_valid_fraction,
                     "scene_cut_detected":result.cut_detected}
                if prev_output is not None and f.scene==prev_frame.scene:
                    dx=f.displacement[0]-prev_frame.displacement[0]
                    dy=f.displacement[1]-prev_frame.displacement[1]
                    row["temporal_delta_mae"]=temporal_delta_mae(
                        result.images[arm],prev_output[arm],
                        f.clean,prev_clean,dx,dy)
                else:
                    row["temporal_delta_mae"]=None
                rows.append(row)
            if seed==seeds[0] and f.index in (1,3,8,13,19,23,27):
                samples.append(montage(f,result.images))
            prev_output=result.images
            prev_clean=f.clean
            prev_frame=f
    with (dest/"per_frame.csv").open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if samples:
        # Representative damaged frames; not only the highest-scoring sample.
        cv2.imwrite(str(dest/"comparison.png"),np.vstack(samples))
    scores={}
    for arm in ARMS:
        damaged=[r for r in rows if r["arm"]==arm and not r["sharp"]]
        pairs=[r for r in rows if r["arm"]==arm and r["temporal_delta_mae"] is not None]
        scores[arm]={
          "damaged_mean_psnr_db":float(np.mean([r["psnr_db"] for r in damaged])),
          "damaged_mean_ssim":float(np.mean([r["ssim"] for r in damaged])),
          "motion_compensated_temporal_delta_mae":float(np.mean([r["temporal_delta_mae"] for r in pairs])),
          "n_damaged":len(damaged),
          "n_pairs":len(pairs)
        }
    per_seed={}
    for seed in seeds:
        per_seed[str(seed)]={}
        for arm in ARMS:
            sub=[r for r in rows if r["seed"]==seed and r["arm"]==arm and not r["sharp"]]
            per_seed[str(seed)][arm]=float(np.mean([r["psnr_db"] for r in sub]))
    strong=scores["flow_cache"]
    candidate=scores["sihti_select"]
    independent=scores["independent"]
    seed_wins_reuse=sum(
        per_seed[str(s)]["flow_cache"]>per_seed[str(s)]["independent"]
        for s in seeds)
    seed_wins_sihti=sum(
        per_seed[str(s)]["sihti_select"]>per_seed[str(s)]["flow_cache"]
        for s in seeds)
    minimum_seed_wins=(2*len(seeds)+2)//3
    reuse_gate=(strong["damaged_mean_psnr_db"]>independent["damaged_mean_psnr_db"]
        and strong["motion_compensated_temporal_delta_mae"]<
            independent["motion_compensated_temporal_delta_mae"]
        and seed_wins_reuse>=minimum_seed_wins)
    sihti_gate=(candidate["damaged_mean_psnr_db"]-
                strong["damaged_mean_psnr_db"]>=0.20
        and candidate["motion_compensated_temporal_delta_mae"]<=
            0.95*strong["motion_compensated_temporal_delta_mae"]
        and seed_wins_sihti>=minimum_seed_wins)
    report={
      "protocol":"video-restoration-gate1-v1",
      "truth_policy":"clean targets only in scoring; all memory writes from explicitly sharp observed frames; causal past only",
      "data":"procedural motion crop with latent texture, motion blur/noise, moving occluder and scene replacement",
      "seeds":list(seeds),"frames_per_seed":frames,
      "height":height,"width":width,"oracle_motion_diagnostic":oracle_flow,
      "arms":scores,"per_seed_damaged_psnr_db":per_seed,
      "runtime_ms_mean_all_frames":float(np.mean(times)),
      "runtime_ms_p95_all_frames":float(np.percentile(times,95)),
      "estimated_peak_state_bytes":int(peak_bytes),
      "registered_quality_gates":{
        "temporal_reuse":bool(reuse_gate),
        "sihti_specific_scale_selection":bool(sihti_gate),
        "per_seed_psnr_wins_reuse":int(seed_wins_reuse),
        "per_seed_psnr_wins_sihti":int(seed_wins_sihti),
        "minimum_seed_wins":int(minimum_seed_wins),
        "sihti_quality_thresholds":{"psnr_improvement_db":0.20,
                                     "temporal_mae_relative_improvement":0.05}
      },
      "tests":{
        "temporal_reuse_beats_frame_only_psnr":
          strong["damaged_mean_psnr_db"]>scores["independent"]["damaged_mean_psnr_db"],
        "sihti_select_beats_single_cache_psnr":
          candidate["damaged_mean_psnr_db"]>strong["damaged_mean_psnr_db"],
        "sihti_select_beats_single_cache_temporal_error":
          candidate["motion_compensated_temporal_delta_mae"]<
          strong["motion_compensated_temporal_delta_mae"]
      },
      "limitations":[
        "Procedural, strongly related image sequences; not a real video benchmark",
        "Flow-cache and Sihti-null use the same anchor high-pass information",
        "Sihti-select holds three full-resolution float32 bands, more bytes than a single cache",
        "Perfect registration is available only under the optional oracle-flow diagnostic",
        "No pretrained Sihti P3 photo codebook or learned recurrent baseline is used",
        "Camera crop shift does not model lens artifacts, rolling shutter or natural video"
      ]
    }
    (dest/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
