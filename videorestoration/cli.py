"""CLI: frozen Gate 1 benchmark and optional transparent Gate 2 video microscope."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .benchmark import run_benchmark
from .diagnostics import save_panel
from .gate2 import MODES, AutoAnchor, SelectiveRestorer, focus_score


def parse_indices(spec: str) -> set[int]:
    out=set()
    for chunk in spec.split(","):
        chunk=chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a,b=map(int,chunk.split("-",1))
            if b<a or b-a>1000000:
                raise ValueError("Invalid anchor range")
            out.update(range(a,b+1))
        else:
            out.add(int(chunk))
    if any(v<0 for v in out):
        raise ValueError("Anchor indices must be nonnegative")
    return out


def restore_video(input_file: str, output_file: str, sharp_indices: set[int],
                  mode: str, comparison: bool, max_frames: int | None,
                  strength: float, cut_threshold: float, *,
                  diagnostics_dir: str | None = None,
                  diagnostics_every: int = 15,
                  progress_every: int = 30,
                  auto_anchors: bool = False,
                  auto_min_score: float = 10.,
                  auto_ratio: float = 1.20,
                  auto_gap: int = 20) -> dict:
    """Process only selected arm, never reinserting generated frames as memory."""
    if not sharp_indices and not auto_anchors:
        raise ValueError("Supply manually verified --anchors or explicitly opt into --auto-anchors")
    if max_frames is not None and max_frames<=0:
        raise ValueError("max_frames must be positive")
    if diagnostics_every<1 or progress_every<0:
        raise ValueError("Invalid diagnostics/progress cadence")
    if Path(input_file).resolve()==Path(output_file).resolve():
        raise ValueError("Output must differ from input")
    cap=cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_file}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        fps=24.0
    total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total=int(total) if np.isfinite(total) and total>0 else None
    if max_frames is not None and total is not None:
        total=min(total,max_frames)
    selector=AutoAnchor(auto_min_score,auto_ratio,auto_gap) if auto_anchors else None
    model=SelectiveRestorer(mode=mode,strength=strength,
                            scene_cut_threshold=cut_threshold)
    folder=Path(diagnostics_dir) if diagnostics_dir else None
    if folder is not None:
        folder.mkdir(parents=True,exist_ok=True)
    writer=None
    count=0
    cuts=[]
    anchors=[]
    automatic=[]
    valid=[]
    correction=[]
    confidence=[]
    times=[]
    peak=0
    peak_state_frame=0
    rows=[]
    start=perf_counter()
    try:
        while max_frames is None or count<max_frames:
            ok,frame=cap.read()
            if not ok:
                break
            manual=count in sharp_indices
            if selector is not None:
                sharp,score,reason=selector.consider(frame,count,manual=manual)
            else:
                sharp=manual
                score=focus_score(frame) if folder is not None and (
                    count%diagnostics_every==0 or sharp) else None
                reason="manual" if manual else "none"

            debug=folder is not None and (count%diagnostics_every==0 or sharp)
            tick=perf_counter()
            result=model.process(frame,sharp=sharp,debug=debug)
            elapsed_ms=(perf_counter()-tick)*1000.
            times.append(elapsed_ms)
            if sharp:
                anchors.append(count)
                if reason=="auto-heuristic":
                    automatic.append(count)
            if result.cut_detected:
                cuts.append(count)
            valid.append(result.memory_valid_fraction)
            correction.append(result.mean_abs_correction)
            confidence.append(result.mean_confidence)
            if model.bytes_used>peak:
                peak=model.bytes_used
                peak_state_frame=count
            out=np.hstack((frame,result.image)) if comparison else result.image
            if writer is None:
                h,w=out.shape[:2]
                Path(output_file).parent.mkdir(parents=True,exist_ok=True)
                writer=cv2.VideoWriter(str(output_file),
                        cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
                if not writer.isOpened():
                    raise RuntimeError("MP4 writer unavailable; try standard .mp4 output")
            writer.write(out)
            if folder is not None:
                rows.append({
                    "frame":count,
                    "seconds":round(count/fps,3),
                    "anchor_source":reason if sharp else "none",
                    "focus_score":None if score is None else round(score,4),
                    "memory_age":result.memory_age,
                    "valid_fraction_gt_0_25":round(result.memory_valid_fraction,5),
                    "mean_confidence":round(result.mean_confidence,5),
                    "mean_abs_output_correction_0_255":round(result.mean_abs_correction,5),
                    "scene_cut_score":round(result.cut_score,5),
                    "scene_cut_detected":result.cut_detected,
                    "processing_ms":round(elapsed_ms,2)
                })
                if debug or result.cut_detected:
                    # Re-run debug only on cut frame? No: the cut result
                    # already conveys zero valid memory and raw output.
                    save_panel(folder,count,frame,result)
            count+=1
            if progress_every and count%progress_every==0:
                duration=perf_counter()-start
                measured=count/max(duration,1e-6)
                if total:
                    eta=max(0,(total-count)/max(measured,1e-6))
                    progress=f"{100*count/total:5.1f}% ETA {eta/60:.1f} min"
                else:
                    progress="duration unknown"
                print(f"[{count}/{total or '?'}] {progress} | "
                      f"{measured:.2f} frames/s | confidence "
                      f"{result.memory_valid_fraction:.1%} | "
                      f"change {result.mean_abs_correction:.2f}/255",
                      file=sys.stderr,flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if count==0:
        raise RuntimeError("No decoded video frames")
    if folder is not None:
        with (folder/"metrics.csv").open("w",newline="",encoding="utf-8") as fp:
            w=csv.DictWriter(fp,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    dur=perf_counter()-start
    summary={
        "input":str(input_file),"output":str(output_file),"mode":mode,
        "frames":count,"fps":fps,"manual_sharp_indices":sorted(sharp_indices),
        "anchors_used":anchors,"automatic_anchors_used":automatic,
        "scene_cuts_detected":cuts,
        "mean_memory_valid_fraction":float(np.mean(valid)),
        "mean_confidence":float(np.mean(confidence)),
        "mean_abs_output_correction_0_255":float(np.mean(correction)),
        "peak_final_state_bytes":model.bytes_used,  # historic key preserved
        "peak_state_bytes":peak,
        "peak_state_frame":peak_state_frame,
        "mean_processing_ms_per_frame":float(np.mean(times)),
        "elapsed_wall_seconds":float(dur),
        "diagnostics_dir":str(folder) if folder is not None else None,
        "auto_anchor_selection":"heuristic-unverified" if auto_anchors else "disabled",
        "quality_metrics":"unavailable (no clean ground-truth video supplied)",
        "warning":"Confidence and sharpness are heuristics, not proof of reconstruction quality; "
                  "OpenCV's MP4 writer does not preserve source audio."
    }
    return summary


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description="VideoRestoration: CPU temporal-memory microscope")
    sub=p.add_subparsers(dest="command",required=True)
    bench=sub.add_parser("benchmark",help="unaltered Gate 1 synthetic benchmark")
    bench.add_argument("--out",default="runs/gate1")
    bench.add_argument("--frames",type=int,default=28)
    bench.add_argument("--height",type=int,default=128)
    bench.add_argument("--width",type=int,default=160)
    bench.add_argument("--oracle-flow",action="store_true",
                       help="DIAGNOSTIC ONLY: simulator true motion, not practical")
    restore=sub.add_parser("restore",help="selected-mode video restoration with optional microscope")
    restore.add_argument("--input",required=True)
    restore.add_argument("--output",required=True)
    restore.add_argument("--anchors",default="",
                         help="verified 0-based sharp frame numbers, e.g. 0,14,28-30")
    restore.add_argument("--mode",choices=MODES,default="flow_cache")
    restore.add_argument("--comparison",action="store_true",
                         help="output observed | restored, side by side")
    restore.add_argument("--max-frames",type=int)
    restore.add_argument("--strength",type=float,default=.90)
    restore.add_argument("--cut-threshold",type=float,default=.23)
    restore.add_argument("--diagnostics-dir",
                         help="export observed/ref/flow/confidence/correction panels and metrics.csv")
    restore.add_argument("--diagnostics-every",type=int,default=15)
    restore.add_argument("--progress-every",type=int,default=30,
                         help="0 disables progress; default every 30 frames")
    restore.add_argument("--auto-anchors",action="store_true",
                         help="EXPERIMENTAL: automatically mark candidate sharp observed frames")
    restore.add_argument("--auto-min-score",type=float,default=10.)
    restore.add_argument("--auto-ratio",type=float,default=1.20)
    restore.add_argument("--auto-gap",type=int,default=20)
    args=p.parse_args(argv)
    if args.command=="benchmark":
        result=run_benchmark(args.out,frames=args.frames,height=args.height,
                             width=args.width,oracle_flow=args.oracle_flow)
        print(json.dumps({
            "arms":result["arms"],"tests":result["tests"],
            "registered_quality_gates":result["registered_quality_gates"],
            "per_seed_damaged_psnr_db":result["per_seed_damaged_psnr_db"],
            "runtime_ms_mean":result["runtime_ms_mean_all_frames"],
            "state_bytes":result["estimated_peak_state_bytes"],
            "output":str(args.out)},indent=2))
    else:
        summary=restore_video(args.input,args.output,parse_indices(args.anchors),
              args.mode,args.comparison,args.max_frames,args.strength,args.cut_threshold,
              diagnostics_dir=args.diagnostics_dir,
              diagnostics_every=args.diagnostics_every,
              progress_every=args.progress_every,
              auto_anchors=args.auto_anchors,auto_min_score=args.auto_min_score,
              auto_ratio=args.auto_ratio,auto_gap=args.auto_gap)
        dest=Path(args.output).with_suffix(".summary.json")
        dest.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2))
    return 0
