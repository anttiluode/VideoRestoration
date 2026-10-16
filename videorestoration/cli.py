"""Command-line entry points for a sealed synthetic gate and practical video I/O."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .benchmark import run_benchmark, ARMS
from .core import ObservedDetailRestorer


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
                  strength: float, cut_threshold: float) -> dict:
    if not sharp_indices:
        raise ValueError("At least one genuinely sharp observed anchor frame is needed")
    cap=cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_file}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps=24.0
    writer=None
    model=ObservedDetailRestorer(strength=strength,
                                 scene_cut_threshold=cut_threshold)
    count=0
    cuts=[]
    anchor_hits=[]
    valid=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok or (max_frames is not None and count>=max_frames):
                break
            sharp=count in sharp_indices
            result=model.process(frame,sharp=sharp)
            if sharp:
                anchor_hits.append(count)
            if result.cut_detected:
                cuts.append(count)
            valid.append(result.memory_valid_fraction)
            restored=result.images[mode]
            view=np.hstack((frame,restored)) if comparison else restored
            if writer is None:
                h,w=view.shape[:2]
                Path(output_file).parent.mkdir(parents=True,exist_ok=True)
                writer=cv2.VideoWriter(str(output_file),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps,(w,h))
                if not writer.isOpened():
                    raise RuntimeError("MP4 writer unavailable; try a standard .mp4 path")
            writer.write(view)
            count+=1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if count==0:
        raise RuntimeError("No decoded input frames")
    summary={
      "input":str(input_file),"output":str(output_file),"mode":mode,
      "frames":count,"fps":fps,"manual_sharp_indices":sorted(sharp_indices),
      "anchors_used":anchor_hits,"scene_cuts_detected":cuts,
      "mean_memory_valid_fraction":float(np.mean(valid)),
      "peak_final_state_bytes":model.bytes_used,
      "quality_metrics":"unavailable (no clean ground-truth video supplied)",
      "warning":"Declared anchors must really be sharp original observations; an arbitrary interval is not proof of sharpness."
    }
    return summary


def main(argv=None) -> int:
    p=argparse.ArgumentParser(
      description="CPU observed-detail video restoration, with explicit controls")
    sub=p.add_subparsers(dest="command",required=True)
    bench=sub.add_parser("benchmark",help="sealed deterministic test movie and ground-truth comparison")
    bench.add_argument("--out",default="runs/gate1")
    bench.add_argument("--frames",type=int,default=28)
    bench.add_argument("--height",type=int,default=128)
    bench.add_argument("--width",type=int,default=160)
    bench.add_argument("--oracle-flow",action="store_true",
                       help="DIAGNOSTIC ONLY: use simulator's true motion, not a practical result")
    restore=sub.add_parser("restore",help="process a user video with manually verified sharp frames")
    restore.add_argument("--input",required=True)
    restore.add_argument("--output",required=True)
    restore.add_argument("--anchors",required=True,
                         help="0-based genuinely sharp frame indices, e.g. 0,15,38-40")
    restore.add_argument("--mode",choices=ARMS,default="sihti_select")
    restore.add_argument("--comparison",action="store_true",
                         help="output side by side: observed | restored")
    restore.add_argument("--max-frames",type=int)
    restore.add_argument("--strength",type=float,default=0.90)
    restore.add_argument("--cut-threshold",type=float,default=0.23)
    args=p.parse_args(argv)
    if args.command=="benchmark":
        result=run_benchmark(args.out,frames=args.frames,height=args.height,
                              width=args.width,oracle_flow=args.oracle_flow)
        print(json.dumps({
          "arms":result["arms"],"tests":result["tests"],
          "runtime_ms_mean":result["runtime_ms_mean_all_frames"],
          "state_bytes":result["estimated_peak_state_bytes"],
          "output":str(args.out)},indent=2))
    else:
        summary=restore_video(args.input,args.output,parse_indices(args.anchors),
                              args.mode,args.comparison,args.max_frames,
                              args.strength,args.cut_threshold)
        out=Path(args.output).with_suffix(".summary.json")
        out.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2))
    return 0
