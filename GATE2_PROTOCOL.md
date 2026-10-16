# Gate 2 — microscope for weak real-video restoration

Started October 16, 2026, prompted by the user's real-camera clip. This gate is a
diagnostic and computational-budget cleanup, not a new verified enhancement.

## Why

In a 436-frame user clip, only about 6.6% of pixels per frame exceeded
confidence 0.25, while 249 MB of full-mode state was retained. In a
90-frame run, approximately 31.9% passed the threshold, yet both sides
still looked nearly identical. The last manual anchor was frame 30.
No ground truth exists for the clip; we cannot claim the small differences
are improvements.

## What is actually measured

The selective Gate 2 engine computes only the chosen mode. The frozen
Gate 1 implementation remains available to the benchmark and to parity
tests. The microscope reports for every frame:

- selected anchor origin (manual or auto heuristic), observed-frame focus proxy,
  memory age, scene-change score
- fraction of pixels where confidence exceeds 0.25 and mean confidence
- actual mean absolute OUTPUT change on an 8-bit 0–255 scale
- per-frame processing time, peak retained state bytes and ETA

The diagnostics panel displays observed frame, restored frame, warped
past observed anchor, confidence, current-to-previous motion and
actual final output difference, magnified 8 times.

Brighter differences mean *more change*, NOT *better restoration*.

## Frozen engineering gates

- Mode-specific output must match the original all-mode Gate 1 code
  to within one 8-bit level on a predetermined synthetic sequence.
- Flow-cache mode must not retain the Sihti band arrays; state bytes
  must be lower than original all-mode code on same size frames.
- No history for raw/frame-only modes. No updates from restored frames.
- A manually labeled sharp observation passes through unchanged.
- Diagnostics never access hidden clean target imagery.
- Automatic anchors are explicitly opt-in and labeled **heuristic**,
  never certified ground-truth sharp frames.
- Original Gate 1 benchmark and recorded results remain unchanged.
- No Gate 2 improvement in actual natural-video quality is claimed without
  paired clean-degraded natural-video tests and aligned strong baselines.

## On Windows, try the conservative instrument first

Run from the VideoRestoration folder after pulling and updating the package:

~~~cmd
git pull
python3.13 -m pip install -e .
python3.13 -m videorestoration restore --input video.mp4 --output gate2_flow.mp4 --anchors 0,14,28-30 --mode flow_cache --comparison --max-frames 90 --diagnostics-dir gate2_maps --diagnostics-every 15
~~~

Inspect gate2_maps/frame_000045.jpg, frame_000060.jpg and
gate2_maps/metrics.csv, as well as the output .summary.json.

Then **as a separate optional experiment**, try a sharpness heuristic:

~~~cmd
python3.13 -m videorestoration restore --input video.mp4 --output gate2_auto.mp4 --auto-anchors --mode flow_cache --comparison --max-frames 90 --diagnostics-dir gate2_auto_maps
~~~

Auto-anchor scores derive solely from the input video. Blurry but noisy
frames, scene changes or changing illumination can deceive this heuristic.
Inspect auto selections before trusting any apparent gain. The heuristic
uses downsampled, preblurred Sobel edge magnitude and requires a minimum
score, improvement over recent median, and a minimum time between anchors.
Manual anchor flags override the automatic threshold.

This is **not** an assertion that a sharper frame was automatically found;
image gradients are not a validated sharpness oracle. Opting into it can make
restoration worse.

## Remaining science

Compare confidence versus actual output corrections before modifying
thresholds. Recheck with degraded footage created from independently
recorded natural sharp video and optical-flow caches matched for memory.
Test when motion warp, occlusion, rolling shutter and frame-to-frame
exposure changes invalidate reuse. Detect and quantify ghosting. For
deployment, OpenCV MP4 writer currently drops audio and encodes with mp4v.
