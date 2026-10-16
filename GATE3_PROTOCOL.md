# Gate 3 — direct-motion, short-horizon, observed-frame denoising

**Protocol frozen before running Gate 3 benchmarks: October 16, 2026.**

Gate 2's Ultra20 diagnostic found that chaining Farneback flow into a single frame-30 anchor badly deformed the aligned guitar, while multiplying confidence from old confidence by repeated mismatch penalties drove mean confidence from 1 to ~0.005 at frame 60 and the actual output change to nearly zero. This is a *diagnosis*, not a performance win.

## Hypothesis and algorithm

Instead of attempting to recreate absent high frequencies from a single distant "sharp" anchor, maintain the **last four genuinely observed frames**, compressed as uint8 BGR, and **directly** align each one to the current observed frame. For each source pixel, compute a reliability weight from *current/reference* smoothed photometric disagreement, forward/backward flow consistency, in-frame validity and temporal age; recompute from scratch at every frame. Blend weighted observed colors, with the current observed pixel weighted 1. No future frames, reference ground truth, clean synthetic target, or restored outputs enter the ring buffer. The output is a weighted temporal estimate, **not an added Sihti residual**.

Our implementation offers four Gate 3 arms:

- `spatial_bilateral`: current frame only, OpenCV bilateral denoiser; no temporal look-ahead.
- `fusion_unaligned`: same recent-frame buffer and photometric gating, but no optical flow (weak temporal control).
- `fusion_pair`: direct flow to just the previous observed frame (strong simple temporal control).
- `temporal_fuse`: same direct-flow/gating estimator using up to four previous observed frames.

Also report input `raw` and the unchanged Gate 1 `flow_cache` with actually labeled sharp anchor frames on the old Gate 1 synthetic dataset. Do **not** claim one task transfers to the other: Gate 3's all-noisy low-light task has no privileged sharp references, while Gate 1 explicitly has them.

## Predefined synthetic evaluation

A new deterministic Gate 3 sequence creates a 2D textured world with slow subpixel camera motion, foreground transients, independent per-frame Gaussian sensor-like noise, an optional scene switch and modest spatial blur. The generator passes only **observed noisy frames** to the restoration methods; *clean* frames are used afterward strictly to score quality, never to compute flow, weights or choose snapshots.

Frozen seeds `(401, 502, 603)`, 22 frames per seed, at 80×96, sensor-noise sigma 18/255, mild blur 0.5, moving occluder, and scene switch frame 13. Test all four Gate 3 modes from the same observations, record full per-frame PSNR, SSIM, motion-compensated *error versus true clean temporal derivative* (not just brightness consistency), frames/sec, estimated **persistent state bytes**, and elapsed processing time. Separate an additional static-world/motion-only control from the moving-object/scene-cut task in a later gate.

**Gate thresholds, defined before results:**
1. `temporal_fuse` must improve damaged-frame PSNR by **at least 1.0 dB** and reduce temporal derivative error by **at least 10%** versus `spatial_bilateral`, win per-seed PSNR on at least **2/3**, and have no catastrophic >3 dB PSNR loss around moving foreground or scene cut.
2. To justify holding four source frames rather than one, `temporal_fuse` must improve PSNR by **0.30 dB** and reduce temporal derivative error by **5%** versus `fusion_pair` at the cost of additional state and runtime.
3. The latter is **not** satisfied by beating unaligned average alone.
4. Quality-only wins never imply CPU efficiency, and even a passing toy gate does not imply a real-world denoising improvement.

A natural-video controlled-corruption benchmark with independent scenes, realistic sensor noise (Poisson/read/chroma), varying exposure, shot boundaries, and a competitive video denoiser remains necessary. No real-video quality claim is possible from the unpaired Ultra20 clip. Auto-selected sharp anchors are unvalidated and are not needed for Gate 3 fusion.

## Engineering invariants

- Only actual observed frames enter storage, and storage is bounded to `N <= 4` by default.
- No recursively multiplied confidence and no chained optical-flow warping.
- Forward/backward geometric disagreement and spatial validity reduce blending at occlusions and disocclusions.
- Scene-cut handling resets the observed-frame ring buffer; spurious false cuts must be reported.
- Only the chosen mode is computed in MP4 CLI and the existing Gate 1/2 modes continue to work.
- Diagnostic panels use the unchanged six-pane UI; `aligned` means the **newest** directly aligned observed source.
- Neither full-frame denoiser quality nor speed is considered proven until measured.
