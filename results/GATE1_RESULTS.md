# Gate 1 frozen procedural results — 2026-10-16

**Corrected fairness run**: [GitHub Actions run 38870580177](https://github.com/anttiluode/VideoRestoration/actions/runs/38870580177), code SHA [07c9115](https://github.com/anttiluode/VideoRestoration/commit/07c91152f48c4490b31bd9d4d5e13e01a6bf8113). All **11 tests pass**, including real MP4 I/O. Complete CSV, JSON and panel figure: [video-gate1-cpu artifact](https://github.com/anttiluode/VideoRestoration/actions/runs/38870580177/artifacts/12033531158). **Important correction:** an earlier run (38870134055) sharpened the already-sharp input only in the independent control and thus unfairly inflated its temporal error; its PSNR numbers for damaged frames were unchanged. All temporal claims below use the corrected all-arms sharp-pass-through version.

**These are deterministic synthetic results, not real camera footage, consumer video superiority, or statistically independent sequences.** Frozen seeds: 101,202,303; 28 frames each; 72 damaged frames; 78 within-scene adjacent pairs. All restoration sees only observed frames, and memory is written only from declared genuinely sharp frames.

| Method | Damaged-frame PSNR dB ↑ | SSIM ↑ | Motion-compensated temporal derivative MAE ↓ |
|---|---:|---:|---:|
| Raw degraded observation | 24.9073 | 0.6022 | 0.024916 |
| Independent per-frame unsharp (sharp inputs passed through) | 25.3828 | 0.6221 | 0.026746 |
| Previous observed high-pass, flow-aligned (simple control) | **27.7412** | **0.7720** | **0.020914** |
| Sihti separately warped bands, equal weight (mathematical null) | **27.7412** | **0.7720** | **0.020914** |
| Sihti per-scale confidence selection (candidate) | 26.9613 | 0.7246 | 0.021709 |

Interpretation:

1. **Real temporal reuse mechanism, positive pilot:** a cache of previously observed fine detail improves damaged-frame PSNR by **+2.3584 dB** over frame-only sharpening, and reduces clean-referenced temporal derivative error by **21.8%**. This is a result for conventional motion-aligned caching in the synthetic fixture; not evidence for Sihti novelty.
2. **Sihti representation-only null works:** warping the sum or each component separately gives the same image up to rounding. Decomposition does not invent additional detail.
3. **Sihti per-scale candidate fails its quality hypothesis:** **−0.7799 dB** PSNR relative to the flow cache, with **3.8% larger** temporal derivative error. A smaller gain over independent sharpening is insufficient; the correct control is the flow cache. It also stores multiple float32 residue bands, unlike a single-residue cache.
4. The entire 5-arm prototype processes **128×160** frames in **17.02 ms average** on GitHub's hosted CPU runner (including flow and all baseline/candidate outputs). Combined estimated peak model state: **1,966,080 bytes**. This is not a benchmark of a production-optimized single arm or 1080p runtime.

**Frozen acceptance gates:** temporal-reuse quality **PASS** (PSNR improves on 3/3 procedural seeds and temporal error decreases). Sihti-specific scale-selection **FAIL** (wins 0/3 seeds, misses both 0.20-dB and 5%-temporal thresholds). These are quality-only, not byte-matched claims.\n\nCaveats: the code picks procedural frames from fixed synthetic worlds with small subpixel pan and repeated patterns, plus a synthetic scene replacement and moving occluders. It is not a held-out natural-video benchmark. Sharp anchors are given by construction rather than detected. The optical flow, photometric thresholds, scene-cut detector and scale gates are heuristic and were not tuned on an independent validation family. Quantitative scene-cut recall, ghosting frequency, and real-video sharp-anchor quality still need measurement. The oracle-registration diagnostic is **not** part of these reported primary results.

### Honest next gate

Use independent natural clips with controlled blur/noise where earlier genuinely sharp frames exist. Include the **exact observed-detail flow cache** as the mandatory baseline and test both restoration PSNR/SSIM and ground-truth-aligned temporal derivative error. Profile **per-arm** bytes and CPU time. If a compact multi-band approach does not beat a single warped residue, prefer the simpler method for deployment.

This file is a receipt, not a retuned model. The implementation parameters and evaluation seeds used to produce these numbers predate this interpretation.
