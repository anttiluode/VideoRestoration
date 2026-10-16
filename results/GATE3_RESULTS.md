# Gate 3 — frozen all-noisy camera benchmark (October 16, 2026)

**Status: BOTH preregistered synthetic quality gates PASS. NOT a real-video result.**

Provenance: [GitHub Actions run 38900257273](https://github.com/anttiluode/VideoRestoration/actions/runs/38900257273), evaluated after implementing frozen [GATE3_PROTOCOL.md](../GATE3_PROTOCOL.md), [code commit 278d2ef](https://github.com/anttiluode/VideoRestoration/commit/278d2ef25ba0a35492a2c18a0e490bb325b001df).
[Complete unselected benchmark receipts: JSON, per-frame CSV and comparison PNG](https://github.com/anttiluode/VideoRestoration/actions/runs/38900257273/artifacts/12045692300).

Fixture: three deterministic procedural worlds, seeds 401/502/603, 22 frames each; 80×96 pixels; Gaussian sensor-like noise sigma=18/255 added independently to *all* observations, mild Gaussian blur 0.5, subpixel pan, moving occluder, hard scene change at frame 13. **No sharp or clean frame is passed to any denoiser.** Clean synthetic targets and true camera displacement are held for evaluation only. Existing Gates 1 and 2 benchmarks and their negative controls remain unchanged.

| Method | Mean PSNR dB ↑ | Mean SSIM ↑ | True-clean temporal derivative MAE ↓ | Mean process ms/frame* | Persistent state bytes |
|---|---:|---:|---:|---:|---:|
| Unprocessed noisy observations | 22.7498 | 0.5913 | 0.06412 | — | 0 |
| Spatial bilateral (no temporal access) | 24.1377 | 0.6520 | 0.05260 | 0.104 | 0 |
| Unaligned recent-frame fusion | 23.5782 | 0.6077 | 0.03537 | 1.040 | 92,160 |
| Direct one-previous-frame fusion | 25.0442 | 0.6932 | 0.03856 | 1.051 | 23,040 |
| **Direct four-frame motion fusion** | **26.2692** | **0.7559** | **0.02815** | 2.968 | 92,160 |

*CPU timings were obtained only on the tiny 80×96 GitHub-hosted runner; they include algorithm processing but not MP4 decode/write or setup. **Do not extrapolate them to 1080p**. Persistent bytes are the stored uint8 ring-buffer frames only; they **exclude temporary optical-flow arrays and working buffers**.

## Frozen decisions

- **PASS: four-frame versus bilateral.** +2.1315 dB PSNR; approximately 46.5% lower clean-referenced temporal derivative error; wins PSNR on 3/3 procedural seeds; no >3 dB per-frame loss to raw input on the registered moving-foreground/cut condition.
- **PASS: four-frame versus direct two-frame fusion.** +1.2250 dB PSNR and approximately 27.0% lower temporal error. Extra history helps on this fixture but costs ~3× the computation and 4× persistent memory.
- **Negative control:** unaligned fusion has lower PSNR (23.5782 dB) than even independent bilateral denoising (24.1377 dB), although its temporal error is low. Lower flicker alone is not evidence of correct detail.

## Boundary of the claim

The experiment validates a classical engineering mechanism on related synthetic sequences, not new mathematical novelty, natural-video fidelity, or advantage over advanced VBM3D/VBM4D/BasicVSR or neural denoisers. Optical flow can fail on parallax, foreground disocclusion, abrupt lighting, rolling shutter and real low-light colored sensor noise. This benchmark does not match compute costs, real-video scene diversity, or portable run time, and three procedural seeds are not independent natural-camera samples.

The highest-value next step is to run **temporal_fuse**, **fusion_pair**, and **spatial_bilateral** on the *same* first 90 frames of the previously uploaded Ultra20 camera clip, export Gate 3 confidence/correction panels, then examine what is really different. No clean target exists for that clip. Please do not infer restoration quality solely from increased difference-map brightness.
