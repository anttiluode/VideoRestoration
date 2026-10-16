# Gate 1 frozen procedural results — 2026-10-16

**Report from the first fully green benchmark**: [GitHub Actions run 38870134055](https://github.com/anttiluode/VideoRestoration/actions/runs/38870134055), code SHA [5069071](https://github.com/anttiluode/VideoRestoration/commit/50690719931ece786e76c3353d21e046633a11db). The run completed with all 10 tests passing; full per-frame CSV, JSON report and five-panel PNG were uploaded as the [video-gate1-cpu artifact](https://github.com/anttiluode/VideoRestoration/actions/runs/38870134055/artifacts/12032731818).

**These are deterministic synthetic results, not real camera footage, consumer video superiority, or statistically independent sequences.** Frozen seeds: 101,202,303; 28 frames each; 72 damaged frames; 78 within-scene adjacent pairs. All restoration sees only observed frames, and memory is written only from declared genuinely sharp frames.

| Method | Damaged-frame PSNR dB ↑ | SSIM ↑ | Motion-compensated temporal derivative MAE ↓ |
|---|---:|---:|---:|
| Raw degraded observation | 24.9073 | 0.6022 | 0.024916 |
| Independent per-frame unsharp | 25.3828 | 0.6221 | 0.032983 |
| Previous observed high-pass, flow-aligned (simple control) | **27.7412** | **0.7720** | **0.020914** |
| Sihti separately warped bands, equal weight (mathematical null) | **27.7412** | **0.7720** | **0.020914** |
| Sihti per-scale confidence selection (candidate) | 26.9613 | 0.7246 | 0.021709 |

Interpretation:

1. **Real temporal reuse mechanism, positive pilot:** a cache of previously observed fine detail improves damaged-frame PSNR by **+2.3584 dB** over frame-only sharpening, and reduces clean-referenced temporal derivative error by **36.6%**. This is a result for conventional motion-aligned caching in the synthetic fixture; not evidence for Sihti novelty.
2. **Sihti representation-only null works:** warping the sum or each component separately gives the same image up to rounding. Decomposition does not invent additional detail.
3. **Sihti per-scale candidate fails its quality hypothesis:** **−0.7799 dB** PSNR relative to the flow cache, with **3.8% larger** temporal derivative error. A smaller gain over independent sharpening is insufficient; the correct control is the flow cache. It also stores multiple float32 residue bands, unlike a single-residue cache.
4. The entire 5-arm prototype processes **128×160** frames in **13.52 ms average** on GitHub's hosted CPU runner (including flow and all baseline/candidate outputs). Combined estimated peak model state: **1,966,080 bytes**. This is not a benchmark of a production-optimized single arm or 1080p runtime.

Caveats: the code picks procedural frames from fixed synthetic worlds with small subpixel pan and repeated patterns, plus a synthetic scene replacement and moving occluders. It is not a held-out natural-video benchmark. Sharp anchors are given by construction rather than detected. The optical flow, photometric thresholds, scene-cut detector and scale gates are heuristic and were not tuned on an independent validation family. Quantitative scene-cut recall, ghosting frequency, and real-video sharp-anchor quality still need measurement. The oracle-registration diagnostic is **not** part of these reported primary results.

### Honest next gate

Use independent natural clips with controlled blur/noise where earlier genuinely sharp frames exist. Include the **exact observed-detail flow cache** as the mandatory baseline and test both restoration PSNR/SSIM and ground-truth-aligned temporal derivative error. Profile **per-arm** bytes and CPU time. If a compact multi-band approach does not beat a single warped residue, prefer the simpler method for deployment.

This file is a receipt, not a retuned model. The implementation parameters and evaluation seeds used to produce these numbers predate this interpretation.
