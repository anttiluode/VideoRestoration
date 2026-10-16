# Gate 1 — prospective criteria before seeing the benchmark output

Date: 2026-10-16. Frozen test seeds: **101, 202, 303**; 28 frames/seed; 128×160; sharp frames 0,7,14,21 (where applicable); a deterministic scene replacement starts at frame 17. Blurred/noisy frames are degraded independently; one moving colored foreground object introduces unmodelled nonrigid content.

**Independent null**: per-frame unsharp. **Serious conventional control**: single past-observed high-pass flow cache. **Mathematical Sihti null**: warp all bands separately at equal weights, which should equal one cache up to numerical rounding. **Candidate**: Sihti band-selective confidence gate, with higher storage cost.

- A **temporal-reuse win** requires `flow_cache` to have both higher damaged-frame PSNR and lower motion-compensated temporal derivative MAE than `independent` over all frozen data, plus no worse PSNR on at least two of the three seeds.
- A **Sihti-specific win** requires `sihti_select` to outperform `flow_cache` by **at least 0.20 dB mean damaged-frame PSNR**, decrease temporal derivative MAE **at least 5%**, and win PSNR on at least two of three seeds. At present these would be *quality* gates, not a CPU or byte-matched victory; memory costs must be reported.
- The candidate is **not established** by beating raw pixels or the unsharp arm only. It must beat the direct cache with identical flow and valid memory writes.
- A cut error or ghost from an old scene is a serious failure independent of PSNR. The scene-change gate is a hand-chosen heuristic; its false-negative/positive rate is not yet evaluated.
- Runtime/state: measure per-frame time and combined bytes, no speed victory without dedicated equal-mode memory profiling. Single-device measurements are not portable performance claims.

`motion_compensated_temporal_delta_mae` compares the temporal derivative **against clean temporal derivative** after the known synthetic camera warp; this avoids rewarding a perfectly frozen or blurred output merely because it looks stable. Report it on all within-scene consecutive frame pairs, including transitions around explicitly sharp observations, which may expose jumps.

Do not compute flow from hidden true clean frames in the primary test. `--oracle-flow` is a separate upper-bound diagnosis, never counted as a deployable result. Inputs to `ObservedDetailRestorer.process` must contain only observed frame pixels, a manually declared sharp flag, and optionally a motion override in the special oracle experiment.

Other honest boundaries: no trainable models are optimized here, so “held out” means not used in adaptation/selection, not an independent natural-world generalization study. The three procedural seeds are not 3 independent real-video samples. The Sihti null is mathematically incapable of exceeding the cache by representation alone; if a large difference appears, treat it as a bug. Fine-tuning thresholds after observing test outcomes invalidates these preregistration criteria.
