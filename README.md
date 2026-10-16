# VideoRestoration — detail already observed is not hallucination

**CPU-first experimental prototype**, based on the Sihti genealogy's most practical question: can a cheap online memory reuse texture the camera *actually saw* to recover later blurry frames, and avoid changing that texture independently every frame?

**Status: Gate 1 research instrument; a measured synthetic temporal-reuse gain, but no demonstrated Sihti-specific or real-world video-quality gain.** The first frozen CPU result is [documented here](results/GATE1_RESULTS.md): the simple flow-aligned detail cache reaches **27.74 dB** vs frame-only sharpening **25.38 dB** on 72 synthetic damaged frames; the Sihti scale-gate candidate instead scores **26.96 dB** and does not beat the cache. Strong multi-frame denoisers, motion-compensated burst fusion and recurrent video super-resolution already do temporal reuse. This repository does **not** claim to invent that principle, beat BasicVSR++, or validate phase/graph memory.

## The one rule that matters

Only a **declared genuinely sharp observed frame** can write to the detail memory. Later blurred, noisy or restored frames can move the *address* and reduce confidence, but **never write their hallucinated pixels back as evidence**.

The implementation is causal: it only sees the present observed frame, past observations and any explicitly supplied earlier sharp anchor. Test-only clean targets and true camera motion are kept outside the restorer.

### Run a synthetic experiment (CPU; no model weights)

Python 3.10+:

```bash
git clone https://github.com/anttiluode/VideoRestoration.git
cd VideoRestoration
python -m pip install -e .
python -m unittest discover -s tests -v
python -m videorestoration benchmark --out runs/gate1
```

Outputs:

- `runs/gate1/comparison.png`: actual observed frame, per-frame unsharp, ordinary warped detail cache, scale-selective Sihti memory and **separate hidden ground truth**. Panels are from predetermined frames, not curated by score.
- `runs/gate1/per_frame.csv`: PSNR, SSIM, valid-memory fraction and motion-compensated temporal derivative error by arm/frame/seed.
- `runs/gate1/report.json`: all three predefined seeds, mean per-arm scores, per-seed PSNR and CPU time/state usage.

This benchmark generates texture, shapes, slow subpixel camera movement, noisy blurred frames, moving occluders and a scene replacement. It is **synthetic**; clean frames are used for evaluation, not restoration.

**Run the stronger oracle registration diagnostic separately** (it is cheating if shown as a deployable result):

```bash
python -m videorestoration benchmark --out runs/oracle-registration --oracle-flow
```

If oracle registration succeeds but estimated flow fails, that's an alignment bottleneck, not evidence against the observed-detail memory idea.

### Restore a real MP4

Find genuine sharp frame indices **by visual inspection** (zero-based). We do not automatically label arbitrary frames as sharp.

```bash
python -m videorestoration restore --input my_clip.mp4 --output restored.mp4 --anchors 0,14,28-30 --mode sihti_select --comparison
```

The result is a side-by-side MP4 if `--comparison` is used and a JSON receipt next to the output MP4. Available `--mode` values: `raw`, `independent`, `flow_cache`, `sihti_null`, `sihti_select`. Use `--max-frames 100` for quick first runs. Run the same clip with `--mode flow_cache` as the essential comparison.

**A real input MP4 has no verified clean target.** Only visual inspection is possible until you provide paired clean/degraded videos. MP4 export itself may add compression. If no sharp anchors are supplied, the command refuses to make up memory.

### Arms and controls

| Arm | What it knows / does | What it establishes |
|---|---|---|
| `raw` | Actually observed damaged frame | Damage baseline |
| `independent` | Per-frame Gaussian unsharp only; no history | Independent frame restoration |
| `flow_cache` | One stored past sharp **high-pass** residual; Farneback motion tracking; confidence gating | The serious simple alternative |
| `sihti_null` | Warp three stored Sihti residues separately, then add; same flow/confidence | **Should equal flow_cache**: linear warping distributes over sums |
| `sihti_select` | Warp each residue scale; gate replacement by how much of that scale is missing in present observation | The unproven extra mechanism |

The Sihti decomposition is an ordinary stable four-neighbour heat operator applied at depths `(0,1,4,16)`. If `R_k = x_{d_k} - x_{d_{k+1}}`, then

```
sharp_observed = R0 + R1 + R2 + core
```

**Exact reconstruction is algebra, not learning or improved image quality.** A fixed-matrix or graph-Laplacian point of view links this to Sihti and to Growth-Microscope; the video pipeline does not borrow Growth-Microscope's semantic-region claims, develop new networks or train an attention mechanism.

All restore arms use OpenCV's CPU Farneback dense optical flow from current observation to the previous observation, with a composed map back to the last sharp anchor. Confidence is reduced for out-of-bounds mappings, observed brightness disagreement and forward/backward flow inconsistency. A large detected cut invalidates memory until another designated sharp observation occurs.

A memory containing **three full-resolution float32 bands** uses more bytes than one high-pass cached float32 residual. This is *not* a byte-matched Sihti-vs-cache comparison. The benchmark reports peak **combined prototype** state bytes and measured CPU time, not deployable video codec quality or GPU usage.

## Hypotheses and decision rules

Read [PREREG_GATE1.md](PREREG_GATE1.md) **before interpreting any numbers**. The test answers two distinct questions:

1. Is reuse of earlier observed detail better than independently sharpening each damaged frame, including temporal derivative accuracy? This is a classic and plausible method, not novel.
2. Does **Sihti-specific multiscale selection** improve over **the same** well-aligned single residual cache enough to justify more memory and work? The answer could be no.

If the candidate loses, the repo still provides a reproducible instrument, a useful clip-processing control and a measured negative result. Do not tune on the frozen three-seed test and still call it held-out; move any tuning to a separate development seed family.

## Relation to earlier work

- [Sihti](https://github.com/anttiluode/Sihti): exact multiscale decomposition; its own-colour affinity graph is **not** used here. A graph built from unseen target pixels would leak the answer.
- [SihtiImageModel](https://github.com/anttiluode/SihtiImageModel): static patch-based conditional reconstruction and honest natural-photo control. **Its offline P3 learned codebook is not implemented here**, so comparisons to it remain future work.
- [Growth-Microscope](https://github.com/anttiluode/Growth-Microscope): changes in edge conductance and task-dependent operators motivate confidence gating, but this code contains no learned topology, damage grouping or Goodfire-style semantic probes.
- [Genealogy](https://anttiluode.github.io/Genealogy/): the historical research map; today's video hypothesis has been recorded as an open question.

**Prior art:** classical motion-compensated video denoising, burst/multi-frame super-resolution, blind temporal-consistency methods and recurrent video restoration. A positive result here would be an engineering result under measured costs, not proof of a new fundamental memory principle.

## Known limitations

No hardware alignment acceleration, no true optical-flow confidence calibration, no learned P3 predictor, no perceptual benchmark, no automatic anchor-quality classifier, no multiple-anchor memory bank, no occlusion-specific ground-truth benchmark, no learned recurrent control and no matched RAM-runtime optimization. The single most important future test is real video with controlled corruptions and carefully chosen sharp source frames.

License: see `LICENSE`.
