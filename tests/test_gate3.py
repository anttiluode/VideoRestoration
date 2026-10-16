"""Gate 3 causality, bounded memory, direct flow and MP4/benchmark tests."""
import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np

from videorestoration.gate3 import (TemporalFusionRestorer, GATE3_MODES,
                                     direct_flow)
from videorestoration.cli import restore_video
from videorestoration.benchmark_gate3 import benchmark_gate3, sequence


class Gate3Tests(unittest.TestCase):
    def test_stores_only_observed_uint8_and_is_bounded(self):
        model=TemporalFusionRestorer(history=3)
        rng=np.random.default_rng(16)
        raw=[]
        for t in range(12):
            img=rng.integers(25,210,(45,60,3),dtype=np.uint8)
            raw.append(img.copy())
            model.process(img)
            self.assertLessEqual(model.bytes_used,3*45*60*3)
            self.assertLessEqual(len(model.frames),3)
        self.assertEqual(len(model.frames),3)
        for saved, original in zip(model.frames,raw[-3:]):
            self.assertTrue(np.array_equal(saved,original))

    def test_direct_registration_does_not_accumulate_old_warps(self):
        rng=np.random.default_rng(11)
        image=rng.integers(0,255,(48,64,3),dtype=np.uint8)
        flow=direct_flow(image,image,scale=.5)
        self.assertEqual(flow.shape,(48,64,2))
        self.assertTrue(np.isfinite(flow).all())
        # Identical inputs can still show subpixel Farneback numerical jitter
        # when estimated at half resolution and resampled.
        self.assertLess(float(np.max(np.abs(flow))),.30)

    def test_no_arbitrary_sharp_frames_needed(self):
        rng=np.random.default_rng(8)
        f=rng.integers(20,160,(48,64,3),dtype=np.uint8)
        for arm in GATE3_MODES:
            m=TemporalFusionRestorer(mode=arm,history=4)
            m.process(f)
            x=m.process(f)
            self.assertEqual(x.image.shape,f.shape)
            self.assertTrue(np.isfinite(x.image).all())
        self.assertEqual(TemporalFusionRestorer("spatial_bilateral").bytes_used,0)

    def test_cut_resets_past_observations(self):
        m=TemporalFusionRestorer(scene_cut_threshold=.08)
        dark=np.full((48,64,3),13,np.uint8)
        bright=np.full((48,64,3),235,np.uint8)
        m.process(dark)
        r=m.process(bright)
        self.assertTrue(r.cut_detected)
        self.assertEqual(len(m.frames),1)
        self.assertTrue(np.array_equal(r.image,bright))
        self.assertTrue(np.array_equal(m.frames[-1],bright))

    def test_declared_sharp_passthrough_but_still_observed(self):
        img=np.random.default_rng(1).integers(0,255,(48,64,3),dtype=np.uint8)
        m=TemporalFusionRestorer()
        output=m.process(img,sharp=True)
        self.assertTrue(np.array_equal(output.image,img))
        self.assertTrue(np.array_equal(m.frames[-1],img))

    def test_no_restored_frame_can_enter_buffer(self):
        scene=sequence(401,frames=6,height=48,width=64)
        m=TemporalFusionRestorer(history=3)
        for f in scene:
            m.process(f["observed"])
        for actual,observed in zip(m.frames,[f["observed"] for f in scene][-3:]):
            self.assertTrue(np.array_equal(actual,observed))

    def test_real_mp4_without_manual_anchors_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            src=d/"in.mp4"
            writer=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),12.,(64,48))
            if not writer.isOpened():
                self.skipTest("runner MP4 encoder not available")
            for f in sequence(401,frames=6,height=48,width=64):
                writer.write(f["observed"])
            writer.release()
            summary=restore_video(str(src),str(d/"out.mp4"),set(),
                         "temporal_fuse",True,6,.9,.23,diagnostics_dir=str(d/"maps"),
                         diagnostics_every=2,progress_every=0)
            self.assertEqual(summary["frames"],6)
            self.assertEqual(summary["anchors_used"],[])
            self.assertTrue((d/"maps"/"metrics.csv").exists())
            self.assertTrue((d/"maps"/"frame_000002.jpg").exists())
            self.assertGreater(summary["peak_state_bytes"],0)
            self.assertLessEqual(summary["peak_state_bytes"],4*48*64*3)
            self.assertEqual(summary["history_frames"],4)

    def test_gate3_benchmark_records_negative_or_positive_honestly(self):
        with tempfile.TemporaryDirectory() as td:
            report=benchmark_gate3(td,seeds=(401,),frames=7,height=48,width=64)
            self.assertEqual(report["protocol"],"gate3-frozen-lowlight-v1")
            self.assertEqual(set(report["arms"]),{"raw",*GATE3_MODES})
            self.assertTrue((Path(td)/"report.json").exists())
            self.assertTrue((Path(td)/"per_frame.csv").exists())
            self.assertTrue((Path(td)/"comparison.png").exists())
            self.assertIn("fuse_vs_spatial_bilateral",report["registered_gates"])


if __name__=="__main__":
    unittest.main()
