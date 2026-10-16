"""Gate 2 controls: frozen-output parity, memory, no target leakage."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np

from videorestoration.core import ObservedDetailRestorer
from videorestoration.gate2 import AutoAnchor, SelectiveRestorer, focus_score
from videorestoration.diagnostics import diagnostic_panel,save_panel
from videorestoration.synthetic import make_sequence
from videorestoration.cli import restore_video


class Gate2Tests(unittest.TestCase):
    def test_single_mode_flow_cache_matches_frozen_gate1(self):
        seq=make_sequence(101,frames=10,width=80,height=64,scene_at=None)
        old=ObservedDetailRestorer()
        new=SelectiveRestorer(mode="flow_cache")
        for f in seq:
            a=old.process(f.observed,sharp=f.sharp)
            b=new.process(f.observed,sharp=f.sharp)
            dist=np.abs(a.images["flow_cache"].astype(np.int16)-b.image.astype(np.int16))
            self.assertLessEqual(int(dist.max()),1)
            self.assertAlmostEqual(a.memory_valid_fraction,b.memory_valid_fraction,delta=.001)
            self.assertLess(new.bytes_used,old.bytes_used)
            self.assertIsNone(new.anchor_bands)

    def test_sihti_select_matches_frozen_gate1(self):
        seq=make_sequence(202,frames=5,width=64,height=52,scene_at=None)
        old=ObservedDetailRestorer()
        new=SelectiveRestorer(mode="sihti_select")
        for f in seq:
            a=old.process(f.observed,sharp=f.sharp)
            b=new.process(f.observed,sharp=f.sharp)
            diff=np.abs(a.images["sihti_select"].astype(np.int16)-
                        b.image.astype(np.int16))
            self.assertLessEqual(int(diff.max()),1)

    def test_no_lost_memory_replaced_with_generated_output(self):
        seq=make_sequence(303,frames=5,width=64,height=52,scene_at=None)
        model=SelectiveRestorer("flow_cache")
        model.process(seq[0].observed,sharp=True)
        anchor=model.anchor_uint8.copy()
        residue=model.anchor_residue.copy()
        for f in seq[1:]:
            model.process(f.observed,sharp=False)
        self.assertTrue(np.array_equal(anchor,model.anchor_uint8))
        self.assertTrue(np.array_equal(residue,model.anchor_residue))

    def test_simple_modes_keep_zero_memory(self):
        frame=np.random.default_rng(2).integers(0,255,(48,50,3),dtype=np.uint8)
        for mode in ("raw","independent"):
            x=SelectiveRestorer(mode)
            x.process(frame,sharp=True)
            x.process(frame,sharp=False)
            self.assertEqual(x.bytes_used,0)

    def test_auto_uses_observed_sharpness_only(self):
        h,w=100,110
        sharp=np.zeros((h,w,3),np.uint8)
        cv2.rectangle(sharp,(18,18),(84,79),(255,255,255),-1)
        blurry=cv2.GaussianBlur(sharp,(0,0),5)
        hard=focus_score(sharp)
        soft=focus_score(blurry)
        self.assertGreater(hard,soft)
        mid=(hard+soft)/2
        auto=AutoAnchor(min_score=mid,min_gap=2,ratio=1.0)
        self.assertFalse(auto.consider(blurry,0)[0])
        yes,score,reason=auto.consider(sharp,1)
        self.assertTrue(yes)
        self.assertEqual(reason,"auto-heuristic")
        self.assertFalse(auto.consider(sharp,2)[0])
        self.assertTrue(auto.consider(blurry,3,manual=True)[0])
        self.assertEqual(auto.last_anchor,3)

    def test_diagnostic_panel_is_six_way_and_not_target(self):
        seq=make_sequence(101,frames=3,width=80,height=64,scene_at=None)
        model=SelectiveRestorer("flow_cache")
        model.process(seq[0].observed,sharp=True)
        res=model.process(seq[1].observed,debug=True)
        panel=diagnostic_panel(seq[1].observed,res)
        self.assertEqual(panel.shape,(2*(64+28),3*80,3))
        self.assertGreater(float(np.std(panel)),1)
        with TemporaryDirectory() as d:
            p=save_panel(Path(d),1,seq[1].observed,res)
            self.assertTrue(p.exists())

    def test_mp4_debug_csv_and_auto_opt_in(self):
        with TemporaryDirectory() as td:
            d=Path(td)
            path=d/"in.mp4"
            out=d/"out.mp4"
            w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.,(80,64))
            if not w.isOpened():
                self.skipTest("MP4 encoder not available")
            seq=make_sequence(66,frames=6,width=80,height=64,scene_at=None)
            for x in seq:
                w.write(x.observed)
            w.release()
            report=restore_video(str(path),str(out),{0},"flow_cache",True,
                                 6,.9,.23,diagnostics_dir=str(d/"maps"),
                                 diagnostics_every=2,progress_every=0)
            self.assertEqual(report["frames"],6)
            self.assertEqual(report["anchors_used"],[0])
            self.assertTrue((d/"maps"/"metrics.csv").exists())
            self.assertTrue((d/"maps"/"frame_000002.jpg").exists())
            self.assertGreater(report["peak_state_bytes"],0)
            self.assertGreaterEqual(report["mean_abs_output_correction_0_255"],0)
            with self.assertRaises(ValueError):
                restore_video(str(path),str(out),set(),"flow_cache",False,6,.9,.23)
            r2=restore_video(str(path),str(d/"auto.mp4"),set(),"flow_cache",
                             False,6,.9,.23,auto_anchors=True,
                             auto_min_score=0,auto_ratio=1,
                             auto_gap=2,progress_every=0)
            self.assertEqual(r2["auto_anchor_selection"],"heuristic-unverified")
            self.assertTrue(r2["automatic_anchors_used"])


if __name__=="__main__":
    unittest.main()
