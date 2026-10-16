import unittest
import numpy as np
from videorestoration.core import (ObservedDetailRestorer, sihti_decompose,
                                   sihti_reconstruct, motion_to_previous)
from videorestoration.synthetic import make_sequence
from videorestoration.benchmark import psnr, temporal_delta_mae
from videorestoration.cli import parse_indices


class CoreGates(unittest.TestCase):
    def test_sieve_is_lossless_not_a_new_information_source(self):
        rng=np.random.default_rng(19)
        for h,w in ((32,40),(71,39)):
            x=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            r,c=sihti_decompose(x)
            target=x.astype(np.float32)/255
            self.assertLess(np.max(np.abs(sihti_reconstruct(r,c)-target)),2e-6)

    def test_no_memory_before_sharp_anchor(self):
        model=ObservedDetailRestorer()
        blank=np.full((36,44,3),100,np.uint8)
        r=model.process(blank,sharp=False)
        self.assertEqual(r.memory_valid_fraction,0.0)
        self.assertTrue(np.array_equal(r.images["independent"],r.images["sihti_select"]))
        self.assertIsNone(model.anchor_bands)

    def test_sharp_observation_only_is_memory_write(self):
        seq=make_sequence(101,frames=4,height=48,width=60)
        model=ObservedDetailRestorer()
        x=model.process(seq[0].observed,sharp=True)
        self.assertTrue(x.sharp_observation)
        before=model.anchor.copy()
        model.process(seq[1].observed,sharp=False)
        self.assertTrue(np.array_equal(before,model.anchor))
        self.assertEqual(model.age,1)

    def test_sihti_linear_null_equals_single_residue_warp(self):
        rng=np.random.default_rng(13)
        anchor=rng.integers(30,230,(48,60,3),dtype=np.uint8)
        blurred=anchor.copy()
        model=ObservedDetailRestorer(disable_confidence=True)
        model.process(anchor,sharp=True)
        z=np.zeros((48,60,2),np.float32)
        out=model.process(blurred,sharp=False,override_flow=z).images
        # Linearity of the warp and sum should survive quantization to uint8.
        difference=np.abs(out["sihti_null"].astype(int) -
                          out["flow_cache"].astype(int))
        self.assertLessEqual(difference.max(),1)

    def test_zero_flow_override_and_oracle_separation(self):
        rng=np.random.default_rng(27)
        x=rng.integers(0,256,(40,42,3),dtype=np.uint8)
        flow=motion_to_previous(x.astype(np.float32)/255,x.astype(np.float32)/255)
        # OpenCV Farneback has small subpixel numerical jitter even on identical inputs.\n        self.assertLess(float(np.max(np.abs(flow))),0.12)

    def test_hard_scene_change_kills_memory(self):
        model=ObservedDetailRestorer(scene_cut_threshold=0.08)
        bright=np.full((48,64,3),235,np.uint8)
        dark=np.full((48,64,3),10,np.uint8)
        model.process(bright,sharp=True)
        response=model.process(dark,sharp=False)
        self.assertTrue(response.cut_detected)
        self.assertIsNone(model.anchor)
        self.assertTrue(np.array_equal(response.images["sihti_select"],
                                       response.images["independent"]))

    def test_anchor_refresh_changes_only_at_declared_observation(self):
        seq=make_sequence(33,frames=9,height=48,width=60,anchor_every=4,scene_at=None)
        model=ObservedDetailRestorer()
        model.process(seq[0].observed,sharp=True)
        first=model.anchor.copy()
        for f in seq[1:4]:
            model.process(f.observed,sharp=False)
        self.assertTrue(np.array_equal(first,model.anchor))
        model.process(seq[4].observed,sharp=True)
        self.assertFalse(np.array_equal(first,model.anchor))

    def test_metric_zero_on_exact_truth(self):
        image=np.random.default_rng(0).integers(0,255,(40,44,3),dtype=np.uint8)
        self.assertGreater(psnr(image,image),110)
        self.assertLess(temporal_delta_mae(image,image,image,image,0,0),1e-7)

    def test_anchor_index_parser(self):
        self.assertEqual(parse_indices("0,2,5-7"),{0,2,5,6,7})
        with self.assertRaises(ValueError):
            parse_indices("-1")


if __name__=="__main__":
    unittest.main()
