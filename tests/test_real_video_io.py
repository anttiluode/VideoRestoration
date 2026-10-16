"""Practical CLI smoke test; skips only when runner has no MP4 codec."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np
from videorestoration.cli import restore_video
from videorestoration.synthetic import make_sequence


class RealVideoIOTests(unittest.TestCase):
    def test_mp4_reader_writer_with_explicit_past_anchors(self):
        with TemporaryDirectory() as directory:
            d=Path(directory)
            source=d/"source.mp4"
            dest=d/"restored.mp4"
            writer=cv2.VideoWriter(str(source),
                                   cv2.VideoWriter_fourcc(*"mp4v"),12.0,(80,64))
            if not writer.isOpened():
                self.skipTest("runner has no MP4 encoder")
            sequence=make_sequence(30,frames=7,width=80,height=64,anchor_every=3,
                                   scene_at=None)
            for frame in sequence:
                writer.write(frame.observed)
            writer.release()
            summary=restore_video(str(source),str(dest),{0,3,6},
                                  "sihti_select",True,None,0.9,0.23)
            self.assertEqual(summary["frames"],7)
            self.assertEqual(summary["anchors_used"],[0,3,6])
            self.assertTrue(dest.exists())
            cap=cv2.VideoCapture(str(dest))
            ok,first=cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(first.shape[:2],(64,160))
            self.assertEqual(summary["quality_metrics"],
                             "unavailable (no clean ground-truth video supplied)")


if __name__=="__main__":
    unittest.main()
