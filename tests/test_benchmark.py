import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from videorestoration.benchmark import run_benchmark, ARMS


class IntegrationGates(unittest.TestCase):
    def test_tiny_sealed_gate_writes_scores_and_images(self):
        with TemporaryDirectory() as d:
            report=run_benchmark(d,frames=9,height=48,width=64,seeds=(101,))
            self.assertEqual(report["protocol"],"video-restoration-gate1-v1")
            self.assertEqual(set(report["arms"]),set(ARMS))
            self.assertGreater(report["estimated_peak_state_bytes"],0)
            for k in ARMS:
                self.assertGreater(report["arms"][k]["n_damaged"],0)
                self.assertTrue(0<=report["arms"][k]["damaged_mean_ssim"]<=1)
            self.assertTrue((Path(d)/"per_frame.csv").exists())
            self.assertTrue((Path(d)/"report.json").exists())
            self.assertTrue((Path(d)/"comparison.png").exists())
            saved=json.loads((Path(d)/"report.json").read_text())
            self.assertFalse(saved["oracle_motion_diagnostic"])

if __name__=="__main__":
    unittest.main()
