from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework import ExperimentConfig, ExperimentRunner


class FrameworkTests(unittest.TestCase):
    def test_runner_emits_report_and_images(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "unit_test_experiment",
                "seed": 7,
                "output_dir": "test_outputs",
                "model": {
                    "backend": "mock",
                    "model_id": "mock-test-backend",
                },
                "adapters": [
                    {
                        "type": "style_prompt",
                        "enabled": True,
                        "params": {"style_prompt": "manga screentone"},
                    }
                ],
                "refinement": {
                    "type": "flux",
                    "enabled": True,
                    "params": {"save_copy": True},
                },
                "evaluation": {
                    "metrics": ["file_integrity", "image_statistics", "histogram_consistency"],
                },
                "prompts": [
                    {
                        "id": "prompt_1",
                        "prompt": "heroic manga portrait",
                        "metadata": {"suite": "alignment", "baseline": "unit"},
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = ExperimentRunner(config=config, project_root=Path(tmp_dir))
            report = runner.run()

            result = report["results"][0]
            artifacts = result["artifacts"]
            self.assertTrue(artifacts)
            self.assertTrue(Path(artifacts[0]["path"]).exists())
            self.assertIn("FileIntegrityMetric", result["scores"])
            self.assertIn("HistogramConsistencyMetric", result["scores"])
            self.assertEqual(result["metadata"]["suite"], "alignment")
            self.assertEqual(result["metadata"]["baseline"], "unit")


if __name__ == "__main__":
    unittest.main()
