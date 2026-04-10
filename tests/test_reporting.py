from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.reporting import flatten_results, summarize_rows, write_csv, write_markdown


class ReportingTests(unittest.TestCase):
    def test_report_rows_export(self) -> None:
        report = {
            "results": [
                {
                    "prompt_id": "scene_1",
                    "metadata": {"backend": "mock", "model_id": "mock-id"},
                    "artifacts": [{"path": "/tmp/a.txt"}],
                    "timings": {"generation": 0.1, "refinement": 0.2},
                    "scores": {
                        "FileIntegrityMetric": 1.0,
                        "LatencyMetric": {"generation": 0.1, "refinement": 0.2},
                    },
                }
            ]
        }
        rows = flatten_results(report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "scene_1")
        self.assertEqual(rows[0]["LatencyMetric.generation"], 0.1)
        self.assertEqual(rows[0]["metadata.backend"], "mock")
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["prompt_id"], "scene_1")
        self.assertEqual(summary[0]["backend"], "mock")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "results.csv"
            md_path = Path(tmp_dir) / "results.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
