from __future__ import annotations

import unittest

from framework.reporting import format_ablation_rows


class AblationOverallTests(unittest.TestCase):
    def test_format_ablation_rows_adds_overall_rows(self) -> None:
        rows = [
            {
                "condition_label": "base",
                "metadata.suite": "alignment",
                "generation_time": 10.0,
                "CLIPTextAlignmentMetric": 0.3,
                "ImageStatisticsMetric.mean_brightness": 120.0,
            },
            {
                "condition_label": "base",
                "metadata.suite": "consistency",
                "generation_time": 20.0,
                "CLIPTextAlignmentMetric": 0.5,
                "ImageStatisticsMetric.mean_brightness": 140.0,
            },
        ]
        table = format_ablation_rows(rows, label_column="condition_label")
        overall = next(item for item in table if item["suite"] == "overall")
        self.assertEqual(overall["condition"], "base")
        self.assertEqual(overall["generation_time.avg"], 15.0)
        self.assertEqual(overall["CLIPTextAlignmentMetric.avg"], 0.4)


if __name__ == "__main__":
    unittest.main()
