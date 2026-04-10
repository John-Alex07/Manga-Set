from __future__ import annotations

import unittest

from framework.reporting import format_ablation_rows


class AblationReportingTests(unittest.TestCase):
    def test_format_ablation_rows(self) -> None:
        rows = [
            {
                "condition_label": "base",
                "metadata.suite": "alignment",
                "generation_time": 10.0,
                "CLIPTextAlignmentMetric": 0.3,
                "CaptionKeywordRecallMetric": 0.1,
                "ImageStatisticsMetric.mean_brightness": 120.0,
            },
            {
                "condition_label": "base",
                "metadata.suite": "alignment",
                "generation_time": 14.0,
                "CLIPTextAlignmentMetric": 0.5,
                "CaptionKeywordRecallMetric": 0.2,
                "ImageStatisticsMetric.mean_brightness": 140.0,
            },
        ]
        table = format_ablation_rows(rows, label_column="condition_label")
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]["condition"], "base")
        self.assertEqual(table[0]["suite"], "alignment")
        self.assertEqual(table[0]["generation_time.avg"], 12.0)
        self.assertEqual(table[0]["CLIPTextAlignmentMetric.avg"], 0.4)
        self.assertEqual(table[1]["suite"], "overall")


if __name__ == "__main__":
    unittest.main()
