from __future__ import annotations

import unittest

from framework.reporting import group_rows


class GroupedReportingTests(unittest.TestCase):
    def test_group_rows_averages_metrics(self) -> None:
        rows = [
            {
                "metadata.suite": "alignment",
                "generation_time": 10.0,
                "FileIntegrityMetric": 1.0,
            },
            {
                "metadata.suite": "alignment",
                "generation_time": 20.0,
                "FileIntegrityMetric": 1.0,
            },
            {
                "metadata.suite": "consistency",
                "generation_time": 30.0,
                "FileIntegrityMetric": 1.0,
            },
        ]
        grouped = group_rows(rows, group_by="metadata.suite")
        self.assertEqual(len(grouped), 2)
        alignment = next(item for item in grouped if item["metadata.suite"] == "alignment")
        self.assertEqual(alignment["count"], 2)
        self.assertEqual(alignment["generation_time.avg"], 15.0)


if __name__ == "__main__":
    unittest.main()
