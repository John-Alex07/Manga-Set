from __future__ import annotations

import unittest

from framework.reporting import group_rows


class CompareReportingTests(unittest.TestCase):
    def test_group_rows_supports_multiple_group_columns(self) -> None:
        rows = [
            {"experiment_name": "base", "metadata.suite": "alignment", "generation_time": 10.0},
            {"experiment_name": "base", "metadata.suite": "alignment", "generation_time": 14.0},
            {"experiment_name": "styled", "metadata.suite": "alignment", "generation_time": 20.0},
        ]
        grouped = group_rows(rows, group_by=["experiment_name", "metadata.suite"])
        self.assertEqual(len(grouped), 2)
        base_alignment = next(item for item in grouped if item["experiment_name"] == "base")
        self.assertEqual(base_alignment["generation_time.avg"], 12.0)


if __name__ == "__main__":
    unittest.main()
