from __future__ import annotations

import unittest

from framework.analysis import build_consistency_rows, infer_consistency_group


class ConsistencyAnalysisTests(unittest.TestCase):
    def test_infer_consistency_group_from_prompt_id(self) -> None:
        row = {"prompt_id": "consistency_hiro_panel_2"}
        self.assertEqual(infer_consistency_group(row), "consistency_hiro_panel")

    def test_build_consistency_rows_groups_rows(self) -> None:
        rows = [
            {"experiment_name": "exp", "prompt_id": "consistency_hiro_panel_1", "artifacts.0.path": "/tmp/a.png"},
            {"experiment_name": "exp", "prompt_id": "consistency_hiro_panel_2", "artifacts.0.path": "/tmp/b.png"},
        ]
        grouped = build_consistency_rows(rows)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
