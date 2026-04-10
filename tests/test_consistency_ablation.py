from __future__ import annotations

import unittest

from framework.analysis import label_consistency_rows


class ConsistencyAblationTests(unittest.TestCase):
    def test_label_consistency_rows(self) -> None:
        rows = [
            {
                "condition_label": "base",
                "consistency_group": "hiro",
                "count": 2,
                "clip_image_similarity.avg": 0.8,
            }
        ]
        labelled = label_consistency_rows(rows)
        self.assertEqual(labelled[0]["condition"], "base")
        self.assertEqual(labelled[0]["consistency_group"], "hiro")
        self.assertEqual(labelled[0]["clip_image_similarity.avg"], 0.8)


if __name__ == "__main__":
    unittest.main()
