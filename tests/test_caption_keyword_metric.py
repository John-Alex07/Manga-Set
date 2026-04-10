from __future__ import annotations

import unittest

from framework.evaluation.metrics import CaptionKeywordRecallMetric


class CaptionKeywordMetricTests(unittest.TestCase):
    def test_extract_keywords_discards_stopwords(self) -> None:
        keywords = CaptionKeywordRecallMetric._extract_keywords(
            "A manga panel with a boy on a rooftop at sunset"
        )
        self.assertIn("manga", keywords)
        self.assertIn("rooftop", keywords)
        self.assertNotIn("with", keywords)
        self.assertNotIn("the", keywords)


if __name__ == "__main__":
    unittest.main()
