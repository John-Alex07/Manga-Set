from __future__ import annotations

import unittest

from framework.registry import METRIC_REGISTRY


class MetricParamsTests(unittest.TestCase):
    def test_metric_registry_passes_params(self) -> None:
        metric = METRIC_REGISTRY.create("latency", {"example": "value"})
        self.assertEqual(metric.params["example"], "value")


if __name__ == "__main__":
    unittest.main()
