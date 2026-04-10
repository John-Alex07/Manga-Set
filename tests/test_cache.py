from __future__ import annotations

import unittest

from framework.cache import prefetch_model_artifacts


class CacheTests(unittest.TestCase):
    def test_prefetch_function_exists(self) -> None:
        self.assertTrue(callable(prefetch_model_artifacts))


if __name__ == "__main__":
    unittest.main()
