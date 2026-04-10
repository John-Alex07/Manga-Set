"""Tests for the LPIPS and DINO similarity metrics.

Uses synthetic test images to validate that the metrics are registered,
instantiable, and produce sensible outputs without requiring real model
weights or GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.registry import METRIC_REGISTRY
from framework.types import GeneratedArtifact, GenerationResult


def _create_test_image(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a minimal solid-color PNG for testing."""
    try:
        from PIL import Image
    except ImportError:
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


class LPIPSMetricRegistrationTests(unittest.TestCase):
    def test_lpips_is_registered(self) -> None:
        self.assertIn("lpips_consistency", METRIC_REGISTRY.keys())

    def test_lpips_instantiates(self) -> None:
        metric = METRIC_REGISTRY.create("lpips_consistency", {"device": "cpu"})
        self.assertIsNotNone(metric)

    def test_lpips_returns_one_for_single_image(self) -> None:
        metric = METRIC_REGISTRY.create("lpips_consistency", {"device": "cpu"})
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "single.png"
            _create_test_image(img_path)
            result = GenerationResult(
                prompt_id="test",
                artifacts=[GeneratedArtifact(path=img_path, kind="image")],
            )
            score = metric.evaluate(result, {})
            self.assertEqual(score, 1.0)

    def test_lpips_identical_images_high_similarity(self) -> None:
        try:
            import lpips  # noqa: F401
        except ImportError:
            self.skipTest("lpips not installed")

        metric = METRIC_REGISTRY.create("lpips_consistency", {"device": "cpu"})
        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "img1.png"
            img2 = Path(tmp) / "img2.png"
            _create_test_image(img1, (100, 100, 100))
            _create_test_image(img2, (100, 100, 100))
            result = GenerationResult(
                prompt_id="test",
                artifacts=[GeneratedArtifact(path=img1, kind="image")],
            )
            context = {"reference_images": [str(img2)]}
            score = metric.evaluate(result, context)
            self.assertGreater(score, 0.8)

    def test_lpips_different_images_lower_similarity(self) -> None:
        try:
            import lpips  # noqa: F401
        except ImportError:
            self.skipTest("lpips not installed")

        metric = METRIC_REGISTRY.create("lpips_consistency", {"device": "cpu"})
        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "img1.png"
            img2 = Path(tmp) / "img2.png"
            _create_test_image(img1, (0, 0, 0))
            _create_test_image(img2, (255, 255, 255))
            result = GenerationResult(
                prompt_id="test",
                artifacts=[GeneratedArtifact(path=img1, kind="image")],
            )
            context = {"reference_images": [str(img2)]}
            score = metric.evaluate(result, context)
            self.assertLess(score, 1.0)


class DINOMetricRegistrationTests(unittest.TestCase):
    def test_dino_is_registered(self) -> None:
        self.assertIn("dino_similarity", METRIC_REGISTRY.keys())

    def test_dino_instantiates(self) -> None:
        metric = METRIC_REGISTRY.create("dino_similarity", {"device": "cpu"})
        self.assertIsNotNone(metric)

    def test_dino_returns_one_for_single_image(self) -> None:
        metric = METRIC_REGISTRY.create("dino_similarity", {"device": "cpu"})
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "single.png"
            _create_test_image(img_path)
            result = GenerationResult(
                prompt_id="test",
                artifacts=[GeneratedArtifact(path=img_path, kind="image")],
            )
            score = metric.evaluate(result, {})
            self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
