from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.config import load_experiment_config


class ConfigLoaderTests(unittest.TestCase):
    def test_config_loader_expands_prompt_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prompts_path = root / "suite.json"
            prompts_path.write_text(
                '[{"id":"p1","prompt":"prompt from suite"},{"id":"p2","prompt":"second prompt"}]',
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "name": "suite_test",
                  "model": {"backend": "mock", "model_id": "mock"},
                  "prompt_sources": ["suite.json"],
                  "prompts": [{"id":"inline","prompt":"inline prompt"}]
                }
                """,
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            self.assertEqual(config.prompt_sources, ["suite.json"])
            self.assertEqual(len(config.prompts), 3)
            self.assertEqual(config.prompts[0].id, "inline")
            self.assertEqual(config.prompts[1].id, "p1")

    def test_legacy_lora_style_prompt_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "name": "legacy_lora_test",
                  "model": {"backend": "mock", "model_id": "mock"},
                  "adapters": [
                    {
                      "type": "lora",
                      "params": {"style_prompt": "manga screentone"}
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            with self.assertWarns(DeprecationWarning):
                config = load_experiment_config(config_path)
            self.assertEqual(config.adapters[0].type, "style_prompt")
            self.assertEqual(config.adapters[0].params["_legacy_alias"], "lora")

    def test_config_loader_filters_prompt_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prompts_path = root / "suite.json"
            prompts_path.write_text(
                """
                [
                  {"id":"a1","prompt":"a1","metadata":{"suite":"alignment"}},
                  {"id":"c1","prompt":"c1","metadata":{"suite":"consistency"}},
                  {"id":"s1","prompt":"s1","metadata":{"suite":"story"}}
                ]
                """,
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "name": "suite_filter_test",
                  "model": {"backend": "mock", "model_id": "mock"},
                  "prompt_sources": ["suite.json"],
                  "prompt_filters": {"suites": ["story", "alignment"], "limit": 2}
                }
                """,
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            self.assertEqual(len(config.prompts), 2)
            self.assertEqual(config.prompts[0].id, "a1")
            self.assertEqual(config.prompts[1].id, "s1")


if __name__ == "__main__":
    unittest.main()
