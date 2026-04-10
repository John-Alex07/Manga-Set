"""Tests that protect the benchmark protocol invariants the paper depends on.

These tests ensure that ablation configs share identical prompt IDs,
per-prompt seeds, guidance scales, and inference step counts so that
cross-condition comparisons are properly controlled.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

QUINTET_CONFIGS = [
    "benchmark_storyboard_sd21_quintet_base_clip.json",
    "benchmark_storyboard_sd21_quintet_clip.json",
    "benchmark_storyboard_sd21_true_lora_quintet_clip.json",
    "benchmark_storyboard_sd21_true_lora_cartoon_quintet_clip.json",
]

CONSISTENCY_CONFIGS = [
    "benchmark_storyboard_sd21_consistency_base_clip.json",
    "benchmark_storyboard_sd21_style_consistency_clip.json",
    "benchmark_storyboard_sd21_true_lora_planogram_consistency_clip.json",
    "benchmark_storyboard_sd21_true_lora_cartoon_consistency_clip.json",
]

CANONICAL_SEEDS = {
    "align_rooftop_duel": 1001,
    "align_library_dialogue": 1002,
    "consistency_hiro_panel_1": 1003,
    "consistency_hiro_panel_2": 1004,
    "story_panel_1_setup": 1005,
    "consistency_akira_panel_1": 1006,
    "consistency_akira_panel_2": 1007,
}

EXPANDED_CONDITIONS = ["base", "style_prompt", "lora_planogram", "lora_cartoon"]
EXPANDED_TRIALS = [0, 1, 2]
EXPANDED_TRIAL_SEEDS = [42, 137, 256]
EXPANDED_PROMPT_COUNT = 30


def _load_config(name: str) -> dict:
    path = CONFIGS_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prompt_seed_map(config: dict) -> dict[str, int]:
    return {p["id"]: p["seed"] for p in config["prompts"]}


def _prompt_ids(config: dict) -> list[str]:
    return [p["id"] for p in config["prompts"]]


class QuintetProtocolTests(unittest.TestCase):
    """Ensure the four quintet ablation configs are properly controlled."""

    def test_all_configs_exist(self) -> None:
        for name in QUINTET_CONFIGS:
            self.assertTrue(
                (CONFIGS_DIR / name).exists(),
                f"Missing config: {name}",
            )

    def test_prompt_ids_match_across_conditions(self) -> None:
        configs = [_load_config(n) for n in QUINTET_CONFIGS]
        reference_ids = _prompt_ids(configs[0])
        for i, cfg in enumerate(configs[1:], start=1):
            self.assertEqual(
                _prompt_ids(cfg),
                reference_ids,
                f"{QUINTET_CONFIGS[i]} has different prompt IDs than {QUINTET_CONFIGS[0]}",
            )

    def test_per_prompt_seeds_match_canonical(self) -> None:
        for name in QUINTET_CONFIGS:
            cfg = _load_config(name)
            seed_map = _prompt_seed_map(cfg)
            for prompt_id, expected_seed in seed_map.items():
                self.assertEqual(
                    expected_seed,
                    CANONICAL_SEEDS[prompt_id],
                    f"{name}: prompt {prompt_id} has seed {expected_seed}, expected {CANONICAL_SEEDS[prompt_id]}",
                )

    def test_seeds_are_identical_across_conditions(self) -> None:
        configs = [_load_config(n) for n in QUINTET_CONFIGS]
        reference_seeds = _prompt_seed_map(configs[0])
        for i, cfg in enumerate(configs[1:], start=1):
            self.assertEqual(
                _prompt_seed_map(cfg),
                reference_seeds,
                f"{QUINTET_CONFIGS[i]} has different seeds than {QUINTET_CONFIGS[0]}",
            )

    def test_guidance_scale_matches_across_conditions(self) -> None:
        configs = [_load_config(n) for n in QUINTET_CONFIGS]
        ref = {p["id"]: p["guidance_scale"] for p in configs[0]["prompts"]}
        for i, cfg in enumerate(configs[1:], start=1):
            current = {p["id"]: p["guidance_scale"] for p in cfg["prompts"]}
            self.assertEqual(
                current,
                ref,
                f"{QUINTET_CONFIGS[i]} has different guidance_scale values",
            )

    def test_inference_steps_match_across_conditions(self) -> None:
        configs = [_load_config(n) for n in QUINTET_CONFIGS]
        ref = {p["id"]: p["num_inference_steps"] for p in configs[0]["prompts"]}
        for i, cfg in enumerate(configs[1:], start=1):
            current = {p["id"]: p["num_inference_steps"] for p in cfg["prompts"]}
            self.assertEqual(
                current,
                ref,
                f"{QUINTET_CONFIGS[i]} has different num_inference_steps values",
            )

    def test_all_configs_have_consistent_local_files_only(self) -> None:
        for name in QUINTET_CONFIGS:
            cfg = _load_config(name)
            clip_params = cfg.get("evaluation", {}).get("params", {}).get("clip_text_alignment", {})
            self.assertIn(
                "local_files_only",
                clip_params,
                f"{name}: clip_text_alignment should specify local_files_only",
            )


class ConsistencyProtocolTests(unittest.TestCase):
    """Ensure the four consistency ablation configs are properly controlled."""

    def test_all_configs_exist(self) -> None:
        for name in CONSISTENCY_CONFIGS:
            self.assertTrue(
                (CONFIGS_DIR / name).exists(),
                f"Missing config: {name}",
            )

    def test_prompt_ids_match_across_conditions(self) -> None:
        configs = [_load_config(n) for n in CONSISTENCY_CONFIGS]
        reference_ids = _prompt_ids(configs[0])
        for i, cfg in enumerate(configs[1:], start=1):
            self.assertEqual(
                _prompt_ids(cfg),
                reference_ids,
                f"{CONSISTENCY_CONFIGS[i]} has different prompt IDs",
            )

    def test_per_prompt_seeds_match_canonical(self) -> None:
        for name in CONSISTENCY_CONFIGS:
            cfg = _load_config(name)
            seed_map = _prompt_seed_map(cfg)
            for prompt_id, expected_seed in seed_map.items():
                self.assertEqual(
                    expected_seed,
                    CANONICAL_SEEDS[prompt_id],
                    f"{name}: prompt {prompt_id} has seed {expected_seed}, expected {CANONICAL_SEEDS[prompt_id]}",
                )

    def test_seeds_are_identical_across_conditions(self) -> None:
        configs = [_load_config(n) for n in CONSISTENCY_CONFIGS]
        reference_seeds = _prompt_seed_map(configs[0])
        for i, cfg in enumerate(configs[1:], start=1):
            self.assertEqual(
                _prompt_seed_map(cfg),
                reference_seeds,
                f"{CONSISTENCY_CONFIGS[i]} has different seeds",
            )


class AdapterTypeTests(unittest.TestCase):
    """Ensure no config silently mislabels adapter types."""

    def test_no_config_uses_lora_type_without_weights(self) -> None:
        for path in CONFIGS_DIR.glob("benchmark_storyboard_sd21_*.json"):
            cfg = _load_config(path.name)
            for adapter in cfg.get("adapters", []):
                if adapter.get("type") == "lora":
                    params = adapter.get("params", {})
                    has_weights = "weights_path" in params or "lora_path" in params
                    self.assertTrue(
                        has_weights,
                        f"{path.name}: adapter type='lora' must have weights_path or lora_path",
                    )


class ExpandedProtocolTests(unittest.TestCase):
    """Ensure the 12 expanded ablation configs are properly controlled."""

    def _config_name(self, condition: str, trial: int) -> str:
        return f"expanded_{condition}_t{trial}.json"

    def test_all_expanded_configs_exist(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                name = self._config_name(cond, trial)
                self.assertTrue(
                    (CONFIGS_DIR / name).exists(),
                    f"Missing expanded config: {name}",
                )

    def test_expanded_prompt_count(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                cfg = _load_config(self._config_name(cond, trial))
                self.assertEqual(
                    len(cfg["prompts"]),
                    EXPANDED_PROMPT_COUNT,
                    f"{self._config_name(cond, trial)}: expected {EXPANDED_PROMPT_COUNT} prompts",
                )

    def test_expanded_prompt_ids_match_across_conditions(self) -> None:
        for trial in EXPANDED_TRIALS:
            configs = [
                _load_config(self._config_name(cond, trial))
                for cond in EXPANDED_CONDITIONS
            ]
            reference_ids = _prompt_ids(configs[0])
            for i, cfg in enumerate(configs[1:], start=1):
                self.assertEqual(
                    _prompt_ids(cfg),
                    reference_ids,
                    f"Trial {trial}: {EXPANDED_CONDITIONS[i]} has different prompt IDs than {EXPANDED_CONDITIONS[0]}",
                )

    def test_expanded_seeds_differ_across_trials(self) -> None:
        cond = EXPANDED_CONDITIONS[0]
        trial_seed_maps = [
            _prompt_seed_map(_load_config(self._config_name(cond, t)))
            for t in EXPANDED_TRIALS
        ]
        for i in range(len(trial_seed_maps)):
            for j in range(i + 1, len(trial_seed_maps)):
                self.assertNotEqual(
                    trial_seed_maps[i],
                    trial_seed_maps[j],
                    f"Trial {i} and trial {j} have identical seed maps (should differ)",
                )

    def test_expanded_seeds_match_across_conditions_within_trial(self) -> None:
        for trial in EXPANDED_TRIALS:
            configs = [
                _load_config(self._config_name(cond, trial))
                for cond in EXPANDED_CONDITIONS
            ]
            reference_seeds = _prompt_seed_map(configs[0])
            for i, cfg in enumerate(configs[1:], start=1):
                self.assertEqual(
                    _prompt_seed_map(cfg),
                    reference_seeds,
                    f"Trial {trial}: {EXPANDED_CONDITIONS[i]} has different seeds than {EXPANDED_CONDITIONS[0]}",
                )

    def test_expanded_guidance_scale_matches(self) -> None:
        for trial in EXPANDED_TRIALS:
            configs = [
                _load_config(self._config_name(cond, trial))
                for cond in EXPANDED_CONDITIONS
            ]
            ref = {p["id"]: p["guidance_scale"] for p in configs[0]["prompts"]}
            for i, cfg in enumerate(configs[1:], start=1):
                current = {p["id"]: p["guidance_scale"] for p in cfg["prompts"]}
                self.assertEqual(
                    current,
                    ref,
                    f"Trial {trial}: {EXPANDED_CONDITIONS[i]} has different guidance_scale",
                )

    def test_expanded_inference_steps_match(self) -> None:
        for trial in EXPANDED_TRIALS:
            configs = [
                _load_config(self._config_name(cond, trial))
                for cond in EXPANDED_CONDITIONS
            ]
            ref = {p["id"]: p["num_inference_steps"] for p in configs[0]["prompts"]}
            for i, cfg in enumerate(configs[1:], start=1):
                current = {p["id"]: p["num_inference_steps"] for p in cfg["prompts"]}
                self.assertEqual(
                    current,
                    ref,
                    f"Trial {trial}: {EXPANDED_CONDITIONS[i]} has different num_inference_steps",
                )

    def test_expanded_lora_configs_have_weights(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                cfg = _load_config(self._config_name(cond, trial))
                for adapter in cfg.get("adapters", []):
                    if adapter.get("type") == "lora":
                        params = adapter.get("params", {})
                        has_weights = "weights_path" in params or "lora_path" in params
                        self.assertTrue(
                            has_weights,
                            f"{self._config_name(cond, trial)}: lora adapter must have weights",
                        )

    def test_expanded_configs_have_consistent_local_files_only(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                name = self._config_name(cond, trial)
                cfg = _load_config(name)
                clip_params = cfg.get("evaluation", {}).get("params", {}).get("clip_text_alignment", {})
                self.assertIn(
                    "local_files_only",
                    clip_params,
                    f"{name}: clip_text_alignment should specify local_files_only",
                )
                dino_params = cfg.get("evaluation", {}).get("params", {}).get("dino_similarity", {})
                self.assertIn(
                    "local_files_only",
                    dino_params,
                    f"{name}: dino_similarity should specify local_files_only",
                )

    def test_expanded_trial_seed_arithmetic(self) -> None:
        """Verify each prompt seed = base_seed + trial_seed for every config."""
        base_seeds = {
            "align_rooftop_duel": 1001, "align_library_dialogue": 1002,
            "align_alley_chase": 1003, "align_market_crowd": 1004,
            "align_classroom_reveal": 1005, "align_bridge_standoff": 1006,
            "align_dojo_training": 1007,
            "consistency_hiro_panel_1": 1101, "consistency_hiro_panel_2": 1102,
            "consistency_hiro_panel_3": 1103, "consistency_akira_panel_1": 1104,
            "consistency_akira_panel_2": 1105, "consistency_akira_panel_3": 1106,
            "consistency_yuki_panel_1": 1107, "consistency_yuki_panel_2": 1108,
            "consistency_yuki_panel_3": 1109, "consistency_kenji_panel_1": 1110,
            "consistency_kenji_panel_2": 1111,
            "story_panel_1_setup": 1201, "story_panel_2_confrontation": 1202,
            "story_panel_3_action": 1203, "story_cafe_1_arrival": 1204,
            "story_cafe_2_recognition": 1205, "story_cafe_3_conversation": 1206,
            "style_screentone_gradient": 1301, "style_speed_lines": 1302,
            "style_chibi_emotion": 1303, "style_dramatic_shadow": 1304,
            "style_shoujo_sparkle": 1305, "style_seinen_detail": 1306,
        }
        for cond in EXPANDED_CONDITIONS:
            for trial_index, trial_seed in enumerate(EXPANDED_TRIAL_SEEDS):
                name = self._config_name(cond, trial_index)
                cfg = _load_config(name)
                seed_map = _prompt_seed_map(cfg)
                for prompt_id, actual_seed in seed_map.items():
                    expected = base_seeds[prompt_id] + trial_seed
                    self.assertEqual(
                        actual_seed,
                        expected,
                        f"{name}: {prompt_id} seed {actual_seed} != base {base_seeds[prompt_id]} + trial {trial_seed} = {expected}",
                    )

    def test_expanded_model_has_revision_pin(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                name = self._config_name(cond, trial)
                cfg = _load_config(name)
                revision = cfg.get("model", {}).get("revision")
                self.assertTrue(
                    revision and len(revision) >= 7,
                    f"{name}: model must have a revision SHA pin",
                )

    def test_expanded_evaluation_models_have_revision_pins(self) -> None:
        for cond in EXPANDED_CONDITIONS:
            for trial in EXPANDED_TRIALS:
                name = self._config_name(cond, trial)
                cfg = _load_config(name)
                params = cfg.get("evaluation", {}).get("params", {})
                clip_rev = params.get("clip_text_alignment", {}).get("revision")
                self.assertTrue(
                    clip_rev and len(clip_rev) >= 7,
                    f"{name}: clip_text_alignment must have a revision SHA pin",
                )
                dino_rev = params.get("dino_similarity", {}).get("revision")
                self.assertTrue(
                    dino_rev and len(dino_rev) >= 7,
                    f"{name}: dino_similarity must have a revision SHA pin",
                )

    def test_expanded_lora_adapters_have_revision_pins(self) -> None:
        for cond in ["lora_planogram", "lora_cartoon"]:
            for trial in EXPANDED_TRIALS:
                name = self._config_name(cond, trial)
                cfg = _load_config(name)
                for adapter in cfg.get("adapters", []):
                    if adapter.get("type") == "lora":
                        rev = adapter.get("params", {}).get("revision")
                        self.assertTrue(
                            rev and len(rev) >= 7,
                            f"{name}: LoRA adapter must have a revision SHA pin",
                        )


class ConsistencyGuidanceTests(unittest.TestCase):
    """Ensure consistency configs have matching guidance and steps across conditions."""

    def test_consistency_guidance_scale_matches(self) -> None:
        configs = [_load_config(n) for n in CONSISTENCY_CONFIGS]
        ref = {p["id"]: p["guidance_scale"] for p in configs[0]["prompts"]}
        for i, cfg in enumerate(configs[1:], start=1):
            current = {p["id"]: p["guidance_scale"] for p in cfg["prompts"]}
            self.assertEqual(
                current,
                ref,
                f"{CONSISTENCY_CONFIGS[i]} has different guidance_scale values",
            )

    def test_consistency_inference_steps_match(self) -> None:
        configs = [_load_config(n) for n in CONSISTENCY_CONFIGS]
        ref = {p["id"]: p["num_inference_steps"] for p in configs[0]["prompts"]}
        for i, cfg in enumerate(configs[1:], start=1):
            current = {p["id"]: p["num_inference_steps"] for p in cfg["prompts"]}
            self.assertEqual(
                current,
                ref,
                f"{CONSISTENCY_CONFIGS[i]} has different num_inference_steps values",
            )


# ---------------------------------------------------------------------------
# SD 2.1 @ 20 steps protocol tests
# ---------------------------------------------------------------------------

SD21_S20_CONDITIONS = ["base", "style_prompt", "lora_planogram", "lora_cartoon"]
SD21_S20_TRIALS = [0, 1, 2]


class SD21_20StepProtocolTests(unittest.TestCase):
    """Ensure SD 2.1 @ 20-step configs are consistent."""

    def _name(self, condition: str, trial: int) -> str:
        return f"sd21_s20_{condition}_t{trial}.json"

    def test_all_configs_exist(self) -> None:
        for cond in SD21_S20_CONDITIONS:
            for trial in SD21_S20_TRIALS:
                name = self._name(cond, trial)
                self.assertTrue((CONFIGS_DIR / name).exists(), f"Missing: {name}")

    def test_prompt_count(self) -> None:
        for cond in SD21_S20_CONDITIONS:
            for trial in SD21_S20_TRIALS:
                cfg = _load_config(self._name(cond, trial))
                self.assertEqual(len(cfg["prompts"]), 30, self._name(cond, trial))

    def test_all_prompts_use_20_steps(self) -> None:
        for cond in SD21_S20_CONDITIONS:
            for trial in SD21_S20_TRIALS:
                cfg = _load_config(self._name(cond, trial))
                for p in cfg["prompts"]:
                    self.assertEqual(p["num_inference_steps"], 20, f"{p['id']} in {self._name(cond, trial)}")

    def test_seeds_match_across_conditions_within_trial(self) -> None:
        for trial in SD21_S20_TRIALS:
            configs = [_load_config(self._name(c, trial)) for c in SD21_S20_CONDITIONS]
            ref = _prompt_seed_map(configs[0])
            for i, cfg in enumerate(configs[1:], start=1):
                self.assertEqual(_prompt_seed_map(cfg), ref)

    def test_step_count_parity_with_4step(self) -> None:
        """4-step and 20-step runs share prompt IDs and per-trial seeds."""
        for trial in SD21_S20_TRIALS:
            cfg_4 = _load_config(f"expanded_base_t{trial}.json")
            cfg_20 = _load_config(self._name("base", trial))
            self.assertEqual(_prompt_ids(cfg_4), _prompt_ids(cfg_20))
            self.assertEqual(_prompt_seed_map(cfg_4), _prompt_seed_map(cfg_20))

    def test_model_has_revision_pin(self) -> None:
        for cond in SD21_S20_CONDITIONS:
            cfg = _load_config(self._name(cond, 0))
            rev = cfg["model"].get("revision", "")
            self.assertTrue(len(rev) >= 7, f"{self._name(cond, 0)}: missing revision pin")


# ---------------------------------------------------------------------------
# SDXL Turbo protocol tests
# ---------------------------------------------------------------------------

SDXL_CONDITIONS = ["base", "style_prompt", "lora_anime", "lora_manga"]
SDXL_TRIALS = [0, 1, 2]


class SDXLTurboProtocolTests(unittest.TestCase):
    """Ensure SDXL Turbo configs are properly controlled."""

    def _name(self, condition: str, trial: int) -> str:
        return f"sdxl_{condition}_t{trial}.json"

    def test_all_configs_exist(self) -> None:
        for cond in SDXL_CONDITIONS:
            for trial in SDXL_TRIALS:
                name = self._name(cond, trial)
                self.assertTrue((CONFIGS_DIR / name).exists(), f"Missing: {name}")

    def test_prompt_count(self) -> None:
        for cond in SDXL_CONDITIONS:
            cfg = _load_config(self._name(cond, 0))
            self.assertEqual(len(cfg["prompts"]), 30)

    def test_guidance_scale_is_zero(self) -> None:
        for cond in SDXL_CONDITIONS:
            cfg = _load_config(self._name(cond, 0))
            for p in cfg["prompts"]:
                self.assertEqual(p["guidance_scale"], 0.0, f"{p['id']} guidance_scale != 0")

    def test_uses_euler_ancestral_scheduler(self) -> None:
        for cond in SDXL_CONDITIONS:
            cfg = _load_config(self._name(cond, 0))
            self.assertEqual(cfg["model"]["params"]["scheduler"], "euler_ancestral")

    def test_seeds_match_across_conditions(self) -> None:
        for trial in SDXL_TRIALS:
            configs = [_load_config(self._name(c, trial)) for c in SDXL_CONDITIONS]
            ref = _prompt_seed_map(configs[0])
            for cfg in configs[1:]:
                self.assertEqual(_prompt_seed_map(cfg), ref)

    def test_model_has_revision_pin(self) -> None:
        cfg = _load_config(self._name("base", 0))
        rev = cfg["model"].get("revision", "")
        self.assertTrue(len(rev) >= 7, "SDXL Turbo model missing revision pin")

    def test_sdxl_lora_adapters_have_revision_pins(self) -> None:
        for cond in ["lora_anime", "lora_manga"]:
            cfg = _load_config(self._name(cond, 0))
            for adapter in cfg.get("adapters", []):
                if adapter.get("type") == "lora":
                    rev = adapter.get("params", {}).get("revision", "")
                    self.assertTrue(len(rev) >= 7, f"SDXL {cond}: LoRA missing revision pin")


# ---------------------------------------------------------------------------
# Flux Schnell protocol tests
# ---------------------------------------------------------------------------


class FluxSchnellProtocolTests(unittest.TestCase):
    """Ensure Flux Schnell configs are properly structured."""

    def test_configs_exist(self) -> None:
        for cond in ["base", "style_prompt"]:
            name = f"flux_{cond}_t0.json"
            self.assertTrue((CONFIGS_DIR / name).exists(), f"Missing: {name}")

    def test_prompt_count(self) -> None:
        cfg = _load_config("flux_base_t0.json")
        self.assertEqual(len(cfg["prompts"]), 10)

    def test_no_negative_prompts(self) -> None:
        cfg = _load_config("flux_base_t0.json")
        for p in cfg["prompts"]:
            self.assertEqual(p.get("negative_prompt", ""), "", f"{p['id']} has negative_prompt")

    def test_guidance_scale_is_zero(self) -> None:
        cfg = _load_config("flux_base_t0.json")
        for p in cfg["prompts"]:
            self.assertEqual(p["guidance_scale"], 0.0)

    def test_model_uses_bfloat16(self) -> None:
        cfg = _load_config("flux_base_t0.json")
        self.assertEqual(cfg["model"]["dtype"], "bfloat16")

    def test_model_has_revision_pin(self) -> None:
        cfg = _load_config("flux_base_t0.json")
        rev = cfg["model"].get("revision", "")
        self.assertTrue(len(rev) >= 7, "Flux Schnell model missing revision pin")


# ---------------------------------------------------------------------------
# ControlNet protocol tests
# ---------------------------------------------------------------------------


class ControlNetProtocolTests(unittest.TestCase):
    """Ensure SD 2.1 + ControlNet Canny configs are correct."""

    def _name(self, trial: int) -> str:
        return f"sd21_controlnet_canny_s20_t{trial}.json"

    def test_all_configs_exist(self) -> None:
        for trial in [0, 1, 2]:
            name = self._name(trial)
            self.assertTrue((CONFIGS_DIR / name).exists(), f"Missing: {name}")

    def test_prompt_count(self) -> None:
        cfg = _load_config(self._name(0))
        self.assertEqual(len(cfg["prompts"]), 30)

    def test_uses_20_steps(self) -> None:
        cfg = _load_config(self._name(0))
        for p in cfg["prompts"]:
            self.assertEqual(p["num_inference_steps"], 20)

    def test_controlnet_adapter_present(self) -> None:
        cfg = _load_config(self._name(0))
        types = [a["type"] for a in cfg.get("adapters", [])]
        self.assertIn("controlnet", types)

    def test_model_has_controlnet_id(self) -> None:
        cfg = _load_config(self._name(0))
        cn_id = cfg["model"]["params"].get("controlnet_id", "")
        self.assertTrue(len(cn_id) > 0, "ControlNet model ID not set")

    def test_controlnet_revision_pinned(self) -> None:
        cfg = _load_config(self._name(0))
        cn_rev = cfg["model"]["params"].get("controlnet_revision", "")
        self.assertTrue(len(cn_rev) >= 7, "ControlNet revision not pinned")

    def test_prompts_have_canny_controls(self) -> None:
        cfg = _load_config(self._name(0))
        for p in cfg["prompts"]:
            controls = p.get("controls", {})
            self.assertEqual(controls.get("type"), "canny", f"{p['id']} missing canny controls")


if __name__ == "__main__":
    unittest.main()
