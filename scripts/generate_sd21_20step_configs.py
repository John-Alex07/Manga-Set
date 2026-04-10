"""Generate SD 2.1 configs with 20 inference steps for quality comparison.

Reuses the same 30 prompts, 4 conditions, and 3 trials as the 4-step study,
but with num_inference_steps=20 to measure the effect of step count on
relative condition rankings.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

SUITE_FILES = [
    "alignment_suite.json",
    "consistency_suite.json",
    "panel_story_suite.json",
    "style_fidelity_suite.json",
]

TRIAL_SEEDS = [42, 137, 256]
NUM_INFERENCE_STEPS = 20

BASE_SEED_MAP = {
    "align_rooftop_duel": 1001,
    "align_library_dialogue": 1002,
    "align_alley_chase": 1003,
    "align_market_crowd": 1004,
    "align_classroom_reveal": 1005,
    "align_bridge_standoff": 1006,
    "align_dojo_training": 1007,
    "consistency_hiro_panel_1": 1101,
    "consistency_hiro_panel_2": 1102,
    "consistency_hiro_panel_3": 1103,
    "consistency_akira_panel_1": 1104,
    "consistency_akira_panel_2": 1105,
    "consistency_akira_panel_3": 1106,
    "consistency_yuki_panel_1": 1107,
    "consistency_yuki_panel_2": 1108,
    "consistency_yuki_panel_3": 1109,
    "consistency_kenji_panel_1": 1110,
    "consistency_kenji_panel_2": 1111,
    "story_panel_1_setup": 1201,
    "story_panel_2_confrontation": 1202,
    "story_panel_3_action": 1203,
    "story_cafe_1_arrival": 1204,
    "story_cafe_2_recognition": 1205,
    "story_cafe_3_conversation": 1206,
    "style_screentone_gradient": 1301,
    "style_speed_lines": 1302,
    "style_chibi_emotion": 1303,
    "style_dramatic_shadow": 1304,
    "style_shoujo_sparkle": 1305,
    "style_seinen_detail": 1306,
}

MODEL_CONFIG = {
    "backend": "diffusers",
    "model_id": "sd-research/stable-diffusion-2-1-base",
    "device": "cpu",
    "dtype": "float32",
    "revision": "0708cecd370b4d1c3a6ff3f7332f5e9aea78896f",
    "params": {
        "use_safetensors": True,
        "scheduler": "euler",
        "local_files_only": True,
        "disable_safety_checker": True,
    },
}

EVALUATION_CONFIG = {
    "metrics": [
        "file_integrity",
        "image_statistics",
        "latency",
        "clip_text_alignment",
        "image_reward",
        "lpips_consistency",
        "dino_similarity",
    ],
    "params": {
        "clip_text_alignment": {
            "model_id": "openai/clip-vit-base-patch32",
            "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "device": "cpu",
            "local_files_only": True,
        },
        "lpips_consistency": {"device": "cpu"},
        "dino_similarity": {
            "model_id": "facebook/dino-vits16",
            "revision": "abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd",
            "device": "cpu",
            "local_files_only": True,
        },
    },
}

CONDITIONS: dict[str, dict] = {
    "base": {"label": "Base (no adapter)", "adapters": []},
    "style_prompt": {
        "label": "Style Prompt conditioning",
        "adapters": [{
            "type": "style_prompt", "enabled": True, "weight": 0.8,
            "params": {"style_prompt": "black and white manga screentone, cinematic line art"},
        }],
    },
    "lora_planogram": {
        "label": "LoRA (Planogram) weight-based adaptation",
        "adapters": [{
            "type": "lora", "enabled": True, "weight": 1.0,
            "params": {
                "lora_path": "bharadwajkg/finetune-stable-diffusion-2-1-planogram-lora",
                "revision": "5129789dc19adb2edf3275cfe9c35327bbca8605",
                "adapter_name": "planogram_sd21_lora", "adapter_scale": 0.8,
            },
        }],
    },
    "lora_cartoon": {
        "label": "LoRA (Cartoon) weight-based adaptation",
        "adapters": [{
            "type": "lora", "enabled": True, "weight": 1.0,
            "params": {
                "lora_path": "dlcvproj/cartoon_sd_lora",
                "revision": "a526614e425b82be098144a862d49aa8e5ded083",
                "adapter_name": "cartoon_sd21_lora", "adapter_scale": 0.8,
            },
        }],
    },
}


def load_all_prompts() -> list[dict]:
    all_prompts = []
    for suite_file in SUITE_FILES:
        path = PROMPTS_DIR / suite_file
        with path.open("r", encoding="utf-8") as f:
            all_prompts.extend(json.load(f))
    return all_prompts


def make_prompt_entry(prompt: dict, condition_key: str, trial_seed: int, trial_index: int) -> dict:
    pid = prompt["id"]
    base_seed = BASE_SEED_MAP[pid]
    entry = {
        "id": pid,
        "seed": base_seed + trial_seed,
        "prompt": prompt["prompt"],
        "negative_prompt": prompt.get("negative_prompt", ""),
        "output_name": f"{pid}_{condition_key}_s20_t{trial_index}",
        "guidance_scale": 6.0,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "metadata": {
            **prompt.get("metadata", {}),
            "condition": condition_key,
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "mode": "cached_local_model",
        },
    }
    if "character_group" in prompt.get("metadata", {}):
        entry["metadata"]["character_group"] = prompt["metadata"]["character_group"]
    return entry


def main() -> None:
    all_prompts = load_all_prompts()
    print(f"Loaded {len(all_prompts)} prompts from {len(SUITE_FILES)} suites")

    for pid in [p["id"] for p in all_prompts]:
        if pid not in BASE_SEED_MAP:
            raise ValueError(f"Prompt {pid!r} not in BASE_SEED_MAP")

    generated = []
    for condition_key, condition_def in CONDITIONS.items():
        for trial_index, trial_seed in enumerate(TRIAL_SEEDS):
            name = f"sd21_s20_{condition_key}_t{trial_index}"
            config = {
                "name": name,
                "description": f"SD 2.1 @ 20 steps: {condition_def['label']}, trial {trial_index} (seed offset {trial_seed}).",
                "seed": trial_seed,
                "output_dir": "outputs",
                "model": MODEL_CONFIG,
                "adapters": condition_def["adapters"],
                "evaluation": EVALUATION_CONFIG,
                "prompts": [
                    make_prompt_entry(p, condition_key, trial_seed, trial_index)
                    for p in all_prompts
                ],
            }
            filename = f"{name}.json"
            output_path = CONFIGS_DIR / filename
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            generated.append(filename)
            print(f"  Generated: {filename} ({len(config['prompts'])} prompts)")

    print(f"\nGenerated {len(generated)} config files.")


if __name__ == "__main__":
    main()
