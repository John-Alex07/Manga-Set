"""Generate Flux Schnell benchmark configs with a reduced prompt set.

Flux is a 12B-parameter model whose CPU inference time is extremely high
(~30-60 min/image at 4 steps).  This generator produces a reduced set of
10 prompts x 2 conditions x 1 trial = 20 images to keep the experiment
feasible on CPU while demonstrating backbone generality.

Flux Schnell uses its default FlowMatchEuler scheduler and does NOT support
negative prompts.  guidance_scale=0.0 is the standard for Schnell.
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

# We only use 1 trial to keep run time manageable.
TRIAL_SEEDS = [42]
PROMPT_LIMIT = 10
NUM_INFERENCE_STEPS = 4

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
    "model_id": "black-forest-labs/FLUX.1-schnell",
    "device": "cpu",
    "dtype": "bfloat16",
    "revision": "741f7c3ce8b383c54771c7003378a50191e9efe9",
    "params": {
        "use_safetensors": True,
        "scheduler": "default",
        "local_files_only": True,
    },
}

EVALUATION_CONFIG = {
    "metrics": [
        "file_integrity",
        "image_statistics",
        "latency",
        "clip_text_alignment",
        "image_reward",
    ],
    "params": {
        "clip_text_alignment": {
            "model_id": "openai/clip-vit-base-patch32",
            "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "device": "cpu",
            "local_files_only": True,
        },
    },
}

CONDITIONS: dict[str, dict] = {
    "base": {"label": "Flux Schnell Base (no adapter)", "adapters": []},
    "style_prompt": {
        "label": "Flux Schnell + Style Prompt",
        "adapters": [{
            "type": "style_prompt", "enabled": True, "weight": 0.8,
            "params": {"style_prompt": "black and white manga screentone, cinematic line art"},
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
    return {
        "id": pid,
        "seed": base_seed + trial_seed,
        "prompt": prompt["prompt"],
        "negative_prompt": "",
        "output_name": f"{pid}_{condition_key}_flux_t{trial_index}",
        "guidance_scale": 0.0,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "metadata": {
            **prompt.get("metadata", {}),
            "condition": condition_key,
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "backbone": "flux_schnell",
            "mode": "cached_local_model",
        },
    }


def main() -> None:
    all_prompts = load_all_prompts()
    print(f"Loaded {len(all_prompts)} prompts from {len(SUITE_FILES)} suites")

    reduced_prompts = all_prompts[:PROMPT_LIMIT]
    print(f"Using reduced set of {len(reduced_prompts)} prompts for Flux Schnell")

    for pid in [p["id"] for p in reduced_prompts]:
        if pid not in BASE_SEED_MAP:
            raise ValueError(f"Prompt {pid!r} not in BASE_SEED_MAP")

    generated = []
    for condition_key, condition_def in CONDITIONS.items():
        for trial_index, trial_seed in enumerate(TRIAL_SEEDS):
            name = f"flux_{condition_key}_t{trial_index}"
            config = {
                "name": name,
                "description": f"Flux Schnell ablation: {condition_def['label']}, trial {trial_index} (seed offset {trial_seed}).",
                "seed": trial_seed,
                "output_dir": "outputs",
                "model": MODEL_CONFIG,
                "adapters": condition_def["adapters"],
                "evaluation": EVALUATION_CONFIG,
                "prompts": [
                    make_prompt_entry(p, condition_key, trial_seed, trial_index)
                    for p in reduced_prompts
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
