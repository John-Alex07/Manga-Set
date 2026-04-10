from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework import ExperimentConfig, ExperimentRunner  # noqa: E402


def build_config(
    backend: str,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    output_name: str,
    use_style_prompt: bool,
    use_controlnet: bool,
    use_ip_adapter: bool,
    use_refinement: bool,
) -> ExperimentConfig:
    adapters = []
    if use_style_prompt:
        adapters.append(
            {
                "type": "style_prompt",
                "enabled": True,
                "params": {"style_prompt": "black and white manga screentone, dramatic linework"},
            }
        )
    if use_controlnet:
        adapters.append(
            {
                "type": "controlnet",
                "enabled": True,
                "params": {
                    "prompt_hint": "clear panel composition, readable silhouettes",
                    "controls": {"panel_layout": "single panel"},
                },
            }
        )
    if use_ip_adapter:
        adapters.append(
            {
                "type": "ip_adapter",
                "enabled": True,
                "params": {"identity_prompt": "preserve main character identity"},
            }
        )

    refinement = None
    if use_refinement:
        refinement = {"type": "flux", "enabled": True, "params": {"save_copy": True}}

    return ExperimentConfig.from_dict(
        {
            "name": "streamlit_demo_run",
            "description": "Interactive run from the framework demo UI.",
            "seed": 42,
            "output_dir": "outputs",
            "model": {
                "backend": backend,
                "model_id": model_id,
                "device": "cpu",
            },
            "adapters": adapters,
            "refinement": refinement,
            "evaluation": {
                "metrics": ["file_integrity", "image_statistics", "histogram_consistency", "latency"],
            },
            "prompts": [
                {
                    "id": "interactive_prompt",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "output_name": output_name,
                    "metadata": {"source": "streamlit_ui"},
                }
            ],
        }
    )


st.set_page_config(page_title="Manga-Set Framework", layout="wide")
st.title("Manga-Set Research Framework")
st.caption("Config-driven experiment runner for controllable manga/storyboard generation.")

col1, col2 = st.columns(2)

with col1:
    backend = st.selectbox("Backend", ["mock", "diffusers"], index=0)
    default_model = "mock-storyboard-backend" if backend == "mock" else "stabilityai/stable-diffusion-2-1"
    model_id = st.text_input("Model ID", value=default_model)
    prompt = st.text_area(
        "Prompt",
        value="A dramatic manga rooftop confrontation at sunset, dynamic line art, cinematic shadows",
        height=160,
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value="blurry, malformed hands, low contrast, extra limbs",
        height=120,
    )

with col2:
    output_name = st.text_input("Output Name", value="interactive_panel")
    use_style_prompt = st.checkbox("Enable Style Prompt stage", value=True)
    use_controlnet = st.checkbox("Enable ControlNet stage", value=True)
    use_ip_adapter = st.checkbox("Enable IP-Adapter stage", value=False)
    use_refinement = st.checkbox("Enable refinement stage", value=True)


if st.button("Run Experiment", type="primary"):
    config = build_config(
        backend=backend,
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_name=output_name,
        use_style_prompt=use_style_prompt,
        use_controlnet=use_controlnet,
        use_ip_adapter=use_ip_adapter,
        use_refinement=use_refinement,
    )

    runner = ExperimentRunner(config=config, project_root=PROJECT_ROOT)
    report = runner.run()
    result = report["results"][0]

    st.subheader("Artifacts")
    for artifact in result["artifacts"]:
        artifact_path = Path(artifact["path"])
        if artifact_path.exists():
            if artifact["kind"] == "image":
                st.image(str(artifact_path), caption=artifact_path.name, use_container_width=True)
            else:
                st.code(artifact_path.read_text(encoding="utf-8"), language="text")

    st.subheader("Scores")
    st.json(result["scores"])

    st.subheader("Metadata")
    st.json(result["metadata"])
