from .config import ExperimentConfig, load_experiment_config
from .experiments.runner import ExperimentRunner
from .pipeline import FrameworkPipeline

# Import built-in components so they self-register with the registries.
from .models import diffusion as _diffusion  # noqa: F401
from .models import mock as _mock  # noqa: F401
from .adapters import controlnet as _controlnet  # noqa: F401
from .adapters import ip_adapter as _ip_adapter  # noqa: F401
from .adapters import lora as _lora  # noqa: F401
from .adapters import style_prompt as _style_prompt  # noqa: F401
from .refinement import flux as _flux  # noqa: F401
from .evaluation import metrics as _metrics  # noqa: F401

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "FrameworkPipeline",
    "load_experiment_config",
]
