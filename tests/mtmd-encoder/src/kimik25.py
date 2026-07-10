"""Harness config for the kimik25 projector family (PROJECTOR_TYPE_KIMIK25).

Reference: moonshotai/Kimi-K2.7-Code.
"""

from .gen_images import DEFAULT_IMAGES

HARNESS_CONFIG = {
    "ref_repo": "moonshotai/Kimi-K2.7-Code",
    "ref_files": [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "model-00063-of-000064.safetensors",  # mm_projector.*
        "model-00064-of-000064.safetensors",  # vision_tower.*
        "configuration_kimi_k25.py",
        "configuration_deepseek.py",
        "modeling_kimi_k25.py",
        "modeling_deepseek.py",
        "kimi_k25_vision_processing.py",
        "kimi_k25_processor.py",
        "media_utils.py",
    ],
    # convert script, relative to the harness directory (kimik25/)
    "convert": "kimik25-convert-image-encoder-to-gguf.py",
    "compare_label": "image",
    "default_images": DEFAULT_IMAGES,
}
