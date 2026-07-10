"""Harness config for the step3vl family (PROJECTOR_TYPE_STEP3VL).

Reference: stepfun-ai/Step-3.7-Flash.
"""

from .gen_images import DEFAULT_IMAGES

HARNESS_CONFIG = {
    "ref_repo": "stepfun-ai/Step-3.7-Flash",
    "ref_files": [
        "config.json",
        "configuration_step3p7.py",
        "modeling_step3p7.py",
        "processing_step3.py",
        "vision_encoder.py",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-vit-00001.safetensors",
        "model-vit-00002.safetensors",
    ],
    # convert script, relative to the harness directory (step3vl/)
    "convert": "step3vl-convert-image-encoder-to-gguf.py",
    "compare_label": "view",
    "default_images": DEFAULT_IMAGES,
}
