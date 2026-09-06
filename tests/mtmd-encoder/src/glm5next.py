"""Harness config for the glm5next family (PROJECTOR_TYPE_GLM5NEXT).

Reference: zai-org/GLM-5.3-Flash.
"""

from .gen_images import DEFAULT_IMAGES

HARNESS_CONFIG = {
    "ref_repo": "zai-org/GLM-5.3-Flash",
    "ref_files": [
        "config.json",
        "processor_config.json",
        "model.safetensors.index.json",
        "model-00062-of-00062.safetensors",  # the whole vision tower and projector
    ],
    # convert script, relative to the harness directory (glm5next/)
    "convert": "glm5next-convert-image-encoder-to-gguf.py",
    "compare_label": "image",
    "default_images": DEFAULT_IMAGES,
}
