"""Harness config for the minimax_m3_vl family (PROJECTOR_TYPE_MINIMAX_M3_VL).

Reference: MiniMaxAI/MiniMax-M3.
"""

from .gen_images import DEFAULT_IMAGES

HARNESS_CONFIG = {
    "ref_repo": "MiniMaxAI/MiniMax-M3",
    "ref_files": [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "model-00026-of-00059.safetensors",
        "model-00059-of-00059.safetensors",
    ],
    # convert script, relative to the harness directory (minimax_m3_vl/)
    "convert": "../../../examples/mtmd/legacy-models/minimaxm3-convert-image-encoder-to-gguf.py",
    "compare_label": "image",
    "default_images": DEFAULT_IMAGES,
}
