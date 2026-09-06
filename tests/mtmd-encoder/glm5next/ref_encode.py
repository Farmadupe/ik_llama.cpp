#!/usr/bin/env python3
"""HF reference driver for the GLM-5.3-Flash vision tower."""
import argparse
import base64
import glob
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextVisionModel,
    Glm5NextVisionRotaryEmbedding,
)

VISION_PREFIX = "model.visual."


def load_vision(model_dir: str):
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        vision = Glm5NextVisionModel(cfg.vision_config)

    state = {}
    for f in sorted(glob.glob(f"{model_dir}/model-000*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for key in h.keys():
                if key.startswith(VISION_PREFIX):
                    state[key[len(VISION_PREFIX):]] = h.get_tensor(key).float()

    # strict=True is the correctness check: any missing/extra/renamed key throws here
    vision.load_state_dict(state, strict=True, assign=True)

    # the rotary inv_freq is a non-persistent buffer, so no state dict entry replaces it
    # and it would stay on the meta device the model was built under; rebuilding the
    # module runs its constructor for real
    head_dim = cfg.vision_config.hidden_size // cfg.vision_config.num_heads
    vision.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)

    return cfg, vision.float().eval()


def load_image_processor(model_dir: str) -> Glm5NextImageProcessor:
    """The checkpoint ships a combined processor_config.json; only its image_processor
    section is needed here, and taking it directly avoids pulling in the tokenizer."""
    with open(Path(model_dir) / "processor_config.json", encoding="utf-8") as f:
        kwargs = dict(json.load(f)["image_processor"])
    kwargs.pop("image_processor_type", None)
    return Glm5NextImageProcessor(**kwargs)


MODEL_DIR = os.environ.get("REFERENCE_DIR", "/tmp/glm5next_reference")
IMAGES = "images"
OUT = "reference_embeddings.json"

# where ik_encode --ik-image-preprocessing dumps its preprocessed images
IK_PRE_DIR = "/tmp/glm5next_pre"


def _load_pre_bin(path: str) -> np.ndarray:
    """Read a preprocessed image dumped by ik_encode --ik-image-preprocessing:
    int32 nx, int32 ny, then nx*ny*3 f32 interleaved RGB row-major. Returns an
    [ny, nx, 3] f32 array; it is already resized+rescaled+normalized, so no
    preprocessing is (re-)applied to it here."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        buf = np.frombuffer(f.read(nx * ny * 3 * 4), dtype=np.float32)
    return buf.reshape(ny, nx, 3)


def _encode_feats(feats) -> list:
    """Serialize a [n_tokens, n_embd] tensor as one base64 string per token row,
    each the row's n_embd raw little-endian f32 bytes. See
    tests/mtmd-encoder/README.md."""
    arr = np.ascontiguousarray(feats.detach().cpu().numpy(), dtype="<f4")
    return [base64.b64encode(row.tobytes()).decode("ascii") for row in arr]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-image-preprocessing", action="store_true")
    cli = ap.parse_args()

    _, vision = load_vision(MODEL_DIR)
    processor = load_image_processor(MODEL_DIR)

    out = {}
    if cli.ik_image_preprocessing:
        # Encoder-only mode: each *.pre.bin is one preprocessed image dumped by
        # ik_encode --ik-image-preprocessing, named <image>.pre.bin, so the output
        # keys pair with ik_embeddings.json by construction. ik's preprocessing feeds
        # both sides; the processor's own patchify still runs on ik's pixels, so the
        # patch unfold stays the checkpoint's.
        for path in sorted(glob.glob(f"{IK_PRE_DIR}/*.pre.bin")):
            name = Path(path).name[:-len(".pre.bin")]
            pre = torch.from_numpy(np.ascontiguousarray(_load_pre_bin(path)))
            pixels = pre.permute(2, 0, 1)[None]  # [1, C, H, W]
            pixel_values, grid_h, grid_w = processor.patchify(
                pixels,
                patch_size=processor.patch_size,
                merge_size=processor.merge_size,
                temporal_patch_size=processor.temporal_patch_size,
            )
            grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
            feats = vision(pixel_values[0].float(), grid_thw=grid_thw).pooler_output
            out[name] = _encode_feats(feats)
            print(f"{name}: {feats.shape[0]} x {feats.shape[1]} (from pre.bin)")
    else:
        for path in sorted(Path(IMAGES).glob("*.png")):
            img = Image.open(path).convert("RGB")
            enc = processor(images=img, return_tensors="pt")
            feats = vision(enc["pixel_values"].float(),
                           grid_thw=enc["image_grid_thw"]).pooler_output
            out[path.name] = _encode_feats(feats)
            print(f"{path.name}: {feats.shape[0]} x {feats.shape[1]}")

    Path(OUT).write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
