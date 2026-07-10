#!/usr/bin/env python3
"""HF reference driver for the MiniMax-M3 vision tower.

"""
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
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector,
    MiniMaxM3VLVisionModel,
)


def map_vision_key(k: str) -> str:
    k = k[len("vision_tower.vision_model.") :]
    k = k.replace("embeddings.patch_embedding.", "embeddings.proj.")
    k = k.replace("encoder.layers.", "layers.")
    return k


def map_proj_key(k: str) -> str:
    if k.startswith("multi_modal_projector."):
        return k[len("multi_modal_projector.") :]
    return k.replace("patch_merge_mlp.linear_1", "merge_linear_1").replace(
        "patch_merge_mlp.linear_2", "merge_linear_2"
    )


def load_modules(model_dir: str):
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        vision = MiniMaxM3VLVisionModel(cfg.vision_config)
        proj = MiniMaxM3VLMultiModalProjector(cfg)

    vision_sd, proj_sd = {}, {}
    for f in sorted(glob.glob(f"{model_dir}/model-000*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for key in h.keys():
                if key.startswith("vision_tower."):
                    vision_sd[map_vision_key(key)] = h.get_tensor(key).float()
                elif key.startswith(("multi_modal_projector.", "patch_merge_mlp.")):
                    proj_sd[map_proj_key(key)] = h.get_tensor(key).float()

    # strict=True is the correctness check: any missing/extra/renamed key throws here
    vision.load_state_dict(vision_sd, strict=True, assign=True)
    proj.load_state_dict(proj_sd, strict=True, assign=True)
    return cfg, vision.float().eval(), proj.float().eval()


MODEL_DIR = os.environ.get("REFERENCE_DIR", "/tmp/minimax_m3_vl_reference")
IMAGES = "images"
OUT = "reference_embeddings.json"

# where ik_encode --ik-image-preprocessing dumps its preprocessed images
IK_PRE_DIR = "/tmp/minimax_m3_vl_pre"


def _load_pre_bin(path: str) -> np.ndarray:
    """Read a preprocessed image dumped by ik_encode --ik-image-preprocessing:
    int32 nx, int32 ny, then nx*ny*3 f32 interleaved RGB row-major. Returns an
    [ny, nx, 3] f32 array; it is already resized+rescaled+normalized, so no
    preprocessing is (re-)applied to it here."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        buf = np.frombuffer(f.read(nx * ny * 3 * 4), dtype=np.float32)
    return buf.reshape(ny, nx, 3)


def _patchify(pre: np.ndarray, patch_size: int, temporal_patch_size: int,
              merge_size: int):
    """Encoder-only patch unfold for a single frame. pre is [H, W, 3] already
    rescaled+normalized pixels; returns (pixel_values [n, C*tps*ps*ps],
    grid_thw [[t, h, w]]).

    NOTE: hand-written replica of the processor's unfold, not round-trip
    validated; see tests/mtmd-encoder/README.md."""
    H, W, C = pre.shape
    patches = torch.from_numpy(np.ascontiguousarray(pre)).permute(2, 0, 1)  # [C, H, W]
    patches = patches[None, None]  # [B=1, T=1, C, H, W]
    if patches.shape[1] % temporal_patch_size != 0:
        reps = patches[:, -1:].repeat(
            1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1)
        patches = torch.cat([patches, reps], dim=1)
    B, t_len = patches.shape[0], patches.shape[1]
    grid_t = t_len // temporal_patch_size
    grid_h, grid_w = H // patch_size, W // patch_size
    patches = patches.view(
        B, grid_t, temporal_patch_size, C,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = patches.reshape(
        B, grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size)
    pixel_values = flat[0].float()  # [n, feat]  (single image, batch dim dropped)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return pixel_values, grid_thw


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

    _, vision, proj = load_modules(MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    out = {}
    if cli.ik_image_preprocessing:
        # Encoder-only mode: each *.pre.bin is one preprocessed image dumped by
        # ik_encode --ik-image-preprocessing, named <image>.pre.bin, so the
        # output keys pair with ik_embeddings.json by construction. ik's
        # preprocessing feeds both sides; we only re-run the patch unfold.
        patch_size = int(getattr(processor, "patch_size", 14))
        temporal_patch_size = int(getattr(processor, "temporal_patch_size", 2))
        merge_size = int(getattr(processor, "merge_size", 2))
        for path in sorted(glob.glob(f"{IK_PRE_DIR}/*.pre.bin")):
            name = Path(path).name[:-len(".pre.bin")]
            pixel_values, grid_thw = _patchify(
                _load_pre_bin(path), patch_size, temporal_patch_size, merge_size)
            vout = vision(pixel_values=pixel_values, grid_thw=grid_thw)
            feats = proj(vout.last_hidden_state.squeeze(0))  # [n_tokens, text_hidden]
            out[name] = _encode_feats(feats)
            print(f"{name}: {feats.shape[0]} x {feats.shape[1]} (from pre.bin)")
    else:
        for path in sorted(Path(IMAGES).glob("*.png")):
            img = Image.open(path).convert("RGB")
            enc = processor(images=img, return_tensors="pt")
            pixel_values = enc["pixel_values"].float()
            grid_thw = enc["image_grid_thw"]

            vout = vision(pixel_values=pixel_values, grid_thw=grid_thw)
            feats = proj(vout.last_hidden_state.squeeze(0))  # [n_tokens, text_hidden]

            out[path.name] = _encode_feats(feats)

    Path(OUT).write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
