#!/usr/bin/env python3
"""Day-1 HF reference driver for the Step-3.7-Flash vision tower + projector.

Produces post-projector vision embeddings (LLM hidden size 4096) for each test
image.

Pipeline (all faithful to the day-1 weights drop, loaded from REFERENCE_DIR):
  - preprocessing:   processing_step3.ImagePatcher + Step3VisionProcessor
  - vision tower:    vision_encoder.StepRoboticsVisionEncoder
  - projector:       vision_model.vit_downsampler1 (Conv2d 1536 to 3072, k3 s2 p1)
                     vision_model.vit_downsampler2 (Conv2d 3072 to 6144, k3 s2 p1)
                     vit_large_projector           (Linear  6144 to 4096)

Everything runs in fp32 with TF32 disabled; REF_DEVICE=cpu forces the CPU
path (GPU fp32 sits within ~1e-6 of it).
"""
import argparse
import base64
import glob
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open

MODEL_DIR = os.environ.get("REFERENCE_DIR", "/tmp/step3vl_reference")
IMAGES = os.environ.get("IK_IMAGES", "images")
OUT = "reference_embeddings.json"
LAYOUT_OUT = "reference_layout.json"

# where ik_encode --ik-image-preprocessing dumps its preprocessed views
IK_PRE_DIR = "/tmp/step3vl_pre"

# REF_DEVICE=cpu forces the CPU path; default is GPU when present.
DEVICE = os.environ.get("REF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Parity-grade fp32: on Ampere+, PyTorch silently runs fp32 matmuls (and cudnn
# convs - the downsampler path) as TF32 (10-bit mantissa) unless told not to.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Import the day-1 python as a namespace package rooted at the parent of MODEL_DIR.
_PARENT = str(Path(MODEL_DIR).resolve().parent)
_PKG = Path(MODEL_DIR).resolve().name
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

_ve = __import__(f"{_PKG}.vision_encoder", fromlist=["StepRoboticsVisionEncoder"])
_cfg = __import__(f"{_PKG}.configuration_step3p7", fromlist=["StepRoboticsVisionEncoderConfig"])
_proc = __import__(f"{_PKG}.processing_step3", fromlist=["Step3VisionProcessor", "ImagePatcher"])
StepRoboticsVisionEncoder = _ve.StepRoboticsVisionEncoder
StepRoboticsVisionEncoderConfig = _cfg.StepRoboticsVisionEncoderConfig
Step3VisionProcessor = _proc.Step3VisionProcessor
ImagePatcher = _proc.ImagePatcher


def load_reference():
    """Build the vision encoder + projector and load only the vit weights."""
    config = json.loads((Path(MODEL_DIR) / "config.json").read_text())
    vision_cfg = StepRoboticsVisionEncoderConfig(**config["vision_config"])
    text_hidden = int(config.get("text_config", {}).get("hidden_size", 4096))
    projector_bias = bool(config.get("projector_bias", False))

    # Build on CPU (not meta): EncoderRope2D registers a non-persistent
    # freqs_cache buffer in __init__ that is absent from the state_dict, so a meta
    # build would leave it un-materialized. The vision tower is ~1.8B params.
    vision = StepRoboticsVisionEncoder(vision_cfg)
    # vit_large_projector is a top-level Linear in Step3p7Model, not inside the
    # vision tower; downsampler1/2 live inside the vision tower.
    projector = nn.Linear(vision_cfg.width * 4, text_hidden, bias=projector_bias)

    vision_sd, proj_sd = {}, {}
    for f in sorted(glob.glob(f"{MODEL_DIR}/model-vit-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for key in h.keys():
                if key.startswith("vision_model."):
                    vision_sd[key[len("vision_model."):]] = h.get_tensor(key).float()
                elif key == "vit_large_projector.weight":
                    proj_sd["weight"] = h.get_tensor(key).float()

    # strict=True is the correctness check: any missing/extra/renamed key throws.
    vision.load_state_dict(vision_sd, strict=True)
    projector.load_state_dict(proj_sd, strict=True)
    return vision.float().eval().to(DEVICE), projector.float().eval().to(DEVICE)


@torch.no_grad()
def process_image_features(vision, projector, feats: torch.Tensor) -> torch.Tensor:
    """feats: [B, P, width] patch features from the vision tower.
    returns: [B, P/16, text_hidden] post-projector features (169 for overview,
    81 for a 504px slice).
    """
    B, P = feats.shape[:2]
    HW = int(round(P ** 0.5))
    x = feats.permute(0, 2, 1).view(B, -1, HW, HW)
    x = vision.vit_downsampler1(x)
    x = vision.vit_downsampler2(x)
    B, C, H2, W2 = x.shape
    x = x.view(B, -1, H2 * W2).permute(0, 2, 1)
    x = projector(x)
    return x


@torch.no_grad()
def encode_image(vision, projector, vproc, patcher, img: Image.Image):
    """Full day-1 path for one PIL image.

    Returns (views, layout): views is a list of (view_name, feats[N,hidden])
    pairs in model order [slice0, ..., overview]."""
    raw_img, patches, newline_mask = patcher(img)

    pv = vproc(raw_img, is_patch=False)["pixel_values"].float().to(DEVICE)   # [1,3,728,728]
    overview = process_image_features(vision, projector, vision(pv))[0]  # [169,H]

    views = []
    if len(patches) > 0:
        ppv = torch.cat([vproc(p, is_patch=True)["pixel_values"].float() for p in patches]).to(DEVICE)  # [K,3,504,504]
        patch_feats = process_image_features(vision, projector, vision(ppv))  # [K,81,H]
        views = [(f"slice{k}", patch_feats[k]) for k in range(patch_feats.shape[0])]
    views.append(("overview", overview))

    layout = {
        "num_patches": len(patches),
        "patch_tokens_each": int(views[0][1].shape[0]) if len(patches) > 0 else 0,
        "overview_tokens": int(overview.shape[0]),
        "total_tokens": int(sum(v.shape[0] for _, v in views)),
        "hidden": int(overview.shape[-1]),
        "patch_newline_mask": [bool(x) for x in newline_mask] if newline_mask else [],
        "order": [name for name, _ in views],
    }
    return views, layout


def _load_pre_bin(path: str) -> torch.Tensor:
    """Read a preprocessed image dumped by the ik_encode wrapper
    (--ik-image-preprocessing):
    int32 nx, int32 ny, then nx*ny*3 f32 interleaved RGB row-major.
    Returns a [1,3,ny,nx] tensor for encoder-only (resizer-free) parity checks."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        buf = torch.frombuffer(bytearray(f.read(nx * ny * 3 * 4)), dtype=torch.float32)
    return buf.view(ny, nx, 3).permute(2, 0, 1).unsqueeze(0).contiguous()


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

    print(f"reference device: {DEVICE} (tf32 disabled)")
    vision, projector = load_reference()
    vproc = Step3VisionProcessor(728, "bilinear", 504)
    patcher = ImagePatcher()

    out, layouts = {}, {}

    if cli.ik_image_preprocessing:
        # Encoder-only mode: each *.pre.bin is one preprocessed view dumped by
        # ik_encode --ik-image-preprocessing, already named <image>.<view>, so
        # the output keys pair with ik_embeddings.json by construction and
        # compare_embeddings.py then measures encoder+projector numerics only (ik's
        # preprocessing feeds both sides).
        for path in sorted(glob.glob(f"{IK_PRE_DIR}/*.pre.bin")):
            name = Path(path).name[:-len(".pre.bin")]
            pv = _load_pre_bin(path).to(DEVICE)
            feats = process_image_features(vision, projector, vision(pv))[0]
            out[name] = _encode_feats(feats)
            layouts[name] = {"total_tokens": int(feats.shape[0]), "hidden": int(feats.shape[-1]),
                             "source": "pre_bin"}
            print(f"{name}: {feats.shape[0]} x {feats.shape[1]} (from pre.bin)")
    else:
        for path in sorted(Path(IMAGES).glob("*.png")):
            img = Image.open(path).convert("RGB")
            views, layout = encode_image(vision, projector, vproc, patcher, img)
            for vname, feats in views:
                out[f"{path.name}.{vname}"] = _encode_feats(feats)
            layouts[path.name] = layout
            allf = torch.cat([f for _, f in views], dim=0)
            fmin = float(allf.min()); fmax = float(allf.max()); fmean = float(allf.mean())
            print(f"{path.name}: {layout['total_tokens']} x {layout['hidden']} tok "
                  f"(views={len(views)}) "
                  f"range[{fmin:.3f},{fmax:.3f}] mean {fmean:.4f} finite={bool(torch.isfinite(allf).all())}")

    Path(OUT).write_text(json.dumps(out))
    Path(LAYOUT_OUT).write_text(json.dumps(layouts, indent=2))
    print(f"wrote {OUT} and {LAYOUT_OUT}")


if __name__ == "__main__":
    main()
