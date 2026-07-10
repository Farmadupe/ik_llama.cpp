#!/usr/bin/env python3
"""Reference driver for the Kimi K2.7 vision tower + projector.

Runs the trust_remote_code modules shipped in the moonshotai/Kimi-K2.7-Code
repo (downloaded into $REFERENCE_DIR) for both preprocessing and encoding:

  - preprocessing: KimiK25VisionProcessor (kimi_k25_vision_processing.py)
  - encoder + projector: MoonViT3dPretrainedModel + PatchMergerMLP (vision +
    projector only)

The checkpoint tensor names are these modules' own state_dict names, so the
weights load strict=True with no remapping. Deviations from a full
`from_pretrained` of the whole model:

  - only the vision_tower / mm_projector submodules are instantiated; their
    weights are read from the two shards that hold them, with the submodule
    prefix stripped (PyTorch state_dict scoping, not a rename).

The reference runs exactly as the deployed model does: bf16 compute on GPU with
_attn_implementation = "flash_attention_2" (the checkpoint default). This is the
production path, fixed with no knobs. flash_attention_2 needs Ampere+, so it is
pinned to CUDA device 0 (the RTX 3090; the sm_75 RTX 2060 SUPER is unsupported).

TORCHDYNAMO_DISABLE: the upstream pos-emb interpolation helper is wrapped in
torch.compile; disabling dynamo runs the identical Python eagerly (a stock
PyTorch escape hatch, not a code change) so the reference does not depend on
a working inductor toolchain.
"""
import argparse
import base64
import glob
import json
import os
import struct
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# flash_attention_2 needs Ampere+; pin to the RTX 3090 (device 0), not the
# sm_75 RTX 2060 SUPER, which FA2 does not support. Set before torch inits CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_DIR = os.environ.get("REFERENCE_DIR", "/tmp/kimik25_reference")
IMAGES = "images"
OUT = "reference_embeddings.json"

# where ik_encode --ik-image-preprocessing dumps its preprocessed images
IK_PRE_DIR = "/tmp/kimik25_pre"

def load_upstream(model_dir: str):
    """Instantiate the checkpoint's own processor, vision tower and projector."""
    Processor = get_class_from_dynamic_module(
        "kimi_k25_vision_processing.KimiK25VisionProcessor", model_dir)
    processor = Processor.from_pretrained(model_dir)

    MoonViT = get_class_from_dynamic_module(
        "modeling_kimi_k25.MoonViT3dPretrainedModel", model_dir)
    mod = sys.modules[MoonViT.__module__]  # the upstream modeling module itself

    Config = get_class_from_dynamic_module(
        "configuration_kimi_k25.KimiK25Config", model_dir)
    cfg = Config.from_pretrained(model_dir)
    cfg.vision_config._attn_implementation = "flash_attention_2"

    # instantiate the vision tower + projector only
    vision = mod.MoonViT3dPretrainedModel(mod.VisionTowerConfig(cfg.vision_config))
    proj_cfg = mod.ProjectorConfig(cfg.vision_config)
    assert proj_cfg.mm_projector_type == "patchmerger", proj_cfg.mm_projector_type
    projector = mod.PatchMergerMLP(proj_cfg)

    vision_sd, proj_sd = {}, {}
    for f in sorted(glob.glob(f"{model_dir}/model-000*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for key in h.keys():
                if key.startswith("vision_tower."):
                    vision_sd[key[len("vision_tower."):]] = h.get_tensor(key).float()
                elif key.startswith("mm_projector."):
                    proj_sd[key[len("mm_projector."):]] = h.get_tensor(key).float()

    vision.load_state_dict(vision_sd, strict=True)
    projector.load_state_dict(proj_sd, strict=True)
    vision = vision.to(device="cuda", dtype=torch.bfloat16).eval()
    projector = projector.to(device="cuda", dtype=torch.bfloat16).eval()
    return processor, vision, projector


def _load_pre_bin(path: str) -> np.ndarray:
    """Read a preprocessed image dumped by ik_encode --ik-image-preprocessing:
    int32 nx, int32 ny, then nx*ny*3 f32 interleaved RGB row-major. Returns an
    [ny, nx, 3] f32 array; it is already resized+normalized, so no preprocessing
    is (re-)applied to it here."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        buf = np.frombuffer(f.read(nx * ny * 3 * 4), dtype=np.float32)
    return buf.reshape(ny, nx, 3)


def _encode_feats(feats) -> list:
    """Serialize a [n_tokens, n_embd] tensor as one base64 string per token row,
    each the row's n_embd raw little-endian f32 bytes. See
    tests/mtmd-encoder/README.md."""
    arr = np.ascontiguousarray(feats.detach().to(torch.float32).cpu().numpy(), dtype="<f4")
    return [base64.b64encode(row.tobytes()).decode("ascii") for row in arr]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-image-preprocessing", action="store_true")
    cli = ap.parse_args()

    processor, vision, projector = load_upstream(MODEL_DIR)

    out = {}
    if cli.ik_image_preprocessing:
        # Encoder-only mode: ik's preprocessing (resize + normalize) feeds both
        # sides, so the comparison isolates the encoder. Rather than reimplement
        # the processor's patch unfold, run the real preprocess() on each source
        # image and swap ik's normalized pixels in at the unfold seam: preprocess
        # calls navit_patchify as a bare global (it is `from .media_utils import
        # navit_patchify` into the processor module's namespace), so patching
        # that name -- in the processor module, NOT media_utils -- is what
        # preprocess actually resolves. The real navit_patchify then runs on ik's
        # pixels, so the reference unfold is the checkpoint's own, not a replica.
        # HF's own pixels are computed and discarded except for their shape,
        # which must match ik's grid or the two cannot be compared per-token.
        proc_mod = sys.modules[type(processor).__module__]
        real_navit_patchify = proc_mod.navit_patchify
        assert callable(real_navit_patchify)
        ik_hold = {"pixels": None}  # armed per image, disarmed on use

        def ik_navit_patchify(pixels, patch_size):
            ik = ik_hold["pixels"]
            assert ik is not None, "ik pixels not armed for this preprocess call"
            assert ik.shape == pixels.shape, \
                f"ik/hf geometry mismatch: ik {ik.shape} vs hf {pixels.shape}"
            ik_hold["pixels"] = None
            return real_navit_patchify(ik, patch_size)

        proc_mod.navit_patchify = ik_navit_patchify

        for png in sorted(Path(IMAGES).glob("*.png")):
            # <image>.pre.bin pairs with ik_embeddings.json by construction; the
            # [None] gives ik's [H, W, 3] the T=1 axis navit_patchify expects.
            ik_hold["pixels"] = _load_pre_bin(
                f"{IK_PRE_DIR}/{png.name}.pre.bin")[None]  # [1, H, W, 3]
            batch = processor.preprocess(
                [{"type": "image", "image": Image.open(png).convert("RGB")}],
                return_tensors="pt")
            tokens = vision(batch["pixel_values"].to("cuda", torch.bfloat16),
                            batch["grid_thws"].to("cuda"))
            feats = projector(tokens)[0]  # [n_merged, text_hidden]
            out[png.name] = _encode_feats(feats)
            print(f"{png.name}: {feats.shape[0]} x {feats.shape[1]} (ik pixels)")
    else:
        for path in sorted(Path(IMAGES).glob("*.png")):
            image = Image.open(path).convert("RGB")
            batch = processor.preprocess([{"type": "image", "image": image}],
                                         return_tensors="pt")
            tokens = vision(batch["pixel_values"].to("cuda", torch.bfloat16),
                            batch["grid_thws"].to("cuda"))
            feats = projector(tokens)[0]  # [n_merged, text_hidden]
            out[path.name] = _encode_feats(feats)

    Path(OUT).write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
