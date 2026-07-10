#!/usr/bin/env python3
# Convert the Step-3.7-Flash vision encoder + projector (day-1 HF safetensors) to
# an mmproj GGUF for ik_llama.cpp's clip loader. This script uses raw
# GGUFWriter.add_* calls (not gguf's clip helpers) so it works regardless of
# whether the repo-local gguf-py knows the step3vl projector type.
#
# Tensor name mapping (left: HF day-1 name; right: GGUF name):
#   vision_model.conv1.weight                                  v.patch_embd.weight
#   vision_model.positional_embedding                          v.position_embd.weight
#   vision_model.ln_pre.{weight,bias}                          v.pre_ln.{weight,bias}
#   vision_model.transformer.resblocks.N.attn.in_proj_weight   v.blk.N.attn_qkv.weight  (fused QKV kept fused)
#   vision_model.transformer.resblocks.N.attn.in_proj_bias     v.blk.N.attn_qkv.bias
#   vision_model.transformer.resblocks.N.attn.out_proj.{w,b}   v.blk.N.attn_out.{weight,bias}
#   vision_model.transformer.resblocks.N.ln_1.{w,b}            v.blk.N.ln1.{weight,bias}
#   vision_model.transformer.resblocks.N.ln_2.{w,b}            v.blk.N.ln2.{weight,bias}
#   vision_model.transformer.resblocks.N.mlp.c_fc.{w,b}        v.blk.N.ffn_up.{weight,bias}
#   vision_model.transformer.resblocks.N.mlp.c_proj.{w,b}      v.blk.N.ffn_down.{weight,bias}
#   vision_model.transformer.resblocks.N.ls_1.gamma            v.blk.N.ls1.weight  (layer scale)
#   vision_model.transformer.resblocks.N.ls_2.gamma            v.blk.N.ls2.weight
#   vision_model.vit_downsampler1.{w,b}                        mm.0.{weight,bias}  (Conv2d 1536 to 3072)
#   vision_model.vit_downsampler2.{w,b}                        mm.1.{weight,bias}  (Conv2d 3072 to 6144)
#   vit_large_projector.weight                                 mm.model.fc.weight  (Linear 6144 to 4096)
import argparse
import json
import re
from pathlib import Path

import torch
from gguf import GGMLQuantizationType, GGUFEndian, GGUFWriter
from safetensors.torch import load_file

# Mistral common dataset normalization constants (Step3's default image_mean/std,
# see _MISTRAL_COMMON_DATASET_MEAN/STD in the StepFun fork conversion/base.py).
MISTRAL_MEAN = [0.48145466, 0.4578275, 0.40821073]
MISTRAL_STD = [0.26862954, 0.26130258, 0.27577711]

# understand_projector_stride ** 2 (from config.json; 2**2 == 4).
DEFAULT_SCALE_FACTOR = 4
# MAX_IMAGE_SIZE from processing_step3.py.
PREPROC_IMAGE_SIZE = 3024

_RESBLOCK_SUFFIX = {
    "attn.in_proj_weight": "attn_qkv.weight",
    "attn.in_proj_bias": "attn_qkv.bias",
    "attn.out_proj.weight": "attn_out.weight",
    "attn.out_proj.bias": "attn_out.bias",
    "ln_1.weight": "ln1.weight",
    "ln_1.bias": "ln1.bias",
    "ln_2.weight": "ln2.weight",
    "ln_2.bias": "ln2.bias",
    "mlp.c_fc.weight": "ffn_up.weight",
    "mlp.c_fc.bias": "ffn_up.bias",
    "mlp.c_proj.weight": "ffn_down.weight",
    "mlp.c_proj.bias": "ffn_down.bias",
    "ls_1.gamma": "ls1.weight",
    "ls_2.gamma": "ls2.weight",
}


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        tensors = load_file(str(shards[0]), device="cpu")
        return {name: shards[0].name for name in tensors}

    raise FileNotFoundError(f"unable to find safetensors index in {model_dir}")


def rename_tensor(name: str) -> str | None:
    m = re.match(r"vision_model\.vit_downsampler(\d+)\.(weight|bias)$", name)
    if m:
        return f"mm.{int(m.group(1)) - 1}.{m.group(2)}"
    if name == "vit_large_projector.weight":
        return "mm.model.fc.weight"
    if name == "vision_model.conv1.weight":
        return "v.patch_embd.weight"
    if name == "vision_model.positional_embedding":
        return "v.position_embd.weight"
    if name in ("vision_model.ln_pre.weight", "vision_model.ln_pre.bias"):
        return name.replace("vision_model.ln_pre", "v.pre_ln")

    m = re.match(r"vision_model\.transformer\.resblocks\.(\d+)\.(.+)$", name)
    if m:
        suffix = _RESBLOCK_SUFFIX.get(m.group(2))
        if suffix is None:
            return None
        return f"v.blk.{int(m.group(1))}.{suffix}"
    return None


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Step-3.7-Flash vision encoder/projector to GGUF")
    parser.add_argument("-m", "--model-dir", required=True, help="Path to the reference model dir (vit shards + config.json)")
    parser.add_argument("-o", "--output", default=None, help="Output GGUF path")
    parser.add_argument("--use-f32", action="store_true", help="Write matmul/conv weights as f32 instead of f16")
    parser.add_argument("--use-bf16", action="store_true", help="Write matmul/conv weights as bf16 instead of f16")
    parser.add_argument("--bigendian", action="store_true", help="Write big-endian GGUF")
    args = parser.parse_args()
    if args.use_f32 and args.use_bf16:
        parser.error("--use-f32 and --use-bf16 are mutually exclusive")

    model_dir = Path(args.model_dir)
    config = read_json(model_dir / "config.json")
    vision_config = config["vision_config"]
    text_config = config.get("text_config", {})

    hidden_size = int(vision_config.get("width", vision_config.get("hidden_size")))
    mlp_ratio = float(vision_config.get("mlp_ratio", 8960 / 1536))
    intermediate_size = int(vision_config.get("intermediate_size", round(hidden_size * mlp_ratio)))
    projection_dim = int(text_config.get("hidden_size", 4096))
    stride = int(config.get("understand_projector_stride", 2))

    output = Path(args.output) if args.output else model_dir / "mmproj-step3vl.gguf"
    ftype = 0 if args.use_f32 else 1

    writer = GGUFWriter(
        path=str(output),
        arch="clip",
        endianess=GGUFEndian.BIG if args.bigendian else GGUFEndian.LITTLE,
    )
    writer.add_string("general.type", "mmproj")
    writer.add_string("general.name", "Step-3.7-Flash vision projector")
    writer.add_uint32("general.file_type", ftype)
    writer.add_uint32("general.quantization_version", 2)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "step3vl")

    writer.add_uint32("clip.vision.image_size", int(vision_config.get("image_size", 728)))
    writer.add_uint32("clip.vision.patch_size", int(vision_config.get("patch_size", 14)))
    writer.add_uint32("clip.vision.embedding_length", hidden_size)
    writer.add_uint32("clip.vision.feed_forward_length", intermediate_size)
    writer.add_uint32("clip.vision.projection_dim", projection_dim)
    writer.add_uint32("clip.vision.attention.head_count", int(vision_config.get("heads", vision_config.get("num_attention_heads", 16))))
    writer.add_uint32("clip.vision.block_count", int(vision_config.get("layers", vision_config.get("num_hidden_layers", 47))))
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", float(vision_config.get("layer_norm_eps", 1e-5)))
    writer.add_uint32("clip.vision.projector.scale_factor", stride * stride)
    writer.add_uint32("clip.vision.preproc_image_size", PREPROC_IMAGE_SIZE)
    writer.add_array("clip.vision.image_mean", MISTRAL_MEAN)
    writer.add_array("clip.vision.image_std", MISTRAL_STD)

    weight_map = load_index(model_dir)
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    n_written = 0
    for src_name in sorted(weight_map):
        dst_name = rename_tensor(src_name)
        if dst_name is None:
            continue

        shard_name = weight_map[src_name]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(model_dir / shard_name), device="cpu")
        data = shard_cache[shard_name][src_name]

        # position_embd stays f32; 1D tensors (biases, norms, layer scales) stay
        # f32; matmul/conv weights follow the storage flag (f16 default).
        is_weight_2d = data.ndim >= 2 and dst_name.endswith(".weight")
        keep_f32 = dst_name == "v.position_embd.weight" or not is_weight_2d

        if keep_f32 or args.use_f32:
            writer.add_tensor(dst_name, data.float().numpy())
        elif args.use_bf16:
            if data.dtype == torch.bfloat16:
                writer.add_tensor(dst_name, data.contiguous().view(torch.int16).numpy(),
                                  raw_dtype=GGMLQuantizationType.BF16)
            else:
                writer.add_tensor(dst_name, data.float().numpy())
        else:
            writer.add_tensor(dst_name, data.half().numpy())
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output} ({n_written} tensors)")


if __name__ == "__main__":
    main()
