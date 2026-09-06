#!/usr/bin/env python3
"""Convert the GLM-5.3-Flash vision tower and projector to a clip mmproj GGUF."""
import argparse
import json
from pathlib import Path

import torch
from gguf import GGMLQuantizationType, GGUFEndian, GGUFWriter
from safetensors.torch import load_file


VISION_PREFIX = "model.visual."

# projector tensors, checked exhaustively so a renamed checkpoint key fails loudly
# instead of silently dropping out of the mmproj
PROJ_MAP = {
    "downsample.weight": "mm.patch_merger.weight",
    "downsample.bias": "mm.patch_merger.bias",
    "merger.proj.weight": "mm.model.fc.weight",
    "merger.post_projection_norm.weight": "mm.post_norm.weight",
    "merger.post_projection_norm.bias": "mm.post_norm.bias",
    "merger.gate_proj.weight": "mm.gate.weight",
    "merger.up_proj.weight": "mm.up.weight",
    "merger.down_proj.weight": "mm.down.weight",
}

BLOCK_MAP = {
    "norm1": "ln1",
    "norm2": "ln2",
    "attn.qkv": "attn_qkv",
    "attn.proj": "attn_out",
    "attn.q_norm": "attn_q_norm",
    "attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return {name: shards[0].name for name in load_file(str(shards[0]), device="cpu")}

    raise FileNotFoundError(f"unable to find safetensors index in {model_dir}")


def rename_tensor(name: str) -> str | None:
    if not name.startswith(VISION_PREFIX):
        return None
    name = name[len(VISION_PREFIX):]

    if name in PROJ_MAP:
        return PROJ_MAP[name]
    if name == "patch_embed.proj.weight":
        return "v.patch_embd.weight"
    if name == "patch_embed.proj.bias":
        return "v.patch_embd.bias"
    if name == "post_layernorm.weight":
        return "v.post_ln.weight"

    if name.startswith("blocks."):
        _, bid, rest = name.split(".", 2)
        stem, _, suffix = rest.rpartition(".")
        if stem not in BLOCK_MAP:
            raise ValueError(f"unmapped vision block tensor: {name}")
        return f"v.blk.{bid}.{BLOCK_MAP[stem]}.{suffix}"

    raise ValueError(f"unmapped vision tensor: {name}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_tensor(writer: GGUFWriter, name: str, data: torch.Tensor, args) -> None:
    """Store one tensor at the requested width. bf16 needs the int16 view because numpy
    has no bfloat16; 1-D tensors (norms and biases) stay f32, as the clip loader expects."""
    if args.use_f32 or data.ndim == 1:
        writer.add_tensor(name, data.float().numpy())
    elif args.use_bf16:
        if data.dtype == torch.bfloat16:
            writer.add_tensor(name, data.contiguous().view(torch.int16).numpy(),
                              raw_dtype=GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, data.float().numpy())
    else:
        writer.add_tensor(name, data.half().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the GLM-5.3-Flash vision encoder/projector to GGUF")
    parser.add_argument("-m", "--model-dir", required=True, help="Path to the GLM-5.3-Flash HF model directory")
    parser.add_argument("-o", "--output", default=None, help="Output GGUF path")
    parser.add_argument("--use-f32", action="store_true", help="Write tensors as f32 instead of f16")
    parser.add_argument("--use-bf16", action="store_true", help="Write tensors as bf16 instead of f16")
    parser.add_argument("--bigendian", action="store_true", help="Write big-endian GGUF")
    args = parser.parse_args()
    if args.use_f32 and args.use_bf16:
        parser.error("--use-f32 and --use-bf16 are mutually exclusive")

    model_dir = Path(args.model_dir)
    vision_config = read_json(model_dir / "config.json")["vision_config"]
    image_processor = read_json(model_dir / "processor_config.json")["image_processor"]

    output = Path(args.output) if args.output else model_dir / "mmproj-glm5next.gguf"

    writer = GGUFWriter(
        path=str(output),
        arch="clip",
        endianess=GGUFEndian.BIG if args.bigendian else GGUFEndian.LITTLE,
    )
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)
    writer.add_string("clip.projector_type", "glm5next")
    writer.add_string("general.name", "GLM-5.3-Flash vision projector")
    writer.add_uint32("general.file_type", 0 if args.use_f32 else 1)

    writer.add_uint32("clip.vision.image_size", vision_config["image_size"])
    writer.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
    writer.add_uint32("clip.vision.embedding_length", vision_config["hidden_size"])
    writer.add_uint32("clip.vision.feed_forward_length", vision_config["intermediate_size"])
    writer.add_uint32("clip.vision.projection_dim", vision_config["out_hidden_size"])
    writer.add_uint32("clip.vision.attention.head_count", vision_config["num_heads"])
    writer.add_uint32("clip.vision.block_count", vision_config["depth"])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_config["rms_norm_eps"])
    writer.add_uint32("clip.vision.spatial_merge_size", vision_config["spatial_merge_size"])
    writer.add_float32("clip.vision.swiglu_limit", vision_config["swiglu_limit"])
    writer.add_array("clip.vision.image_mean", image_processor["image_mean"])
    writer.add_array("clip.vision.image_std", image_processor["image_std"])
    writer.add_bool("clip.use_silu", True)

    temporal_patch_size = vision_config["temporal_patch_size"]

    weight_map = load_index(model_dir)
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}
    written = set()

    for src_name in sorted(weight_map):
        dst_name = rename_tensor(src_name)
        if dst_name is None:
            continue

        shard_name = weight_map[src_name]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(model_dir / shard_name), device="cpu")

        data = shard_cache[shard_name][src_name]

        if src_name.endswith("patch_embed.proj.weight"):
            # a conv3d over the temporal patch, stored as one conv2d kernel per temporal
            # slice; the graph sums their outputs
            if data.ndim != 5 or data.shape[2] != temporal_patch_size:
                raise ValueError(f"{src_name}: expected a 5-D kernel with temporal size "
                                 f"{temporal_patch_size}, got shape {tuple(data.shape)}")
            for i in range(temporal_patch_size):
                slice_name = dst_name if i == 0 else f"{dst_name}.{i}"
                add_tensor(writer, slice_name, data[:, :, i], args)
                written.add(slice_name)
            continue

        add_tensor(writer, dst_name, data, args)
        written.add(dst_name)

    missing = sorted(set(PROJ_MAP.values()) - written)
    if missing:
        raise ValueError(f"projector tensors missing from the checkpoint: {missing}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output} ({len(written)} tensors)")


if __name__ == "__main__":
    main()
