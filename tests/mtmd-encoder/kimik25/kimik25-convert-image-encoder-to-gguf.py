#!/usr/bin/env python3
"""Convert the Kimi K2.7 (MoonViT3d) vision tower + projector to a CLIP mmproj GGUF
for ik_llama.cpp's PROJECTOR_TYPE_KIMIK25 graph (examples/mtmd/clip.cpp).

Tensors are transported verbatim (renamed only) except:
  - fused wqkv is split into attn_q / attn_k / attn_v (the graph uses three matmuls);
  - q/k rows are permuted from the checkpoint's interleaved 2D-RoPE layout to the
    split layout consumed by build_rope_2d (see _qk_reindex).
"""
import argparse
import json
from pathlib import Path

import torch
from gguf import GGMLQuantizationType, GGUFEndian, GGUFWriter
from safetensors.torch import load_file

ENCODER_PREFIX = "vision_tower.encoder.blocks."


def _qk_reindex(head_dim: int) -> torch.Tensor:
    """Per-head row permutation from the checkpoint's RoPE layout to build_rope_2d's.

    Checkpoint (Rope2DPosEmbRepeated/apply_rope): adjacent pairs (4i, 4i+1) rotate by
    x*f_i and (4i+2, 4i+3) by y*f_i. build_rope_2d (interleave_freq=false): pairs
    (2i, 2i+1) of the first half rotate by x*f_i, of the second half by y*f_i.
    Hence: x-pairs first, then y-pairs, frequency order preserved.
    """
    d = torch.arange(head_dim)
    return d.reshape(head_dim // 4, 2, 2).permute(1, 0, 2).reshape(-1)


def permute_qk_weight(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """Reindex a Q or K weight [out_dim, in_dim] head-wise."""
    out_dim, in_dim = w.shape
    idx = _qk_reindex(out_dim // n_head)
    return w.reshape(n_head, out_dim // n_head, in_dim)[:, idx, :].reshape(out_dim, in_dim)


def permute_qk_bias(b: torch.Tensor, n_head: int) -> torch.Tensor:
    """Same reindexing for a Q or K bias [out_dim]."""
    out_dim = b.shape[0]
    idx = _qk_reindex(out_dim // n_head)
    return b.reshape(n_head, out_dim // n_head)[:, idx].reshape(-1)


def rename_static(name: str) -> str | None:
    """Non-block tensors: patch embed, position embed, final norm, projector."""
    table = {
        "vision_tower.patch_embed.proj.weight": "v.patch_embd.weight",
        "vision_tower.patch_embed.proj.bias": "v.patch_embd.bias",
        "vision_tower.patch_embed.pos_emb.weight": "v.position_embd.weight",
        "vision_tower.encoder.final_layernorm.weight": "v.post_ln.weight",
        "vision_tower.encoder.final_layernorm.bias": "v.post_ln.bias",
        "mm_projector.pre_norm.weight": "mm.input_norm.weight",
        "mm_projector.pre_norm.bias": "mm.input_norm.bias",
        "mm_projector.proj.0.weight": "mm.1.weight",
        "mm_projector.proj.0.bias": "mm.1.bias",
        "mm_projector.proj.2.weight": "mm.2.weight",
        "mm_projector.proj.2.bias": "mm.2.bias",
    }
    return table.get(name)


# checkpoint block-tensor suffix mapped to ik "v.blk.<il>." suffix
BLOCK_MAP = {
    "norm0.weight": "ln1.weight", "norm0.bias": "ln1.bias",     # pre-attn norm
    "norm1.weight": "ln2.weight", "norm1.bias": "ln2.bias",     # pre-mlp norm
    "wo.weight": "attn_out.weight", "wo.bias": "attn_out.bias",
    "mlp.fc0.weight": "ffn_up.weight", "mlp.fc0.bias": "ffn_up.bias",
    "mlp.fc1.weight": "ffn_down.weight", "mlp.fc1.bias": "ffn_down.bias",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vision_state(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load only vision_tower.* / mm_projector.* tensors from the shard(s) that hold them."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = read_json(index_path)["weight_map"]
        shards = sorted({weight_map[k] for k in weight_map
                         if k.startswith(("vision_tower.", "mm_projector."))})
    else:
        shards = [p.name for p in sorted(model_dir.glob("*.safetensors"))]

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        tensors = load_file(str(model_dir / shard), device="cpu")
        for k, v in tensors.items():
            if k.startswith(("vision_tower.", "mm_projector.")):
                state[k] = v
    return state


def add_tensor(writer: GGUFWriter, name: str, data: torch.Tensor, args) -> None:
    """Write one tensor honoring the storage flag.

    Only 2D linear weights are stored in the compressed type; conv kernels, position
    embeddings, norms and biases stay f32 (consumed by f32-only ggml kernels).
    """
    compress = data.ndim == 2 and name.endswith(".weight")
    if args.use_f32 or not compress:
        writer.add_tensor(name, data.float().numpy())
    elif args.use_bf16:
        if data.dtype == torch.bfloat16:
            writer.add_tensor(name, data.contiguous().view(torch.int16).numpy(),
                              raw_dtype=GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, data.float().numpy())
    else:  # f16 default
        writer.add_tensor(name, data.half().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kimi K2.7 vision encoder/projector to GGUF")
    parser.add_argument("-m", "--model-dir", required=True, help="Path to Kimi K2.7 HF model directory")
    parser.add_argument("-o", "--output", default=None, help="Output GGUF path")
    parser.add_argument("--use-f32", action="store_true", help="Write tensors as f32 instead of f16")
    parser.add_argument("--use-bf16", action="store_true", help="Write tensors as bf16 instead of f16")
    parser.add_argument("--bigendian", action="store_true", help="Write big-endian GGUF")
    args = parser.parse_args()
    if args.use_f32 and args.use_bf16:
        parser.error("--use-f32 and --use-bf16 are mutually exclusive")

    model_dir = Path(args.model_dir)
    config = read_json(model_dir / "config.json")
    vc = config["vision_config"]

    preprocessor = read_json(model_dir / "preprocessor_config.json")
    mpc = preprocessor.get("media_proc_cfg", preprocessor)

    n_head = int(vc["vt_num_attention_heads"])
    patch_size = int(vc["patch_size"])
    merge = vc["merge_kernel_size"]
    merge = merge[0] if isinstance(merge, (list, tuple)) else int(merge)
    pos_emb_h = int(vc.get("init_pos_emb_height", 64))
    in_patch_limit = int(mpc.get("in_patch_limit", 16384))
    ln_eps = float(vc.get("projector_ln_eps", 1e-5))

    output = Path(args.output) if args.output else model_dir / "mmproj-kimik25.gguf"
    ftype = 0 if args.use_f32 else 1

    writer = GGUFWriter(
        path=str(output),
        arch="clip",
        endianess=GGUFEndian.BIG if args.bigendian else GGUFEndian.LITTLE,
    )
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)
    writer.add_string("clip.projector_type", "kimik25")  # loader keys the graph off this string
    writer.add_string("general.name", "Kimi-K2.7 vision projector")
    writer.add_uint32("general.file_type", ftype)

    writer.add_uint32("clip.vision.image_size", pos_emb_h * patch_size)  # reference size, for compat
    writer.add_uint32("clip.vision.patch_size", patch_size)
    writer.add_uint32("clip.vision.embedding_length", int(vc["vt_hidden_size"]))
    writer.add_uint32("clip.vision.feed_forward_length", int(vc["vt_intermediate_size"]))
    writer.add_uint32("clip.vision.projection_dim", int(vc["text_hidden_size"]))
    writer.add_uint32("clip.vision.attention.head_count", n_head)
    writer.add_uint32("clip.vision.block_count", int(vc["vt_num_hidden_layers"]))
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", ln_eps)
    writer.add_uint32("clip.vision.projector.scale_factor", merge)  # loader's n_merge
    # min/max pixels bound ik's aspect-preserving resize
    writer.add_uint32("clip.vision.image_min_pixels", 8 * patch_size * patch_size)
    writer.add_uint32("clip.vision.image_max_pixels", in_patch_limit * patch_size * patch_size)
    writer.add_array("clip.vision.image_mean", [float(x) for x in mpc.get("image_mean", [0.5, 0.5, 0.5])])
    writer.add_array("clip.vision.image_std", [float(x) for x in mpc.get("image_std", [0.5, 0.5, 0.5])])
    writer.add_bool("clip.use_gelu", vc.get("projector_hidden_act", "gelu") == "gelu")

    state = load_vision_state(model_dir)
    written = 0

    for src in sorted(state):
        data = state[src]

        # static (non-block) tensors
        dst = rename_static(src)
        if dst is not None:
            add_tensor(writer, dst, data, args)
            written += 1
            continue

        if not src.startswith(ENCODER_PREFIX):
            continue  # ignore anything we don't explicitly map (e.g. rotary buffers)
        rest = src[len(ENCODER_PREFIX):]          # "<il>.<suffix>"
        il, suffix = rest.split(".", 1)
        blk = f"v.blk.{int(il)}."

        # fused QKV split into q/k/v, q/k permuted to build_rope_2d's layout
        if suffix == "wqkv.weight":
            qkv_dim = data.shape[0] // 3
            wq, wk, wv = data[:qkv_dim], data[qkv_dim:2 * qkv_dim], data[2 * qkv_dim:]
            add_tensor(writer, blk + "attn_q.weight", permute_qk_weight(wq, n_head), args)
            add_tensor(writer, blk + "attn_k.weight", permute_qk_weight(wk, n_head), args)
            add_tensor(writer, blk + "attn_v.weight", wv, args)
            written += 3
            continue
        if suffix == "wqkv.bias":
            qkv_dim = data.shape[0] // 3
            bq, bk, bv = data[:qkv_dim], data[qkv_dim:2 * qkv_dim], data[2 * qkv_dim:]
            add_tensor(writer, blk + "attn_q.bias", permute_qk_bias(bq, n_head), args)
            add_tensor(writer, blk + "attn_k.bias", permute_qk_bias(bk, n_head), args)
            add_tensor(writer, blk + "attn_v.bias", bv, args)
            written += 3
            continue

        mapped = BLOCK_MAP.get(suffix)
        if mapped is None:
            raise ValueError(f"unmapped vision tensor: {src}")
        add_tensor(writer, blk + mapped, data, args)
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output}  ({written} tensors)")


if __name__ == "__main__":
    main()
