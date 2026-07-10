#!/usr/bin/env python3
"""Entry point for the mtmd-encoder parity harnesses; see README.md.

    uv run python run_all.py --model <family> [args]
"""
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent      # tests/mtmd-encoder
REPO = HERE.parent.parent                   # repo root
BUILD = REPO / "build"
SRC = HERE / "src"

sys.path.insert(0, str(HERE))  # so `import src.*` resolves regardless of cwd
from src import gen_images, parity_report  # noqa: E402
from src.compare_embeddings import ImageEmbeddingSequence  # noqa: E402

QUANTS = ["f16", "bf16", "f32"]
CONVERT_FLAG = {"f16": [], "bf16": ["--use-bf16"], "f32": ["--use-f32"]}
CUDACXX = "/usr/local/cuda-13.2/bin/nvcc"
CUDA_HOME = str(Path(CUDACXX).parent.parent)  # /usr/local/cuda-13.2

# Build env for uv sync's one from-source dependency (flash-attn, kimik25 only):
# pin nvcc 13.2 (13.0/13.1 miscompile the fp16 rsqrt path), build only the sm_80
# kernels (they run on the RTX 3090's sm_86), and cap parallel nvcc jobs so the
# compile stays within host RAM (the default job count OOMs a 64G box).
SYNC_ENV = {
    "CUDA_HOME": CUDA_HOME,
    "PATH": f"{CUDA_HOME}/bin:" + os.environ.get("PATH", ""),
    "MAX_JOBS": "8",
    "FLASH_ATTN_CUDA_ARCHS": "80",
}

# src modules that are shared infrastructure, not a projector family.
SHARED = {"__init__", "gen_images", "compare_embeddings", "parity_report"}
FAMILIES = sorted(p.stem for p in SRC.glob("*.py") if p.stem not in SHARED)


def child_env(extra):
    """Environment for a harness subprocess: drop VIRTUAL_ENV so each harness's own
    `uv run` selects that harness's .venv, not the orchestrator's."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    if extra:
        env.update(extra)
    return env


def run(cmd, cwd, env, quiet):
    printable = " ".join(str(c) for c in cmd)
    print(f"+ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True,
                   stdout=subprocess.DEVNULL if quiet else None)


@dataclass
class Ctx:
    fam: str
    cfg: dict
    args: argparse.Namespace
    harness: Path
    reference_dir: Path
    target: str
    binary: Path

    def mmproj(self, quant: str) -> Path:
        stem = f"mmproj-{self.fam}" + {"f16": "", "bf16": "-bf16", "f32": "-f32"}[quant]
        return self.harness / f"{stem}.gguf"


def generate_images(ctx: Ctx, specs) -> None:
    img_dir = ctx.harness / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    for spec in specs:
        gen_images.generate(spec, img_dir / (spec.replace(":", "_") + ".png"), REPO)


def fetch_reference(ctx: Ctx) -> None:
    """Fetch the reference shards + metadata into REFERENCE_DIR. Idempotent: fetches
    on first run, no-op once every file is present."""
    ctx.reference_dir.mkdir(parents=True, exist_ok=True)
    files = ctx.cfg["ref_files"]
    if all((ctx.reference_dir / f).is_file() and (ctx.reference_dir / f).stat().st_size > 0
           for f in files):
        return
    repo = ctx.cfg["ref_repo"]
    for f in files:
        print(f">>> {f}")
        run(["wget", "-c", "-O", str(ctx.reference_dir / f),
             f"https://huggingface.co/{repo}/resolve/main/{f}"],
            cwd=None, env=None, quiet=False)


def build_wrapper(ctx: Ctx) -> None:
    """Configure + build the ik_encode_<family> parity wrapper (CUDA build)."""
    env = child_env({"CUDACXX": CUDACXX})
    run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DGGML_CUDA=ON", "-DBUILD_SHARED_LIBS=OFF",
         f"-DLLAMA_BUILD_{ctx.fam.upper()}_PARITY=ON"], cwd=None, env=env, quiet=True)
    run(["cmake", "--build", str(BUILD), "--target", ctx.target, "-j", str(os.cpu_count() or 1)],
        cwd=None, env=env, quiet=False)


def convert(ctx: Ctx, quant: str) -> Path:
    """Build the mmproj GGUF for one storage quant if it is not already present.
    Runs in the harness venv (needs gguf + torch + safetensors)."""
    mmproj = ctx.mmproj(quant)
    if not mmproj.exists():
        run(["uv", "run", "python", ctx.cfg["convert"],
             "-m", str(ctx.reference_dir), "-o", str(mmproj), *CONVERT_FLAG[quant]],
            cwd=ctx.harness, env=child_env(None), quiet=False)
    return mmproj


def encode(ctx: Ctx, mmproj: Path) -> None:
    """Run both drivers, ik_encode first. With --ik-image-preprocessing, ik_encode
    wipes and refills /tmp/<fam>_pre, which ref_encode.py (same flag) then encodes
    instead of preprocessing the images itself, so the per-key pairing holds."""
    images = sorted((ctx.harness / "images").glob("*.png"))
    img_args = [str(p.relative_to(ctx.harness)) for p in images]  # "images/<name>.png", cwd=harness

    ik_flags = []
    if ctx.args.cpu:
        ik_flags.append("--cpu")
    if ctx.args.ik_image_preprocessing:
        ik_flags.append("--ik-image-preprocessing")
    run([str(ctx.binary), *ik_flags, str(mmproj), *img_args], cwd=ctx.harness, env=None, quiet=False)

    ref_flags = ["--ik-image-preprocessing"] if ctx.args.ik_image_preprocessing else []
    ref_env = {"REFERENCE_DIR": str(ctx.reference_dir)}
    if ctx.args.cpu:
        ref_env["REF_DEVICE"] = "cpu"  # only read by families whose reference has a GPU path
    run(["uv", "run", "python", "ref_encode.py", *ref_flags],
        cwd=ctx.harness, env=child_env(ref_env), quiet=False)


def display_cmd(ctx: Ctx, quant: str):
    """The command echoed into parity_results.md for this quant."""
    cmd = ["python", "run_all.py", "--model", ctx.fam, "--quant", quant]
    if ctx.args.cpu:
        cmd.append("--cpu")
    if ctx.args.ik_image_preprocessing:
        cmd.append("--ik-image-preprocessing")
    return cmd


def compare(ref_path: Path, ik_path: Path, label: str) -> dict:
    """Load both embedding dumps, pair them per key, and return the per-key metrics
    consumed by parity_report. label is the key column header ("image", or "view"
    for per-view harnesses). A ref-vs-ik shape mismatch is recorded as an error row
    instead of being compared. See README.md."""
    ref = json.loads(Path(ref_path).read_text())
    ik = json.loads(Path(ik_path).read_text())

    only_ref = sorted(set(ref) - set(ik))
    only_ik = sorted(set(ik) - set(ref))
    if only_ref:
        print(f"WARNING: only in reference: {only_ref}")
    if only_ik:
        print(f"WARNING: only in ik: {only_ik}")

    shared = sorted(set(ref) & set(ik))
    # width fits the header and the longest key so the table stays aligned
    # regardless of whether keys are image basenames or longer <image>.<view>.
    width = max([len(label)] + [len(n) for n in shared])

    metrics = {}
    print(f"{label:<{width}} {'min_cosim':>16} {'worst_rel_l2':>16}")
    for n in shared:
        ref_seq = ImageEmbeddingSequence(ref[n])
        ik_seq = ImageEmbeddingSequence(ik[n])
        try:
            cosim_min = float(ref_seq.cosim(ik_seq).min())
            worst_rel_l2 = float(ref_seq.rel_l2(ik_seq).max())
        except AssertionError:
            ref_shape, ik_shape = ref_seq.shape(), ik_seq.shape()
            metrics[n] = {"error": "shape_mismatch",
                          "ref_shape": list(ref_shape), "ik_shape": list(ik_shape)}
            print(f"{n:<{width}} SHAPE MISMATCH ref={list(ref_shape)} ik={list(ik_shape)}")
            continue
        metrics[n] = {"per_token_cosim_min": cosim_min, "per_token_l2_worst": worst_rel_l2}
        print(f"{n:<{width}} {cosim_min:>16.10f} {worst_rel_l2:>16.10f}")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=FAMILIES)
    ap.add_argument("--image", dest="images", nargs="+", metavar="NAME:W:H", default=None)
    ap.add_argument("--quant", dest="quants", nargs="+", choices=QUANTS, default=["f32"],
                    metavar="QUANT")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ik-image-preprocessing", action="store_true")
    args = ap.parse_args()

    cfg = importlib.import_module(f"src.{args.model}").HARNESS_CONFIG

    ctx = Ctx(
        fam=args.model,
        cfg=cfg,
        args=args,
        harness=HERE / args.model,
        reference_dir=Path(f"/tmp/{args.model}_reference"),
        target=f"ik_encode_{args.model}",
        binary=BUILD / "bin" / f"ik_encode_{args.model}",
    )

    images = args.images if args.images is not None else cfg["default_images"]
    generate_images(ctx, list(dict.fromkeys(images)))  # dedupe, keep order

    fetch_reference(ctx)
    run(["uv", "sync"], cwd=ctx.harness, env=child_env(SYNC_ENV), quiet=False)
    build_wrapper(ctx)

    quants = list(dict.fromkeys(args.quants))  # dedupe, keep order
    cmds = {v: display_cmd(ctx, v) for v in quants}
    results = {}
    for quant in quants:
        print(f"### {' '.join(cmds[quant])}", flush=True)
        mmproj = convert(ctx, quant)
        encode(ctx, mmproj)
        results[quant] = compare(
            ctx.harness / "reference_embeddings.json",
            ctx.harness / "ik_embeddings.json",
            cfg.get("compare_label", "image"),
        )

    parity_report.write_parity_results(ctx.harness / "parity_results.md", quants, cmds, results)


if __name__ == "__main__":
    main()
