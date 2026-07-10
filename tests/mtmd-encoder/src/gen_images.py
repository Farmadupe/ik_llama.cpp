"""Deterministic test-image generator for the parity harnesses; see README.md."""
from pathlib import Path

import numpy as np
from PIL import Image

# Named real photos, resolved relative to the repo root the caller passes in.
REAL_SUBPATHS = {
    "test-1":       "examples/mtmd/test-1.jpeg",
    "llama-leader": "media/llama-leader.jpeg",
}

SYNTHETIC = ("rgb_gradient_h", "rgb_gradient_v", "bw_checkerboard",
             "quadrants", "rings", "noise")

# Default image set shared by every family's HARNESS_CONFIG for now: the widest
# assortment on hand - each synthetic pattern at two sizes plus both real photos.
# A family that needs its own set later overrides "default_images" in its config.
#
# Sizes are deliberately not multiples of the patch size (14). A grid-aligned
# image resizes at scale 1.0 (identity), which hides any divergence between ik's
# resize and the HF processor's resize; these jittered sizes (each dim of a round
# base perturbed within +/-36) force a real rescale so the resize path is
# actually exercised. Do not round them back to tidy multiples.
DEFAULT_IMAGES = [
    "rgb_gradient_h:203:217",
    "rgb_gradient_v:309:213",
    "bw_checkerboard:211:327",
    "quadrants:287:274",
    "rings:470:220",
    "noise:148:173",
    "rgb_gradient_h:87:38",
    "rgb_gradient_v:662:648",
    "bw_checkerboard:135:83",
    "quadrants:100:146",
    "rings:577:141",
    "test-1:635:508",
    "llama-leader:642:656",
]

CHECKER_PERIOD = 9  # fixed cosmetic period
NOISE_SEED = 1       # fixed seed: every noise image is deterministic and reproducible


def synthetic(name: str, w: int, h: int) -> np.ndarray:
    """Return an [h, w, 3] float32 image in [0, 1] for the named synthetic pattern.

    Structure makes axis/patch scrambling visible; distinctness across names makes
    any file mispairing show up in compare_embeddings."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / max(w - 1, 1), yy / max(h - 1, 1)
    if name == "rgb_gradient_h":
        img = np.stack([xn, 1 - xn, np.full_like(xn, 0.5)], axis=-1)
    elif name == "rgb_gradient_v":
        img = np.stack([yn, np.full_like(yn, 0.3), 1 - yn], axis=-1)
    elif name == "bw_checkerboard":
        c = (((xx // CHECKER_PERIOD).astype(int) + (yy // CHECKER_PERIOD).astype(int)) % 2).astype(np.float32)
        img = np.stack([c, 1 - c, (xn + yn) / 2], axis=-1)
    elif name == "quadrants":
        img = np.zeros((h, w, 3), np.float32)
        img[: h // 2, : w // 2] = [0.9, 0.1, 0.1]
        img[: h // 2, w // 2:] = [0.1, 0.9, 0.1]
        img[h // 2:, : w // 2] = [0.1, 0.1, 0.9]
        img[h // 2:, w // 2:] = [0.9, 0.9, 0.1]
    elif name == "rings":
        cx, cy = xn - 0.5, yn - 0.5
        r = np.sqrt(cx * cx + cy * cy)
        rings = 0.5 + 0.5 * np.sin(r * 40.0)
        img = np.stack([rings, (1 - rings), rings * xn], axis=-1)
    elif name == "noise":
        rng = np.random.default_rng(NOISE_SEED)
        base = rng.random((h // CHECKER_PERIOD + 1, w // CHECKER_PERIOD + 1, 3)).astype(np.float32)
        img = np.asarray(
            Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
    else:
        raise ValueError(
            f"unknown image name: {name!r}\n"
            f"  synthetic patterns: {', '.join(SYNTHETIC)}\n"
            f"  real photos:        {', '.join(sorted(REAL_SUBPATHS))}")
    return np.clip(img, 0.0, 1.0)


def build(name: str, w: int, h: int, repo_root: Path) -> Image.Image:
    """Build the W x H RGB image for a synthetic pattern or a registered real photo."""
    if name in REAL_SUBPATHS:
        src = repo_root / REAL_SUBPATHS[name]
        if not src.exists():
            raise FileNotFoundError(f"real photo {name!r} source not found: {src}")
        return Image.open(src).convert("RGB").resize((w, h), Image.BICUBIC)
    arr = (synthetic(name, w, h) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def parse_spec(spec: str):
    """Parse a "<name>:<W>:<H>" spec into (name, w, h)."""
    try:
        name, w_s, h_s = spec.rsplit(":", 2)
        w, h = int(w_s), int(h_s)
    except ValueError:
        raise ValueError(f"bad spec {spec!r}: expected <name>:<W>:<H>")
    if w <= 0 or h <= 0:
        raise ValueError(f"bad spec {spec!r}: W and H must be positive")
    return name, w, h


def generate(spec: str, out: Path, repo_root: Path) -> None:
    """Build the image for one "<name>:<W>:<H>" spec and write it to out."""
    name, w, h = parse_spec(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(name, w, h, repo_root).save(out)
    print(f"wrote {out}  ({name} {w}x{h})")
