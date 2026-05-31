"""m-rope spatial-localisation regression test (bounding-box oracle).

Why this exists
---------------
When m-rope positions are scrambled across a ubatch boundary, a Qwen3-VL /
Qwen3.5-VL model keeps *what* is in the image but loses *where*: the (h, w)
position components are read from the wrong offsets, so the spatial layout is
corrupted. A coarse "is the colour present" assertion does not catch this -- the
model still says "black square" -- but a bounding-box centroid check does,
because a scrambled grid sends the box to the wrong location.

This is the precise failure mode of the bug fixed by the AoS `llama_mrope_pos`
change: with the old SoA layout, `batch.pos + cur_token` mis-sliced the h/w
sections whenever an image straddled a 64-token ubatch boundary.

Strategy
--------
  * Render a white image on the 32px merged-cell grid (one cell == one image
    token). Fill exactly one cell black, so exactly one image token is black and
    its (h, w) position is the only thing the model must localise.
  * Ask the model for the bbox_2d of the black cell.
  * PASS iff the centroid of the returned box lands inside the real cell
    (with a one-cell tolerance).
  * Sweep cell positions (deterministic RNG) and ubatch *phase* offsets
    (variable-length low-salience filler prepended before the image) so the
    image rows straddle the 64-token ubatch boundary in many alignments.

Server config
-------------
Same model/CPU config as test_coalesced.py. Launched with --batch-size 512
--ubatch-size 64 and a 1024px image (32x32 = 1024 image tokens => 16 ubatches),
so every image decode is split across ubatches inside llama_decode -- the exact
path that was broken before the fix. Pre-fix these assertions drift; post-fix
the centroid lands in the cell regardless of phase.
"""

from __future__ import annotations

import base64
import io
import math
import random
import re
import subprocess
import time
from pathlib import Path

import pytest
import requests
from huggingface_hub import hf_hub_download
from openai import OpenAI
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SERVER_BIN = HERE.parents[1] / "build/bin/llama-server"
MODEL_DIR = HERE / "models"
SERVER_LOG = HERE / "server_mrope.log"
HF_REPO = "bartowski/Qwen_Qwen3-VL-4B-Instruct-GGUF"
MODEL_FILE = "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf"
SERVER_URL = "http://127.0.0.1:18081"

CELL = 32           # merged ViT cell (16px patch x 2x2 merge) = one image token
UBATCH = 64         # must match --ubatch-size below
THRESHOLD_TILES = 1.0   # pass iff centroid is within this many tiles of the cell

# Every run writes an HTML report (no CSS) of every case here.
REPORT_FILE = HERE / "mrope_bbox_report.html"
RESULTS: list[dict] = []

_proc: subprocess.Popen | None = None


def _log_tail(n: int = 4000) -> str:
    return SERVER_LOG.read_text()[-n:] if SERVER_LOG.exists() else "(no server log)"


def _require_alive() -> None:
    if _proc is None or _proc.poll() is not None:
        rc = _proc.returncode if _proc else "unstarted"
        pytest.fail(f"llama-server is not running (rc={rc}). Log tail:\n{_log_tail()}")


@pytest.fixture(scope="session")
def server():
    global _proc
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = hf_hub_download(HF_REPO, MODEL_FILE, local_dir=str(MODEL_DIR))
    mmproj = hf_hub_download(HF_REPO, MMPROJ_FILE, local_dir=str(MODEL_DIR))

    logf = SERVER_LOG.open("w")
    _proc = subprocess.Popen([
        str(SERVER_BIN),
        "--model", model,
        "--mmproj", mmproj,
        "--parallel", "1",
        "--n-gpu-layers", "0",
        "--ctx-size", "8192",
        "--batch-size", "512",
        "--ubatch-size", str(UBATCH),
        "--port", "18081",
        "--reasoning", "off",
        "--temp", "0",
    ], stdout=logf, stderr=subprocess.STDOUT)

    try:
        for _ in range(180):
            if _proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (rc={_proc.returncode}). "
                    f"Log:\n{_log_tail()}")
            try:
                if requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(
                f"llama-server did not become healthy within 180s. Log:\n{_log_tail()}")

        # Guard against a false pass: this test only exercises the m-rope bug if
        # the image decode is actually split across ubatches (n_ubatch < n_batch).
        # If anything forces n_ubatch == n_batch (e.g. the old get_batch_ubatch
        # mtmd workaround) the split never happens and every case passes trivially.
        log = SERVER_LOG.read_text()
        assert "Adjust batch size for mtmd" not in log, (
            "n_ubatch was forced to n_batch (mtmd batch-size workaround active) -- "
            "the ubatch split this test depends on is suppressed")
        m_b = re.search(r"n_batch\s*=\s*(\d+)", log)
        m_u = re.search(r"n_ubatch\s*=\s*(\d+)", log)
        assert m_b and m_u, f"could not read n_batch/n_ubatch from server log:\n{_log_tail()}"
        n_batch, n_ubatch = int(m_b.group(1)), int(m_u.group(1))
        assert n_ubatch < n_batch, (
            f"n_ubatch ({n_ubatch}) must be < n_batch ({n_batch}) so the image "
            f"straddles ubatch boundaries; n_ubatch == n_batch means no split")
        yield SERVER_URL
    finally:
        if _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait()
        logf.close()


@pytest.fixture(scope="session")
def client(server):
    return OpenAI(base_url=f"{server}/v1", api_key="not-needed")


# --------------------------------------------------------------------------- #
# Image generation                                                            #
# --------------------------------------------------------------------------- #

def _png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def black_square_on_white_grid(grid_cells: tuple[int, int],
                               square_cell: tuple[int, int]):
    """White image of (grid_w x grid_h) 32px cells with a single black cell at
    cell coordinate `square_cell`. One cell == one image token, so exactly one
    token is black -- its (h, w) position is what m-rope must localise.

    Returns (img, square_rect_norm, (W, H)) where img is the PIL image and
    square_rect_norm is the black cell's (x1, y1, x2, y2) in normalised [0, 1]
    image coordinates.
    """
    gw, gh = grid_cells
    sx, sy = square_cell
    assert 0 <= sx < gw and 0 <= sy < gh
    W, H = gw * CELL, gh * CELL
    img = Image.new("RGB", (W, H), "white")
    x0, y0 = sx * CELL, sy * CELL
    x1, y1 = x0 + CELL, y0 + CELL
    ImageDraw.Draw(img).rectangle((x0, y0, x1 - 1, y1 - 1), fill="black")
    rect_norm = (x0 / W, y0 / H, x1 / W, y1 / H)
    return img, rect_norm, (W, H)


def _img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _txt(t):
    return {"type": "text", "text": t}


def _annotate_b64(img: Image.Image, box_norm) -> str:
    """Return a base64 PNG of the image with the model's predicted box drawn in
    red. The black cell is the ground truth; a pass has the red box over it, a
    fail has it drifted off. Embedded directly into the HTML report."""
    annotated = img.copy()
    if box_norm is not None:
        W, H = annotated.size
        rect = (box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H)
        ImageDraw.Draw(annotated).rectangle(rect, outline=(255, 0, 0), width=4)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ubatch_phase_filler(n_chunks: int) -> str:
    """Low-salience alternating tokens prepended before the image to shift it
    off the ubatch grid. Distinct chars + spaces keep ~1 token each so the
    image's start position advances roughly one token per chunk."""
    return "".join(". " if i % 2 == 0 else ", " for i in range(n_chunks))


# --------------------------------------------------------------------------- #
# Bounding-box parsing (tolerant to Qwen's coordinate conventions)            #
# --------------------------------------------------------------------------- #

def parse_bbox(text: str):
    """Return [x1, y1, x2, y2] from the reply, or None.

    Match the first run of four comma-separated integers -- that's the box. We
    deliberately key on the commas so the digit in the "bbox_2d" key name isn't
    mistaken for a coordinate.
    """
    m = re.search(r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)', text)
    if not m:
        return None
    return [int(g) for g in m.groups()]


def normalise_box(box):
    """Qwen3-VL grounding returns normalised 0-1000 coordinates (relative to the
    image, resize-independent -- see the official 2d_grounding cookbook). Map to
    [0, 1]; return None if the box is degenerate or out of range."""
    x1, y1, x2, y2 = (v / 1000.0 for v in box)
    if x2 <= x1 or y2 <= y1 or not all(-0.05 <= v <= 1.05 for v in (x1, y1, x2, y2)):
        return None
    return (x1, y1, x2, y2)


PROMPT = (
    "The image is white with exactly one solid black square on it. "
    "Output ONLY its bounding box as JSON on a single line: "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "black square"}.'
)


def _localise(client, grid, sx, sy, offset, label) -> dict:
    img, rect, _ = black_square_on_white_grid((grid, grid), (sx, sy))
    content = []
    if offset:
        content.append(_txt(ubatch_phase_filler(offset)))
    content.append(_img(_png_url(img)))
    content.append(_txt(PROMPT))

    _require_alive()
    resp = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": content}],
        max_tokens=64,
        temperature=0.0,
    )
    out = (resp.choices[0].message.content or "").strip()
    box = parse_bbox(out)
    norm = normalise_box(box) if box is not None else None

    # Distance (in tiles) from the predicted box centroid to the target cell
    # centroid. 1 tile == 1/grid in normalised coords, so scale back up by grid.
    if norm is not None:
        cx, cy = (norm[0] + norm[2]) / 2, (norm[1] + norm[3]) / 2
        scx, scy = (sx + 0.5) / grid, (sy + 0.5) / grid
        distance = grid * math.hypot(cx - scx, cy - scy)
        passed = distance < THRESHOLD_TILES
    else:
        distance = None
        passed = False

    rec = {
        "label": label,
        "tiles": grid * grid,
        "target": (sx, sy),
        "bbox": box,
        "distance": distance,
        "passed": passed,
        "out": out,
        "img_b64": _annotate_b64(img, norm),
    }
    RESULTS.append(rec)
    return rec


# 32x32 cells * 32px = 1024px image => 32x32 = 1024 image tokens => 16 ubatches at ub=64.
GRID = 32


def _cases(n=10, seed=20240531):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        sx = rng.randint(0, GRID - 1)
        sy = rng.randint(0, GRID - 1)
        # offset in [0, UBATCH] so the image start sweeps a full ubatch period
        offset = rng.randint(0, UBATCH)
        cases.append(pytest.param(sx, sy, offset, id=f"sq{sx}-{sy}_off{offset}"))
    return cases


@pytest.mark.parametrize("sx,sy,offset", _cases())
def test_mrope_bbox_localisation(client, sx, sy, offset):
    """The returned bbox centroid must land inside the real black cell, for every
    cell position and ubatch phase offset. A scrambled m-rope grid fails this."""
    label = f"sq{sx}-{sy}_off{offset}"
    r = _localise(client, GRID, sx, sy, offset, label)
    assert r["distance"] is not None, f"no/invalid bbox parsed ({label}): {r['out']!r}"
    assert r["passed"], (
        f"centroid distance {r['distance']:.3g} tiles >= {THRESHOLD_TILES} from "
        f"target {r['target']} ({label}): {r['out']!r}")


def test_mrope_bbox_no_offset_baseline(client):
    """Sanity baseline with no text offset: the 1024-token image spans 16 ubatches
    on its own, so this alone exercises the inner split."""
    r = _localise(client, GRID, GRID // 2, GRID // 2, 0, "baseline")
    assert r["distance"] is not None, f"no/invalid bbox parsed (baseline): {r['out']!r}"
    assert r["passed"], (
        f"centre cell not localised: distance {r['distance']:.3g} tiles "
        f">= {THRESHOLD_TILES}: {r['out']!r}")


# --------------------------------------------------------------------------- #
# HTML report (no CSS) -- written once at the end of the session              #
# --------------------------------------------------------------------------- #

def _fmt_dist(d) -> str:
    return "&mdash;" if d is None else f"{d:.3g}"


def _fmt_bbox(b) -> str:
    return "&mdash;" if b is None else "(" + ", ".join(str(v) for v in b) + ")"


def _write_report(results: list[dict]) -> None:
    def img_block(r: dict) -> str:
        return (f'<div><img src="data:image/png;base64,{r["img_b64"]}" width="512">'
                f"<p>distance: {_fmt_dist(r['distance'])} tiles</p></div>")

    p = ["<html><head><meta charset='utf-8'><title>m-rope bbox results</title>"
         "</head><body bgcolor='gray'>"]

    p.append("<h1>Results</h1>")
    p.append("<table border='1' cellpadding='4'>")
    p.append("<tr><th>tiles</th><th>target</th><th>bbox</th>"
             "<th>distance</th><th>passed</th></tr>")
    for r in results:
        sx, sy = r["target"]
        p.append("<tr>"
                 f"<td>{r['tiles']}</td>"
                 f"<td>({sx}, {sy})</td>"
                 f"<td>{_fmt_bbox(r['bbox'])}</td>"
                 f"<td>{_fmt_dist(r['distance'])}</td>"
                 f"<td>{'&#9989;' if r['passed'] else '&#10060;'}</td>"
                 "</tr>")
    p.append("</table>")

    p.append("<h1>Failures</h1>")
    p += [img_block(r) for r in results if not r["passed"]]

    p.append("<h1>Passes</h1>")
    p += [img_block(r) for r in results if r["passed"]]

    p.append("</body></html>")
    REPORT_FILE.write_text("\n".join(p), encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def _html_report():
    """Always emit the HTML report at session end, whatever the outcomes."""
    yield
    _write_report(RESULTS)
