#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "pytest~=8.4",
#     "openai~=1.60",
#     "Pillow~=11.0",
#     "huggingface_hub~=0.27.0",
#     "requests~=2.32",
# ]
# ///
"""Smoke tests for the coalesced multimodal prefill path in llama-server."""

from __future__ import annotations

import base64
import io
import subprocess
import sys
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
SERVER_LOG = HERE / "server.log"
HF_REPO = "bartowski/Qwen_Qwen3-VL-4B-Instruct-GGUF"
MODEL_FILE = "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf"
SERVER_URL = "http://127.0.0.1:18080"

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
        "--ctx-size", "4096",
        "--batch-size", "512",
        "--ubatch-size", "64",
        "--port", "18080",
        "--reasoning", "off",
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


def _png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def red_square_on_white(size: int = 256) -> str:
    img = Image.new("RGB", (size, size), "white")
    pad = size // 4
    ImageDraw.Draw(img).rectangle((pad, pad, size - pad, size - pad), fill="red")
    return _png_url(img)


def blue_triangle_on_black(size: int = 256) -> str:
    img = Image.new("RGB", (size, size), "black")
    pad = size // 8
    ImageDraw.Draw(img).polygon(
        [(size // 2, pad), (pad, size - pad), (size - pad, size - pad)], fill="blue")
    return _png_url(img)


def colored_square(color: str, bg: str = "white", size: int = 256) -> str:
    img = Image.new("RGB", (size, size), bg)
    pad = size // 4
    ImageDraw.Draw(img).rectangle((pad, pad, size - pad, size - pad), fill=color)
    return _png_url(img)


# ~120 reps × ~9 tokens each ≈ 1080 tokens of filler. Comfortably above n_batch=512.
def long_filler(salt: str) -> str:
    """Per-test unique long filler so each test's first run hits a cold KV cache."""
    return f"Test scenario {salt}. " + "The quick brown fox jumps over the lazy dog. " * 120


def chat(client: OpenAI, content, max_tokens: int = 200) -> str:
    _require_alive()
    try:
        resp = client.chat.completions.create(
            model="local",
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception:
        _require_alive()  # surface server death instead of generic connection error
        raise
    return (resp.choices[0].message.content or "").lower()


def test_server_health(server):
    assert requests.get(f"{server}/health", timeout=5).status_code == 200


def test_text_completion(client):
    assert chat(client, "Say hello in one short sentence.", 50).strip()


def test_red_square(client):
    out = chat(client, [{"type": "image_url", "image_url": {"url": red_square_on_white()}}])
    assert "red" in out, out
    assert "square" in out, out


def test_background_question(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "text", "text": "What colour is the background? Reply with one word."},
    ])
    assert "white" in out, out


def test_two_images_describe(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "image_url", "image_url": {"url": blue_triangle_on_black()}},
        {"type": "text", "text": "Describe both images."},
    ], max_tokens=400)
    for needle in ("red", "square", "blue", "triangle"):
        assert needle in out, (needle, out)


def test_two_images_count(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "image_url", "image_url": {"url": blue_triangle_on_black()}},
        {"type": "text", "text": "How many images are there? Reply with one word."},
    ], max_tokens=50)
    assert "two" in out or "2" in out, out


# The tests below force the coalesced batch to exceed n_batch=512, so they exercise
# the get_view loop in mtmd_helper_eval_coalesced. Each Qwen3.5-VL 256x256 image
# contributes ~64 patches, so 8 images alone push us above the boundary.

def test_many_images_colours(client):
    colors = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta")
    content = [{"type": "image_url", "image_url": {"url": colored_square(c)}} for c in colors]
    content.append({"type": "text", "text":
                    "Name the colour of each square in order. Reply with one word per line."})
    out = chat(client, content, max_tokens=200)
    for c in colors:
        assert c in out, (c, out)


def test_long_text_prefill(client):
    out = chat(client, [
        {"type": "text", "text": long_filler("long-prefill")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "text", "text": "What colour is the shape in the image? Reply with one word."},
    ], max_tokens=50)
    assert "red" in out, out


def test_prompt_cache_reuse(client):
    """Same prompt twice: the warm run should reuse the cached KV state (image included)
    and finish substantially faster than the cold run.
    Note: requires a non-recurrent-state, non-SWA model for the upstream checkpoint
    mechanism to restore the KV cache properly."""
    msg = [
        {"type": "text", "text": long_filler("cache-reuse")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "text", "text": "What colour is the shape? Reply with one word."},
    ]

    t0 = time.time()
    chat(client, msg, max_tokens=20)
    t_cold = time.time() - t0

    t0 = time.time()
    chat(client, msg, max_tokens=20)
    t_warm = time.time() - t0

    assert t_warm < t_cold * 0.6, f"expected cache reuse: cold={t_cold:.2f}s warm={t_warm:.2f}s"


def test_prompt_cache_rewind(client):
    """Two prompts sharing a long prefix; second prompt diverges at the suffix.
    Verifies (a) the shared prefix is reused (warm < cold), and (b) the model
    correctly answers the divergent suffix (not a stale cached answer)."""
    shared_prefix = [
        {"type": "text", "text": long_filler("cache-rewind")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
    ]

    t0 = time.time()
    out_a = chat(client, shared_prefix + [
        {"type": "text", "text": "What colour is the shape? Reply with one word."},
    ], max_tokens=10)
    t_cold = time.time() - t0
    assert "red" in out_a.lower(), out_a

    t0 = time.time()
    out_b = chat(client, shared_prefix + [
        {"type": "text", "text": "What shape is in the image? One word: square, circle, or triangle."},
    ], max_tokens=10)
    t_warm = time.time() - t0

    assert "square" in out_b.lower(), f"stale or wrong answer: {out_b!r}"
    assert t_warm < t_cold * 0.7, f"expected partial-prefix reuse: cold={t_cold:.2f}s warm={t_warm:.2f}s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
