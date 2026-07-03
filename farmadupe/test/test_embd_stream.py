"""Smoke tests for the stream-driven multimodal prefill in llama-server.

The server owns every llama_decode: per-slot mtmd embd streams drain image and
text rows into one shared batch per update_slots round, so these tests cover
cross-slot continuous batching, prefix-cache reuse over mirrored chunks, and
cancellation mid-prefill.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import subprocess
import time
from pathlib import Path

import pytest
import requests
from huggingface_hub import hf_hub_download
from openai import OpenAI
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SERVER_BIN = REPO_ROOT / "build/bin/llama-server"
COMPILE_SH = REPO_ROOT / "farmadupe/scripts/compile.sh"
MODEL_DIR = HERE / "models"
SERVER_LOG = HERE / "server_embd_stream.log"
HF_REPO = "bartowski/Qwen_Qwen3-VL-4B-Instruct-GGUF"
MODEL_FILE = "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf"
PORT = 18081
SERVER_URL = f"http://127.0.0.1:{PORT}"

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
    build = subprocess.run([str(COMPILE_SH)], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    if build.returncode != 0:
        pytest.exit("build failed:\n" + (build.stdout + build.stderr)[-4000:], returncode=1)

    # the compile script builds static (BUILD_SHARED_LIBS=OFF); a dynamically
    # linked server could silently run against stale shared libs
    ldd = subprocess.run(["ldd", str(SERVER_BIN)], capture_output=True, text=True)
    if "libllama" in ldd.stdout or "libggml" in ldd.stdout:
        pytest.exit("llama-server is not statically linked:\n" + ldd.stdout, returncode=1)

    # a hard-killed pytest leaks its server; a leftover answering our health
    # checks while the fresh server is still loading would shadow the new
    # binary and poison every result
    subprocess.run(["pkill", "-f", "build/bin/llama-server"])
    time.sleep(1)
    leftover = subprocess.run(["pgrep", "-af", "build/bin/llama-server"],
                              capture_output=True, text=True)
    if leftover.stdout.strip():
        pytest.exit("instance of build/llama-server found:\n" + leftover.stdout,
                    returncode=1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = hf_hub_download(HF_REPO, MODEL_FILE, local_dir=str(MODEL_DIR))
    mmproj = hf_hub_download(HF_REPO, MMPROJ_FILE, local_dir=str(MODEL_DIR))

    logf = SERVER_LOG.open("w")
    # Small n_batch so multi-image prompts span several rounds and images
    # straddle round boundaries; two slots so streams from different requests
    # drain into the same batch.
    _proc = subprocess.Popen([
        str(SERVER_BIN),
        "--model", model,
        "--mmproj", mmproj,
        "--parallel", "2",
        "--n-gpu-layers", "999",  # drop to 0 for CPU-only debugging
        "--ctx-size", "8192",
        "--batch-size", "128",
        "--ubatch-size", "64",
        "--port", str(PORT),
        "--reasoning", "off",
    ], stdout=logf, stderr=subprocess.STDOUT)

    try:
        for _ in range(180):
            if _proc.poll() is not None:
                pytest.exit(
                    f"llama-server exited during startup (rc={_proc.returncode}). "
                    f"Log:\n{_log_tail()}", returncode=1)
            try:
                if requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.exit(
                f"llama-server did not become healthy within 180s. Log:\n{_log_tail()}",
                returncode=1)

        # health returned 200: it must have been served by the process we
        # spawned, not a squatter that survived the sweep above
        if _proc.poll() is not None:
            pytest.exit(
                f"health check answered but our llama-server is dead "
                f"(rc={_proc.returncode}); a foreign server is on port {PORT}. "
                f"Log:\n{_log_tail()}", returncode=1)

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
    return OpenAI(base_url=f"{server}/v1", api_key="not-needed", timeout=600)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(img)).decode()


def _red_square_img(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    pad = size // 4
    ImageDraw.Draw(img).rectangle((pad, pad, size - pad, size - pad), fill="red")
    return img


def red_square_on_white(size: int = 256) -> str:
    return _png_url(_red_square_img(size))


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


# ~120 reps x ~9 tokens each, ~1080 tokens of filler. Many rounds at n_batch=128.
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


def test_health_endpoint_returns_200(server):
    assert requests.get(f"{server}/health", timeout=5).status_code == 200


def test_text_prompt_returns_nonempty_reply(client):
    # pure-text requests must take the token-round path, untouched by streaming
    assert chat(client, "Say hello in one short sentence.", 50).strip()


def test_red_square_image_identified_as_red_square(client):
    out = chat(client, [{"type": "image_url", "image_url": {"url": red_square_on_white()}}])
    assert "red" in out, out
    assert "square" in out, out


def test_white_background_identified_as_white(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "text", "text": "What colour is the background? Reply with one word."},
    ])
    assert "white" in out, out


def test_two_images_described_with_both_colours_and_shapes(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "image_url", "image_url": {"url": blue_triangle_on_black()}},
        {"type": "text", "text": "Describe both images."},
    ], max_tokens=400)
    for needle in ("red", "square", "blue", "triangle"):
        assert needle in out, (needle, out)


# Counting is a sharp probe for two separate bugs: the chat template (jinja)
# must expand one image placeholder per image so the model sees two, and the
# M-RoPE positions of the second image must be laid down correctly on top of
# the first. A miscount here usually means one of those two is wrong.
def test_two_images_counted_as_two(client):
    out = chat(client, [
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "image_url", "image_url": {"url": blue_triangle_on_black()}},
        {"type": "text", "text": "How many images are there? Reply with one word."},
    ], max_tokens=50)
    assert "two" in out or "2" in out, out


# Each Qwen3-VL 256x256 image contributes ~64 patches; with n_batch=128 the
# eight-image prompt spans several embd rounds and images straddle round
# boundaries, exercising mid-image stream suspension and M-RoPE positions
# written across separate drains.

def test_eight_images_each_colour_named_in_order(client):
    colors = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta")
    content = [{"type": "image_url", "image_url": {"url": colored_square(c)}} for c in colors]
    content.append({"type": "text", "text":
                    "Name the colour of each square in order. Reply with one word per line."})
    out = chat(client, content, max_tokens=200)
    for c in colors:
        assert c in out, (c, out)


def test_long_text_prefill_still_identifies_colour(client):
    out = chat(client, [
        {"type": "text", "text": long_filler("long-prefill")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
        {"type": "text", "text": "What colour is the shape in the image? Reply with one word."},
    ], max_tokens=50)
    assert "red" in out, out


def test_prompt_cache_repeated_prompt_warm_faster_than_cold(client):
    """Same prompt twice: the warm run reuses the cached KV state (image included,
    mirrored into cache_tokens chunk-atomically after decode) and finishes
    substantially faster than the cold run."""
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


def test_prompt_cache_divergent_suffix_returns_fresh_answer(client):
    """Two prompts share a long prefix but diverge at the suffix; the second prompt
    must be answered from its own suffix, not from the prefix's cached answer."""
    shared_prefix = [
        {"type": "text", "text": long_filler("cache-rewind-correctness")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
    ]

    out_a = chat(client, shared_prefix + [
        {"type": "text", "text": "What colour is the shape? Reply with one word."},
    ], max_tokens=10)
    assert "red" in out_a, out_a

    out_b = chat(client, shared_prefix + [
        {"type": "text", "text": "What shape is in the image? One word: square, circle, or triangle."},
    ], max_tokens=10)
    assert "square" in out_b, f"stale or wrong answer: {out_b!r}"


def test_prompt_cache_shared_prefix_reused_so_warm_faster_than_cold(client):
    """Two prompts share a long prefix and diverge only at the suffix: the second
    run reuses the cached prefix (image chunk included) and finishes faster than
    the cold first run. The perf companion to the divergent-suffix correctness
    test above: that one proves the answer is fresh, this one proves the prefix
    was actually reused rather than recomputed."""
    shared_prefix = [
        {"type": "text", "text": long_filler("cache-rewind-perf")},
        {"type": "image_url", "image_url": {"url": red_square_on_white()}},
    ]

    t0 = time.time()
    chat(client, shared_prefix + [
        {"type": "text", "text": "What colour is the shape? Reply with one word."},
    ], max_tokens=10)
    t_cold = time.time() - t0

    t0 = time.time()
    chat(client, shared_prefix + [
        {"type": "text", "text": "What shape is in the image? One word: square, circle, or triangle."},
    ], max_tokens=10)
    t_warm = time.time() - t0

    assert t_warm < t_cold * 0.7, f"expected partial-prefix reuse: cold={t_cold:.2f}s warm={t_warm:.2f}s"


def test_parallel_image_prefills_with_concurrent_generation(client):
    """Two multi-image requests plus a text request in flight at once: streams
    from different slots drain into the same embd rounds, and generation-token
    rows ride along converted through the input-embedding LUT."""
    def image_task(salt: str) -> str:
        return chat(client, [
            {"type": "text", "text": f"Request {salt}."},
            {"type": "image_url", "image_url": {"url": red_square_on_white()}},
            {"type": "image_url", "image_url": {"url": blue_triangle_on_black()}},
            {"type": "text", "text": "Name the colour of each shape in order, one word each."},
        ], max_tokens=60)

    def text_task() -> str:
        return chat(client, "Count from one to ten as words.", 80)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_img_a = pool.submit(image_task, "alpha")
        f_img_b = pool.submit(image_task, "beta")
        f_txt = pool.submit(text_task)
        out_a = f_img_a.result()
        out_b = f_img_b.result()
        out_t = f_txt.result()

    for out in (out_a, out_b):
        assert "red" in out, out
        assert "blue" in out, out
    assert "three" in out_t or "3" in out_t, out_t


def test_cancel_mid_prefill_then_retry_answers_correctly(client, server):
    """Disconnect while a large multimodal prompt is prefilling, then resend it.
    The rewind must land on a chunk boundary: the retry reuses whatever prefix
    was mirrored and still answers from the image."""
    colors = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta")
    content = [{"type": "text", "text": long_filler("cancel-retry")}]
    content += [{"type": "image_url", "image_url": {"url": colored_square(c)}} for c in colors]
    content.append({"type": "text", "text": "Name the colour of the first square. One word."})
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 20,
        "temperature": 0.0,
    }

    # abort mid-prefill: the read timeout fires while the ~1500-token prefill
    # is still in flight (0.2s is mid-prefill even on a GPU), closing the
    # connection
    with pytest.raises(requests.RequestException):
        requests.post(f"{server}/v1/chat/completions", json=payload, timeout=(5, 0.2))

    time.sleep(1)  # let the server observe the disconnect
    _require_alive()

    out = chat(client, content, max_tokens=20)
    assert "red" in out, out
    _require_alive()
