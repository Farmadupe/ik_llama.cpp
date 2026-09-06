# mtmd vision encoder parity tests

Each subdirectory here is a standalone parity harness for one multimodal
projector (vision tower + mmproj). A harness runs two encoders over the same
images - the HuggingFace reference implementation and the ik_llama.cpp mmproj
port - and reports per-token cosine similarity and L2 error between the two
sets of image embeddings.

Each harness is named after its mtmd projector suffix (the lowercased
`PROJECTOR_TYPE_*` from `examples/mtmd/clip.cpp`), and every artifact it owns -
directory, build target, mmproj filename, `/tmp` scratch dirs, `LLAMA_BUILD_*`
option - uses that same family token. The actual upstream checkpoint each
harness pulls its HF reference from is recorded here, and nowhere else in the
naming:

| harness (family token) | projector type                 | reference model (HF repo)                                                       |
|------------------------|--------------------------------|---------------------------------------------------------------------------------|
| `kimik25`              | `PROJECTOR_TYPE_KIMIK25`       | [`moonshotai/Kimi-K2.7-Code`](https://huggingface.co/moonshotai/Kimi-K2.7-Code) |
| `minimax_m3_vl`        | `PROJECTOR_TYPE_MINIMAX_M3_VL` | [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3)            |
| `step3vl`              | `PROJECTOR_TYPE_STEP3VL`       | [`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash) |
| `glm5next`             | `PROJECTOR_TYPE_GLM5NEXT`      | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)          |

The convert scripts and `ref_encode.py` still reference the real checkpoint's
own class and module names (e.g. `KimiK25*`, `MiniMaxM3VL*`, `Step3p7*`,
`Glm5Next*`) because they load it directly; those are functional, not naming
choices.

## Layout

A single top-level orchestrator drives every family; each family directory holds
only its own reference stack.

Top-level (`tests/mtmd-encoder/`):

- `run_all.py` - the one entry point: runs one family's storage quants end to
  end and writes `<family>/parity_results.md`. Takes `--model <family>`.
- `pyproject.toml` - the orchestrator venv (numpy + pillow only). The
  model-agnostic steps run here in-process; the model-specific steps shell out to
  each family's own venv.
- `src/` - the orchestrator's modules:
  - `src/<family>.py` - one per family (`kimik25`, `minimax_m3_vl`, `step3vl`,
    `glm5next`), each exporting `HARNESS_CONFIG`: reference repo + file list,
    convert-script path, compare label, default images. All of them currently point
    `default_images` at the shared `DEFAULT_IMAGES` set in `src/gen_images.py`.
  - `src/gen_images.py` - shared, model-agnostic test-image generator; also
    defines `DEFAULT_IMAGES`, the default image set shared by every family for now
  - `src/parity_report.py` - shared `parity_results.md` renderer

Per family (`tests/mtmd-encoder/<family>/`):

- `ref_encode.py` - HuggingFace reference encoder
- `pyproject.toml` + `uv.lock` - the reference/convert venv (pinned torch /
  transformers / gguf), used only by the convert and reference-encode steps
- the convert script (local, or in-tree under `examples/mtmd/legacy-models`)
- `wrapper/ik_encode.cpp` - ik_llama.cpp encoder driver (each family builds its
  own `ik_encode_<family>` target)

`run_all.py` generates the images (via `src/gen_images.py`), runs the comparison
(step3vl uses the `view` label since its keys are per-view), and renders the
report (via `src/parity_report.py`) all in-process. It
shells out with `uv run` inside `<family>/` only for the two steps that need the
model's pinned venv: building the mmproj (the convert script) and running
`ref_encode.py`. The C++ wrapper build (`cmake`) and the reference-weight fetch
(`wget`) are plain subprocesses.

## Embedding dump format

Both encoders write the same JSON contract, independently: the ik driver emits
`ik_embeddings.json` and `ref_encode.py` emits `reference_embeddings.json`. Each
file is a single JSON object mapping one string key per encoded unit to a
row-major `[n_tokens x n_embd]` matrix of the post-projector embeddings, stored
as one base64 string per token row. Each string is that row's `n_embd` embedding
channels (`n_embd = clip_n_mmproj_embd`) as raw little-endian float32 bytes:

    { "<key>": [ "<base64 of n_embd LE f32>", ... n_tokens rows ... ], ... }

The shape is not written down anywhere - it is recovered from the data: `n_tokens`
is the array length, and `n_embd` is `len(base64_decode(row)) / 4`.

The key differs by tower shape:

| harness         | key(s) per image                          | extra reference file    |
|-----------------|-------------------------------------------|-------------------------|
| `kimik25`       | image basename (`cat.png`)                | none                    |
| `minimax_m3_vl` | image basename                            | none                    |
| `glm5next`      | image basename                            | none                    |
| `step3vl`       | `<image>.<view>` (`overview`, `slice0`..) | `reference_layout.json` |

`step3vl` splits an image into one key per view and records the per-image model
order (`[slices..., overview]`) in `reference_layout.json` (reference side only,
`indent=2`); the special layout tokens are text-embedding rows, not vision
features, so they are not emitted here.

**Precision.** The bytes on disk are the bytes in memory: each row is its raw
little-endian float32 buffer, base64-encoded, so no decimal formatting or parsing
sits between either encoder and the comparison. Two equal float32 values serialize
to byte-identical base64 with nothing to trust about anyone's float printer, and
any difference the comparator reports is therefore a genuine encoder difference.
NaN and Inf survive verbatim (they were the one asymmetry of the old decimal
format, where nlohmann wrote `null` and Python wrote `NaN`/`Infinity`). The C++
side encodes with the in-tree `base64.hpp` (reachable through the `common` include
path, same as `nlohmann/json.hpp`); the Python side uses `numpy.ndarray.tobytes`
on a `<f4` array plus `base64.b64encode`.

The comparison step loads both files, warns on any key present on only one side, decodes
each paired key by stacking its rows into a float64 `[n_tokens, n_embd]` array, and
reports per-token minimum cosine similarity and worst relative L2
(`||ref_i - ik_i|| / ||ref_i||`) per key. It produces the per-key metrics in-process
(consumed directly by `parity_report.py`). A ref-vs-ik shape mismatch is recorded
as an error row instead of being compared; a key whose own rows are ragged (or
empty) makes the stack raise, since that is a producer bug rather than a parity
result.

## Running

From `tests/mtmd-encoder/`, with the orchestrator venv:

    uv run python run_all.py --model <family> [args]

`run_all.py` runs the selected family's storage quants (fetch reference weights,
sync the family venv, build the wrapper, then per quant: convert the mmproj,
encode with both drivers, compare) and regenerates `<family>/parity_results.md`
from the collected metrics.

## Agent conduct when running these tests

An agent asked to run these tests runs them and returns the results verbatim.
That is the whole job.

NEVER pass opinion on the quality of the metrics. Do not label a `min_cosim` or
`worst_rel_l2` value as good, bad, acceptable, healthy, passing, failing, weak,
near-exact, or any other judgement, and do not rank the families against each
other by quality. Report the numbers (and whether each run exited cleanly);
whether those numbers are good enough is the reader's call, not the agent's.

## Common arguments

`--model <family>` (required) selects the family: `kimik25`, `minimax_m3_vl`,
`step3vl`, or `glm5next`. The remaining flags - `--image`, `--quant`, `--cpu`, and
`--ik-image-preprocessing` - apply across families.

### `--image <name>:<W>:<H> [...]`

The set of images to generate, each `name:width:height` in
pixels; defaults to the family's `default_images` (currently the shared
`DEFAULT_IMAGES` set in `src/gen_images.py`) when omitted. `name` is a synthetic pattern
(`rgb_gradient_h`, `rgb_gradient_v`, `bw_checkerboard`, `quadrants`, `rings`,
`noise`) or a registered real photo (`test-1`, `llama-leader`). Each spec is
written to `images/<name>_<W>_<H>.png` 

### `--quant <name> [<name> ...]`

Storage quants to run: `f16` (default), `bf16`, `f32`. 

### `--cpu`

Run the encoder on CPU (and the HF reference on CPU for families whose reference
defaults to GPU, i.e. step3vl).

### `--ik-image-preprocessing`

Uses ik_llama's preprocessing, so that the encoders get the same pixels. 

| harness         | dump directory           |
|-----------------|--------------------------|
| `kimik25`       | `/tmp/kimik25_pre`       |
| `minimax_m3_vl` | `/tmp/minimax_m3_vl_pre` |
| `step3vl`       | `/tmp/step3vl_pre`       |
| `glm5next`      | `/tmp/glm5next_pre`      |

**Parity risk (not yet validated).** The `minimax_m3_vl` patch unfold is a
hand-written replica of the HF processor's unfold and is not validated against
it. `kimik25` no longer carries this risk: under `--ik-image-preprocessing` it
runs the checkpoint's own `navit_patchify` on ik's pixels (a scoped monkeypatch
swaps ik's normalized pixels in at the processor's unfold seam, asserting the
grid matches), so the reference unfold is the processor's own. `glm5next` never
carried it either: its processor exposes `patchify` as a plain method, which
`ref_encode.py` calls directly on ik's pixels.

**Temp-dir cleanup (future work).** The dump directory names above are hardcoded
in two places per harness (the C++ wrapper and `ref_encode.py`) and diverge by
model. A future pass should centralize them (a shared constant or an argument).

## Support matrix

| argument                   | `kimik25` | `minimax_m3_vl` | `step3vl` | `glm5next` |
|----------------------------|:---------:|:---------------:|:---------:|:----------:|
| `--image`                  |    yes    |       yes       |    yes    |    yes     |
| `--quant`                  |    yes    |       yes       |    yes    |    yes     |
| `--cpu`                    |    yes    |       yes       |    yes    |    yes     |
| `--ik-image-preprocessing` |    yes    |       yes       |    yes    |    yes     |

## Comment and docstring conventions

Keep comments and docstrings for what the code alone does not make clear: control
flow, non-obvious mechanics, dependency and config rationale, binary formats, and
function parameters. Two things do not belong in them:

- **Do not restate this README.** The argument semantics, directory layout, JSON
  dump format, naming scheme, and precision guarantees are documented here and
  only here; comments and docstrings must not repeat them - point back to this
  file instead. In particular, drop the descriptive help text for any argument
  covered above: a bare `usage:` synopsis is fine, the per-argument explanation
  is not.
- **Do not cross-reference other code by name as "matches X" / "same as X" /
  "mirrors X".** Such notes claim an equivalence to code that can change
  underneath them, so they go stale and turn into maintenance overhead. State
  what the code does; do not assert that it tracks some other file, class, or
  function.

## todo
* suspicious comment in step3vl: "NOTE: hand-written replica of the processor's unfold, not round-trip validated; see tests/mtmd-encoder/README.md."