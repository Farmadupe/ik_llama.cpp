# Project rules

- **Write only ASCII while operating in this codebase.** Do not introduce
  non-ASCII characters (em-dashes, smart quotes, arrows, multiplication/approx
  signs, etc.) into code, comments, or docs. Use ASCII equivalents: `-` for
  em-dashes, `x` for multiplication signs, "approximately" for approx
  signs, plain `"`/`'` for smart quotes.
- **Do not use ASCII arrows (`->`) either.** The user dislikes them. Express the
  relationship in plain English (e.g. "from A to B", "A becomes B", "A then B")
  instead of `A -> B`. This is about prose arrows. The `->` of real language
  syntax is fine and must not be "fixed": Python return-type hints
  (`def f() -> str:`), Rust/C++ returns and member access, shell/Make rules, etc.
  The rule targets arrows used as prose shorthand, not code tokens.

- **Do not use `@property`.** Prefer a plain method with explicit call syntax
  (e.g. `obj.shape()` over `obj.shape`).

- **Build with CUDA for testing.** Use `./farmadupe/scripts/compile.sh`
  when building for the python test suites in `farmadupe/test/`; the machine
  has an RTX 3090 and qwen3vl-4b-q4 fits comfortably. Drop back to a CPU build
  (`./farmadupe/scripts/compile.sh`) during development or when debugging calls for it.

- **Build with CPU for development.** Use `./farmadupe/scripts/compile.sh --cpu` for normal day-to-day development, to speed up iteration times. 

# JJ
The user is using JJ for change management.

# Deferred issues (embd-only decode rounds)

llama-server's update_slots decode loop is embd-only: staged token rows are
converted through llama_input_embeddings and media rows are drained from mtmd
streams, so llama_decode only ever receives batch.embd from the server. The
following known issues were deliberately deferred when the token path was
removed. Functionality is allowed to degrade as described; do not add
workarounds or assertions for these without being asked.

- **TODO gemma3/gemma4 input scaling**: build_gemma3.cpp and build_gemma4.cpp
  apply the sqrt(n_embd) input scale only when batch.token is set (raw mmproj
  image embeddings must not be re-scaled), so text rows delivered as
  embeddings skip the scale and arrive roughly 50x too small. This already
  affected gemma3 multimodal prompts under the stream design, and embd-only
  rounds extend it to all gemma3/gemma4 text. Intended fix: change
  llama_input_embeddings' contract from "rows of tok_embd" to "what the token
  path feeds the first layer" (apply the arch-conditional input transforms);
  per-row correctness then falls out, since media rows are pre-scaled by the
  mmproj and text rows by the LUT. gemma1/gemma2/minicpm scale
  unconditionally in-graph and are unaffected. Deferred to a dedicated
  "gemma update" work item; gemma needs its own test coverage anyway.

- **gemma4 per-layer embeddings (PLE)**: build_gemma4.cpp looks up
  tok_embd_per_layer rows by token id; embd batches substitute the
  padding-token row for every position, so token identity is lost and text
  quality degrades. This cannot be fixed by scaling. gemma4 is to be
  considered broken under embd-only rounds. This may change the design
  later; the likely shape is batches carrying both tokens and embeddings
  into libllama (embd for the first-layer input, tokens for side-table
  lookups).

- **MTP / speculative target features**: the server hands the decoded (embd)
  batch view to common_speculative_on_target_seq_batch. qwen35 MTP works
  (its MTP graph reads model.tok_embd and has an embd branch). openPangu MTP
  hits GGML_ASSERT(batch.token) in build_openpangu.cpp and aborts the
  server - accepted for now, the assertion is allowed to fire. gemma4 target
  features silently take a different graph path. Likely future fix is the
  same combined tokens-plus-embeddings batch in libllama (or handing spec
  replay a token view built from the token staging batch, which retains
  token ids).