# Ornith reference logits

Captures llama.cpp's full next-token logits for the prompts in `prompts/`, the
independent oracle for the scalar reference in `src/models/ornith15/`.
`ornith15.reference_oracle` replays each capture's token IDs through the
reference and compares every position's distribution. Token IDs come from
the capture, so this checks the model graph, not tokenization or templates.

Prompts are ChatML rendered the way Ornith's `chat_template.jinja` would render
them: a user turn, a multi-turn history whose past reasoning is kept, and a
tool call with its response. Each ends in an open `<think>` generation prompt.

```sh
LLAMA=/path/to/llama.cpp  # checkout at the revision in .devops/nix/llama-cpp-reference.nix
cmake -S "$LLAMA" -B "$LLAMA/build-cpu" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build "$LLAMA/build-cpu" --target llama
g++ -std=c++17 -O2 -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  tools/ornith15/llama_logits.cpp -L"$LLAMA/build-cpu/bin" \
  -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA/build-cpu/bin" -o llama_logits
MODEL=/path/to/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf
OUT=$(mktemp -d)
for p in tools/ornith15/prompts/*.txt; do
  ./llama_logits "$MODEL" "$p" "$OUT/$(basename "$p" .txt).ologits" 28
done
cmake --build --preset cpu-test --target ornith15_reference_oracle_test
GUFO_ORNITH15_APEX_MTP_MODEL="$MODEL" GUFO_ORNITH15_LLAMA_LOGITS="$OUT" \
  ctest --preset cpu-test -R '^ornith15\.reference_oracle$'
```

The test needs both captures of every prompt: `NAME.ologits` (one batched
decode) and `NAME.sequential.ologits` (pass `sequential` as the fifth argument
to decode one token per call). Captures are about 1 MB per position and belong
in ignored storage, not in the tree. Set `GUFO_ORNITH15_ORACLE_VERBOSE=1` for
per-position rows.

llama.cpp's CPU backend quantizes activations for its quantized dot products
and stores the KV cache in F16; the reference keeps float32 activations, so
the two never agree exactly, and llama.cpp's batched and sequential decodes
already disagree with each other. The test therefore bounds the reference by
that measured spread:

- no top-1 disagreement where llama.cpp's two decodes agree and its top two
  logits are more than 1.0 apart;
- KL(llama.cpp || reference) at most 0.1 nats where llama.cpp's entropy is
  below 0.5 nats (its own batch/sequential KL reached 0.10 there);
- mean KL at most twice llama.cpp's own batch/sequential mean.

On October 23, 2026 (llama.cpp 68d9053a CPU, APEX MTP Quality, 183
positions) the reference had no clear flips, a confident-position KL of at
most 0.036, and mean KL 0.025–0.065 against llama.cpp's own 0.032–0.085. Two
injected faults, a sigmoid DeltaNet output gate and a `h / 2` value-to-key head
mapping, raised mean KL to 8.3 and 6.0 and failed every criterion.
High-entropy positions are bounded only through the mean: KL on near-uniform
distributions (the largest, 1.7 nats, at about 9 nats of entropy) does not
separate faults from noise.

# Ornith chat-template goldens

`chat_goldens.py` regenerates `tests/fixtures/ornith15_chat_goldens.json`: the
fixture's messages rendered by transformers' `apply_chat_template` with
Ornith's `chat_template.jinja`, then tokenized by the Hugging Face tokenizer.
`tests/fixtures/ornith15_chat_template.jinja` is that template, copied from
the pinned revision. Use the flake's pinned transformers; torch is not needed:

```sh
HF=$(mktemp -d)
for f in chat_template.jinja tokenizer.json tokenizer_config.json; do
  curl -sSL -o "$HF/$f" \
    "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/10fbf86fed7ecee4a061f8b499a618f46001cac1/$f"
done
nix shell --impure --expr '((builtins.getFlake (toString ./.)).inputs.nixpkgs.legacyPackages.x86_64-linux.python313.withPackages (ps: [ ps.transformers ps.jinja2 ]))' \
  -c python3 tools/ornith15/chat_goldens.py "$HF" tests/fixtures/ornith15_chat_goldens.json
```

`ornith15.chat_template` replays the rendered bytes without a model.
`ornith15.chat_tokens` checks the token IDs through the GGUF's tokenizer and
needs `GUFO_ORNITH15_APEX_MTP_MODEL`. Ornith keeps every past assistant turn's
reasoning and adds no reasoning-effort instruction, so client effort and
`preserve_thinking: false` leave the rendering unchanged. System and developer
messages after the first turn are moved to the system turn, as for the other
Qwen models; the reference template would reject them.
