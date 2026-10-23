// Writes llama.cpp's full next-token logits for every position of one prompt,
// the independent oracle for the Ornith scalar reference.
// Usage: llama_logits MODEL.gguf PROMPT.txt OUT.ologits [THREADS] [sequential]
//
// The default decodes the prompt as one batch. `sequential` decodes one token
// per call instead, which selects llama.cpp's autoregressive kernels; the two
// captures bound llama.cpp's own numerical spread.
//
// The prompt is tokenized with special tokens parsed and no BOS (the GGUF sets
// add_bos_token false). OUT.ologits layout, little-endian:
//   char[4] "ORNL", u32 version (1), u32 n_tokens, u32 n_vocab,
//   i32 tokens[n_tokens], f32 logits[n_tokens][n_vocab]
// Row t holds the distribution predicting token t + 1.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6 ||
      (argc == 6 && std::string(argv[5]) != "sequential")) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf PROMPT.txt OUT.ologits [THREADS] "
                 "[sequential]\n",
                 argv[0]);
    return 2;
  }
  const bool sequential = argc == 6;
  std::ifstream prompt_file(argv[2], std::ios::binary);
  const std::string prompt((std::istreambuf_iterator<char>(prompt_file)),
                           std::istreambuf_iterator<char>());
  if (!prompt_file || prompt.empty()) {
    std::fprintf(stderr, "cannot read prompt %s\n", argv[2]);
    return 1;
  }

  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (model == nullptr) {
    std::fprintf(stderr, "model load failed\n");
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  std::vector<llama_token> tokens(prompt.size() + 8);
  const int n =
      llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                     tokens.data(), static_cast<int32_t>(tokens.size()),
                     /*add_special=*/false, /*parse_special=*/true);
  if (n <= 0) {
    std::fprintf(stderr, "tokenization failed\n");
    return 1;
  }
  tokens.resize(n);

  auto cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>((n + 255) / 256 * 256);
  cp.n_batch = cp.n_ctx;
  cp.n_ubatch = cp.n_ctx;
  cp.n_seq_max = 1;
  const int threads = argc >= 5 ? std::stoi(argv[4]) : 16;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (ctx == nullptr) {
    std::fprintf(stderr, "context creation failed\n");
    return 1;
  }

  std::vector<float> logits(static_cast<std::size_t>(n) * n_vocab);
  llama_batch batch = llama_batch_init(n, 0, 1);
  const int step = sequential ? 1 : n;
  for (int begin = 0; begin < n; begin += step) {
    batch.n_tokens = 0;
    for (int i = begin; i < begin + step; ++i) {
      const int j = batch.n_tokens++;
      batch.token[j] = tokens[i];
      batch.pos[j] = i;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
      std::fprintf(stderr, "decode failed\n");
      return 1;
    }
    for (int j = 0; j < step; ++j) {
      const float* row = llama_get_logits_ith(ctx, j);
      std::copy(row, row + n_vocab,
                logits.begin() + static_cast<std::size_t>(begin + j) * n_vocab);
    }
  }

  std::ofstream out(argv[3], std::ios::binary);
  const auto u32 = [&](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };
  out.write("ORNL", 4);
  u32(1);
  u32(static_cast<uint32_t>(n));
  u32(static_cast<uint32_t>(n_vocab));
  out.write(reinterpret_cast<const char*>(tokens.data()),
            static_cast<std::streamsize>(n * sizeof(llama_token)));
  out.write(reinterpret_cast<const char*>(logits.data()),
            static_cast<std::streamsize>(logits.size() * sizeof(float)));
  if (!out) {
    std::fprintf(stderr, "write failed\n");
    return 1;
  }
  std::printf("%d tokens, vocab %d:", n, n_vocab);
  for (int i = 0; i < n; ++i)
    std::printf(" %d", tokens[i]);
  std::printf("\n");

  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
