// Wide prefill must match consecutive 16-token cells bit for bit.
// One call at chunk 512, a 16-token cell schedule, and grid-aligned splits
// (32+rest, 400+112, 16 + middle + tail) publish the same logits at every
// position and the same snapshot at every shared boundary.
// A test-only column-dependent GEMM must make the wide call disagree.
//
// Exits 77 when GUFO_ORNITH15_APEX_MTP_MODEL or GUFO_ORNITH15_LLAMA_LOGITS
// is missing. Scores that directory and, when present beside it, oracle
// and heldout. Also scores one rendered tool-agent prefix.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/ornith15/chat_template.hpp"
#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/ornith15/weights.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kGrid = 16;
constexpr std::uint32_t kWide = 512;
constexpr std::uint32_t kContext = 1024;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

std::vector<std::int32_t> LoadTokens(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4]{};
  std::uint32_t header[3]{};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(static_cast<bool>(in) && std::string(magic, 4) == "ORNL" &&
              header[0] == 1 && header[1] > 0 && header[2] == 248320,
          "invalid capture header: " + path.string());
  std::vector<std::int32_t> tokens(header[1]);
  in.read(reinterpret_cast<char*>(tokens.data()),
          static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
  Require(static_cast<bool>(in), "short capture: " + path.string());
  return tokens;
}

std::vector<std::int32_t> RepeatTo(std::vector<std::int32_t> tokens,
                                   std::size_t n) {
  if (tokens.empty() || tokens.size() >= n)
    return tokens;
  const auto base = tokens;
  while (tokens.size() < n)
    tokens.insert(tokens.end(), base.begin(), base.end());
  tokens.resize(n);
  return tokens;
}

std::vector<std::int32_t> AgentTokens(
    const gufo::tokenization::QwenTokenizer& tokenizer) {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::ChatTool;
  ChatTool read;
  read.name = "read";
  read.description = "Read a file";
  read.parameters_json =
      R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})";
  ChatTool edit;
  edit.name = "edit";
  edit.description = "Replace text in a file";
  edit.parameters_json =
      R"({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]})";
  const std::vector<ChatTool> tools = {read, edit};
  ChatMessage user(
      ChatRole::kUser,
      "For calc0.py then calc1.py: read the file with read, fix its "
      "subtraction bug using edit, then read it again to verify. Do all six "
      "actions, one tool call per turn.");
  ChatMessage assistant(ChatRole::kAssistant, "I'll read calc0.py.");
  assistant.tool_calls.push_back(
      {.id = "call_read_0",
       .name = "read",
       .arguments = {{.name = "path", .value = "calc0.py"}}});
  ChatMessage tool(ChatRole::kTool, "def sub(a, b):\n    return a + b\n");
  tool.name = "read";
  tool.tool_call_id = "call_read_0";
  const std::vector<ChatMessage> messages = {user, assistant, tool};
  gufo::ReasoningOptions reasoning;
  reasoning.enabled = false;
  std::string error;
  const auto tokens = gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
      tokenizer, messages, tools, orn::ResolveChatOptions(reasoning), &error);
  Require(tokens.has_value() && !tokens->empty(), "agent render: " + error);
  std::vector<std::int32_t> ids;
  ids.reserve(tokens->size());
  for (const auto token : *tokens) {
    Require(token <= static_cast<gufo::tokenization::TokenId>(
                         std::numeric_limits<std::int32_t>::max()),
            "agent token range");
    ids.push_back(static_cast<std::int32_t>(token));
  }
  return ids;
}

std::vector<std::uint32_t> Cells(std::uint32_t n) {
  std::vector<std::uint32_t> parts;
  std::uint32_t left = n;
  while (left > 0) {
    const std::uint32_t count = std::min(kGrid, left);
    parts.push_back(count);
    left -= count;
  }
  return parts;
}

std::vector<std::uint32_t> SixteenMiddleTail(std::uint32_t n) {
  if (n <= kGrid)
    return {n};
  const std::uint32_t tail = n % kGrid;
  const std::uint32_t middle = n - kGrid - tail;
  std::vector<std::uint32_t> parts{kGrid};
  if (middle != 0)
    parts.push_back(middle);
  if (tail != 0)
    parts.push_back(tail);
  return parts;
}

struct Outcome {
  std::vector<float> logits;
  std::map<std::uint32_t, std::vector<std::uint8_t>> snaps;
  double ms = 0;
};

Outcome Run(const orn::rocm::DeviceModel& device,
            std::span<const std::int32_t> tokens, std::uint32_t vocab,
            const std::vector<std::uint32_t>& parts,
            const std::set<std::uint32_t>& snap_at, bool mutate) {
  std::uint32_t sum = 0;
  for (const auto part : parts)
    sum += part;
  Require(sum == tokens.size(), "schedule does not cover the prompt");
  std::string error;
  auto session = orn::rocm::Session::Create(device, kContext, &error);
  Require(session != nullptr && session->SetCanonicalGrid(kGrid),
          "session: " + error);
  if (mutate)
    session->SetChunkDependentReductionForTest(true);
  Outcome out;
  out.logits.assign(tokens.size() * static_cast<std::size_t>(vocab), 0.f);
  std::vector<float> last(vocab);
  std::uint32_t at = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto part : parts) {
    std::vector<float> rows(static_cast<std::size_t>(part) * vocab);
    Require(session->Prefill(std::span(tokens.data() + at, part), last, &error,
                             part, rows),
            "prefill: " + error);
    std::copy(rows.begin(), rows.end(),
              out.logits.begin() + static_cast<std::ptrdiff_t>(at * vocab));
    at += part;
    if (snap_at.count(at) != 0) {
      std::vector<std::uint8_t> payload;
      Require(session->SaveSnapshot(&payload, &error), "snapshot: " + error);
      out.snaps.emplace(at, std::move(payload));
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  Require(at == tokens.size(), "prefill position");
  return out;
}

bool SameBytes(const std::vector<std::uint8_t>& left,
               const std::vector<std::uint8_t>& right) {
  return left.size() == right.size() &&
         (left.empty() ||
          std::memcmp(left.data(), right.data(), left.size()) == 0);
}

// Returns the first mismatched token, or tokens.size() when they match.
std::size_t FirstMismatch(const std::vector<float>& left,
                          const std::vector<float>& right,
                          std::uint32_t vocab) {
  const std::size_t n = std::min(left.size(), right.size()) / vocab;
  for (std::size_t token = 0; token < n; ++token) {
    const auto* a = left.data() + token * vocab;
    const auto* b = right.data() + token * vocab;
    if (std::memcmp(a, b, static_cast<std::size_t>(vocab) * sizeof(float)) != 0)
      return token;
  }
  if (left.size() != right.size())
    return n;
  return n;
}

bool Compare(const std::string& name, const Outcome& ref, const Outcome& got,
             std::uint32_t vocab, std::uint32_t ntokens) {
  const auto token = FirstMismatch(ref.logits, got.logits, vocab);
  bool snaps = true;
  std::uint32_t snap_at = 0;
  for (const auto& [pos, payload] : got.snaps) {
    const auto it = ref.snaps.find(pos);
    if (it == ref.snaps.end() || !SameBytes(it->second, payload)) {
      snaps = false;
      snap_at = pos;
      break;
    }
  }
  const bool ok = token == ntokens && snaps;
  const double tok_s = got.ms > 0 ? (1000.0 * ntokens) / got.ms : 0;
  std::cout << name << (ok ? " PASS" : " FAIL") << " ms=" << got.ms
            << " tok_s=" << tok_s;
  if (!ok) {
    std::cout << " first_bad_token=" << token << " snap_ok=" << snaps
              << " snap_at=" << snap_at;
  }
  std::cout << std::endl;
  return ok;
}

bool ScorePrompt(const orn::rocm::DeviceModel& device, const std::string& name,
                 const std::vector<std::int32_t>& tokens, std::uint32_t vocab) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  std::vector<std::pair<std::string, std::vector<std::uint32_t>>> schedules;
  schedules.push_back({"one", {n}});
  if (n > 32)
    schedules.push_back({"split_32_rest", {32, n - 32}});
  if (n >= 512)
    schedules.push_back({"split_400_112", {400, 112}});
  const auto mid = SixteenMiddleTail(n);
  if (mid.size() > 1)
    schedules.push_back({"split_16_mid_tail", mid});

  std::set<std::uint32_t> bounds{n};
  for (const auto& schedule : schedules) {
    std::uint32_t at = 0;
    for (const auto part : schedule.second) {
      at += part;
      bounds.insert(at);
    }
  }
  const auto cells = Run(device, tokens, vocab, Cells(n), bounds, false);
  std::cout << name << " tokens=" << n << " cells_ms=" << cells.ms
            << " cells_tok_s=" << (1000.0 * n) / cells.ms << std::endl;
  bool ok = true;
  for (const auto& schedule : schedules) {
    const auto got = Run(device, tokens, vocab, schedule.second, bounds, false);
    ok = Compare(name + " " + schedule.first, cells, got, vocab, n) && ok;
  }
  return ok;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  if (model_path == nullptr || logits_dir == nullptr ||
      !fs::is_regular_file(model_path) || !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_LLAMA_LOGITS\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, "bind trunk: " + error);
    const auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer: " + error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, "upload: " + error);
    const std::uint32_t vocab = weights->config.vocab_size;

    std::vector<fs::path> dirs{fs::path(logits_dir)};
    for (const char* name : {"oracle", "heldout"}) {
      const auto path = fs::path(logits_dir).parent_path() / name;
      if (fs::is_directory(path) &&
          fs::canonical(path) != fs::canonical(dirs.front()) &&
          std::find(dirs.begin(), dirs.end(), path) == dirs.end())
        dirs.push_back(path);
    }

    bool ok = true;
    for (const auto& dir : dirs) {
      for (const auto& entry : fs::directory_iterator(dir)) {
        const auto file = entry.path().filename().string();
        if (!file.ends_with(".ologits") ||
            file.ends_with(".sequential.ologits"))
          continue;
        const auto base = LoadTokens(entry.path());
        const auto stem = file.substr(0, file.size() - std::strlen(".ologits"));
        const auto label = dir.filename().string() + "/" + stem;
        if (const char* only = std::getenv("GUFO_CHUNK_ONLY");
            only != nullptr && only[0] != '\0' &&
            label.find(only) == std::string::npos)
          continue;
        ok = ScorePrompt(*device, label + "/native", base, vocab) && ok;
        if (base.size() < kWide)
          ok = ScorePrompt(*device, label + "/wide", RepeatTo(base, kWide),
                           vocab) &&
               ok;
      }
    }
    const auto agent = AgentTokens(*tokenizer);
    ok = ScorePrompt(*device, "agent/native", agent, vocab) && ok;
    if (agent.size() < kWide)
      ok = ScorePrompt(*device, "agent/wide", RepeatTo(agent, kWide), vocab) &&
           ok;

    auto mutation_tokens = agent;
    if (mutation_tokens.size() < 128)
      mutation_tokens = RepeatTo(std::move(mutation_tokens), 128);
    else
      mutation_tokens.resize(128);
    const auto n = static_cast<std::uint32_t>(mutation_tokens.size());
    const auto cells =
        Run(*device, mutation_tokens, vocab, Cells(n), {n}, false);
    const auto mutated = Run(*device, mutation_tokens, vocab, {n}, {n}, true);
    const auto bad = FirstMismatch(cells.logits, mutated.logits, vocab);
    const bool diverged = bad < n;
    std::cout << "mutation " << (diverged ? "DIVERGED" : "FAILED_TO_DIVERGE")
              << " first_bad_token=" << bad << std::endl;
    ok = ok && diverged;

    if (!ok)
      return 1;
    std::cout << "chunk identity bit-identical" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cout << "FAIL: " << e.what() << std::endl;
    return 1;
  }
}
