// Wide prefill must match consecutive 16-token cells bit for bit.
// One call at the prompt width, including a 2,048-token launch, a 16-token
// cell schedule, and off-grid splits (32+rest, 400+112, 16 + middle + tail,
// whole cells + lone tail) publish the same logits at every position and the
// same snapshot at every shared boundary; a sweep covers final cells of 1, 2,
// 3, 8, 9 and 15 tokens. In-pass capture is also checked inside a 2,048-token
// launch.
// Sessions run the server's one-token grid, so off-grid splits are included.
// PrefillThrough's boundary image must equal a stop-at-boundary snapshot.
// A test-only column-dependent GEMM must make the wide call disagree, and so
// must a lone final token on the decode dots.
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

// Reference cell width. Sessions run the server's grid, under which any
// split is a cell boundary.
constexpr std::uint32_t kGrid = 16;
constexpr std::uint32_t kServerGrid = 1;
constexpr std::uint32_t kWide = 512;
constexpr std::uint32_t kLaunch = 2048;
constexpr std::uint32_t kContext = 4096;

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

// 16-token cells, also cut at every compared boundary so the reference has
// a snapshot there.
std::vector<std::uint32_t> CellsThrough(std::uint32_t n,
                                        const std::set<std::uint32_t>& cuts) {
  std::vector<std::uint32_t> parts;
  std::uint32_t at = 0;
  while (at < n) {
    std::uint32_t next = std::min(n, (at / kGrid + 1) * kGrid);
    const auto cut = cuts.upper_bound(at);
    if (cut != cuts.end() && *cut < next)
      next = *cut;
    parts.push_back(next - at);
    at = next;
  }
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
            const std::set<std::uint32_t>& snap_at, bool mutate,
            bool decode_dots = false) {
  std::uint32_t sum = 0;
  for (const auto part : parts)
    sum += part;
  Require(sum == tokens.size(), "schedule does not cover the prompt");
  std::string error;
  auto session = orn::rocm::Session::Create(device, kContext, &error);
  Require(session != nullptr && session->SetCanonicalGrid(kServerGrid),
          "session: " + error);
  if (mutate)
    session->SetChunkDependentReductionForTest(true);
  if (decode_dots)
    session->SetLoneTokenDecodeDotsForTest(true);
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
  if (n == 512)
    schedules.push_back({"split_400_112", {400, 112}});
  else if (n > 512)
    schedules.push_back({"split_400_112", {400, 112, n - 512}});
  if (n > 40) {
    schedules.push_back({"split_37_rest", {37, n - 37}});
    schedules.push_back({"split_5_11_rest", {5, 11, n - 16}});
    schedules.push_back({"split_rest_3", {n - 3, 3}});
    schedules.push_back({"split_rest_1", {n - 1, 1}});
  }
  const auto mid = SixteenMiddleTail(n);
  if (mid.size() > 1)
    schedules.push_back({"split_16_mid_tail", mid});
  // A cached turn resumes on the last cell boundary and may prefill only a
  // short cell, down to one token, that a cold call ran inside a wide launch.
  if (n > kGrid && n % kGrid != 0)
    schedules.push_back({"split_cells_lone_tail", {n - n % kGrid, n % kGrid}});

  std::set<std::uint32_t> bounds{n};
  for (const auto& schedule : schedules) {
    std::uint32_t at = 0;
    for (const auto part : schedule.second) {
      at += part;
      bounds.insert(at);
    }
  }
  const auto cells =
      Run(device, tokens, vocab, CellsThrough(n, bounds), bounds, false);
  std::cout << name << " tokens=" << n << " cells_ms=" << cells.ms
            << " cells_tok_s=" << (1000.0 * n) / cells.ms << std::endl;
  bool ok = true;
  for (const auto& schedule : schedules) {
    const auto got = Run(device, tokens, vocab, schedule.second, bounds, false);
    ok = Compare(name + " " + schedule.first, cells, got, vocab, n) && ok;
  }
  return ok;
}

// PrefillThrough from `start` must publish the same final logits as a plain
// prefill and freeze a boundary image byte-identical to SaveSnapshot after a
// prefill that stopped at `boundary`.
bool InPassCheck(const orn::rocm::DeviceModel& device,
                 const std::vector<std::int32_t>& tokens, std::uint32_t vocab,
                 std::uint32_t start, std::uint32_t boundary) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const auto make = [&] {
    std::string error;
    auto session = orn::rocm::Session::Create(device, kContext, &error);
    Require(session != nullptr && session->SetCanonicalGrid(kServerGrid),
            "session: " + error);
    return session;
  };
  const auto span = [&](std::uint32_t from, std::uint32_t to) {
    return std::span<const std::int32_t>(tokens.data() + from, to - from);
  };
  std::string error;
  std::vector<float> ref_last(vocab);
  std::vector<float> got_last(vocab);
  std::vector<std::uint8_t> ref_payload;
  auto ref = make();
  if (start > 0)
    Require(ref->Prefill(span(0, start), ref_last, &error), error);
  Require(ref->Prefill(span(start, boundary), ref_last, &error), error);
  Require(ref->SaveSnapshot(&ref_payload, &error), error);
  Require(ref->Prefill(span(boundary, n), ref_last, &error), error);

  auto got = make();
  if (start > 0)
    Require(got->Prefill(span(0, start), got_last, &error), error);
  std::shared_ptr<orn::rocm::StagedSnapshot> image;
  Require(
      got->PrefillThrough(span(start, n), got_last, boundary, &image, &error) &&
          image != nullptr && got->Position() == n,
      "prefill through: " + error);
  std::vector<std::uint8_t> payload;
  Require(got->ExportStagedSnapshot(image, &payload, &error), error);

  // The image must also resume: restore it elsewhere and finish the prompt.
  auto resumed = make();
  std::vector<float> resumed_last(vocab);
  Require(resumed->RestoreStagedSnapshot(*image, {}, &error) &&
              resumed->Prefill(span(boundary, n), resumed_last, &error),
          "resume: " + error);

  const bool same_payload = SameBytes(ref_payload, payload);
  const bool same_last = ref_last == got_last;
  const bool same_resumed = ref_last == resumed_last;
  const bool ok = same_payload && same_last && same_resumed;
  std::cout << "in_pass start=" << start << " boundary=" << boundary
            << " end=" << n << (ok ? " PASS" : " FAIL")
            << " payload=" << same_payload << " last=" << same_last
            << " resumed=" << same_resumed << std::endl;
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
    if (orn::rocm::Session::kMaxPrefillChunk >= kLaunch)
      ok = ScorePrompt(*device, "agent/chunk2048", RepeatTo(agent, kLaunch),
                       vocab) &&
           ok;

    // Every short-cell length class: one token, the two-row attention
    // group, the saved-history convolution, and the wider entries.
    for (const std::uint32_t tail : {1u, 2u, 3u, 8u, 9u, 15u}) {
      auto tokens = RepeatTo(agent, 6 * kGrid + tail);
      tokens.resize(6 * kGrid + tail);
      ok = ScorePrompt(*device, "agent/tail" + std::to_string(tail), tokens,
                       vocab) &&
           ok;
    }

    for (const auto& [start, boundary] :
         std::vector<std::pair<std::uint32_t, std::uint32_t>>{
             {0, 1},
             {0, 37},
             {0, static_cast<std::uint32_t>(agent.size()) - 7},
             {0, static_cast<std::uint32_t>(agent.size()) - 1},
             {32, static_cast<std::uint32_t>(agent.size()) - 5}}) {
      ok = InPassCheck(*device, agent, vocab, start, boundary) && ok;
    }
    if (orn::rocm::Session::kMaxPrefillChunk >= kLaunch) {
      const auto wide = RepeatTo(agent, kLaunch);
      ok = InPassCheck(*device, wide, vocab, 0, 1000) && ok;
      ok = InPassCheck(*device, wide, vocab, 400, 1500) && ok;
    }

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

    // A lone final token on the decode dots must not match its row in the
    // one-call reference.
    auto lone_tokens = RepeatTo(agent, 6 * kGrid + 1);
    lone_tokens.resize(6 * kGrid + 1);
    const auto lone_n = static_cast<std::uint32_t>(lone_tokens.size());
    const auto wide = Run(*device, lone_tokens, vocab, {lone_n}, {}, false);
    const auto lone =
        Run(*device, lone_tokens, vocab, {lone_n - 1, 1}, {}, false, true);
    const auto lone_bad = FirstMismatch(wide.logits, lone.logits, vocab);
    const bool lone_diverged = lone_bad < lone_n;
    std::cout << "mutation_lone_token "
              << (lone_diverged ? "DIVERGED" : "FAILED_TO_DIVERGE")
              << " first_bad_token=" << lone_bad << std::endl;
    ok = ok && lone_diverged;

    if (!ok)
      return 1;
    std::cout << "chunk identity bit-identical" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cout << "FAIL: " << e.what() << std::endl;
    return 1;
  }
}
