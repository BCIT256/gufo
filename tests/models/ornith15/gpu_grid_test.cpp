// Canonical 16-token prefill grid. A cached suffix must match a cold prefill
// of the same tokens bit for bit: RAM rewind in the session that produced the
// checkpoint, restore into a session that already ran other work, and a host
// payload round trip. Histories are the capture prompts, repeated when they
// are shorter than two cells, and one rendered tool-agent conversation.
// The direct chunk sizes scored by gpu_decode stay off this grid.
//
// Exits 77 when GUFO_ORNITH15_APEX_MTP_MODEL or GUFO_ORNITH15_LLAMA_LOGITS
// is missing.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/ornith15/chat_template.hpp"
#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;
namespace {

constexpr std::uint32_t kGrid = 16;
constexpr std::uint32_t kHistory = 128;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

struct Capture {
  std::uint32_t vocab = 0;
  std::vector<std::int32_t> tokens;
};

Capture Load(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4]{};
  std::uint32_t header[3]{};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(static_cast<bool>(in) && std::string(magic, 4) == "ORNL" &&
              header[0] == 1 && header[1] > 0 && header[2] == 248320,
          "invalid capture header: " + path.string());
  Capture c;
  c.vocab = header[2];
  c.tokens.resize(header[1]);
  in.read(reinterpret_cast<char*>(c.tokens.data()),
          static_cast<std::streamsize>(c.tokens.size() * sizeof(std::int32_t)));
  // Logits follow. The grid gate compares Gufo paths with each other.
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::uint64_t>(in.tellg());
  const auto expect =
      16ull + c.tokens.size() * sizeof(std::int32_t) +
      static_cast<std::uint64_t>(c.tokens.size()) * c.vocab * sizeof(float);
  Require(bytes == expect, "truncated/oversized capture");
  return c;
}

std::vector<std::int32_t> Fit(std::vector<std::int32_t> tokens) {
  if (tokens.empty())
    return tokens;
  const auto base = tokens;
  while (tokens.size() < kHistory)
    tokens.insert(tokens.end(), base.begin(), base.end());
  tokens.resize(kHistory);
  return tokens;
}

struct Bits {
  bool match = false;
  std::size_t index = 0;
  float left = 0;
  float right = 0;
};

Bits Compare(const float* cold, const float* got, std::size_t n) {
  Bits bits;
  bits.match = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(cold + i, got + i, sizeof(float)) != 0) {
      bits.match = false;
      bits.index = i;
      bits.left = cold[i];
      bits.right = got[i];
      return bits;
    }
  }
  return bits;
}

std::unique_ptr<orn::rocm::Session> Open(const orn::rocm::DeviceModel& device,
                                         std::uint32_t context,
                                         std::string* error) {
  auto session = orn::rocm::Session::Create(device, context, error);
  Require(session != nullptr && session->SetCanonicalGrid(kGrid),
          "grid session: " + *error);
  return session;
}

void Prefill(orn::rocm::Session& session, std::span<const std::int32_t> tokens,
             std::uint32_t vocab, std::vector<float>* all, std::string* error) {
  const auto before = session.Position();
  std::vector<float> last(vocab);
  std::span<float> rows;
  if (all != nullptr) {
    all->assign(tokens.size() * static_cast<std::size_t>(vocab), 0.f);
    rows = *all;
  }
  Require(session.Prefill(tokens, last, error,
                          orn::rocm::Session::kMaxPrefillChunk, rows),
          "prefill: " + *error);
  Require(
      session.Position() == before + static_cast<std::uint32_t>(tokens.size()),
      "prefill position");
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
  Require(tokens.has_value() && tokens->size() > kGrid,
          "agent render: " + error);
  std::vector<std::int32_t> ids;
  ids.reserve(tokens->size());
  for (const auto token : *tokens) {
    Require(token <= static_cast<gufo::tokenization::TokenId>(
                         std::numeric_limits<std::int32_t>::max()),
            "agent token range");
    ids.push_back(static_cast<std::int32_t>(token));
  }
  std::cout << "agent_replay_rendered tokens=" << ids.size() << std::endl;
  return Fit(std::move(ids));
}

struct TurnResult {
  bool ram_match = false;
  bool ram_skipped = false;
  bool used_match = false;
  bool used_copied = false;
  bool disk_match = false;
  bool disk_copied = false;
  std::uint32_t published = 0;
  std::uint32_t rep_refill = 0;
  std::uint32_t steps = 0;
};

TurnResult RunTurn(const orn::rocm::DeviceModel& device,
                   const std::vector<std::int32_t>& history,
                   std::uint32_t prev_end, std::uint32_t end,
                   std::uint32_t vocab, std::string* error) {
  const std::uint32_t published = (prev_end / kGrid) * kGrid;
  Require(published > 0 && published < end && prev_end < end, "turn span");
  TurnResult result;
  result.published = published;
  result.rep_refill = end - published;
  result.steps = std::min<std::uint32_t>(4, end - prev_end);

  auto cold = Open(device, end, error);
  std::vector<float> cold_logits;
  Prefill(*cold, std::span(history.data(), end), vocab, &cold_logits, error);
  cold.reset();

  auto origin = Open(device, end, error);
  Prefill(*origin, std::span(history.data(), published), vocab, nullptr, error);
  auto image = origin->StageSnapshot(error);
  Require(image != nullptr && image->position == published,
          "stage boundary: " + *error);
  if (prev_end > published) {
    Prefill(*origin,
            std::span(history.data() + published, prev_end - published), vocab,
            nullptr, error);
  }
  std::vector<float> step_logits(vocab);
  for (std::uint32_t i = 0; i < result.steps; ++i) {
    Require(origin->Step(history[prev_end + i], step_logits, error),
            "dirty step: " + *error);
  }
  std::vector<std::uint8_t> payload;
  Require(origin->ExportStagedSnapshot(image, &payload, error),
          "export: " + *error);
  Require(origin->RestoreStagedSnapshot(*image, step_logits, error),
          "ram rewind: " + *error);
  result.ram_skipped = origin->LastRestoreSkippedKv();
  Require(origin->Position() == published, "ram position");
  std::vector<float> suffix;
  Prefill(*origin, std::span(history.data() + published, end - published),
          vocab, &suffix, error);
  const auto ram =
      Compare(cold_logits.data() + static_cast<std::size_t>(published) * vocab,
              suffix.data(), suffix.size());
  result.ram_match = ram.match;
  if (!ram.match) {
    std::cout << " ram_mismatch index=" << ram.index << " cold=" << ram.left
              << " got=" << ram.right << std::endl;
  }
  origin.reset();

  auto used = Open(device, end, error);
  std::vector<std::int32_t> junk(8);
  for (std::size_t i = 0; i < junk.size(); ++i)
    junk[i] = history[(i * 3) % history.size()];
  Prefill(*used, junk, vocab, nullptr, error);
  Require(used->RestoreStagedSnapshot(*image, step_logits, error),
          "used restore: " + *error);
  result.used_copied = !used->LastRestoreSkippedKv();
  Require(used->Position() == published, "used position");
  Prefill(*used, std::span(history.data() + published, end - published), vocab,
          &suffix, error);
  const auto used_bits =
      Compare(cold_logits.data() + static_cast<std::size_t>(published) * vocab,
              suffix.data(), suffix.size());
  result.used_match = used_bits.match;
  if (!used_bits.match) {
    std::cout << " used_mismatch index=" << used_bits.index
              << " cold=" << used_bits.left << " got=" << used_bits.right
              << std::endl;
  }
  used.reset();

  auto disk = Open(device, end, error);
  Prefill(*disk, junk, vocab, nullptr, error);
  Require(disk->RestoreSnapshot(payload, step_logits, error),
          "disk restore: " + *error);
  result.disk_copied = !disk->LastRestoreSkippedKv();
  Require(disk->Position() == published, "disk position");
  Prefill(*disk, std::span(history.data() + published, end - published), vocab,
          &suffix, error);
  const auto disk_bits =
      Compare(cold_logits.data() + static_cast<std::size_t>(published) * vocab,
              suffix.data(), suffix.size());
  result.disk_match = disk_bits.match;
  if (!disk_bits.match) {
    std::cout << " disk_mismatch index=" << disk_bits.index
              << " cold=" << disk_bits.left << " got=" << disk_bits.right
              << std::endl;
  }
  return result;
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

    struct History {
      std::string name;
      std::vector<std::int32_t> tokens;
    };
    std::vector<History> histories;
    for (const auto& entry : fs::directory_iterator(logits_dir)) {
      const auto name = entry.path().filename().string();
      if (!name.ends_with(".ologits") || name.ends_with(".sequential.ologits"))
        continue;
      auto capture = Load(entry.path());
      Require(capture.vocab == weights->config.vocab_size, "vocab");
      histories.push_back(
          {name.substr(0, name.size() - 8), Fit(std::move(capture.tokens))});
    }
    std::ranges::sort(histories, {}, &History::name);
    Require(!histories.empty(), "no captures");
    histories.push_back({"agent_replay", AgentTokens(*tokenizer)});

    bool ok = true;
    for (const auto& history : histories) {
      Require(history.tokens.size() == kHistory, "history length");
      const std::uint32_t vocab = weights->config.vocab_size;
      // 100 sits inside the cell that starts at 96. 128 ends on a boundary.
      const auto turn =
          RunTurn(*device, history.tokens, 100, kHistory, vocab, &error);
      const bool turn_ok = turn.ram_match && turn.ram_skipped &&
                           turn.used_match && turn.used_copied &&
                           turn.disk_match && turn.disk_copied;
      ok = ok && turn_ok;
      std::cout << history.name << " published=" << turn.published
                << " re_prefill=" << turn.rep_refill << " steps=" << turn.steps
                << " ram_skip=" << turn.ram_skipped
                << " ram_match=" << turn.ram_match
                << " used_copied=" << turn.used_copied
                << " used_match=" << turn.used_match
                << " disk_copied=" << turn.disk_copied
                << " disk_match=" << turn.disk_match
                << (turn_ok ? " PASS" : " FAIL") << std::endl;
    }
    if (!ok)
      return 1;
    std::cout << "canonical grid bit-identical on " << histories.size()
              << " histories" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cout << "FAIL: " << e.what() << std::endl;
    return 1;
  }
}
