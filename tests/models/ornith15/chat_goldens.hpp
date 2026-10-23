#ifndef GUFO_TESTS_MODELS_ORNITH15_CHAT_GOLDENS_HPP_
#define GUFO_TESTS_MODELS_ORNITH15_CHAT_GOLDENS_HPP_

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"

/// Reads tests/fixtures/ornith15_chat_goldens.json, written by
/// tools/ornith15/chat_goldens.py from Hugging Face's rendering. The inputs
/// live in the fixture so the generator and these tests replay one set of
/// messages.
namespace gufo::tests::ornith15::chat {

namespace json = gufo::json;
namespace qwen = gufo::tokenization;
using gufo::testing::chat_goldens::Expect;

struct Case {
  std::string name;
  std::vector<qwen::ChatMessage> messages;
  bool tools{false};
  bool enable_thinking{true};
  std::string rendered;
  std::string rendered_sha256;
  std::size_t token_count{0};
  std::string token_sha256;
};

struct Fixture {
  json::Value root;
  std::vector<qwen::ChatTool> tools;
  std::vector<Case> cases;
};

inline qwen::ChatRole Role(const std::string& role) {
  if (role == "system")
    return qwen::ChatRole::kSystem;
  if (role == "assistant")
    return qwen::ChatRole::kAssistant;
  if (role == "tool")
    return qwen::ChatRole::kTool;
  Expect(role == "user", "golden fixture role is known");
  return qwen::ChatRole::kUser;
}

/// Client reasoning controls with only thinking on/off set.
inline gufo::ReasoningOptions Thinking(bool enabled) {
  gufo::ReasoningOptions reasoning;
  reasoning.enabled = enabled;
  return reasoning;
}

inline Fixture Load() {
  std::ifstream input(GUFO_ORNITH15_CHAT_GOLDENS, std::ios::binary);
  Expect(input.good(), "Ornith golden fixture opens");
  const std::string text{std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>()};
  Fixture f;
  f.root = json::parse(text);
  Expect(f.root.member_str("schema") == "gufo.ornith15-chat-goldens.v1",
         "Ornith golden fixture schema");
  for (const auto& tool : f.root.find("tools")->items()) {
    qwen::ChatTool definition;
    definition.name = tool.member_str("name");
    definition.description = tool.member_str("description");
    definition.parameters_json = tool.member_str("parameters_json");
    f.tools.push_back(std::move(definition));
  }
  for (const auto& item : f.root.find("cases")->items()) {
    Case c;
    c.name = item.member_str("name");
    c.tools = item.find("tools")->as_bool();
    c.enable_thinking = item.find("enable_thinking")->as_bool();
    c.rendered = item.member_str("rendered");
    c.rendered_sha256 = item.member_str("rendered_sha256");
    c.token_count = item.member_size("token_count");
    c.token_sha256 = item.member_str("token_ids_le_u32_sha256");
    for (const auto& m : item.find("messages")->items()) {
      qwen::ChatMessage message(Role(m.member_str("role")),
                                m.member_str("content"), "",
                                m.member_str("thought"));
      if (const auto* calls = m.find("tool_calls")) {
        for (const auto& call : calls->items()) {
          qwen::ChatMessage::ToolCall tool_call;
          tool_call.name = call.member_str("name");
          for (const auto& argument : call.find("arguments")->items())
            tool_call.arguments.push_back({.name = argument.items()[0].str(),
                                           .value = argument.items()[1].str()});
          message.tool_calls.push_back(std::move(tool_call));
        }
      }
      c.messages.push_back(std::move(message));
    }
    f.cases.push_back(std::move(c));
  }
  Expect(!f.cases.empty(), "Ornith golden fixture has cases");
  return f;
}

}  // namespace gufo::tests::ornith15::chat

#endif  // GUFO_TESTS_MODELS_ORNITH15_CHAT_GOLDENS_HPP_
