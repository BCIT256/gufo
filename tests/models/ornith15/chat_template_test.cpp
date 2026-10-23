// Ornith's rendering policy against Hugging Face's rendering of the reference
// chat_template.jinja. Rendered bytes only; token IDs need the GGUF and are
// checked by ornith15.chat_tokens.

#include "src/models/ornith15/chat_template.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "tests/models/ornith15/chat_goldens.hpp"
#include "tests/models/ornith15/gguf_builder.hpp"

namespace orn = gufo::models::ornith15;
namespace qwen = gufo::tokenization;
namespace golden = gufo::tests::ornith15::chat;
namespace build = gufo::tests::ornith15;
using gufo::testing::chat_goldens::Expect;
using gufo::testing::chat_goldens::Sha256;

namespace {

std::string Render(const golden::Fixture& f, const golden::Case& c,
                   const gufo::ReasoningOptions& reasoning) {
  std::string error;
  const auto rendered = qwen::QwenChatTemplate::Render(
      c.messages,
      c.tools ? std::span<const qwen::ChatTool>(f.tools)
              : std::span<const qwen::ChatTool>{},
      orn::ResolveChatOptions(reasoning), &error);
  Expect(rendered.has_value(), "Ornith case renders: " + error);
  return *rendered;
}

void ReportDifference(const std::string& name, const std::string& want,
                      const std::string& got) {
  const auto mismatch =
      std::mismatch(want.begin(), want.end(), got.begin(), got.end());
  const auto at = static_cast<std::size_t>(mismatch.first - want.begin());
  const auto from = at < 80 ? 0 : at - 80;
  std::cerr << "case " << name << " differs at byte " << at
            << "\n--- Hugging Face:\n"
            << want.substr(from, 160) << "\n--- Gufo:\n"
            << got.substr(from, 160) << '\n';
}

void CheckGoldens(const golden::Fixture& f) {
  for (const auto& c : f.cases) {
    const std::string rendered =
        Render(f, c, golden::Thinking(c.enable_thinking));
    // The fixture's text and hash must agree before either is trusted.
    Expect(Sha256(c.rendered) == c.rendered_sha256,
           "fixture text matches its hash: " + c.name);
    if (rendered != c.rendered)
      ReportDifference(c.name, c.rendered, rendered);
    Expect(Sha256(rendered) == c.rendered_sha256,
           "Ornith rendering matches Hugging Face: " + c.name);
  }
}

void CheckPolicyIsFixed(const golden::Fixture& f) {
  // Effort and a request to drop past reasoning have no counterpart in the
  // reference template; the rendering must not change.
  for (const auto& c : f.cases) {
    for (const auto effort :
         {gufo::ReasoningEffort::kLow, gufo::ReasoningEffort::kXHigh}) {
      gufo::ReasoningOptions reasoning = golden::Thinking(c.enable_thinking);
      reasoning.effort = effort;
      reasoning.preserve_thinking = false;
      const std::string rendered = Render(f, c, reasoning);
      Expect(Sha256(rendered) == c.rendered_sha256,
             "effort and preserve_thinking do not change Ornith: " + c.name);
    }
  }
  // Thinking defaults on, as in the reference template.
  const auto& thinking =
      *std::find_if(f.cases.begin(), f.cases.end(),
                    [](const auto& c) { return c.name == "thinking"; });
  Expect(Sha256(Render(f, thinking, gufo::ReasoningOptions{})) ==
             thinking.rendered_sha256,
         "thinking is the default");
}

void CheckQwen38DefaultUnchanged() {
  // The shared option defaults to Qwen3.8's effort instruction.
  const std::vector<qwen::ChatMessage> base = {
      {qwen::ChatRole::kUser, "Name one color."}};
  const auto rendered =
      qwen::QwenChatTemplate::Render(base, qwen::ChatTemplateOptions{});
  Expect(rendered.has_value() &&
             rendered->find("Reasoning effort is set to xhigh") !=
                 std::string::npos,
         "Qwen3.8 rendering keeps its effort instruction by default");
}

void CheckEmbeddedTemplateRecognition() {
  const auto validate = [](const std::string* chat_template) {
    auto fields = build::OrnithMetadata();
    if (chat_template != nullptr)
      fields["tokenizer.chat_template"] = *chat_template;
    const auto image = build::Build(fields, {});
    std::string error;
    const bool ok = orn::ValidateEmbeddedTemplate(*image.reader, &error);
    Expect(ok == error.empty(), "validation reports exactly its failures");
    return error;
  };
  std::ifstream input(GUFO_ORNITH15_CHAT_TEMPLATE, std::ios::binary);
  const std::string reference{std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>()};
  Expect(Sha256(reference) == orn::kReferenceTemplateSha256,
         "checked-in reference template is the pinned one");
  Expect(validate(&reference).empty(), "reference template recognized");
  const std::string unknown = "{{ messages }}";
  Expect(validate(nullptr) ==
             "Ornith GGUF is missing tokenizer.chat_template metadata",
         "missing template rejected");
  Expect(validate(&unknown).starts_with(
             "Ornith GGUF chat template SHA-256 is not a recognized version"),
         "unknown template rejected");
}

}  // namespace

int main() {
  const golden::Fixture f = golden::Load();
  Expect(f.root.member_str("template_sha256") == orn::kReferenceTemplateSha256,
         "fixture was rendered from the reference template");
  CheckGoldens(f);
  CheckPolicyIsFixed(f);
  CheckQwen38DefaultUnchanged();
  CheckEmbeddedTemplateRecognition();
  std::cout << "Ornith chat template renders " << f.cases.size()
            << " Hugging Face goldens byte-exactly.\n";
}
