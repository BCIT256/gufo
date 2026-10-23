#include "src/models/ornith15/chat_template.hpp"

#include <algorithm>
#include <cstdint>

#include "src/core/crypto/sha256.hpp"

namespace gufo::models::ornith15 {

bool ValidateEmbeddedTemplate(const core::GgufReader& reader,
                              std::string* error_msg) {
  const auto value = reader.GetMetadataString("tokenizer.chat_template");
  if (!value.has_value() || value->empty()) {
    if (error_msg != nullptr)
      *error_msg = "Ornith GGUF is missing tokenizer.chat_template metadata";
    return false;
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value->data());
  const std::string hash = crypto::Sha256Hex({begin, value->size()});
  if (std::find(kRecognizedTemplateSha256.begin(),
                kRecognizedTemplateSha256.end(),
                hash) == kRecognizedTemplateSha256.end()) {
    if (error_msg != nullptr)
      *error_msg =
          "Ornith GGUF chat template SHA-256 is not a recognized version: " +
          hash;
    return false;
  }
  return true;
}

tokenization::ChatTemplateOptions ResolveChatOptions(
    const ReasoningOptions& reasoning) {
  tokenization::ChatTemplateOptions options;
  options.enable_thinking = reasoning.enabled.value_or(true);
  options.reasoning_instruction = false;
  options.preserve_thinking = true;
  return options;
}

}  // namespace gufo::models::ornith15
