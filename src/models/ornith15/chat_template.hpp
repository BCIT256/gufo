#ifndef GUFO_MODELS_ORNITH15_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_ORNITH15_CHAT_TEMPLATE_HPP_

#include <array>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

/// Ornith's chat policy on top of the shared Qwen ChatML formatter. The
/// reference is Ornith's own chat_template.jinja (ornith-ai/Ornith-1.5-35B-A3B
/// @ 10fbf86f), which the model card names as training-consistent; no GGUF
/// embeds it unmodified, so the embedded template only identifies the file.
namespace gufo::models::ornith15 {

/// SHA-256 of the reference chat_template.jinja.
inline constexpr std::string_view kReferenceTemplateSha256 =
    "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e";

/// Templates embedded in the inspected Ornith GGUFs: the stock Qwen3.6
/// template left in tokenizer_config.json (APEX files, converted by
/// llama.cpp) and the publisher's Unsloth-patched template (official files).
inline constexpr std::array<std::string_view, 3> kRecognizedTemplateSha256 = {
    kReferenceTemplateSha256,
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b",
};

/// Cache and snapshot identity of the rendering policy below.
inline constexpr std::string_view kTemplateId = "ornith15-chatml-compiled-v1";

/// Rejects a GGUF whose embedded template is not one of the recognized
/// Ornith templates. Stock Qwen3.6-35B-A3B shares the geometry and the
/// `e84f32a2` template, so this identifies the template family, not the
/// checkpoint.
[[nodiscard]] bool ValidateEmbeddedTemplate(const core::GgufReader& reader,
                                            std::string* error_msg = nullptr);

/// Formatter options for Ornith's reference template: every past assistant
/// turn keeps its reasoning, and no reasoning-effort instruction is added.
/// Thinking on/off follows `reasoning.enabled`; effort and a request to drop
/// past reasoning have no counterpart in the template and are not applied.
[[nodiscard]] tokenization::ChatTemplateOptions ResolveChatOptions(
    const ReasoningOptions& reasoning);

}  // namespace gufo::models::ornith15

#endif  // GUFO_MODELS_ORNITH15_CHAT_TEMPLATE_HPP_
