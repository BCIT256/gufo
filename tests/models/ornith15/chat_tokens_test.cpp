// Token IDs of the Ornith goldens through the GGUF's own tokenizer, against
// the Hugging Face tokenizer at the pinned revision.
//
//   GUFO_ORNITH15_APEX_MTP_MODEL=/path/to/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf
//
// Exits 77 (skipped, never passed) when the model is not supplied.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith15/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/ornith15/chat_goldens.hpp"

namespace orn = gufo::models::ornith15;
namespace qwen = gufo::tokenization;
namespace golden = gufo::tests::ornith15::chat;
using gufo::testing::chat_goldens::Expect;
using gufo::testing::chat_goldens::TokenSha256;

int main() {
  const char* path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  if (path == nullptr || !std::filesystem::is_regular_file(path)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL to the APEX MTP "
                 "Quality GGUF\n";
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "Ornith GGUF opens: " + error);
  Expect(reader->GetMetadataString("tokenizer.ggml.pre") ==
             std::optional<std::string_view>{"qwen35"},
         "Ornith GGUF declares the qwen35 pre-tokenizer");
  Expect(orn::ValidateEmbeddedTemplate(*reader, &error),
         "embedded template recognized: " + error);
  const auto tokenizer = qwen::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, "Ornith tokenizer loads: " + error);

  // generation_config.json at the pinned revision stops on both IDs; the GGUF
  // records only the first as eos.
  const auto im_end = tokenizer->FindSpecialToken(qwen::kImEnd);
  const auto end_of_text = tokenizer->FindSpecialToken(qwen::kEndOfText);
  Expect(im_end == std::optional<qwen::TokenId>{248046} &&
             end_of_text == std::optional<qwen::TokenId>{248044},
         "Ornith control token IDs");
  Expect(tokenizer->GetEosTokenId() == 248046, "GGUF eos is <|im_end|>");
  Expect(tokenizer->IsStopToken(248046) && tokenizer->IsStopToken(248044),
         "both generation_config EOS IDs stop generation");

  const golden::Fixture f = golden::Load();
  for (const auto& example : f.root.find("tokenization")->items()) {
    const auto tokens = tokenizer->Encode(example.member_str("text"));
    const auto name = example.member_str("name");
    Expect(tokens.size() == example.member_size("token_count") &&
               TokenSha256<qwen::TokenId>(tokens) ==
                   example.member_str("token_ids_le_u32_sha256"),
           "Hugging Face token IDs: " + name);
  }
  for (const auto& c : f.cases) {
    const auto tokens = qwen::QwenChatTemplate::RenderAndTokenize(
        *tokenizer, c.messages,
        c.tools ? std::span<const qwen::ChatTool>(f.tools)
                : std::span<const qwen::ChatTool>{},
        orn::ResolveChatOptions(golden::Thinking(c.enable_thinking)), &error);
    Expect(tokens.has_value(), "Ornith case tokenizes: " + error);
    if (tokens->size() != c.token_count)
      std::cerr << "case " << c.name << ": expected " << c.token_count
                << " tokens, got " << tokens->size() << '\n';
    Expect(tokens->size() == c.token_count &&
               TokenSha256<qwen::TokenId>(*tokens) == c.token_sha256,
           "Ornith token IDs match Hugging Face: " + c.name);
  }
  std::cout << "Ornith chat goldens tokenize to Hugging Face's "
            << f.cases.size() << " token sequences.\n";
  return 0;
}
