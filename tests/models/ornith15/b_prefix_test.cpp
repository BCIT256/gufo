// Ordinary production B on code_diff tokens 0..75. This binary links the
// session built without GUFO_ORNITH15_TEST_PROBE.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/ornith15/kernels/rocm/executor.hpp"

namespace orn = gufo::models::ornith15;
namespace fs = std::filesystem;
namespace {

constexpr int kTokens = 76;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  const char* out_path = std::getenv("GUFO_ORNITH15_UNPROBED_LOGITS");
  if (!model_path || !logits_dir || !out_path ||
      !fs::is_regular_file(model_path) || !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL, "
                 "GUFO_ORNITH15_LLAMA_LOGITS and GUFO_ORNITH15_UNPROBED_LOGITS\n";
    return 77;
  }
  try {
    std::cout << "build=unprobed_production_session tokens=0-75\n";
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, error);
    const auto vocab = device->config().vocab_size;
    std::ifstream in(fs::path(logits_dir) / "code_diff.ologits", std::ios::binary);
    char magic[4]{};
    std::uint32_t header[3]{};
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    Require(std::string(magic, 4) == "ORNL" && header[0] == 1 &&
                header[1] >= static_cast<std::uint32_t>(kTokens) &&
                header[2] == vocab,
            "capture");
    std::vector<std::int32_t> tokens(header[1]);
    in.read(reinterpret_cast<char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
    Require(static_cast<bool>(in), "tokens");
    auto session = orn::rocm::Session::Create(*device, kTokens, &error);
    Require(session != nullptr, error);
    std::vector<float> logits(vocab), all(std::size_t(kTokens) * vocab);
    for (int t = 0; t < kTokens; ++t) {
      Require(session->Step(tokens[t], logits, &error), error);
      std::copy(logits.begin(), logits.end(),
                all.begin() + std::size_t(t) * vocab);
      std::cout << "arm=unprobed token=" << t << " done\n";
    }
    fs::create_directories(fs::path(out_path).parent_path());
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(all.data()),
              static_cast<std::streamsize>(all.size() * sizeof(float)));
    Require(static_cast<bool>(out), "write logits");
    std::cout << "wrote=" << out_path << " floats=" << all.size() << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
