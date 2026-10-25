// Tool-history routing audit for unchanged policy B. Position 96 is the
// corroborated clear-flip anchor. The session linked here is the test-only
// probe build. It does not replace QKV.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/ornith15/cpu_ops.hpp"
#include "tests/models/ornith15/route_margin.hpp"
#include "src/models/ornith15/kernels/rocm/executor.hpp"
#include "src/models/ornith15/reference.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace orn = gufo::models::ornith15;
namespace fn = gufo::models::qwen38_flash_next::rocm;
namespace fs = std::filesystem;
namespace {

constexpr int kTokens = 97;
constexpr int kPosition = 96;
constexpr int kExpectLayer = 6;
constexpr double kClearMargin = 1.0;
constexpr double kFloor = 6.103515625e-5;

void Require(bool ok, const std::string& why) {
  if (!ok)
    throw std::runtime_error(why);
}

struct Capture {
  std::uint32_t vocab{0};
  std::vector<std::int32_t> tokens;
  std::vector<float> logits;
  const float* Row(std::size_t t) const { return logits.data() + t * vocab; }
};

Capture Load(const fs::path& path, std::uint32_t vocab) {
  std::ifstream in(path, std::ios::binary);
  char magic[4]{};
  std::uint32_t header[3]{};
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  Require(in && std::string(magic, 4) == "ORNL" && header[0] == 1 &&
              header[1] >= static_cast<std::uint32_t>(kTokens) &&
              header[2] == vocab,
          "capture " + path.string());
  Capture c{header[2], std::vector<std::int32_t>(header[1]),
            std::vector<float>(std::size_t(header[1]) * header[2])};
  in.read(reinterpret_cast<char*>(c.tokens.data()),
          static_cast<std::streamsize>(c.tokens.size() * sizeof(std::int32_t)));
  in.read(reinterpret_cast<char*>(c.logits.data()),
          static_cast<std::streamsize>(c.logits.size() * sizeof(float)));
  Require(in && in.peek() == EOF, "truncated capture");
  return c;
}

std::size_t Top(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

double Margin(const float* x, std::size_t n, std::size_t top) {
  float second = -INFINITY;
  for (std::size_t i = 0; i < n; ++i)
    if (i != top)
      second = std::max(second, x[i]);
  return static_cast<double>(x[top]) - second;
}

double MaxAbs(const float* a, const float* b, std::size_t n) {
  double m = 0;
  for (std::size_t i = 0; i < n; ++i)
    m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

bool SameSet(const std::int32_t* a, const std::int32_t* b, int k) {
  std::vector<std::int32_t> x(a, a + k), y(b, b + k);
  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  return x == y;
}

void HostMatchedRouter(const float* weights, const float* x, float* out,
                       int rows, int k) {
  for (int row = 0; row < rows; ++row) {
    const float* wrow = weights + static_cast<std::size_t>(row) * k;
    float lane[32] = {};
    for (int lane_i = 0; lane_i < 32; ++lane_i) {
      float acc = 0.0F;
      for (int base = lane_i * 4; base < k; base += 512) {
        for (int step = 0; step < 4; ++step) {
          const int i0 = base + step * 128;
          float a = acc;
          for (int r = 0; r < 4; ++r)
            a = std::fmaf(wrow[i0 + r], x[i0 + r], a);
          volatile float stored = a;
          acc = stored;
        }
      }
      lane[lane_i] = acc;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      float next[32];
      for (int i = 0; i < 32; ++i) {
        volatile float sum = lane[i] + lane[i ^ offset];
        next[i] = sum;
      }
      std::copy(next, next + 32, lane);
    }
    out[row] = lane[0];
  }
}

struct Pick {
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  int ties{0};
  double probability_margin{0};
  double logit_margin{0};
};

Pick Select(const std::vector<double>& probs, const float* logits, int k) {
  const int n = static_cast<int>(probs.size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) {
                      if (probs[a] != probs[b])
                        return probs[a] > probs[b];
                      return a < b;
                    });
  Pick pick;
  pick.ids.resize(k);
  pick.weights.resize(k);
  double sum = 0;
  for (int i = 0; i < k; ++i) {
    pick.ids[i] = order[i];
    sum += probs[order[i]];
    if (i > 0 && probs[order[i]] == probs[order[i - 1]])
      ++pick.ties;
  }
  for (int e = 0; e < n; ++e) {
    if (std::find(order.begin(), order.begin() + k, e) != order.begin() + k)
      continue;
    for (int i = 0; i < k; ++i)
      if (probs[e] == probs[order[i]])
        ++pick.ties;
  }
  const RankBoundary boundary = RankTop(probs, logits, k);
  pick.probability_margin = boundary.probability_margin;
  pick.logit_margin = boundary.logit_margin;
  sum = std::max(sum, static_cast<double>(kFloor));
  for (int i = 0; i < k; ++i)
    pick.weights[i] = static_cast<float>(probs[order[i]] / sum);
  return pick;
}

std::vector<double> Softmax(const float* logits, int n, bool f64) {
  double peak = logits[0];
  for (int i = 1; i < n; ++i)
    peak = std::max(peak, static_cast<double>(logits[i]));
  std::vector<double> p(n);
  if (f64) {
    double denom = 0;
    for (int i = 0; i < n; ++i) {
      p[i] = std::exp(logits[i] - peak);
      denom += p[i];
    }
    for (double& v : p)
      v /= denom;
  } else {
    float denom = 0;
    for (int i = 0; i < n; ++i)
      denom += std::exp(static_cast<float>(logits[i]) - static_cast<float>(peak));
    for (int i = 0; i < n; ++i)
      p[i] = std::exp(static_cast<float>(logits[i]) - static_cast<float>(peak)) /
             denom;
  }
  return p;
}

void Reduce(const float* rows, const float* weights, const float* shared,
            float gate, float* out, int k, int dim, bool f64) {
  const float sig = 1.0F / (1.0F + std::exp(-gate));
  for (int i = 0; i < dim; ++i) {
    if (f64) {
      double acc = 0;
      for (int s = 0; s < k; ++s)
        acc += static_cast<double>(weights[s]) * rows[s * dim + i];
      out[i] = static_cast<float>(acc + static_cast<double>(sig) * shared[i]);
    } else {
      float acc = 0;
      for (int s = 0; s < k; ++s)
        acc = std::fmaf(weights[s], rows[s * dim + i], acc);
      out[i] = acc + sig * shared[i];
    }
  }
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_ORNITH15_APEX_MTP_MODEL");
  const char* logits_dir = std::getenv("GUFO_ORNITH15_LLAMA_LOGITS");
  const char* out_env = std::getenv("GUFO_ORNITH15_ROUTE_DIR");
  if (!model_path || !logits_dir || !fs::is_regular_file(model_path) ||
      !fs::is_directory(logits_dir)) {
    std::cout << "SKIP: set GUFO_ORNITH15_APEX_MTP_MODEL and "
                 "GUFO_ORNITH15_LLAMA_LOGITS\n";
    return 77;
  }
  try {
    std::cout << std::setprecision(17);
    std::cout << "experiment=tool_two_parameters_position96_route_audit\n"
                 "policy=B unchanged\n"
                 "anchor=position 96 corroborated clear flip\n"
                 "expected_first_expert_set_layer=6\n";
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    Require(reader != nullptr, error);
    const auto weights =
        orn::ModelWeights::Bind(*reader, orn::BindOptions{}, &error);
    Require(weights.has_value() && !weights->mtp, error);
    const auto device =
        orn::rocm::DeviceModel::Upload(*weights, *reader, &error);
    Require(device != nullptr, error);
    const auto& c = device->config();
    const int layers = static_cast<int>(c.num_layers);
    const int used = static_cast<int>(c.num_experts_used);
    const int experts = static_cast<int>(c.num_experts);
    const int hidden = static_cast<int>(c.hidden_size);
    Require(layers == 40 && used == 8 && experts == 256 && hidden == 2048 &&
                c.IsLinearLayer(kExpectLayer),
            "shape");
    const auto batch =
        Load(fs::path(logits_dir) / "tool_two_parameters.ologits", c.vocab_size);
    const auto sequential = Load(
        fs::path(logits_dir) / "tool_two_parameters.sequential.ologits",
        c.vocab_size);
    Require(batch.tokens == sequential.tokens, "token identity");
    std::vector<std::int32_t> tokens(batch.tokens.begin(),
                                     batch.tokens.begin() + kTokens);

    const std::size_t route_n =
        static_cast<std::size_t>(kTokens) * layers * used;
    std::vector<std::int32_t> scalar_ids(route_n), gpu_ids(route_n);
    std::vector<float> scalar_w(route_n), gpu_w(route_n);
    std::vector<float> scalar_logits(c.vocab_size), gpu_logits(c.vocab_size);
    std::vector<float> scalar_in(static_cast<std::size_t>(layers) * hidden);
    std::vector<float> moe_in(hidden), router(experts),
        expert_down(static_cast<std::size_t>(used) * hidden), shared(hidden),
        moe_out(hidden), shared_gate(layers);
    int first_residual_token = -1, first_residual_layer = -1;

    orn::ReferenceModel ref(*weights, kTokens);
    auto session = orn::rocm::Session::Create(*device, kTokens, &error);
    Require(session != nullptr, error);
    std::vector<float> row(c.vocab_size);
    std::vector<std::int32_t> step_ids(static_cast<std::size_t>(layers) * used);
    std::vector<float> step_w(step_ids.size()), step_in(scalar_in.size());
    orn::LayerTrace scalar_trace, gpu_trace;
    for (int t = 0; t < kTokens; ++t) {
      Require(ref.Step(tokens[t], row, &error, &scalar_trace, step_ids.data(),
                       step_w.data(), step_in.data()),
              "reference " + error);
      std::copy(step_ids.begin(), step_ids.end(),
                scalar_ids.begin() + static_cast<std::size_t>(t) * layers * used);
      std::copy(step_w.begin(), step_w.end(),
                scalar_w.begin() + static_cast<std::size_t>(t) * layers * used);
      if (t == kPosition) {
        scalar_logits = row;
        scalar_in = step_in;
      }
      orn::rocm::Layer0Probe probe;
      probe.route_ids = step_ids.data();
      probe.route_weights = step_w.data();
      if (t == kPosition) {
        probe.detail_layer = kExpectLayer;
        probe.detail_moe_in = moe_in.data();
        probe.detail_router = router.data();
        probe.detail_expert_down = expert_down.data();
        probe.detail_shared = shared.data();
        probe.detail_moe_out = moe_out.data();
        probe.shared_gate = shared_gate.data();
      }
      Require(session->Step(tokens[t], row, &error, &gpu_trace, &probe),
              "gpu " + error);
      std::copy(step_ids.begin(), step_ids.end(),
                gpu_ids.begin() + static_cast<std::size_t>(t) * layers * used);
      std::copy(step_w.begin(), step_w.end(),
                gpu_w.begin() + static_cast<std::size_t>(t) * layers * used);
      if (t == kPosition)
        gpu_logits = row;
      if (first_residual_token < 0) {
        for (int layer = 0; layer < layers; ++layer) {
          if (std::memcmp(scalar_trace[layer].data(), gpu_trace[layer].data(),
                          hidden * sizeof(float)) != 0) {
            first_residual_token = t;
            first_residual_layer = layer;
            break;
          }
        }
      }
      std::cout << "token=" << t << " done\n";
    }

    const auto rt = Top(scalar_logits.data(), c.vocab_size);
    const auto gt = Top(gpu_logits.data(), c.vocab_size);
    const auto lt = Top(batch.Row(kPosition), c.vocab_size);
    const auto st = Top(sequential.Row(kPosition), c.vocab_size);
    const double margin = Margin(scalar_logits.data(), c.vocab_size, rt);
    const bool llama_agree = lt == rt && st == rt;
    const bool clear_flip = rt != gt && margin > kClearMargin && llama_agree;
    std::cout << "position=96 margin=" << margin << " top_ref=" << rt
              << " top_gpu=" << gt << " top_llama=" << lt
              << " top_sequential=" << st << " llama_agree=" << llama_agree
              << " clear_flip=" << clear_flip << "\n";

    int first_set_token = -1, first_set_layer = -1;
    int pos_first_set = -1, pos_first_order = -1;
    for (int t = 0; t < kTokens; ++t) {
      for (int layer = 0; layer < layers; ++layer) {
        const auto off = (static_cast<std::size_t>(t) * layers + layer) * used;
        const bool set =
            SameSet(scalar_ids.data() + off, gpu_ids.data() + off, used);
        const bool order = std::memcmp(scalar_ids.data() + off,
                                       gpu_ids.data() + off,
                                       used * sizeof(std::int32_t)) == 0;
        if (!set && first_set_token < 0) {
          first_set_token = t;
          first_set_layer = layer;
        }
        if (t == kPosition && !set && pos_first_set < 0)
          pos_first_set = layer;
        if (t == kPosition && set && !order && pos_first_order < 0)
          pos_first_order = layer;
      }
    }
    std::cout << "first_expert_set_mismatch token=" << first_set_token
              << " layer=" << first_set_layer << "\n";
    std::cout << "position96_first_expert_set_layer=" << pos_first_set
              << " position96_first_order_only_layer=" << pos_first_order
              << " verified_against_6=" << (pos_first_set == kExpectLayer)
              << "\n";
    std::cout << "first_residual_mismatch token=" << first_residual_token
              << " layer=" << first_residual_layer << "\n";

    const auto off96 =
        (static_cast<std::size_t>(kPosition) * layers + kExpectLayer) * used;
    std::cout << "layer6_gpu_ids";
    for (int i = 0; i < used; ++i)
      std::cout << " " << gpu_ids[off96 + i];
    std::cout << "\nlayer6_scalar_ids";
    for (int i = 0; i < used; ++i)
      std::cout << " " << scalar_ids[off96 + i];
    std::cout << "\nlayer6_set_match="
              << SameSet(gpu_ids.data() + off96, scalar_ids.data() + off96, used)
              << " layer6_order_match="
              << (std::memcmp(gpu_ids.data() + off96, scalar_ids.data() + off96,
                              used * sizeof(std::int32_t)) == 0)
              << " moe_in_max_abs="
              << MaxAbs(moe_in.data(),
                        scalar_in.data() +
                            static_cast<std::size_t>(kExpectLayer) * hidden,
                        hidden)
              << "\n";

    hipStream_t stream = nullptr;
    Require(hipStreamCreate(&stream) == hipSuccess, "stream");
    float *dx = nullptr, *dy = nullptr;
    std::int32_t* dids = nullptr;
    float* dweights = nullptr;
    Require(hipMalloc(&dx, hidden * sizeof(float)) == hipSuccess, "x");
    Require(hipMalloc(&dy, experts * sizeof(float)) == hipSuccess, "y");
    Require(hipMalloc(&dids, used * sizeof(std::int32_t)) == hipSuccess, "ids");
    Require(hipMalloc(&dweights, used * sizeof(float)) == hipSuccess, "w");
    Require(hipMemcpy(dx, moe_in.data(), hidden * sizeof(float),
                      hipMemcpyHostToDevice) == hipSuccess,
            "upload x");
    const auto& router_w = device->layers().at(kExpectLayer).router;
    Require(router_w.type == gufo::core::GgmlType::kF32 &&
                router_w.rows == static_cast<std::uint32_t>(experts) &&
                router_w.cols == static_cast<std::uint32_t>(hidden),
            "router tensor");
    fn::SmallGemm(router_w.data, fn::WeightType::kF32, dx, dy, 1, experts,
                  hidden, stream);
    std::vector<float> relaunch(experts);
    Require(hipMemcpy(relaunch.data(), dy, experts * sizeof(float),
                      hipMemcpyDeviceToHost) == hipSuccess,
            "download router");
    fn::RouterTopK(dy, experts, dids, dweights, 1, experts, used, stream);
    std::vector<std::int32_t> relaunch_ids(used);
    std::vector<float> relaunch_w(used);
    Require(hipMemcpy(relaunch_ids.data(), dids, used * sizeof(std::int32_t),
                      hipMemcpyDeviceToHost) == hipSuccess,
            "download ids");
    Require(hipMemcpy(relaunch_w.data(), dweights, used * sizeof(float),
                      hipMemcpyDeviceToHost) == hipSuccess,
            "download weights");
    Require(hipStreamSynchronize(stream) == hipSuccess, "sync relaunch");
    const bool router_relaunch =
        std::memcmp(relaunch.data(), router.data(), experts * sizeof(float)) ==
        0;
    const bool route_relaunch =
        std::memcmp(relaunch_ids.data(), gpu_ids.data() + off96,
                    used * sizeof(std::int32_t)) == 0 &&
        std::memcmp(relaunch_w.data(), gpu_w.data() + off96,
                    used * sizeof(float)) == 0;
    std::cout << "identical_input_router_byte_identical=" << router_relaunch
              << " identical_input_topk_byte_identical=" << route_relaunch
              << "\n";

    const float* host_w = static_cast<const float*>(
        weights->layers.at(kExpectLayer).router.data);
    std::vector<float> matched(experts), f64(experts, 0.0F);
    HostMatchedRouter(host_w, moe_in.data(), matched.data(), experts, hidden);
    orn::cpu::MatVec(weights->layers.at(kExpectLayer).router, 0, moe_in, f64);
    std::cout << "router_G_vs_M_max_abs="
              << MaxAbs(router.data(), matched.data(), experts)
              << " router_G_vs_M_bytes="
              << (std::memcmp(router.data(), matched.data(),
                              experts * sizeof(float)) == 0)
              << " router_M_vs_F_max_abs="
              << MaxAbs(matched.data(), f64.data(), experts)
              << " router_G_vs_F_max_abs="
              << MaxAbs(router.data(), f64.data(), experts) << "\n";

    const auto g_probs = Softmax(router.data(), experts, true);
    const auto f_probs = Softmax(f64.data(), experts, true);
    const Pick g_pick = Select(g_probs, router.data(), used);
    const Pick f_pick = Select(f_probs, f64.data(), used);
    std::cout << "gpu_logits_mathematical_top8";
    for (int id : g_pick.ids)
      std::cout << " " << id;
    std::cout << "\nf64_logits_mathematical_top8";
    for (int id : f_pick.ids)
      std::cout << " " << id;
    std::cout << "\nselection_matches_gpu_logits="
              << (std::memcmp(g_pick.ids.data(), gpu_ids.data() + off96,
                              used * sizeof(std::int32_t)) == 0)
              << " selection_matches_f64_logits="
              << (std::memcmp(f_pick.ids.data(), gpu_ids.data() + off96,
                              used * sizeof(std::int32_t)) == 0)
              << " gpu_probability_margin=" << g_pick.probability_margin
              << " gpu_logit_margin=" << g_pick.logit_margin
              << " f64_probability_margin=" << f_pick.probability_margin
              << " f64_logit_margin=" << f_pick.logit_margin
              << " tie_count_gpu_logits=" << g_pick.ties
              << " tie_count_f64_logits=" << f_pick.ties
              << " weight_max_abs_vs_gpu_softmax="
              << MaxAbs(g_pick.weights.data(), gpu_w.data() + off96, used)
              << "\n";

    std::vector<float> reduced(hidden), reduced_f(hidden);
    const float gate = shared_gate[kExpectLayer];
    Reduce(expert_down.data(), gpu_w.data() + off96, shared.data(), gate,
           reduced.data(), used, hidden, false);
    Reduce(expert_down.data(), gpu_w.data() + off96, shared.data(), gate,
           reduced_f.data(), used, hidden, true);
    std::cout << "expert_slot_order=selection_order_consumed_by_mmvq\n"
              << "shared_gate=" << gate
              << " reduction_G_vs_fma_max_abs="
              << MaxAbs(moe_out.data(), reduced.data(), hidden)
              << " reduction_G_vs_fma_bytes="
              << (std::memcmp(moe_out.data(), reduced.data(),
                              hidden * sizeof(float)) == 0)
              << " reduction_G_vs_f64_max_abs="
              << MaxAbs(moe_out.data(), reduced_f.data(), hidden)
              << " reduction_fma_vs_f64_max_abs="
              << MaxAbs(reduced.data(), reduced_f.data(), hidden) << "\n";
    const fs::path out_dir = out_env ? fs::path(out_env)
                                     : fs::path("/tmp/ornith-route-audit");
    fs::create_directories(out_dir);
    auto save = [&](const char* name, const float* data, std::size_t n) {
      std::ofstream f(out_dir / name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(n * sizeof(float)));
    };
    save("layer6_moe_in.f32", moe_in.data(), hidden);
    save("layer6_router_gpu.f32", router.data(), experts);
    save("layer6_router_matched.f32", matched.data(), experts);
    save("layer6_router_f64.f32", f64.data(), experts);
    save("layer6_expert_down.f32", expert_down.data(),
         static_cast<std::size_t>(used) * hidden);
    save("layer6_shared.f32", shared.data(), hidden);
    save("layer6_moe_out.f32", moe_out.data(), hidden);
    save("layer6_weights_gpu.f32", gpu_w.data() + off96, used);
    std::cout << "fixtures=" << out_dir.string() << "\nmeasurement_complete=1\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
