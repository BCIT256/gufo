#ifndef GUFO_MODELS_ORNITH15_TRACE_HPP_
#define GUFO_MODELS_ORNITH15_TRACE_HPP_

#include <vector>

namespace gufo::models::ornith15 {

/// Optional diagnostic residual after each complete trunk layer, before the
/// final output norm. Requesting a trace does not change device arithmetic.
using LayerTrace = std::vector<std::vector<float>>;

}  // namespace gufo::models::ornith15

#endif
