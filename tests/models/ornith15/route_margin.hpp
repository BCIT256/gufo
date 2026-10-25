#ifndef GUFO_TESTS_ORNITH15_ROUTE_MARGIN_HPP_
#define GUFO_TESTS_ORNITH15_ROUTE_MARGIN_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// Top-k boundary after a full ordering of the tail. partial_sort of only k
// elements leaves the entry at position k in the unsorted remainder, so that
// entry is not the ninth place.
struct RankBoundary {
  int eighth{-1};
  int ninth{-1};
  double probability_margin{0};
  double logit_margin{0};
  int ties{0};
};

inline RankBoundary RankTop(const std::vector<double>& probability,
                            const float* logits, int k) {
  const int n = static_cast<int>(probability.size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + (k + 1), order.end(),
                    [&](int a, int b) {
                      if (probability[a] != probability[b])
                        return probability[a] > probability[b];
                      return a < b;
                    });
  RankBoundary out;
  out.eighth = order[k - 1];
  out.ninth = order[k];
  out.probability_margin = probability[out.eighth] - probability[out.ninth];
  if (logits) {
    out.logit_margin = static_cast<double>(logits[out.eighth]) -
                       static_cast<double>(logits[out.ninth]);
  }
  for (int i = 0; i < k; ++i)
    if (probability[order[i]] == probability[order[i + 1]])
      ++out.ties;
  return out;
}

#endif
