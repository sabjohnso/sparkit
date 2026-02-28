#pragma once

//
// ... Standard header files
//
#include <iterator>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/info.hpp>

namespace sparkit::data::detail {

  // Column sum preconditioner setup.
  //
  // Computes the absolute column sums (1-norms) of A and inverts each entry.
  // Returns a vector of inverse absolute column sums.
  //
  // Throws std::invalid_argument if any column sum is zero.

  template <typename T>
  std::vector<T>
  column_sum(Compressed_row_matrix<T> const& A) {
    auto col_norms = column_norms_1(A);
    for (auto& c : col_norms) {
      if (c == T{0}) {
        throw std::invalid_argument("column_sum: zero column sum");
      }
      c = T{1} / c;
    }
    return col_norms;
  }

  // Column sum preconditioner apply.
  //
  // Computes z[i] = inv_col_sums[i] * r[i] (element-wise scaling).

  template <typename T, typename Iter, typename OutIter>
  void
  column_sum_apply(
    std::vector<T> const& inv_col_sums, Iter first, Iter last, OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    for (std::size_t i = 0; i < n; ++i, ++first, ++out) {
      *out = inv_col_sums[i] * (*first);
    }
  }

} // end of namespace sparkit::data::detail
