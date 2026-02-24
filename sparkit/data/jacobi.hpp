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
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // Jacobi preconditioner setup.
  //
  // Extracts the diagonal of A and inverts each entry.
  // Returns a vector of inverse diagonal entries.
  //
  // Throws std::invalid_argument if any diagonal entry is zero.

  template <typename T>
  std::vector<T>
  jacobi(Compressed_row_matrix<T> const& A) {
    auto d = extract_diagonal(A);
    for (auto& di : d) {
      if (di == T{0}) {
        throw std::invalid_argument("jacobi: zero diagonal entry");
      }
      di = T{1} / di;
    }
    return d;
  }

  // Jacobi preconditioner apply.
  //
  // Computes z[i] = inv_diag[i] * r[i] (element-wise scaling).

  template <typename T, typename Iter, typename OutIter>
  void
  jacobi_apply(
    std::vector<T> const& inv_diag, Iter first, Iter last, OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    for (std::size_t i = 0; i < n; ++i, ++first, ++out) {
      *out = inv_diag[i] * (*first);
    }
  }

} // end of namespace sparkit::data::detail
