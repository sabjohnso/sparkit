#pragma once

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // IC(0) incomplete Cholesky factorization.
  //
  // Computes L such that L*L^T approximates A, where L has the same
  // sparsity pattern as the lower triangle of A (no fill-in). This
  // delegates to numeric_cholesky with the lower-triangle pattern,
  // which naturally drops fill entries not present in the pattern.
  //
  // Throws std::invalid_argument if A is not square.
  // Throws std::domain_error if A is not sufficiently positive definite.

  template <typename T>
  Compressed_row_matrix<T>
  incomplete_cholesky(Compressed_row_matrix<T> const& A) {
    auto lower = extract_lower_triangle(A, true);
    return numeric_cholesky(A, lower.sparsity());
  }

} // end of namespace sparkit::data::detail
