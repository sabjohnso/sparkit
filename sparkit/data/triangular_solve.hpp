#pragma once

//
// ... Standard header files
//
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // Forward substitution: solve L*x = b where L is lower triangular CSR.
  // L must be square with nonzero diagonal (last entry in each row).
  //
  // Row-by-row, top to bottom (cf. CSparse cs_lsolve):
  //   x = copy of b
  //   for i = 0 to n-1:
  //     for j in L.row(i) where j < i:
  //       x[i] -= L(i,j) * x[j]
  //     x[i] /= L(i,i)

  template <typename T>
  std::vector<T>
  forward_solve(Compressed_row_matrix<T> const& L, std::span<T const> b) {
    auto n = L.shape().row();

    if (L.shape().row() != L.shape().column()) {
      throw std::invalid_argument("forward_solve requires a square matrix");
    }

    auto rp = L.row_ptr();
    auto ci = L.col_ind();
    auto vals = L.values();

    std::vector<T> x(b.begin(), b.end());

    for (config::size_type i = 0; i < n; ++i) {
      // Off-diagonal entries: j < i
      for (auto p = rp[i]; p < rp[i + 1] - 1; ++p) {
        x[static_cast<std::size_t>(i)] -=
            vals[p] * x[static_cast<std::size_t>(ci[p])];
      }

      // Diagonal: last entry in row
      auto diag_pos = rp[i + 1] - 1;
      x[static_cast<std::size_t>(i)] /= vals[diag_pos];
    }

    return x;
  }

  // Backward substitution: solve U*x = b where U is upper triangular CSR.
  // U must be square with nonzero diagonal (first entry in each row).
  //
  // Row-by-row, bottom to top:
  //   x = copy of b
  //   for i = n-1 downto 0:
  //     for j in U.row(i) where j > i:
  //       x[i] -= U(i,j) * x[j]
  //     x[i] /= U(i,i)

  template <typename T>
  std::vector<T>
  backward_solve(Compressed_row_matrix<T> const& U, std::span<T const> b) {
    auto n = U.shape().row();

    if (U.shape().row() != U.shape().column()) {
      throw std::invalid_argument("backward_solve requires a square matrix");
    }

    auto rp = U.row_ptr();
    auto ci = U.col_ind();
    auto vals = U.values();

    std::vector<T> x(b.begin(), b.end());

    for (auto ii = n; ii > 0; --ii) {
      auto i = ii - 1;

      // Diagonal: first entry in row
      auto diag_pos = rp[i];

      // Off-diagonal entries: j > i
      for (auto p = rp[i] + 1; p < rp[i + 1]; ++p) {
        x[static_cast<std::size_t>(i)] -=
            vals[p] * x[static_cast<std::size_t>(ci[p])];
      }

      x[static_cast<std::size_t>(i)] /= vals[diag_pos];
    }

    return x;
  }

  // Solve L^T*x = b using L (lower triangular CSR) without forming L^T.
  //
  // L^T is upper triangular. Traverse L's rows in reverse, scattering
  // contributions — the column-oriented view of L^T:
  //   x = copy of b
  //   for i = n-1 downto 0:
  //     x[i] /= L(i,i)
  //     for j in L.row(i) where j < i:
  //       x[j] -= L(i,j) * x[i]

  template <typename T>
  std::vector<T>
  forward_solve_transpose(Compressed_row_matrix<T> const& L,
                          std::span<T const> b) {
    auto n = L.shape().row();

    if (L.shape().row() != L.shape().column()) {
      throw std::invalid_argument(
          "forward_solve_transpose requires a square matrix");
    }

    auto rp = L.row_ptr();
    auto ci = L.col_ind();
    auto vals = L.values();

    std::vector<T> x(b.begin(), b.end());

    for (auto ii = n; ii > 0; --ii) {
      auto i = ii - 1;

      // Diagonal of L^T row i = L(i,i), which is last entry in L's row i
      auto diag_pos = rp[i + 1] - 1;
      x[static_cast<std::size_t>(i)] /= vals[diag_pos];

      // Off-diagonal: L(i,j) for j < i  =>  L^T(j,i) = L(i,j)
      for (auto p = rp[i]; p < rp[i + 1] - 1; ++p) {
        x[static_cast<std::size_t>(ci[p])] -=
            vals[p] * x[static_cast<std::size_t>(i)];
      }
    }

    return x;
  }

} // end of namespace sparkit::data::detail
