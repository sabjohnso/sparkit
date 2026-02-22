#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::data::detail {

  // Numeric Cholesky factorization with a pre-computed sparsity pattern.
  //
  // Up-looking row Cholesky (CSparse cs_chol style):
  // For each row i of L, scatter A's lower-triangle entries into a dense
  // workspace, update off-diagonal entries via two-pointer dot products
  // with previously computed rows, then compute the diagonal.
  //
  // A must be symmetric positive definite and contain at least the lower
  // triangle. L_pattern must be the result of symbolic_cholesky on A's
  // sparsity (or a compatible superset).
  //
  // Throws std::invalid_argument for non-square or mismatched pattern.
  // Throws std::domain_error if A is not positive definite.

  template <typename T>
  Compressed_row_matrix<T>
  numeric_cholesky(Compressed_row_matrix<T> const& A,
                   Compressed_row_sparsity const& L_pattern) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("numeric_cholesky requires a square matrix");
    }

    if (L_pattern.shape().row() != n || L_pattern.shape().column() != n) {
      throw std::invalid_argument(
          "numeric_cholesky: pattern dimensions do not match matrix");
    }

    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();

    auto l_rp = L_pattern.row_ptr();
    auto l_ci = L_pattern.col_ind();

    auto nnz = L_pattern.size();
    std::vector<T> l_vals(static_cast<std::size_t>(nnz), T{0});

    // Dense workspace for row i under construction
    std::vector<T> x(static_cast<std::size_t>(n), T{0});

    for (config::size_type i = 0; i < n; ++i) {
      // 1. Scatter A's lower-triangle entries for row i into x[]
      for (auto p = a_rp[i]; p < a_rp[i + 1]; ++p) {
        if (a_ci[p] <= i) { x[static_cast<std::size_t>(a_ci[p])] = a_vals[p]; }
      }

      // L's row i: l_ci[l_rp[i] .. l_rp[i+1]-1]
      // Off-diagonals: l_rp[i] .. l_rp[i+1]-2 (columns < i, sorted)
      // Diagonal: l_rp[i+1]-1 (column == i)

      auto row_begin = l_rp[i];
      auto row_end = l_rp[i + 1];

      // 2. Off-diagonal entries: for each column j < i in L's row i
      for (auto pi = row_begin; pi < row_end - 1; ++pi) {
        auto j = l_ci[pi];

        // Two-pointer dot product: sum L(i,k)*L(j,k) for k < j
        // L(i,k) values come from x[k] (workspace, being built)
        // L(j,k) values come from l_vals (already computed)
        auto j_begin = l_rp[j];
        auto j_end = l_rp[j + 1] - 1; // exclude diagonal of row j

        T dot{0};
        auto qi = row_begin; // walk L row i
        auto qj = j_begin;   // walk L row j

        while (qi < pi && qj < j_end) {
          auto ci_k = l_ci[qi];
          auto cj_k = l_ci[qj];
          if (ci_k < cj_k) {
            ++qi;
          } else if (cj_k < ci_k) {
            ++qj;
          } else {
            // Same column k: L(i,k) is in x[k], L(j,k) is in l_vals[qj]
            dot += x[static_cast<std::size_t>(ci_k)] *
                   l_vals[static_cast<std::size_t>(qj)];
            ++qi;
            ++qj;
          }
        }

        // L(i,j) = (A(i,j) - dot) / L(j,j)
        auto j_diag = l_vals[static_cast<std::size_t>(l_rp[j + 1] - 1)];
        x[static_cast<std::size_t>(j)] =
            (x[static_cast<std::size_t>(j)] - dot) / j_diag;
      }

      // 3. Diagonal: x[i] = sqrt(A(i,i) - sum of x[k]^2 for off-diag k)
      T diag_sum{0};
      for (auto pi = row_begin; pi < row_end - 1; ++pi) {
        auto k = l_ci[pi];
        auto xk = x[static_cast<std::size_t>(k)];
        diag_sum += xk * xk;
      }

      auto diag_val = x[static_cast<std::size_t>(i)] - diag_sum;
      if (diag_val <= T{0}) {
        throw std::domain_error(
            "numeric_cholesky: matrix is not positive definite");
      }
      x[static_cast<std::size_t>(i)] = std::sqrt(diag_val);

      // 4. Store x[] into l_vals, then clear workspace
      for (auto pi = row_begin; pi < row_end; ++pi) {
        auto col = l_ci[pi];
        l_vals[static_cast<std::size_t>(pi)] = x[static_cast<std::size_t>(col)];
        x[static_cast<std::size_t>(col)] = T{0};
      }
    }

    return Compressed_row_matrix<T>{L_pattern, std::move(l_vals)};
  }

  // Convenience: combined symbolic + numeric Cholesky.
  template <typename T>
  Compressed_row_matrix<T>
  cholesky(Compressed_row_matrix<T> const& A) {
    auto L_pattern = symbolic_cholesky(A.sparsity());
    return numeric_cholesky(A, L_pattern);
  }

} // end of namespace sparkit::data::detail
