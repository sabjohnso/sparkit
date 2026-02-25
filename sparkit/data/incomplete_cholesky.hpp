#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/triangular_solve.hpp>
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

  // Apply IC preconditioner: z = L^{-T} L^{-1} r.
  // Delegates to forward_solve (L) then forward_solve_transpose (L).

  template <typename T, typename Iter, typename OutIter>
  void
  ic_apply(
    Compressed_row_matrix<T> const& L, Iter first, Iter last, OutIter out) {
    auto y = forward_solve(L, std::span<T const>{first, last});
    auto z = forward_solve_transpose(L, std::span<T const>{y});
    std::copy(z.begin(), z.end(), out);
  }

  // MIC(0): modified incomplete Cholesky factorization.
  //
  // Column-based (left-looking) factorization with symmetric diagonal
  // compensation for dropped fill (Gustafsson 1978). Same sparsity
  // pattern as the lower triangle of A, but when a fill entry (i,j)
  // is dropped, both A(i,i) and A(j,j) are compensated, preserving
  // row sums: (L*L^T)*e = A*e.
  //
  // For each column k, computes L(k,k) and all L(i,k) for i > k,
  // then applies the rank-1 update L(:,k)*L(:,k)^T to entries (i,j)
  // with i,j > k. If (i,j) is not in the pattern, both diagonals
  // A(i,i) and A(j,j) are reduced by L(i,k)*L(j,k).
  //
  // Throws std::invalid_argument if A is not square.
  // Throws std::domain_error if A is not sufficiently positive definite.

  template <typename T>
  Compressed_row_matrix<T>
  mic0(Compressed_row_matrix<T> const& A) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("mic0 requires a square matrix");
    }

    // Extract lower triangle with diagonal
    auto lower = extract_lower_triangle(A, true);
    auto l_rp = lower.row_ptr();
    auto l_ci = lower.col_ind();

    // Mutable copy of values for in-place factorization
    auto vals_span = lower.values();
    std::vector<T> vals(vals_span.begin(), vals_span.end());

    auto un = static_cast<std::size_t>(n);

    // Build column lists: col_list[k] = [(row_j, position)] for rows
    // j > k that have column k as an off-diagonal entry. Entries are
    // in ascending row order (since we iterate rows 0..n-1).
    using row_pos = std::pair<config::size_type, config::size_type>;
    std::vector<std::vector<row_pos>> col_list(un);

    for (config::size_type i = 0; i < n; ++i) {
      for (auto p = l_rp[i]; p < l_rp[i + 1] - 1; ++p) {
        auto k = l_ci[p];
        col_list[static_cast<std::size_t>(k)].push_back({i, p});
      }
    }

    // Column-to-position map (reset per inner loop iteration)
    std::vector<config::size_type> col_map(un, -1);

    for (config::size_type k = 0; k < n; ++k) {
      auto uk = static_cast<std::size_t>(k);
      auto k_diag_p = static_cast<std::size_t>(l_rp[k + 1] - 1);

      // Step 1: L(k,k) = sqrt(current diagonal value)
      auto diag_val = vals[k_diag_p];
      if (diag_val <= T{0}) {
        throw std::domain_error("mic0: matrix is not positive definite");
      }
      vals[k_diag_p] = std::sqrt(diag_val);
      auto l_kk = vals[k_diag_p];

      // Step 2: L(i,k) = current value / L(k,k) for all i in col_list[k]
      for (auto& [i, pos_ik] : col_list[uk]) {
        vals[static_cast<std::size_t>(pos_ik)] /= l_kk;
      }

      // Step 3: Rank-1 update with symmetric dropped-fill compensation
      auto const& col_entries = col_list[uk];

      for (std::size_t ii = 0; ii < col_entries.size(); ++ii) {
        auto [ri, pos_ik] = col_entries[ii];
        auto l_ik = vals[static_cast<std::size_t>(pos_ik)];
        auto ri_diag_p = static_cast<std::size_t>(l_rp[ri + 1] - 1);

        // Build col_map for row ri
        for (auto p = l_rp[ri]; p < l_rp[ri + 1]; ++p) {
          col_map[static_cast<std::size_t>(l_ci[p])] = p;
        }

        // Diagonal update: L(i,k)^2 (always in pattern)
        vals[ri_diag_p] -= l_ik * l_ik;

        // Off-diagonal: for each j < i in col_list[k]
        for (std::size_t jj = 0; jj < ii; ++jj) {
          auto [rj, pos_jk] = col_entries[jj];
          auto l_jk = vals[static_cast<std::size_t>(pos_jk)];

          auto target = col_map[static_cast<std::size_t>(rj)];
          if (target >= 0) {
            // (ri, rj) in pattern: normal update
            vals[static_cast<std::size_t>(target)] -= l_ik * l_jk;
          } else {
            // Dropped fill: compensate both diagonals for symmetry.
            // Fill at (ri,rj) also implies fill at (rj,ri) in L*L^T,
            // so both row ri and row rj sums are affected.
            auto rj_diag_p = static_cast<std::size_t>(l_rp[rj + 1] - 1);
            vals[ri_diag_p] -= l_ik * l_jk;
            vals[rj_diag_p] -= l_ik * l_jk;
          }
        }

        // Clean up col_map
        for (auto p = l_rp[ri]; p < l_rp[ri + 1]; ++p) {
          col_map[static_cast<std::size_t>(l_ci[p])] = -1;
        }
      }
    }

    return Compressed_row_matrix<T>{lower.sparsity(), std::move(vals)};
  }

} // end of namespace sparkit::data::detail
