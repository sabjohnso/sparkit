#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::data::detail {

  // Incomplete LU factors: L (lower, unit diagonal last) and U (upper,
  // computed diagonal first). Compatible with forward_solve / backward_solve.

  template <typename T>
  struct Ilu_factors {
    Compressed_row_matrix<T> L;
    Compressed_row_matrix<T> U;
  };

  // Shared ILU(0)/MILU(0) implementation.
  //
  // IKJ row-by-row factorization (Saad, Algorithm 10.4). When `modified`
  // is true, dropped fill is accumulated and added to the diagonal,
  // preserving row sums.

  template <typename T>
  Ilu_factors<T>
  ilu0_impl(Compressed_row_matrix<T> const& A, bool modified) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("ilu0 requires a square matrix");
    }

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto a_vals = A.values();

    // Step 1: Mutable copy of values for in-place factorization
    std::vector<T> w(a_vals.begin(), a_vals.end());

    // Step 2: Locate diagonal position in each row
    std::vector<config::size_type> diag_pos(static_cast<std::size_t>(n));

    for (config::size_type i = 0; i < n; ++i) {
      bool found = false;
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] == i) {
          diag_pos[static_cast<std::size_t>(i)] = p;
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::invalid_argument("ilu0: missing diagonal entry");
      }
    }

    // Step 3: Dense column-to-position map
    std::vector<config::size_type> col_map(static_cast<std::size_t>(n), -1);

    // Step 4: Row-by-row elimination
    for (config::size_type i = 0; i < n; ++i) {
      // Scatter: mark positions of row i entries
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        col_map[static_cast<std::size_t>(ci[p])] = p;
      }

      T dropped_sum{0};

      // Eliminate: for each k < i in row i
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto k = ci[p];
        if (k >= i) { break; }

        // l_ik = a_ik / u_kk
        w[static_cast<std::size_t>(p)] /=
          w[static_cast<std::size_t>(diag_pos[static_cast<std::size_t>(k)])];

        auto l_ik = w[static_cast<std::size_t>(p)];

        // Update row i from U's row k (entries after diagonal of k)
        for (auto q = diag_pos[static_cast<std::size_t>(k)] + 1; q < rp[k + 1];
             ++q) {
          auto j = ci[q];
          auto pos = col_map[static_cast<std::size_t>(j)];
          if (pos >= 0) {
            w[static_cast<std::size_t>(pos)] -=
              l_ik * w[static_cast<std::size_t>(q)];
          } else if (modified) {
            dropped_sum += l_ik * w[static_cast<std::size_t>(q)];
          }
        }
      }

      // MILU: compensate diagonal for dropped fill (preserve row sums)
      if (modified) {
        w[static_cast<std::size_t>(diag_pos[static_cast<std::size_t>(i)])] -=
          dropped_sum;
      }

      // Clean up col_map
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        col_map[static_cast<std::size_t>(ci[p])] = -1;
      }
    }

    // Step 5: Extract L and U
    std::vector<Index> l_indices;
    std::vector<T> l_vals;
    std::vector<Index> u_indices;
    std::vector<T> u_vals;

    for (config::size_type i = 0; i < n; ++i) {
      // L: entries with j < i, then unit diagonal
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] < i) {
          l_indices.push_back(Index{i, ci[p]});
          l_vals.push_back(w[static_cast<std::size_t>(p)]);
        }
      }
      l_indices.push_back(Index{i, i});
      l_vals.push_back(T{1});

      // U: diagonal then entries with j > i
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] >= i) {
          u_indices.push_back(Index{i, ci[p]});
          u_vals.push_back(w[static_cast<std::size_t>(p)]);
        }
      }
    }

    auto shape = A.shape();
    Compressed_row_sparsity l_sp{shape, l_indices.begin(), l_indices.end()};
    Compressed_row_sparsity u_sp{shape, u_indices.begin(), u_indices.end()};

    return Ilu_factors<T>{
      Compressed_row_matrix<T>{std::move(l_sp), std::move(l_vals)},
      Compressed_row_matrix<T>{std::move(u_sp), std::move(u_vals)}};
  }

  // ILU(0): zero-fill incomplete LU factorization.
  // Maintains the sparsity pattern of A.

  template <typename T>
  Ilu_factors<T>
  ilu0(Compressed_row_matrix<T> const& A) {
    return ilu0_impl(A, false);
  }

  // MILU(0): modified ILU(0).
  // Dropped fill is added to diagonal, preserving row sums: (L*U)*e = A*e.

  template <typename T>
  Ilu_factors<T>
  milu0(Compressed_row_matrix<T> const& A) {
    return ilu0_impl(A, true);
  }

  // Apply ILU preconditioner: z = U^{-1} L^{-1} r.
  // Delegates to forward_solve (L) then backward_solve (U).

  template <typename T, typename Iter, typename OutIter>
  void
  ilu_apply(Ilu_factors<T> const& factors, Iter first, Iter last, OutIter out) {
    auto y = forward_solve(factors.L, std::span<T const>{first, last});
    auto z = backward_solve(factors.U, std::span<T const>{y});
    std::copy(z.begin(), z.end(), out);
  }

} // end of namespace sparkit::data::detail
