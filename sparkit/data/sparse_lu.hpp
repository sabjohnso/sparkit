#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/reordering.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::data::detail {

  // LU factors: L (unit lower triangular, diagonal last) and U (upper
  // triangular, diagonal first), with row and column permutations.
  // P_r * A * P_c = L * U.

  template <typename T>
  struct Lu_factors {
    Compressed_row_matrix<T> L;
    Compressed_row_matrix<T> U;
    std::vector<config::size_type> row_perm;
    std::vector<config::size_type> col_perm;
  };

  // Sparse LU factorization with partial pivoting.
  //
  // Column-by-column Gaussian elimination with partial row pivoting.
  // Optionally applies COLAMD fill-reducing column ordering.
  //
  // Returns Lu_factors with:
  //   L: unit lower triangular (1s stored, diagonal last per row)
  //   U: upper triangular (diagonal first per row)
  //   row_perm: row_perm[k] = original row that ended up in position k
  //   col_perm: COLAMD column permutation (empty if not applied)
  //
  // Throws std::invalid_argument if A is not square.
  // Throws std::domain_error if A is singular (zero pivot).

  template <typename T>
  Lu_factors<T>
  sparse_lu(Compressed_row_matrix<T> const& A, bool apply_colamd = true) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("sparse_lu requires a square matrix");
    }

    auto un = static_cast<std::size_t>(n);

    // Optional COLAMD column reordering
    std::vector<config::size_type> col_perm;
    std::optional<Compressed_row_matrix<T>> B_storage;

    if (apply_colamd) {
      col_perm = column_approximate_minimum_degree(A.sparsity());
      B_storage.emplace(cperm(A, std::span<config::size_type const>{col_perm}));
    }

    auto const& B = B_storage.has_value() ? *B_storage : A;
    auto b_rp = B.row_ptr();
    auto b_ci = B.col_ind();
    auto b_vals = B.values();

    // Per-row sparse storage: each row[i] = sorted vector of (col, val)
    using col_val = std::pair<config::size_type, T>;
    std::vector<std::vector<col_val>> rows(un);

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);
      for (auto p = b_rp[i]; p < b_rp[i + 1]; ++p) {
        rows[ui].push_back({b_ci[p], b_vals[p]});
      }
    }

    // Row permutation: row_perm[k] = original row in position k
    std::vector<config::size_type> row_perm(un);
    for (config::size_type i = 0; i < n; ++i) {
      row_perm[static_cast<std::size_t>(i)] = i;
    }

    // Column-by-column Gaussian elimination with partial pivoting
    for (config::size_type k = 0; k < n; ++k) {
      auto uk = static_cast<std::size_t>(k);

      // Find pivot: row in [k..n-1] with max |value at column k|
      config::size_type pivot_row = -1;
      T pivot_val{0};

      for (config::size_type i = k; i < n; ++i) {
        auto ui = static_cast<std::size_t>(i);
        for (auto const& [col, val] : rows[ui]) {
          if (col == k) {
            if (std::abs(val) > std::abs(pivot_val)) {
              pivot_val = val;
              pivot_row = i;
            }
            break;
          }
        }
      }

      if (pivot_row < 0 || pivot_val == T{0}) {
        throw std::domain_error("sparse_lu: singular matrix (zero pivot)");
      }

      // Swap rows k and pivot_row
      if (pivot_row != k) {
        std::swap(rows[uk], rows[static_cast<std::size_t>(pivot_row)]);
        std::swap(row_perm[uk], row_perm[static_cast<std::size_t>(pivot_row)]);
      }

      // Eliminate: for each row i > k that has column k
      auto u_kk = pivot_val;

      for (config::size_type i = k + 1; i < n; ++i) {
        auto ui = static_cast<std::size_t>(i);

        // Find column k in row i
        config::size_type col_k_pos = -1;
        for (std::size_t idx = 0; idx < rows[ui].size(); ++idx) {
          if (rows[ui][idx].first == k) {
            col_k_pos = static_cast<config::size_type>(idx);
            break;
          }
        }

        if (col_k_pos < 0) { continue; }

        auto l_ik = rows[ui][static_cast<std::size_t>(col_k_pos)].second / u_kk;

        // Build col-to-index map for row i
        std::vector<config::size_type> col_to_idx(un, -1);
        for (std::size_t idx = 0; idx < rows[ui].size(); ++idx) {
          col_to_idx[static_cast<std::size_t>(rows[ui][idx].first)] =
            static_cast<config::size_type>(idx);
        }

        // Merge-subtract: row[i] -= l_ik * row[k] (for columns > k)
        for (auto const& [col, val] : rows[uk]) {
          if (col <= k) { continue; }
          auto target = col_to_idx[static_cast<std::size_t>(col)];
          if (target >= 0) {
            rows[ui][static_cast<std::size_t>(target)].second -= l_ik * val;
          } else {
            // Fill-in: insert new entry
            rows[ui].push_back({col, -l_ik * val});
          }
        }

        // Store l_ik in column k position
        rows[ui][static_cast<std::size_t>(col_k_pos)].second = l_ik;

        // Re-sort row i by column index (fill-in may have added entries)
        std::sort(
          rows[ui].begin(), rows[ui].end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
          });
      }
    }

    // Extract L and U from per-row storage
    std::vector<Index> l_indices;
    std::vector<T> l_vals;
    std::vector<Index> u_indices;
    std::vector<T> u_vals;

    auto shape = B.shape();

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);

      // L: entries with col < i (multipliers), then unit diagonal
      for (auto const& [col, val] : rows[ui]) {
        if (col < i) {
          l_indices.push_back(Index{i, col});
          l_vals.push_back(val);
        }
      }
      l_indices.push_back(Index{i, i});
      l_vals.push_back(T{1});

      // U: diagonal first (col == i), then entries with col > i
      for (auto const& [col, val] : rows[ui]) {
        if (col >= i) {
          u_indices.push_back(Index{i, col});
          u_vals.push_back(val);
        }
      }
    }

    Compressed_row_sparsity l_sp{shape, l_indices.begin(), l_indices.end()};
    Compressed_row_sparsity u_sp{shape, u_indices.begin(), u_indices.end()};

    return Lu_factors<T>{
      Compressed_row_matrix<T>{std::move(l_sp), std::move(l_vals)},
      Compressed_row_matrix<T>{std::move(u_sp), std::move(u_vals)},
      std::move(row_perm),
      std::move(col_perm)};
  }

  // Solve A*x = b using pre-computed LU factors.
  //
  // b' = P_r * b  (apply row permutation)
  // L * z = b'    (forward substitution)
  // U * y = z     (backward substitution)
  // x[col_inv[i]] = y[i]  (undo column permutation)

  template <typename T>
  std::vector<T>
  lu_solve(Lu_factors<T> const& factors, std::span<T const> b) {
    auto n = factors.L.shape().row();
    auto un = static_cast<std::size_t>(n);

    // Apply row permutation: b'[k] = b[row_perm[k]]
    std::vector<T> b_perm(un);
    for (config::size_type k = 0; k < n; ++k) {
      b_perm[static_cast<std::size_t>(k)] = b[static_cast<std::size_t>(
        factors.row_perm[static_cast<std::size_t>(k)])];
    }

    // Forward solve: L * z = b'
    auto z = forward_solve(factors.L, std::span<T const>{b_perm});

    // Backward solve: U * y = z
    auto y = backward_solve(factors.U, std::span<T const>{z});

    // Undo column permutation
    if (!factors.col_perm.empty()) {
      auto inv_p = inverse_permutation(
        std::span<config::size_type const>{factors.col_perm});
      std::vector<T> x_out(un);
      for (config::size_type i = 0; i < n; ++i) {
        x_out[static_cast<std::size_t>(inv_p[static_cast<std::size_t>(i)])] =
          y[static_cast<std::size_t>(i)];
      }
      return x_out;
    }

    return y;
  }

  // Apply LU preconditioner: z = U^{-1} L^{-1} P_r * r.
  // Iterator-based interface compatible with solver preconditioner API.

  template <typename T, typename Iter, typename OutIter>
  void
  lu_apply(Lu_factors<T> const& factors, Iter first, Iter last, OutIter out) {
    auto z = lu_solve(factors, std::span<T const>{first, last});
    std::copy(z.begin(), z.end(), out);
  }

} // end of namespace sparkit::data::detail
