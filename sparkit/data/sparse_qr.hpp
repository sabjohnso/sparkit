#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/reordering.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>
#include <sparkit/data/triangular_solve.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // Result of sparse QR factorization: A*P = Q*R
  //
  // R: n×n upper triangular (CSR, diagonal first in each row)
  // V: m×n Householder vectors (CSR, v_j[j]=1 stored explicitly)
  // beta: n Householder scalars, H_j = I - beta[j] * v_j * v_j^T
  // column_perm: COLAMD permutation (empty if none applied)

  template <typename T>
  struct Qr_factors {
    Compressed_row_matrix<T> R;
    Compressed_row_matrix<T> V;
    std::vector<T> beta;
    std::vector<config::size_type> column_perm;
  };

  // -- Private helpers for sparse QR --

  // Form A^T*A sparsity pattern without computing values.
  // Duplicated from reordering.cpp (static there) because we need it
  // in this header-only context.

  inline Compressed_row_sparsity
  qr_form_ata_pattern(Compressed_row_sparsity const& sp) {
    using size_type = config::size_type;
    auto nrow = sp.shape().row();
    auto ncol = sp.shape().column();
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto un = static_cast<std::size_t>(ncol);

    // Phase 1: Build column-to-row mapping (col_ptr, row_ind)
    std::vector<size_type> col_count(un, 0);
    for (size_type i = 0; i < nrow; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        ++col_count[static_cast<std::size_t>(ci[p])];
      }
    }

    std::vector<size_type> col_ptr(un + 1, 0);
    for (size_type j = 0; j < ncol; ++j) {
      col_ptr[static_cast<std::size_t>(j + 1)] =
        col_ptr[static_cast<std::size_t>(j)] +
        col_count[static_cast<std::size_t>(j)];
    }

    auto total_entries = col_ptr[un];
    std::vector<size_type> row_ind(static_cast<std::size_t>(total_entries));
    std::vector<size_type> col_pos(col_ptr.begin(), col_ptr.end() - 1);

    for (size_type i = 0; i < nrow; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        row_ind[static_cast<std::size_t>(
          col_pos[static_cast<std::size_t>(j)]++)] = i;
      }
    }

    // Phase 2: Collect A^T*A entries as Index pairs
    std::vector<Index> indices;
    std::vector<size_type> marker(un, -1);

    for (size_type j = 0; j < ncol; ++j) {
      auto uj = static_cast<std::size_t>(j);

      marker[uj] = j;
      indices.push_back(Index{j, j});

      for (auto r = col_ptr[uj]; r < col_ptr[uj + 1]; ++r) {
        auto row = row_ind[static_cast<std::size_t>(r)];
        for (auto p = rp[row]; p < rp[row + 1]; ++p) {
          auto k = ci[p];
          if (marker[static_cast<std::size_t>(k)] != j) {
            marker[static_cast<std::size_t>(k)] = j;
            indices.push_back(Index{j, k});
          }
        }
      }
    }

    return Compressed_row_sparsity{
      Shape{ncol, ncol}, indices.begin(), indices.end()};
  }

  // Transpose a sparsity pattern (no values).

  inline Compressed_row_sparsity
  qr_transpose_sparsity(Compressed_row_sparsity const& sp) {
    using size_type = config::size_type;
    auto nrow = sp.shape().row();
    auto ncol = sp.shape().column();
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sp.size()));

    for (size_type i = 0; i < nrow; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        indices.push_back(Index{ci[p], i});
      }
    }

    return Compressed_row_sparsity{
      Shape{ncol, nrow}, indices.begin(), indices.end()};
  }

  // Sparse Householder QR factorization (CSparse-style left-looking).
  //
  // For m×n matrix A (m >= n):
  // 1. Optional COLAMD column ordering for fill reduction
  // 2. Symbolic analysis via A^T*A → elimination tree → symbolic Cholesky
  // 3. Numeric phase: column-by-column Householder reflections
  //
  // Returns Qr_factors<T> with R, V, beta, and column_perm.

  template <typename T>
  Qr_factors<T>
  qr(Compressed_row_matrix<T> const& A, bool apply_colamd = true) {
    using size_type = config::size_type;
    auto m = A.shape().row();
    auto n = A.shape().column();

    if (m < n) { throw std::invalid_argument("sparse_qr: requires m >= n"); }

    // 1. Optional COLAMD column ordering
    std::vector<size_type> perm;
    Compressed_row_matrix<T> const* A_ptr = &A;
    Compressed_row_matrix<T> A_perm{A.sparsity(), std::vector<T>{}};

    if (apply_colamd) {
      perm = column_approximate_minimum_degree(A.sparsity());
      A_perm = cperm(A, std::span<size_type const>{perm});
      A_ptr = &A_perm;
    }

    auto const& Aw = *A_ptr;

    // 2. Build transpose for column access
    auto At = transpose(Aw);

    // 3. Symbolic analysis: R has the same sparsity as chol(A^T*A)^T
    auto ata = qr_form_ata_pattern(Aw.sparsity());
    auto L_pattern = symbolic_cholesky(ata);
    auto R_pattern = qr_transpose_sparsity(L_pattern);

    // 4. Pre-build column→position map for R
    //    col_map[j] = list of (row_k, offset in r_vals)
    auto r_rp = R_pattern.row_ptr();
    auto r_ci = R_pattern.col_ind();

    std::vector<std::vector<std::pair<size_type, size_type>>> col_map(
      static_cast<std::size_t>(n));
    for (size_type i = 0; i < n; ++i) {
      for (auto p = r_rp[i]; p < r_rp[i + 1]; ++p) {
        auto j = r_ci[p];
        col_map[static_cast<std::size_t>(j)].push_back({i, p});
      }
    }

    // 5. Allocate workspace
    auto r_nnz = R_pattern.size();
    std::vector<T> r_vals(static_cast<std::size_t>(r_nnz), T{0});
    std::vector<T> x(static_cast<std::size_t>(m), T{0});
    std::vector<T> beta_vec(static_cast<std::size_t>(n));

    // v_cols[j] = list of (row, value) for Householder vector j
    std::vector<std::vector<std::pair<size_type, T>>> v_cols(
      static_cast<std::size_t>(n));

    std::vector<size_type> x_nz;

    // L_pattern row j lists columns k < j where H_k affects column j,
    // plus the diagonal j. We need L's rows for the reflection loop.
    auto l_rp = L_pattern.row_ptr();
    auto l_ci = L_pattern.col_ind();

    auto at_rp = At.row_ptr();
    auto at_ci = At.col_ind();
    auto at_vals = At.values();

    // 6. Column loop
    for (size_type j = 0; j < n; ++j) {
      auto uj = static_cast<std::size_t>(j);

      // A. Scatter A's column j (= At's row j) into x
      for (auto p = at_rp[j]; p < at_rp[j + 1]; ++p) {
        auto row = at_ci[p];
        x[static_cast<std::size_t>(row)] = at_vals[p];
        x_nz.push_back(row);
      }

      // B. Apply H_k for each k < j in L row j
      //    L row j: l_ci[l_rp[j] .. l_rp[j+1]-1], diagonal is last
      for (auto lp = l_rp[j]; lp < l_rp[j + 1] - 1; ++lp) {
        auto k = l_ci[lp];
        if (k >= j) { break; }

        // dot = v_k^T * x
        auto const& vk = v_cols[static_cast<std::size_t>(k)];
        T dot{0};
        for (auto const& [row, val] : vk) {
          dot += val * x[static_cast<std::size_t>(row)];
        }

        if (dot == T{0}) { continue; }

        // x -= beta[k] * dot * v_k
        for (auto const& [row, val] : vk) {
          auto ur = static_cast<std::size_t>(row);
          if (x[ur] == T{0}) { x_nz.push_back(row); }
          x[ur] -= beta_vec[static_cast<std::size_t>(k)] * dot * val;
        }
      }

      // C. Extract R column j above diagonal: R(k,j) = x[k] for k < j
      for (auto const& [row_k, offset] : col_map[uj]) {
        if (row_k < j) {
          r_vals[static_cast<std::size_t>(offset)] =
            x[static_cast<std::size_t>(row_k)];
        }
      }

      // D. Compute Householder from x[j:m]
      T sigma_sq{0};
      for (auto idx : x_nz) {
        if (idx >= j) {
          auto val = x[static_cast<std::size_t>(idx)];
          sigma_sq += val * val;
        }
      }

      auto sigma = std::sqrt(sigma_sq);
      if (sigma == T{0}) {
        beta_vec[uj] = T{0};
        // Clear workspace
        for (auto idx : x_nz) {
          x[static_cast<std::size_t>(idx)] = T{0};
        }
        x_nz.clear();
        continue;
      }

      auto xj = x[uj];
      if (xj >= T{0}) { sigma = -sigma; }

      auto denom = xj - sigma;
      beta_vec[uj] = -denom / sigma;

      // Store R(j,j) = sigma
      for (auto const& [row_k, offset] : col_map[uj]) {
        if (row_k == j) {
          r_vals[static_cast<std::size_t>(offset)] = sigma;
          break;
        }
      }

      // Store v_j: v[j] = 1, v[i] = x[i] / denom for i > j
      v_cols[uj].push_back({j, T{1}});
      for (auto idx : x_nz) {
        if (idx > j) {
          auto val = x[static_cast<std::size_t>(idx)];
          if (val != T{0}) { v_cols[uj].push_back({idx, val / denom}); }
        }
      }

      // Sort v_cols[j] by row for consistent access
      std::sort(
        v_cols[uj].begin(), v_cols[uj].end(), [](auto const& a, auto const& b) {
          return a.first < b.first;
        });

      // E. Clear workspace
      for (auto idx : x_nz) {
        x[static_cast<std::size_t>(idx)] = T{0};
      }
      x_nz.clear();
    }

    // 7. Assemble V into Compressed_row_matrix from v_cols
    std::vector<Index> v_indices;
    std::vector<T> v_values;
    for (size_type j = 0; j < n; ++j) {
      for (auto const& [row, val] : v_cols[static_cast<std::size_t>(j)]) {
        v_indices.push_back(Index{row, j});
        v_values.push_back(val);
      }
    }

    // Sort by (row, col) for CSR construction
    std::vector<std::size_t> v_perm(v_indices.size());
    std::iota(v_perm.begin(), v_perm.end(), std::size_t{0});
    std::sort(v_perm.begin(), v_perm.end(), [&](std::size_t a, std::size_t b) {
      if (v_indices[a].row() != v_indices[b].row())
        return v_indices[a].row() < v_indices[b].row();
      return v_indices[a].column() < v_indices[b].column();
    });

    std::vector<Index> sorted_v_idx;
    std::vector<T> sorted_v_vals;
    sorted_v_idx.reserve(v_perm.size());
    sorted_v_vals.reserve(v_perm.size());
    for (auto k : v_perm) {
      sorted_v_idx.push_back(v_indices[k]);
      sorted_v_vals.push_back(v_values[k]);
    }

    Compressed_row_sparsity v_sp{
      Shape{m, n}, sorted_v_idx.begin(), sorted_v_idx.end()};
    Compressed_row_matrix<T> V{std::move(v_sp), std::move(sorted_v_vals)};

    // 8. Assemble R
    Compressed_row_matrix<T> R{R_pattern, std::move(r_vals)};

    return Qr_factors<T>{
      std::move(R), std::move(V), std::move(beta_vec), std::move(perm)};
  }

  // Apply Q^T to vector: y = Q^T * b
  // Q^T = H_{n-1} * ... * H_1 * H_0 applied in forward order:
  //   for j = 0 to n-1: y -= beta[j] * (v_j^T * y) * v_j

  template <typename T>
  std::vector<T>
  qr_apply_qt(Qr_factors<T> const& factors, std::span<T const> b) {
    auto const& V = factors.V;
    auto const& beta = factors.beta;
    auto n = V.shape().column();

    // Build CSC view of V by transposing
    auto Vt = transpose(V);
    auto vt_rp = Vt.row_ptr();
    auto vt_ci = Vt.col_ind();
    auto vt_vals = Vt.values();

    std::vector<T> y(b.begin(), b.end());

    for (config::size_type j = 0; j < n; ++j) {
      // dot = v_j^T * y (Vt row j = V column j)
      T dot{0};
      for (auto p = vt_rp[j]; p < vt_rp[j + 1]; ++p) {
        dot += vt_vals[p] * y[static_cast<std::size_t>(vt_ci[p])];
      }

      // y -= beta[j] * dot * v_j
      for (auto p = vt_rp[j]; p < vt_rp[j + 1]; ++p) {
        y[static_cast<std::size_t>(vt_ci[p])] -=
          beta[static_cast<std::size_t>(j)] * dot * vt_vals[p];
      }
    }

    return y;
  }

  // Apply Q to vector: y = Q * x
  // Q = H_0 * H_1 * ... * H_{n-1} applied in reverse order:
  //   for j = n-1 downto 0: y -= beta[j] * (v_j^T * y) * v_j

  template <typename T>
  std::vector<T>
  qr_apply_q(Qr_factors<T> const& factors, std::span<T const> x) {
    auto const& V = factors.V;
    auto const& beta = factors.beta;
    auto n = V.shape().column();

    auto Vt = transpose(V);
    auto vt_rp = Vt.row_ptr();
    auto vt_ci = Vt.col_ind();
    auto vt_vals = Vt.values();

    std::vector<T> y(x.begin(), x.end());

    for (auto jj = n; jj > 0; --jj) {
      auto j = jj - 1;

      T dot{0};
      for (auto p = vt_rp[j]; p < vt_rp[j + 1]; ++p) {
        dot += vt_vals[p] * y[static_cast<std::size_t>(vt_ci[p])];
      }

      for (auto p = vt_rp[j]; p < vt_rp[j + 1]; ++p) {
        y[static_cast<std::size_t>(vt_ci[p])] -=
          beta[static_cast<std::size_t>(j)] * dot * vt_vals[p];
      }
    }

    return y;
  }

  // Least-squares solve: min ||Ax - b||_2
  //
  // Computes x = P * R^{-1} * (Q^T * b)[0:n]
  // where A*P = Q*R from the QR factorization.

  template <typename T>
  std::vector<T>
  qr_solve(Qr_factors<T> const& factors, std::span<T const> b) {
    auto n = factors.R.shape().row();

    // c = Q^T * b
    auto c = qr_apply_qt(factors, b);

    // Truncate to first n entries
    std::vector<T> c_n(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(n));

    // y = R^{-1} * c_n
    auto y = backward_solve(factors.R, std::span<T const>{c_n});

    // Undo column permutation if applied
    if (!factors.column_perm.empty()) {
      auto inv_p = inverse_permutation(
        std::span<config::size_type const>{factors.column_perm});
      std::vector<T> x_out(static_cast<std::size_t>(n));
      for (config::size_type i = 0; i < n; ++i) {
        x_out[static_cast<std::size_t>(inv_p[static_cast<std::size_t>(i)])] =
          y[static_cast<std::size_t>(i)];
      }
      return x_out;
    }

    return y;
  }

} // end of namespace sparkit::data::detail
