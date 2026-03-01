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
#include <sparkit/data/symbolic_cholesky.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::data::detail {

  // LDL^T factorization result.
  //
  // L is unit lower triangular (explicit 1s on diagonal).
  // D is block diagonal with 1x1 and 2x2 blocks.
  //   - D_diag[k] stores D(k,k) for all k.
  //   - D_subdiag[k] stores D(k+1,k), nonzero only at 2x2 pivot starts.
  //   - pivot_size[k] = 1 for a 1x1 pivot, 2 for the start of a 2x2
  //     block, 0 for the continuation column of a 2x2 block.

  template <typename T>
  struct Ldl_factors {
    Compressed_row_matrix<T> L;
    std::vector<T> D_diag;
    std::vector<T> D_subdiag;
    std::vector<config::size_type> pivot_size;
  };

  // Solve D*z = y where D is block diagonal from LDL^T factors.
  // 1x1 block: z[k] = y[k] / D(k,k).
  // 2x2 block: Cramer's rule on [[d11,d21],[d21,d22]] * [z_k,z_{k+1}]^T.

  template <typename T>
  std::vector<T>
  block_diagonal_solve(Ldl_factors<T> const& factors, std::span<T const> y) {
    auto n = static_cast<config::size_type>(factors.D_diag.size());
    std::vector<T> z(y.begin(), y.end());

    config::size_type k = 0;
    while (k < n) {
      auto uk = static_cast<std::size_t>(k);
      if (factors.pivot_size[uk] == 1) {
        z[uk] = y[uk] / factors.D_diag[uk];
        k += 1;
      } else {
        // 2x2 block at (k, k+1)
        auto uk1 = uk + 1;
        auto d11 = factors.D_diag[uk];
        auto d21 = factors.D_subdiag[uk];
        auto d22 = factors.D_diag[uk1];
        auto det = d11 * d22 - d21 * d21;
        z[uk] = (d22 * y[uk] - d21 * y[uk1]) / det;
        z[uk1] = (d11 * y[uk1] - d21 * y[uk]) / det;
        k += 2;
      }
    }

    return z;
  }

  // Compute the working column for column k of L.
  //
  // Scatters A's lower-triangle entries for column k into w[], then
  // applies updates from previously factored columns j < k using
  // column lists built from L_pattern.

  template <typename T>
  void
  compute_working_column(
    config::size_type k,
    Compressed_row_matrix<T> const& A,
    std::vector<
      std::vector<std::pair<config::size_type, config::size_type>>> const&
      a_col_list,
    std::span<config::size_type const> l_rp,
    std::span<config::size_type const> l_ci,
    std::span<T const> l_vals,
    std::vector<
      std::vector<std::pair<config::size_type, config::size_type>>> const&
      col_list,
    std::vector<T> const& D_diag,
    std::vector<T> const& D_subdiag,
    std::vector<config::size_type> const& pivot_size,
    std::vector<T>& w) {
    auto uk = static_cast<std::size_t>(k);

    // Step A: Scatter A(i, k) for i >= k from A's lower triangle.
    // A is stored by row (CSR), so A(i,k) for i >= k means scanning
    // rows i >= k that have column k. Use a_col_list for this.
    auto a_vals = A.values();
    w[uk] = A(k, k); // diagonal
    for (auto const& [row_i, pos] : a_col_list[uk]) {
      if (row_i >= k) { w[static_cast<std::size_t>(row_i)] = a_vals[pos]; }
    }

    // Step B: Update from previously factored columns j < k.
    // For each off-diagonal j < k in L's row k:
    for (auto p = l_rp[k]; p < l_rp[k + 1] - 1; ++p) {
      auto j = l_ci[p];
      if (j >= k) { break; }
      auto uj = static_cast<std::size_t>(j);

      if (pivot_size[uj] == 1) {
        // 1x1 pivot at j: t = L(k,j) * D(j,j)
        auto l_kj = l_vals[p];
        auto t = l_kj * D_diag[uj];

        // Update w[k] -= L(k,j) * t
        w[uk] -= l_kj * t;

        // Update w[row_i] -= L(row_i, j) * t for row_i > k
        for (auto const& [row_i, pos_ij] : col_list[uj]) {
          if (row_i > k) {
            w[static_cast<std::size_t>(row_i)] -= l_vals[pos_ij] * t;
          }
        }
      } else if (pivot_size[uj] == 2) {
        // 2x2 pivot starting at j, using columns j and j+1.
        auto uj1 = uj + 1;
        auto j1 = j + 1;

        // Find L(k, j) and L(k, j+1) in L's row k.
        auto l_kj = l_vals[p];
        T l_kj1{0};
        // Search for column j+1 in row k
        for (auto q = l_rp[k]; q < l_rp[k + 1] - 1; ++q) {
          if (l_ci[q] == j1) {
            l_kj1 = l_vals[q];
            break;
          }
        }

        // D * [L(k,j), L(k,j+1)]^T
        auto dv0 = D_diag[uj] * l_kj + D_subdiag[uj] * l_kj1;
        auto dv1 = D_subdiag[uj] * l_kj + D_diag[uj1] * l_kj1;

        // Update w[k]
        w[uk] -= l_kj * dv0 + l_kj1 * dv1;

        // Update w[row_i] for row_i > k from both columns j and j+1
        for (auto const& [row_i, pos_ij] : col_list[uj]) {
          if (row_i > k) {
            auto uri = static_cast<std::size_t>(row_i);
            // Find L(row_i, j+1)
            T l_ij1{0};
            for (auto q = l_rp[row_i]; q < l_rp[row_i + 1] - 1; ++q) {
              if (l_ci[q] == j1) {
                l_ij1 = l_vals[q];
                break;
              }
            }
            w[uri] -= l_vals[pos_ij] * dv0 + l_ij1 * dv1;
          }
        }
      }
      // pivot_size == 0: continuation column of 2x2 block, skip
    }
  }

  // Clear working column entries for column k using L_pattern.

  template <typename T>
  void
  clear_working_column(
    config::size_type k,
    std::span<config::size_type const> l_rp,
    std::span<config::size_type const> l_ci,
    std::vector<
      std::vector<std::pair<config::size_type, config::size_type>>> const&
      col_list,
    std::vector<T>& w) {
    auto uk = static_cast<std::size_t>(k);
    w[uk] = T{0};
    for (auto const& [row_i, pos] : col_list[uk]) {
      w[static_cast<std::size_t>(row_i)] = T{0};
    }
    // Also clear any entries from L's row k (for A scatter)
    // Not needed since we only scatter into positions >= k which are
    // covered by col_list and the diagonal.
    (void)l_rp;
    (void)l_ci;
  }

  // Numeric LDL^T factorization with a pre-computed sparsity pattern.
  //
  // Left-looking column-by-column with bounded Bunch-Kaufman pivoting.
  // 2x2 pivots are restricted to adjacent columns (k, k+1), preserving
  // the symbolic sparsity pattern from symbolic_cholesky.
  //
  // A must be symmetric and contain at least the lower triangle.
  // L_pattern must be the result of symbolic_cholesky on A's sparsity.
  //
  // L stores explicit 1s on the diagonal so that forward_solve and
  // forward_solve_transpose work directly.
  //
  // Throws std::invalid_argument for non-square or mismatched pattern.
  // Throws std::domain_error for singular pivot (both 1x1 and 2x2 det zero).

  template <typename T>
  Ldl_factors<T>
  numeric_ldlt(
    Compressed_row_matrix<T> const& A,
    Compressed_row_sparsity const& L_pattern) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("numeric_ldlt requires a square matrix");
    }

    if (L_pattern.shape().row() != n || L_pattern.shape().column() != n) {
      throw std::invalid_argument(
        "numeric_ldlt: pattern dimensions do not match matrix");
    }

    auto un = static_cast<std::size_t>(n);

    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();

    auto l_rp = L_pattern.row_ptr();
    auto l_ci = L_pattern.col_ind();

    auto nnz = L_pattern.size();
    std::vector<T> l_vals(static_cast<std::size_t>(nnz), T{0});

    std::vector<T> D_diag(un, T{0});
    std::vector<T> D_subdiag(un, T{0});
    std::vector<config::size_type> piv_size(un, 0);

    // Build column lists from L_pattern: col_list[k] = [(row_i, position)]
    // for rows i > k that have column k as an off-diagonal entry.
    using row_pos = std::pair<config::size_type, config::size_type>;
    std::vector<std::vector<row_pos>> col_list(un);

    for (config::size_type i = 0; i < n; ++i) {
      for (auto p = l_rp[i]; p < l_rp[i + 1] - 1; ++p) {
        auto k = l_ci[p];
        col_list[static_cast<std::size_t>(k)].push_back({i, p});
      }
    }

    // Build A column lists for efficient scatter of A's lower triangle.
    // a_col_list[k] = [(row_i, position)] for off-diagonal A(row_i, k)
    // where row_i > k.
    std::vector<std::vector<row_pos>> a_col_list(un);

    for (config::size_type i = 0; i < n; ++i) {
      for (auto p = a_rp[i]; p < a_rp[i + 1]; ++p) {
        auto col = a_ci[p];
        if (col < i) {
          a_col_list[static_cast<std::size_t>(col)].push_back({i, p});
        }
      }
    }

    // Bunch-Kaufman threshold: alpha = (1 + sqrt(17)) / 8
    auto const alpha = (T{1} + std::sqrt(T{17})) / T{8};

    // Dense workspace for column under construction
    std::vector<T> w(un, T{0});
    std::vector<T> w2(un, T{0}); // second workspace for 2x2 pivots

    config::size_type k = 0;
    while (k < n) {
      auto uk = static_cast<std::size_t>(k);

      // Compute working column k
      compute_working_column(
        k,
        A,
        a_col_list,
        l_rp,
        l_ci,
        std::span<T const>{l_vals},
        col_list,
        D_diag,
        D_subdiag,
        piv_size,
        w);

      // Find lambda = max |w[i]| for i > k (off-diagonal magnitude)
      T lambda{0};
      for (auto const& [row_i, pos] : col_list[uk]) {
        auto val = std::abs(w[static_cast<std::size_t>(row_i)]);
        if (val > lambda) { lambda = val; }
      }

      auto abs_wk = std::abs(w[uk]);

      // Pivot decision
      if (abs_wk >= alpha * lambda || k == n - 1) {
        // 1x1 pivot
        if (abs_wk == T{0} && lambda == T{0}) {
          throw std::domain_error(
            "numeric_ldlt: zero pivot encountered (singular matrix)");
        }

        D_diag[uk] = w[uk];
        piv_size[uk] = 1;

        // Store L column k: L(k,k) = 1, L(i,k) = w[i] / D(k,k)
        // Diagonal is last entry in L's row k
        l_vals[static_cast<std::size_t>(l_rp[k + 1] - 1)] = T{1};

        for (auto const& [row_i, pos] : col_list[uk]) {
          l_vals[static_cast<std::size_t>(pos)] =
            w[static_cast<std::size_t>(row_i)] / D_diag[uk];
        }

        // Clear workspace
        clear_working_column(k, l_rp, l_ci, col_list, w);

        k += 1;
      } else {
        // 2x2 pivot at (k, k+1)
        auto k1 = k + 1;
        auto uk1 = static_cast<std::size_t>(k1);

        // Compute working column k+1
        compute_working_column(
          k1,
          A,
          a_col_list,
          l_rp,
          l_ci,
          std::span<T const>{l_vals},
          col_list,
          D_diag,
          D_subdiag,
          piv_size,
          w2);

        // D block: [[w[k], w2[k]], [w2[k], w2[k+1]]]
        // But w[k+1] is the off-diagonal from column k's perspective,
        // and w2[k] is the same entry from column k+1's perspective.
        // Use w[k+1] as D_subdiag (from column k's working column).
        auto d11 = w[uk];
        auto d21 = w[uk1]; // = w2[uk] by symmetry
        auto d22 = w2[uk1];
        auto det = d11 * d22 - d21 * d21;

        if (std::abs(det) == T{0}) {
          throw std::domain_error(
            "numeric_ldlt: singular 2x2 pivot encountered");
        }

        D_diag[uk] = d11;
        D_subdiag[uk] = d21;
        D_diag[uk1] = d22;
        piv_size[uk] = 2;
        piv_size[uk1] = 0;

        // L(k,k) = 1 (diagonal of row k)
        l_vals[static_cast<std::size_t>(l_rp[k + 1] - 1)] = T{1};
        // L(k+1,k+1) = 1 (diagonal of row k+1)
        l_vals[static_cast<std::size_t>(l_rp[k1 + 1] - 1)] = T{1};
        // L(k+1,k) = 0 (coupling is in D, not L)
        for (auto p = l_rp[k1]; p < l_rp[k1 + 1] - 1; ++p) {
          if (l_ci[p] == k) {
            l_vals[static_cast<std::size_t>(p)] = T{0};
            break;
          }
        }

        // L(i,k) and L(i,k+1) for i > k+1:
        // [L(i,k), L(i,k+1)] = D^{-1} * [w[i], w2[i]]
        // D^{-1} = (1/det) * [[d22, -d21], [-d21, d11]]
        for (auto const& [row_i, pos_ik] : col_list[uk]) {
          if (row_i <= k1) { continue; }
          auto uri = static_cast<std::size_t>(row_i);
          auto wi = w[uri];
          auto w2i = w2[uri];
          l_vals[static_cast<std::size_t>(pos_ik)] =
            (d22 * wi - d21 * w2i) / det;
        }

        for (auto const& [row_i, pos_ik1] : col_list[uk1]) {
          if (row_i <= k1) { continue; }
          auto uri = static_cast<std::size_t>(row_i);
          auto wi = w[uri];
          auto w2i = w2[uri];
          l_vals[static_cast<std::size_t>(pos_ik1)] =
            (d11 * w2i - d21 * wi) / det;
        }

        // Clear workspaces
        clear_working_column(k, l_rp, l_ci, col_list, w);
        clear_working_column(k1, l_rp, l_ci, col_list, w2);

        k += 2;
      }
    }

    return Ldl_factors<T>{
      Compressed_row_matrix<T>{L_pattern, std::move(l_vals)},
      std::move(D_diag),
      std::move(D_subdiag),
      std::move(piv_size)};
  }

  // Convenience: combined symbolic + numeric LDL^T.

  template <typename T>
  Ldl_factors<T>
  ldlt(Compressed_row_matrix<T> const& A) {
    auto L_pattern = symbolic_cholesky(A.sparsity());
    return numeric_ldlt(A, L_pattern);
  }

  // Solve A*x = b given pre-computed LDL^T factors.
  // y = L^{-1} b, z = D^{-1} y, x = L^{-T} z.

  template <typename T>
  std::vector<T>
  ldlt_solve(Ldl_factors<T> const& factors, std::span<T const> b) {
    auto y = forward_solve(factors.L, b);
    auto z = block_diagonal_solve(factors, std::span<T const>{y});
    return forward_solve_transpose(factors.L, std::span<T const>{z});
  }

  // Preconditioner: z = L^{-T} D^{-1} L^{-1} r.
  // Iterator-based interface compatible with solver preconditioner API.

  template <typename T, typename Iter, typename OutIter>
  void
  ldlt_apply(
    Ldl_factors<T> const& factors, Iter first, Iter last, OutIter out) {
    auto z = ldlt_solve(factors, std::span<T const>{first, last});
    std::copy(z.begin(), z.end(), out);
  }

} // end of namespace sparkit::data::detail
