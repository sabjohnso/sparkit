#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/multifrontal_symbolic.hpp>

namespace sparkit::data::detail {

  template <typename T>
  struct Supernode_factor {
    std::vector<T> L_diag; // snode_size * snode_size, column-major lower tri
    std::vector<T> L_sub;  // update_size * snode_size, column-major
    config::size_type snode_size;
    config::size_type update_size;
  };

  template <typename T>
  struct Multifrontal_factor {
    Multifrontal_symbolic symbolic;
    std::vector<Supernode_factor<T>> factors;
    std::vector<config::size_type> perm;
    std::vector<config::size_type> inv_perm;
  };

  // In-place column-major dense Cholesky factorization.
  // L is p x p column-major; on exit, lower triangle holds L.
  template <typename T>
  void
  dense_cholesky(std::span<T> L, config::size_type p) {
    for (config::size_type j = 0; j < p; ++j) {
      // Diagonal: L[j,j] = sqrt(L[j,j] - sum of L[j,k]^2 for k < j)
      T diag = L[static_cast<std::size_t>(j * p + j)];
      for (config::size_type k = 0; k < j; ++k) {
        auto ljk = L[static_cast<std::size_t>(k * p + j)];
        diag -= ljk * ljk;
      }
      if (diag <= T{0}) {
        throw std::domain_error(
          "dense_cholesky: matrix is not positive definite");
      }
      auto ljj = std::sqrt(diag);
      L[static_cast<std::size_t>(j * p + j)] = ljj;

      // Off-diagonal: L[i,j] = (L[i,j] - sum L[i,k]*L[j,k]) / L[j,j]
      for (config::size_type i = j + 1; i < p; ++i) {
        T sum = L[static_cast<std::size_t>(j * p + i)];
        for (config::size_type k = 0; k < j; ++k) {
          sum -= L[static_cast<std::size_t>(k * p + i)] *
                 L[static_cast<std::size_t>(k * p + j)];
        }
        L[static_cast<std::size_t>(j * p + i)] = sum / ljj;
      }
    }
  }

  // Solve L * X^T = B^T where L is p x p lower triangular column-major,
  // B is q x p column-major (q rows, p columns). On exit, B holds X^T
  // (i.e., each row of B is solved against L).
  //
  // Equivalent to: for each row of B, solve L * x = b.
  // Column-major layout: B[col * q + row].
  template <typename T>
  void
  dense_trsm(
    std::span<T const> L,
    config::size_type p,
    std::span<T> B,
    config::size_type q) {
    // Solve L * X = B column by column of X
    // B is stored as q rows by p columns, column-major: B[j*q + i]
    // We solve: for each "column" j of the system (j=0..p-1):
    //   B[j*q + i] for all i = 0..q-1 is the i-th RHS
    //
    // Actually, we want to solve L * X^T = B^T.
    // B is q x p column-major, so B^T is p x q.
    // X^T is p x q.
    // This means: for each of q right-hand sides (columns of B^T = rows of B),
    // solve L * x = b where b is a column of B^T (= a row of B).
    //
    // A row of B in column-major: row i = B[0*q+i], B[1*q+i], ..., B[(p-1)*q+i]
    // Forward substitution for each row i:
    for (config::size_type i = 0; i < q; ++i) {
      for (config::size_type j = 0; j < p; ++j) {
        T sum = B[static_cast<std::size_t>(j * q + i)];
        for (config::size_type k = 0; k < j; ++k) {
          sum -= L[static_cast<std::size_t>(k * p + j)] *
                 B[static_cast<std::size_t>(k * q + i)];
        }
        B[static_cast<std::size_t>(j * q + i)] =
          sum / L[static_cast<std::size_t>(j * p + j)];
      }
    }
  }

  // C -= A * A^T where A is m x k column-major, C is m x m column-major.
  // Only the lower triangle of C is updated.
  template <typename T>
  void
  dense_syrk(
    std::span<T const> A,
    config::size_type m,
    config::size_type k,
    std::span<T> C) {
    for (config::size_type j = 0; j < m; ++j) {
      for (config::size_type i = j; i < m; ++i) {
        T sum{0};
        for (config::size_type l = 0; l < k; ++l) {
          sum += A[static_cast<std::size_t>(l * m + i)] *
                 A[static_cast<std::size_t>(l * m + j)];
        }
        C[static_cast<std::size_t>(j * m + i)] -= sum;
      }
    }
  }

  // Multi-frontal numeric Cholesky factorization.
  template <typename T>
  std::vector<Supernode_factor<T>>
  multifrontal_factorize(
    Compressed_row_matrix<T> const& A, Multifrontal_symbolic const& symbolic) {
    auto ns = symbolic.partition.n_supernodes;
    auto const& postorder = symbolic.tree.postorder;

    std::vector<Supernode_factor<T>> factors(static_cast<std::size_t>(ns));

    // Contribution blocks from children, indexed by supernode
    std::vector<std::vector<T>> contributions(static_cast<std::size_t>(ns));

    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();

    for (size_type step = 0; step < ns; ++step) {
      auto s = postorder[static_cast<std::size_t>(step)];
      auto const& map = symbolic.maps[static_cast<std::size_t>(s)];
      auto snode_size = map.snode_size;
      auto front_size = map.front_size;
      auto update_size = front_size - snode_size;
      auto const& row_indices = map.row_indices;

      // Allocate and zero the dense frontal matrix (column-major)
      auto f_size = static_cast<std::size_t>(front_size * front_size);
      std::vector<T> F(f_size, T{0});

      // Build reverse map: global row -> local index in this front
      // Using linear scan since front sizes are typically small
      auto local_index =
        [&](config::size_type global_row) -> config::size_type {
        for (config::size_type k = 0; k < front_size; ++k) {
          if (row_indices[static_cast<std::size_t>(k)] == global_row) {
            return k;
          }
        }
        return -1;
      };

      // Scatter A's entries into F (lower triangle only).
      // Only place entries where the column belongs to this supernode
      // [col_start, col_end). The update portion of F comes from
      // children's Schur complements, not from A directly.
      auto col_start =
        symbolic.partition.snode_start[static_cast<std::size_t>(s)];
      auto col_end =
        symbolic.partition.snode_start[static_cast<std::size_t>(s + 1)];

      for (config::size_type fi = 0; fi < front_size; ++fi) {
        auto global_i = row_indices[static_cast<std::size_t>(fi)];
        for (auto p = a_rp[global_i]; p < a_rp[global_i + 1]; ++p) {
          auto global_j = a_ci[p];
          // Only scatter if column belongs to this supernode
          if (global_j < col_start || global_j >= col_end) { continue; }
          if (global_j > global_i) { continue; } // skip upper triangle
          auto fj = local_index(global_j);
          if (fj == -1) { continue; }
          F[static_cast<std::size_t>(fj * front_size + fi)] += a_vals[p];
        }
      }

      // Extend-add: assemble each child's contribution block
      auto const& children =
        symbolic.tree.snode_children[static_cast<std::size_t>(s)];
      for (auto c : children) {
        auto const& child_contrib = contributions[static_cast<std::size_t>(c)];
        auto const& child_map = symbolic.maps[static_cast<std::size_t>(c)];
        auto child_update_size = child_map.front_size - child_map.snode_size;
        auto const& rmap = symbolic.relative_maps[static_cast<std::size_t>(c)];

        // Child contribution is child_update_size x child_update_size,
        // col-major
        for (config::size_type cj = 0; cj < child_update_size; ++cj) {
          auto pj = rmap[static_cast<std::size_t>(cj)];
          for (config::size_type ci = cj; ci < child_update_size; ++ci) {
            auto pi = rmap[static_cast<std::size_t>(ci)];
            // Ensure we add to lower triangle (pi >= pj)
            if (pi >= pj) {
              F[static_cast<std::size_t>(pj * front_size + pi)] +=
                child_contrib[static_cast<std::size_t>(
                  cj * child_update_size + ci)];
            } else {
              F[static_cast<std::size_t>(pi * front_size + pj)] +=
                child_contrib[static_cast<std::size_t>(
                  cj * child_update_size + ci)];
            }
          }
        }

        // Free child contribution
        contributions[static_cast<std::size_t>(c)].clear();
        contributions[static_cast<std::size_t>(c)].shrink_to_fit();
      }

      // Factor the diagonal block: dense_cholesky on F[0:s, 0:s]
      // Extract diagonal block into L_diag
      auto diag_size = static_cast<std::size_t>(snode_size * snode_size);
      std::vector<T> L_diag(diag_size);
      for (config::size_type j = 0; j < snode_size; ++j) {
        for (config::size_type i = j; i < snode_size; ++i) {
          L_diag[static_cast<std::size_t>(j * snode_size + i)] =
            F[static_cast<std::size_t>(j * front_size + i)];
        }
      }
      dense_cholesky<T>(L_diag, snode_size);

      // Extract sub-diagonal block and solve: L_sub = F_sub * L_diag^{-T}
      auto sub_size = static_cast<std::size_t>(update_size * snode_size);
      std::vector<T> L_sub(sub_size);
      for (config::size_type j = 0; j < snode_size; ++j) {
        for (config::size_type i = 0; i < update_size; ++i) {
          L_sub[static_cast<std::size_t>(j * update_size + i)] =
            F[static_cast<std::size_t>(j * front_size + snode_size + i)];
        }
      }
      dense_trsm<T>(
        std::span<T const>{L_diag},
        snode_size,
        std::span<T>{L_sub},
        update_size);

      // Form contribution block: C = F_22 - L_sub * L_sub^T
      if (update_size > 0) {
        auto contrib_size = static_cast<std::size_t>(update_size * update_size);
        std::vector<T> contrib(contrib_size);
        // Copy F_22 into contrib
        for (config::size_type j = 0; j < update_size; ++j) {
          for (config::size_type i = j; i < update_size; ++i) {
            contrib[static_cast<std::size_t>(j * update_size + i)] =
              F[static_cast<std::size_t>(
                (snode_size + j) * front_size + snode_size + i)];
          }
        }
        dense_syrk<T>(
          std::span<T const>{L_sub},
          update_size,
          snode_size,
          std::span<T>{contrib});

        contributions[static_cast<std::size_t>(s)] = std::move(contrib);
      }

      factors[static_cast<std::size_t>(s)] = Supernode_factor<T>{
        std::move(L_diag), std::move(L_sub), snode_size, update_size};
    }

    return factors;
  }

} // end of namespace sparkit::data::detail
