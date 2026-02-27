#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>

namespace sparkit::data::detail {

  using size_type = sparkit::config::size_type;

  /**
   * @brief Configuration for the bidiagonal SVD solver.
   */
  template <typename T>
  struct Bidiagonal_svd_config {
    T tolerance{};
    size_type max_iterations{};
  };

  /**
   * @brief Result of the bidiagonal SVD solver.
   *
   * Singular values are non-negative. left_vectors[k] and
   * right_vectors[k] are the left and right singular vectors
   * corresponding to singular_values[k], stored as dense vectors
   * of size k (the bidiagonal dimension).
   */
  template <typename T>
  struct Bidiagonal_svd_result {
    std::vector<T> singular_values{};
    std::vector<std::vector<T>> left_vectors{};
    std::vector<std::vector<T>> right_vectors{};
  };

  /**
   * @brief Bidiagonal SVD via Golub-Kahan QR iteration.
   *
   * Computes all singular values (and optionally singular vectors)
   * of an upper bidiagonal matrix B given by its diagonal and
   * superdiagonal.
   *
   * Reference: Golub & Van Loan, "Matrix Computations", Section
   * 8.6.
   *
   * @param alpha       Main diagonal entries (length k).
   * @param beta        Superdiagonal entries (length k-1).
   * @param cfg         Solver configuration.
   * @param compute_vectors  If true, accumulate left/right singular
   *                         vectors.
   * @return Singular values and (optionally) singular vectors.
   */
  template <typename T>
  Bidiagonal_svd_result<T>
  bidiagonal_svd(
    std::vector<T> alpha,
    std::vector<T> beta,
    Bidiagonal_svd_config<T> cfg,
    bool compute_vectors) {
    auto const k = static_cast<size_type>(alpha.size());

    if (k == 0) { return {}; }

    // Initialize left/right vector accumulators as identity.
    auto sk = static_cast<std::size_t>(k);
    std::vector<std::vector<T>> U;
    std::vector<std::vector<T>> V;
    if (compute_vectors) {
      U.resize(sk);
      V.resize(sk);
      for (size_type i = 0; i < k; ++i) {
        auto si = static_cast<std::size_t>(i);
        U[si].assign(sk, T{0});
        V[si].assign(sk, T{0});
        U[si][si] = T{1};
        V[si][si] = T{1};
      }
    }

    // Working copies (modified in place).
    auto& d = alpha;
    auto& e = beta;

    // Apply a Givens rotation acting on rows/columns i and j.
    // For a right rotation on B columns (V accumulation):
    //   B[:, i] = cs * B[:, i] + sn * B[:, j]
    //   B[:, j] = -sn * B[:, i] + cs * B[:, j]
    // For a left rotation on B rows (U accumulation):
    //   B[i, :] = cs * B[i, :] + sn * B[j, :]
    //   B[j, :] = -sn * B[i, :] + cs * B[j, :]

    // Compute Givens rotation to zero out b, given [a; b].
    auto givens = [](T a, T b, T& cs, T& sn) {
      using std::sqrt;
      using std::abs;
      T r = sqrt(a * a + b * b);
      if (r == T{0}) {
        cs = T{1};
        sn = T{0};
      } else {
        cs = a / r;
        sn = b / r;
      }
      return r;
    };

    // Apply a right Givens rotation on columns p, q of the
    // bidiagonal. This modifies d and e entries and accumulates
    // into V.
    // Returns the "bulge" element created at position (q, p-1) or
    // similar.

    // Implicit QR step on bidiagonal B[lo..hi].
    // Reference: Golub & Van Loan, Algorithm 8.6.2.
    auto implicit_qr_step = [&](size_type lo, size_type hi) {
      // Wilkinson shift: eigenvalue of trailing 2x2 of T = B^T B
      // closest to t_nn.
      auto shi = static_cast<std::size_t>(hi);
      auto shim1 = static_cast<std::size_t>(hi - 1);

      // T = B^T * B, trailing 2x2:
      //   t11 = d[hi-1]^2 + e[hi-2]^2  (if hi-2 >= lo-1, else
      //   just d[hi-1]^2) t12 = d[hi-1] * e[hi-1] t22 =
      //   d[hi]^2 + e[hi-1]^2
      T d_him1_sq = d[shim1] * d[shim1];
      T d_hi_sq = d[shi] * d[shi];
      T e_him1 = e[shim1];
      T e_him1_sq = e_him1 * e_him1;

      T t11 = d_him1_sq;
      if (hi - 1 > lo) {
        T e_prev = e[static_cast<std::size_t>(hi - 2)];
        t11 += e_prev * e_prev;
      }
      T t12 = d[shim1] * e_him1;
      T t22 = d_hi_sq + e_him1_sq;

      // Eigenvalue of [[t11, t12], [t12, t22]] closest to t22.
      T delta = (t11 - t22) / T{2};
      T mu{};
      using std::abs;
      using std::sqrt;
      if (delta == T{0}) {
        mu = t22 - abs(t12);
      } else {
        T sign_d = (delta > T{0}) ? T{1} : T{-1};
        mu =
          t22 - t12 * t12 / (delta + sign_d * sqrt(delta * delta + t12 * t12));
      }

      // First column of B^T B - mu * I determines the initial
      // rotation.
      auto slo = static_cast<std::size_t>(lo);
      T y = d[slo] * d[slo] - mu;
      T z = d[slo] * e[slo];

      for (size_type j = lo; j < hi; ++j) {
        auto sj = static_cast<std::size_t>(j);
        auto sj1 = static_cast<std::size_t>(j + 1);

        // Right Givens: zero z by rotating columns j, j+1.
        T cs{}, sn{};
        givens(y, z, cs, sn);

        // Apply to B columns j, j+1.
        // Row j: (d[j], e[j]) -> rotated
        // Row j+1: (0, d[j+1]) -> picks up bulge at (j+1, j)
        if (j > lo) { e[sj - 1] = cs * y + sn * z; }

        T old_d_j = d[sj];
        T old_e_j = e[sj];
        T old_d_j1 = d[sj1];

        d[sj] = cs * old_d_j + sn * old_e_j;
        e[sj] = -sn * old_d_j + cs * old_e_j;

        // Bulge created at B(j+1, j).
        T bulge = sn * old_d_j1;
        d[sj1] = cs * old_d_j1;

        // Accumulate V: V_new = V * G_right.
        if (compute_vectors) {
          for (std::size_t i = 0; i < sk; ++i) {
            T v_j = V[sj][i];
            T v_j1 = V[sj1][i];
            V[sj][i] = cs * v_j + sn * v_j1;
            V[sj1][i] = -sn * v_j + cs * v_j1;
          }
        }

        // Left Givens: zero bulge at (j+1, j) by rotating rows j,
        // j+1.
        y = d[sj];
        z = bulge;
        givens(y, z, cs, sn);

        d[sj] = cs * y + sn * z;

        // Apply to remaining entries in rows j, j+1.
        T old_e_j2 = e[sj];
        T old_d_j1b = d[sj1];

        e[sj] = cs * old_e_j2 + sn * old_d_j1b;
        d[sj1] = -sn * old_e_j2 + cs * old_d_j1b;

        // New bulge at B(j, j+2) if j+1 < hi.
        if (j + 1 < hi) {
          T old_e_j1 = e[sj1];
          y = e[sj];
          z = sn * old_e_j1;
          e[sj1] = cs * old_e_j1;
        }

        // Accumulate U: U_new = U * G_left.
        if (compute_vectors) {
          for (std::size_t i = 0; i < sk; ++i) {
            T u_j = U[sj][i];
            T u_j1 = U[sj1][i];
            U[sj][i] = cs * u_j + sn * u_j1;
            U[sj1][i] = -sn * u_j + cs * u_j1;
          }
        }
      }
    };

    // Main deflation loop.
    size_type hi = k - 1;
    size_type total_iterations = 0;

    while (hi > 0 && total_iterations < cfg.max_iterations * k) {
      // Deflate: check if e[hi-1] is negligible.
      using std::abs;
      auto shi = static_cast<std::size_t>(hi);
      auto shim1 = static_cast<std::size_t>(hi - 1);
      T tol_hi = cfg.tolerance * (abs(d[shim1]) + abs(d[shi]));
      if (tol_hi == T{0}) { tol_hi = cfg.tolerance; }
      if (abs(e[shim1]) <= tol_hi) {
        --hi;
        continue;
      }

      // Find the lowest index of the active block.
      size_type lo = hi - 1;
      while (lo > 0) {
        auto slom1 = static_cast<std::size_t>(lo - 1);
        auto slo = static_cast<std::size_t>(lo);
        T tol_lo = cfg.tolerance * (abs(d[slom1]) + abs(d[slo]));
        if (tol_lo == T{0}) { tol_lo = cfg.tolerance; }
        if (abs(e[slom1]) <= tol_lo) { break; }
        --lo;
      }

      // Check for zero diagonal entries in the active block.
      // If d[j] ≈ 0, use left rotation to zero e[j] and deflate.
      bool deflated_zero_diag = false;
      for (size_type j = lo; j <= hi; ++j) {
        if (abs(d[static_cast<std::size_t>(j)]) <= cfg.tolerance) {
          // Zero out e[j] (or e[j-1]) using a left rotation.
          if (j < hi) {
            T cs{}, sn{};
            auto sj = static_cast<std::size_t>(j);
            givens(d[sj + 1], e[sj], cs, sn);
            d[sj + 1] = cs * d[sj + 1] + sn * e[sj];
            e[sj] = T{0};

            if (compute_vectors) {
              for (std::size_t i = 0; i < sk; ++i) {
                auto sj2 = static_cast<std::size_t>(j);
                auto sj2p1 = static_cast<std::size_t>(j + 1);
                T u_j = U[sj2][i];
                T u_j1 = U[sj2p1][i];
                U[sj2][i] = cs * u_j + sn * u_j1;
                U[sj2p1][i] = -sn * u_j + cs * u_j1;
              }
            }
            deflated_zero_diag = true;
            break;
          }
        }
      }
      if (deflated_zero_diag) { continue; }

      implicit_qr_step(lo, hi);
      ++total_iterations;
    }

    // Build result: singular values = |d[i]|, fix signs in U.
    Bidiagonal_svd_result<T> result;
    result.singular_values.resize(sk);
    for (size_type i = 0; i < k; ++i) {
      auto si = static_cast<std::size_t>(i);
      using std::abs;
      result.singular_values[si] = abs(d[si]);
      if (compute_vectors && d[si] < T{0}) {
        // Negate corresponding U column.
        for (std::size_t r = 0; r < sk; ++r) {
          U[si][r] = -U[si][r];
        }
      }
    }

    if (compute_vectors) {
      result.left_vectors = std::move(U);
      result.right_vectors = std::move(V);
    }

    return result;
  }

} // end of namespace sparkit::data::detail
