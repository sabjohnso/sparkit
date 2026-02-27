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
   * @brief Configuration for the tridiagonal eigensolver.
   */
  template <typename T>
  struct Tridiagonal_eigen_config {
    T tolerance{};
    size_type max_iterations{};
  };

  /**
   * @brief Result of the tridiagonal eigensolver.
   *
   * Eigenvalues are unsorted. Eigenvectors[k] is the eigenvector
   * corresponding to eigenvalues[k], stored as a dense vector.
   */
  template <typename T>
  struct Tridiagonal_eigen_result {
    std::vector<T> eigenvalues{};
    std::vector<std::vector<T>> eigenvectors{};
  };

  /**
   * @brief Symmetric tridiagonal eigensolver via implicit QR with
   *        Wilkinson shift.
   *
   * Computes all eigenvalues (and optionally eigenvectors) of a
   * symmetric tridiagonal matrix T given by its diagonal and
   * subdiagonal.
   *
   * Reference: Golub & Van Loan, "Matrix Computations", §8.3.
   *
   * @param diagonal       Main diagonal entries (length n).
   * @param subdiagonal    Sub/super-diagonal entries (length n-1).
   * @param cfg            Solver configuration.
   * @param compute_eigenvectors  If true, accumulate eigenvectors.
   * @return Eigenvalues and (optionally) eigenvectors.
   */
  template <typename T>
  Tridiagonal_eigen_result<T>
  tridiagonal_eigen(
    std::vector<T> diagonal,
    std::vector<T> subdiagonal,
    Tridiagonal_eigen_config<T> cfg,
    bool compute_eigenvectors) {
    auto const n = static_cast<size_type>(diagonal.size());

    if (n == 0) { return {}; }

    // Initialize eigenvector accumulator as identity.
    std::vector<std::vector<T>> Q;
    if (compute_eigenvectors) {
      Q.resize(static_cast<std::size_t>(n));
      for (size_type i = 0; i < n; ++i) {
        Q[static_cast<std::size_t>(i)].assign(
          static_cast<std::size_t>(n), T{0});
        Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = T{1};
      }
    }

    // Working copies (modified in place).
    auto& d = diagonal;
    auto& e = subdiagonal;

    // QR iteration with implicit Wilkinson shift on submatrix
    // [lo..hi].
    auto apply_givens = [&](size_type lo, size_type hi) {
      // Wilkinson shift: eigenvalue of trailing 2x2 block closest to
      // d[hi].
      auto const sz_hi = static_cast<std::size_t>(hi);
      auto const sz_him1 = static_cast<std::size_t>(hi - 1);
      T delta = (d[sz_him1] - d[sz_hi]) / T{2};
      T e_him1 = e[sz_him1];
      T shift{};
      using std::abs;
      using std::sqrt;
      if (delta == T{0}) {
        shift = d[sz_hi] - abs(e_him1);
      } else {
        T sign_d = (delta > T{0}) ? T{1} : T{-1};
        shift =
          d[sz_hi] - e_him1 * e_him1 /
                       (delta + sign_d * sqrt(delta * delta + e_him1 * e_him1));
      }

      // Implicit QR step (bulge chase).
      T x = d[static_cast<std::size_t>(lo)] - shift;
      T z = e[static_cast<std::size_t>(lo)];

      for (size_type k = lo; k < hi; ++k) {
        auto const sk = static_cast<std::size_t>(k);
        auto const sk1 = static_cast<std::size_t>(k + 1);

        // Compute Givens rotation to zero z.
        T cs{}, sn{};
        using std::sqrt;
        T r = sqrt(x * x + z * z);
        if (r == T{0}) {
          cs = T{1};
          sn = T{0};
        } else {
          cs = x / r;
          sn = z / r;
        }

        // Apply rotation to tridiagonal entries.
        if (k > lo) { e[static_cast<std::size_t>(k - 1)] = r; }

        T d_k = d[sk];
        T d_k1 = d[sk1];
        T e_k = e[sk];

        T w = cs * e_k + sn * d_k1;
        d[sk] = cs * cs * d_k + T{2} * cs * sn * e_k + sn * sn * d_k1;
        d[sk1] = sn * sn * d_k - T{2} * cs * sn * e_k + cs * cs * d_k1;
        e[sk] = w - (d[sk] - d_k) * sn / cs;

        // More numerically stable version:
        e[sk] = cs * (d_k1 - d_k) * sn + (cs * cs - sn * sn) * e_k;

        // Prepare next bulge.
        if (k + 1 < hi) {
          T e_k1 = e[sk1];
          z = sn * e_k1;
          e[sk1] = cs * e_k1;
          x = e[sk];
        }

        // Accumulate eigenvector rotations: Q_new = Q * G.
        if (compute_eigenvectors) {
          for (size_type i = 0; i < n; ++i) {
            auto si = static_cast<std::size_t>(i);
            T q_ik = Q[sk][si];
            T q_ik1 = Q[sk1][si];
            Q[sk][si] = cs * q_ik + sn * q_ik1;
            Q[sk1][si] = -sn * q_ik + cs * q_ik1;
          }
        }
      }
    };

    // Main deflation loop.
    size_type hi = n - 1;
    size_type total_iterations = 0;

    while (hi > 0 && total_iterations < cfg.max_iterations * n) {
      // Find active submatrix by checking convergence from the
      // bottom.
      using std::abs;
      T tol_hi = cfg.tolerance * (abs(d[static_cast<std::size_t>(hi - 1)]) +
                                  abs(d[static_cast<std::size_t>(hi)]));
      if (tol_hi == T{0}) { tol_hi = cfg.tolerance; }
      if (abs(e[static_cast<std::size_t>(hi - 1)]) <= tol_hi) {
        --hi;
        continue;
      }

      // Find the lowest index of the active block.
      size_type lo = hi - 1;
      while (lo > 0) {
        T tol_lo = cfg.tolerance * (abs(d[static_cast<std::size_t>(lo - 1)]) +
                                    abs(d[static_cast<std::size_t>(lo)]));
        if (tol_lo == T{0}) { tol_lo = cfg.tolerance; }
        if (abs(e[static_cast<std::size_t>(lo - 1)]) <= tol_lo) { break; }
        --lo;
      }

      apply_givens(lo, hi);
      ++total_iterations;
    }

    // Build result.
    Tridiagonal_eigen_result<T> result;
    result.eigenvalues = d;
    if (compute_eigenvectors) { result.eigenvectors = std::move(Q); }
    return result;
  }

} // end of namespace sparkit::data::detail
