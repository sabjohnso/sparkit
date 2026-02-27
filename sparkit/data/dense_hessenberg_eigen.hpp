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
   * @brief Configuration for the Hessenberg eigensolver.
   */
  template <typename T>
  struct Hessenberg_eigen_config {
    T tolerance{};
    size_type max_iterations{};
  };

  /**
   * @brief Result of the Hessenberg eigensolver.
   *
   * Real eigenvalues have eigenvalues_imag[k] = 0. Complex conjugate
   * pairs appear as consecutive entries with opposite imaginary parts.
   */
  template <typename T>
  struct Hessenberg_eigen_result {
    std::vector<T> eigenvalues_real{};
    std::vector<T> eigenvalues_imag{};
  };

  /**
   * @brief Compute a Householder reflector for a 2- or 3-element
   *        vector [x, y, z] to zero y and z.
   *
   * Reflector: P = I - beta * v * v^T, P * [x,y,z]^T =
   * [alpha,0,0]^T.
   */
  template <typename T>
  void
  householder_3(T x, T y, T z, size_type size, T& v0, T& v1, T& v2, T& beta) {
    using std::sqrt;

    if (size == 2) {
      T norm_val = sqrt(x * x + y * y);
      if (norm_val == T{0}) {
        v0 = T{1};
        v1 = T{0};
        v2 = T{0};
        beta = T{0};
        return;
      }
      T sign = (x >= T{0}) ? T{1} : T{-1};
      v0 = x + sign * norm_val;
      v1 = y;
      v2 = T{0};
      T denom = v0 * v0 + v1 * v1;
      beta = T{2} / denom;
      return;
    }

    // size == 3
    T norm_val = sqrt(x * x + y * y + z * z);
    if (norm_val == T{0}) {
      v0 = T{1};
      v1 = T{0};
      v2 = T{0};
      beta = T{0};
      return;
    }
    T sign = (x >= T{0}) ? T{1} : T{-1};
    v0 = x + sign * norm_val;
    v1 = y;
    v2 = z;
    T denom = v0 * v0 + v1 * v1 + v2 * v2;
    beta = T{2} / denom;
  }

  /**
   * @brief Apply one Francis double-shift QR step on the Hessenberg
   *        submatrix H[lo..hi, lo..hi].
   *
   * Uses 3×1 Householder reflectors for the bulge chase.
   */
  template <typename T>
  void
  francis_double_shift(
    std::vector<T>& H, size_type n, size_type lo, size_type hi, T s, T t) {
    auto h = [&](size_type i, size_type j) -> T& {
      return H[static_cast<std::size_t>(i * n + j)];
    };

    // First column of M = H² - s*H + t*I.
    T h00 = h(lo, lo);
    T h10 = h(lo + 1, lo);
    T h01 = h(lo, lo + 1);
    T h11 = h(lo + 1, lo + 1);

    T x = h00 * h00 + h01 * h10 - s * h00 + t;
    T y = h10 * (h00 + h11 - s);
    T z = h10 * h(lo + 2, lo + 1);

    // Bulge chase with Householder reflectors.
    for (size_type k = lo; k <= hi - 2; ++k) {
      size_type r = std::min(k + 3, hi);
      size_type bulge_size = r - k + 1;

      T v0{}, v1{}, v2{};
      T beta_h{};
      householder_3(x, y, z, bulge_size, v0, v1, v2, beta_h);

      // Apply reflector from left.
      for (size_type j = (k > lo ? k - 1 : lo); j < n; ++j) {
        T dot = v0 * h(k, j);
        if (bulge_size >= 2) { dot += v1 * h(k + 1, j); }
        if (bulge_size >= 3) { dot += v2 * h(k + 2, j); }
        dot *= beta_h;
        h(k, j) -= v0 * dot;
        if (bulge_size >= 2) { h(k + 1, j) -= v1 * dot; }
        if (bulge_size >= 3) { h(k + 2, j) -= v2 * dot; }
      }

      // Apply reflector from right.
      size_type row_end = std::min(r + 1, hi) + 1;
      for (size_type i = 0; i < row_end; ++i) {
        T dot = v0 * h(i, k);
        if (bulge_size >= 2) { dot += v1 * h(i, k + 1); }
        if (bulge_size >= 3) { dot += v2 * h(i, k + 2); }
        dot *= beta_h;
        h(i, k) -= v0 * dot;
        if (bulge_size >= 2) { h(i, k + 1) -= v1 * dot; }
        if (bulge_size >= 3) { h(i, k + 2) -= v2 * dot; }
      }

      // Prepare next bulge.
      if (k + 3 <= hi) {
        x = h(k + 1, k);
        y = h(k + 2, k);
        z = (k + 3 <= hi) ? h(k + 3, k) : T{0};
      }
    }

    // Final 2×2 Givens rotation at bottom of bulge.
    {
      size_type k = hi - 1;
      T cs{}, sn{};
      T r_val = std::sqrt(
        h(k, k - 1) * h(k, k - 1) + h(k + 1, k - 1) * h(k + 1, k - 1));
      if (r_val > T{0}) {
        cs = h(k, k - 1) / r_val;
        sn = h(k + 1, k - 1) / r_val;

        // Apply from left.
        for (size_type j = k - 1; j < n; ++j) {
          T h_k = h(k, j);
          T h_k1 = h(k + 1, j);
          h(k, j) = cs * h_k + sn * h_k1;
          h(k + 1, j) = -sn * h_k + cs * h_k1;
        }

        // Apply from right.
        for (size_type i = 0; i <= std::min(k + 2, hi); ++i) {
          T h_ik = h(i, k);
          T h_ik1 = h(i, k + 1);
          h(i, k) = cs * h_ik + sn * h_ik1;
          h(i, k + 1) = -sn * h_ik + cs * h_ik1;
        }
      }
    }

    // Zero out below-Hessenberg entries (numerical cleanup).
    for (size_type i = lo + 2; i <= hi; ++i) {
      h(i, i - 2) = T{0};
      if (i > lo + 2) { h(i, i - 3) = T{0}; }
    }
  }

  /**
   * @brief Compute eigenvalues of an upper Hessenberg matrix via
   *        Francis double-shift implicit QR iteration.
   *
   * H is stored row-major in a flat vector of size n*n.
   *
   * Reference: Golub & Van Loan, "Matrix Computations", §7.5.
   */
  template <typename T>
  Hessenberg_eigen_result<T>
  hessenberg_eigen(
    std::vector<T> H, size_type n, Hessenberg_eigen_config<T> cfg) {
    Hessenberg_eigen_result<T> result;
    result.eigenvalues_real.resize(static_cast<std::size_t>(n), T{0});
    result.eigenvalues_imag.resize(static_cast<std::size_t>(n), T{0});

    if (n == 0) { return result; }
    if (n == 1) {
      result.eigenvalues_real[0] = H[0];
      return result;
    }

    auto h = [&](size_type i, size_type j) -> T& {
      return H[static_cast<std::size_t>(i * n + j)];
    };

    auto extract_2x2 = [&](size_type p) {
      auto sp = static_cast<std::size_t>(p);
      auto sp1 = static_cast<std::size_t>(p + 1);
      T a = h(p, p);
      T b = h(p, p + 1);
      T c = h(p + 1, p);
      T d = h(p + 1, p + 1);

      T trace = a + d;
      T det = a * d - b * c;
      T disc = trace * trace - T{4} * det;

      using std::sqrt;

      if (disc >= T{0}) {
        T sq = sqrt(disc);
        result.eigenvalues_real[sp] = (trace + sq) / T{2};
        result.eigenvalues_real[sp1] = (trace - sq) / T{2};
        result.eigenvalues_imag[sp] = T{0};
        result.eigenvalues_imag[sp1] = T{0};
      } else {
        T sq = sqrt(-disc);
        result.eigenvalues_real[sp] = trace / T{2};
        result.eigenvalues_real[sp1] = trace / T{2};
        result.eigenvalues_imag[sp] = sq / T{2};
        result.eigenvalues_imag[sp1] = -sq / T{2};
      }
    };

    size_type hi = n - 1;
    size_type total_iter = 0;
    size_type stall_count = 0;

    while (hi > 0) {
      if (total_iter >= cfg.max_iterations * n) { break; }

      using std::abs;

      // Check for 1×1 deflation.
      T tol_hi = cfg.tolerance * (abs(h(hi - 1, hi - 1)) + abs(h(hi, hi)));
      if (tol_hi == T{0}) { tol_hi = cfg.tolerance; }

      if (abs(h(hi, hi - 1)) <= tol_hi) {
        result.eigenvalues_real[static_cast<std::size_t>(hi)] = h(hi, hi);
        result.eigenvalues_imag[static_cast<std::size_t>(hi)] = T{0};
        --hi;
        stall_count = 0;
        continue;
      }

      if (hi == 1) {
        extract_2x2(0);
        break;
      }

      // Check for 2×2 deflation.
      T tol_him1 =
        cfg.tolerance * (abs(h(hi - 2, hi - 2)) + abs(h(hi - 1, hi - 1)));
      if (tol_him1 == T{0}) { tol_him1 = cfg.tolerance; }

      if (abs(h(hi - 1, hi - 2)) <= tol_him1) {
        extract_2x2(hi - 1);
        hi -= 2;
        stall_count = 0;
        continue;
      }

      // Find bottom of unreduced block.
      size_type lo = hi - 1;
      while (lo > 0) {
        T tol_lo = cfg.tolerance * (abs(h(lo - 1, lo - 1)) + abs(h(lo, lo)));
        if (tol_lo == T{0}) { tol_lo = cfg.tolerance; }
        if (abs(h(lo, lo - 1)) <= tol_lo) {
          h(lo, lo - 1) = T{0};
          break;
        }
        --lo;
      }

      // Francis double-shift.
      T s = h(hi - 1, hi - 1) + h(hi, hi);
      T t_val = h(hi - 1, hi - 1) * h(hi, hi) - h(hi - 1, hi) * h(hi, hi - 1);

      ++stall_count;
      if (stall_count > 10) {
        s = abs(h(hi, hi - 1)) + abs(h(hi - 1, hi - 2));
        t_val = s * s;
        stall_count = 0;
      }

      francis_double_shift(H, n, lo, hi, s, t_val);
      ++total_iter;
    }

    if (hi == 0) {
      result.eigenvalues_real[0] = h(0, 0);
      result.eigenvalues_imag[0] = T{0};
    }

    return result;
  }

} // end of namespace sparkit::data::detail
