#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <iterator>
#include <numbers>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::data::detail {

  // Chebyshev polynomial preconditioner data.
  //
  // Approximates A^{-1} via Chebyshev polynomial approximation of
  // f(λ) = 1/λ on the interval [λ_min, λ_max]. The polynomial is
  // evaluated using the Clenshaw recurrence with matrix argument.

  template <typename T>
  struct Chebyshev_preconditioner {
    Compressed_row_matrix<T> const* A;
    std::vector<T> coeffs;
    T lambda_min;
    T lambda_max;
    config::size_type degree;
    mutable std::vector<T> b_prev;
    mutable std::vector<T> b_curr;
  };

  // Chebyshev polynomial preconditioner setup.
  //
  // Computes Chebyshev coefficients for f(λ) = 1/λ on [λ_min, λ_max]
  // using discrete cosine transform sampling, then stores them for
  // Clenshaw evaluation.

  template <typename T>
  Chebyshev_preconditioner<T>
  chebyshev_preconditioner(
    Compressed_row_matrix<T> const& A,
    T lambda_min,
    T lambda_max,
    config::size_type degree) {
    auto n = static_cast<std::size_t>(A.shape().row());

    // Compute Chebyshev coefficients for f(λ) = 1/λ on [λ_min, λ_max]
    auto d = static_cast<std::size_t>(degree);
    auto N = std::max(d + 1, std::size_t{64});

    T center = (lambda_max + lambda_min) / T{2};
    T half_width = (lambda_max - lambda_min) / T{2};

    std::vector<T> coeffs(d + 1, T{0});
    for (std::size_t k = 0; k <= d; ++k) {
      T ck = T{0};
      for (std::size_t j = 0; j < N; ++j) {
        T tj = std::cos(std::numbers::pi_v<T> * (T(j) + T{0.5}) / T(N));
        T lj = center + half_width * tj;
        T fj = T{1} / lj;
        T cos_k =
          std::cos(std::numbers::pi_v<T> * T(k) * (T(j) + T{0.5}) / T(N));
        ck += fj * cos_k;
      }
      coeffs[k] = T{2} * ck / T(N);
    }
    coeffs[0] *= T{0.5};

    return Chebyshev_preconditioner<T>{
      &A,
      std::move(coeffs),
      lambda_min,
      lambda_max,
      degree,
      std::vector<T>(n, T{0}),
      std::vector<T>(n, T{0})};
  }

  // Chebyshev polynomial preconditioner apply.
  //
  // Evaluates z = p(A) r using the Clenshaw recurrence, where
  // p is the Chebyshev polynomial approximation of 1/λ.
  //
  // Ā = (2A - (λ_max + λ_min)I) / (λ_max - λ_min)
  // b_{d+1} = 0
  // b_d = c_d · r
  // For k = d-1 down to 1: b_k = c_k · r + 2 · Ā · b_{k+1} - b_{k+2}
  // z = c_0 · r + Ā · b_1 - b_2

  template <typename T, typename Iter, typename OutIter>
  void
  chebyshev_preconditioner_apply(
    Chebyshev_preconditioner<T> const& prec,
    Iter first,
    Iter last,
    OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    auto d = prec.degree;

    T sum = prec.lambda_max + prec.lambda_min;
    T diff = prec.lambda_max - prec.lambda_min;

    // Helper: apply Ā to vector v, store in result
    // Ā·v = (2·A·v - sum·v) / diff
    auto apply_Abar = [&](std::vector<T> const& v, std::vector<T>& result) {
      auto av = multiply(*prec.A, std::span<T const>{v});
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = (T{2} * av[i] - sum * v[i]) / diff;
      }
    };

    // Copy input to a vector for reuse
    std::vector<T> r(n);
    {
      auto it = first;
      for (std::size_t i = 0; i < n; ++i, ++it) {
        r[i] = *it;
      }
    }

    if (d == 0 || diff == T{0}) {
      // z = c_0 * r (higher coefficients are zero when diff == 0)
      auto z_it = out;
      for (std::size_t i = 0; i < n; ++i, ++z_it) {
        *z_it = prec.coeffs[0] * r[i];
      }
      return;
    }

    // b_prev = b_{k+2}, b_curr = b_{k+1}
    auto& b_prev = prec.b_prev;
    auto& b_curr = prec.b_curr;

    // b_{d+1} = 0
    std::fill(b_prev.begin(), b_prev.end(), T{0});

    // b_d = c_d * r
    for (std::size_t i = 0; i < n; ++i) {
      b_curr[i] = prec.coeffs[static_cast<std::size_t>(d)] * r[i];
    }

    // Clenshaw: for k = d-1 down to 1
    std::vector<T> abar_b(n);
    for (config::size_type k = d - 1; k >= 1; --k) {
      apply_Abar(b_curr, abar_b);
      for (std::size_t i = 0; i < n; ++i) {
        T b_new = prec.coeffs[static_cast<std::size_t>(k)] * r[i] +
                  T{2} * abar_b[i] - b_prev[i];
        b_prev[i] = b_curr[i];
        b_curr[i] = b_new;
      }
    }

    // z = c_0 * r + Ā * b_1 - b_2
    // At this point b_curr = b_1, b_prev = b_2
    apply_Abar(b_curr, abar_b);
    auto z_it = out;
    for (std::size_t i = 0; i < n; ++i, ++z_it) {
      *z_it = prec.coeffs[0] * r[i] + abar_b[i] - b_prev[i];
    }
  }

} // end of namespace sparkit::data::detail
