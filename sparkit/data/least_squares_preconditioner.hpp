#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <iterator>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::data::detail {

  // Least-squares polynomial preconditioner data.
  //
  // Approximates A^{-1} via a polynomial p(λ) fitted by least squares
  // to minimise ‖1 - p(λ)·λ‖ at Chebyshev sample points on
  // [λ_min, λ_max]. The polynomial is evaluated using Horner's method.

  template <typename T>
  struct Least_squares_preconditioner {
    Compressed_row_matrix<T> const* A;
    std::vector<T> coeffs;
    config::size_type degree;
    mutable std::vector<T> tmp;
  };

  // Least-squares polynomial preconditioner setup.
  //
  // Samples m Chebyshev nodes on [λ_min, λ_max], builds a Vandermonde
  // system for p(λ) ≈ 1/λ, and solves the (d+1)×(d+1) normal equations
  // via dense Cholesky.

  template <typename T>
  Least_squares_preconditioner<T>
  least_squares_preconditioner(
    Compressed_row_matrix<T> const& A,
    T lambda_min,
    T lambda_max,
    config::size_type degree) {
    auto n = static_cast<std::size_t>(A.shape().row());
    auto d = static_cast<std::size_t>(degree);
    auto dp1 = d + 1;

    // Sample points: Chebyshev nodes on [λ_min, λ_max]
    auto m = std::max(dp1 * 2, std::size_t{64});
    T center = (lambda_max + lambda_min) / T{2};
    T half_width = (lambda_max - lambda_min) / T{2};

    std::vector<T> samples(m);
    std::vector<T> targets(m);
    for (std::size_t j = 0; j < m; ++j) {
      T tj = std::cos(std::numbers::pi_v<T> * (T(j) + T{0.5}) / T(m));
      samples[j] = center + half_width * tj;
      targets[j] = T{1} / samples[j];
    }

    // Build normal equations: V^T V c = V^T f
    // where V_{jk} = λ_j^k, f_j = 1/λ_j
    // G = V^T V  (dp1 × dp1)
    // rhs = V^T f (dp1)
    std::vector<T> G(dp1 * dp1, T{0});
    std::vector<T> rhs(dp1, T{0});

    for (std::size_t j = 0; j < m; ++j) {
      // Compute powers of λ_j
      std::vector<T> powers(dp1);
      powers[0] = T{1};
      for (std::size_t k = 1; k < dp1; ++k) {
        powers[k] = powers[k - 1] * samples[j];
      }

      for (std::size_t p = 0; p < dp1; ++p) {
        rhs[p] += powers[p] * targets[j];
        for (std::size_t q = 0; q < dp1; ++q) {
          G[p * dp1 + q] += powers[p] * powers[q];
        }
      }
    }

    // Solve G c = rhs via dense Cholesky (G is SPD)
    // In-place Cholesky: G = L L^T
    for (std::size_t i = 0; i < dp1; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        T sum = G[i * dp1 + j];
        for (std::size_t k = 0; k < j; ++k) {
          sum -= G[i * dp1 + k] * G[j * dp1 + k];
        }
        G[i * dp1 + j] = sum / G[j * dp1 + j];
      }
      T sum = G[i * dp1 + i];
      for (std::size_t k = 0; k < i; ++k) {
        sum -= G[i * dp1 + k] * G[i * dp1 + k];
      }
      if (sum <= T{0}) {
        throw std::invalid_argument(
          "least_squares_preconditioner: normal equations not SPD");
      }
      G[i * dp1 + i] = std::sqrt(sum);
    }

    // Forward solve: L y = rhs
    for (std::size_t i = 0; i < dp1; ++i) {
      T sum = rhs[i];
      for (std::size_t k = 0; k < i; ++k) {
        sum -= G[i * dp1 + k] * rhs[k];
      }
      rhs[i] = sum / G[i * dp1 + i];
    }

    // Backward solve: L^T c = y
    for (std::size_t i = dp1; i-- > 0;) {
      T sum = rhs[i];
      for (std::size_t k = i + 1; k < dp1; ++k) {
        sum -= G[k * dp1 + i] * rhs[k];
      }
      rhs[i] = sum / G[i * dp1 + i];
    }

    return Least_squares_preconditioner<T>{
      &A, std::move(rhs), degree, std::vector<T>(n, T{0})};
  }

  // Least-squares polynomial preconditioner apply.
  //
  // Evaluates z = p(A) r using Horner's method:
  //   z = c_d · r
  //   for k = d-1 down to 0: z = c_k · r + A · z

  template <typename T, typename Iter, typename OutIter>
  void
  least_squares_preconditioner_apply(
    Least_squares_preconditioner<T> const& prec,
    Iter first,
    Iter last,
    OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    auto d = prec.degree;

    // Copy input to vector for reuse
    std::vector<T> r(n);
    {
      auto it = first;
      for (std::size_t i = 0; i < n; ++i, ++it) {
        r[i] = *it;
      }
    }

    // z = c_d * r
    std::vector<T> z(n);
    for (std::size_t i = 0; i < n; ++i) {
      z[i] = prec.coeffs[static_cast<std::size_t>(d)] * r[i];
    }

    // Horner: z = c_k * r + A * z
    for (config::size_type k = d - 1; k >= 0; --k) {
      auto az = multiply(*prec.A, std::span<T const>{z});
      for (std::size_t i = 0; i < n; ++i) {
        z[i] = prec.coeffs[static_cast<std::size_t>(k)] * r[i] + az[i];
      }
    }

    // Copy to output
    auto z_it = out;
    for (std::size_t i = 0; i < n; ++i, ++z_it) {
      *z_it = z[i];
    }
  }

} // end of namespace sparkit::data::detail
