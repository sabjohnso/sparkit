#pragma once

//
// ... Standard header files
//
#include <iterator>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/info.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::data::detail {

  // Neumann series preconditioner data.
  //
  // Approximates A^{-1} via the truncated Neumann series:
  //   M^{-1} ≈ ω Σ_{k=0}^{d} (I - ωA)^k
  //
  // Requires ‖I - ωA‖ < 1 for convergence. A safe default is
  // ω = 1 / ‖A‖_∞.

  template <typename T>
  struct Neumann_preconditioner {
    Compressed_row_matrix<T> const* A;
    T omega;
    config::size_type degree;
    mutable std::vector<T> tmp;
  };

  // Neumann series preconditioner setup.
  //
  // Constructs a Neumann preconditioner with the given scaling parameter
  // omega and polynomial degree.

  template <typename T>
  Neumann_preconditioner<T>
  neumann_preconditioner(
    Compressed_row_matrix<T> const& A, T omega, config::size_type degree) {
    auto n = static_cast<std::size_t>(A.shape().row());
    return Neumann_preconditioner<T>{
      &A, omega, degree, std::vector<T>(n, T{0})};
  }

  // Neumann series preconditioner setup with automatic omega.
  //
  // Uses ω = 1 / ‖A‖_∞ as a safe default.

  template <typename T>
  Neumann_preconditioner<T>
  neumann_preconditioner(
    Compressed_row_matrix<T> const& A, config::size_type degree) {
    return neumann_preconditioner(A, T{1} / norm_inf(A), degree);
  }

  // Neumann series preconditioner apply.
  //
  // Computes z = ω Σ_{k=0}^{d} (I - ωA)^k r via Horner evaluation:
  //   z = r
  //   for k = 1..d: z = r + (I - ωA)z  →  tmp = A*z; z = r + z - ω*tmp
  //   z *= ω

  template <typename T, typename Iter, typename OutIter>
  void
  neumann_preconditioner_apply(
    Neumann_preconditioner<T> const& prec, Iter first, Iter last, OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));

    // Copy input r into output z
    auto r_it = first;
    auto z_it = out;
    for (std::size_t i = 0; i < n; ++i, ++r_it, ++z_it) {
      *z_it = *r_it;
    }

    // Horner iterations: z = r + (I - ωA)z
    for (config::size_type k = 1; k <= prec.degree; ++k) {
      // tmp = A * z
      std::vector<T> z_vec(n);
      z_it = out;
      for (std::size_t i = 0; i < n; ++i, ++z_it) {
        z_vec[i] = *z_it;
      }
      auto az = multiply(*prec.A, std::span<T const>{z_vec});

      // z = r + z - ω * tmp
      r_it = first;
      z_it = out;
      for (std::size_t i = 0; i < n; ++i, ++r_it, ++z_it) {
        *z_it = *r_it + z_vec[i] - prec.omega * az[i];
      }
    }

    // Final scale: z *= ω
    z_it = out;
    for (std::size_t i = 0; i < n; ++i, ++z_it) {
      *z_it *= prec.omega;
    }
  }

} // end of namespace sparkit::data::detail
