#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // Precomputed data for SSOR sweeps.

  template <typename T>
  struct Ssor_factors {
    std::vector<T> inv_diag;
    std::vector<T> diag;
    T omega;
  };

  // SSOR preconditioner setup.
  //
  // Extracts the diagonal of A, inverts each entry, and stores omega.
  //
  // Throws std::invalid_argument if any diagonal entry is zero.

  template <typename T>
  Ssor_factors<T>
  ssor(Compressed_row_matrix<T> const& A, T omega) {
    auto d = extract_diagonal(A);
    std::vector<T> inv_d(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) {
      if (d[i] == T{0}) {
        throw std::invalid_argument("ssor: zero diagonal entry");
      }
      inv_d[i] = T{1} / d[i];
    }
    return Ssor_factors<T>{std::move(inv_d), std::move(d), omega};
  }

  // SSOR preconditioner apply.
  //
  // Computes z = M_SSOR^{-1} r where
  //   M_SSOR = 1/(omega*(2-omega)) * (D + omega*L) * D^{-1} * (D + omega*U)
  //
  // Algorithm:
  //   1. Forward sweep:  solve (D + omega*L) y = r
  //   2. Diagonal scale: z_mid[i] = d[i] * y[i]
  //   3. Backward sweep: solve (D + omega*U) w = z_mid
  //   4. Final scale:    z[i] = omega*(2-omega) * w[i]
  //
  // All steps operate in-place on the output buffer.

  template <typename T, typename Iter, typename OutIter>
  void
  ssor_apply(
    Compressed_row_matrix<T> const& A,
    Ssor_factors<T> const& factors,
    Iter first,
    Iter last,
    OutIter out) {
    auto n = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();
    auto const& inv_diag = factors.inv_diag;
    auto const& diag = factors.diag;
    auto omega = factors.omega;

    // Copy r to output buffer
    std::vector<T> z(first, last);

    // Step 1: Forward sweep — solve (D + omega*L) y = r
    for (config::size_type i = 0; i < n; ++i) {
      T sum{0};
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j < i) { sum += vals[p] * z[static_cast<std::size_t>(j)]; }
      }
      z[static_cast<std::size_t>(i)] =
        (z[static_cast<std::size_t>(i)] - omega * sum) *
        inv_diag[static_cast<std::size_t>(i)];
    }

    // Step 2: Diagonal scale — z_mid[i] = d[i] * y[i]
    for (config::size_type i = 0; i < n; ++i) {
      z[static_cast<std::size_t>(i)] *= diag[static_cast<std::size_t>(i)];
    }

    // Step 3: Backward sweep — solve (D + omega*U) w = z_mid
    for (config::size_type i = n - 1; i >= 0; --i) {
      T sum{0};
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j > i) { sum += vals[p] * z[static_cast<std::size_t>(j)]; }
      }
      z[static_cast<std::size_t>(i)] =
        (z[static_cast<std::size_t>(i)] - omega * sum) *
        inv_diag[static_cast<std::size_t>(i)];
      if (i == 0) { break; }
    }

    // Step 4: Final scale — z[i] = omega*(2-omega) * w[i]
    T scale = omega * (T{2} - omega);
    for (config::size_type i = 0; i < n; ++i) {
      z[static_cast<std::size_t>(i)] *= scale;
    }

    std::copy(z.begin(), z.end(), out);
  }

} // end of namespace sparkit::data::detail
