#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // Tentative prolongation for smoothed aggregation AMG.
  //
  // P_tent(i, agg[i]) = 1/sqrt(|agg|) where |agg| is the size of
  // node i's aggregate. Shape: n × n_agg.

  template <typename T>
  Compressed_row_matrix<T>
  tentative_prolongation(
    std::vector<config::size_type> const& aggregates,
    config::size_type n_aggregates,
    config::size_type n) {
    // Count aggregate sizes.
    std::vector<config::size_type> agg_size(
      static_cast<std::size_t>(n_aggregates), config::size_type{0});
    for (config::size_type i = 0; i < n; ++i) {
      ++agg_size[static_cast<std::size_t>(
        aggregates[static_cast<std::size_t>(i)])];
    }

    // Build entries: one per row.
    std::vector<Index> indices;
    std::vector<T> vals;
    indices.reserve(static_cast<std::size_t>(n));
    vals.reserve(static_cast<std::size_t>(n));

    for (config::size_type i = 0; i < n; ++i) {
      auto agg = aggregates[static_cast<std::size_t>(i)];
      indices.push_back(Index{i, agg});
      vals.push_back(
        T{1} /
        std::sqrt(static_cast<T>(agg_size[static_cast<std::size_t>(agg)])));
    }

    Compressed_row_sparsity sp{
      Shape{n, n_aggregates}, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sp), std::move(vals)};
  }

  // Smoothed prolongation: P = (I - omega * D^{-1} * A) * P_tent.
  //
  // Computes S = D^{-1} * A (scaled by omega), then P = P_tent - omega * S *
  // P_tent. Delegates to extract_diagonal, multiply_left_diagonal, and SpMM.

  template <typename T>
  Compressed_row_matrix<T>
  smooth_prolongation(
    Compressed_row_matrix<T> const& A,
    Compressed_row_matrix<T> const& P_tent,
    T omega) {
    if (omega == T{0}) { return P_tent; }

    auto diag = extract_diagonal(A);

    // inv_diag = omega * D^{-1}
    std::vector<T> scaled_inv_diag(diag.size());
    for (std::size_t i = 0; i < diag.size(); ++i) {
      scaled_inv_diag[i] = omega / diag[i];
    }

    // S = omega * D^{-1} * A
    auto S = multiply_left_diagonal(A, std::span<T const>{scaled_inv_diag});

    // P = P_tent - S * P_tent  (i.e., P = (I - omega*D^{-1}*A) * P_tent)
    auto S_P = multiply(S, P_tent);
    return add(P_tent, T{-1}, S_P);
  }

} // end of namespace sparkit::data::detail
