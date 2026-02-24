#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/ilu.hpp>

namespace sparkit::data::detail {

  // ILUT configuration: dual-threshold incomplete LU.

  template <typename T>
  struct Ilut_config {
    T drop_tolerance;             // tau: |entry| < tau * ||row||_2 -> drop
    config::size_type fill_limit; // lfil: max entries per row in L and U
  };

  // ILUT: dual-threshold incomplete LU factorization (Saad).
  //
  // Produces L (lower, unit diagonal last) and U (upper, computed diagonal
  // first), compatible with forward_solve / backward_solve via ilu_apply.
  //
  // Throws std::invalid_argument if A is not square.
  // Throws std::domain_error on zero pivot.

  template <typename T>
  Ilu_factors<T>
  ilut(Compressed_row_matrix<T> const& A, Ilut_config<T> const& cfg) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("ilut requires a square matrix");
    }

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto a_vals = A.values();
    auto tau = cfg.drop_tolerance;
    auto lfil = cfg.fill_limit;

    auto un = static_cast<std::size_t>(n);

    // Per-row storage for L and U factors
    using col_val = std::pair<config::size_type, T>;
    std::vector<std::vector<col_val>> l_rows(un);
    std::vector<std::vector<col_val>> u_rows(un);
    std::vector<T> u_diag(un);

    // Dense workspace and active marker
    std::vector<T> w(un, T{0});
    std::vector<bool> active(un, false);

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);

      // A. Scatter row i into workspace, compute row norm
      T norm_sq{0};

      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        auto uj = static_cast<std::size_t>(j);
        w[uj] = a_vals[p];
        active[uj] = true;
        norm_sq += a_vals[p] * a_vals[p];
      }
      T norm_i = std::sqrt(norm_sq);

      // B. IKJ elimination: sweep k = 0..i-1, process active entries
      for (config::size_type k = 0; k < i; ++k) {
        auto uk = static_cast<std::size_t>(k);
        if (!active[uk]) { continue; }

        w[uk] /= u_diag[uk];

        // Drop small l_ik
        if (std::abs(w[uk]) < tau * norm_i) {
          w[uk] = T{0};
          active[uk] = false;
          continue;
        }

        auto l_ik = w[uk];

        // Update from U's row k (entries after diagonal)
        for (auto const& [j, u_kj] : u_rows[uk]) {
          if (j <= k) { continue; }
          auto uj = static_cast<std::size_t>(j);
          w[uj] -= l_ik * u_kj;
          active[uj] = true;
        }
      }

      // C. Collect L entries (columns < i) and apply dual-drop
      std::vector<col_val> l_cands;
      for (config::size_type j = 0; j < i; ++j) {
        auto uj = static_cast<std::size_t>(j);
        if (!active[uj]) { continue; }
        if (std::abs(w[uj]) >= tau * norm_i) { l_cands.push_back({j, w[uj]}); }
      }

      // Keep at most lfil largest by magnitude
      if (static_cast<config::size_type>(l_cands.size()) > lfil) {
        std::partial_sort(
          l_cands.begin(),
          l_cands.begin() + static_cast<std::ptrdiff_t>(lfil),
          l_cands.end(),
          [](auto const& a, auto const& b) {
            return std::abs(a.second) > std::abs(b.second);
          });
        l_cands.resize(static_cast<std::size_t>(lfil));
        std::sort(
          l_cands.begin(), l_cands.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
          });
      }

      l_rows[ui] = std::move(l_cands);

      // D. Collect U entries (columns > i) and apply dual-drop
      T diag_val = w[ui];
      if (diag_val == T{0}) { throw std::domain_error("ilut: zero pivot"); }

      std::vector<col_val> u_cands;
      for (config::size_type j = i + 1; j < n; ++j) {
        auto uj = static_cast<std::size_t>(j);
        if (!active[uj]) { continue; }
        if (std::abs(w[uj]) >= tau * norm_i) { u_cands.push_back({j, w[uj]}); }
      }

      // Keep at most lfil largest by magnitude
      if (static_cast<config::size_type>(u_cands.size()) > lfil) {
        std::partial_sort(
          u_cands.begin(),
          u_cands.begin() + static_cast<std::ptrdiff_t>(lfil),
          u_cands.end(),
          [](auto const& a, auto const& b) {
            return std::abs(a.second) > std::abs(b.second);
          });
        u_cands.resize(static_cast<std::size_t>(lfil));
        std::sort(
          u_cands.begin(), u_cands.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
          });
      }

      std::vector<col_val> u_row;
      u_row.reserve(u_cands.size() + 1);
      u_row.push_back({i, diag_val});
      for (auto& cv : u_cands) {
        u_row.push_back(cv);
      }

      u_diag[ui] = diag_val;
      u_rows[ui] = std::move(u_row);

      // E. Clean up workspace
      for (config::size_type j = 0; j < n; ++j) {
        auto uj = static_cast<std::size_t>(j);
        w[uj] = T{0};
        active[uj] = false;
      }
    }

    // Build L and U CSR matrices
    std::vector<Index> l_indices;
    std::vector<T> l_vals;
    std::vector<Index> u_indices;
    std::vector<T> u_vals;

    auto shape = A.shape();

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);

      // L: off-diagonal entries then unit diagonal
      for (auto const& [j, v] : l_rows[ui]) {
        l_indices.push_back(Index{i, j});
        l_vals.push_back(v);
      }
      l_indices.push_back(Index{i, i});
      l_vals.push_back(T{1});

      // U: diagonal first, then off-diagonal
      for (auto const& [j, v] : u_rows[ui]) {
        u_indices.push_back(Index{i, j});
        u_vals.push_back(v);
      }
    }

    Compressed_row_sparsity l_sp{shape, l_indices.begin(), l_indices.end()};
    Compressed_row_sparsity u_sp{shape, u_indices.begin(), u_indices.end()};

    return Ilu_factors<T>{
      Compressed_row_matrix<T>{std::move(l_sp), std::move(l_vals)},
      Compressed_row_matrix<T>{std::move(u_sp), std::move(u_vals)}};
  }

} // end of namespace sparkit::data::detail
