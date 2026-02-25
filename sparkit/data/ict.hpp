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

namespace sparkit::data::detail {

  // ICT configuration: dual-threshold incomplete Cholesky.

  template <typename T>
  struct Ict_config {
    T drop_tolerance;             // tau: |entry| < tau * ||row||_2 -> drop
    config::size_type fill_limit; // lfil: max off-diagonal entries per row
  };

  // ICT: dual-threshold incomplete Cholesky factorization.
  //
  // Symmetric analog of ILUT (cf. Saad, Chapter 10). Row-by-row
  // up-looking factorization with dynamic fill discovery and dual
  // dropping (tolerance + fill limit).
  //
  // For each row i, scatters the lower-triangle entries of A into a
  // dense workspace, then eliminates using previously computed rows
  // of L (IKJ approach with column lists). New fill positions are
  // activated during elimination. After elimination, off-diagonal
  // entries are filtered by tolerance and fill limit.
  //
  // Returns Compressed_row_matrix<T> (lower triangular L with
  // diagonal last in each row), compatible with ic_apply.
  //
  // Throws std::invalid_argument if A is not square.
  // Throws std::domain_error if A is not sufficiently positive definite.

  template <typename T>
  Compressed_row_matrix<T>
  ict(Compressed_row_matrix<T> const& A, Ict_config<T> const& cfg) {
    auto n = A.shape().row();

    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument("ict requires a square matrix");
    }

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto a_vals = A.values();
    auto tau = cfg.drop_tolerance;
    auto lfil = cfg.fill_limit;

    auto un = static_cast<std::size_t>(n);

    // Per-row storage for L factor
    using col_val = std::pair<config::size_type, T>;
    std::vector<std::vector<col_val>> l_rows(un);
    std::vector<T> l_diag(un);

    // Dense workspace and active marker
    std::vector<T> w(un, T{0});
    std::vector<bool> active(un, false);

    // Column lists: col_list[k] = [(row, index_in_l_rows[row])] for
    // rows j > k that have column k in L. Built incrementally as
    // each row is finalized.
    using row_idx = std::pair<config::size_type, config::size_type>;
    std::vector<std::vector<row_idx>> col_list(un);

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);

      // A. Scatter lower-triangle entries of A's row i into w[]
      T norm_sq{0};

      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j <= i) {
          auto uj = static_cast<std::size_t>(j);
          w[uj] = a_vals[p];
          active[uj] = true;
          norm_sq += a_vals[p] * a_vals[p];
        }
      }
      T norm_i = std::sqrt(norm_sq);

      // B. IKJ elimination: for each active k < i, divide by L(k,k)
      //    and propagate updates from L's column k.
      for (config::size_type k = 0; k < i; ++k) {
        auto uk = static_cast<std::size_t>(k);
        if (!active[uk]) { continue; }

        w[uk] /= l_diag[uk];

        // Drop small L(i,k)
        if (std::abs(w[uk]) < tau * norm_i) {
          w[uk] = T{0};
          active[uk] = false;
          continue;
        }

        auto l_ik = w[uk];

        // Update from L's column k: for each j in col_list[k],
        // subtract l_ik * L(j,k) from w[j] (activate j if new fill,
        // only if j < i since L is lower triangular).
        for (auto const& [j, idx_jk] : col_list[uk]) {
          if (j >= i) { break; }
          auto uj = static_cast<std::size_t>(j);
          auto l_jk = l_rows[static_cast<std::size_t>(j)]
                            [static_cast<std::size_t>(idx_jk)]
                              .second;
          w[uj] -= l_ik * l_jk;
          active[uj] = true;
        }

        // Also update the diagonal w[i]: w[i] -= l_ik * l_ik
        // (from L(i,k) contribution to diagonal)
        // Wait -- this is handled below in the diagonal computation.
      }

      // C. Collect off-diagonal entries (columns < i) that survive
      //    tolerance drop, then keep at most lfil largest.
      std::vector<col_val> off_diag;
      for (config::size_type j = 0; j < i; ++j) {
        auto uj = static_cast<std::size_t>(j);
        if (!active[uj]) { continue; }
        if (std::abs(w[uj]) >= tau * norm_i) { off_diag.push_back({j, w[uj]}); }
      }

      // Keep at most lfil largest by magnitude
      if (static_cast<config::size_type>(off_diag.size()) > lfil) {
        std::partial_sort(
          off_diag.begin(),
          off_diag.begin() + static_cast<std::ptrdiff_t>(lfil),
          off_diag.end(),
          [](auto const& a, auto const& b) {
            return std::abs(a.second) > std::abs(b.second);
          });
        off_diag.resize(static_cast<std::size_t>(lfil));
        std::sort(
          off_diag.begin(), off_diag.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
          });
      }

      l_rows[ui] = off_diag;

      // D. Diagonal: subtract off-diagonal squared terms
      T sq_sum{0};
      for (auto const& [j, v] : off_diag) {
        sq_sum += v * v;
      }

      auto diag_val = w[ui] - sq_sum;
      if (diag_val <= T{0}) {
        throw std::domain_error("ict: matrix is not positive definite");
      }
      l_diag[ui] = std::sqrt(diag_val);

      // E. Update column lists for the entries we kept
      for (config::size_type idx = 0;
           idx < static_cast<config::size_type>(off_diag.size());
           ++idx) {
        auto col = off_diag[static_cast<std::size_t>(idx)].first;
        col_list[static_cast<std::size_t>(col)].push_back({i, idx});
      }

      // F. Clean up workspace
      for (config::size_type j = 0; j < n; ++j) {
        auto uj = static_cast<std::size_t>(j);
        w[uj] = T{0};
        active[uj] = false;
      }
    }

    // Build L CSR: for each row i, off-diagonal entries sorted by
    // column, then diagonal (last entry in row).
    std::vector<Index> l_indices;
    std::vector<T> l_vals;

    for (config::size_type i = 0; i < n; ++i) {
      auto ui = static_cast<std::size_t>(i);

      for (auto const& [j, v] : l_rows[ui]) {
        l_indices.push_back(Index{i, j});
        l_vals.push_back(v);
      }
      l_indices.push_back(Index{i, i});
      l_vals.push_back(l_diag[ui]);
    }

    auto shape = A.shape();
    Compressed_row_sparsity l_sp{shape, l_indices.begin(), l_indices.end()};

    return Compressed_row_matrix<T>{std::move(l_sp), std::move(l_vals)};
  }

} // end of namespace sparkit::data::detail
