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
#include <sparkit/data/amg_aggregation.hpp>
#include <sparkit/data/amg_config.hpp>
#include <sparkit/data/amg_prolongation.hpp>
#include <sparkit/data/amg_strength.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/jacobi.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/triangular_solve.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::data::detail {

  // Build the AMG hierarchy via smoothed aggregation.
  //
  // For each level: extract Jacobi preconditioner, build strength graph,
  // aggregate, construct prolongation/restriction, form Galerkin coarse
  // operator R*A*P. Coarsest level gets a direct Cholesky factorization.

  template <typename T>
  Amg_hierarchy<T>
  amg_setup(Compressed_row_matrix<T> const& A, Amg_config<T> const& cfg) {
    std::vector<Amg_level<T>> levels;
    std::vector<Amg_transfer<T>> transfers;

    auto current_A = A;

    for (config::size_type lvl = 0; lvl < cfg.max_levels; ++lvl) {
      auto inv_d = jacobi(current_A);
      auto n = current_A.shape().row();

      levels.push_back(Amg_level<T>{current_A, std::move(inv_d)});

      if (n <= cfg.coarsest_size || lvl == cfg.max_levels - 1) { break; }

      auto S = strength_of_connection(current_A, cfg.strength_threshold);
      auto [agg_ids, n_agg] = aggregate(S, n);

      auto P_tent = tentative_prolongation<T>(agg_ids, n_agg, n);

      auto P = (cfg.prolongation_smoothing_weight == T{0})
                 ? P_tent
                 : smooth_prolongation(
                     current_A, P_tent, cfg.prolongation_smoothing_weight);

      auto R = transpose(P);
      auto AP = multiply(current_A, P);
      auto coarse_A = multiply(R, AP);

      transfers.push_back(Amg_transfer<T>{std::move(P), std::move(R)});
      current_A = std::move(coarse_A);
    }

    auto coarse_factor = cholesky(levels.back().A);

    return Amg_hierarchy<T>{
      std::move(levels), std::move(transfers), std::move(coarse_factor), cfg};
  }

  // Weighted Jacobi smoothing: x += omega * D^{-1} * (rhs - A*x).

  template <typename T>
  void
  amg_smooth(
    Amg_level<T> const& level,
    T omega,
    config::size_type steps,
    std::span<T const> rhs,
    std::span<T> x) {
    auto n = static_cast<std::size_t>(level.A.shape().row());

    for (config::size_type s = 0; s < steps; ++s) {
      auto Ax = multiply(level.A, std::span<T const>{x.data(), x.size()});
      for (std::size_t i = 0; i < n; ++i) {
        x[i] += omega * level.inv_diag[i] * (rhs[i] - Ax[i]);
      }
    }
  }

  // Recursive V-cycle.
  //
  // Coarsest level: direct solve via pre-computed Cholesky factor.
  // Other levels: pre-smooth, restrict residual, recurse, prolongate
  // correction, post-smooth.

  template <typename T>
  void
  amg_vcycle(
    Amg_hierarchy<T> const& h,
    config::size_type level,
    std::span<T const> rhs,
    std::span<T> x) {
    auto last_level = static_cast<config::size_type>(h.levels.size()) - 1;

    if (level == last_level) {
      // Direct solve: x = L^{-T} L^{-1} rhs
      auto y = forward_solve(h.coarse_factor, rhs);
      auto z = forward_solve_transpose(h.coarse_factor, std::span<T const>{y});
      for (std::size_t i = 0; i < z.size(); ++i) {
        x[i] = z[i];
      }
      return;
    }

    auto const& lvl = h.levels[static_cast<std::size_t>(level)];
    auto const& tr = h.transfers[static_cast<std::size_t>(level)];
    auto n = static_cast<std::size_t>(lvl.A.shape().row());

    // Pre-smooth
    amg_smooth(
      lvl, h.config.jacobi_weight, h.config.pre_smoothing_steps, rhs, x);

    // Compute residual: r = rhs - A*x
    auto Ax = multiply(lvl.A, std::span<T const>{x.data(), x.size()});
    std::vector<T> r(n);
    for (std::size_t i = 0; i < n; ++i) {
      r[i] = rhs[i] - Ax[i];
    }

    // Restrict: r_c = R * r
    auto r_c = multiply(tr.R, std::span<T const>{r});

    // Recurse on coarse level
    auto n_c = static_cast<std::size_t>(
      h.levels[static_cast<std::size_t>(level + 1)].A.shape().row());
    std::vector<T> x_c(n_c, T{0});
    amg_vcycle(h, level + 1, std::span<T const>{r_c}, std::span<T>{x_c});

    // Prolongate correction: x += P * x_c
    auto correction = multiply(tr.P, std::span<T const>{x_c});
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += correction[i];
    }

    // Post-smooth
    amg_smooth(
      lvl, h.config.jacobi_weight, h.config.post_smoothing_steps, rhs, x);
  }

  // Apply AMG as a preconditioner: z = AMG^{-1} * r.
  // One V-cycle from zero initial guess.

  template <typename T, typename Iter, typename OutIter>
  void
  amg_apply(Amg_hierarchy<T> const& h, Iter first, Iter last, OutIter out) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    std::vector<T> rhs(first, last);
    std::vector<T> x(n, T{0});

    amg_vcycle(h, 0, std::span<T const>{rhs}, std::span<T>{x});

    for (std::size_t i = 0; i < n; ++i, ++out) {
      *out = x[i];
    }
  }

} // end of namespace sparkit::data::detail
