#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/amg_cycle.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::data::detail {

  // Configuration for the standalone AMG iterative solver.

  template <typename T>
  struct Amg_solver_config {
    Amg_config<T> amg{};
    T tolerance{};
    config::size_type max_iterations{};
    bool collect_residuals{};
  };

  // Convergence summary returned by the AMG solver.

  template <typename T>
  struct Amg_solver_summary {
    T residual_norm{};
    config::size_type computed_iterations{};
    bool converged{};
    std::vector<T> iteration_residuals{};
  };

  // Standalone AMG iterative solver.
  //
  // Solves Ax = b by applying repeated V-cycles with convergence
  // monitoring. Each iteration applies one V-cycle to the current
  // residual and adds the correction to x.
  //
  // Takes Compressed_row_matrix directly because amg_setup requires
  // the explicit matrix.

  template <typename T, typename BIter, typename XIter>
  class AmgSolver {
  public:
    AmgSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Amg_solver_config<T> cfg,
      Compressed_row_matrix<T> const& A)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , A_{A} {
      init();
      run();
    }

    Amg_solver_summary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_size();
      build_hierarchy();
      allocate_storage();
      initialize_residual_collection();
    }

    void
    compute_size() {
      n_ = std::distance(bfirst_, blast_);
    }

    void
    build_hierarchy() {
      hierarchy_.emplace(amg_setup(A_, cfg_.amg));
    }

    void
    allocate_storage() {
      auto un = static_cast<std::size_t>(n_);
      r_.assign(un, T{0});
      delta_.assign(un, T{0});
    }

    void
    initialize_residual_collection() {
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.reserve(cfg_.max_iterations);
      }
    }

    void
    run() {
      bnorm_ = compute_norm(bfirst_, blast_);
      if (bnorm_ == T{0}) {
        std::fill(xfirst_, xlast_, T{0});
        summary_.converged = true;
        return;
      }

      compute_residual();

      while (has_budget() && !summary_.converged) {
        amg_step();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    compute_residual() {
      auto un = static_cast<std::size_t>(n_);
      std::vector<T> x_vec(xfirst_, xlast_);
      auto Ax = multiply(A_, std::span<T const>{x_vec});
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r_[i] = *bit - Ax[i];
      }
    }

    void
    amg_step() {
      apply_vcycle();
      update_solution();
      compute_residual();
      ++summary_.computed_iterations;
      check_convergence();
    }

    void
    apply_vcycle() {
      auto un = static_cast<std::size_t>(n_);
      std::fill(delta_.begin(), delta_.end(), T{0});
      amg_vcycle(
        *hierarchy_,
        config::size_type{0},
        std::span<T const>{r_.data(), un},
        std::span<T>{delta_.data(), un});
    }

    void
    update_solution() {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += delta_[i];
      }
    }

    void
    check_convergence() {
      T r_norm = compute_norm(r_.begin(), r_.end());
      summary_.residual_norm = r_norm;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(r_norm);
      }
      if (r_norm / bnorm_ < cfg_.tolerance) { summary_.converged = true; }
    }

    template <typename Iter>
    static T
    compute_norm(Iter first, Iter last) {
      using std::sqrt;
      return sqrt(std::inner_product(first, last, first, T{0}));
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    Amg_solver_config<T> cfg_;
    Compressed_row_matrix<T> const& A_;
    config::size_type n_;
    Amg_solver_summary<T> summary_{};
    T bnorm_{};
    std::optional<Amg_hierarchy<T>> hierarchy_;
    std::vector<T> r_{};
    std::vector<T> delta_{};
  };

  // Solve Ax = b with standalone AMG iteration.
  //
  // Convenience wrapper that constructs an AmgSolver, runs it to
  // completion, and returns the convergence summary. The solution is
  // written in-place into [xfirst, xlast).

  template <typename BIter, typename XIter, typename T>
  Amg_solver_summary<T>
  amg_solve(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Amg_solver_config<T> cfg,
    Compressed_row_matrix<T> const& A) {
    auto solver =
      AmgSolver<T, BIter, XIter>(bfirst, blast, xfirst, xlast, cfg, A);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
