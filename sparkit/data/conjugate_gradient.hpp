#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>

namespace sparkit::data::detail {

  template <typename T>
  struct CGSummary {
    T residual_norm{};
    size_type computed_iterations{};
    bool converged{};
    std::vector<T> iteration_residuals{};
  };

  template <typename T>
  struct CGConfig {
    T tolerance{};
    size_type restart_iterations{};
    size_type max_iterations{};
    bool collect_residuals{};
  };

  // Unpreconditioned conjugate gradient solver.
  //
  // Solves A*x = b where A is symmetric positive definite.
  // The linear operator has output-iterator signature:
  //   linear_operator(first, last, output_first)
  // The solution x is modified in-place via [xfirst, xlast).
  // Periodic reorthogonalization recomputes the true residual
  // every restart_iterations steps.

  template <typename T, typename BIter, typename XIter, typename LinearOperator>
  class ConjugateGradientSolver {
  public:
    ConjugateGradientSolver(BIter bfirst, BIter blast, XIter xfirst,
                            XIter xlast, CGConfig<T> cfg,
                            LinearOperator linear_operator)
        : bfirst_{bfirst}, blast_{blast}, xfirst_{xfirst}, xlast_{xlast},
          cfg_{cfg}, linear_operator_{linear_operator} {
      init();
      run();
    }

    CGSummary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_size();
      initialize_vector(ax_);
      initialize_vector(residual_);
      initialize_vector(p_);
      initialize_vector(q_);
      initialize_residual_norms();
    }

    void
    compute_size() {
      n_ = std::distance(bfirst_, blast_);
    }

    void
    initialize_vector(auto& v) {
      v.resize(n_);
      std::fill(std::begin(v), std::end(v), T{0});
    }

    void
    initialize_residual_norms() {
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.reserve(cfg_.max_iterations);
      }
    }

    void
    run() {
      using std::sqrt;
      bnorm_ = sqrt(std::inner_product(bfirst_, blast_, bfirst_, T{0}));
      if (bnorm_ == T{0}) {
        std::fill(xfirst_, xlast_, T{0});
        summary_.converged = true;
        return;
      }
      for (size_type k = 0; k < cfg_.max_iterations; ++k) {
        step(k);
        ++summary_.computed_iterations;
        if (is_converged()) break;
        update_direction();
      }
    }

    void
    step(size_type k) {
      if (k % cfg_.restart_iterations == 0) { reorthogonalize(); }
      auto alpha = compute_step_size();
      compute_new_solution(alpha);
      compute_new_residual(alpha);
    }

    void
    reorthogonalize() {
      linear_operator_(xfirst_, xlast_, std::begin(ax_));
      std::transform(bfirst_, blast_, std::begin(ax_), std::begin(residual_),
                     std::minus<T>{});
      rdotr_ = std::inner_product(std::cbegin(residual_), std::cend(residual_),
                                  std::begin(residual_), T{0});
      std::copy(std::cbegin(residual_), std::cend(residual_), std::begin(p_));
    }

    auto
    compute_step_size() {
      linear_operator_(std::begin(p_), std::end(p_), std::begin(q_));
      T pq = std::inner_product(std::begin(p_), std::end(p_), std::begin(q_),
                                T{0});
      T alpha = rdotr_ / pq;
      return alpha;
    }

    void
    compute_new_solution(auto alpha) {
      std::transform(
          std::begin(p_), std::end(p_), xfirst_, xfirst_,
          [&](const auto pi, const auto xi) { return alpha * pi + xi; });
    }

    void
    compute_new_residual(auto alpha) {
      std::transform(std::begin(q_), std::end(q_), std::begin(residual_),
                     std::begin(residual_), [&](const auto qi, const auto ri) {
                       return ri - alpha * qi;
                     });
    }

    bool
    is_converged() {
      using std::sqrt;
      auto residual_norm =
          sqrt(std::inner_product(std::begin(residual_), std::end(residual_),
                                  std::begin(residual_), T{0}));
      summary_.residual_norm = residual_norm;
      summary_.converged = residual_norm / bnorm_ < cfg_.tolerance;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_norm);
      }
      return summary_.converged;
    }

    void
    update_direction() {
      auto rdotr_new =
          std::inner_product(std::begin(residual_), std::end(residual_),
                             std::begin(residual_), T{0});
      const auto beta = rdotr_new / rdotr_;
      std::transform(std::begin(residual_), std::end(residual_), std::begin(p_),
                     std::begin(p_), [&](const auto ri, const auto pi) {
                       return ri + beta * pi;
                     });
      rdotr_ = rdotr_new;
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    CGConfig<T> cfg_;
    LinearOperator linear_operator_;
    size_type n_;
    CGSummary<T> summary_{};
    T bnorm_{};
    T rdotr_{};
    std::vector<T> ax_{};
    std::vector<T> residual_{};
    std::vector<T> p_{};
    std::vector<T> q_{};
  };

  template <typename BIter, typename XIter, typename T, typename LinearOperator>
  CGSummary<T>
  conjugate_gradient(BIter bfirst, BIter blast, XIter xfirst, XIter xlast,
                     CGConfig<T> cfg, LinearOperator linear_operator) {
    auto solver = ConjugateGradientSolver(bfirst, blast, xfirst, xlast, cfg,
                                          linear_operator);
    return solver.summary();
  }

  // Preconditioned conjugate gradient solver.
  //
  // Solves A*x = b where A is symmetric positive definite, using
  // a preconditioner M^{-1} to accelerate convergence.
  // Both the linear operator and preconditioner use output-iterator
  // signature: callable(first, last, output_first).
  // Convergence is checked on the unpreconditioned residual norm.

  template <typename T, typename BIter, typename XIter, typename LinearOperator,
            typename Preconditioner>
  class PreconditionedConjugateGradientSolver {
  public:
    PreconditionedConjugateGradientSolver(BIter bfirst, BIter blast,
                                          XIter xfirst, XIter xlast,
                                          CGConfig<T> cfg,
                                          LinearOperator linear_operator,
                                          Preconditioner preconditioner)
        : bfirst_{bfirst}, blast_{blast}, xfirst_{xfirst}, xlast_{xlast},
          cfg_{cfg}, linear_operator_{linear_operator},
          preconditioner_{preconditioner} {
      init();
      run();
    }

    CGSummary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_size();
      initialize_vector(ax_);
      initialize_vector(residual_);
      initialize_vector(z_);
      initialize_vector(p_);
      initialize_vector(q_);
      initialize_residual_norms();
    }

    void
    compute_size() {
      n_ = std::distance(bfirst_, blast_);
    }

    void
    initialize_vector(auto& v) {
      v.resize(n_);
      std::fill(std::begin(v), std::end(v), T{0});
    }

    void
    initialize_residual_norms() {
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.reserve(cfg_.max_iterations);
      }
    }

    void
    run() {
      using std::sqrt;
      bnorm_ = sqrt(std::inner_product(bfirst_, blast_, bfirst_, T{0}));
      if (bnorm_ == T{0}) {
        std::fill(xfirst_, xlast_, T{0});
        summary_.converged = true;
        return;
      }
      for (size_type k = 0; k < cfg_.max_iterations; ++k) {
        step(k);
        ++summary_.computed_iterations;
        if (is_converged()) break;
        update_direction();
      }
    }

    void
    step(size_type k) {
      if (k % cfg_.restart_iterations == 0) { reorthogonalize(); }
      auto alpha = compute_step_size();
      compute_new_solution(alpha);
      compute_new_residual(alpha);
    }

    void
    reorthogonalize() {
      linear_operator_(xfirst_, xlast_, std::begin(ax_));
      std::transform(bfirst_, blast_, std::begin(ax_), std::begin(residual_),
                     std::minus<T>{});
      preconditioner_(std::cbegin(residual_), std::cend(residual_),
                      std::begin(z_));
      rz_ = std::inner_product(std::cbegin(residual_), std::cend(residual_),
                               std::begin(z_), T{0});
      std::copy(std::cbegin(z_), std::cend(z_), std::begin(p_));
    }

    auto
    compute_step_size() {
      linear_operator_(std::begin(p_), std::end(p_), std::begin(q_));
      T pq = std::inner_product(std::begin(p_), std::end(p_), std::begin(q_),
                                T{0});
      return rz_ / pq;
    }

    void
    compute_new_solution(auto alpha) {
      std::transform(
          std::begin(p_), std::end(p_), xfirst_, xfirst_,
          [&](const auto pi, const auto xi) { return alpha * pi + xi; });
    }

    void
    compute_new_residual(auto alpha) {
      std::transform(std::begin(q_), std::end(q_), std::begin(residual_),
                     std::begin(residual_), [&](const auto qi, const auto ri) {
                       return ri - alpha * qi;
                     });
    }

    bool
    is_converged() {
      using std::sqrt;
      auto residual_norm =
          sqrt(std::inner_product(std::begin(residual_), std::end(residual_),
                                  std::begin(residual_), T{0}));
      summary_.residual_norm = residual_norm;
      summary_.converged = residual_norm / bnorm_ < cfg_.tolerance;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_norm);
      }
      return summary_.converged;
    }

    void
    update_direction() {
      preconditioner_(std::cbegin(residual_), std::cend(residual_),
                      std::begin(z_));
      auto rz_new = std::inner_product(
          std::begin(residual_), std::end(residual_), std::begin(z_), T{0});
      const auto beta = rz_new / rz_;
      std::transform(
          std::begin(z_), std::end(z_), std::begin(p_), std::begin(p_),
          [&](const auto zi, const auto pi) { return zi + beta * pi; });
      rz_ = rz_new;
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    CGConfig<T> cfg_;
    LinearOperator linear_operator_;
    Preconditioner preconditioner_;
    size_type n_;
    CGSummary<T> summary_{};
    T bnorm_{};
    T rz_{};
    std::vector<T> ax_{};
    std::vector<T> residual_{};
    std::vector<T> z_{};
    std::vector<T> p_{};
    std::vector<T> q_{};
  };

  template <typename BIter, typename XIter, typename T, typename LinearOperator,
            typename Preconditioner>
  CGSummary<T>
  preconditioned_conjugate_gradient(BIter bfirst, BIter blast, XIter xfirst,
                                    XIter xlast, CGConfig<T> cfg,
                                    LinearOperator linear_operator,
                                    Preconditioner preconditioner) {
    auto solver = PreconditionedConjugateGradientSolver(
        bfirst, blast, xfirst, xlast, cfg, linear_operator, preconditioner);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
