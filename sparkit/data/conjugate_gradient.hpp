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

  /**
   * @brief Convergence summary returned by conjugate gradient solvers.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct CGSummary {
    /** Final residual 2-norm, @f$ \|r\|_2 @f$. */
    T residual_norm{};

    /** Number of iterations actually performed. */
    size_type computed_iterations{};

    /** True when @f$ \|r\|_2 / \|b\|_2 < @f$ tolerance. */
    bool converged{};

    /** Per-iteration residual norms (populated when
     *  CGConfig::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief Configuration for conjugate gradient solvers.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct CGConfig {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ \|r\|_2 / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Reorthogonalisation interval
     *
     * @details Every this many iterations the
     *  true residual @f$ b - Ax @f$ is recomputed from scratch to
     *  prevent drift from accumulated floating-point error.
     */
    size_type restart_iterations{};

    /**
     * @brief Hard upper bound on the number of iterations.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each iteration's residual norm is appended to
     *  CGSummary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Unpreconditioned conjugate gradient solver.
   *
   * Solves the symmetric positive-definite system @f$ Ax = b @f$ using
   * the standard CG algorithm (Hestenes & Stiefel, 1952).  The
   * solution is written in-place into the caller's buffer
   * @c [xfirst, xlast), and a CGSummary is returned with convergence
   * diagnostics.
   *
   * The linear operator uses an output-iterator signature so that
   * matrix-free operators are supported without allocation:
   * @code
   *   linear_operator(first, last, output_first)  // y = A * x
   * @endcode
   *
   * Periodic reorthogonalisation (controlled by
   * CGConfig::restart_iterations) recomputes the true residual
   * @f$ r = b - Ax @f$ to guard against floating-point drift.
   *
   * When @f$ \|b\|_2 = 0 @f$ the solver returns immediately with
   * @f$ x = 0 @f$ and @c converged = true.
   *
   * @tparam T               Value type (e.g. double).
   * @tparam BIter           Iterator over the right-hand side @a b.
   * @tparam XIter           Iterator over the solution vector @a x.
   * @tparam LinearOperator  Callable with signature
   *                         @c (BIter, BIter, XIter) implementing
   *                         @f$ y \leftarrow Ax @f$.
   *
   * @see conjugate_gradient  Free-function convenience wrapper.
   * @see PreconditionedConjugateGradientSolver  Preconditioned variant.
   */
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

    /**
     * @brief Return a summary of the solution process
     */
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

  /**
   * @brief Solve @f$ Ax = b @f$ with the conjugate gradient method.
   *
   * Convenience wrapper that constructs a ConjugateGradientSolver,
   * runs it to completion, and returns the convergence summary.
   * The solution is written in-place into @c [xfirst, xlast).
   *
   * @param bfirst           Start of right-hand side @a b.
   * @param blast            Past-the-end of @a b.
   * @param xfirst           Start of initial guess / solution @a x.
   * @param xlast            Past-the-end of @a x.
   * @param cfg              Solver configuration.
   * @param linear_operator  Output-iterator callable implementing
   *                         @f$ y \leftarrow Ax @f$.
   * @return CGSummary with convergence diagnostics.
   */
  template <typename BIter, typename XIter, typename T, typename LinearOperator>
  CGSummary<T>
  conjugate_gradient(BIter bfirst, BIter blast, XIter xfirst, XIter xlast,
                     CGConfig<T> cfg, LinearOperator linear_operator) {
    auto solver = ConjugateGradientSolver(bfirst, blast, xfirst, xlast, cfg,
                                          linear_operator);
    return solver.summary();
  }

  /**
   * @brief Preconditioned conjugate gradient solver.
   *
   * Solves the symmetric positive-definite system @f$ Ax = b @f$ using
   * the preconditioned CG algorithm.  A preconditioner
   * @f$ M^{-1} \approx A^{-1} @f$ clusters eigenvalues and accelerates
   * convergence relative to unpreconditioned CG.
   *
   * Both the linear operator and the preconditioner use an
   * output-iterator signature:
   * @code
   *   linear_operator(first, last, output_first)   // y = A * x
   *   preconditioner(first, last, output_first)     // z = M^{-1} * r
   * @endcode
   *
   * Internally the solver tracks the inner product
   * @f$ \langle r, z \rangle @f$ (where @f$ z = M^{-1}r @f$) for
   * step-size and direction updates, while convergence is still
   * checked on the unpreconditioned residual norm
   * @f$ \|r\|_2 / \|b\|_2 @f$.
   *
   * @tparam T               Value type (e.g. double).
   * @tparam BIter           Iterator over the right-hand side @a b.
   * @tparam XIter           Iterator over the solution vector @a x.
   * @tparam LinearOperator  Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam Preconditioner  Callable implementing
   *                         @f$ z \leftarrow M^{-1}r @f$.
   *
   * @see preconditioned_conjugate_gradient  Free-function convenience
   *      wrapper.
   * @see ConjugateGradientSolver  Unpreconditioned variant.
   */
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

    /**
     * @brief Return a summary of the solution process
     */
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

  /**
   * @brief Solve @f$ Ax = b @f$ with the preconditioned conjugate
   *        gradient method.
   *
   * Convenience wrapper that constructs a
   * PreconditionedConjugateGradientSolver, runs it to completion,
   * and returns the convergence summary.  The solution is written
   * in-place into @c [xfirst, xlast).
   *
   * @param bfirst           Start of right-hand side @a b.
   * @param blast            Past-the-end of @a b.
   * @param xfirst           Start of initial guess / solution @a x.
   * @param xlast            Past-the-end of @a x.
   * @param cfg              Solver configuration.
   * @param linear_operator  Output-iterator callable implementing
   *                         @f$ y \leftarrow Ax @f$.
   * @param preconditioner   Output-iterator callable implementing
   *                         @f$ z \leftarrow M^{-1}r @f$.
   * @return CGSummary with convergence diagnostics.
   */
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
