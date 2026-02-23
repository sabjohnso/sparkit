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
#include <sparkit/data/import.hpp>

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
   * @brief Dual-preconditioned conjugate gradient solver.
   *
   * Solves the symmetric positive-definite system @f$ Ax = b @f$ using
   * a conjugate gradient algorithm with optional left and right
   * preconditioners @f$ M_L @f$ and @f$ M_R @f$.
   *
   * The unified algorithm is:
   * @f{align*}{
   *   r_0 &= b - A x_0 \\
   *   z_0 &= M_L^{-1} r_0 \\
   *   p_0 &= z_0 \\
   *   \hat{p}_k &= M_R^{-1} p_k \\
   *   q_k &= A \hat{p}_k \\
   *   \alpha_k &= \langle r_k, z_k \rangle / \langle p_k, q_k \rangle \\
   *   x_{k+1} &= x_k + \alpha_k \hat{p}_k \\
   *   r_{k+1} &= r_k - \alpha_k q_k \\
   *   z_{k+1} &= M_L^{-1} r_{k+1} \\
   *   \beta_k &= \langle r_{k+1}, z_{k+1} \rangle
   *            / \langle r_k, z_k \rangle \\
   *   p_{k+1} &= z_{k+1} + \beta_k p_k
   * @f}
   *
   * Setting @f$ M_L = I @f$ and @f$ M_R = I @f$ recovers unpreconditioned
   * CG. Setting only @f$ M_L @f$ gives left PCG. Setting only @f$ M_R @f$
   * gives right PCG.
   *
   * Both the linear operator and preconditioners use an output-iterator
   * signature:
   * @code
   *   linear_operator(first, last, output_first)       // y = A * x
   *   left_preconditioner(first, last, output_first)    // z = M_L^{-1} * r
   *   right_preconditioner(first, last, output_first)   // w = M_R^{-1} * p
   * @endcode
   *
   * @tparam T                    Value type (e.g. double).
   * @tparam BIter                Iterator over the right-hand side @a b.
   * @tparam XIter                Iterator over the solution vector @a x.
   * @tparam LinearOperator       Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam LeftPreconditioner   Callable implementing
   *                              @f$ z \leftarrow M_L^{-1}r @f$.
   * @tparam RightPreconditioner  Callable implementing
   *                              @f$ w \leftarrow M_R^{-1}p @f$.
   *
   * @see conjugate_gradient  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class ConjugateGradientSolver {
  public:
    ConjugateGradientSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      CGConfig<T> cfg,
      LinearOperator linear_operator,
      LeftPreconditioner left_preconditioner,
      RightPreconditioner right_preconditioner)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , left_preconditioner_{left_preconditioner}
        , right_preconditioner_{right_preconditioner} {
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
      initialize_vector(hatp_);
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
      std::transform(
        bfirst_,
        blast_,
        std::begin(ax_),
        std::begin(residual_),
        std::minus<T>{});
      left_preconditioner_(
        std::cbegin(residual_), std::cend(residual_), std::begin(z_));
      rz_ = std::inner_product(
        std::cbegin(residual_), std::cend(residual_), std::begin(z_), T{0});
      std::copy(std::cbegin(z_), std::cend(z_), std::begin(p_));
    }

    auto
    compute_step_size() {
      right_preconditioner_(std::cbegin(p_), std::cend(p_), std::begin(hatp_));
      linear_operator_(std::begin(hatp_), std::end(hatp_), std::begin(q_));
      T pq =
        std::inner_product(std::begin(p_), std::end(p_), std::begin(q_), T{0});
      return rz_ / pq;
    }

    void
    compute_new_solution(auto alpha) {
      std::transform(
        std::begin(hatp_),
        std::end(hatp_),
        xfirst_,
        xfirst_,
        [&](const auto hi, const auto xi) { return alpha * hi + xi; });
    }

    void
    compute_new_residual(auto alpha) {
      std::transform(
        std::begin(q_),
        std::end(q_),
        std::begin(residual_),
        std::begin(residual_),
        [&](const auto qi, const auto ri) { return ri - alpha * qi; });
    }

    bool
    is_converged() {
      using std::sqrt;
      auto residual_norm = sqrt(
        std::inner_product(
          std::begin(residual_),
          std::end(residual_),
          std::begin(residual_),
          T{0}));
      summary_.residual_norm = residual_norm;
      summary_.converged = residual_norm / bnorm_ < cfg_.tolerance;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_norm);
      }
      return summary_.converged;
    }

    void
    update_direction() {
      left_preconditioner_(
        std::cbegin(residual_), std::cend(residual_), std::begin(z_));
      auto rz_new = std::inner_product(
        std::begin(residual_), std::end(residual_), std::begin(z_), T{0});
      const auto beta = rz_new / rz_;
      std::transform(
        std::begin(z_),
        std::end(z_),
        std::begin(p_),
        std::begin(p_),
        [&](const auto zi, const auto pi) { return zi + beta * pi; });
      rz_ = rz_new;
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    CGConfig<T> cfg_;
    LinearOperator linear_operator_;
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_;
    CGSummary<T> summary_{};
    T bnorm_{};
    T rz_{};
    std::vector<T> ax_{};
    std::vector<T> residual_{};
    std::vector<T> z_{};
    std::vector<T> hatp_{};
    std::vector<T> p_{};
    std::vector<T> q_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the dual-preconditioned conjugate
   *        gradient method.
   *
   * Convenience wrapper that constructs a ConjugateGradientSolver,
   * runs it to completion, and returns the convergence summary.
   * The solution is written in-place into @c [xfirst, xlast).
   *
   * Pass identity callables (copying input to output) for unused
   * preconditioner slots to recover unpreconditioned, left-only, or
   * right-only CG.
   *
   * @param bfirst               Start of right-hand side @a b.
   * @param blast                Past-the-end of @a b.
   * @param xfirst               Start of initial guess / solution @a x.
   * @param xlast                Past-the-end of @a x.
   * @param cfg                  Solver configuration.
   * @param linear_operator      Output-iterator callable implementing
   *                             @f$ y \leftarrow Ax @f$.
   * @param left_preconditioner  Output-iterator callable implementing
   *                             @f$ z \leftarrow M_L^{-1}r @f$.
   * @param right_preconditioner Output-iterator callable implementing
   *                             @f$ w \leftarrow M_R^{-1}p @f$.
   * @return CGSummary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  CGSummary<T>
  conjugate_gradient(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    CGConfig<T> cfg,
    LinearOperator linear_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = ConjugateGradientSolver(
      bfirst,
      blast,
      xfirst,
      xlast,
      cfg,
      linear_operator,
      left_preconditioner,
      right_preconditioner);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
