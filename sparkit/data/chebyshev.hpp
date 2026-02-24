#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/import.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Configuration for the Chebyshev iteration solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Chebyshev_config {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ \|r\|_2 / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Hard upper bound on total iterations.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each step's residual norm is appended to
     *  Chebyshev_summary::iteration_residuals.
     */
    bool collect_residuals{};

    /**
     * @brief Lower bound on eigenvalues of @f$ M^{-1}A @f$.
     */
    T lambda_min{};

    /**
     * @brief Upper bound on eigenvalues of @f$ M^{-1}A @f$.
     */
    T lambda_max{};
  };

  /**
   * @brief Convergence summary returned by the Chebyshev solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Chebyshev_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of iterations actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Chebyshev_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief Preconditioned Chebyshev iteration solver with left and right
   *  preconditioning.
   *
   * Solves a linear system @f$ Ax = b @f$ using the Chebyshev iteration
   * with optional preconditioning (Templates book, Algorithm 5.2).
   * The combined preconditioner is @f$ M = M_L M_R @f$, applied as
   * @f$ z = M_R^{-1}(M_L^{-1}(r)) @f$. Chebyshev iteration accelerates
   * Richardson iteration using polynomial coefficients derived from
   * eigenvalue bounds, requiring no inner products.
   *
   * @f{align*}{
   *   r_0 &= b - Ax_0 \\
   *   z_k &= M_R^{-1}(M_L^{-1}(r_k)) \\
   *   p_1 &= z_0, \quad \alpha_1 = 1/d \\
   *   p_k &= z_{k-1} + \beta_k p_{k-1}, \quad k \geq 2 \\
   *   x_k &= x_{k-1} + \alpha_k p_k \\
   *   r_k &= r_{k-1} - \alpha_k A p_k
   * @f}
   *
   * where @f$ d = (\lambda_{\max} + \lambda_{\min}) / 2 @f$,
   * @f$ c = (\lambda_{\max} - \lambda_{\min}) / 2 @f$,
   * @f$ \beta_k = (c \alpha_{k-1} / 2)^2 @f$, and
   * @f$ \alpha_k = 1 / (d - \beta_k) @f$.
   *
   * @note The eigenvalue bounds apply to the combined preconditioned
   *  operator @f$ M^{-1}A @f$.
   *
   * @tparam T                    Value type (e.g. double).
   * @tparam BIter                Iterator over the right-hand side @a b.
   * @tparam XIter                Iterator over the solution vector @a x.
   * @tparam LinearOperator       Callable implementing
   *                              @f$ y \leftarrow Ax @f$.
   * @tparam LeftPreconditioner   Callable implementing
   *                              @f$ z \leftarrow M_L^{-1}r @f$.
   * @tparam RightPreconditioner  Callable implementing
   *                              @f$ w \leftarrow M_R^{-1}p @f$.
   *
   * @see chebyshev  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class ChebyshevSolver {
  public:
    ChebyshevSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Chebyshev_config<T> cfg,
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
     * @brief Return a summary of the solution process.
     */
    Chebyshev_summary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_size();
      compute_spectral_parameters();
      allocate_storage();
      initialize_residual_norms();
    }

    void
    compute_size() {
      n_ = std::distance(bfirst_, blast_);
    }

    void
    compute_spectral_parameters() {
      d_ = (cfg_.lambda_max + cfg_.lambda_min) / T{2};
      c_ = (cfg_.lambda_max - cfg_.lambda_min) / T{2};
    }

    void
    allocate_storage() {
      auto un = static_cast<std::size_t>(n_);
      r_.assign(un, T{0});
      z_.assign(un, T{0});
      p_.assign(un, T{0});
      w_.assign(un, T{0});
      tmp_.assign(un, T{0});
    }

    void
    initialize_residual_norms() {
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

      compute_initial_residual();

      while (has_budget() && !summary_.converged) {
        chebyshev_step();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    compute_initial_residual() {
      auto un = static_cast<std::size_t>(n_);
      linear_operator_(xfirst_, xlast_, w_.begin());
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r_[i] = *bit - w_[i];
      }
    }

    void
    chebyshev_step() {
      apply_preconditioner();
      compute_direction();
      compute_matvec();
      update_solution();
      update_residual();
      ++summary_.computed_iterations;
      check_convergence();
    }

    void
    apply_preconditioner() {
      left_preconditioner_(r_.cbegin(), r_.cend(), tmp_.begin());
      right_preconditioner_(tmp_.cbegin(), tmp_.cend(), z_.begin());
    }

    void
    compute_direction() {
      auto un = static_cast<std::size_t>(n_);
      if (summary_.computed_iterations == 0) {
        std::copy(z_.begin(), z_.end(), p_.begin());
        alpha_ = T{1} / d_;
      } else {
        T beta = c_ * alpha_prev_ / T{2};
        beta = beta * beta;
        alpha_ = T{1} / (d_ - beta);
        for (std::size_t i = 0; i < un; ++i) {
          p_[i] = z_[i] + beta * p_[i];
        }
      }
      alpha_prev_ = alpha_;
    }

    void
    compute_matvec() {
      linear_operator_(p_.begin(), p_.end(), w_.begin());
    }

    void
    update_solution() {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += alpha_ * p_[i];
      }
    }

    void
    update_residual() {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        r_[i] -= alpha_ * w_[i];
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
    Chebyshev_config<T> cfg_;
    LinearOperator linear_operator_;
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_;
    Chebyshev_summary<T> summary_{};
    T bnorm_{};
    T d_{};
    T c_{};
    T alpha_{};
    T alpha_prev_{};
    std::vector<T> r_{};
    std::vector<T> z_{};
    std::vector<T> p_{};
    std::vector<T> w_{};
    std::vector<T> tmp_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with preconditioned Chebyshev iteration.
   *
   * Convenience wrapper that constructs a ChebyshevSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
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
   * @return Chebyshev_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  Chebyshev_summary<T>
  chebyshev(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Chebyshev_config<T> cfg,
    LinearOperator linear_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = ChebyshevSolver(
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
