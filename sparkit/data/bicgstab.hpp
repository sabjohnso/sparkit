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

namespace sparkit::data::detail {

  /**
   * @brief Configuration for BiCGSTAB solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Bicgstab_config {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ \|r\|_2 / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Hard upper bound on total matrix-vector products.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each step's residual norm is appended to
     *  Bicgstab_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the BiCGSTAB solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Bicgstab_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Bicgstab_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief BiCGSTAB solver with left and right preconditioning.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the Biconjugate Gradient Stabilized method (van der Vorst, 1992).
   * The algorithm follows the Templates book (Algorithm 7.7).
   *
   * BiCGSTAB does not require the transpose of A, making it suitable for
   * problems where A^T is unavailable or expensive.
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
   * @see bicgstab  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class BicgstabSolver {
  public:
    BicgstabSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Bicgstab_config<T> cfg,
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
    Bicgstab_summary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_size();
      allocate_storage();
      initialize_residual_norms();
    }

    void
    compute_size() {
      n_ = std::distance(bfirst_, blast_);
    }

    void
    allocate_storage() {
      auto un = static_cast<std::size_t>(n_);
      r_.assign(un, T{0});
      r_hat_.assign(un, T{0});
      p_.assign(un, T{0});
      v_.assign(un, T{0});
      s_.assign(un, T{0});
      t_.assign(un, T{0});
      p_hat_.assign(un, T{0});
      s_hat_.assign(un, T{0});
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
      initialize_shadow_residual();

      rho_ = T{1};
      alpha_ = T{1};
      omega_ = T{1};
      std::fill(v_.begin(), v_.end(), T{0});
      std::fill(p_.begin(), p_.end(), T{0});

      while (has_budget() && !summary_.converged) {
        bicgstab_step();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    compute_initial_residual() {
      auto un = static_cast<std::size_t>(n_);
      linear_operator_(xfirst_, xlast_, tmp_.begin());
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r_[i] = *bit - tmp_[i];
      }
    }

    void
    initialize_shadow_residual() {
      std::copy(r_.begin(), r_.end(), r_hat_.begin());
    }

    void
    bicgstab_step() {
      T rho_new = dot(r_hat_, r_);
      if (rho_new == T{0}) return;

      T beta = (rho_new / rho_) * (alpha_ / omega_);
      update_search_direction(beta);
      apply_preconditioner_to_p();
      compute_v();

      alpha_ = rho_new / dot(r_hat_, v_);
      compute_s();

      if (check_half_step_convergence()) {
        update_solution_half_step();
        return;
      }

      apply_preconditioner_to_s();
      compute_t();

      omega_ = compute_omega();
      if (omega_ == T{0}) return;

      update_solution_full_step();
      update_residual();
      rho_ = rho_new;

      check_full_step_convergence();
    }

    void
    update_search_direction(T beta) {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        p_[i] = r_[i] + beta * (p_[i] - omega_ * v_[i]);
      }
    }

    void
    apply_preconditioner_to_p() {
      left_preconditioner_(p_.cbegin(), p_.cend(), tmp_.begin());
      right_preconditioner_(tmp_.cbegin(), tmp_.cend(), p_hat_.begin());
    }

    void
    compute_v() {
      linear_operator_(p_hat_.begin(), p_hat_.end(), v_.begin());
      summary_.computed_iterations += 1;
    }

    void
    compute_s() {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        s_[i] = r_[i] - alpha_ * v_[i];
      }
    }

    bool
    check_half_step_convergence() {
      T s_norm = compute_norm(s_.begin(), s_.end());
      summary_.residual_norm = s_norm;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(s_norm);
      }
      return s_norm / bnorm_ < cfg_.tolerance;
    }

    void
    update_solution_half_step() {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += alpha_ * p_hat_[i];
      }
      summary_.converged = true;
    }

    void
    apply_preconditioner_to_s() {
      left_preconditioner_(s_.cbegin(), s_.cend(), tmp_.begin());
      right_preconditioner_(tmp_.cbegin(), tmp_.cend(), s_hat_.begin());
    }

    void
    compute_t() {
      linear_operator_(s_hat_.begin(), s_hat_.end(), t_.begin());
      summary_.computed_iterations += 1;
    }

    T
    compute_omega() const {
      T tt = dot(t_, t_);
      if (tt == T{0}) return T{0};
      return dot(t_, s_) / tt;
    }

    void
    update_solution_full_step() {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += alpha_ * p_hat_[i] + omega_ * s_hat_[i];
      }
    }

    void
    update_residual() {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        r_[i] = s_[i] - omega_ * t_[i];
      }
    }

    void
    check_full_step_convergence() {
      T r_norm = compute_norm(r_.begin(), r_.end());
      summary_.residual_norm = r_norm;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(r_norm);
      }
      if (r_norm / bnorm_ < cfg_.tolerance) { summary_.converged = true; }
    }

    T
    dot(std::vector<T> const& a, std::vector<T> const& b) const {
      return std::inner_product(a.begin(), a.end(), b.begin(), T{0});
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
    Bicgstab_config<T> cfg_;
    LinearOperator linear_operator_;
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_;
    Bicgstab_summary<T> summary_{};
    T bnorm_{};
    T rho_{};
    T alpha_{};
    T omega_{};
    std::vector<T> r_{};
    std::vector<T> r_hat_{};
    std::vector<T> p_{};
    std::vector<T> v_{};
    std::vector<T> s_{};
    std::vector<T> t_{};
    std::vector<T> p_hat_{};
    std::vector<T> s_hat_{};
    std::vector<T> tmp_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the BiCGSTAB method.
   *
   * Convenience wrapper that constructs a BicgstabSolver, runs it to
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
   * @return Bicgstab_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  Bicgstab_summary<T>
  bicgstab(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Bicgstab_config<T> cfg,
    LinearOperator linear_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = BicgstabSolver(
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
