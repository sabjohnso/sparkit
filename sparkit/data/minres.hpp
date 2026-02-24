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
   * @brief Configuration for the MINRES solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Minres_config {
    /**
     * @brief Relative convergence tolerance.
     *
     * For unpreconditioned MINRES: @f$ \|r_k\|_2 / \|b\|_2 < \text{tol} @f$.
     * For preconditioned MINRES: @f$ \|r_k\|_{M^{-1}} / \|r_0\|_{M^{-1}}
     *   < \text{tol} @f$.
     */
    T tolerance{};

    /**
     * @brief Hard upper bound on the number of iterations.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each iteration's residual norm estimate is appended
     *  to Minres_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the MINRES solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Minres_summary {
    /** Final residual norm estimate. */
    T residual_norm{};

    /** Number of iterations actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-iteration residual norm estimates (populated when
     *  Minres_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief Dual-preconditioned MINRES solver for symmetric (possibly
   *        indefinite) systems.
   *
   * Solves @f$ Ax = b @f$ where @f$ A @f$ is symmetric using Lanczos
   * tridiagonalization with Givens QR factorization and d-vector solution
   * updates. No @f$ A^T @f$ operator is needed.
   *
   * The combined preconditioner is @f$ M = M_L M_R @f$ (both symmetric,
   * @f$ M @f$ SPD), applied as @f$ z = M_R^{-1}(M_L^{-1}(v)) @f$.
   *
   * Reference: Paige & Saunders (1975); Templates book section 2.3.2.
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
   * @see minres  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class MinresSolver {
  public:
    MinresSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Minres_config<T> cfg,
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
    Minres_summary<T>
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
      initialize_vector(v_prev_);
      initialize_vector(v_);
      initialize_vector(z_);
      initialize_vector(w_);
      initialize_vector(tmp_);
      initialize_vector(d_prev_);
      initialize_vector(d_);
    }

    void
    initialize_vector(auto& vec) {
      vec.resize(n_);
      std::fill(std::begin(vec), std::end(vec), T{0});
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
      compute_initial_residual();
      if (summary_.converged) { return; }
      while (has_budget() && !summary_.converged) {
        minres_step();
      }
    }

    void
    compute_initial_residual() {
      using std::sqrt;
      // r = b - A*x → v_
      linear_operator_(xfirst_, xlast_, std::begin(w_));
      std::transform(
        bfirst_, blast_, std::begin(w_), std::begin(v_), std::minus<T>{});
      // z = M_R^{-1}(M_L^{-1}(r)) → z_
      left_preconditioner_(std::cbegin(v_), std::cend(v_), std::begin(tmp_));
      right_preconditioner_(std::cbegin(tmp_), std::cend(tmp_), std::begin(z_));
      // beta = sqrt(r^T z)
      beta_ = sqrt(
        std::inner_product(
          std::cbegin(v_), std::cend(v_), std::cbegin(z_), T{0}));
      if (beta_ == T{0}) {
        summary_.converged = true;
        return;
      }
      // normalize v_ and z_
      auto inv_beta = T{1} / beta_;
      std::transform(
        std::cbegin(v_), std::cend(v_), std::begin(v_), [inv_beta](auto vi) {
          return vi * inv_beta;
        });
      std::transform(
        std::cbegin(z_), std::cend(z_), std::begin(z_), [inv_beta](auto zi) {
          return zi * inv_beta;
        });
      // initialize scalars
      phi_bar_ = beta_;
      cs_prev_ = T{1};
      sn_prev_ = T{0};
      cs_ = T{1};
      sn_ = T{0};
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    minres_step() {
      lanczos_step();
      auto beta_new = precondition_lanczos();
      apply_givens_rotations(beta_new);
      update_solution();
      check_convergence();
      ++summary_.computed_iterations;
      if (summary_.converged || beta_new == T{0}) { return; }
      rotate_state(beta_new);
    }

    void
    lanczos_step() {
      // w_ = A * z_
      linear_operator_(std::cbegin(z_), std::cend(z_), std::begin(w_));
      // alpha = dot(z_, w_)
      alpha_ = std::inner_product(
        std::cbegin(z_), std::cend(z_), std::cbegin(w_), T{0});
      // w_ -= alpha*v_ + beta*v_prev_
      for (size_type i = 0; i < n_; ++i) {
        w_[i] -= alpha_ * v_[i] + beta_ * v_prev_[i];
      }
    }

    auto
    precondition_lanczos() {
      using std::sqrt;
      // Apply M^{-1} to w_: left(w_, tmp_); right(tmp_, v_prev_)
      // Reuse v_prev_ since it's no longer needed after lanczos_step
      left_preconditioner_(std::cbegin(w_), std::cend(w_), std::begin(tmp_));
      right_preconditioner_(
        std::cbegin(tmp_), std::cend(tmp_), std::begin(v_prev_));
      // beta_new = sqrt(dot(w_, v_prev_))
      auto beta_new = sqrt(
        std::inner_product(
          std::cbegin(w_), std::cend(w_), std::cbegin(v_prev_), T{0}));
      return beta_new;
    }

    void
    apply_givens_rotations(T beta_new) {
      using std::sqrt;
      // QR factorization: apply previous two Givens rotations to new column
      epsilon_ = sn_prev_ * beta_;
      auto delta_1 = cs_prev_ * beta_;
      auto delta = cs_ * delta_1 + sn_ * alpha_;
      auto rho_bar = -sn_ * delta_1 + cs_ * alpha_;
      auto rho = sqrt(rho_bar * rho_bar + beta_new * beta_new);
      if (rho == T{0}) {
        // QR breakdown — extremely rare
        summary_.converged = true;
        return;
      }
      cs_new_ = rho_bar / rho;
      sn_new_ = beta_new / rho;
      // Residual norm update
      phi_ = cs_new_ * phi_bar_;
      phi_bar_ = -sn_new_ * phi_bar_;
      rho_ = rho;
      delta_ = delta;
    }

    void
    update_solution() {
      // d-vector and solution update
      for (size_type i = 0; i < n_; ++i) {
        auto d_old = d_[i];
        d_prev_[i] = (z_[i] - delta_ * d_[i] - epsilon_ * d_prev_[i]) / rho_;
        d_[i] = d_prev_[i];
        d_prev_[i] = d_old;
        xfirst_[i] += phi_ * d_[i];
      }
    }

    void
    check_convergence() {
      using std::abs;
      auto residual_estimate = abs(phi_bar_);
      summary_.residual_norm = residual_estimate;
      summary_.converged = residual_estimate / bnorm_ < cfg_.tolerance;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_estimate);
      }
    }

    void
    rotate_state(T beta_new) {
      // z_ ← normalized z_new (v_prev_ holds unnormalized z_new)
      auto inv_beta_new = T{1} / beta_new;
      for (size_type i = 0; i < n_; ++i) {
        auto v_old = v_[i];
        z_[i] = v_prev_[i] * inv_beta_new;
        v_prev_[i] = v_old;
        v_[i] = w_[i] * inv_beta_new;
      }
      // Rotate Givens scalars
      cs_prev_ = cs_;
      sn_prev_ = sn_;
      cs_ = cs_new_;
      sn_ = sn_new_;
      beta_ = beta_new;
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    Minres_config<T> cfg_;
    LinearOperator linear_operator_;
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_{};
    Minres_summary<T> summary_{};
    T bnorm_{};
    T beta_{};
    T alpha_{};
    T epsilon_{};
    T delta_{};
    T rho_{};
    T phi_{};
    T phi_bar_{};
    T cs_prev_{};
    T sn_prev_{};
    T cs_{};
    T sn_{};
    T cs_new_{};
    T sn_new_{};
    std::vector<T> v_prev_{};
    std::vector<T> v_{};
    std::vector<T> z_{};
    std::vector<T> w_{};
    std::vector<T> tmp_{};
    std::vector<T> d_prev_{};
    std::vector<T> d_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the dual-preconditioned MINRES method.
   *
   * Convenience wrapper that constructs a MinresSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
   *
   * MINRES handles symmetric indefinite systems — unlike CG, it does not
   * require positive definiteness. It minimizes @f$ \|r_k\|_2 @f$ over
   * the Krylov subspace.
   *
   * Pass identity callables (copying input to output) for unused
   * preconditioner slots to recover unpreconditioned, left-only, or
   * right-only MINRES.
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
   * @return Minres_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  Minres_summary<T>
  minres(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Minres_config<T> cfg,
    LinearOperator linear_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = MinresSolver(
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
