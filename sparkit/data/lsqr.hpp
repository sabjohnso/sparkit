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
   * @brief Configuration for LSQR solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Lsqr_config {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ |\bar{\phi}| / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Hard upper bound on total matrix-vector products.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each step's residual norm is appended to
     *  Lsqr_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the LSQR solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Lsqr_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Lsqr_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief LSQR solver for least-squares problems with right
   * preconditioning.
   *
   * Solves @f$ \min_x \|Ax - b\|_2 @f$ using the LSQR algorithm
   * (Paige & Saunders, 1982). For square nonsingular systems, this
   * is equivalent to solving @f$ Ax = b @f$.
   *
   * Right preconditioning solves
   * @f$ \min_y \|A M^{-1} y - b\|_2 @f$, then recovers
   * @f$ x = M^{-1} y @f$. This preserves the least-squares structure
   * (left preconditioning would change the norm).
   *
   * The preconditioner @f$ M @f$ is assumed symmetric so that
   * @f$ M^{-T} = M^{-1} @f$. This covers IC(0), Jacobi, and SSOR.
   *
   * The algorithm is based on Lanczos bidiagonalization and requires
   * both @f$ A @f$ and @f$ A^T @f$ as callable operators. Each
   * iteration uses one matvec with A and one with A^T.
   *
   * @tparam T                 Value type (e.g. double).
   * @tparam BIter             Iterator over the right-hand side @a b.
   * @tparam XIter             Iterator over the solution vector @a x.
   * @tparam LinearOperator    Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam TransposeOperator Callable implementing
   *                           @f$ y \leftarrow A^T x @f$.
   * @tparam Preconditioner    Callable implementing
   *                           @f$ z \leftarrow M^{-1}r @f$.
   *
   * @see lsqr  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename TransposeOperator,
    typename Preconditioner>
  class LsqrSolver {
  public:
    LsqrSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Lsqr_config<T> cfg,
      LinearOperator linear_operator,
      TransposeOperator transpose_operator,
      Preconditioner preconditioner)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , transpose_operator_{transpose_operator}
        , preconditioner_{preconditioner} {
      init();
      run();
    }

    /**
     * @brief Return a summary of the solution process.
     */
    Lsqr_summary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      compute_sizes();
      allocate_storage();
      initialize_residual_norms();
    }

    void
    compute_sizes() {
      m_ = std::distance(bfirst_, blast_);
      n_ = std::distance(xfirst_, xlast_);
    }

    void
    allocate_storage() {
      auto um = static_cast<std::size_t>(m_);
      auto un = static_cast<std::size_t>(n_);
      u_.assign(um, T{0});
      v_.assign(un, T{0});
      w_.assign(un, T{0});
      z_.assign(un, T{0});
      u_work_.assign(um, T{0});
      v_work_.assign(un, T{0});
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

      initialize_bidiagonalization();

      while (has_budget() && !summary_.converged) {
        lsqr_step();
      }

      recover_solution();
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    initialize_bidiagonalization() {
      // u_1 = b / ||b||
      auto um = static_cast<std::size_t>(m_);
      auto bit = bfirst_;
      for (std::size_t i = 0; i < um; ++i, ++bit) {
        u_[i] = *bit;
      }
      beta_ = compute_norm(u_.begin(), u_.end());
      scale_vector(u_, T{1} / beta_);

      // v_work = A^T * u_1
      transpose_operator_(u_.cbegin(), u_.cend(), v_work_.begin());
      // z_ = M^{-1} * v_work
      preconditioner_(v_work_.cbegin(), v_work_.cend(), z_.begin());
      alpha_ = compute_norm(z_.begin(), z_.end());
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        v_[i] = z_[i] / alpha_;
      }

      // Initialize search direction
      std::copy(v_.begin(), v_.end(), w_.begin());

      // Initialize QR parameters
      phi_bar_ = beta_;
      rho_bar_ = alpha_;
    }

    void
    lsqr_step() {
      continue_bidiagonalization();
      apply_rotation();
      update_solution();
      update_search_direction();

      summary_.computed_iterations += 2;
      record_residual();
      check_convergence();
    }

    void
    continue_bidiagonalization() {
      auto um = static_cast<std::size_t>(m_);
      auto un = static_cast<std::size_t>(n_);

      // Forward: u_work = A * M^{-1} * v - alpha * u
      preconditioner_(v_.cbegin(), v_.cend(), z_.begin());
      linear_operator_(z_.cbegin(), z_.cend(), u_work_.begin());
      for (std::size_t i = 0; i < um; ++i) {
        u_work_[i] -= alpha_ * u_[i];
      }
      beta_ = compute_norm(u_work_.begin(), u_work_.end());
      for (std::size_t i = 0; i < um; ++i) {
        u_[i] = u_work_[i] / beta_;
      }

      // Backward: v_work = M^{-1} * A^T * u - beta * v
      transpose_operator_(u_.cbegin(), u_.cend(), v_work_.begin());
      preconditioner_(v_work_.cbegin(), v_work_.cend(), z_.begin());
      for (std::size_t i = 0; i < un; ++i) {
        v_work_[i] = z_[i] - beta_ * v_[i];
      }
      alpha_ = compute_norm(v_work_.begin(), v_work_.end());
      for (std::size_t i = 0; i < un; ++i) {
        v_[i] = v_work_[i] / alpha_;
      }
    }

    void
    apply_rotation() {
      using std::sqrt;

      rho_ = sqrt(rho_bar_ * rho_bar_ + beta_ * beta_);
      c_ = rho_bar_ / rho_;
      s_ = beta_ / rho_;
      theta_ = s_ * alpha_;
      rho_bar_ = -c_ * alpha_;
      phi_ = c_ * phi_bar_;
      phi_bar_ = s_ * phi_bar_;
    }

    void
    update_solution() {
      auto un = static_cast<std::size_t>(n_);
      T factor = phi_ / rho_;
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += factor * w_[i];
      }
    }

    void
    update_search_direction() {
      auto un = static_cast<std::size_t>(n_);
      T factor = theta_ / rho_;
      for (std::size_t i = 0; i < un; ++i) {
        w_[i] = v_[i] - factor * w_[i];
      }
    }

    void
    record_residual() {
      using std::abs;
      T residual_estimate = abs(phi_bar_);
      summary_.residual_norm = residual_estimate;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_estimate);
      }
    }

    void
    check_convergence() {
      using std::abs;
      if (abs(phi_bar_) / bnorm_ < cfg_.tolerance) {
        summary_.converged = true;
      }
    }

    void
    recover_solution() {
      // The iteration accumulated y (in the right-preconditioned system
      // min||AM^{-1}y - b||). Recover x = M^{-1} * y.
      auto un = static_cast<std::size_t>(n_);
      std::vector<T> y(un, T{0});
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        y[i] = *xit;
      }
      preconditioner_(y.cbegin(), y.cend(), z_.begin());
      xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit = z_[i];
      }
    }

    template <typename Iter>
    static T
    compute_norm(Iter first, Iter last) {
      using std::sqrt;
      return sqrt(std::inner_product(first, last, first, T{0}));
    }

    static void
    scale_vector(std::vector<T>& v, T factor) {
      for (auto& val : v) {
        val *= factor;
      }
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    Lsqr_config<T> cfg_;
    LinearOperator linear_operator_;
    TransposeOperator transpose_operator_;
    Preconditioner preconditioner_;
    size_type m_;
    size_type n_;
    Lsqr_summary<T> summary_{};
    T bnorm_{};
    T alpha_{};
    T beta_{};
    T phi_bar_{};
    T rho_bar_{};
    T rho_{};
    T c_{};
    T s_{};
    T theta_{};
    T phi_{};
    std::vector<T> u_{};
    std::vector<T> v_{};
    std::vector<T> w_{};
    std::vector<T> z_{};
    std::vector<T> u_work_{};
    std::vector<T> v_work_{};
  };

  /**
   * @brief Solve @f$ \min_x \|Ax - b\|_2 @f$ with the preconditioned
   * LSQR method.
   *
   * Convenience wrapper that constructs an LsqrSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
   *
   * @param bfirst             Start of right-hand side @a b.
   * @param blast              Past-the-end of @a b.
   * @param xfirst             Start of initial guess / solution @a x.
   * @param xlast              Past-the-end of @a x.
   * @param cfg                Solver configuration.
   * @param linear_operator    Output-iterator callable implementing
   *                           @f$ y \leftarrow Ax @f$.
   * @param transpose_operator Output-iterator callable implementing
   *                           @f$ y \leftarrow A^T x @f$.
   * @param preconditioner     Output-iterator callable implementing
   *                           @f$ z \leftarrow M^{-1}r @f$.
   * @return Lsqr_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename TransposeOperator,
    typename Preconditioner>
  Lsqr_summary<T>
  lsqr(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Lsqr_config<T> cfg,
    LinearOperator linear_operator,
    TransposeOperator transpose_operator,
    Preconditioner preconditioner) {
    auto solver = LsqrSolver(
      bfirst,
      blast,
      xfirst,
      xlast,
      cfg,
      linear_operator,
      transpose_operator,
      preconditioner);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
