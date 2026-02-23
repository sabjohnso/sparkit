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

  using size_type = config::size_type;

  /**
   * @brief Configuration for QMR solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Qmr_config {
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
     *  Qmr_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the QMR solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Qmr_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Qmr_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief QMR (Quasi-Minimal Residual) solver.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the Quasi-Minimal Residual method (Freund & Nachtigal, 1991).
   * The algorithm is based on Lanczos biorthogonalization with
   * quasi-residual minimization, following the Templates book
   * (Algorithm 7.6).
   *
   * QMR produces a smoother convergence curve than BiCG and avoids
   * the erratic residual behavior of BiCG.
   *
   * @note The d-vector recurrence is
   *  @f$ d_k = \eta_k p_k + (\theta_{k-1} \gamma_k)^2 d_{k-1} @f$.
   *  The coefficient of @f$ d_{k-1} @f$ uses @f$ \theta_{k-1} @f$
   *  (the value from the *previous* iteration), not the newly computed
   *  @f$ \theta_k @f$.  Using @f$ \theta_k @f$ instead causes the
   *  quasi-residual norm to decrease while the true solution stagnates,
   *  giving false convergence.
   *
   * @tparam T                 Value type (e.g. double).
   * @tparam BIter             Iterator over the right-hand side @a b.
   * @tparam XIter             Iterator over the solution vector @a x.
   * @tparam LinearOperator    Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam TransposeOperator Callable implementing
   *                           @f$ y \leftarrow A^T x @f$.
   *
   * @see qmr  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename TransposeOperator>
  class QmrSolver {
  public:
    QmrSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Qmr_config<T> cfg,
      LinearOperator linear_operator,
      TransposeOperator transpose_operator)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , transpose_operator_{transpose_operator} {
      init();
      run();
    }

    /**
     * @brief Return a summary of the solution process.
     */
    Qmr_summary<T>
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
      v_.assign(un, T{0});
      w_.assign(un, T{0});
      v_tilde_.assign(un, T{0});
      w_tilde_.assign(un, T{0});
      p_.assign(un, T{0});
      q_.assign(un, T{0});
      d_.assign(un, T{0});
      work_.assign(un, T{0});
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

      initialize_lanczos();

      while (has_budget() && !summary_.converged) {
        qmr_step();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    initialize_lanczos() {
      auto un = static_cast<std::size_t>(n_);

      // r0 = b - A*x
      linear_operator_(xfirst_, xlast_, work_.begin());
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        v_tilde_[i] = *bit - work_[i];
      }
      std::copy(v_tilde_.begin(), v_tilde_.end(), w_tilde_.begin());

      rho_ = compute_norm(v_tilde_.begin(), v_tilde_.end());
      xi_ = compute_norm(w_tilde_.begin(), w_tilde_.end());

      scale_into(v_tilde_, T{1} / rho_, v_);
      scale_into(w_tilde_, T{1} / xi_, w_);

      gamma_ = T{1};
      eta_ = T{-1};
      theta_ = T{0};
      epsilon_ = T{0};
      tau_ = rho_;

      std::fill(d_.begin(), d_.end(), T{0});
      iteration_ = 0;
    }

    void
    qmr_step() {
      ++iteration_;

      T delta = dot(w_, v_);
      if (delta == T{0}) return;

      update_search_directions(delta);
      compute_lanczos_step();

      T rho_new = compute_norm(v_tilde_.begin(), v_tilde_.end());
      T xi_new = compute_norm(w_tilde_.begin(), w_tilde_.end());

      apply_qmr_smoothing(rho_new);
      update_solution();

      summary_.computed_iterations += 2;
      record_residual();

      if (check_convergence()) return;

      if (rho_new == T{0} || xi_new == T{0}) return;

      advance_lanczos(rho_new, xi_new);
    }

    void
    update_search_directions(T delta) {
      auto un = static_cast<std::size_t>(n_);

      if (iteration_ == 1) {
        std::copy(v_.begin(), v_.end(), p_.begin());
        std::copy(w_.begin(), w_.end(), q_.begin());
      } else {
        T factor = (xi_ * delta) / epsilon_;
        for (std::size_t i = 0; i < un; ++i) {
          p_[i] = v_[i] - factor * p_[i];
        }
        T factor2 = (rho_ * delta) / epsilon_;
        for (std::size_t i = 0; i < un; ++i) {
          q_[i] = w_[i] - factor2 * q_[i];
        }
      }
    }

    void
    compute_lanczos_step() {
      auto un = static_cast<std::size_t>(n_);

      // a_p = A * p
      linear_operator_(p_.begin(), p_.end(), work_.begin());
      epsilon_ = dot(q_, work_);

      T beta = epsilon_ / dot(w_, v_);

      // v_tilde = A*p - beta*v
      for (std::size_t i = 0; i < un; ++i) {
        v_tilde_[i] = work_[i] - beta * v_[i];
      }

      // w_tilde = A^T*q - beta*w
      transpose_operator_(q_.begin(), q_.end(), work_.begin());
      for (std::size_t i = 0; i < un; ++i) {
        w_tilde_[i] = work_[i] - beta * w_[i];
      }

      beta_ = beta;
    }

    void
    apply_qmr_smoothing(T rho_new) {
      using std::abs;
      using std::sqrt;

      T theta_new = rho_new / (gamma_ * abs(beta_));
      T gamma_new = T{1} / sqrt(T{1} + theta_new * theta_new);

      // Update tau: |tau_k| = |tau_{k-1}| * theta_k * gamma_k
      tau_ = tau_ * theta_new * gamma_new;

      T eta_new =
        -eta_ * (rho_ / beta_) * (gamma_new * gamma_new) / (gamma_ * gamma_);

      // d_k = eta_k * p_k + (theta_{k-1} * gamma_k)^2 * d_{k-1}
      // Must save theta_old before overwriting
      theta_old_ = theta_;

      theta_ = theta_new;
      gamma_ = gamma_new;
      eta_ = eta_new;
    }

    void
    update_solution() {
      auto un = static_cast<std::size_t>(n_);

      T coeff = (theta_old_ * gamma_) * (theta_old_ * gamma_);
      for (std::size_t i = 0; i < un; ++i) {
        d_[i] = eta_ * p_[i] + coeff * d_[i];
      }

      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += d_[i];
      }
    }

    void
    record_residual() {
      using std::abs;
      T residual_estimate = abs(tau_);
      summary_.residual_norm = residual_estimate;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(residual_estimate);
      }
    }

    bool
    check_convergence() {
      using std::abs;
      if (abs(tau_) / bnorm_ < cfg_.tolerance) {
        summary_.converged = true;
        return true;
      }
      return false;
    }

    void
    advance_lanczos(T rho_new, T xi_new) {
      auto un = static_cast<std::size_t>(n_);

      for (std::size_t i = 0; i < un; ++i) {
        v_[i] = v_tilde_[i] / rho_new;
      }
      for (std::size_t i = 0; i < un; ++i) {
        w_[i] = w_tilde_[i] / xi_new;
      }

      rho_ = rho_new;
      xi_ = xi_new;
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

    static void
    scale_into(std::vector<T> const& src, T factor, std::vector<T>& dst) {
      for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i] * factor;
      }
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    Qmr_config<T> cfg_;
    LinearOperator linear_operator_;
    TransposeOperator transpose_operator_;
    size_type n_;
    Qmr_summary<T> summary_{};
    T bnorm_{};
    T rho_{};
    T xi_{};
    T gamma_{};
    T eta_{};
    T theta_{};
    T theta_old_{};
    T epsilon_{};
    T beta_{};
    T tau_{};
    size_type iteration_{};
    std::vector<T> v_{};
    std::vector<T> w_{};
    std::vector<T> v_tilde_{};
    std::vector<T> w_tilde_{};
    std::vector<T> p_{};
    std::vector<T> q_{};
    std::vector<T> d_{};
    std::vector<T> work_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the QMR method.
   *
   * Convenience wrapper that constructs a QmrSolver, runs it to
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
   * @return Qmr_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename TransposeOperator>
  Qmr_summary<T>
  qmr(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Qmr_config<T> cfg,
    LinearOperator linear_operator,
    TransposeOperator transpose_operator) {
    auto solver = QmrSolver(
      bfirst, blast, xfirst, xlast, cfg, linear_operator, transpose_operator);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
