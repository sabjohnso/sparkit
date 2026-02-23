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
   * @brief Configuration for CGS solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Cgs_config {
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
     *  Cgs_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the CGS solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Cgs_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Cgs_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief CGS (Conjugate Gradient Squared) solver.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the Conjugate Gradient Squared method (Templates book,
   * Algorithm 7.5).
   *
   * CGS avoids the transpose operator required by BiCG by "squaring" the
   * BiCG polynomial. Each iteration performs two matrix-vector products
   * with A. Convergence can be irregular but each iteration is cheaper
   * than BiCGSTAB (no stabilization step).
   *
   * @tparam T               Value type (e.g. double).
   * @tparam BIter           Iterator over the right-hand side @a b.
   * @tparam XIter           Iterator over the solution vector @a x.
   * @tparam LinearOperator  Callable implementing @f$ y \leftarrow Ax @f$.
   *
   * @see cgs  Free-function convenience wrapper.
   */
  template <typename T, typename BIter, typename XIter, typename LinearOperator>
  class CgsSolver {
  public:
    CgsSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Cgs_config<T> cfg,
      LinearOperator linear_operator)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator} {
      init();
      run();
    }

    /**
     * @brief Return a summary of the solution process.
     */
    Cgs_summary<T>
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
      u_.assign(un, T{0});
      q_.assign(un, T{0});
      q_hat_.assign(un, T{0});
      u_hat_.assign(un, T{0});
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
      initialize_search_directions();
      rho_ = dot(r_hat_, r_);

      while (has_budget() && !summary_.converged) {
        cgs_step();
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
    initialize_search_directions() {
      std::copy(r_.begin(), r_.end(), p_.begin());
      std::copy(r_.begin(), r_.end(), u_.begin());
    }

    void
    cgs_step() {
      compute_q_hat();
      T alpha = compute_alpha();

      compute_q(alpha);
      compute_u_hat();
      update_solution(alpha);
      compute_residual_update(alpha);

      summary_.computed_iterations += 2;

      if (check_convergence()) return;

      T rho_new = dot(r_hat_, r_);
      if (rho_new == T{0}) return;

      T beta = rho_new / rho_;
      update_search_directions(beta);
      rho_ = rho_new;
    }

    void
    compute_q_hat() {
      linear_operator_(p_.begin(), p_.end(), q_hat_.begin());
    }

    T
    compute_alpha() const {
      return rho_ / dot(r_hat_, q_hat_);
    }

    void
    compute_q(T alpha) {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        q_[i] = u_[i] - alpha * q_hat_[i];
      }
    }

    void
    compute_u_hat() {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        u_hat_[i] = u_[i] + q_[i];
      }
    }

    void
    update_solution(T alpha) {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += alpha * u_hat_[i];
      }
    }

    void
    compute_residual_update(T alpha) {
      linear_operator_(u_hat_.begin(), u_hat_.end(), tmp_.begin());
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        r_[i] -= alpha * tmp_[i];
      }
    }

    bool
    check_convergence() {
      T r_norm = compute_norm(r_.begin(), r_.end());
      summary_.residual_norm = r_norm;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(r_norm);
      }
      if (r_norm / bnorm_ < cfg_.tolerance) {
        summary_.converged = true;
        return true;
      }
      return false;
    }

    void
    update_search_directions(T beta) {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        u_[i] = r_[i] + beta * q_[i];
      }
      for (std::size_t i = 0; i < un; ++i) {
        p_[i] = u_[i] + beta * (q_[i] + beta * p_[i]);
      }
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
    Cgs_config<T> cfg_;
    LinearOperator linear_operator_;
    size_type n_;
    Cgs_summary<T> summary_{};
    T bnorm_{};
    T rho_{};
    std::vector<T> r_{};
    std::vector<T> r_hat_{};
    std::vector<T> p_{};
    std::vector<T> u_{};
    std::vector<T> q_{};
    std::vector<T> q_hat_{};
    std::vector<T> u_hat_{};
    std::vector<T> tmp_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the CGS method.
   *
   * Convenience wrapper that constructs a CgsSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
   *
   * @param bfirst           Start of right-hand side @a b.
   * @param blast            Past-the-end of @a b.
   * @param xfirst           Start of initial guess / solution @a x.
   * @param xlast            Past-the-end of @a x.
   * @param cfg              Solver configuration.
   * @param linear_operator  Output-iterator callable implementing
   *                         @f$ y \leftarrow Ax @f$.
   * @return Cgs_summary with convergence diagnostics.
   */
  template <typename BIter, typename XIter, typename T, typename LinearOperator>
  Cgs_summary<T>
  cgs(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Cgs_config<T> cfg,
    LinearOperator linear_operator) {
    auto solver = CgsSolver(bfirst, blast, xfirst, xlast, cfg, linear_operator);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
