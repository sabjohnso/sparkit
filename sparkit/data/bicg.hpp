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
   * @brief Configuration for BiCG solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Bicg_config {
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
     *  Bicg_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the BiCG solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Bicg_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Bicg_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief BiCG (Biconjugate Gradient) solver.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the biconjugate gradient method (Templates book, Algorithm 7.3).
   *
   * This method requires both @f$ A @f$ and @f$ A^T @f$ as callable
   * operators. Each iteration uses one matvec with A and one with A^T.
   *
   * @tparam T                 Value type (e.g. double).
   * @tparam BIter             Iterator over the right-hand side @a b.
   * @tparam XIter             Iterator over the solution vector @a x.
   * @tparam LinearOperator    Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam TransposeOperator Callable implementing
   *                           @f$ y \leftarrow A^T x @f$.
   *
   * @see bicg  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename TransposeOperator>
  class BicgSolver {
  public:
    BicgSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Bicg_config<T> cfg,
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
    Bicg_summary<T>
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
      r_tilde_.assign(un, T{0});
      p_.assign(un, T{0});
      p_tilde_.assign(un, T{0});
      q_.assign(un, T{0});
      q_tilde_.assign(un, T{0});
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
      rho_ = dot(r_tilde_, r_);

      while (has_budget() && !summary_.converged) {
        bicg_step();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    compute_initial_residual() {
      auto un = static_cast<std::size_t>(n_);
      std::vector<T> tmp(un, T{0});
      linear_operator_(xfirst_, xlast_, tmp.begin());
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r_[i] = *bit - tmp[i];
      }
    }

    void
    initialize_shadow_residual() {
      std::copy(r_.begin(), r_.end(), r_tilde_.begin());
    }

    void
    initialize_search_directions() {
      std::copy(r_.begin(), r_.end(), p_.begin());
      std::copy(r_tilde_.begin(), r_tilde_.end(), p_tilde_.begin());
    }

    void
    bicg_step() {
      compute_matvec_forward();
      T ptq = dot(p_tilde_, q_);
      T alpha = rho_ / ptq;

      update_solution(alpha);
      update_residual(alpha);
      update_shadow_residual(alpha);

      summary_.computed_iterations += 2;

      if (check_convergence()) return;

      T rho_new = dot(r_tilde_, r_);
      if (rho_new == T{0}) return;

      T beta = rho_new / rho_;
      update_search_directions(beta);
      rho_ = rho_new;
    }

    void
    compute_matvec_forward() {
      linear_operator_(p_.begin(), p_.end(), q_.begin());
    }

    void
    update_solution(T alpha) {
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        *xit += alpha * p_[i];
      }
    }

    void
    update_residual(T alpha) {
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        r_[i] -= alpha * q_[i];
      }
    }

    void
    update_shadow_residual(T alpha) {
      transpose_operator_(p_tilde_.begin(), p_tilde_.end(), q_tilde_.begin());
      auto un = static_cast<std::size_t>(n_);
      for (std::size_t i = 0; i < un; ++i) {
        r_tilde_[i] -= alpha * q_tilde_[i];
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
        p_[i] = r_[i] + beta * p_[i];
      }
      for (std::size_t i = 0; i < un; ++i) {
        p_tilde_[i] = r_tilde_[i] + beta * p_tilde_[i];
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
    Bicg_config<T> cfg_;
    LinearOperator linear_operator_;
    TransposeOperator transpose_operator_;
    size_type n_;
    Bicg_summary<T> summary_{};
    T bnorm_{};
    T rho_{};
    std::vector<T> r_{};
    std::vector<T> r_tilde_{};
    std::vector<T> p_{};
    std::vector<T> p_tilde_{};
    std::vector<T> q_{};
    std::vector<T> q_tilde_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the BiCG method.
   *
   * Convenience wrapper that constructs a BicgSolver, runs it to
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
   * @return Bicg_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename TransposeOperator>
  Bicg_summary<T>
  bicg(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Bicg_config<T> cfg,
    LinearOperator linear_operator,
    TransposeOperator transpose_operator) {
    auto solver = BicgSolver(
      bfirst, blast, xfirst, xlast, cfg, linear_operator, transpose_operator);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
