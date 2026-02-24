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
   * @brief BiCG (Biconjugate Gradient) solver with left and right
   *  preconditioning.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the biconjugate gradient method (Templates book, Algorithm 7.3).
   * The combined preconditioner is @f$ M = M_L M_R @f$, applied as:
   * - Primal: @f$ z = M_R^{-1}(M_L^{-1}(r)) @f$ (left first, then right)
   * - Adjoint: @f$ \tilde{z} = M_L^{-1}(M_R^{-1}(\tilde{r})) @f$
   *   (reversed order, since @f$ M^{-T} = M_L^{-T} M_R^{-T} @f$ and
   *   both factors are symmetric)
   *
   * This method requires both @f$ A @f$ and @f$ A^T @f$ as callable
   * operators. Each iteration uses one matvec with A and one with A^T.
   *
   * Both @f$ M_L @f$ and @f$ M_R @f$ are assumed symmetric so that
   * @f$ M^{-T} = M^{-1} @f$. This covers IC(0), Jacobi, and SSOR.
   *
   * @tparam T                    Value type (e.g. double).
   * @tparam BIter                Iterator over the right-hand side @a b.
   * @tparam XIter                Iterator over the solution vector @a x.
   * @tparam LinearOperator       Callable implementing
   *                              @f$ y \leftarrow Ax @f$.
   * @tparam TransposeOperator    Callable implementing
   *                              @f$ y \leftarrow A^T x @f$.
   * @tparam LeftPreconditioner   Callable implementing
   *                              @f$ z \leftarrow M_L^{-1}r @f$.
   * @tparam RightPreconditioner  Callable implementing
   *                              @f$ w \leftarrow M_R^{-1}p @f$.
   *
   * @see bicg  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename TransposeOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class BicgSolver {
  public:
    BicgSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Bicg_config<T> cfg,
      LinearOperator linear_operator,
      TransposeOperator transpose_operator,
      LeftPreconditioner left_preconditioner,
      RightPreconditioner right_preconditioner)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , transpose_operator_{transpose_operator}
        , left_preconditioner_{left_preconditioner}
        , right_preconditioner_{right_preconditioner} {
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
      z_.assign(un, T{0});
      z_tilde_.assign(un, T{0});
      p_.assign(un, T{0});
      p_tilde_.assign(un, T{0});
      q_.assign(un, T{0});
      q_tilde_.assign(un, T{0});
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
      apply_preconditioner();
      initialize_search_directions();
      rho_ = dot(r_tilde_, z_);

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
      linear_operator_(xfirst_, xlast_, tmp_.begin());
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r_[i] = *bit - tmp_[i];
      }
    }

    void
    initialize_shadow_residual() {
      std::copy(r_.begin(), r_.end(), r_tilde_.begin());
    }

    void
    apply_preconditioner() {
      // Primal: left first, then right
      left_preconditioner_(r_.cbegin(), r_.cend(), tmp_.begin());
      right_preconditioner_(tmp_.cbegin(), tmp_.cend(), z_.begin());
      // Adjoint: reversed order (right first, then left)
      right_preconditioner_(r_tilde_.cbegin(), r_tilde_.cend(), tmp_.begin());
      left_preconditioner_(tmp_.cbegin(), tmp_.cend(), z_tilde_.begin());
    }

    void
    initialize_search_directions() {
      std::copy(z_.begin(), z_.end(), p_.begin());
      std::copy(z_tilde_.begin(), z_tilde_.end(), p_tilde_.begin());
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

      apply_preconditioner();
      T rho_new = dot(r_tilde_, z_);
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
        p_[i] = z_[i] + beta * p_[i];
      }
      for (std::size_t i = 0; i < un; ++i) {
        p_tilde_[i] = z_tilde_[i] + beta * p_tilde_[i];
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
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_;
    Bicg_summary<T> summary_{};
    T bnorm_{};
    T rho_{};
    std::vector<T> r_{};
    std::vector<T> r_tilde_{};
    std::vector<T> z_{};
    std::vector<T> z_tilde_{};
    std::vector<T> p_{};
    std::vector<T> p_tilde_{};
    std::vector<T> q_{};
    std::vector<T> q_tilde_{};
    std::vector<T> tmp_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the preconditioned BiCG method.
   *
   * Convenience wrapper that constructs a BicgSolver, runs it to
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
   * @param transpose_operator   Output-iterator callable implementing
   *                             @f$ y \leftarrow A^T x @f$.
   * @param left_preconditioner  Output-iterator callable implementing
   *                             @f$ z \leftarrow M_L^{-1}r @f$.
   * @param right_preconditioner Output-iterator callable implementing
   *                             @f$ w \leftarrow M_R^{-1}p @f$.
   * @return Bicg_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename TransposeOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  Bicg_summary<T>
  bicg(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Bicg_config<T> cfg,
    LinearOperator linear_operator,
    TransposeOperator transpose_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = BicgSolver(
      bfirst,
      blast,
      xfirst,
      xlast,
      cfg,
      linear_operator,
      transpose_operator,
      left_preconditioner,
      right_preconditioner);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
