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
   * @brief Configuration for FGMRES(m) solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Fgmres_config {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ |g_{j+1}| / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Krylov subspace dimension per restart cycle (m in FGMRES(m)).
     */
    size_type restart_dimension{};

    /**
     * @brief Hard upper bound on total matrix-vector products.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each step's residual norm is appended to
     *  Fgmres_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the FGMRES solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Fgmres_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Fgmres_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief Flexible GMRES(m) solver with variable right preconditioning.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using Flexible GMRES (Saad, 1993). Unlike standard GMRES, FGMRES
   * allows the preconditioner to change between iterations, which is
   * essential for inner-outer Krylov methods and variable preconditioners.
   *
   * The preconditioned system is:
   * @f$ A M^{-1}(M x) = b @f$ (right preconditioning only)
   *
   * The linear operator and preconditioner use an output-iterator signature:
   * @code
   *   linear_operator(first, last, output_first)   // y = A * x
   *   preconditioner(first, last, output_first)     // z = M^{-1} * r
   * @endcode
   *
   * @tparam T               Value type (e.g. double).
   * @tparam BIter           Iterator over the right-hand side @a b.
   * @tparam XIter           Iterator over the solution vector @a x.
   * @tparam LinearOperator  Callable implementing @f$ y \leftarrow Ax @f$.
   * @tparam Preconditioner  Callable implementing
   *                         @f$ z \leftarrow M^{-1}r @f$.
   *
   * @see fgmres  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename Preconditioner>
  class FgmresSolver {
  public:
    FgmresSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Fgmres_config<T> cfg,
      LinearOperator linear_operator,
      Preconditioner preconditioner)
        : bfirst_{bfirst}
        , blast_{blast}
        , xfirst_{xfirst}
        , xlast_{xlast}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , preconditioner_{preconditioner} {
      init();
      run();
    }

    /**
     * @brief Return a summary of the solution process.
     */
    Fgmres_summary<T>
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
      m_ = cfg_.restart_dimension;
    }

    void
    allocate_storage() {
      auto un = static_cast<std::size_t>(n_);
      auto um = static_cast<std::size_t>(m_);

      V_.resize(um + 1, std::vector<T>(un, T{0}));
      Z_.resize(um, std::vector<T>(un, T{0}));
      H_.assign((um + 1) * um, T{0});
      cs_.assign(um, T{0});
      sn_.assign(um, T{0});
      g_.assign(um + 1, T{0});
      w_.assign(un, T{0});
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

      while (has_budget() && !summary_.converged) {
        restart_cycle();
      }
    }

    bool
    has_budget() const {
      return summary_.computed_iterations < cfg_.max_iterations;
    }

    void
    restart_cycle() {
      compute_initial_residual();
      clear_hessenberg();

      auto m_eff = effective_restart_dimension();
      size_type j_final = 0;

      for (size_type j = 0; j < m_eff; ++j) {
        arnoldi_step(j);
        apply_givens_rotations(j);
        ++summary_.computed_iterations;
        j_final = j;

        record_residual(j);
        if (check_convergence(j)) break;
      }

      form_solution(j_final);
    }

    void
    compute_initial_residual() {
      auto un = static_cast<std::size_t>(n_);

      linear_operator_(xfirst_, xlast_, w_.begin());

      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        V_[0][i] = *bit - w_[i];
      }

      T beta = compute_norm(V_[0].begin(), V_[0].end());
      scale_vector(V_[0], T{1} / beta);

      std::fill(g_.begin(), g_.end(), T{0});
      g_[0] = beta;

      clear_preconditioned_vectors();
    }

    void
    clear_hessenberg() {
      std::fill(H_.begin(), H_.end(), T{0});
      std::fill(cs_.begin(), cs_.end(), T{0});
      std::fill(sn_.begin(), sn_.end(), T{0});
    }

    void
    clear_preconditioned_vectors() {
      auto um = static_cast<std::size_t>(m_);
      for (std::size_t j = 0; j < um; ++j) {
        std::fill(Z_[j].begin(), Z_[j].end(), T{0});
      }
    }

    size_type
    effective_restart_dimension() const {
      return std::min(m_, cfg_.max_iterations - summary_.computed_iterations);
    }

    void
    arnoldi_step(size_type j) {
      auto uj = static_cast<std::size_t>(j);

      preconditioner_(V_[uj].cbegin(), V_[uj].cend(), Z_[uj].begin());
      linear_operator_(Z_[uj].begin(), Z_[uj].end(), w_.begin());
      std::copy(w_.begin(), w_.end(), V_[uj + 1].begin());

      orthogonalize(j);
      normalize_basis_vector(j);
    }

    void
    orthogonalize(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto un = static_cast<std::size_t>(n_);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      for (size_type i = 0; i <= j; ++i) {
        auto ui = static_cast<std::size_t>(i);
        T h_ij = std::inner_product(
          V_[ui].begin(), V_[ui].end(), V_[uj + 1].begin(), T{0});
        H_[ui + uj * um1] = h_ij;
        for (std::size_t k = 0; k < un; ++k) {
          V_[uj + 1][k] -= h_ij * V_[ui][k];
        }
      }
    }

    void
    normalize_basis_vector(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      T h_j1j = compute_norm(V_[uj + 1].begin(), V_[uj + 1].end());
      H_[uj + 1 + uj * um1] = h_j1j;

      if (h_j1j > T{0}) { scale_vector(V_[uj + 1], T{1} / h_j1j); }
    }

    void
    apply_givens_rotations(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      apply_previous_rotations(j);
      generate_new_rotation(j);

      H_[uj + uj * um1] =
        cs_[uj] * H_[uj + uj * um1] + sn_[uj] * H_[uj + 1 + uj * um1];
      H_[uj + 1 + uj * um1] = T{0};

      T temp = cs_[uj] * g_[uj] + sn_[uj] * g_[uj + 1];
      g_[uj + 1] = -sn_[uj] * g_[uj] + cs_[uj] * g_[uj + 1];
      g_[uj] = temp;
    }

    void
    apply_previous_rotations(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      for (size_type i = 0; i < j; ++i) {
        auto ui = static_cast<std::size_t>(i);
        T temp = cs_[ui] * H_[ui + uj * um1] + sn_[ui] * H_[ui + 1 + uj * um1];
        H_[ui + 1 + uj * um1] =
          -sn_[ui] * H_[ui + uj * um1] + cs_[ui] * H_[ui + 1 + uj * um1];
        H_[ui + uj * um1] = temp;
      }
    }

    void
    generate_new_rotation(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      T a = H_[uj + uj * um1];
      T b = H_[uj + 1 + uj * um1];

      if (b == T{0}) {
        cs_[uj] = T{1};
        sn_[uj] = T{0};
      } else if (a == T{0}) {
        cs_[uj] = T{0};
        sn_[uj] = T{1};
      } else {
        using std::sqrt;
        T r = sqrt(a * a + b * b);
        cs_[uj] = a / r;
        sn_[uj] = b / r;
      }
    }

    void
    record_residual(size_type j) {
      auto absres = std::abs(g_[static_cast<std::size_t>(j + 1)]);
      summary_.residual_norm = absres;
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.push_back(absres);
      }
    }

    bool
    check_convergence(size_type j) {
      auto absres = std::abs(g_[static_cast<std::size_t>(j + 1)]);
      if (absres / bnorm_ < cfg_.tolerance) {
        summary_.converged = true;
        return true;
      }
      return false;
    }

    void
    form_solution(size_type j_final) {
      auto y = back_substitute(j_final);
      update_solution(y, j_final);
    }

    std::vector<T>
    back_substitute(size_type j_final) const {
      auto um1 = static_cast<std::size_t>(m_ + 1);
      auto nj = static_cast<std::size_t>(j_final + 1);

      std::vector<T> y(nj, T{0});
      for (auto ii = nj; ii > 0; --ii) {
        auto i = ii - 1;
        y[i] = g_[i];
        for (std::size_t k = i + 1; k < nj; ++k) {
          y[i] -= H_[i + k * um1] * y[k];
        }
        y[i] /= H_[i + i * um1];
      }
      return y;
    }

    void
    update_solution(std::vector<T> const& y, size_type j_final) {
      auto un = static_cast<std::size_t>(n_);
      auto nj = static_cast<std::size_t>(j_final + 1);

      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        T sum = T{0};
        for (std::size_t k = 0; k < nj; ++k) {
          sum += Z_[k][i] * y[k];
        }
        *xit += sum;
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
    Fgmres_config<T> cfg_;
    LinearOperator linear_operator_;
    Preconditioner preconditioner_;
    size_type n_;
    size_type m_;
    Fgmres_summary<T> summary_{};
    T bnorm_{};
    std::vector<std::vector<T>> V_{};
    std::vector<std::vector<T>> Z_{};
    std::vector<T> H_{};
    std::vector<T> cs_{};
    std::vector<T> sn_{};
    std::vector<T> g_{};
    std::vector<T> w_{};
  };

  /**
   * @brief Solve @f$ Ax = b @f$ with the Flexible GMRES(m) method.
   *
   * Convenience wrapper that constructs an FgmresSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
   *
   * Pass an identity callable (copying input to output) for the
   * preconditioner slot to recover unpreconditioned FGMRES (equivalent
   * to standard GMRES without left preconditioning).
   *
   * @param bfirst          Start of right-hand side @a b.
   * @param blast           Past-the-end of @a b.
   * @param xfirst          Start of initial guess / solution @a x.
   * @param xlast           Past-the-end of @a x.
   * @param cfg             Solver configuration.
   * @param linear_operator Output-iterator callable implementing
   *                        @f$ y \leftarrow Ax @f$.
   * @param preconditioner  Output-iterator callable implementing
   *                        @f$ z \leftarrow M^{-1}r @f$.
   * @return Fgmres_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename Preconditioner>
  Fgmres_summary<T>
  fgmres(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Fgmres_config<T> cfg,
    LinearOperator linear_operator,
    Preconditioner preconditioner) {
    auto solver = FgmresSolver(
      bfirst, blast, xfirst, xlast, cfg, linear_operator, preconditioner);
    return solver.summary();
  }

} // end of namespace sparkit::data::detail
