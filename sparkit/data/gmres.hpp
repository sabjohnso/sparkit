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
   * @brief Configuration for GMRES(m) solver.
   *
   * @tparam T  Value type matching the linear system.
   */
  template <typename T>
  struct Gmres_config {
    /**
     * @brief Relative convergence tolerance:
     *  @f$ |g_{j+1}| / \|b\|_2 < \text{tolerance} @f$.
     */
    T tolerance{};

    /**
     * @brief Krylov subspace dimension per restart cycle (m in GMRES(m)).
     */
    size_type restart_dimension{};

    /**
     * @brief Hard upper bound on total matrix-vector products.
     */
    size_type max_iterations{};

    /**
     * @brief When true, each step's residual norm is appended to
     *  Gmres_summary::iteration_residuals.
     */
    bool collect_residuals{};
  };

  /**
   * @brief Convergence summary returned by the GMRES solver.
   *
   * @tparam T  Value type (e.g. double).
   */
  template <typename T>
  struct Gmres_summary {
    /** Final residual 2-norm. */
    T residual_norm{};

    /** Number of matrix-vector products actually performed. */
    size_type computed_iterations{};

    /** True when relative residual fell below tolerance. */
    bool converged{};

    /** Per-step residual norms (populated when
     *  Gmres_config::collect_residuals is true). */
    std::vector<T> iteration_residuals{};
  };

  /**
   * @brief Restarted GMRES(m) solver with left and right preconditioning.
   *
   * Solves a general (possibly nonsymmetric) linear system @f$ Ax = b @f$
   * using the restarted Generalized Minimum Residual method. The algorithm
   * follows the Templates book (Barrett et al., Chapter 2).
   *
   * The preconditioned system is:
   * @f$ M_L^{-1} A M_R^{-1} (M_R x) = M_L^{-1} b @f$
   *
   * The linear operator and preconditioners use an output-iterator
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
   * @see gmres  Free-function convenience wrapper.
   */
  template <
    typename T,
    typename BIter,
    typename XIter,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  class GmresSolver {
  public:
    GmresSolver(
      BIter bfirst,
      BIter blast,
      XIter xfirst,
      XIter xlast,
      Gmres_config<T> cfg,
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
    Gmres_summary<T>
    summary() const {
      return summary_;
    }

  private:
    void
    init() {
      n_ = std::distance(bfirst_, blast_);
      m_ = cfg_.restart_dimension;
      allocate_storage();
      if (cfg_.collect_residuals) {
        summary_.iteration_residuals.reserve(cfg_.max_iterations);
      }
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
    run() {
      using std::sqrt;
      bnorm_ = sqrt(std::inner_product(bfirst_, blast_, bfirst_, T{0}));
      if (bnorm_ == T{0}) {
        std::fill(xfirst_, xlast_, T{0});
        summary_.converged = true;
        return;
      }

      while (summary_.computed_iterations < cfg_.max_iterations &&
             !summary_.converged) {
        restart_cycle();
      }
    }

    void
    restart_cycle() {
      compute_initial_residual();

      std::fill(H_.begin(), H_.end(), T{0});
      std::fill(cs_.begin(), cs_.end(), T{0});
      std::fill(sn_.begin(), sn_.end(), T{0});

      auto m_eff =
        std::min(m_, cfg_.max_iterations - summary_.computed_iterations);

      size_type j_final = 0;
      for (size_type j = 0; j < m_eff; ++j) {
        arnoldi_step(j);
        apply_givens_rotations(j);
        ++summary_.computed_iterations;
        j_final = j;

        auto absres = std::abs(g_[static_cast<std::size_t>(j + 1)]);
        summary_.residual_norm = absres;
        if (cfg_.collect_residuals) {
          summary_.iteration_residuals.push_back(absres);
        }

        if (absres / bnorm_ < cfg_.tolerance) {
          summary_.converged = true;
          break;
        }
      }

      form_solution(j_final);
    }

    void
    compute_initial_residual() {
      // r0 = b - A*x
      linear_operator_(xfirst_, xlast_, w_.begin());
      auto un = static_cast<std::size_t>(n_);
      std::vector<T> r0(un);
      auto bit = bfirst_;
      for (std::size_t i = 0; i < un; ++i, ++bit) {
        r0[i] = *bit - w_[i];
      }

      // r0 = M_L^{-1} * r0
      left_preconditioner_(r0.cbegin(), r0.cend(), V_[0].begin());

      using std::sqrt;
      T beta = sqrt(
        std::inner_product(V_[0].begin(), V_[0].end(), V_[0].begin(), T{0}));

      for (std::size_t i = 0; i < un; ++i) {
        V_[0][i] /= beta;
      }

      auto um = static_cast<std::size_t>(m_);
      std::fill(g_.begin(), g_.end(), T{0});
      g_[0] = beta;

      // Clear Z storage for this cycle
      for (std::size_t j = 0; j < um; ++j) {
        std::fill(Z_[j].begin(), Z_[j].end(), T{0});
      }
    }

    void
    arnoldi_step(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto un = static_cast<std::size_t>(n_);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      // z_j = M_R^{-1} * v_j
      right_preconditioner_(V_[uj].cbegin(), V_[uj].cend(), Z_[uj].begin());

      // w = A * z_j
      linear_operator_(Z_[uj].begin(), Z_[uj].end(), w_.begin());

      // v_{j+1} = M_L^{-1} * w
      left_preconditioner_(w_.cbegin(), w_.cend(), V_[uj + 1].begin());

      // Modified Gram-Schmidt orthogonalization
      for (size_type i = 0; i <= j; ++i) {
        auto ui = static_cast<std::size_t>(i);
        T h_ij = std::inner_product(
          V_[ui].begin(), V_[ui].end(), V_[uj + 1].begin(), T{0});
        H_[ui + uj * um1] = h_ij;
        for (std::size_t k = 0; k < un; ++k) {
          V_[uj + 1][k] -= h_ij * V_[ui][k];
        }
      }

      // Normalize
      using std::sqrt;
      T h_j1j = sqrt(
        std::inner_product(
          V_[uj + 1].begin(), V_[uj + 1].end(), V_[uj + 1].begin(), T{0}));
      H_[uj + 1 + uj * um1] = h_j1j;

      if (h_j1j > T{0}) {
        for (std::size_t k = 0; k < un; ++k) {
          V_[uj + 1][k] /= h_j1j;
        }
      }
    }

    void
    apply_givens_rotations(size_type j) {
      auto uj = static_cast<std::size_t>(j);
      auto um1 = static_cast<std::size_t>(m_ + 1);

      // Apply previous rotations to the new column of H
      for (size_type i = 0; i < j; ++i) {
        auto ui = static_cast<std::size_t>(i);
        T temp = cs_[ui] * H_[ui + uj * um1] + sn_[ui] * H_[ui + 1 + uj * um1];
        H_[ui + 1 + uj * um1] =
          -sn_[ui] * H_[ui + uj * um1] + cs_[ui] * H_[ui + 1 + uj * um1];
        H_[ui + uj * um1] = temp;
      }

      // Generate new Givens rotation
      generate_givens(
        H_[uj + uj * um1], H_[uj + 1 + uj * um1], cs_[uj], sn_[uj]);

      // Apply new rotation to H column
      H_[uj + uj * um1] =
        cs_[uj] * H_[uj + uj * um1] + sn_[uj] * H_[uj + 1 + uj * um1];
      H_[uj + 1 + uj * um1] = T{0};

      // Apply new rotation to g
      T temp = cs_[uj] * g_[uj] + sn_[uj] * g_[uj + 1];
      g_[uj + 1] = -sn_[uj] * g_[uj] + cs_[uj] * g_[uj + 1];
      g_[uj] = temp;
    }

    static void
    generate_givens(T a, T b, T& cs, T& sn) {
      using std::sqrt;
      if (b == T{0}) {
        cs = T{1};
        sn = T{0};
      } else if (a == T{0}) {
        cs = T{0};
        sn = T{1};
      } else {
        T r = sqrt(a * a + b * b);
        cs = a / r;
        sn = b / r;
      }
    }

    void
    form_solution(size_type j_final) {
      auto um1 = static_cast<std::size_t>(m_ + 1);
      auto nj = static_cast<std::size_t>(j_final + 1);

      // Back-substitution: solve R * y = g
      std::vector<T> y(nj, T{0});
      for (auto ii = nj; ii > 0; --ii) {
        auto i = ii - 1;
        y[i] = g_[i];
        for (std::size_t k = i + 1; k < nj; ++k) {
          y[i] -= H_[i + k * um1] * y[k];
        }
        y[i] /= H_[i + i * um1];
      }

      // x += Z * y
      auto un = static_cast<std::size_t>(n_);
      auto xit = xfirst_;
      for (std::size_t i = 0; i < un; ++i, ++xit) {
        T sum = T{0};
        for (std::size_t k = 0; k < nj; ++k) {
          sum += Z_[k][i] * y[k];
        }
        *xit += sum;
      }
    }

    BIter bfirst_;
    BIter blast_;
    XIter xfirst_;
    XIter xlast_;
    Gmres_config<T> cfg_;
    LinearOperator linear_operator_;
    LeftPreconditioner left_preconditioner_;
    RightPreconditioner right_preconditioner_;
    size_type n_;
    size_type m_;
    Gmres_summary<T> summary_{};
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
   * @brief Solve @f$ Ax = b @f$ with the restarted GMRES(m) method.
   *
   * Convenience wrapper that constructs a GmresSolver, runs it to
   * completion, and returns the convergence summary. The solution is
   * written in-place into @c [xfirst, xlast).
   *
   * Pass identity callables (copying input to output) for unused
   * preconditioner slots.
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
   * @return Gmres_summary with convergence diagnostics.
   */
  template <
    typename BIter,
    typename XIter,
    typename T,
    typename LinearOperator,
    typename LeftPreconditioner,
    typename RightPreconditioner>
  Gmres_summary<T>
  gmres(
    BIter bfirst,
    BIter blast,
    XIter xfirst,
    XIter xlast,
    Gmres_config<T> cfg,
    LinearOperator linear_operator,
    LeftPreconditioner left_preconditioner,
    RightPreconditioner right_preconditioner) {
    auto solver = GmresSolver(
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
