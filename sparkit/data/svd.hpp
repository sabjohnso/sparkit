#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/eigen_target.hpp>
#include <sparkit/data/lanczos.hpp>

namespace sparkit::data::detail {

  using size_type = sparkit::config::size_type;

  /**
   * @brief Configuration for the sparse SVD solver.
   */
  template <typename T>
  struct Svd_config {
    size_type num_singular_values{};
    size_type krylov_dimension{};
    T tolerance{};
    size_type max_restarts{};
    Eigen_target target{};
    bool collect_residuals{};
  };

  /**
   * @brief Result of the sparse SVD solver.
   */
  template <typename T>
  struct Svd_result {
    std::vector<T> singular_values{};
    std::vector<std::vector<T>> left_singular_vectors{};
    std::vector<std::vector<T>> right_singular_vectors{};
    std::vector<T> residual_norms{};
    size_type computed_restarts{};
    size_type converged_count{};
    bool converged{};
    std::vector<T> restart_residuals{};
  };

  /**
   * @brief Sparse SVD solver via Lanczos on the cross-product
   *        matrix.
   *
   * Computes a few singular values/vectors of a large sparse matrix
   * through the matrix-free callable interface, requiring both A
   * and A^T operators.
   *
   * The approach computes eigenvalues of A^T*A (or A*A^T for tall
   * matrices) using the proven Lanczos eigensolver, then recovers
   * singular triplets. This leverages the implicitly restarted
   * Lanczos with full reorthogonalization already implemented.
   *
   * For the smaller dimension d = min(m, n):
   * - If d == n: eigendecompose A^T*A (n x n), right vectors from
   *   eigenvectors, left vectors via u = Av / sigma.
   * - If d == m: eigendecompose A*A^T (m x m), left vectors from
   *   eigenvectors, right vectors via v = A^T u / sigma.
   *
   * Reference: Golub & Van Loan, "Matrix Computations", Section
   * 8.6.
   *
   * @tparam T                 Value type.
   * @tparam LinearOperator    Callable: op(first, last,
   *                           output_first) implementing y = A*x.
   * @tparam TransposeOperator Callable: op(first, last,
   *                           output_first) implementing y = A^T*x.
   */
  template <typename T, typename LinearOperator, typename TransposeOperator>
  class SvdSolver {
  public:
    SvdSolver(
      size_type m_rows,
      size_type n_cols,
      Svd_config<T> cfg,
      LinearOperator linear_operator,
      TransposeOperator transpose_operator)
        : m_rows_{m_rows}
        , n_cols_{n_cols}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , transpose_operator_{transpose_operator} {
      run();
    }

    Svd_result<T>
    result() const {
      return result_;
    }

  private:
    size_type m_rows_{};
    size_type n_cols_{};
    Svd_config<T> cfg_{};
    LinearOperator linear_operator_;
    TransposeOperator transpose_operator_;

    Svd_result<T> result_{};

    void
    run() {
      if (n_cols_ <= m_rows_) {
        run_ata();
      } else {
        run_aat();
      }
    }

    // Eigendecompose A^T*A (n x n) when n <= m.
    // Eigenvectors are right singular vectors.
    // Left singular vectors recovered via u_i = A*v_i / sigma_i.
    void
    run_ata() {
      auto sm_rows = static_cast<std::size_t>(m_rows_);
      std::vector<T> tmp(sm_rows, T{0});

      // op(x) = A^T * (A * x)
      auto ata_op = [&](auto first, auto last, auto out) {
        linear_operator_(first, last, tmp.data());
        transpose_operator_(tmp.data(), tmp.data() + m_rows_, out);
      };

      auto lanczos_result = run_lanczos(n_cols_, ata_op);

      convert_eigenvalues_to_singular_values(lanczos_result);

      // Right singular vectors = Lanczos eigenvectors.
      result_.right_singular_vectors = std::move(lanczos_result.eigenvectors);

      // Left singular vectors: u_i = A * v_i / sigma_i.
      compute_left_vectors();
    }

    // Eigendecompose A*A^T (m x m) when m < n.
    // Eigenvectors are left singular vectors.
    // Right singular vectors recovered via v_i = A^T*u_i / sigma_i.
    void
    run_aat() {
      auto sn_cols = static_cast<std::size_t>(n_cols_);
      std::vector<T> tmp(sn_cols, T{0});

      // op(x) = A * (A^T * x)
      auto aat_op = [&](auto first, auto last, auto out) {
        transpose_operator_(first, last, tmp.data());
        linear_operator_(tmp.data(), tmp.data() + n_cols_, out);
      };

      auto lanczos_result = run_lanczos(m_rows_, aat_op);

      convert_eigenvalues_to_singular_values(lanczos_result);

      // Left singular vectors = Lanczos eigenvectors.
      result_.left_singular_vectors = std::move(lanczos_result.eigenvectors);

      // Right singular vectors: v_i = A^T * u_i / sigma_i.
      compute_right_vectors();
    }

    template <typename Op>
    Lanczos_result<T>
    run_lanczos(size_type dim, Op op) {
      Lanczos_config<T> lcfg{
        .num_eigenvalues = cfg_.num_singular_values,
        .krylov_dimension = cfg_.krylov_dimension,
        .tolerance = cfg_.tolerance,
        .max_restarts = cfg_.max_restarts,
        .target = cfg_.target,
        .collect_residuals = cfg_.collect_residuals};

      return lanczos(dim, lcfg, op);
    }

    void
    convert_eigenvalues_to_singular_values(Lanczos_result<T>& lr) {
      using std::abs;
      using std::sqrt;

      result_.singular_values.clear();
      for (auto ev : lr.eigenvalues) {
        result_.singular_values.push_back(sqrt(abs(ev)));
      }
      result_.residual_norms = std::move(lr.residual_norms);
      result_.computed_restarts = lr.computed_restarts;
      result_.converged_count = lr.converged_count;
      result_.converged = lr.converged;
      result_.restart_residuals = std::move(lr.restart_residuals);
    }

    // u_i = A * v_i / sigma_i
    void
    compute_left_vectors() {
      auto sm_rows = static_cast<std::size_t>(m_rows_);
      result_.left_singular_vectors.clear();

      for (std::size_t k = 0; k < result_.singular_values.size(); ++k) {
        auto const& v = result_.right_singular_vectors[k];
        std::vector<T> u(sm_rows, T{0});

        linear_operator_(v.data(), v.data() + n_cols_, u.data());

        T sigma = result_.singular_values[k];
        if (sigma > T{0}) {
          for (auto& val : u) {
            val /= sigma;
          }
        }

        result_.left_singular_vectors.push_back(std::move(u));
      }
    }

    // v_i = A^T * u_i / sigma_i
    void
    compute_right_vectors() {
      auto sn_cols = static_cast<std::size_t>(n_cols_);
      result_.right_singular_vectors.clear();

      for (std::size_t k = 0; k < result_.singular_values.size(); ++k) {
        auto const& u = result_.left_singular_vectors[k];
        std::vector<T> v(sn_cols, T{0});

        transpose_operator_(u.data(), u.data() + m_rows_, v.data());

        T sigma = result_.singular_values[k];
        if (sigma > T{0}) {
          for (auto& val : v) {
            val /= sigma;
          }
        }

        result_.right_singular_vectors.push_back(std::move(v));
      }
    }
  };

  /**
   * @brief Convenience wrapper for the sparse SVD solver.
   *
   * @param m_rows             Number of rows.
   * @param n_cols             Number of columns.
   * @param cfg                Solver configuration.
   * @param linear_operator    Callable implementing y = A*x.
   * @param transpose_operator Callable implementing y = A^T*x.
   * @return Svd_result with singular values and vectors.
   */
  template <typename T, typename LinearOperator, typename TransposeOperator>
  Svd_result<T>
  svd(
    size_type m_rows,
    size_type n_cols,
    Svd_config<T> cfg,
    LinearOperator linear_operator,
    TransposeOperator transpose_operator) {
    auto solver = SvdSolver<T, LinearOperator, TransposeOperator>(
      m_rows, n_cols, cfg, linear_operator, transpose_operator);
    return solver.result();
  }

} // end of namespace sparkit::data::detail
