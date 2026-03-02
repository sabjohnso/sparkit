#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/eigen_target.hpp>
#include <sparkit/data/info.hpp>
#include <sparkit/data/lanczos.hpp>

namespace sparkit::data::detail {

  // --- Condition number estimation ---

  // Estimates ||A^{-1}||_1 using Hager's 1-norm power method.
  //
  // This is the scalar version of LAPACK dlacon (Hager 1984, O'Leary 1980).
  // Estimates ||A^{-1}||_1 via a power iteration on the sign matrix,
  // requiring only forward and adjoint solves — never forming A^{-1}.
  //
  // solve       : callable (b: span<T const>) -> vector<T>, solving A*x = b
  // trans_solve : callable (b: span<T const>) -> vector<T>, solving A^T*x = b
  // max_iter    : number of power iterations (5-10 is usually enough)
  template <typename T, typename SolveFn, typename TransSolveFn>
  T
  estimate_norm_1_inverse(
    config::size_type n,
    SolveFn solve,
    TransSolveFn trans_solve,
    config::size_type max_iter = 5) {
    auto sn = static_cast<std::size_t>(n);

    // Starting vector: x = (1/n, ..., 1/n)
    std::vector<T> x(sn, T{1} / static_cast<T>(n));
    T gamma{0};

    for (config::size_type iter = 0; iter < max_iter; ++iter) {
      // y = A^{-1} x
      auto y = solve(std::span<T const>{x});

      // gamma = ||y||_1
      T gamma_new{0};
      for (auto v : y) {
        gamma_new += std::abs(v);
      }

      // xi = sign(y), component-wise (+1 or -1)
      std::vector<T> xi(sn);
      for (std::size_t i = 0; i < sn; ++i) {
        xi[i] = y[i] >= T{0} ? T{1} : T{-1};
      }

      // z = A^{-T} xi
      auto z = trans_solve(std::span<T const>{xi});

      // Check convergence: ||z||_inf <= xi^T y
      T xi_dot_y{0};
      for (std::size_t i = 0; i < sn; ++i) {
        xi_dot_y += xi[i] * y[i];
      }
      T z_inf{0};
      config::size_type i_star{0};
      for (config::size_type i = 0; i < n; ++i) {
        T az = std::abs(z[static_cast<std::size_t>(i)]);
        if (az > z_inf) {
          z_inf = az;
          i_star = i;
        }
      }

      gamma = gamma_new;

      // Optimality: xi^T y = ||z||_inf implies convergence
      if (z_inf <= xi_dot_y) { break; }

      // Next iterate: x = e_{i*}
      std::fill(x.begin(), x.end(), T{0});
      x[static_cast<std::size_t>(i_star)] = T{1};
    }
    return gamma;
  }

  // Estimates cond_1(A) = ||A||_1 * ||A^{-1}||_1
  template <typename T, typename SolveFn, typename TransSolveFn>
  T
  estimate_condition_1(
    Compressed_row_matrix<T> const& A,
    SolveFn solve,
    TransSolveFn trans_solve,
    config::size_type max_iter = 5) {
    T norm_A = norm_1(A);
    T norm_A_inv =
      estimate_norm_1_inverse<T>(A.shape().row(), solve, trans_solve, max_iter);
    return norm_A * norm_A_inv;
  }

  // --- Eigenvalue bounds (symmetric matrices) ---

  // Estimates spectral radius rho(A) = max |lambda_i| via a thin Lanczos run.
  //
  // op       : callable (first, last, out_first) — matrix-vector product A*x
  // num_iter : number of Krylov steps
  template <typename T, typename LinearOperator>
  T
  estimate_spectral_radius(
    config::size_type n, LinearOperator op, config::size_type num_iter = 20) {
    auto m = std::min(n, num_iter + 1);
    Lanczos_config<T> cfg{
      .num_eigenvalues = 1,
      .krylov_dimension = m,
      .tolerance = T{1e-10},
      .max_restarts = 10,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos<T>(n, cfg, op);
    if (result.eigenvalues.empty()) { return T{0}; }
    return std::abs(result.eigenvalues[0]);
  }

  // Estimates {lambda_min, lambda_max} for a symmetric matrix via Lanczos.
  // Returns {lower_bound, upper_bound} as a pair.
  //
  // Runs two Lanczos passes: one for largest algebraic, one for smallest.
  template <typename T, typename LinearOperator>
  std::pair<T, T>
  estimate_eigenvalue_bounds(
    config::size_type n, LinearOperator op, config::size_type num_iter = 30) {
    auto m = std::min(n, num_iter + 1);

    // Largest algebraic eigenvalue
    Lanczos_config<T> cfg_max{
      .num_eigenvalues = 1,
      .krylov_dimension = m,
      .tolerance = T{1e-10},
      .max_restarts = 10,
      .target = Eigen_target::largest_algebraic,
      .collect_residuals = false};

    auto result_max = lanczos<T>(n, cfg_max, op);

    // Smallest algebraic eigenvalue
    Lanczos_config<T> cfg_min{
      .num_eigenvalues = 1,
      .krylov_dimension = m,
      .tolerance = T{1e-10},
      .max_restarts = 10,
      .target = Eigen_target::smallest_algebraic,
      .collect_residuals = false};

    auto result_min = lanczos<T>(n, cfg_min, op);

    T lambda_max =
      result_max.eigenvalues.empty() ? T{0} : result_max.eigenvalues[0];
    T lambda_min =
      result_min.eigenvalues.empty() ? T{0} : result_min.eigenvalues[0];

    return {lambda_min, lambda_max};
  }

} // end of namespace sparkit::data::detail
