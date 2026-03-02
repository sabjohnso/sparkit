//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/conditioning.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/sparse_lu.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::diagonal_matrix;
  using sparkit::data::detail::estimate_condition_1;
  using sparkit::data::detail::estimate_eigenvalue_bounds;
  using sparkit::data::detail::estimate_norm_1_inverse;
  using sparkit::data::detail::estimate_spectral_radius;
  using sparkit::data::detail::lu_solve;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::sparse_lu;
  using sparkit::data::detail::tridiagonal_matrix;

  // ================================================================
  // estimate_norm_1_inverse
  // ================================================================

  TEST_CASE(
    "conditioning - estimate_norm_1_inverse identity matrix",
    "[conditioning]") {
    // For identity matrix: A^{-1} = I, ||I||_1 = 1
    auto A = diagonal_matrix({1.0, 1.0, 1.0});
    auto factors = sparse_lu(A, false);

    auto solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };
    auto trans_solve = [&](std::span<double const> b) {
      return lu_solve(factors, b); // symmetric: same as forward solve
    };

    auto est = estimate_norm_1_inverse<double>(3, solve, trans_solve, 10);
    CHECK(est == Catch::Approx(1.0).epsilon(0.05));
  }

  TEST_CASE(
    "conditioning - estimate_norm_1_inverse diagonal exact", "[conditioning]") {
    // D = diag(1, 2, 4): ||D^{-1}||_1 = max col-sum = max(1, 0.5, 0.25) = 1
    auto A = diagonal_matrix({1.0, 2.0, 4.0});
    auto factors = sparse_lu(A, false);

    auto solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };
    auto trans_solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };

    auto est = estimate_norm_1_inverse<double>(3, solve, trans_solve, 10);
    // ||D^{-1}||_1 = max(1/1, 1/2, 1/4) = 1.0
    CHECK(est == Catch::Approx(1.0).epsilon(0.01));
  }

  TEST_CASE(
    "conditioning - estimate_norm_1_inverse ill-conditioned diagonal",
    "[conditioning]") {
    // D = diag(1, 0.01): ||D^{-1}||_1 = 100
    auto A = diagonal_matrix({1.0, 0.01});
    auto factors = sparse_lu(A, false);

    auto solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };
    auto trans_solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };

    auto est = estimate_norm_1_inverse<double>(2, solve, trans_solve, 10);
    // Exact ||D^{-1}||_1 = 100
    CHECK(est == Catch::Approx(100.0).epsilon(0.01));
  }

  // ================================================================
  // estimate_condition_1
  // ================================================================

  TEST_CASE("conditioning - estimate_condition_1 identity", "[conditioning]") {
    auto A = diagonal_matrix({1.0, 1.0, 1.0, 1.0});
    auto factors = sparse_lu(A, false);

    auto solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };
    auto trans_solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };

    auto cond = estimate_condition_1(A, solve, trans_solve, 10);
    // cond_1(I) = 1
    CHECK(cond == Catch::Approx(1.0).epsilon(0.05));
  }

  TEST_CASE(
    "conditioning - estimate_condition_1 ill-conditioned diagonal",
    "[conditioning]") {
    // D = diag(1, 100): ||D||_1 = 100, ||D^{-1}||_1 = 1, cond = 100
    auto A = diagonal_matrix({1.0, 100.0});
    auto factors = sparse_lu(A, false);

    auto solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };
    auto trans_solve = [&](std::span<double const> b) {
      return lu_solve(factors, b);
    };

    auto cond = estimate_condition_1(A, solve, trans_solve, 10);
    // Exact cond_1(D) = 100 * 1 = 100
    CHECK(cond == Catch::Approx(100.0).epsilon(0.01));
  }

  // ================================================================
  // estimate_spectral_radius
  // ================================================================

  TEST_CASE(
    "conditioning - estimate_spectral_radius diagonal exact",
    "[conditioning]") {
    // D = diag(1, 3, 2): spectral radius = 3
    auto A = diagonal_matrix({1.0, 3.0, 2.0});

    auto op = [&A](auto first, auto last, auto out) {
      auto y = multiply(A, std::span<double const>{first, last});
      std::copy(y.begin(), y.end(), out);
    };

    auto rho = estimate_spectral_radius<double>(3, op, 20);
    CHECK(rho == Catch::Approx(3.0).epsilon(1e-6));
  }

  TEST_CASE(
    "conditioning - estimate_spectral_radius tridiagonal", "[conditioning]") {
    // tridiagonal(n, -1, 2, -1): largest eigenvalue = 2 + 2*cos(pi/(n+1))
    // For n=5: 2 + 2*cos(pi/6) ≈ 3.732
    auto A = tridiagonal_matrix<double>(5, -1.0, 2.0, -1.0);

    auto op = [&A](auto first, auto last, auto out) {
      auto y = multiply(A, std::span<double const>{first, last});
      std::copy(y.begin(), y.end(), out);
    };

    auto rho = estimate_spectral_radius<double>(5, op, 30);
    double expected = 2.0 + 2.0 * std::cos(M_PI / 6.0);
    CHECK(rho == Catch::Approx(expected).epsilon(1e-4));
  }

  // ================================================================
  // estimate_eigenvalue_bounds
  // ================================================================

  TEST_CASE(
    "conditioning - estimate_eigenvalue_bounds diagonal exact",
    "[conditioning]") {
    // D = diag(1, 4, 2): lambda_min = 1, lambda_max = 4
    auto A = diagonal_matrix({1.0, 4.0, 2.0});

    auto op = [&A](auto first, auto last, auto out) {
      auto y = multiply(A, std::span<double const>{first, last});
      std::copy(y.begin(), y.end(), out);
    };

    auto [lmin, lmax] = estimate_eigenvalue_bounds<double>(3, op, 30);
    CHECK(lmin == Catch::Approx(1.0).epsilon(1e-6));
    CHECK(lmax == Catch::Approx(4.0).epsilon(1e-6));
  }

  TEST_CASE(
    "conditioning - estimate_eigenvalue_bounds tridiagonal", "[conditioning]") {
    // tridiagonal(n, -1, 2, -1):
    //   lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n
    //   lambda_min = 2 - 2*cos(pi/(n+1))
    //   lambda_max = 2 + 2*cos(pi/(n+1))  [for odd k near n/2 / k=n]
    // For n=6:
    //   lambda_min = 2 - 2*cos(pi/7) ≈ 0.198
    //   lambda_max = 2 - 2*cos(6*pi/7) = 2 + 2*cos(pi/7) ≈ 3.802
    auto A = tridiagonal_matrix<double>(6, -1.0, 2.0, -1.0);

    auto op = [&A](auto first, auto last, auto out) {
      auto y = multiply(A, std::span<double const>{first, last});
      std::copy(y.begin(), y.end(), out);
    };

    auto [lmin, lmax] = estimate_eigenvalue_bounds<double>(6, op, 30);

    double lambda_min_exact = 2.0 - 2.0 * std::cos(M_PI / 7.0);
    double lambda_max_exact = 2.0 + 2.0 * std::cos(M_PI / 7.0);

    CHECK(lmin == Catch::Approx(lambda_min_exact).epsilon(1e-4));
    CHECK(lmax == Catch::Approx(lambda_max_exact).epsilon(1e-4));
  }

  TEST_CASE(
    "conditioning - estimate_eigenvalue_bounds large diagonal",
    "[conditioning]") {
    // 100-point diagonal with d_i = 1 + 0.1*i, i=0..99.
    // lambda_min = 1.0, lambda_max = 10.9.
    // With n=100 and num_iter=30 (m=31 << 100), Lanczos avoids
    // the degenerate m==n case. Diagonal eigenvectors are canonical,
    // so convergence is fast.
    std::vector<double> diag;
    for (int i = 0; i < 100; ++i) {
      diag.push_back(1.0 + 0.1 * i);
    }
    auto A = diagonal_matrix(diag);
    auto n = A.shape().row();

    auto op = [&A](auto first, auto last, auto out) {
      auto y = multiply(A, std::span<double const>{first, last});
      std::copy(y.begin(), y.end(), out);
    };

    auto [lmin, lmax] = estimate_eigenvalue_bounds<double>(n, op, 30);

    CHECK(lmin == Catch::Approx(1.0).epsilon(1e-6));
    CHECK(lmax == Catch::Approx(10.9).epsilon(1e-6));
  }

} // end of namespace sparkit::testing
