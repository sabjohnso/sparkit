//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/chebyshev.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/incomplete_cholesky.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Chebyshev_config;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::chebyshev;
  using sparkit::data::detail::convdiff_centered_2d;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::nonsymmetric_sample;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // Unpreconditioned Chebyshev tests (identity preconditioner)
  // ================================================================

  TEST_CASE("chebyshev - identity", "[chebyshev]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0},
       Entry<double>{Index{3, 3}, 1.0}});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-12,
      .max_iterations = 10,
      .lambda_min = 1.0,
      .lambda_max = 1.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(b[i]).margin(1e-10));
    }
  }

  TEST_CASE("chebyshev - diagonal", "[chebyshev]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-10,
      .max_iterations = 100,
      .lambda_min = 2.0,
      .lambda_max = 5.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    CHECK(x[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(x[1] == Catch::Approx(4.0).margin(1e-8));
    CHECK(x[2] == Catch::Approx(5.0).margin(1e-8));
    CHECK(x[3] == Catch::Approx(6.0).margin(1e-8));
  }

  TEST_CASE("chebyshev - symmetric tridiag 4x4", "[chebyshev]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-10,
      .max_iterations = 100,
      .lambda_min = 2.0,
      .lambda_max = 6.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("chebyshev - nonsymmetric 4x4", "[chebyshev]") {
    auto A = nonsymmetric_sample<double>();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-10,
      .max_iterations = 100,
      .lambda_min = 2.0,
      .lambda_max = 6.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("chebyshev - grid 16-node", "[chebyshev]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-8,
      .max_iterations = 200,
      .lambda_min = 3.0,
      .lambda_max = 9.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("chebyshev - convection-diffusion 16-node", "[chebyshev]") {
    auto A = convdiff_centered_2d<double>(4, 4, 1.0, 0.3, 0.3, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-8,
      .max_iterations = 200,
      .lambda_min = 3.0,
      .lambda_max = 10.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("chebyshev - zero rhs", "[chebyshev]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-12,
      .max_iterations = 100,
      .lambda_min = 2.0,
      .lambda_max = 6.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-14));
    }
  }

  TEST_CASE("chebyshev - max iterations exceeded", "[chebyshev]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-14,
      .max_iterations = 2,
      .lambda_min = 3.0,
      .lambda_max = 9.0};

    auto summary = chebyshev(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    CHECK_FALSE(summary.converged);
    CHECK(summary.computed_iterations <= 2);
  }

  // ================================================================
  // Right-preconditioned Chebyshev tests (IC(0))
  // ================================================================

  TEST_CASE("right-preconditioned chebyshev - tridiag", "[chebyshev]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto L = incomplete_cholesky(A);
    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      auto y = forward_solve(L, std::span<double const>{first, last});
      auto z = forward_solve_transpose(L, std::span<double const>{y});
      std::copy(z.begin(), z.end(), out);
    };

    std::vector<double> x(4, 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-10,
      .max_iterations = 100,
      .lambda_min = 2.0,
      .lambda_max = 6.0};

    auto summary = chebyshev(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      identity,
      apply_inv_M);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("right-preconditioned chebyshev - grid", "[chebyshev]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    auto L = incomplete_cholesky(A);
    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      auto y = forward_solve(L, std::span<double const>{first, last});
      auto z = forward_solve_transpose(L, std::span<double const>{y});
      std::copy(z.begin(), z.end(), out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Chebyshev_config<double> cfg{
      .tolerance = 1e-8,
      .max_iterations = 200,
      .lambda_min = 3.0,
      .lambda_max = 9.0};

    auto summary = chebyshev(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      identity,
      apply_inv_M);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

} // end of namespace sparkit::testing
