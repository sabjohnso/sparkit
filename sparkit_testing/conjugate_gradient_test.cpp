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
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conjugate_gradient.hpp>
#include <sparkit/data/incomplete_cholesky.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::CGSummary;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // Unpreconditioned CG tests
  // ================================================================

  TEST_CASE("conjugate gradient - identity", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    CHECK(summary.computed_iterations <= 1);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(b[i]).margin(1e-10));
    }
  }

  TEST_CASE("conjugate gradient - diagonal", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    CHECK(x[0] == Catch::Approx(3.0).margin(1e-10));
    CHECK(x[1] == Catch::Approx(4.0).margin(1e-10));
    CHECK(x[2] == Catch::Approx(5.0).margin(1e-10));
    CHECK(x[3] == Catch::Approx(6.0).margin(1e-10));
  }

  TEST_CASE("conjugate gradient - tridiag 4x4", "[conjugate_gradient]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("conjugate gradient - grid 16-node", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Left-preconditioned CG tests
  // ================================================================

  TEST_CASE("left preconditioned CG - tridiag", "[conjugate_gradient]") {
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

    std::vector<double> x_pcg(4, 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary_pcg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_pcg.begin(),
      x_pcg.end(),
      cfg,
      apply_A,
      apply_inv_M,
      identity);

    std::vector<double> x_cg(4, 0.0);
    auto summary_cg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_cg.begin(),
      x_cg.end(),
      cfg,
      apply_A,
      identity,
      identity);

    REQUIRE(summary_pcg.converged);
    CHECK(summary_pcg.computed_iterations <= summary_cg.computed_iterations);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_pcg[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("left preconditioned CG - grid", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_inv_M,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Right-preconditioned CG tests
  // ================================================================

  TEST_CASE("right preconditioned CG - tridiag", "[conjugate_gradient]") {
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

    std::vector<double> x_rpcg(4, 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary_rpcg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_rpcg.begin(),
      x_rpcg.end(),
      cfg,
      apply_A,
      identity,
      apply_inv_M);

    std::vector<double> x_cg(4, 0.0);
    auto summary_cg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_cg.begin(),
      x_cg.end(),
      cfg,
      apply_A,
      identity,
      identity);

    REQUIRE(summary_rpcg.converged);
    CHECK(summary_rpcg.computed_iterations <= summary_cg.computed_iterations);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_rpcg[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("right preconditioned CG - grid", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
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

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("conjugate gradient - zero rhs", "[conjugate_gradient]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> x(4, 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-14));
    }
  }

  TEST_CASE("conjugate gradient - max iterations", "[conjugate_gradient]") {
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
    CGConfig<double> cfg{
      .tolerance = 1e-14, .restart_iterations = 50, .max_iterations = 1};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, identity);

    CHECK_FALSE(summary.converged);
    CHECK(summary.computed_iterations == 1);
  }

} // end of namespace sparkit::testing
