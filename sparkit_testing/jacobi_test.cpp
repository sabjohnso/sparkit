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
#include <sparkit/data/jacobi.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::jacobi;
  using sparkit::data::detail::jacobi_apply;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // Jacobi preconditioner tests
  // ================================================================

  TEST_CASE("jacobi - diagonal matrix", "[jacobi]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    auto inv_d = jacobi(A);

    REQUIRE(inv_d.size() == 4);
    CHECK(inv_d[0] == Catch::Approx(0.5));
    CHECK(inv_d[1] == Catch::Approx(1.0 / 3.0));
    CHECK(inv_d[2] == Catch::Approx(0.25));
    CHECK(inv_d[3] == Catch::Approx(0.2));
  }

  TEST_CASE("jacobi - tridiag 4x4", "[jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto inv_d = jacobi(A);

    REQUIRE(inv_d.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(inv_d[i] == Catch::Approx(0.25));
    }
  }

  TEST_CASE("jacobi - apply to vector", "[jacobi]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 5.0},
       Entry<double>{Index{3, 3}, 10.0}});

    auto inv_d = jacobi(A);

    std::vector<double> r = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> z(4, 0.0);
    jacobi_apply(inv_d, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(3.0));
    CHECK(z[1] == Catch::Approx(3.0));
    CHECK(z[2] == Catch::Approx(4.0));
    CHECK(z[3] == Catch::Approx(3.0));
  }

  TEST_CASE("jacobi - left-prec CG tridiag", "[jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto inv_d = jacobi(A);
    auto apply_jacobi = [&inv_d](auto first, auto last, auto out) {
      jacobi_apply(inv_d, first, last, out);
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
      apply_jacobi,
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

  TEST_CASE("jacobi - left-prec CG grid", "[jacobi]") {
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

    auto inv_d = jacobi(A);
    auto apply_jacobi = [&inv_d](auto first, auto last, auto out) {
      jacobi_apply(inv_d, first, last, out);
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
      apply_jacobi,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("jacobi - right-prec CG tridiag", "[jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto inv_d = jacobi(A);
    auto apply_jacobi = [&inv_d](auto first, auto last, auto out) {
      jacobi_apply(inv_d, first, last, out);
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
      apply_jacobi);

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

  TEST_CASE("jacobi - right-prec CG grid", "[jacobi]") {
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

    auto inv_d = jacobi(A);
    auto apply_jacobi = [&inv_d](auto first, auto last, auto out) {
      jacobi_apply(inv_d, first, last, out);
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
      apply_jacobi);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("jacobi - zero diagonal throws", "[jacobi]") {
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{2, 2}, 3.0}});

    CHECK_THROWS_AS(jacobi(A), std::invalid_argument);
  }

} // end of namespace sparkit::testing
