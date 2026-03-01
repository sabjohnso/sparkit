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
#include <numeric>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/incomplete_cholesky.hpp>
#include <sparkit/data/lsqr.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/triangular_solve.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Lsqr_config;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::convdiff_centered_2d;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::lsqr;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::nonsymmetric_sample;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::transpose;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // 6x4 overdetermined system (full column rank, diagonally dominant)
  static Compressed_row_matrix<double>
  make_overdetermined_6x4() {
    return make_matrix(
      Shape{6, 4},
      {// Row 0
       Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       // Row 1
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 2}, -1.0},
       // Row 2
       Entry<double>{Index{2, 1}, -1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, -1.0},
       // Row 3
       Entry<double>{Index{3, 2}, -1.0},
       Entry<double>{Index{3, 3}, 4.0},
       // Row 4
       Entry<double>{Index{4, 0}, 2.0},
       Entry<double>{Index{4, 3}, 1.0},
       // Row 5
       Entry<double>{Index{5, 1}, 1.0},
       Entry<double>{Index{5, 2}, 2.0}});
  }

  // ================================================================
  // LSQR tests for square systems (identity preconditioner)
  // ================================================================

  TEST_CASE("lsqr - identity", "[lsqr]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0},
       Entry<double>{Index{3, 3}, 1.0}});

    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(b[i]).margin(1e-10));
    }
  }

  TEST_CASE("lsqr - diagonal", "[lsqr]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    CHECK(x[0] == Catch::Approx(3.0).margin(1e-10));
    CHECK(x[1] == Catch::Approx(4.0).margin(1e-10));
    CHECK(x[2] == Catch::Approx(5.0).margin(1e-10));
    CHECK(x[3] == Catch::Approx(6.0).margin(1e-10));
  }

  TEST_CASE("lsqr - symmetric tridiag 4x4", "[lsqr]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("lsqr - nonsymmetric 4x4", "[lsqr]") {
    auto A = nonsymmetric_sample<double>();
    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("lsqr - grid 16-node", "[lsqr]") {
    auto A = poisson_2d<double>(4, 5.0);
    auto AT = transpose(A);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-10, .max_iterations = 200};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("lsqr - convection-diffusion 16-node", "[lsqr]") {
    auto A = convdiff_centered_2d<double>(4, 4, 1.0, 0.3, 0.3, 5.0);
    auto AT = transpose(A);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-10, .max_iterations = 200};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("lsqr - zero rhs", "[lsqr]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-14));
    }
  }

  TEST_CASE("lsqr - max iterations exceeded", "[lsqr]") {
    auto A = poisson_2d<double>(4, 5.0);
    auto AT = transpose(A);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-14, .max_iterations = 2};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    CHECK_FALSE(summary.converged);
    CHECK(summary.computed_iterations <= 2);
  }

  // ================================================================
  // Overdetermined (least-squares) system
  // ================================================================

  TEST_CASE("lsqr - overdetermined system", "[lsqr]") {
    auto A = make_overdetermined_6x4();
    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    // b = A * [1, 2, 3, 4]^T -- consistent system, so LS solution is exact
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(4, 0.0);
    Lsqr_config<double> cfg{.tolerance = 1e-10, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("lsqr - residual collection", "[lsqr]") {
    auto A = poisson_2d<double>(4, 5.0);
    auto AT = transpose(A);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Lsqr_config<double> cfg{
      .tolerance = 1e-10, .max_iterations = 200, .collect_residuals = true};

    auto summary = lsqr(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_AT, identity);

    REQUIRE(summary.converged);
    CHECK(summary.iteration_residuals.size() > 0);
  }

  // ================================================================
  // Preconditioned LSQR tests (IC(0), right preconditioning)
  // ================================================================

  TEST_CASE("preconditioned lsqr - tridiag", "[lsqr]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto AT = transpose(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
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
    Lsqr_config<double> cfg{.tolerance = 1e-12, .max_iterations = 100};

    auto summary = lsqr(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_AT,
      apply_inv_M);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("preconditioned lsqr - grid", "[lsqr]") {
    auto A = poisson_2d<double>(4, 5.0);
    auto AT = transpose(A);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_AT = [&AT](auto first, auto last, auto out) {
      auto result = multiply(AT, std::span<double const>{first, last});
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
    Lsqr_config<double> cfg{.tolerance = 1e-10, .max_iterations = 200};

    auto summary = lsqr(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_AT,
      apply_inv_M);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

} // end of namespace sparkit::testing
