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
#include <sparkit/data/block_jacobi.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conjugate_gradient.hpp>
#include <sparkit/data/jacobi.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Block_jacobi_factors;
  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::block_jacobi;
  using sparkit::data::detail::block_jacobi_apply;
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
  // Block Jacobi preconditioner tests
  // ================================================================

  TEST_CASE("block jacobi - diagonal bs=2", "[block_jacobi]") {
    // Diagonal matrix: each 2x2 block is diagonal, so block solve = Jacobi
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 5.0},
       Entry<double>{Index{3, 3}, 10.0}});

    auto factors = block_jacobi(A, 2);

    std::vector<double> r = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> z(4, 0.0);
    block_jacobi_apply(factors, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(3.0));
    CHECK(z[1] == Catch::Approx(3.0));
    CHECK(z[2] == Catch::Approx(4.0));
    CHECK(z[3] == Catch::Approx(3.0));
  }

  TEST_CASE("block jacobi - tridiag bs=2", "[block_jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = block_jacobi(A, 2);

    // Block 0: [[4,-1],[-1,4]], Block 1: [[4,-1],[-1,4]]
    // Solve block 0 for r=[1,0]: [4,-1;-1,4]^{-1}[1;0] = [4/15; 1/15]
    std::vector<double> r = {1.0, 0.0, 0.0, 0.0};
    std::vector<double> z(4, 0.0);
    block_jacobi_apply(factors, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(4.0 / 15.0).margin(1e-12));
    CHECK(z[1] == Catch::Approx(1.0 / 15.0).margin(1e-12));
    CHECK(z[2] == Catch::Approx(0.0).margin(1e-12));
    CHECK(z[3] == Catch::Approx(0.0).margin(1e-12));
  }

  TEST_CASE("block jacobi - apply to vector", "[block_jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = block_jacobi(A, 2);

    // Each block is [[4,-1],[-1,4]], det=15
    // Inverse: [[4,1],[1,4]]/15
    // r = [3, 2, 1, 5]
    // Block 0: (4*3+1*2)/15 = 14/15, (1*3+4*2)/15 = 11/15
    // Block 1: (4*1+1*5)/15 = 9/15 = 3/5, (1*1+4*5)/15 = 21/15 = 7/5
    std::vector<double> r = {3.0, 2.0, 1.0, 5.0};
    std::vector<double> z(4, 0.0);
    block_jacobi_apply(factors, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(14.0 / 15.0).margin(1e-12));
    CHECK(z[1] == Catch::Approx(11.0 / 15.0).margin(1e-12));
    CHECK(z[2] == Catch::Approx(3.0 / 5.0).margin(1e-12));
    CHECK(z[3] == Catch::Approx(7.0 / 5.0).margin(1e-12));
  }

  TEST_CASE("block jacobi - bs=1 matches Jacobi", "[block_jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto bj_factors = block_jacobi(A, 1);
    auto inv_d = jacobi(A);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z_bj(4, 0.0);
    std::vector<double> z_j(4, 0.0);

    block_jacobi_apply(bj_factors, r.begin(), r.end(), z_bj.begin());
    jacobi_apply(inv_d, r.begin(), r.end(), z_j.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z_bj[i] == Catch::Approx(z_j[i]).margin(1e-12));
    }
  }

  TEST_CASE("block jacobi - left-prec CG tridiag", "[block_jacobi]") {
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

    auto factors = block_jacobi(A, 2);
    auto apply_bj = [&factors](auto first, auto last, auto out) {
      block_jacobi_apply(factors, first, last, out);
    };

    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    std::vector<double> x_bj(4, 0.0);
    auto summary_bj = conjugate_gradient(
      b.begin(),
      b.end(),
      x_bj.begin(),
      x_bj.end(),
      cfg,
      apply_A,
      apply_bj,
      identity);

    std::vector<double> x_j(4, 0.0);
    auto summary_j = conjugate_gradient(
      b.begin(),
      b.end(),
      x_j.begin(),
      x_j.end(),
      cfg,
      apply_A,
      apply_jacobi,
      identity);

    REQUIRE(summary_bj.converged);
    CHECK(summary_bj.computed_iterations <= summary_j.computed_iterations);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_bj[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("block jacobi - left-prec CG grid", "[block_jacobi]") {
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

    auto factors = block_jacobi(A, 4);
    auto apply_bj = [&factors](auto first, auto last, auto out) {
      block_jacobi_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_bj, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("block jacobi - right-prec CG tridiag", "[block_jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto factors = block_jacobi(A, 2);
    auto apply_bj = [&factors](auto first, auto last, auto out) {
      block_jacobi_apply(factors, first, last, out);
    };

    std::vector<double> x(4, 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, apply_bj);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("block jacobi - right-prec CG grid", "[block_jacobi]") {
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

    auto factors = block_jacobi(A, 4);
    auto apply_bj = [&factors](auto first, auto last, auto out) {
      block_jacobi_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, identity, apply_bj);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("block jacobi - non-divisible size", "[block_jacobi]") {
    // n=4, bs=3 -> block 0 is 3x3, block 1 is 1x1
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = block_jacobi(A, 3);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    block_jacobi_apply(factors, r.begin(), r.end(), z.begin());

    // Block 0: [[4,-1,0],[-1,4,-1],[0,-1,4]]
    // det = 4*(16-1) - (-1)*(-4) = 60 - 4 = 56
    // Inv = [[15,4,1],[4,16,4],[1,4,15]]/56
    // z[0] = (15*1+4*2+1*3)/56 = 26/56 = 13/28
    // z[1] = (4*1+16*2+4*3)/56 = 48/56 = 6/7
    // z[2] = (1*1+4*2+15*3)/56 = 54/56 = 27/28
    CHECK(z[0] == Catch::Approx(13.0 / 28.0).margin(1e-12));
    CHECK(z[1] == Catch::Approx(6.0 / 7.0).margin(1e-12));
    CHECK(z[2] == Catch::Approx(27.0 / 28.0).margin(1e-12));

    // Block 1: [[4]], z[3] = 4/4 = 1
    CHECK(z[3] == Catch::Approx(1.0).margin(1e-12));
  }

  TEST_CASE("block jacobi - bs=n exact solve", "[block_jacobi]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = block_jacobi(A, 4);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(4, 0.0);
    block_jacobi_apply(factors, b.begin(), b.end(), x.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-10));
    }
  }

} // end of namespace sparkit::testing
