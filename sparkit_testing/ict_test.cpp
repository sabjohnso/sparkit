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
#include <sparkit/data/ict.hpp>
#include <sparkit/data/incomplete_cholesky.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Ict_config;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::arrow_matrix;
  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::ic_apply;
  using sparkit::data::detail::ict;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::transpose;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // ICT tests
  // ================================================================

  TEST_CASE("ict - diagonal", "[ict]") {
    // A = diag(4, 9, 16) → L = diag(2, 3, 4)
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0}}};

    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto L = ict(A, cfg);

    REQUIRE(L.shape().row() == 3);
    REQUIRE(L.shape().column() == 3);
    REQUIRE(L.size() == 3);

    CHECK(L(0, 0) == Catch::Approx(2.0));
    CHECK(L(1, 1) == Catch::Approx(3.0));
    CHECK(L(2, 2) == Catch::Approx(4.0));
  }

  TEST_CASE("ict - tridiag exact", "[ict]") {
    // With tau=0 and large fill limit, ICT should be exact Cholesky
    // for tridiagonal (no fill-in anyway).
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};

    auto L_ict = ict(A, cfg);
    auto L_ic = incomplete_cholesky(A);

    REQUIRE(L_ict.size() == L_ic.size());
    auto ict_vals = L_ict.values();
    auto ic_vals = L_ic.values();
    for (size_type i = 0; i < L_ict.size(); ++i) {
      CHECK(ict_vals[i] == Catch::Approx(ic_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE("ict - apply to vector", "[ict]") {
    // Tridiag: ICT with tau=0 is exact, so ic_apply gives exact solve
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto L = ict(A, cfg);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> z(4);

    ic_apply(L, b.begin(), b.end(), z.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(x_true[i]).margin(1e-10));
    }
  }

  TEST_CASE("ict - large tau drops aggressively", "[ict]") {
    // Large drop tolerance should produce fewer nonzeros
    auto A = arrow_matrix<double>(5, 10.0, 1.0);
    Ict_config<double> cfg_generous{.drop_tolerance = 0.0, .fill_limit = 10};
    Ict_config<double> cfg_aggressive{.drop_tolerance = 0.5, .fill_limit = 10};

    auto L_generous = ict(A, cfg_generous);
    auto L_aggressive = ict(A, cfg_aggressive);

    CHECK(L_aggressive.size() <= L_generous.size());
  }

  TEST_CASE("ict - small tau preserves more fill", "[ict]") {
    auto A = poisson_2d<double>(4, 5.0);
    Ict_config<double> cfg_small{.drop_tolerance = 1e-4, .fill_limit = 20};
    Ict_config<double> cfg_large{.drop_tolerance = 0.1, .fill_limit = 20};

    auto L_small = ict(A, cfg_small);
    auto L_large = ict(A, cfg_large);

    CHECK(L_small.size() >= L_large.size());
  }

  TEST_CASE("ict - left-prec CG tridiag", "[ict]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto L = ict(A, cfg);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      ic_apply(L, first, last, out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cg_cfg,
      apply_A,
      apply_inv_M,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ict - left-prec CG grid", "[ict]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;
    Ict_config<double> cfg{.drop_tolerance = 1e-3, .fill_limit = 10};
    auto L = ict(A, cfg);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      ic_apply(L, first, last, out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cg_cfg,
      apply_A,
      apply_inv_M,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ict - left-prec CG arrow", "[ict]") {
    auto A = arrow_matrix<double>(5, 10.0, 1.0);
    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto L = ict(A, cfg);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      ic_apply(L, first, last, out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(5, 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cg_cfg,
      apply_A,
      apply_inv_M,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ict - non-square throws", "[ict]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    Ict_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    CHECK_THROWS_AS(ict(A, cfg), std::invalid_argument);
  }

  TEST_CASE("ict - fewer iters than ic0", "[ict]") {
    // With generous fill, ICT should produce a better preconditioner
    // than IC(0), requiring fewer CG iterations.
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

    // IC(0)
    auto L_ic = incomplete_cholesky(A);
    auto apply_ic = [&L_ic](auto first, auto last, auto out) {
      ic_apply(L_ic, first, last, out);
    };

    std::vector<double> x_ic(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};
    auto summary_ic = conjugate_gradient(
      b.begin(),
      b.end(),
      x_ic.begin(),
      x_ic.end(),
      cg_cfg,
      apply_A,
      apply_ic,
      identity);

    // ICT with generous fill
    Ict_config<double> ict_cfg{.drop_tolerance = 1e-6, .fill_limit = 20};
    auto L_ict = ict(A, ict_cfg);
    auto apply_ict = [&L_ict](auto first, auto last, auto out) {
      ic_apply(L_ict, first, last, out);
    };

    std::vector<double> x_ict(static_cast<std::size_t>(n), 0.0);
    auto summary_ict = conjugate_gradient(
      b.begin(),
      b.end(),
      x_ict.begin(),
      x_ict.end(),
      cg_cfg,
      apply_A,
      apply_ict,
      identity);

    REQUIRE(summary_ic.converged);
    REQUIRE(summary_ict.converged);
    CHECK(summary_ict.computed_iterations <= summary_ic.computed_iterations);
  }

} // end of namespace sparkit::testing
