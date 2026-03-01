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
#include <sparkit/data/least_squares_preconditioner.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::least_squares_preconditioner;
  using sparkit::data::detail::least_squares_preconditioner_apply;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // Least-squares preconditioner tests
  // ================================================================

  TEST_CASE("ls_prec - degree 0", "[least_squares_preconditioner]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    double lmin = 2.0;
    double lmax = 6.0;
    auto prec = least_squares_preconditioner(A, lmin, lmax, 0);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    least_squares_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    // Degree 0: single coefficient c_0, z = c_0 * r (uniform scaling)
    double scale = z[0] / r[0];
    CHECK(scale > 0.0);
    for (std::size_t i = 1; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(scale * r[i]).epsilon(1e-12));
    }
  }

  TEST_CASE(
    "ls_prec - left-prec CG tridiag", "[least_squares_preconditioner]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto prec = least_squares_preconditioner(A, 2.0, 6.0, 3);
    auto apply_ls = [&prec](auto first, auto last, auto out) {
      least_squares_preconditioner_apply(prec, first, last, out);
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
      apply_ls,
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

  TEST_CASE("ls_prec - left-prec CG grid", "[least_squares_preconditioner]") {
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

    auto prec = least_squares_preconditioner(A, 3.0, 13.0, 5);
    auto apply_ls = [&prec](auto first, auto last, auto out) {
      least_squares_preconditioner_apply(prec, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(
      b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_ls, identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE(
    "ls_prec - higher degree fewer iterations",
    "[least_squares_preconditioner]") {
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

    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    // Low degree
    auto prec_low = least_squares_preconditioner(A, 3.0, 13.0, 1);
    auto apply_low = [&prec_low](auto first, auto last, auto out) {
      least_squares_preconditioner_apply(prec_low, first, last, out);
    };
    std::vector<double> x_low(static_cast<std::size_t>(n), 0.0);
    auto summary_low = conjugate_gradient(
      b.begin(),
      b.end(),
      x_low.begin(),
      x_low.end(),
      cfg,
      apply_A,
      apply_low,
      identity);

    // Higher degree
    auto prec_high = least_squares_preconditioner(A, 3.0, 13.0, 5);
    auto apply_high = [&prec_high](auto first, auto last, auto out) {
      least_squares_preconditioner_apply(prec_high, first, last, out);
    };
    std::vector<double> x_high(static_cast<std::size_t>(n), 0.0);
    auto summary_high = conjugate_gradient(
      b.begin(),
      b.end(),
      x_high.begin(),
      x_high.end(),
      cfg,
      apply_A,
      apply_high,
      identity);

    REQUIRE(summary_low.converged);
    REQUIRE(summary_high.converged);
    CHECK(summary_high.computed_iterations <= summary_low.computed_iterations);
  }

} // end of namespace sparkit::testing
