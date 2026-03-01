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
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/neumann_preconditioner.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::neumann_preconditioner;
  using sparkit::data::detail::neumann_preconditioner_apply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // Neumann preconditioner tests
  // ================================================================

  TEST_CASE("neumann - identity matrix", "[neumann_preconditioner]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0},
       Entry<double>{Index{3, 3}, 1.0}});

    auto prec = neumann_preconditioner(A, 1.0, 3);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    neumann_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    // For identity: (I - wA) = 0 when w=1, so z = w*r = r
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(r[i]));
    }
  }

  TEST_CASE("neumann - diagonal matrix", "[neumann_preconditioner]") {
    // Diagonal matrix with entries d_i.
    // omega = 1 / max(d_i) = 1/4
    // (I - wA) is diagonal with entries (1 - w*d_i)
    // Neumann series: w * sum_{k=0}^{d} (1 - w*d_i)^k
    //               = w * (1 - (1-w*d_i)^{d+1}) / (w*d_i)   [geometric sum]
    //               = (1 - (1-w*d_i)^{d+1}) / d_i
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 1.0}});

    double omega = 0.25; // 1/max(diag) = 1/4
    size_type degree = 5;
    auto prec = neumann_preconditioner(A, omega, degree);

    std::vector<double> r = {1.0, 1.0, 1.0};
    std::vector<double> z(3, 0.0);
    neumann_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    // z_i = (1 - (1 - omega*d_i)^{degree+1}) / d_i
    auto expected = [&](double d) {
      double ratio = 1.0 - omega * d;
      double geom = 1.0;
      for (size_type k = 0; k <= degree; ++k) {
        geom *= ratio;
      }
      return (1.0 - geom) / d;
    };

    CHECK(z[0] == Catch::Approx(expected(2.0)).epsilon(1e-12));
    CHECK(z[1] == Catch::Approx(expected(4.0)).epsilon(1e-12));
    CHECK(z[2] == Catch::Approx(expected(1.0)).epsilon(1e-12));
  }

  TEST_CASE("neumann - degree 0", "[neumann_preconditioner]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    double omega = 0.2;
    auto prec = neumann_preconditioner(A, omega, 0);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    neumann_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    // Degree 0: z = omega * r
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(omega * r[i]));
    }
  }

  TEST_CASE("neumann - left-prec CG tridiag", "[neumann_preconditioner]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto prec = neumann_preconditioner(A, 3);
    auto apply_neumann = [&prec](auto first, auto last, auto out) {
      neumann_preconditioner_apply(prec, first, last, out);
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
      apply_neumann,
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

  TEST_CASE("neumann - left-prec CG grid", "[neumann_preconditioner]") {
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

    auto prec = neumann_preconditioner(A, 3);
    auto apply_neumann = [&prec](auto first, auto last, auto out) {
      neumann_preconditioner_apply(prec, first, last, out);
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
      apply_neumann,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

} // end of namespace sparkit::testing
