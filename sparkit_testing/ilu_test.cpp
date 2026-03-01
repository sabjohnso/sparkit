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
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/gmres.hpp>
#include <sparkit/data/ilu.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Gmres_config;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::convdiff_centered_2d;
  using sparkit::data::detail::gmres;
  using sparkit::data::detail::ilu0;
  using sparkit::data::detail::ilu_apply;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::milu0;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::nonsymmetric_sample;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // ILU(0) tests
  // ================================================================

  TEST_CASE("ilu0 - diagonal matrix", "[ilu]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    auto factors = ilu0(A);

    // L should be identity: each row has one entry (i,i) with value 1.0
    for (size_type i = 0; i < 4; ++i) {
      CHECK(factors.L(i, i) == Catch::Approx(1.0).margin(1e-14));
    }

    // U should equal diag(A)
    CHECK(factors.U(0, 0) == Catch::Approx(2.0).margin(1e-14));
    CHECK(factors.U(1, 1) == Catch::Approx(3.0).margin(1e-14));
    CHECK(factors.U(2, 2) == Catch::Approx(4.0).margin(1e-14));
    CHECK(factors.U(3, 3) == Catch::Approx(5.0).margin(1e-14));
  }

  TEST_CASE("ilu0 - tridiag exact LU", "[ilu]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ilu0(A);

    // Tridiag has no fill, so L*U == A exactly.
    auto LU = multiply(factors.L, factors.U);

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();
    for (size_type i = 0; i < 4; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(LU(i, ci[p]) == Catch::Approx(vals[p]).margin(1e-12));
      }
    }
  }

  TEST_CASE("ilu0 - apply to vector", "[ilu]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ilu0(A);

    // Since L*U == A for tridiag, ilu_apply should give exact solve
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(4, 0.0);
    ilu_apply(factors, b.begin(), b.end(), x.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-12));
    }
  }

  TEST_CASE("ilu0 - left-prec GMRES tridiag", "[ilu]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto factors = ilu0(A);
    auto apply_ilu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    Gmres_config<double> cfg{
      .tolerance = 1e-12, .restart_dimension = 10, .max_iterations = 100};

    std::vector<double> x_prec(4, 0.0);
    auto summary_prec = gmres(
      b.begin(),
      b.end(),
      x_prec.begin(),
      x_prec.end(),
      cfg,
      apply_A,
      apply_ilu,
      identity);

    std::vector<double> x_unprec(4, 0.0);
    auto summary_unprec = gmres(
      b.begin(),
      b.end(),
      x_unprec.begin(),
      x_unprec.end(),
      cfg,
      apply_A,
      identity,
      identity);

    REQUIRE(summary_prec.converged);
    CHECK(
      summary_prec.computed_iterations <= summary_unprec.computed_iterations);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_prec[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ilu0 - left-prec GMRES grid", "[ilu]") {
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

    auto factors = ilu0(A);
    auto apply_ilu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Gmres_config<double> cfg{
      .tolerance = 1e-10, .restart_dimension = 20, .max_iterations = 200};

    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_ilu,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ilu0 - left-prec GMRES nonsymmetric", "[ilu]") {
    auto A = nonsymmetric_sample<double>();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto factors = ilu0(A);
    auto apply_ilu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    std::vector<double> x(4, 0.0);
    Gmres_config<double> cfg{
      .tolerance = 1e-12, .restart_dimension = 10, .max_iterations = 100};

    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_ilu,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ilu0 - left-prec GMRES convdiff", "[ilu]") {
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

    auto factors = ilu0(A);
    auto apply_ilu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Gmres_config<double> cfg{
      .tolerance = 1e-10, .restart_dimension = 20, .max_iterations = 200};

    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_ilu,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ilu0 - non-square throws", "[ilu]") {
    auto A = make_matrix(
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 2.0},
       Entry<double>{Index{2, 2}, 3.0}});

    CHECK_THROWS_AS(ilu0(A), std::invalid_argument);
  }

  // ================================================================
  // MILU(0) tests
  // ================================================================

  TEST_CASE("milu0 - tridiag same as ilu0", "[ilu]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors_ilu = ilu0(A);
    auto factors_milu = milu0(A);

    // No fill to drop on tridiag, so L and U should be identical.
    auto vals_ilu_l = factors_ilu.L.values();
    auto vals_milu_l = factors_milu.L.values();

    for (size_type i = 0; i < factors_ilu.L.size(); ++i) {
      CHECK(vals_milu_l[i] == Catch::Approx(vals_ilu_l[i]).margin(1e-14));
    }

    auto vals_ilu_u = factors_ilu.U.values();
    auto vals_milu_u = factors_milu.U.values();

    for (size_type i = 0; i < factors_ilu.U.size(); ++i) {
      CHECK(vals_milu_u[i] == Catch::Approx(vals_ilu_u[i]).margin(1e-14));
    }
  }

  TEST_CASE("milu0 - preserves row sums", "[ilu]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;
    auto factors = milu0(A);

    // (L*U)*e should equal A*e where e is the ones vector.
    auto LU = multiply(factors.L, factors.U);

    std::vector<double> ones(static_cast<std::size_t>(n), 1.0);
    auto ae = multiply(A, std::span<double const>{ones});
    auto lue = multiply(LU, std::span<double const>{ones});

    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(lue[i] == Catch::Approx(ae[i]).margin(1e-10));
    }
  }

  TEST_CASE("milu0 - left-prec GMRES grid", "[ilu]") {
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

    auto factors = milu0(A);
    auto apply_milu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Gmres_config<double> cfg{
      .tolerance = 1e-10, .restart_dimension = 20, .max_iterations = 200};

    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_milu,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("milu0 - left-prec GMRES convdiff", "[ilu]") {
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

    auto factors = milu0(A);
    auto apply_milu = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    Gmres_config<double> cfg{
      .tolerance = 1e-10, .restart_dimension = 20, .max_iterations = 200};

    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_milu,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

} // end of namespace sparkit::testing
