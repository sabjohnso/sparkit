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
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::cholesky;
  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::extract_lower_triangle;
  using sparkit::data::detail::ic_apply;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::mic0;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::transpose;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  // ================================================================
  // incomplete_cholesky tests
  // ================================================================

  TEST_CASE("incomplete cholesky - diagonal", "[incomplete_cholesky]") {
    // A = diag(4, 9, 16, 25)  ->  IC(0) == full Cholesky = diag(2, 3, 4, 5)
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0},
       Entry<double>{Index{3, 3}, 25.0}}};

    auto L = incomplete_cholesky(A);

    REQUIRE(L.shape().row() == 4);
    REQUIRE(L.shape().column() == 4);
    REQUIRE(L.size() == 4);

    CHECK(L(0, 0) == Catch::Approx(2.0));
    CHECK(L(1, 1) == Catch::Approx(3.0));
    CHECK(L(2, 2) == Catch::Approx(4.0));
    CHECK(L(3, 3) == Catch::Approx(5.0));
  }

  TEST_CASE("incomplete cholesky - 2x2 SPD", "[incomplete_cholesky]") {
    // A = [[4, 2], [2, 5]]  ->  IC(0) == full Cholesky (no fill possible)
    auto A = make_matrix(
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 5.0}});

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    REQUIRE(L_ic.size() == L_full.size());
    auto ic_vals = L_ic.values();
    auto full_vals = L_full.values();
    for (size_type i = 0; i < L_ic.size(); ++i) {
      CHECK(ic_vals[i] == Catch::Approx(full_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE("incomplete cholesky - tridiag 4x4", "[incomplete_cholesky]") {
    // Tridiagonal has no fill, so IC(0) == full Cholesky
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    REQUIRE(L_ic.size() == L_full.size());
    auto ic_vals = L_ic.values();
    auto full_vals = L_full.values();
    for (size_type i = 0; i < L_ic.size(); ++i) {
      CHECK(ic_vals[i] == Catch::Approx(full_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE(
    "incomplete cholesky - arrow 5x5 approximation", "[incomplete_cholesky]") {
    // Arrow matrix has fill in full Cholesky, so IC(0) != full Cholesky.
    // But L*L^T should approximate A well.
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    // IC(0) should have fewer nonzeros than full Cholesky (arrow has fill)
    CHECK(L_ic.size() <= L_full.size());

    // L*L^T should approximate A: check diagonal entries match reasonably
    auto Lt = transpose(L_ic);
    auto LLt = multiply(L_ic, Lt);
    for (size_type i = 0; i < 5; ++i) {
      CHECK(LLt(i, i) == Catch::Approx(A(i, i)).margin(0.5));
    }
  }

  TEST_CASE(
    "incomplete cholesky - L is lower triangular", "[incomplete_cholesky]") {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L = incomplete_cholesky(A);
    auto rp = L.row_ptr();
    auto ci = L.col_ind();

    for (size_type i = 0; i < L.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] <= i);
      }
    }
  }

  TEST_CASE(
    "incomplete cholesky - same nnz as lower triangle",
    "[incomplete_cholesky]") {
    // IC(0) factor should have exactly the same nnz as lower(A)
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L = incomplete_cholesky(A);
    auto lower_A = extract_lower_triangle(A, true);

    CHECK(L.size() == lower_A.size());
  }

  TEST_CASE(
    "incomplete cholesky - rectangular rejected", "[incomplete_cholesky]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    CHECK_THROWS_AS(incomplete_cholesky(A), std::invalid_argument);
  }

  // ================================================================
  // ic_apply tests
  // ================================================================

  TEST_CASE("ic_apply - diagonal", "[ic_apply]") {
    // A = diag(4, 9, 16), L = diag(2, 3, 4).
    // ic_apply(L, r) = L^{-T} L^{-1} r = diag(1/4, 1/9, 1/16) * r
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0}}};

    auto L = incomplete_cholesky(A);
    std::vector<double> r = {8.0, 27.0, 64.0};
    std::vector<double> z(3);

    ic_apply(L, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(2.0).margin(1e-12));
    CHECK(z[1] == Catch::Approx(3.0).margin(1e-12));
    CHECK(z[2] == Catch::Approx(4.0).margin(1e-12));
  }

  TEST_CASE("ic_apply - tridiag", "[ic_apply]") {
    // Tridiag: IC(0) == full Cholesky, so ic_apply gives exact A^{-1}r
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L = incomplete_cholesky(A);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> z(4);

    ic_apply(L, b.begin(), b.end(), z.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(x_true[i]).margin(1e-10));
    }
  }

  TEST_CASE("ic_apply - left-prec CG tridiag", "[ic_apply]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L = incomplete_cholesky(A);

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
    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

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
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ic_apply - left-prec CG grid", "[ic_apply]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto L = incomplete_cholesky(A);
    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      ic_apply(L, first, last, out);
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
  // mic0 tests
  // ================================================================

  TEST_CASE("mic0 - diagonal same as ic0", "[mic0]") {
    // Diagonal matrix: no fill to drop, so MIC(0) == IC(0)
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0},
       Entry<double>{Index{3, 3}, 25.0}}};

    auto L_mic = mic0(A);
    auto L_ic = incomplete_cholesky(A);

    REQUIRE(L_mic.size() == L_ic.size());
    auto mic_vals = L_mic.values();
    auto ic_vals = L_ic.values();
    for (size_type i = 0; i < L_mic.size(); ++i) {
      CHECK(mic_vals[i] == Catch::Approx(ic_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE("mic0 - tridiag same as ic0", "[mic0]") {
    // Tridiag: no fill, so MIC(0) == IC(0)
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);

    auto L_mic = mic0(A);
    auto L_ic = incomplete_cholesky(A);

    REQUIRE(L_mic.size() == L_ic.size());
    auto mic_vals = L_mic.values();
    auto ic_vals = L_ic.values();
    for (size_type i = 0; i < L_mic.size(); ++i) {
      CHECK(mic_vals[i] == Catch::Approx(ic_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE("mic0 - preserves row sums", "[mic0]") {
    // Arrow matrix: has fill, so MIC(0) compensates diagonal.
    // Key property: (L*L^T)*e ≈ A*e (row-sum preservation)
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L = mic0(A);
    auto Lt = transpose(L);
    auto LLt = multiply(L, Lt);

    // Compute A*e and (L*L^T)*e
    std::vector<double> ones(5, 1.0);
    auto Ae = multiply(A, std::span<double const>{ones});
    auto LLte = multiply(LLt, std::span<double const>{ones});

    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(LLte[i] == Catch::Approx(Ae[i]).margin(1e-10));
    }
  }

  TEST_CASE("mic0 - left-prec CG grid", "[mic0]") {
    auto A = poisson_2d<double>(4, 5.0);
    size_type const n = 16;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    auto L = mic0(A);
    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      ic_apply(L, first, last, out);
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

  TEST_CASE("mic0 - non-square throws", "[mic0]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    CHECK_THROWS_AS(mic0(A), std::invalid_argument);
  }

} // end of namespace sparkit::testing
