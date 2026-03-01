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
#include <sparkit/data/chebyshev_preconditioner.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conjugate_gradient.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::chebyshev_preconditioner;
  using sparkit::data::detail::chebyshev_preconditioner_apply;
  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::multiply;

  using size_type = sparkit::config::size_type;

  static auto const identity = [](auto first, auto last, auto out) {
    std::copy(first, last, out);
  };

  static Compressed_row_matrix<double>
  make_matrix(Shape shape, std::vector<Entry<double>> const& entries) {
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Compressed_row_sparsity sp{shape, indices.begin(), indices.end()};

    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    std::vector<double> vals(static_cast<std::size_t>(sp.size()), 0.0);

    for (auto const& e : entries) {
      auto row = e.index.row();
      auto col = e.index.column();
      for (auto p = rp[row]; p < rp[row + 1]; ++p) {
        if (ci[p] == col) {
          vals[static_cast<std::size_t>(p)] = e.value;
          break;
        }
      }
    }

    return Compressed_row_matrix<double>{std::move(sp), std::move(vals)};
  }

  static Compressed_row_matrix<double>
  make_tridiag_4() {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    return make_matrix(Shape{4, 4}, entries);
  }

  static Compressed_row_matrix<double>
  make_grid_16() {
    size_type const grid = 4;
    size_type const n = grid * grid;

    std::vector<Entry<double>> entries;
    for (size_type r = 0; r < grid; ++r) {
      for (size_type c = 0; c < grid; ++c) {
        auto node = r * grid + c;
        size_type degree = 0;
        if (c > 0) {
          entries.push_back(Entry<double>{Index{node, node - 1}, -1.0});
          ++degree;
        }
        if (c + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + 1}, -1.0});
          ++degree;
        }
        if (r > 0) {
          entries.push_back(Entry<double>{Index{node, node - grid}, -1.0});
          ++degree;
        }
        if (r + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + grid}, -1.0});
          ++degree;
        }
        entries.push_back(
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 5.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // ================================================================
  // Chebyshev preconditioner tests
  // ================================================================

  TEST_CASE("chebyshev_prec - identity matrix", "[chebyshev_preconditioner]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0},
       Entry<double>{Index{3, 3}, 1.0}});

    // lambda_min = lambda_max = 1, so f(lambda) = 1/lambda = 1
    // p(A) r should approximate r
    auto prec = chebyshev_preconditioner(A, 1.0, 1.0, 5);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    chebyshev_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(r[i]).epsilon(1e-6));
    }
  }

  TEST_CASE("chebyshev_prec - degree 0", "[chebyshev_preconditioner]") {
    auto A = make_tridiag_4();

    // Degree 0: p(lambda) = c_0, which is the average of 1/lambda over
    // the interval. z = c_0 * r (just scaling).
    double lmin = 2.0;
    double lmax = 6.0;
    auto prec = chebyshev_preconditioner(A, lmin, lmax, 0);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);
    chebyshev_preconditioner_apply(prec, r.begin(), r.end(), z.begin());

    // Degree 0: z should be c_0 * r, where c_0 is a scalar.
    // All elements scaled by the same factor.
    double scale = z[0] / r[0];
    CHECK(scale > 0.0);
    for (std::size_t i = 1; i < 4; ++i) {
      CHECK(z[i] == Catch::Approx(scale * r[i]).epsilon(1e-12));
    }
  }

  TEST_CASE(
    "chebyshev_prec - left-prec CG tridiag", "[chebyshev_preconditioner]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    // Eigenvalues of 4x4 tridiag(4,-1,-1) are in [2, 6] approximately
    auto prec = chebyshev_preconditioner(A, 2.0, 6.0, 5);
    auto apply_cheb = [&prec](auto first, auto last, auto out) {
      chebyshev_preconditioner_apply(prec, first, last, out);
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
      apply_cheb,
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

  TEST_CASE(
    "chebyshev_prec - left-prec CG grid", "[chebyshev_preconditioner]") {
    auto A = make_grid_16();
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

    // Grid Laplacian with +5 shift: eigenvalues in [3, 13] roughly
    auto prec = chebyshev_preconditioner(A, 3.0, 13.0, 5);
    auto apply_cheb = [&prec](auto first, auto last, auto out) {
      chebyshev_preconditioner_apply(prec, first, last, out);
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
      apply_cheb,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE(
    "chebyshev_prec - tighter bounds fewer iterations",
    "[chebyshev_preconditioner]") {
    auto A = make_grid_16();
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

    // Loose bounds
    auto prec_loose = chebyshev_preconditioner(A, 1.0, 20.0, 5);
    auto apply_loose = [&prec_loose](auto first, auto last, auto out) {
      chebyshev_preconditioner_apply(prec_loose, first, last, out);
    };
    std::vector<double> x_loose(static_cast<std::size_t>(n), 0.0);
    auto summary_loose = conjugate_gradient(
      b.begin(),
      b.end(),
      x_loose.begin(),
      x_loose.end(),
      cfg,
      apply_A,
      apply_loose,
      identity);

    // Tighter bounds
    auto prec_tight = chebyshev_preconditioner(A, 3.0, 13.0, 5);
    auto apply_tight = [&prec_tight](auto first, auto last, auto out) {
      chebyshev_preconditioner_apply(prec_tight, first, last, out);
    };
    std::vector<double> x_tight(static_cast<std::size_t>(n), 0.0);
    auto summary_tight = conjugate_gradient(
      b.begin(),
      b.end(),
      x_tight.begin(),
      x_tight.end(),
      cfg,
      apply_A,
      apply_tight,
      identity);

    REQUIRE(summary_loose.converged);
    REQUIRE(summary_tight.converged);
    CHECK(
      summary_tight.computed_iterations <= summary_loose.computed_iterations);
  }

} // end of namespace sparkit::testing
