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
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/triangular_solve.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::CGSummary;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::preconditioned_conjugate_gradient;

  using size_type = sparkit::config::size_type;

  // Build a CSR matrix from a list of (row, col, value) entries.
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

  // Build a 4x4 tridiagonal SPD matrix: diag=4, off-diag=-1.
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

  // Build a 4x4 grid Laplacian + 5*I (16 nodes), SPD.
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
        entries.push_back(Entry<double>{Index{node, node},
                                        static_cast<double>(degree) + 5.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // ================================================================
  // Unpreconditioned CG tests
  // ================================================================

  TEST_CASE("conjugate gradient - identity", "[conjugate_gradient]") {
    auto A = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 1.0},
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

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    REQUIRE(summary.converged);
    CHECK(summary.computed_iterations <= 1);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(b[i]).margin(1e-10));
    }
  }

  TEST_CASE("conjugate gradient - diagonal", "[conjugate_gradient]") {
    auto A = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 2.0},
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

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    REQUIRE(summary.converged);
    CHECK(x[0] == Catch::Approx(3.0).margin(1e-10));
    CHECK(x[1] == Catch::Approx(4.0).margin(1e-10));
    CHECK(x[2] == Catch::Approx(5.0).margin(1e-10));
    CHECK(x[3] == Catch::Approx(6.0).margin(1e-10));
  }

  TEST_CASE("conjugate gradient - tridiag 4x4", "[conjugate_gradient]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    std::vector<double> x(4, 0.0);
    CGConfig<double> cfg{
        .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("conjugate gradient - grid 16-node", "[conjugate_gradient]") {
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
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
        .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Preconditioned CG tests
  // ================================================================

  TEST_CASE("preconditioned CG - tridiag", "[conjugate_gradient]") {
    auto A = make_tridiag_4();

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

    auto summary_pcg = preconditioned_conjugate_gradient(
        b.begin(), b.end(), x_pcg.begin(), x_pcg.end(), cfg, apply_A,
        apply_inv_M);

    std::vector<double> x_cg(4, 0.0);
    auto summary_cg = conjugate_gradient(b.begin(), b.end(), x_cg.begin(),
                                         x_cg.end(), cfg, apply_A);

    REQUIRE(summary_pcg.converged);
    CHECK(summary_pcg.computed_iterations <= summary_cg.computed_iterations);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_pcg[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("preconditioned CG - grid", "[conjugate_gradient]") {
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

    auto L = incomplete_cholesky(A);
    auto apply_inv_M = [&L](auto first, auto last, auto out) {
      auto y = forward_solve(L, std::span<double const>{first, last});
      auto z = forward_solve_transpose(L, std::span<double const>{y});
      std::copy(z.begin(), z.end(), out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
        .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary = preconditioned_conjugate_gradient(
        b.begin(), b.end(), x.begin(), x.end(), cfg, apply_A, apply_inv_M);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("conjugate gradient - zero rhs", "[conjugate_gradient]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> b = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> x(4, 0.0);
    CGConfig<double> cfg{
        .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-14));
    }
  }

  TEST_CASE("conjugate gradient - max iterations", "[conjugate_gradient]") {
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
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
        .tolerance = 1e-14, .restart_iterations = 50, .max_iterations = 1};

    auto summary = conjugate_gradient(b.begin(), b.end(), x.begin(), x.end(),
                                      cfg, apply_A);

    CHECK_FALSE(summary.converged);
    CHECK(summary.computed_iterations == 1);
  }

} // end of namespace sparkit::testing
