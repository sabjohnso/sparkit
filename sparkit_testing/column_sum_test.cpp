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
#include <sparkit/data/column_sum.hpp>
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

  using sparkit::data::detail::column_sum;
  using sparkit::data::detail::column_sum_apply;
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
  // Column sum preconditioner tests
  // ================================================================

  TEST_CASE("column_sum - values on 4x4 matrix", "[column_sum]") {
    // Matrix:
    //   [2  1  0  0]
    //   [1  3  1  0]
    //   [0  1  4  1]
    //   [0  0  1  5]
    //
    // Column sums (absolute): col0=3, col1=5, col2=6, col3=6
    // Inverse column sums:    1/3, 1/5, 1/6, 1/6

    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 1.0},
       Entry<double>{Index{2, 1}, 1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, 1.0},
       Entry<double>{Index{3, 2}, 1.0},
       Entry<double>{Index{3, 3}, 5.0}});

    auto inv_cs = column_sum(A);

    REQUIRE(inv_cs.size() == 4);
    CHECK(inv_cs[0] == Catch::Approx(1.0 / 3.0));
    CHECK(inv_cs[1] == Catch::Approx(1.0 / 5.0));
    CHECK(inv_cs[2] == Catch::Approx(1.0 / 6.0));
    CHECK(inv_cs[3] == Catch::Approx(1.0 / 6.0));
  }

  TEST_CASE("column_sum - apply to vector", "[column_sum]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 5.0},
       Entry<double>{Index{3, 3}, 10.0}});

    auto inv_cs = column_sum(A);

    std::vector<double> r = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> z(4, 0.0);
    column_sum_apply(inv_cs, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(3.0));
    CHECK(z[1] == Catch::Approx(3.0));
    CHECK(z[2] == Catch::Approx(4.0));
    CHECK(z[3] == Catch::Approx(3.0));
  }

  TEST_CASE("column_sum - zero column sum throws", "[column_sum]") {
    // Column 1 has no entries => column sum is zero
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 2}, 1.0},
       Entry<double>{Index{2, 0}, 1.0},
       Entry<double>{Index{2, 2}, 3.0}});

    CHECK_THROWS_AS(column_sum(A), std::invalid_argument);
  }

  TEST_CASE("column_sum - left-prec CG tridiag", "[column_sum]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto inv_cs = column_sum(A);
    auto apply_cs = [&inv_cs](auto first, auto last, auto out) {
      column_sum_apply(inv_cs, first, last, out);
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
      apply_cs,
      identity);

    REQUIRE(summary_pcg.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x_pcg[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("column_sum - left-prec CG grid", "[column_sum]") {
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

    auto inv_cs = column_sum(A);
    auto apply_cs = [&inv_cs](auto first, auto last, auto out) {
      column_sum_apply(inv_cs, first, last, out);
    };

    std::vector<double> x_pcg(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary_pcg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_pcg.begin(),
      x_pcg.end(),
      cfg,
      apply_A,
      apply_cs,
      identity);

    REQUIRE(summary_pcg.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x_pcg[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

} // end of namespace sparkit::testing
