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
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Ict_config;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::ic_apply;
  using sparkit::data::detail::ict;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::transpose;

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

  static Compressed_row_matrix<double>
  make_arrow_5() {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    return make_matrix(Shape{5, 5}, entries);
  }

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
    auto A = make_tridiag_4();
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
    auto A = make_tridiag_4();
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
    auto A = make_arrow_5();
    Ict_config<double> cfg_generous{.drop_tolerance = 0.0, .fill_limit = 10};
    Ict_config<double> cfg_aggressive{.drop_tolerance = 0.5, .fill_limit = 10};

    auto L_generous = ict(A, cfg_generous);
    auto L_aggressive = ict(A, cfg_aggressive);

    CHECK(L_aggressive.size() <= L_generous.size());
  }

  TEST_CASE("ict - small tau preserves more fill", "[ict]") {
    auto A = make_grid_16();
    Ict_config<double> cfg_small{.drop_tolerance = 1e-4, .fill_limit = 20};
    Ict_config<double> cfg_large{.drop_tolerance = 0.1, .fill_limit = 20};

    auto L_small = ict(A, cfg_small);
    auto L_large = ict(A, cfg_large);

    CHECK(L_small.size() >= L_large.size());
  }

  TEST_CASE("ict - left-prec CG tridiag", "[ict]") {
    auto A = make_tridiag_4();
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
    auto A = make_grid_16();
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
    auto A = make_arrow_5();
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
