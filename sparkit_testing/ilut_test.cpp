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
#include <sparkit/data/gmres.hpp>
#include <sparkit/data/ilu.hpp>
#include <sparkit/data/ilut.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Gmres_config;
  using sparkit::data::detail::Ilut_config;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::gmres;
  using sparkit::data::detail::ilu0;
  using sparkit::data::detail::ilu_apply;
  using sparkit::data::detail::ilut;
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

  static Compressed_row_matrix<double>
  make_nonsymmetric_4() {
    return make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{0, 2}, 0.5},
       Entry<double>{Index{1, 0}, -0.5},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 3}, -1.0},
       Entry<double>{Index{2, 0}, 0.3},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, -0.8},
       Entry<double>{Index{3, 1}, -0.6},
       Entry<double>{Index{3, 2}, 0.2},
       Entry<double>{Index{3, 3}, 4.0}});
  }

  static Compressed_row_matrix<double>
  make_convdiff_16() {
    size_type const grid = 4;
    size_type const n = grid * grid;
    double const convection = 0.3;

    std::vector<Entry<double>> entries;
    for (size_type r = 0; r < grid; ++r) {
      for (size_type c = 0; c < grid; ++c) {
        auto node = r * grid + c;
        size_type degree = 0;
        if (c > 0) {
          entries.push_back(
            Entry<double>{Index{node, node - 1}, -1.0 - convection});
          ++degree;
        }
        if (c + 1 < grid) {
          entries.push_back(
            Entry<double>{Index{node, node + 1}, -1.0 + convection});
          ++degree;
        }
        if (r > 0) {
          entries.push_back(
            Entry<double>{Index{node, node - grid}, -1.0 - convection});
          ++degree;
        }
        if (r + 1 < grid) {
          entries.push_back(
            Entry<double>{Index{node, node + grid}, -1.0 + convection});
          ++degree;
        }
        entries.push_back(
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 5.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // ================================================================
  // ILUT tests
  // ================================================================

  TEST_CASE("ilut - diagonal", "[ilut]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    Ilut_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto factors = ilut(A, cfg);

    for (size_type i = 0; i < 4; ++i) {
      CHECK(factors.L(i, i) == Catch::Approx(1.0).margin(1e-14));
    }

    CHECK(factors.U(0, 0) == Catch::Approx(2.0).margin(1e-14));
    CHECK(factors.U(1, 1) == Catch::Approx(3.0).margin(1e-14));
    CHECK(factors.U(2, 2) == Catch::Approx(4.0).margin(1e-14));
    CHECK(factors.U(3, 3) == Catch::Approx(5.0).margin(1e-14));
  }

  TEST_CASE("ilut - tridiag exact LU", "[ilut]") {
    auto A = make_tridiag_4();

    Ilut_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto factors = ilut(A, cfg);

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

  TEST_CASE("ilut - apply to vector", "[ilut]") {
    auto A = make_tridiag_4();

    Ilut_config<double> cfg{.drop_tolerance = 0.0, .fill_limit = 10};
    auto factors = ilut(A, cfg);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(4, 0.0);
    ilu_apply(factors, b.begin(), b.end(), x.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-12));
    }
  }

  TEST_CASE("ilut - large tau drops aggressively", "[ilut]") {
    auto A = make_grid_16();

    // Very small tau: keep everything
    Ilut_config<double> cfg_small{.drop_tolerance = 0.0, .fill_limit = 100};
    auto factors_small = ilut(A, cfg_small);
    auto nnz_small = factors_small.L.size() + factors_small.U.size();

    // Large tau: drop aggressively
    Ilut_config<double> cfg_large{.drop_tolerance = 0.5, .fill_limit = 100};
    auto factors_large = ilut(A, cfg_large);
    auto nnz_large = factors_large.L.size() + factors_large.U.size();

    CHECK(nnz_large <= nnz_small);
  }

  TEST_CASE("ilut - small tau preserves more fill", "[ilut]") {
    auto A = make_grid_16();

    // Small tau: preserve more
    Ilut_config<double> cfg_small{.drop_tolerance = 1e-6, .fill_limit = 100};
    auto factors_small = ilut(A, cfg_small);
    auto nnz_small = factors_small.L.size() + factors_small.U.size();

    // Large tau: drop more
    Ilut_config<double> cfg_large{.drop_tolerance = 0.1, .fill_limit = 100};
    auto factors_large = ilut(A, cfg_large);
    auto nnz_large = factors_large.L.size() + factors_large.U.size();

    CHECK(nnz_small >= nnz_large);
  }

  TEST_CASE("ilut - left-prec GMRES tridiag", "[ilut]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    Ilut_config<double> ilut_cfg{.drop_tolerance = 1e-4, .fill_limit = 10};
    auto factors = ilut(A, ilut_cfg);
    auto apply_ilut = [&factors](auto first, auto last, auto out) {
      ilu_apply(factors, first, last, out);
    };

    Gmres_config<double> cfg{
      .tolerance = 1e-12, .restart_dimension = 10, .max_iterations = 100};

    std::vector<double> x(4, 0.0);
    auto summary = gmres(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cfg,
      apply_A,
      apply_ilut,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ilut - left-prec GMRES grid", "[ilut]") {
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

    Ilut_config<double> ilut_cfg{.drop_tolerance = 1e-4, .fill_limit = 10};
    auto factors = ilut(A, ilut_cfg);
    auto apply_ilut = [&factors](auto first, auto last, auto out) {
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
      apply_ilut,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ilut - left-prec GMRES nonsymmetric", "[ilut]") {
    auto A = make_nonsymmetric_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    Ilut_config<double> ilut_cfg{.drop_tolerance = 1e-4, .fill_limit = 10};
    auto factors = ilut(A, ilut_cfg);
    auto apply_ilut = [&factors](auto first, auto last, auto out) {
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
      apply_ilut,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ilut - left-prec GMRES convdiff", "[ilut]") {
    auto A = make_convdiff_16();
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

    Ilut_config<double> ilut_cfg{.drop_tolerance = 1e-4, .fill_limit = 10};
    auto factors = ilut(A, ilut_cfg);
    auto apply_ilut = [&factors](auto first, auto last, auto out) {
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
      apply_ilut,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ilut - fewer iters than ilu0", "[ilut]") {
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

    Gmres_config<double> cfg{
      .tolerance = 1e-10, .restart_dimension = 20, .max_iterations = 200};

    // ILUT with no dropping (approaches exact LU)
    Ilut_config<double> ilut_cfg{.drop_tolerance = 0.0, .fill_limit = 100};
    auto factors_ilut = ilut(A, ilut_cfg);
    auto apply_ilut = [&factors_ilut](auto first, auto last, auto out) {
      ilu_apply(factors_ilut, first, last, out);
    };

    std::vector<double> x_ilut(static_cast<std::size_t>(n), 0.0);
    auto summary_ilut = gmres(
      b.begin(),
      b.end(),
      x_ilut.begin(),
      x_ilut.end(),
      cfg,
      apply_A,
      apply_ilut,
      identity);

    // ILU(0)
    auto factors_ilu0 = ilu0(A);
    auto apply_ilu0 = [&factors_ilu0](auto first, auto last, auto out) {
      ilu_apply(factors_ilu0, first, last, out);
    };

    std::vector<double> x_ilu0(static_cast<std::size_t>(n), 0.0);
    auto summary_ilu0 = gmres(
      b.begin(),
      b.end(),
      x_ilu0.begin(),
      x_ilu0.end(),
      cfg,
      apply_A,
      apply_ilu0,
      identity);

    REQUIRE(summary_ilut.converged);
    REQUIRE(summary_ilu0.converged);
    CHECK(summary_ilut.computed_iterations <= summary_ilu0.computed_iterations);
  }

} // end of namespace sparkit::testing
