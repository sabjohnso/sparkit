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
#include <sparkit/data/amg_solver.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/richardson.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Amg_solver_config;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::amg_solve;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::richardson;
  using sparkit::data::detail::Richardson_config;

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
  make_tridiag(size_type n) {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < n; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < n) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  static Compressed_row_matrix<double>
  make_grid(size_type grid) {
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
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 1.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // ================================================================
  // Standalone AMG solver tests
  // ================================================================

  TEST_CASE(
    "amg solver - diagonal system converges in 1 iteration", "[amg_solver]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    std::vector<double> b = {4.0, 9.0, 16.0, 25.0};
    std::vector<double> x(4, 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.max_levels = 1;
    cfg.tolerance = 1e-10;
    cfg.max_iterations = 100;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    REQUIRE(summary.converged);
    CHECK(summary.computed_iterations == 1);
    CHECK(x[0] == Catch::Approx(2.0).margin(1e-10));
    CHECK(x[1] == Catch::Approx(3.0).margin(1e-10));
    CHECK(x[2] == Catch::Approx(4.0).margin(1e-10));
    CHECK(x[3] == Catch::Approx(5.0).margin(1e-10));
  }

  TEST_CASE("amg solver - tridiag 4x4 converges", "[amg_solver]") {
    size_type const n = 4;
    auto A = make_tridiag(n);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.max_levels = 1;
    cfg.tolerance = 1e-10;
    cfg.max_iterations = 200;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("amg solver - 4x4 grid converges", "[amg_solver]") {
    size_type const grid = 4;
    size_type const n = grid * grid;
    auto A = make_grid(grid);

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.coarsest_size = 4;
    cfg.tolerance = 1e-10;
    cfg.max_iterations = 500;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-4));
    }
  }

  TEST_CASE("amg solver - zero rhs gives zero solution", "[amg_solver]") {
    auto A = make_tridiag(4);

    std::vector<double> b(4, 0.0);
    std::vector<double> x(4, 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.max_levels = 1;
    cfg.tolerance = 1e-10;
    cfg.max_iterations = 100;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-14));
    }
  }

  TEST_CASE("amg solver - stops at max iterations", "[amg_solver]") {
    size_type const grid = 4;
    size_type const n = grid * grid;
    auto A = make_grid(grid);

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.coarsest_size = 4;
    cfg.tolerance = 1e-15;
    cfg.max_iterations = 3;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    CHECK_FALSE(summary.converged);
    CHECK(summary.computed_iterations == 3);
  }

  TEST_CASE(
    "amg solver - residual collection shows decreasing trend", "[amg_solver]") {
    size_type const grid = 4;
    size_type const n = grid * grid;
    auto A = make_grid(grid);

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    Amg_solver_config<double> cfg;
    cfg.amg.coarsest_size = 4;
    cfg.tolerance = 1e-10;
    cfg.max_iterations = 500;
    cfg.collect_residuals = true;

    auto summary = amg_solve(b.begin(), b.end(), x.begin(), x.end(), cfg, A);

    REQUIRE(summary.converged);
    REQUIRE(summary.iteration_residuals.size() > 1);

    // Residuals should generally decrease
    auto const& res = summary.iteration_residuals;
    CHECK(res.back() < res.front());
  }

  TEST_CASE(
    "amg solver - converges in fewer iterations than Richardson",
    "[amg_solver]") {
    size_type const grid = 4;
    size_type const n = grid * grid;
    auto A = make_grid(grid);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    // Unpreconditioned Richardson (damped for stability)
    std::vector<double> x_rich(static_cast<std::size_t>(n), 0.0);
    Richardson_config<double> rich_cfg;
    rich_cfg.tolerance = 1e-10;
    rich_cfg.max_iterations = 5000;
    rich_cfg.omega = 0.2;

    auto summary_rich = richardson(
      b.begin(),
      b.end(),
      x_rich.begin(),
      x_rich.end(),
      rich_cfg,
      apply_A,
      identity,
      identity);

    // Standalone AMG solver
    std::vector<double> x_amg(static_cast<std::size_t>(n), 0.0);
    Amg_solver_config<double> amg_cfg;
    amg_cfg.amg.coarsest_size = 4;
    amg_cfg.tolerance = 1e-10;
    amg_cfg.max_iterations = 5000;

    auto summary_amg =
      amg_solve(b.begin(), b.end(), x_amg.begin(), x_amg.end(), amg_cfg, A);

    REQUIRE(summary_rich.converged);
    REQUIRE(summary_amg.converged);
    CHECK(summary_amg.computed_iterations < summary_rich.computed_iterations);
  }

} // end of namespace sparkit::testing
