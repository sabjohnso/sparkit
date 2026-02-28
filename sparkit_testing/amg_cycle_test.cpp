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
#include <sparkit/data/amg_cycle.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conjugate_gradient.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Amg_config;
  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::amg_apply;
  using sparkit::data::detail::amg_setup;
  using sparkit::data::detail::amg_vcycle;
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
  // V-cycle tests
  // ================================================================

  TEST_CASE("amg vcycle - diagonal system exact", "[amg_cycle]") {
    // Diagonal matrix: AMG should solve exactly in one cycle.
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 5.0}});

    Amg_config<double> cfg;
    cfg.max_levels = 1;

    auto h = amg_setup(A, cfg);

    std::vector<double> rhs = {4.0, 9.0, 16.0, 25.0};
    std::vector<double> x(4, 0.0);

    amg_vcycle(h, 0, std::span<double const>{rhs}, std::span<double>{x});

    CHECK(x[0] == Catch::Approx(2.0).margin(1e-10));
    CHECK(x[1] == Catch::Approx(3.0).margin(1e-10));
    CHECK(x[2] == Catch::Approx(4.0).margin(1e-10));
    CHECK(x[3] == Catch::Approx(5.0).margin(1e-10));
  }

  TEST_CASE("amg vcycle - reduces residual", "[amg_cycle]") {
    auto A = make_tridiag(8);
    size_type const n = 8;

    Amg_config<double> cfg;
    cfg.coarsest_size = 4;

    auto h = amg_setup(A, cfg);

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    // Compute initial residual norm
    auto r0 = multiply(A, std::span<double const>{x});
    double norm0 = 0.0;
    for (size_type i = 0; i < n; ++i) {
      auto ri =
        b[static_cast<std::size_t>(i)] - r0[static_cast<std::size_t>(i)];
      norm0 += ri * ri;
    }
    norm0 = std::sqrt(norm0);

    // Apply one V-cycle
    amg_vcycle(h, 0, std::span<double const>{b}, std::span<double>{x});

    // Compute residual after one cycle
    auto r1 = multiply(A, std::span<double const>{x});
    double norm1 = 0.0;
    for (size_type i = 0; i < n; ++i) {
      auto ri =
        b[static_cast<std::size_t>(i)] - r1[static_cast<std::size_t>(i)];
      norm1 += ri * ri;
    }
    norm1 = std::sqrt(norm1);

    CHECK(norm1 < norm0);
  }

  // ================================================================
  // AMG-preconditioned CG tests
  // ================================================================

  TEST_CASE("amg + CG - tridiag system", "[amg_cycle]") {
    size_type const n = 16;
    auto A = make_tridiag(n);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    Amg_config<double> amg_cfg;
    amg_cfg.coarsest_size = 4;
    auto h = amg_setup(A, amg_cfg);

    auto apply_amg = [&h](auto first, auto last, auto out) {
      amg_apply(h, first, last, out);
    };

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
      apply_amg,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("amg + CG - 4x4 grid system", "[amg_cycle]") {
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

    Amg_config<double> amg_cfg;
    amg_cfg.coarsest_size = 4;
    auto h = amg_setup(A, amg_cfg);

    auto apply_amg = [&h](auto first, auto last, auto out) {
      amg_apply(h, first, last, out);
    };

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
      apply_amg,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("amg + CG - 8x8 grid system", "[amg_cycle]") {
    size_type const grid = 8;
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

    Amg_config<double> amg_cfg;
    amg_cfg.coarsest_size = 8;
    auto h = amg_setup(A, amg_cfg);

    auto apply_amg = [&h](auto first, auto last, auto out) {
      amg_apply(h, first, last, out);
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-10, .restart_iterations = 100, .max_iterations = 500};

    auto summary = conjugate_gradient(
      b.begin(),
      b.end(),
      x.begin(),
      x.end(),
      cg_cfg,
      apply_A,
      apply_amg,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-4));
    }
  }

  TEST_CASE("amg + CG improves over unpreconditioned CG", "[amg_cycle]") {
    size_type const n = 16;
    auto A = make_tridiag(n);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    // Unpreconditioned CG
    std::vector<double> x_cg(static_cast<std::size_t>(n), 0.0);
    CGConfig<double> cg_cfg{
      .tolerance = 1e-10, .restart_iterations = 50, .max_iterations = 200};

    auto summary_cg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_cg.begin(),
      x_cg.end(),
      cg_cfg,
      apply_A,
      identity,
      identity);

    // AMG-preconditioned CG
    Amg_config<double> amg_cfg;
    amg_cfg.coarsest_size = 4;
    auto h = amg_setup(A, amg_cfg);

    auto apply_amg = [&h](auto first, auto last, auto out) {
      amg_apply(h, first, last, out);
    };

    std::vector<double> x_amg(static_cast<std::size_t>(n), 0.0);
    auto summary_amg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_amg.begin(),
      x_amg.end(),
      cg_cfg,
      apply_A,
      apply_amg,
      identity);

    REQUIRE(summary_cg.converged);
    REQUIRE(summary_amg.converged);
    CHECK(summary_amg.computed_iterations <= summary_cg.computed_iterations);
  }

} // end of namespace sparkit::testing
