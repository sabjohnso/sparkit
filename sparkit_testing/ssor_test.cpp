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
#include <sparkit/data/jacobi.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/ssor.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::CGConfig;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::conjugate_gradient;
  using sparkit::data::detail::jacobi;
  using sparkit::data::detail::jacobi_apply;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::ssor;
  using sparkit::data::detail::ssor_apply;

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
  // SSOR preconditioner tests
  // ================================================================

  TEST_CASE("ssor - diagonal omega=1 matches Jacobi", "[ssor]") {
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 5.0},
       Entry<double>{Index{3, 3}, 10.0}});

    auto sf = ssor(A, 1.0);
    auto inv_d = jacobi(A);

    std::vector<double> r = {6.0, 12.0, 20.0, 30.0};
    std::vector<double> z_ssor(4, 0.0);
    std::vector<double> z_jac(4, 0.0);

    ssor_apply(A, sf, r.begin(), r.end(), z_ssor.begin());
    jacobi_apply(inv_d, r.begin(), r.end(), z_jac.begin());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z_ssor[i] == Catch::Approx(z_jac[i]).margin(1e-12));
    }
  }

  TEST_CASE("ssor - tridiag omega=1", "[ssor]") {
    // A = [[4,-1,0,0],[-1,4,-1,0],[0,-1,4,-1],[0,0,-1,4]]
    // omega=1 (SGS)
    // r = [1, 0, 0, 0]
    //
    // Forward sweep: solve (D + L) y = r
    //   y[0] = 1/4 = 0.25
    //   y[1] = (0 - (-1)*0.25)/4 = 0.25/4 = 0.0625
    //   y[2] = (0 - (-1)*0.0625)/4 = 0.015625
    //   y[3] = (0 - (-1)*0.015625)/4 = 0.00390625
    //
    // Diagonal scale: z_mid[i] = d[i] * y[i]
    //   z_mid = [1, 0.25, 0.0625, 0.015625]
    //
    // Backward sweep: solve (D + U) w = z_mid
    //   w[3] = 0.015625/4 = 0.00390625
    //   w[2] = (0.0625 - (-1)*0.00390625)/4 = 0.06640625/4 = 0.016601...
    //   w[1] = (0.25 - (-1)*0.016601...)/4 = 0.266601.../4 = 0.06665039...
    //   w[0] = (1 - (-1)*0.06665039...)/4 = 1.06665039.../4 = 0.26666259...
    //
    // Final scale: z[i] = 1*(2-1) * w[i] = w[i]

    auto A = make_tridiag_4();
    auto sf = ssor(A, 1.0);

    std::vector<double> r = {1.0, 0.0, 0.0, 0.0};
    std::vector<double> z(4, 0.0);
    ssor_apply(A, sf, r.begin(), r.end(), z.begin());

    CHECK(z[0] == Catch::Approx(0.26666259765625).margin(1e-12));
    CHECK(z[1] == Catch::Approx(0.0666503906250).margin(1e-12));
    CHECK(z[2] == Catch::Approx(0.01660156250).margin(1e-12));
    CHECK(z[3] == Catch::Approx(0.00390625).margin(1e-12));
  }

  TEST_CASE("ssor - apply correctness", "[ssor]") {
    // Verify SSOR on tridiag with a different RHS
    auto A = make_tridiag_4();
    auto sf = ssor(A, 1.0);

    std::vector<double> r = {4.0, 4.0, 4.0, 4.0};
    std::vector<double> z(4, 0.0);
    ssor_apply(A, sf, r.begin(), r.end(), z.begin());

    // All entries should be positive for a positive RHS with SPD matrix
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] > 0.0);
    }

    // SSOR should produce a result closer to A^{-1}r than Jacobi
    // Just check it's reasonable (not zero, not huge)
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(z[i] < 2.0);
      CHECK(z[i] > 0.5);
    }
  }

  TEST_CASE("ssor - left-prec CG tridiag omega=1", "[ssor]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto sf = ssor(A, 1.0);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
    };

    CGConfig<double> cfg{
      .tolerance = 1e-12, .restart_iterations = 50, .max_iterations = 100};

    std::vector<double> x_pcg(4, 0.0);
    auto summary_pcg = conjugate_gradient(
      b.begin(),
      b.end(),
      x_pcg.begin(),
      x_pcg.end(),
      cfg,
      apply_A,
      apply_ssor,
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

  TEST_CASE("ssor - left-prec CG grid omega=1", "[ssor]") {
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

    auto sf = ssor(A, 1.0);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
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
      apply_ssor,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ssor - right-prec CG tridiag omega=1", "[ssor]") {
    auto A = make_tridiag_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    auto sf = ssor(A, 1.0);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
    };

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
      identity,
      apply_ssor);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-8));
    }
  }

  TEST_CASE("ssor - right-prec CG grid omega=1", "[ssor]") {
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

    auto sf = ssor(A, 1.0);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
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
      identity,
      apply_ssor);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ssor - omega=0.5 converges", "[ssor]") {
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

    auto sf = ssor(A, 0.5);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
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
      apply_ssor,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ssor - omega=1.5 converges", "[ssor]") {
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

    auto sf = ssor(A, 1.5);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
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
      apply_ssor,
      identity);

    REQUIRE(summary.converged);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-6));
    }
  }

  TEST_CASE("ssor - fewer iters than Jacobi on grid", "[ssor]") {
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

    // SSOR (omega=1)
    auto sf = ssor(A, 1.0);
    auto apply_ssor = [&A, &sf](auto first, auto last, auto out) {
      ssor_apply(A, sf, first, last, out);
    };

    std::vector<double> x_ssor(static_cast<std::size_t>(n), 0.0);
    auto summary_ssor = conjugate_gradient(
      b.begin(),
      b.end(),
      x_ssor.begin(),
      x_ssor.end(),
      cfg,
      apply_A,
      apply_ssor,
      identity);

    // Jacobi
    auto inv_d = jacobi(A);
    auto apply_jacobi = [&inv_d](auto first, auto last, auto out) {
      jacobi_apply(inv_d, first, last, out);
    };

    std::vector<double> x_jac(static_cast<std::size_t>(n), 0.0);
    auto summary_jac = conjugate_gradient(
      b.begin(),
      b.end(),
      x_jac.begin(),
      x_jac.end(),
      cfg,
      apply_A,
      apply_jacobi,
      identity);

    REQUIRE(summary_ssor.converged);
    REQUIRE(summary_jac.converged);
    CHECK(summary_ssor.computed_iterations <= summary_jac.computed_iterations);
  }

} // end of namespace sparkit::testing
