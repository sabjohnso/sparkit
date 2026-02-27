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
#include <sparkit/data/eigen_target.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/svd.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Eigen_target;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Svd_config;

  using sparkit::data::detail::multiply;
  using sparkit::data::detail::svd;

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

  // Build a diagonal matrix.
  static Compressed_row_matrix<double>
  make_diagonal(std::vector<double> const& diag) {
    auto n = static_cast<size_type>(diag.size());
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < n; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, diag[i]});
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // Build a rectangular matrix for SVD testing.
  // 3x5 matrix with known structure.
  static Compressed_row_matrix<double>
  make_rect_3x5() {
    return make_matrix(
      Shape{3, 5},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{0, 4}, 1.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 1.0},
       Entry<double>{Index{2, 0}, 2.0},
       Entry<double>{Index{2, 3}, 4.0}});
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
        entries.push_back(
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 5.0});
      }
    }
    return make_matrix(Shape{n, n}, entries);
  }

  // Transpose a CSR matrix (for A^T operator).
  static Compressed_row_matrix<double>
  transpose_matrix(Compressed_row_matrix<double> const& A) {
    auto shape = A.sparsity().shape();
    auto m = shape.row();
    auto n = shape.column();
    auto rp = A.sparsity().row_ptr();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();

    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        entries.push_back(Entry<double>{Index{j, i}, vals[p]});
      }
    }
    return make_matrix(Shape{n, m}, entries);
  }

  // ================================================================
  // Diagonal matrix: singular values = |diagonal entries|
  // ================================================================

  TEST_CASE("svd - diagonal 4x4, largest 2", "[svd]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A; // Symmetric diagonal

    Svd_config<double> cfg{
      .num_singular_values = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = svd(4, 4, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    REQUIRE(result.singular_values.size() == 2);

    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    CHECK(svals[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(svals[1] == Catch::Approx(4.0).margin(1e-8));
  }

  // ================================================================
  // Rectangular matrix
  // ================================================================

  TEST_CASE("svd - rectangular 3x5", "[svd]") {
    auto A = make_rect_3x5();
    auto At = transpose_matrix(A);

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = [&At](auto first, auto last, auto out) {
      auto result = multiply(At, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Svd_config<double> cfg{
      .num_singular_values = 2,
      .krylov_dimension = 3,
      .tolerance = 1e-10,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = svd(3, 5, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    REQUIRE(result.singular_values.size() == 2);

    // Check that singular values are positive.
    for (auto s : result.singular_values) {
      CHECK(s > 0.0);
    }

    // Check left/right vector dimensions.
    for (std::size_t i = 0; i < result.singular_values.size(); ++i) {
      REQUIRE(result.left_singular_vectors[i].size() == 3);
      REQUIRE(result.right_singular_vectors[i].size() == 5);
    }
  }

  // ================================================================
  // Grid test (requires restart)
  // ================================================================

  TEST_CASE("svd - grid 16 largest 3", "[svd]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A; // Symmetric

    Svd_config<double> cfg{
      .num_singular_values = 3,
      .krylov_dimension = 10,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = true};

    auto result = svd(16, 16, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    REQUIRE(result.singular_values.size() == 3);
    REQUIRE(result.left_singular_vectors.size() == 3);
    REQUIRE(result.right_singular_vectors.size() == 3);
  }

  // ================================================================
  // Singular vector residual: ||Av - σu|| / σ < tol
  // ================================================================

  TEST_CASE("svd - singular vector residual", "[svd]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A; // Symmetric

    Svd_config<double> cfg{
      .num_singular_values = 3,
      .krylov_dimension = 10,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = svd(16, 16, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    for (std::size_t k = 0; k < result.singular_values.size(); ++k) {
      auto const& v = result.right_singular_vectors[k];
      auto av = multiply(A, std::span<double const>{v});

      double sigma = result.singular_values[k];
      auto const& u = result.left_singular_vectors[k];

      double residual_sq = 0.0;
      for (std::size_t i = 0; i < u.size(); ++i) {
        double diff = av[i] - sigma * u[i];
        residual_sq += diff * diff;
      }
      double rel_residual = std::sqrt(residual_sq) / sigma;
      CHECK(rel_residual < 1e-8);
    }
  }

  // ================================================================
  // Frobenius norm property: sum(sigma_i^2) <= ||A||_F^2
  // For a truncated SVD with all singular values,
  // sum(sigma_i^2) = ||A||_F^2.
  // ================================================================

  TEST_CASE("svd - Frobenius norm property", "[svd]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A;

    // Request all 4 singular values.
    Svd_config<double> cfg{
      .num_singular_values = 4,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = svd(4, 4, cfg, apply_A, apply_At);

    REQUIRE(result.converged);

    double sigma_sq = 0.0;
    for (auto s : result.singular_values) {
      sigma_sq += s * s;
    }

    // ||A||_F^2 = 1 + 16 + 4 + 9 = 30
    CHECK(sigma_sq == Catch::Approx(30.0).margin(1e-6));
  }

  // ================================================================
  // Smallest singular values
  // ================================================================

  TEST_CASE("svd - smallest singular values", "[svd]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A;

    Svd_config<double> cfg{
      .num_singular_values = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::smallest_magnitude,
      .collect_residuals = false};

    auto result = svd(4, 4, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    REQUIRE(result.singular_values.size() == 2);

    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    CHECK(svals[0] == Catch::Approx(1.0).margin(1e-8));
    CHECK(svals[1] == Catch::Approx(2.0).margin(1e-8));
  }

  // ================================================================
  // Residual norms reported
  // ================================================================

  TEST_CASE("svd - residual norms reported", "[svd]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };
    auto apply_At = apply_A;

    Svd_config<double> cfg{
      .num_singular_values = 2,
      .krylov_dimension = 8,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = svd(16, 16, cfg, apply_A, apply_At);

    REQUIRE(result.converged);
    REQUIRE(result.residual_norms.size() == result.singular_values.size());
    for (auto rn : result.residual_norms) {
      CHECK(rn < 1e-8);
    }
  }

} // end of namespace sparkit::testing
