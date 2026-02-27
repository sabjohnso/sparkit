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
#include <sparkit/data/lanczos.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Eigen_target;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Lanczos_config;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::lanczos;
  using sparkit::data::detail::multiply;

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

  // Build a symmetric indefinite matrix.
  static Compressed_row_matrix<double>
  make_indefinite_4() {
    return make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, -1.0},
       Entry<double>{Index{1, 2}, -1.0},
       Entry<double>{Index{2, 1}, -1.0},
       Entry<double>{Index{2, 2}, 3.0},
       Entry<double>{Index{2, 3}, -1.0},
       Entry<double>{Index{3, 2}, -1.0},
       Entry<double>{Index{3, 3}, -2.0}});
  }

  // ================================================================
  // Basic Lanczos tests
  // ================================================================

  TEST_CASE("lanczos - diagonal 4x4, largest 2", "[lanczos]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(4, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 2);

    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    // Largest magnitude eigenvalues of {1,4,2,3} are {3,4}.
    CHECK(evals[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(4.0).margin(1e-8));
  }

  TEST_CASE("lanczos - diagonal 4x4, smallest 2", "[lanczos]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::smallest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(4, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 2);

    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(2.0).margin(1e-8));
  }

  TEST_CASE("lanczos - diagonal 4x4, largest algebraic", "[lanczos]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_algebraic,
      .collect_residuals = false};

    auto result = lanczos(4, cfg, apply_A);

    REQUIRE(result.converged);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(4.0).margin(1e-8));
  }

  TEST_CASE("lanczos - diagonal 4x4, smallest algebraic", "[lanczos]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::smallest_algebraic,
      .collect_residuals = false};

    auto result = lanczos(4, cfg, apply_A);

    REQUIRE(result.converged);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(2.0).margin(1e-8));
  }

  // ================================================================
  // Grid test (requires restart)
  // ================================================================

  TEST_CASE("lanczos - grid 16 largest 3", "[lanczos]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 3,
      .krylov_dimension = 10,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = true};

    auto result = lanczos(16, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 3);
    REQUIRE(result.eigenvectors.size() == 3);
  }

  // ================================================================
  // Eigenvector residual property: ||Ax - λx|| / ||x|| < tol
  // ================================================================

  TEST_CASE("lanczos - eigenvector residual", "[lanczos]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 3,
      .krylov_dimension = 10,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(16, cfg, apply_A);

    REQUIRE(result.converged);
    for (std::size_t k = 0; k < result.eigenvalues.size(); ++k) {
      auto const& x = result.eigenvectors[k];
      auto ax = multiply(A, std::span<double const>{x});

      double lambda = result.eigenvalues[k];
      double residual_sq = 0.0;
      double x_sq = 0.0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        double diff = ax[i] - lambda * x[i];
        residual_sq += diff * diff;
        x_sq += x[i] * x[i];
      }
      double rel_residual = std::sqrt(residual_sq / x_sq);
      CHECK(rel_residual < 1e-8);
    }
  }

  // ================================================================
  // Indefinite symmetric matrix
  // ================================================================

  TEST_CASE("lanczos - indefinite symmetric", "[lanczos]") {
    auto A = make_indefinite_4();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    // Request largest magnitude: should get the extreme eigenvalues.
    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(4, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 2);

    // Verify eigenvector residuals.
    for (std::size_t k = 0; k < result.eigenvalues.size(); ++k) {
      auto const& x = result.eigenvectors[k];
      auto ax = multiply(A, std::span<double const>{x});

      double lambda = result.eigenvalues[k];
      double residual_sq = 0.0;
      double x_sq = 0.0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        double diff = ax[i] - lambda * x[i];
        residual_sq += diff * diff;
        x_sq += x[i] * x[i];
      }
      CHECK(std::sqrt(residual_sq / x_sq) < 1e-8);
    }
  }

  // ================================================================
  // Shift-and-invert for generalized eigenvalue problem
  // ================================================================

  TEST_CASE(
    "lanczos - shift-and-invert for interior eigenvalues", "[lanczos]") {
    // A = diag(1, 2, 3, 4, 5), seek eigenvalues near sigma=3.
    // op(x) = (A - 3*I)^{-1} * x
    // op eigenvalues: 1/(1-3)=-0.5, 1/(2-3)=-1, 1/(3-3)=inf,
    //                 1/(4-3)=1, 1/(5-3)=0.5
    // Largest magnitude of op = the eigenvalue of A nearest to
    // sigma=3.
    // Actually lambda=3 maps to infinity, so Lanczos should find it
    // easily. Use sigma=2.5 to avoid singularity:
    // op eigenvalues: 1/(1-2.5)=-0.667, 1/(2-2.5)=-2,
    //                 1/(3-2.5)=2, 1/(4-2.5)=0.667, 1/(5-2.5)=0.4
    // Largest magnitude of op: {-2, 2} → correspond to A eigenvalues
    // {2, 3}.

    double sigma = 2.5;
    std::vector<double> diag = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto apply_op = [&diag, sigma](auto first, auto last, auto out) {
      auto n = std::distance(first, last);
      for (decltype(n) i = 0; i < n; ++i) {
        auto idx = static_cast<std::size_t>(i);
        *(out + i) = *(first + i) / (diag[idx] - sigma);
      }
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 5,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(5, cfg, apply_op);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 2);

    // Convert back: A eigenvalue = sigma + 1/theta.
    std::vector<double> a_evals;
    for (auto theta : result.eigenvalues) {
      a_evals.push_back(sigma + 1.0 / theta);
    }
    std::sort(a_evals.begin(), a_evals.end());

    CHECK(a_evals[0] == Catch::Approx(2.0).margin(1e-8));
    CHECK(a_evals[1] == Catch::Approx(3.0).margin(1e-8));
  }

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("lanczos - all eigenvalues of small matrix", "[lanczos]") {
    // 3x3 tridiagonal: diag=2, subdiag=-1.
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 2.0},
       Entry<double>{Index{1, 2}, -1.0},
       Entry<double>{Index{2, 1}, -1.0},
       Entry<double>{Index{2, 2}, 2.0}});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    // Request all 3 eigenvalues (nev = m = n).
    Lanczos_config<double> cfg{
      .num_eigenvalues = 3,
      .krylov_dimension = 3,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(3, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues.size() == 3);

    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());

    // Eigenvalues of 3x3 tridiag(2,-1): 2 - 2*cos(k*pi/4), k=1,2,3.
    double pi = std::acos(-1.0);
    CHECK(
      evals[0] ==
      Catch::Approx(2.0 - 2.0 * std::cos(1.0 * pi / 4.0)).margin(1e-8));
    CHECK(
      evals[1] ==
      Catch::Approx(2.0 - 2.0 * std::cos(2.0 * pi / 4.0)).margin(1e-8));
    CHECK(
      evals[2] ==
      Catch::Approx(2.0 - 2.0 * std::cos(3.0 * pi / 4.0)).margin(1e-8));
  }

  TEST_CASE("lanczos - residual norms reported", "[lanczos]") {
    auto A = make_grid_16();

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Lanczos_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 8,
      .tolerance = 1e-10,
      .max_restarts = 100,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = lanczos(16, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.residual_norms.size() == result.eigenvalues.size());
    for (auto rn : result.residual_norms) {
      CHECK(rn < 1e-8);
    }
  }

} // end of namespace sparkit::testing
