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
#include <sparkit/data/arnoldi_eig.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/eigen_target.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Arnoldi_eig_config;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Eigen_target;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::arnoldi_eig;
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

  // ================================================================
  // Basic Arnoldi eigenvalue tests
  // ================================================================

  TEST_CASE("arnoldi_eig - diagonal 4x4, largest 2", "[arnoldi_eig]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(4, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues_real.size() == 2);

    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(4.0).margin(1e-8));

    // Imaginary parts should be zero.
    for (auto im : result.eigenvalues_imag) {
      CHECK(std::abs(im) < 1e-8);
    }
  }

  TEST_CASE("arnoldi_eig - diagonal 4x4, smallest magnitude", "[arnoldi_eig]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::smallest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(4, cfg, apply_A);

    REQUIRE(result.converged);
    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(2.0).margin(1e-8));
  }

  // ================================================================
  // Nonsymmetric matrix tests
  // ================================================================

  TEST_CASE(
    "arnoldi_eig - nonsymmetric 4x4 real eigenvalues", "[arnoldi_eig]") {
    // Upper triangular matrix with known eigenvalues on the diagonal.
    // A = [[5, 1, 0, 0], [0, 3, 1, 0], [0, 0, 1, 1], [0, 0, 0, 7]]
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 5.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 1.0},
       Entry<double>{Index{2, 2}, 1.0},
       Entry<double>{Index{2, 3}, 1.0},
       Entry<double>{Index{3, 3}, 7.0}});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(4, cfg, apply_A);

    REQUIRE(result.converged);
    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(5.0).margin(1e-6));
    CHECK(evals[1] == Catch::Approx(7.0).margin(1e-6));
  }

  // ================================================================
  // Eigenvector residual property
  // ================================================================

  TEST_CASE(
    "arnoldi_eig - eigenvector residual for real eigenvalues",
    "[arnoldi_eig]") {
    auto A = make_diagonal({1.0, 4.0, 2.0, 3.0, 5.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 3,
      .krylov_dimension = 5,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(5, cfg, apply_A);

    REQUIRE(result.converged);
    for (std::size_t k = 0; k < result.eigenvalues_real.size(); ++k) {
      // Skip complex eigenvalues.
      if (std::abs(result.eigenvalues_imag[k]) > 1e-8) { continue; }
      auto const& x = result.eigenvectors[k];
      auto ax = multiply(A, std::span<double const>{x});

      double lambda = result.eigenvalues_real[k];
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
  // Trace invariant
  // ================================================================

  TEST_CASE("arnoldi_eig - trace invariant", "[arnoldi_eig]") {
    // Request all eigenvalues (nev = m = n).
    auto A = make_diagonal({2.0, -3.0, 5.0, 1.0});
    double trace = 2.0 + (-3.0) + 5.0 + 1.0;

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 4,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(4, cfg, apply_A);

    REQUIRE(result.converged);
    double eval_sum = 0.0;
    for (auto r : result.eigenvalues_real) {
      eval_sum += r;
    }
    CHECK(eval_sum == Catch::Approx(trace).margin(1e-8));
  }

  // ================================================================
  // Largest real target
  // ================================================================

  TEST_CASE("arnoldi_eig - largest real target", "[arnoldi_eig]") {
    auto A = make_diagonal({-5.0, 3.0, 1.0, 4.0});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 4,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_real,
      .collect_residuals = false};

    auto result = arnoldi_eig(4, cfg, apply_A);

    REQUIRE(result.converged);
    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(3.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(4.0).margin(1e-8));
  }

  // ================================================================
  // Matrix with complex eigenvalues
  // ================================================================

  TEST_CASE("arnoldi_eig - matrix with complex eigenvalues", "[arnoldi_eig]") {
    // A = [[0, -2], [2, 0]]
    // Eigenvalues: ±2i
    // Use n=2, nev=2, m=2 (no restart needed).
    auto A = make_matrix(
      Shape{2, 2},
      {Entry<double>{Index{0, 1}, -2.0}, Entry<double>{Index{1, 0}, 2.0}});

    auto apply_A = [&A](auto first, auto last, auto out) {
      auto result = multiply(A, std::span<double const>{first, last});
      std::copy(result.begin(), result.end(), out);
    };

    Arnoldi_eig_config<double> cfg{
      .num_eigenvalues = 2,
      .krylov_dimension = 2,
      .tolerance = 1e-12,
      .max_restarts = 50,
      .target = Eigen_target::largest_magnitude,
      .collect_residuals = false};

    auto result = arnoldi_eig(2, cfg, apply_A);

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues_real.size() == 2);

    // Real parts should be ~0.
    for (auto r : result.eigenvalues_real) {
      CHECK(std::abs(r) < 1e-8);
    }
    // Imaginary parts should be ±2.
    auto imag = result.eigenvalues_imag;
    std::sort(imag.begin(), imag.end());
    CHECK(imag[0] == Catch::Approx(-2.0).margin(1e-8));
    CHECK(imag[1] == Catch::Approx(2.0).margin(1e-8));
  }

} // end of namespace sparkit::testing
