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
#include <numeric>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/dense_tridiagonal_eigen.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::tridiagonal_eigen;
  using sparkit::data::detail::Tridiagonal_eigen_config;

  using size_type = sparkit::config::size_type;

  // Helper: compute Q^T * Q to check orthogonality.
  // Q is stored column-major: Q[col][row].
  static std::vector<double>
  qtq(std::vector<std::vector<double>> const& Q, size_type n) {
    std::vector<double> result(static_cast<std::size_t>(n * n), 0.0);
    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_type k = 0; k < n; ++k) {
          sum += Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                 Q[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
        }
        result[static_cast<std::size_t>(i * n + j)] = sum;
      }
    }
    return result;
  }

  // Helper: compute Q * diag(lambda) * Q^T to verify factorization.
  // Q[col][row], so (Q * diag * Q^T)[i][j] = sum_k Q[k][i] * lambda[k] *
  // Q[k][j].
  static std::vector<double>
  q_lambda_qt(
    std::vector<std::vector<double>> const& Q,
    std::vector<double> const& lambda,
    size_type n) {
    std::vector<double> result(static_cast<std::size_t>(n * n), 0.0);
    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_type k = 0; k < n; ++k) {
          sum += Q[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] *
                 lambda[static_cast<std::size_t>(k)] *
                 Q[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
        }
        result[static_cast<std::size_t>(i * n + j)] = sum;
      }
    }
    return result;
  }

  // ================================================================
  // Basic eigenvalue tests
  // ================================================================

  TEST_CASE("tridiagonal_eigen - 2x2 diagonal", "[dense_tridiagonal_eigen]") {
    // T = diag(3, 7), eigenvalues = {3, 7}
    std::vector<double> diag = {3.0, 7.0};
    std::vector<double> subdiag = {0.0};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvalues.size() == 2);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(3.0).margin(1e-12));
    CHECK(evals[1] == Catch::Approx(7.0).margin(1e-12));
  }

  TEST_CASE(
    "tridiagonal_eigen - 2x2 with off-diagonal", "[dense_tridiagonal_eigen]") {
    // T = [[2, 1], [1, 2]], eigenvalues = {1, 3}
    std::vector<double> diag = {2.0, 2.0};
    std::vector<double> subdiag = {1.0};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvalues.size() == 2);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-12));
    CHECK(evals[1] == Catch::Approx(3.0).margin(1e-12));
  }

  TEST_CASE(
    "tridiagonal_eigen - 4x4 tridiagonal", "[dense_tridiagonal_eigen]") {
    // T = tridiag(-1, 2, -1) of size 4.
    // Eigenvalues: 2 - 2*cos(k*pi/5) for k=1..4
    std::vector<double> diag = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> subdiag = {-1.0, -1.0, -1.0};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvalues.size() == 4);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());

    double pi = std::acos(-1.0);
    std::vector<double> expected;
    for (int k = 1; k <= 4; ++k) {
      expected.push_back(2.0 - 2.0 * std::cos(k * pi / 5.0));
    }
    std::sort(expected.begin(), expected.end());

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(evals[i] == Catch::Approx(expected[i]).margin(1e-10));
    }
  }

  // ================================================================
  // Eigenvector property tests
  // ================================================================

  TEST_CASE(
    "tridiagonal_eigen - eigenvectors orthogonal (Q^T Q = I)",
    "[dense_tridiagonal_eigen]") {
    std::vector<double> diag = {4.0, 3.0, 2.0, 1.0};
    std::vector<double> subdiag = {1.0, 0.5, 0.25};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvectors.size() == 4);
    size_type n = 4;
    auto QTQ = qtq(result.eigenvectors, n);

    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n; ++j) {
        double expected = (i == j) ? 1.0 : 0.0;
        CHECK(
          QTQ[static_cast<std::size_t>(i * n + j)] ==
          Catch::Approx(expected).margin(1e-10));
      }
    }
  }

  TEST_CASE(
    "tridiagonal_eigen - factorization (Q Lambda Q^T = T)",
    "[dense_tridiagonal_eigen]") {
    std::vector<double> diag = {4.0, 3.0, 2.0, 1.0};
    std::vector<double> subdiag = {1.0, 0.5, 0.25};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    size_type n = 4;
    auto QLQT = q_lambda_qt(result.eigenvectors, result.eigenvalues, n);

    // Reconstruct T as dense matrix.
    std::vector<double> T(static_cast<std::size_t>(n * n), 0.0);
    for (size_type i = 0; i < n; ++i) {
      T[static_cast<std::size_t>(i * n + i)] =
        diag[static_cast<std::size_t>(i)];
      if (i + 1 < n) {
        T[static_cast<std::size_t>(i * n + i + 1)] =
          subdiag[static_cast<std::size_t>(i)];
        T[static_cast<std::size_t>((i + 1) * n + i)] =
          subdiag[static_cast<std::size_t>(i)];
      }
    }

    for (size_type i = 0; i < n * n; ++i) {
      CHECK(
        QLQT[static_cast<std::size_t>(i)] ==
        Catch::Approx(T[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // ================================================================
  // Edge cases
  // ================================================================

  TEST_CASE("tridiagonal_eigen - 1x1 matrix", "[dense_tridiagonal_eigen]") {
    std::vector<double> diag = {5.0};
    std::vector<double> subdiag = {};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvalues.size() == 1);
    CHECK(result.eigenvalues[0] == Catch::Approx(5.0).margin(1e-14));
    REQUIRE(result.eigenvectors.size() == 1);
    CHECK(
      std::abs(result.eigenvectors[0][0]) == Catch::Approx(1.0).margin(1e-14));
  }

  TEST_CASE(
    "tridiagonal_eigen - eigenvalues only (no eigenvectors)",
    "[dense_tridiagonal_eigen]") {
    std::vector<double> diag = {2.0, 2.0};
    std::vector<double> subdiag = {1.0};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, false);

    REQUIRE(result.eigenvalues.size() == 2);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-12));
    CHECK(evals[1] == Catch::Approx(3.0).margin(1e-12));
    CHECK(result.eigenvectors.empty());
  }

  TEST_CASE(
    "tridiagonal_eigen - negative eigenvalues", "[dense_tridiagonal_eigen]") {
    // T = [[-2, 1], [1, -3]], eigenvalues = (-5 ± sqrt(5))/2
    std::vector<double> diag = {-2.0, -3.0};
    std::vector<double> subdiag = {1.0};

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, true);

    REQUIRE(result.eigenvalues.size() == 2);
    auto evals = result.eigenvalues;
    std::sort(evals.begin(), evals.end());

    double trace = -2.0 + (-3.0);
    double det = (-2.0) * (-3.0) - 1.0;
    double disc = std::sqrt(trace * trace - 4.0 * det);
    double e1 = (trace - disc) / 2.0;
    double e2 = (trace + disc) / 2.0;

    CHECK(evals[0] == Catch::Approx(e1).margin(1e-12));
    CHECK(evals[1] == Catch::Approx(e2).margin(1e-12));
  }

  TEST_CASE(
    "tridiagonal_eigen - trace invariant", "[dense_tridiagonal_eigen]") {
    std::vector<double> diag = {5.0, -3.0, 2.0, 7.0, -1.0};
    std::vector<double> subdiag = {1.0, 2.0, 0.5, 1.5};

    double trace = 0.0;
    for (auto d : diag) {
      trace += d;
    }

    Tridiagonal_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 300};
    auto result = tridiagonal_eigen(diag, subdiag, cfg, false);

    double eval_sum = 0.0;
    for (auto e : result.eigenvalues) {
      eval_sum += e;
    }

    CHECK(eval_sum == Catch::Approx(trace).margin(1e-10));
  }

} // end of namespace sparkit::testing
