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
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/dense_bidiagonal_svd.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::bidiagonal_svd;
  using sparkit::data::detail::Bidiagonal_svd_config;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // Helper: reconstruct B from U * diag(sigma) * V^T and check
  // against original bidiagonal.
  //
  // B is k x k upper bidiagonal with diagonal alpha, superdiagonal
  // beta.
  // Reconstruction: B_ij = sum_l U[l][i] * sigma[l] * V[l][j]
  // where U[l] and V[l] are column vectors of size k.
  // ================================================================

  static void
  check_reconstruction(
    std::vector<double> const& alpha,
    std::vector<double> const& beta,
    std::vector<double> const& sigma,
    std::vector<std::vector<double>> const& U,
    std::vector<std::vector<double>> const& V,
    double tol) {
    auto k = static_cast<size_type>(alpha.size());

    for (size_type i = 0; i < k; ++i) {
      for (size_type j = 0; j < k; ++j) {
        double expected = 0.0;
        if (i == j) {
          expected = alpha[static_cast<std::size_t>(i)];
        } else if (j == i + 1) {
          expected = beta[static_cast<std::size_t>(i)];
        }

        double reconstructed = 0.0;
        for (size_type l = 0; l < k; ++l) {
          auto sl = static_cast<std::size_t>(l);
          auto si = static_cast<std::size_t>(i);
          auto sj = static_cast<std::size_t>(j);
          reconstructed += U[sl][si] * sigma[sl] * V[sl][sj];
        }

        CHECK(reconstructed == Catch::Approx(expected).margin(tol));
      }
    }
  }

  // Helper: check that columns are orthonormal (Q^T Q = I).
  // Q[col][row].
  static void
  check_orthonormal(
    std::vector<std::vector<double>> const& Q, size_type k, double tol) {
    for (size_type i = 0; i < k; ++i) {
      for (size_type j = 0; j < k; ++j) {
        double dot = 0.0;
        for (size_type r = 0; r < k; ++r) {
          dot += Q[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)] *
                 Q[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)];
        }
        double expected = (i == j) ? 1.0 : 0.0;
        CHECK(dot == Catch::Approx(expected).margin(tol));
      }
    }
  }

  // ================================================================
  // 1x1 bidiagonal
  // ================================================================

  TEST_CASE("bidiagonal_svd - 1x1 matrix", "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {5.0};
    std::vector<double> beta = {};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.singular_values.size() == 1);
    CHECK(result.singular_values[0] == Catch::Approx(5.0).margin(1e-14));
    REQUIRE(result.left_vectors.size() == 1);
    REQUIRE(result.right_vectors.size() == 1);
    CHECK(
      std::abs(result.left_vectors[0][0]) == Catch::Approx(1.0).margin(1e-14));
    CHECK(
      std::abs(result.right_vectors[0][0]) == Catch::Approx(1.0).margin(1e-14));
  }

  TEST_CASE("bidiagonal_svd - 1x1 negative value", "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {-3.0};
    std::vector<double> beta = {};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.singular_values.size() == 1);
    CHECK(result.singular_values[0] == Catch::Approx(3.0).margin(1e-14));
  }

  // ================================================================
  // 2x2 bidiagonal
  // ================================================================

  TEST_CASE("bidiagonal_svd - 2x2 matrix", "[dense_bidiagonal_svd]") {
    // B = [[3, 1], [0, 2]]
    // B^T B = [[9, 3], [3, 5]]
    // eigenvalues of B^T B: 7 ± sqrt(13)
    // singular values: sqrt(7 - sqrt(13)), sqrt(7 + sqrt(13))
    std::vector<double> alpha = {3.0, 2.0};
    std::vector<double> beta = {1.0};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.singular_values.size() == 2);
    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    double s13 = std::sqrt(13.0);
    CHECK(svals[0] == Catch::Approx(std::sqrt(7.0 - s13)).margin(1e-10));
    CHECK(svals[1] == Catch::Approx(std::sqrt(7.0 + s13)).margin(1e-10));

    check_reconstruction(
      alpha,
      beta,
      result.singular_values,
      result.left_vectors,
      result.right_vectors,
      1e-10);
  }

  // ================================================================
  // Frobenius norm invariant: sum(sigma_i^2) = sum(alpha_i^2 +
  // beta_i^2)
  // ================================================================

  TEST_CASE(
    "bidiagonal_svd - Frobenius norm invariant", "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {4.0, 3.0, 2.0, 1.0};
    std::vector<double> beta = {1.5, 0.5, 2.0};

    double frob_sq = 0.0;
    for (auto a : alpha) {
      frob_sq += a * a;
    }
    for (auto b : beta) {
      frob_sq += b * b;
    }

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 300};
    auto result = bidiagonal_svd(alpha, beta, cfg, false);

    double sigma_sq = 0.0;
    for (auto s : result.singular_values) {
      sigma_sq += s * s;
    }

    CHECK(sigma_sq == Catch::Approx(frob_sq).margin(1e-10));
  }

  // ================================================================
  // Full reconstruction: B = U * diag(sigma) * V^T
  // ================================================================

  TEST_CASE(
    "bidiagonal_svd - reconstruction (U Sigma V^T = B)",
    "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {4.0, 3.0, 2.0, 1.0};
    std::vector<double> beta = {1.5, 0.5, 2.0};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 300};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.left_vectors.size() == 4);
    REQUIRE(result.right_vectors.size() == 4);

    check_reconstruction(
      alpha,
      beta,
      result.singular_values,
      result.left_vectors,
      result.right_vectors,
      1e-10);

    // Also check orthogonality of U and V.
    check_orthonormal(result.left_vectors, 4, 1e-10);
    check_orthonormal(result.right_vectors, 4, 1e-10);
  }

  // ================================================================
  // Diagonal matrix (all beta = 0): already converged
  // ================================================================

  TEST_CASE(
    "bidiagonal_svd - diagonal (all beta zero)", "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {5.0, 3.0, 1.0};
    std::vector<double> beta = {0.0, 0.0};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.singular_values.size() == 3);
    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    CHECK(svals[0] == Catch::Approx(1.0).margin(1e-12));
    CHECK(svals[1] == Catch::Approx(3.0).margin(1e-12));
    CHECK(svals[2] == Catch::Approx(5.0).margin(1e-12));
  }

  // ================================================================
  // Pre-deflated entry: a zero in the superdiagonal
  // ================================================================

  TEST_CASE(
    "bidiagonal_svd - zero superdiagonal entry (pre-deflated)",
    "[dense_bidiagonal_svd]") {
    // B = [[2, 0, 0], [0, 3, 1], [0, 0, 4]]
    // Splits into 1x1 block {2} and 2x2 block [[3,1],[0,4]].
    std::vector<double> alpha = {2.0, 3.0, 4.0};
    std::vector<double> beta = {0.0, 1.0};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, true);

    REQUIRE(result.singular_values.size() == 3);

    // One singular value must be exactly 2.0 (isolated 1x1 block).
    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    CHECK(svals[0] == Catch::Approx(2.0).margin(1e-10));

    // Frobenius check: sum(sigma^2) = 4 + 9 + 1 + 16 = 30
    double sigma_sq = 0.0;
    for (auto s : svals) {
      sigma_sq += s * s;
    }
    CHECK(sigma_sq == Catch::Approx(30.0).margin(1e-10));

    check_reconstruction(
      alpha,
      beta,
      result.singular_values,
      result.left_vectors,
      result.right_vectors,
      1e-10);
  }

  // ================================================================
  // Singular values only (no vectors)
  // ================================================================

  TEST_CASE(
    "bidiagonal_svd - values only (no vectors)", "[dense_bidiagonal_svd]") {
    std::vector<double> alpha = {3.0, 2.0};
    std::vector<double> beta = {1.0};

    Bidiagonal_svd_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = bidiagonal_svd(alpha, beta, cfg, false);

    REQUIRE(result.singular_values.size() == 2);
    CHECK(result.left_vectors.empty());
    CHECK(result.right_vectors.empty());

    auto svals = result.singular_values;
    std::sort(svals.begin(), svals.end());
    double s13 = std::sqrt(13.0);
    CHECK(svals[0] == Catch::Approx(std::sqrt(7.0 - s13)).margin(1e-10));
    CHECK(svals[1] == Catch::Approx(std::sqrt(7.0 + s13)).margin(1e-10));
  }

} // end of namespace sparkit::testing
