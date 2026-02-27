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
#include <sparkit/data/dense_hessenberg_eigen.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::hessenberg_eigen;
  using sparkit::data::detail::Hessenberg_eigen_config;

  using size_type = sparkit::config::size_type;

  // Helper: access H[i][j] in row-major storage of n×n matrix.
  static double&
  at(std::vector<double>& H, size_type n, size_type i, size_type j) {
    return H[static_cast<std::size_t>(i * n + j)];
  }

  // ================================================================
  // Real eigenvalue tests
  // ================================================================

  TEST_CASE(
    "hessenberg_eigen - 2x2 real eigenvalues", "[dense_hessenberg_eigen]") {
    // H = [[3, 1], [2, 4]], eigenvalues of [[3,1],[2,4]]:
    // trace = 7, det = 10, disc = 49-40=9, eigenvalues = (7±3)/2 =
    // {2, 5}.
    size_type n = 2;
    std::vector<double> H(4, 0.0);
    at(H, n, 0, 0) = 3.0;
    at(H, n, 0, 1) = 1.0;
    at(H, n, 1, 0) = 2.0;
    at(H, n, 1, 1) = 4.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = hessenberg_eigen(H, n, cfg);

    REQUIRE(result.eigenvalues_real.size() == 2);
    REQUIRE(result.eigenvalues_imag.size() == 2);

    // Sort by real part.
    std::vector<std::pair<double, double>> evals;
    for (std::size_t i = 0; i < 2; ++i) {
      evals.emplace_back(
        result.eigenvalues_real[i], result.eigenvalues_imag[i]);
    }
    std::sort(evals.begin(), evals.end(), [](auto& a, auto& b) {
      return a.first < b.first;
    });

    CHECK(evals[0].first == Catch::Approx(2.0).margin(1e-10));
    CHECK(evals[0].second == Catch::Approx(0.0).margin(1e-10));
    CHECK(evals[1].first == Catch::Approx(5.0).margin(1e-10));
    CHECK(evals[1].second == Catch::Approx(0.0).margin(1e-10));
  }

  TEST_CASE(
    "hessenberg_eigen - 2x2 complex conjugate pair",
    "[dense_hessenberg_eigen]") {
    // H = [[0, -1], [1, 0]] → rotation matrix, eigenvalues ±i.
    size_type n = 2;
    std::vector<double> H(4, 0.0);
    at(H, n, 0, 0) = 0.0;
    at(H, n, 0, 1) = -1.0;
    at(H, n, 1, 0) = 1.0;
    at(H, n, 1, 1) = 0.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = hessenberg_eigen(H, n, cfg);

    REQUIRE(result.eigenvalues_real.size() == 2);
    REQUIRE(result.eigenvalues_imag.size() == 2);

    // Should be conjugate pair: real parts = 0, imag parts = ±1.
    std::vector<std::pair<double, double>> evals;
    for (std::size_t i = 0; i < 2; ++i) {
      evals.emplace_back(
        result.eigenvalues_real[i], result.eigenvalues_imag[i]);
    }
    std::sort(evals.begin(), evals.end(), [](auto& a, auto& b) {
      return a.second < b.second;
    });

    CHECK(evals[0].first == Catch::Approx(0.0).margin(1e-10));
    CHECK(evals[0].second == Catch::Approx(-1.0).margin(1e-10));
    CHECK(evals[1].first == Catch::Approx(0.0).margin(1e-10));
    CHECK(evals[1].second == Catch::Approx(1.0).margin(1e-10));
  }

  TEST_CASE(
    "hessenberg_eigen - 4x4 upper Hessenberg", "[dense_hessenberg_eigen]") {
    // Companion matrix for p(x) = x^4 - 10x^3 + 35x^2 - 50x + 24
    // = (x-1)(x-2)(x-3)(x-4).
    // Companion form (upper Hessenberg):
    //   [[0, 0, 0, -24],
    //    [1, 0, 0,  50],
    //    [0, 1, 0, -35],
    //    [0, 0, 1,  10]]
    size_type n = 4;
    std::vector<double> H(16, 0.0);
    at(H, n, 0, 3) = -24.0;
    at(H, n, 1, 0) = 1.0;
    at(H, n, 1, 3) = 50.0;
    at(H, n, 2, 1) = 1.0;
    at(H, n, 2, 3) = -35.0;
    at(H, n, 3, 2) = 1.0;
    at(H, n, 3, 3) = 10.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = hessenberg_eigen(H, n, cfg);

    REQUIRE(result.eigenvalues_real.size() == 4);

    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());

    CHECK(evals[0] == Catch::Approx(1.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(2.0).margin(1e-8));
    CHECK(evals[2] == Catch::Approx(3.0).margin(1e-8));
    CHECK(evals[3] == Catch::Approx(4.0).margin(1e-8));

    // All imaginary parts should be zero.
    for (auto im : result.eigenvalues_imag) {
      CHECK(std::abs(im) < 1e-8);
    }
  }

  // ================================================================
  // Property tests
  // ================================================================

  TEST_CASE("hessenberg_eigen - trace invariant", "[dense_hessenberg_eigen]") {
    // H = [[5, 2, 1], [3, -1, 4], [0, 2, 3]]
    // trace = 5 + (-1) + 3 = 7
    size_type n = 3;
    std::vector<double> H(9, 0.0);
    at(H, n, 0, 0) = 5.0;
    at(H, n, 0, 1) = 2.0;
    at(H, n, 0, 2) = 1.0;
    at(H, n, 1, 0) = 3.0;
    at(H, n, 1, 1) = -1.0;
    at(H, n, 1, 2) = 4.0;
    at(H, n, 2, 1) = 2.0;
    at(H, n, 2, 2) = 3.0;

    double trace = 5.0 + (-1.0) + 3.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = hessenberg_eigen(H, n, cfg);

    double eval_sum = 0.0;
    for (auto r : result.eigenvalues_real) {
      eval_sum += r;
    }

    CHECK(eval_sum == Catch::Approx(trace).margin(1e-8));
  }

  TEST_CASE("hessenberg_eigen - 1x1 matrix", "[dense_hessenberg_eigen]") {
    size_type n = 1;
    std::vector<double> H = {7.0};

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = hessenberg_eigen(H, n, cfg);

    REQUIRE(result.eigenvalues_real.size() == 1);
    CHECK(result.eigenvalues_real[0] == Catch::Approx(7.0).margin(1e-14));
    CHECK(result.eigenvalues_imag[0] == Catch::Approx(0.0).margin(1e-14));
  }

  TEST_CASE("hessenberg_eigen - diagonal matrix", "[dense_hessenberg_eigen]") {
    // Already in Schur form.
    size_type n = 3;
    std::vector<double> H(9, 0.0);
    at(H, n, 0, 0) = 2.0;
    at(H, n, 1, 1) = -3.0;
    at(H, n, 2, 2) = 5.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 100};
    auto result = hessenberg_eigen(H, n, cfg);

    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());

    CHECK(evals[0] == Catch::Approx(-3.0).margin(1e-12));
    CHECK(evals[1] == Catch::Approx(2.0).margin(1e-12));
    CHECK(evals[2] == Catch::Approx(5.0).margin(1e-12));
  }

  TEST_CASE(
    "hessenberg_eigen - 3x3 upper Hessenberg", "[dense_hessenberg_eigen]") {
    // H = [[1, 2, 3], [1, 1, 2], [0, 1, 1]].
    // Characteristic polynomial: λ(-λ² + 3λ + 1)
    // Roots: λ = 0 and λ = (3 ± √13)/2.
    size_type n = 3;
    std::vector<double> H(9, 0.0);
    at(H, n, 0, 0) = 1.0;
    at(H, n, 0, 1) = 2.0;
    at(H, n, 0, 2) = 3.0;
    at(H, n, 1, 0) = 1.0;
    at(H, n, 1, 1) = 1.0;
    at(H, n, 1, 2) = 2.0;
    at(H, n, 2, 1) = 1.0;
    at(H, n, 2, 2) = 1.0;

    Hessenberg_eigen_config<double> cfg{
      .tolerance = 1e-14, .max_iterations = 200};
    auto result = hessenberg_eigen(H, n, cfg);

    auto evals = result.eigenvalues_real;
    std::sort(evals.begin(), evals.end());

    double sqrt13 = std::sqrt(13.0);
    CHECK(evals[0] == Catch::Approx((3.0 - sqrt13) / 2.0).margin(1e-8));
    CHECK(evals[1] == Catch::Approx(0.0).margin(1e-8));
    CHECK(evals[2] == Catch::Approx((3.0 + sqrt13) / 2.0).margin(1e-8));
  }

} // end of namespace sparkit::testing
