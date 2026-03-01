//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::arrow_matrix;
  using sparkit::data::detail::convection_diffusion_2d;
  using sparkit::data::detail::diagonal_matrix;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::manufactured_solution;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::poisson_2d;
  using sparkit::data::detail::random_sparse;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // make_matrix
  // ================================================================

  TEST_CASE("matgen - make_matrix round-trip", "[matgen]") {
    std::vector<Entry<double>> entries = {
      {Index{0, 0}, 1.0},
      {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0},
      {Index{2, 2}, 5.0}};

    auto A = make_matrix(Shape{3, 3}, entries);

    REQUIRE(A.shape() == Shape{3, 3});
    REQUIRE(A.size() == 5);
    CHECK(A(0, 0) == Catch::Approx(1.0));
    CHECK(A(0, 2) == Catch::Approx(2.0));
    CHECK(A(1, 1) == Catch::Approx(3.0));
    CHECK(A(2, 0) == Catch::Approx(4.0));
    CHECK(A(2, 2) == Catch::Approx(5.0));
    CHECK(A(0, 1) == Catch::Approx(0.0));
    CHECK(A(1, 0) == Catch::Approx(0.0));
  }

  // ================================================================
  // diagonal_matrix
  // ================================================================

  TEST_CASE("matgen - diagonal_matrix", "[matgen]") {
    std::vector<double> diag = {2.0, 3.0, 5.0, 7.0};
    auto A = diagonal_matrix(diag);

    REQUIRE(A.shape() == Shape{4, 4});
    REQUIRE(A.size() == 4);
    for (size_type i = 0; i < 4; ++i) {
      CHECK(A(i, i) == Catch::Approx(diag[static_cast<std::size_t>(i)]));
      for (size_type j = 0; j < 4; ++j) {
        if (j != i) { CHECK(A(i, j) == Catch::Approx(0.0)); }
      }
    }
  }

  // ================================================================
  // tridiagonal_matrix
  // ================================================================

  TEST_CASE("matgen - tridiagonal_matrix structure", "[matgen]") {
    auto A = tridiagonal_matrix<double>(5, -1.0, 4.0, -1.0);

    REQUIRE(A.shape() == Shape{5, 5});
    // nnz = n + 2*(n-1) = 5 + 8 = 13
    REQUIRE(A.size() == 13);

    // Check diagonal
    for (size_type i = 0; i < 5; ++i) {
      CHECK(A(i, i) == Catch::Approx(4.0));
    }
    // Check sub- and super-diagonals
    for (size_type i = 0; i < 4; ++i) {
      CHECK(A(i, i + 1) == Catch::Approx(-1.0));
      CHECK(A(i + 1, i) == Catch::Approx(-1.0));
    }
    // Check beyond bandwidth is zero
    CHECK(A(0, 2) == Catch::Approx(0.0));
    CHECK(A(0, 4) == Catch::Approx(0.0));
  }

  TEST_CASE("matgen - tridiagonal_matrix symmetry", "[matgen]") {
    auto A = tridiagonal_matrix<double>(6, -2.0, 10.0, -2.0);

    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = i; j < 6; ++j) {
        CHECK(A(i, j) == Catch::Approx(A(j, i)));
      }
    }
  }

  // ================================================================
  // arrow_matrix
  // ================================================================

  TEST_CASE("matgen - arrow_matrix structure", "[matgen]") {
    auto A = arrow_matrix<double>(5, 10.0, 1.0);

    REQUIRE(A.shape() == Shape{5, 5});

    // Diagonal entries
    for (size_type i = 0; i < 5; ++i) {
      CHECK(A(i, i) == Catch::Approx(10.0));
    }
    // First row and column (arrow part)
    for (size_type j = 1; j < 5; ++j) {
      CHECK(A(0, j) == Catch::Approx(1.0));
      CHECK(A(j, 0) == Catch::Approx(1.0));
    }
    // Interior off-diagonal should be zero
    CHECK(A(1, 2) == Catch::Approx(0.0));
    CHECK(A(2, 3) == Catch::Approx(0.0));
    CHECK(A(3, 4) == Catch::Approx(0.0));

    // nnz = n (diag) + 2*(n-1) (arrow) = 5 + 8 = 13
    REQUIRE(A.size() == 13);
  }

  // ================================================================
  // poisson_2d
  // ================================================================

  TEST_CASE("matgen - poisson_2d 4x4 matches manual", "[matgen]") {
    // Build the manual 4x4 grid Laplacian that tests have been using
    size_type const grid = 4;
    size_type const n = grid * grid;
    auto A = poisson_2d(grid);

    REQUIRE(A.shape() == Shape{n, n});

    // Check a corner node (0,0): 2 neighbors, diag = 2
    CHECK(A(0, 0) == Catch::Approx(2.0));
    CHECK(A(0, 1) == Catch::Approx(-1.0));
    CHECK(A(0, grid) == Catch::Approx(-1.0));

    // Check an interior node (1,1): 4 neighbors, diag = 4
    auto interior = 1 * grid + 1;
    CHECK(A(interior, interior) == Catch::Approx(4.0));
    CHECK(A(interior, interior - 1) == Catch::Approx(-1.0));
    CHECK(A(interior, interior + 1) == Catch::Approx(-1.0));
    CHECK(A(interior, interior - grid) == Catch::Approx(-1.0));
    CHECK(A(interior, interior + grid) == Catch::Approx(-1.0));

    // Check an edge node (0,1): 3 neighbors, diag = 3
    CHECK(A(1, 1) == Catch::Approx(3.0));
  }

  TEST_CASE("matgen - poisson_2d rectangular grid", "[matgen]") {
    auto A = poisson_2d(static_cast<size_type>(3), static_cast<size_type>(5));
    size_type const n = 3 * 5;

    REQUIRE(A.shape() == Shape{n, n});

    // Corner (0,0): 2 neighbors
    CHECK(A(0, 0) == Catch::Approx(2.0));

    // Edge along x at (0,1): 3 neighbors (west, east, south)
    CHECK(A(1, 1) == Catch::Approx(3.0));

    // Interior node (1,1): node = 1*3+1 = 4, 4 neighbors
    auto node = static_cast<size_type>(1 * 3 + 1);
    CHECK(A(node, node) == Catch::Approx(4.0));
  }

  TEST_CASE("matgen - poisson_2d SPD", "[matgen]") {
    auto A = poisson_2d(static_cast<size_type>(5));
    size_type const n = 25;

    // Test x^T A x > 0 for a non-zero vector
    std::vector<double> x(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      x[i] = static_cast<double>(i + 1);
    }

    auto Ax = multiply(A, std::span<double const>{x});

    double xtAx = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      xtAx += x[i] * Ax[i];
    }
    CHECK(xtAx > 0.0);

    // Verify symmetry
    for (size_type i = 0; i < n; ++i) {
      for (size_type j = i + 1; j < n; ++j) {
        CHECK(A(i, j) == Catch::Approx(A(j, i)));
      }
    }
  }

  // ================================================================
  // convection_diffusion_2d
  // ================================================================

  TEST_CASE("matgen - convection_diffusion_2d nonsymmetric", "[matgen]") {
    auto A = convection_diffusion_2d(
      static_cast<size_type>(4), static_cast<size_type>(4), 1.0, 0.5, 0.3);

    size_type const n = 16;
    REQUIRE(A.shape() == Shape{n, n});

    // With nonzero convection, A should be nonsymmetric
    bool found_asymmetry = false;
    for (size_type i = 0; i < n && !found_asymmetry; ++i) {
      for (size_type j = i + 1; j < n && !found_asymmetry; ++j) {
        if (std::abs(A(i, j) - A(j, i)) > 1e-14) { found_asymmetry = true; }
      }
    }
    CHECK(found_asymmetry);
  }

  TEST_CASE("matgen - convection_diffusion_2d reduces to poisson", "[matgen]") {
    size_type const grid = 4;
    size_type const n = grid * grid;

    auto A_cd = convection_diffusion_2d(grid, grid, 1.0, 0.0, 0.0);
    auto A_p = poisson_2d(grid);

    REQUIRE(A_cd.shape() == A_p.shape());

    // With zero convection, should match poisson_2d
    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n; ++j) {
        CHECK(A_cd(i, j) == Catch::Approx(A_p(i, j)));
      }
    }
  }

  // ================================================================
  // random_sparse
  // ================================================================

  TEST_CASE("matgen - random_sparse dimensions", "[matgen]") {
    size_type const n = 20;
    size_type const nnz_per_row = 5;
    auto A = random_sparse<double>(n, nnz_per_row, 42u);

    REQUIRE(A.shape() == Shape{n, n});
    // At least n entries (diagonal), at most n * nnz_per_row
    CHECK(A.size() >= n);
    CHECK(A.size() <= n * nnz_per_row);
  }

  TEST_CASE("matgen - random_sparse diagonal dominance", "[matgen]") {
    size_type const n = 30;
    size_type const nnz_per_row = 6;
    auto A = random_sparse<double>(n, nnz_per_row, 123u);

    for (size_type i = 0; i < n; ++i) {
      double diag_val = std::abs(A(i, i));
      double off_diag_sum = 0.0;
      for (size_type j = 0; j < n; ++j) {
        if (j != i) { off_diag_sum += std::abs(A(i, j)); }
      }
      CHECK(diag_val > off_diag_sum);
    }
  }

  TEST_CASE("matgen - random_sparse reproducible", "[matgen]") {
    auto A1 = random_sparse<double>(15, 4, 999u);
    auto A2 = random_sparse<double>(15, 4, 999u);

    REQUIRE(A1.size() == A2.size());
    auto v1 = A1.values();
    auto v2 = A2.values();
    for (size_type i = 0; i < A1.size(); ++i) {
      CHECK(v1[i] == Catch::Approx(v2[i]));
    }
  }

  TEST_CASE("matgen - random_sparse different seeds", "[matgen]") {
    auto A1 = random_sparse<double>(15, 4, 111u);
    auto A2 = random_sparse<double>(15, 4, 222u);

    // At least one value should differ
    bool any_different = false;
    auto v1 = A1.values();
    auto v2 = A2.values();
    auto common = std::min(A1.size(), A2.size());
    for (size_type i = 0; i < common; ++i) {
      if (std::abs(v1[i] - v2[i]) > 1e-14) {
        any_different = true;
        break;
      }
    }
    CHECK(any_different);
  }

  // ================================================================
  // manufactured_solution
  // ================================================================

  TEST_CASE("matgen - manufactured_solution consistency", "[matgen]") {
    size_type const n = 20;
    std::vector<double> x_exact;
    std::vector<double> b;

    auto A = manufactured_solution<double>(n, x_exact, b);

    REQUIRE(A.shape() == Shape{n, n});
    REQUIRE(std::ssize(x_exact) == n);
    REQUIRE(std::ssize(b) == n);

    // Verify A * x_exact == b
    auto Ax = multiply(A, std::span<double const>{x_exact});
    for (size_type i = 0; i < n; ++i) {
      CHECK(
        Ax[static_cast<std::size_t>(i)] ==
        Catch::Approx(b[static_cast<std::size_t>(i)]).epsilon(1e-10));
    }
  }

} // end of namespace sparkit::testing
