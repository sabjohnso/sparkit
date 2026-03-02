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
#include <string>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/diagnostics.hpp>
#include <sparkit/data/matgen.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::column_dominance_ratios;
  using sparkit::data::detail::diagonal_matrix;
  using sparkit::data::detail::is_column_diagonally_dominant;
  using sparkit::data::detail::is_numerically_symmetric;
  using sparkit::data::detail::is_positive_definite;
  using sparkit::data::detail::is_row_diagonally_dominant;
  using sparkit::data::detail::is_strictly_column_diagonally_dominant;
  using sparkit::data::detail::is_strictly_row_diagonally_dominant;
  using sparkit::data::detail::is_structurally_symmetric;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::row_dominance_ratios;
  using sparkit::data::detail::spy;
  using sparkit::data::detail::spy_svg;
  using sparkit::data::detail::tridiagonal_matrix;

  // ================================================================
  // Structural symmetry
  // ================================================================

  TEST_CASE(
    "diagnostics - is_structurally_symmetric identity", "[diagnostics]") {
    auto A = diagonal_matrix({1.0, 2.0, 3.0});
    CHECK(is_structurally_symmetric(A));
  }

  TEST_CASE(
    "diagnostics - is_structurally_symmetric upper triangular false",
    "[diagnostics]") {
    // Upper triangular: (0,1) exists but (1,0) does not
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{2, 2}, 4.0}});
    CHECK_FALSE(is_structurally_symmetric(A));
  }

  TEST_CASE(
    "diagnostics - is_structurally_symmetric tridiagonal", "[diagnostics]") {
    auto A = tridiagonal_matrix<double>(5, -1.0, 2.0, -1.0);
    CHECK(is_structurally_symmetric(A));
  }

  TEST_CASE(
    "diagnostics - is_structurally_symmetric non-square", "[diagnostics]") {
    auto A = make_matrix(
      Shape{2, 3},
      {Entry<double>{Index{0, 0}, 1.0}, Entry<double>{Index{1, 2}, 1.0}});
    CHECK_FALSE(is_structurally_symmetric(A));
  }

  // ================================================================
  // Numerical symmetry
  // ================================================================

  TEST_CASE(
    "diagnostics - is_numerically_symmetric exact match", "[diagnostics]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 3.0, -1.0);
    CHECK(is_numerically_symmetric(A));
  }

  TEST_CASE(
    "diagnostics - is_numerically_symmetric perturbed false", "[diagnostics]") {
    // Make an asymmetric matrix: (0,1) != (1,0)
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 2.0}, // asymmetric
       Entry<double>{Index{1, 1}, 2.0},
       Entry<double>{Index{2, 2}, 2.0}});
    CHECK_FALSE(is_numerically_symmetric(A));
  }

  TEST_CASE(
    "diagnostics - is_numerically_symmetric within tolerance",
    "[diagnostics]") {
    // (0,1) = 1.0, (1,0) = 1.001 — within 1% tolerance
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.001},
       Entry<double>{Index{1, 1}, 2.0},
       Entry<double>{Index{2, 2}, 2.0}});
    CHECK(is_numerically_symmetric(A, 0.01));
    CHECK_FALSE(is_numerically_symmetric(A, 1e-6));
  }

  // ================================================================
  // Row dominance ratios
  // ================================================================

  TEST_CASE(
    "diagnostics - row_dominance_ratios diagonal matrix", "[diagnostics]") {
    // Diagonal matrix: off-diagonal sum = 0, so ratio = inf
    auto A = diagonal_matrix({2.0, 3.0, 4.0});
    auto ratios = row_dominance_ratios(A);
    REQUIRE(std::ssize(ratios) == 3);
    CHECK(std::isinf(ratios[0]));
    CHECK(std::isinf(ratios[1]));
    CHECK(std::isinf(ratios[2]));
  }

  TEST_CASE(
    "diagnostics - row_dominance_ratios known values", "[diagnostics]") {
    // Row 0: diag=4, off=|-1| = 1 (boundary)    -> ratio = 4.0
    // Row 1: diag=4, off=|-1|+|-1| = 2 (interior) -> ratio = 2.0
    // Row 2: diag=4, off=|-1| = 1 (boundary)    -> ratio = 4.0
    auto A = tridiagonal_matrix<double>(3, -1.0, 4.0, -1.0);
    auto ratios = row_dominance_ratios(A);
    REQUIRE(std::ssize(ratios) == 3);
    CHECK(ratios[0] == Catch::Approx(4.0));
    CHECK(ratios[1] == Catch::Approx(2.0));
    CHECK(ratios[2] == Catch::Approx(4.0));
  }

  TEST_CASE(
    "diagnostics - is_row_diagonally_dominant SPD tridiagonal",
    "[diagnostics]") {
    // tridiagonal(n, -1, 2, -1): row i has |2| >= |-1|+|-1| for interior (>=1)
    auto A = tridiagonal_matrix<double>(5, -1.0, 2.0, -1.0);
    CHECK(is_row_diagonally_dominant(A));
  }

  TEST_CASE(
    "diagnostics - is_row_diagonally_dominant false for dominant column",
    "[diagnostics]") {
    // Matrix where off-diagonal sum exceeds diagonal:
    //   [1, 2; 2, 1] — each row: |1| < |2|
    auto A = make_matrix(
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 1.0}});
    CHECK_FALSE(is_row_diagonally_dominant(A));
  }

  TEST_CASE(
    "diagnostics - is_strictly_row_diagonally_dominant boundary",
    "[diagnostics]") {
    // Boundary row of tridiagonal(n,-1,2,-1): ratio = |2|/|-1| = 2 > 1
    // Interior row: ratio = |2|/(|-1|+|-1|) = 1, not strictly > 1
    auto A = tridiagonal_matrix<double>(3, -1.0, 2.0, -1.0);
    // Interior row 1 has ratio exactly 1 => NOT strictly dominant
    CHECK_FALSE(is_strictly_row_diagonally_dominant(A));
  }

  TEST_CASE(
    "diagnostics - is_strictly_row_diagonally_dominant for 2x2",
    "[diagnostics]") {
    // Each row: |3| > |1|
    auto A = tridiagonal_matrix<double>(2, -1.0, 3.0, -1.0);
    CHECK(is_strictly_row_diagonally_dominant(A));
  }

  // ================================================================
  // Column dominance ratios
  // ================================================================

  TEST_CASE(
    "diagnostics - column_dominance_ratios diagonal matrix", "[diagnostics]") {
    auto A = diagonal_matrix({5.0, 6.0, 7.0});
    auto ratios = column_dominance_ratios(A);
    REQUIRE(std::ssize(ratios) == 3);
    for (auto r : ratios) {
      CHECK(std::isinf(r));
    }
  }

  TEST_CASE(
    "diagnostics - is_column_diagonally_dominant symmetric SPD",
    "[diagnostics]") {
    // Symmetric tridiagonal(-1,2,-1) is also column diagonally dominant
    auto A = tridiagonal_matrix<double>(5, -1.0, 2.0, -1.0);
    CHECK(is_column_diagonally_dominant(A));
  }

  TEST_CASE(
    "diagnostics - is_strictly_column_diagonally_dominant false at boundary",
    "[diagnostics]") {
    // The interior column of tridiagonal(-1,2,-1) with n>=3: ratio = 1, not
    // strict
    auto A = tridiagonal_matrix<double>(3, -1.0, 2.0, -1.0);
    CHECK_FALSE(is_strictly_column_diagonally_dominant(A));
  }

  // ================================================================
  // Positive definiteness
  // ================================================================

  TEST_CASE(
    "diagnostics - is_positive_definite SPD tridiagonal", "[diagnostics]") {
    // tridiagonal(5, -1, 2, -1) is SPD
    auto A = tridiagonal_matrix<double>(5, -1.0, 2.0, -1.0);
    CHECK(is_positive_definite(A));
  }

  TEST_CASE(
    "diagnostics - is_positive_definite indefinite false", "[diagnostics]") {
    // Diagonal matrix with a negative entry: not positive definite
    auto A = diagonal_matrix({2.0, -1.0, 3.0});
    CHECK_FALSE(is_positive_definite(A));
  }

  TEST_CASE(
    "diagnostics - is_positive_definite non-square throws", "[diagnostics]") {
    auto A = make_matrix(
      Shape{2, 3},
      {Entry<double>{Index{0, 0}, 1.0}, Entry<double>{Index{1, 1}, 1.0}});
    CHECK_THROWS_AS(is_positive_definite(A), std::invalid_argument);
  }

  // ================================================================
  // Spy (ASCII)
  // ================================================================

  TEST_CASE(
    "diagnostics - spy small matrix character counts", "[diagnostics]") {
    // 3x3 diagonal: 3 nonzeros
    auto A = diagonal_matrix({1.0, 2.0, 3.0});

    // Width >= n, height >= m -> 1:1 mapping
    auto s = spy(A, 10, 10);

    // Count '#' and '.'
    long hash_count = std::count(s.begin(), s.end(), '#');
    long dot_count = std::count(s.begin(), s.end(), '.');

    // 3 nonzeros -> 3 '#'. Grid is 3x3 so 9 cells total: 3 '#' + 6 '.'
    CHECK(hash_count == 3);
    CHECK(dot_count == 6);
  }

  TEST_CASE("diagnostics - spy output line structure", "[diagnostics]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 2.0, -1.0);

    auto s = spy(A, 40, 20);

    // Count newlines — should equal height (one per row)
    long nl_count = std::count(s.begin(), s.end(), '\n');
    CHECK(nl_count == 4); // height capped at min(height, m) = min(20, 4) = 4
  }

  TEST_CASE("diagnostics - spy empty matrix", "[diagnostics]") {
    // Zero matrix (no nonzeros): all dots
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 0.0},
       Entry<double>{Index{1, 1}, 0.0},
       Entry<double>{Index{2, 2}, 0.0}});

    auto s = spy(A, 10, 10);
    long hash_count = std::count(s.begin(), s.end(), '#');
    // The 0.0 entries still occupy sparsity slots — they will appear as '#'
    // This is correct: spy reflects the sparsity pattern, not the values.
    CHECK(hash_count == 3);
  }

  // ================================================================
  // Spy SVG
  // ================================================================

  TEST_CASE("diagnostics - spy_svg contains svg tag", "[diagnostics]") {
    auto A = diagonal_matrix({1.0, 2.0, 3.0});
    auto svg = spy_svg(A, 4);

    CHECK(svg.find("<svg") != std::string::npos);
    CHECK(svg.find("</svg>") != std::string::npos);
  }

  TEST_CASE("diagnostics - spy_svg rect count matches nnz", "[diagnostics]") {
    // 3x3 diagonal has nnz = 3
    auto A = diagonal_matrix({1.0, 2.0, 3.0});
    auto svg = spy_svg(A, 4);

    // Count occurrences of "<rect"
    std::size_t rect_count = 0;
    std::size_t pos = 0;
    while ((pos = svg.find("<rect", pos)) != std::string::npos) {
      ++rect_count;
      ++pos;
    }
    CHECK(rect_count == 3);
  }

  TEST_CASE("diagnostics - spy_svg tridiagonal rect count", "[diagnostics]") {
    // 4x4 tridiagonal: 4 diagonal + 3 sub + 3 super = 10 nonzeros
    auto A = tridiagonal_matrix<double>(4, -1.0, 2.0, -1.0);
    auto svg = spy_svg(A, 4);

    std::size_t rect_count = 0;
    std::size_t pos = 0;
    while ((pos = svg.find("<rect", pos)) != std::string::npos) {
      ++rect_count;
      ++pos;
    }
    CHECK(rect_count == static_cast<std::size_t>(A.size()));
  }

} // end of namespace sparkit::testing
