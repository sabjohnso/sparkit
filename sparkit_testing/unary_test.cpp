//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::extract_diagonal;
  using sparkit::data::detail::extract_lower_triangle;
  using sparkit::data::detail::extract_upper_triangle;
  using sparkit::data::detail::filter;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::submatrix;
  using sparkit::data::detail::transpose;

  // ================================================================
  // extract_diagonal
  // ================================================================

  TEST_CASE("unary - extract_diagonal known values", "[unary]") {
    // A = [[1,2,0],[0,3,4],[5,0,6]]
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{1, 1}, 3.0},
                                     {Index{1, 2}, 4.0},
                                     {Index{2, 0}, 5.0},
                                     {Index{2, 2}, 6.0}}};

    auto d = extract_diagonal(A);

    REQUIRE(std::ssize(d) == 3);
    CHECK(d[0] == Catch::Approx(1.0));
    CHECK(d[1] == Catch::Approx(3.0));
    CHECK(d[2] == Catch::Approx(6.0));
  }

  TEST_CASE("unary - extract_diagonal rectangular", "[unary]") {
    // 2x4 matrix: diagonal has min(2,4)=2 entries
    Compressed_row_matrix<double> A{Shape{2, 4},
                                    {{Index{0, 0}, 7.0},
                                     {Index{0, 3}, 8.0},
                                     {Index{1, 1}, 9.0},
                                     {Index{1, 2}, 10.0}}};

    auto d = extract_diagonal(A);

    REQUIRE(std::ssize(d) == 2);
    CHECK(d[0] == Catch::Approx(7.0));
    CHECK(d[1] == Catch::Approx(9.0));
  }

  TEST_CASE("unary - extract_diagonal empty matrix", "[unary]") {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    auto d = extract_diagonal(A);

    REQUIRE(std::ssize(d) == 3);
    CHECK(d[0] == Catch::Approx(0.0));
    CHECK(d[1] == Catch::Approx(0.0));
    CHECK(d[2] == Catch::Approx(0.0));
  }

  TEST_CASE("unary - extract_diagonal zero diagonal entries", "[unary]") {
    // No structural diagonal entries
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 1}, 5.0}, {Index{1, 0}, 6.0}}};

    auto d = extract_diagonal(A);

    REQUIRE(std::ssize(d) == 2);
    CHECK(d[0] == Catch::Approx(0.0));
    CHECK(d[1] == Catch::Approx(0.0));
  }

  // ================================================================
  // extract_lower_triangle
  // ================================================================

  TEST_CASE("unary - extract_lower_triangle strict", "[unary]") {
    // A = [[1,2,3],[4,5,6],[7,8,9]]
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{1, 0}, 4.0},
                                     {Index{1, 1}, 5.0},
                                     {Index{1, 2}, 6.0},
                                     {Index{2, 0}, 7.0},
                                     {Index{2, 1}, 8.0},
                                     {Index{2, 2}, 9.0}}};

    auto L = extract_lower_triangle(A);

    CHECK(L.shape() == Shape(3, 3));
    CHECK(L.size() == 3);
    CHECK(L(1, 0) == Catch::Approx(4.0));
    CHECK(L(2, 0) == Catch::Approx(7.0));
    CHECK(L(2, 1) == Catch::Approx(8.0));
    CHECK(L(0, 0) == Catch::Approx(0.0));
    CHECK(L(1, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("unary - extract_lower_triangle with diagonal", "[unary]") {
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{1, 0}, 4.0},
                                     {Index{1, 1}, 5.0},
                                     {Index{1, 2}, 6.0},
                                     {Index{2, 0}, 7.0},
                                     {Index{2, 1}, 8.0},
                                     {Index{2, 2}, 9.0}}};

    auto L = extract_lower_triangle(A, true);

    CHECK(L.shape() == Shape(3, 3));
    CHECK(L.size() == 6);
    CHECK(L(0, 0) == Catch::Approx(1.0));
    CHECK(L(1, 0) == Catch::Approx(4.0));
    CHECK(L(1, 1) == Catch::Approx(5.0));
    CHECK(L(2, 0) == Catch::Approx(7.0));
    CHECK(L(2, 1) == Catch::Approx(8.0));
    CHECK(L(2, 2) == Catch::Approx(9.0));
    CHECK(L(0, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("unary - extract_lower_triangle empty result", "[unary]") {
    // Upper triangular matrix — strict lower is empty
    Compressed_row_matrix<double> A{
        Shape{2, 2},
        {{Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}, {Index{1, 1}, 3.0}}};

    auto L = extract_lower_triangle(A);

    CHECK(L.shape() == Shape(2, 2));
    CHECK(L.size() == 0);
  }

  // ================================================================
  // extract_upper_triangle
  // ================================================================

  TEST_CASE("unary - extract_upper_triangle strict", "[unary]") {
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{1, 0}, 4.0},
                                     {Index{1, 1}, 5.0},
                                     {Index{1, 2}, 6.0},
                                     {Index{2, 0}, 7.0},
                                     {Index{2, 1}, 8.0},
                                     {Index{2, 2}, 9.0}}};

    auto U = extract_upper_triangle(A);

    CHECK(U.shape() == Shape(3, 3));
    CHECK(U.size() == 3);
    CHECK(U(0, 1) == Catch::Approx(2.0));
    CHECK(U(0, 2) == Catch::Approx(3.0));
    CHECK(U(1, 2) == Catch::Approx(6.0));
    CHECK(U(0, 0) == Catch::Approx(0.0));
    CHECK(U(1, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("unary - extract_upper_triangle with diagonal", "[unary]") {
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{1, 0}, 4.0},
                                     {Index{1, 1}, 5.0},
                                     {Index{1, 2}, 6.0},
                                     {Index{2, 0}, 7.0},
                                     {Index{2, 1}, 8.0},
                                     {Index{2, 2}, 9.0}}};

    auto U = extract_upper_triangle(A, true);

    CHECK(U.shape() == Shape(3, 3));
    CHECK(U.size() == 6);
    CHECK(U(0, 0) == Catch::Approx(1.0));
    CHECK(U(0, 1) == Catch::Approx(2.0));
    CHECK(U(0, 2) == Catch::Approx(3.0));
    CHECK(U(1, 1) == Catch::Approx(5.0));
    CHECK(U(1, 2) == Catch::Approx(6.0));
    CHECK(U(2, 2) == Catch::Approx(9.0));
    CHECK(U(1, 0) == Catch::Approx(0.0));
  }

  TEST_CASE("unary - extract_upper_triangle empty result", "[unary]") {
    // Lower triangular matrix — strict upper is empty
    Compressed_row_matrix<double> A{
        Shape{2, 2},
        {{Index{0, 0}, 1.0}, {Index{1, 0}, 2.0}, {Index{1, 1}, 3.0}}};

    auto U = extract_upper_triangle(A);

    CHECK(U.shape() == Shape(2, 2));
    CHECK(U.size() == 0);
  }

  // ================================================================
  // transpose
  // ================================================================

  TEST_CASE("unary - transpose known result", "[unary]") {
    // A = [[1,2],[3,4]]
    // A^T = [[1,3],[2,4]]
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{1, 0}, 3.0},
                                     {Index{1, 1}, 4.0}}};

    auto At = transpose(A);

    CHECK(At.shape() == Shape(2, 2));
    CHECK(At(0, 0) == Catch::Approx(1.0));
    CHECK(At(0, 1) == Catch::Approx(3.0));
    CHECK(At(1, 0) == Catch::Approx(2.0));
    CHECK(At(1, 1) == Catch::Approx(4.0));
  }

  TEST_CASE("unary - transpose rectangular", "[unary]") {
    // A is 2x3: [[1,0,2],[0,3,0]]
    // A^T is 3x2: [[1,0],[0,3],[2,0]]
    Compressed_row_matrix<double> A{
        Shape{2, 3},
        {{Index{0, 0}, 1.0}, {Index{0, 2}, 2.0}, {Index{1, 1}, 3.0}}};

    auto At = transpose(A);

    CHECK(At.shape() == Shape(3, 2));
    CHECK(At.size() == 3);
    CHECK(At(0, 0) == Catch::Approx(1.0));
    CHECK(At(1, 1) == Catch::Approx(3.0));
    CHECK(At(2, 0) == Catch::Approx(2.0));
  }

  TEST_CASE("unary - transpose double transpose equals identity", "[unary]") {
    Compressed_row_matrix<double> A{Shape{3, 3},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 2}, 2.0},
                                     {Index{1, 1}, 3.0},
                                     {Index{2, 0}, 4.0},
                                     {Index{2, 2}, 5.0}}};

    auto Att = transpose(transpose(A));

    CHECK(Att.shape() == A.shape());
    CHECK(Att.size() == A.size());
    for (config::size_type i = 0; i < 3; ++i) {
      for (config::size_type j = 0; j < 3; ++j) {
        CHECK(Att(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("unary - transpose empty matrix", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 3}, {}};

    auto At = transpose(A);

    CHECK(At.shape() == Shape(3, 2));
    CHECK(At.size() == 0);
  }

  // ================================================================
  // filter
  // ================================================================

  TEST_CASE("unary - filter drops below threshold", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, 0.1},
                                     {Index{0, 1}, 5.0},
                                     {Index{1, 0}, 0.01},
                                     {Index{1, 1}, 3.0}}};

    auto B = filter(A, 0.5);

    CHECK(B.shape() == Shape(2, 2));
    CHECK(B.size() == 2);
    CHECK(B(0, 1) == Catch::Approx(5.0));
    CHECK(B(1, 1) == Catch::Approx(3.0));
  }

  TEST_CASE("unary - filter all survive", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{1, 0}, 3.0},
                                     {Index{1, 1}, 4.0}}};

    auto B = filter(A, 0.0);

    CHECK(B.size() == 4);
  }

  TEST_CASE("unary - filter none survive", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, 0.1},
                                     {Index{0, 1}, 0.2},
                                     {Index{1, 0}, 0.3},
                                     {Index{1, 1}, 0.4}}};

    auto B = filter(A, 1.0);

    CHECK(B.shape() == Shape(2, 2));
    CHECK(B.size() == 0);
  }

  TEST_CASE("unary - filter negative values", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, -5.0},
                                     {Index{0, 1}, 0.1},
                                     {Index{1, 0}, -0.2},
                                     {Index{1, 1}, 3.0}}};

    auto B = filter(A, 1.0);

    CHECK(B.size() == 2);
    CHECK(B(0, 0) == Catch::Approx(-5.0));
    CHECK(B(1, 1) == Catch::Approx(3.0));
  }

  // ================================================================
  // submatrix
  // ================================================================

  TEST_CASE("unary - submatrix interior block", "[unary]") {
    // 4x4 matrix, extract rows [1,3) cols [1,3) → 2x2
    Compressed_row_matrix<double> A{Shape{4, 4},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{0, 3}, 4.0},
                                     {Index{1, 0}, 5.0},
                                     {Index{1, 1}, 6.0},
                                     {Index{1, 2}, 7.0},
                                     {Index{1, 3}, 8.0},
                                     {Index{2, 0}, 9.0},
                                     {Index{2, 1}, 10.0},
                                     {Index{2, 2}, 11.0},
                                     {Index{2, 3}, 12.0},
                                     {Index{3, 0}, 13.0},
                                     {Index{3, 1}, 14.0},
                                     {Index{3, 2}, 15.0},
                                     {Index{3, 3}, 16.0}}};

    auto S = submatrix(A, 1, 3, 1, 3);

    CHECK(S.shape() == Shape(2, 2));
    CHECK(S(0, 0) == Catch::Approx(6.0));
    CHECK(S(0, 1) == Catch::Approx(7.0));
    CHECK(S(1, 0) == Catch::Approx(10.0));
    CHECK(S(1, 1) == Catch::Approx(11.0));
  }

  TEST_CASE("unary - submatrix full matrix", "[unary]") {
    Compressed_row_matrix<double> A{Shape{2, 2},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{1, 0}, 3.0},
                                     {Index{1, 1}, 4.0}}};

    auto S = submatrix(A, 0, 2, 0, 2);

    CHECK(S.shape() == Shape(2, 2));
    CHECK(S.size() == 4);
    CHECK(S(0, 0) == Catch::Approx(1.0));
    CHECK(S(0, 1) == Catch::Approx(2.0));
    CHECK(S(1, 0) == Catch::Approx(3.0));
    CHECK(S(1, 1) == Catch::Approx(4.0));
  }

  TEST_CASE("unary - submatrix two rows", "[unary]") {
    // 4x4 matrix, extract rows [1,3) cols [0,3) → 2x3
    Compressed_row_matrix<double> A{Shape{4, 4},
                                    {{Index{0, 0}, 1.0},
                                     {Index{0, 1}, 2.0},
                                     {Index{0, 2}, 3.0},
                                     {Index{0, 3}, 4.0},
                                     {Index{1, 0}, 5.0},
                                     {Index{1, 1}, 6.0},
                                     {Index{1, 2}, 7.0},
                                     {Index{1, 3}, 8.0},
                                     {Index{2, 0}, 9.0},
                                     {Index{2, 1}, 10.0},
                                     {Index{2, 2}, 11.0},
                                     {Index{2, 3}, 12.0},
                                     {Index{3, 0}, 13.0},
                                     {Index{3, 1}, 14.0},
                                     {Index{3, 2}, 15.0},
                                     {Index{3, 3}, 16.0}}};

    auto S = submatrix(A, 1, 3, 0, 3);

    CHECK(S.shape() == Shape(2, 3));
    CHECK(S.size() == 6);
    CHECK(S(0, 0) == Catch::Approx(5.0));
    CHECK(S(0, 1) == Catch::Approx(6.0));
    CHECK(S(0, 2) == Catch::Approx(7.0));
    CHECK(S(1, 0) == Catch::Approx(9.0));
    CHECK(S(1, 1) == Catch::Approx(10.0));
    CHECK(S(1, 2) == Catch::Approx(11.0));
  }

  TEST_CASE("unary - submatrix empty result", "[unary]") {
    // Sparse matrix, submatrix region has no entries
    Compressed_row_matrix<double> A{Shape{4, 4},
                                    {{Index{0, 0}, 1.0}, {Index{3, 3}, 2.0}}};

    auto S = submatrix(A, 0, 2, 2, 4);

    CHECK(S.shape() == Shape(2, 2));
    CHECK(S.size() == 0);
  }

} // end of namespace sparkit::testing
