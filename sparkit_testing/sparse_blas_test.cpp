//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::multiply_transpose;
  using sparkit::data::detail::multiply_left_diagonal;
  using sparkit::data::detail::multiply_right_diagonal;
  using sparkit::data::detail::add_diagonal;
  using sparkit::data::detail::add;
  using sparkit::data::detail::add_transpose;
  using sparkit::data::detail::transpose;

  // -- Phase 1: SpMV --

  TEST_CASE("sparse_blas - spmv_known_3x3", "[sparse_blas]")
  {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {2*1+3*3, 4*2, 5*1+6*3} = {11, 8, 23}
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 2.0}, {Index{0, 2}, 3.0},
      {Index{1, 1}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(11.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("sparse_blas - spmv_empty_matrix", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("sparse_blas - spmv_rectangular", "[sparse_blas]")
  {
    // 2x3 matrix: A = [[1,0,2],[0,3,0]], x = {1,2,3}
    // y = {1+6, 6} = {7, 6}
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 2);
    CHECK(y[0] == Catch::Approx(7.0));
    CHECK(y[1] == Catch::Approx(6.0));
  }

  TEST_CASE("sparse_blas - spmv_identity", "[sparse_blas]")
  {
    Compressed_row_matrix<double> I{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{1, 1}, 1.0},
      {Index{2, 2}, 1.0}
    }};

    std::vector<double> x{4.0, 5.0, 6.0};
    auto y = multiply(I, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(4.0));
    CHECK(y[1] == Catch::Approx(5.0));
    CHECK(y[2] == Catch::Approx(6.0));
  }

  // -- Phase 2: Transpose SpMV --

  TEST_CASE("sparse_blas - transpose_spmv_known_3x3", "[sparse_blas]")
  {
    // Same A as spmv_known_3x3
    // A^T = [[2,0,5],[0,4,0],[3,0,6]], x = {1,2,3}
    // y = {2*1+5*3, 4*2, 3*1+6*3} = {17, 8, 21}
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 2.0}, {Index{0, 2}, 3.0},
      {Index{1, 1}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("sparse_blas - transpose_spmv_rectangular", "[sparse_blas]")
  {
    // A is 2x3: [[1,0,2],[0,3,0]]
    // A^T is 3x2, x has length 2, y has length 3
    // A^T*x = [[1,0],[0,3],[2,0]] * {1,2} = {1, 6, 2}
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    std::vector<double> x{1.0, 2.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(1.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(2.0));
  }

  TEST_CASE("sparse_blas - transpose_spmv_symmetric_equals_spmv", "[sparse_blas]")
  {
    // For symmetric A, A^T*x = A*x
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}, {Index{0, 2}, 3.0},
      {Index{1, 0}, 2.0}, {Index{1, 1}, 4.0}, {Index{1, 2}, 5.0},
      {Index{2, 0}, 3.0}, {Index{2, 1}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y1 = multiply(A, std::span<double const>{x});
    auto y2 = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y1) == std::ssize(y2));
    for (std::ptrdiff_t i = 0; i < std::ssize(y1); ++i) {
      CHECK(y1[static_cast<std::size_t>(i)]
        == Catch::Approx(y2[static_cast<std::size_t>(i)]));
    }
  }

  TEST_CASE("sparse_blas - transpose_spmv_empty_matrix", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{3, 4}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 4);
    for (auto val : y) {
      CHECK(val == Catch::Approx(0.0));
    }
  }

  // -- Phase 3: Diagonal operations --

  TEST_CASE("sparse_blas - left_diagonal_scales_rows", "[sparse_blas]")
  {
    // A = [[2,3],[4,5]], d = {10, 100}
    // diag(d)*A = [[20,30],[400,500]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{10.0, 100.0};
    auto C = multiply_left_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C(0, 0) == Catch::Approx(20.0));
    CHECK(C(0, 1) == Catch::Approx(30.0));
    CHECK(C(1, 0) == Catch::Approx(400.0));
    CHECK(C(1, 1) == Catch::Approx(500.0));
  }

  TEST_CASE("sparse_blas - left_diagonal_identity", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{1.0, 1.0};
    auto C = multiply_left_diagonal(A, std::span<double const>{d});

    CHECK(C(0, 0) == Catch::Approx(2.0));
    CHECK(C(0, 1) == Catch::Approx(3.0));
    CHECK(C(1, 0) == Catch::Approx(4.0));
    CHECK(C(1, 1) == Catch::Approx(5.0));
  }

  TEST_CASE("sparse_blas - right_diagonal_scales_columns", "[sparse_blas]")
  {
    // A = [[2,3],[4,5]], d = {10, 100}
    // A*diag(d) = [[20,300],[40,500]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{10.0, 100.0};
    auto C = multiply_right_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C(0, 0) == Catch::Approx(20.0));
    CHECK(C(0, 1) == Catch::Approx(300.0));
    CHECK(C(1, 0) == Catch::Approx(40.0));
    CHECK(C(1, 1) == Catch::Approx(500.0));
  }

  TEST_CASE("sparse_blas - right_diagonal_identity", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{1.0, 1.0};
    auto C = multiply_right_diagonal(A, std::span<double const>{d});

    CHECK(C(0, 0) == Catch::Approx(2.0));
    CHECK(C(0, 1) == Catch::Approx(3.0));
    CHECK(C(1, 0) == Catch::Approx(4.0));
    CHECK(C(1, 1) == Catch::Approx(5.0));
  }

  TEST_CASE("sparse_blas - add_diagonal_present", "[sparse_blas]")
  {
    // A = [[1,2],[3,4]], d = {10, 20}
    // A + diag(d) = [[11,2],[3,24]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C(0, 0) == Catch::Approx(11.0));
    CHECK(C(0, 1) == Catch::Approx(2.0));
    CHECK(C(1, 0) == Catch::Approx(3.0));
    CHECK(C(1, 1) == Catch::Approx(24.0));
  }

  TEST_CASE("sparse_blas - add_diagonal_absent", "[sparse_blas]")
  {
    // A has no diagonal entries: A = [[0,2],[3,0]], d = {10, 20}
    // A + diag(d) = [[10,2],[3,20]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C.size() == 4);
    CHECK(C(0, 0) == Catch::Approx(10.0));
    CHECK(C(0, 1) == Catch::Approx(2.0));
    CHECK(C(1, 0) == Catch::Approx(3.0));
    CHECK(C(1, 1) == Catch::Approx(20.0));
  }

  TEST_CASE("sparse_blas - add_diagonal_empty_matrix", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> d{1.0, 2.0, 3.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(3, 3));
    CHECK(C.size() == 3);
    CHECK(C(0, 0) == Catch::Approx(1.0));
    CHECK(C(1, 1) == Catch::Approx(2.0));
    CHECK(C(2, 2) == Catch::Approx(3.0));
  }

  TEST_CASE("sparse_blas - add_diagonal_rectangular", "[sparse_blas]")
  {
    // 2x3 matrix, d has min(2,3)=2 entries
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    CHECK(C.shape() == Shape(2, 3));
    CHECK(C(0, 0) == Catch::Approx(11.0));
    CHECK(C(0, 2) == Catch::Approx(2.0));
    CHECK(C(1, 1) == Catch::Approx(23.0));
  }

  // -- Phase 4: Sparse addition --

  TEST_CASE("sparse_blas - add_disjoint_patterns", "[sparse_blas]")
  {
    // A has entries at (0,0) and (1,1)
    // B has entries at (0,1) and (1,0)
    // C = A + B has all four entries
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 1}, 3.0}, {Index{1, 0}, 4.0}
    }};

    auto C = add(A, B);

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C.size() == 4);
    CHECK(C(0, 0) == Catch::Approx(1.0));
    CHECK(C(0, 1) == Catch::Approx(3.0));
    CHECK(C(1, 0) == Catch::Approx(4.0));
    CHECK(C(1, 1) == Catch::Approx(2.0));
  }

  TEST_CASE("sparse_blas - add_identical_patterns", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 10.0}, {Index{0, 1}, 20.0},
      {Index{1, 0}, 30.0}, {Index{1, 1}, 40.0}
    }};

    auto C = add(A, B);

    CHECK(C.size() == 4);
    CHECK(C(0, 0) == Catch::Approx(11.0));
    CHECK(C(0, 1) == Catch::Approx(22.0));
    CHECK(C(1, 0) == Catch::Approx(33.0));
    CHECK(C(1, 1) == Catch::Approx(44.0));
  }

  TEST_CASE("sparse_blas - add_partial_overlap", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 3}, {
      {Index{0, 0}, 10.0}, {Index{0, 1}, 20.0},
      {Index{1, 2}, 30.0}
    }};

    auto C = add(A, B);

    CHECK(C.shape() == Shape(2, 3));
    CHECK(C(0, 0) == Catch::Approx(11.0));
    CHECK(C(0, 1) == Catch::Approx(20.0));
    CHECK(C(0, 2) == Catch::Approx(2.0));
    CHECK(C(1, 1) == Catch::Approx(3.0));
    CHECK(C(1, 2) == Catch::Approx(30.0));
  }

  TEST_CASE("sparse_blas - add_scaled", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C = add(A, 2.0, B);

    CHECK(C(0, 0) == Catch::Approx(7.0));  // 1 + 2*3
    CHECK(C(1, 1) == Catch::Approx(10.0)); // 2 + 2*4
  }

  TEST_CASE("sparse_blas - add_one_empty", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}
    }};
    Compressed_row_matrix<double> empty{Shape{2, 2}, {}};

    auto C = add(A, empty);

    CHECK(C.size() == 2);
    CHECK(C(0, 0) == Catch::Approx(1.0));
    CHECK(C(1, 1) == Catch::Approx(2.0));
  }

  TEST_CASE("sparse_blas - add_commutativity", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C1 = add(A, B);
    auto C2 = add(B, A);

    CHECK(C1(0, 0) == Catch::Approx(C2(0, 0)));
    CHECK(C1(0, 1) == Catch::Approx(C2(0, 1)));
    CHECK(C1(1, 1) == Catch::Approx(C2(1, 1)));
  }

  TEST_CASE("sparse_blas - add_negation", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 5.0}, {Index{0, 1}, 7.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 9.0}
    }};

    auto C = add(A, -1.0, A);

    for (config::size_type i = 0; i < 2; ++i) {
      for (config::size_type j = 0; j < 2; ++j) {
        CHECK(C(i, j) == Catch::Approx(0.0));
      }
    }
  }

  // -- Phase 4b: Sparse transpose-add (APLSBT) --

  TEST_CASE("sparse_blas - add_transpose_known_result", "[sparse_blas]")
  {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // B^T = [[5,7],[6,8]]
    // C = A + B^T = [[6,9],[9,12]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 5.0}, {Index{0, 1}, 6.0},
      {Index{1, 0}, 7.0}, {Index{1, 1}, 8.0}
    }};

    auto C = add_transpose(A, B);

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C(0, 0) == Catch::Approx(6.0));
    CHECK(C(0, 1) == Catch::Approx(9.0));
    CHECK(C(1, 0) == Catch::Approx(9.0));
    CHECK(C(1, 1) == Catch::Approx(12.0));
  }

  TEST_CASE("sparse_blas - add_transpose_cross_validation", "[sparse_blas]")
  {
    // add_transpose(A, s, B) == add(A, s, transpose(B))
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}, {Index{2, 2}, 5.0}
    }};
    Compressed_row_matrix<double> B{Shape{3, 3}, {
      {Index{0, 0}, 10.0}, {Index{0, 1}, 20.0},
      {Index{1, 2}, 30.0},
      {Index{2, 1}, 40.0}
    }};
    double s = 2.5;

    auto fused = add_transpose(A, s, B);
    auto naive = add(A, s, transpose(B));

    CHECK(fused.shape() == naive.shape());
    for (config::size_type i = 0; i < 3; ++i) {
      for (config::size_type j = 0; j < 3; ++j) {
        CHECK(fused(i, j) == Catch::Approx(naive(i, j)));
      }
    }
  }

  TEST_CASE("sparse_blas - add_transpose_rectangular", "[sparse_blas]")
  {
    // A is (2,3), B is (3,2), B^T is (2,3)
    // C = A + B^T is (2,3)
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};
    Compressed_row_matrix<double> B{Shape{3, 2}, {
      {Index{0, 0}, 10.0}, {Index{0, 1}, 20.0},
      {Index{1, 0}, 30.0},
      {Index{2, 1}, 40.0}
    }};

    auto C = add_transpose(A, B);

    CHECK(C.shape() == Shape(2, 3));

    auto naive = add(A, transpose(B));
    for (config::size_type i = 0; i < 2; ++i) {
      for (config::size_type j = 0; j < 3; ++j) {
        CHECK(C(i, j) == Catch::Approx(naive(i, j)));
      }
    }
  }

  TEST_CASE("sparse_blas - add_transpose_identity", "[sparse_blas]")
  {
    // A + I^T = A + I
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}
    }};
    Compressed_row_matrix<double> I{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{1, 1}, 1.0},
      {Index{2, 2}, 1.0}
    }};

    auto C = add_transpose(A, I);
    auto expected = add(A, I);

    for (config::size_type i = 0; i < 3; ++i) {
      for (config::size_type j = 0; j < 3; ++j) {
        CHECK(C(i, j) == Catch::Approx(expected(i, j)));
      }
    }
  }

  TEST_CASE("sparse_blas - add_transpose_symmetric_B", "[sparse_blas]")
  {
    // For symmetric B, add_transpose(A, B) == add(A, B)
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 5.0}, {Index{0, 1}, 6.0},
      {Index{1, 0}, 6.0}, {Index{1, 1}, 7.0}
    }};

    auto with_transpose = add_transpose(A, B);
    auto direct = add(A, B);

    for (config::size_type i = 0; i < 2; ++i) {
      for (config::size_type j = 0; j < 2; ++j) {
        CHECK(with_transpose(i, j) == Catch::Approx(direct(i, j)));
      }
    }
  }

  TEST_CASE("sparse_blas - add_transpose_scale_zero", "[sparse_blas]")
  {
    // add_transpose(A, 0, B) == A (only A's pattern)
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 99.0}, {Index{0, 1}, 99.0},
      {Index{1, 0}, 99.0}, {Index{1, 1}, 99.0}
    }};

    auto C = add_transpose(A, 0.0, B);

    for (config::size_type i = 0; i < 2; ++i) {
      for (config::size_type j = 0; j < 2; ++j) {
        CHECK(C(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("sparse_blas - add_transpose_empty_B", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};
    Compressed_row_matrix<double> empty{Shape{3, 2}, {}};

    auto C = add_transpose(A, 2.0, empty);

    CHECK(C.shape() == Shape(2, 3));
    CHECK(C.size() == 3);
    CHECK(C(0, 0) == Catch::Approx(1.0));
    CHECK(C(0, 2) == Catch::Approx(2.0));
    CHECK(C(1, 1) == Catch::Approx(3.0));
  }

  // -- Phase 5: Sparse matrix-matrix multiply --

  TEST_CASE("sparse_blas - matmul_known_2x2", "[sparse_blas]")
  {
    // [[2,3],[4,5]] * [[6,7],[8,9]] = [[36,41],[64,73]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 6.0}, {Index{0, 1}, 7.0},
      {Index{1, 0}, 8.0}, {Index{1, 1}, 9.0}
    }};

    auto C = multiply(A, B);

    CHECK(C.shape() == Shape(2, 2));
    CHECK(C(0, 0) == Catch::Approx(36.0));
    CHECK(C(0, 1) == Catch::Approx(41.0));
    CHECK(C(1, 0) == Catch::Approx(64.0));
    CHECK(C(1, 1) == Catch::Approx(73.0));
  }

  TEST_CASE("sparse_blas - matmul_identity_left", "[sparse_blas]")
  {
    Compressed_row_matrix<double> I{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 1.0}
    }};
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    auto C = multiply(I, A);

    CHECK(C(0, 0) == Catch::Approx(2.0));
    CHECK(C(0, 1) == Catch::Approx(3.0));
    CHECK(C(1, 0) == Catch::Approx(4.0));
    CHECK(C(1, 1) == Catch::Approx(5.0));
  }

  TEST_CASE("sparse_blas - matmul_identity_right", "[sparse_blas]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};
    Compressed_row_matrix<double> I{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 1.0}
    }};

    auto C = multiply(A, I);

    CHECK(C(0, 0) == Catch::Approx(2.0));
    CHECK(C(0, 1) == Catch::Approx(3.0));
    CHECK(C(1, 0) == Catch::Approx(4.0));
    CHECK(C(1, 1) == Catch::Approx(5.0));
  }

  TEST_CASE("sparse_blas - matmul_result_shape", "[sparse_blas]")
  {
    // A(2,3) * B(3,4) = C(2,4)
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{1, 2}, 1.0}
    }};
    Compressed_row_matrix<double> B{Shape{3, 4}, {
      {Index{0, 0}, 1.0}, {Index{2, 3}, 1.0}
    }};

    auto C = multiply(A, B);

    CHECK(C.shape() == Shape(2, 4));
  }

  TEST_CASE("sparse_blas - matmul_associativity_with_vector", "[sparse_blas]")
  {
    // A*(B*x) should equal (A*B)*x
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};
    Compressed_row_matrix<double> B{Shape{3, 2}, {
      {Index{0, 0}, 4.0}, {Index{0, 1}, 5.0},
      {Index{1, 0}, 6.0},
      {Index{2, 1}, 7.0}
    }};

    std::vector<double> x{1.0, 2.0};

    auto Bx = multiply(B, std::span<double const>{x});
    auto lhs = multiply(A, std::span<double const>{Bx});

    auto AB = multiply(A, B);
    auto rhs = multiply(AB, std::span<double const>{x});

    REQUIRE(std::ssize(lhs) == std::ssize(rhs));
    for (std::ptrdiff_t i = 0; i < std::ssize(lhs); ++i) {
      CHECK(lhs[static_cast<std::size_t>(i)]
        == Catch::Approx(rhs[static_cast<std::size_t>(i)]));
    }
  }

  TEST_CASE("sparse_blas - matmul_empty_row_propagation", "[sparse_blas]")
  {
    // Row 1 of A is empty, so row 1 of C should also be empty
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C = multiply(A, B);

    CHECK(C(0, 0) == Catch::Approx(3.0));
    CHECK(C(0, 1) == Catch::Approx(8.0));
    CHECK(C(1, 0) == Catch::Approx(0.0));
    CHECK(C(1, 1) == Catch::Approx(0.0));
  }

} // end of namespace sparkit::testing
