//
// ... Test header files
//
#include <gtest/gtest.h>

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

  // -- Phase 1: SpMV --

  TEST(sparse_blas, spmv_known_3x3)
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

    ASSERT_EQ(std::ssize(y), 3);
    EXPECT_DOUBLE_EQ(y[0], 11.0);
    EXPECT_DOUBLE_EQ(y[1], 8.0);
    EXPECT_DOUBLE_EQ(y[2], 23.0);
  }

  TEST(sparse_blas, spmv_empty_matrix)
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    ASSERT_EQ(std::ssize(y), 3);
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
  }

  TEST(sparse_blas, spmv_rectangular)
  {
    // 2x3 matrix: A = [[1,0,2],[0,3,0]], x = {1,2,3}
    // y = {1+6, 6} = {7, 6}
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    ASSERT_EQ(std::ssize(y), 2);
    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
  }

  TEST(sparse_blas, spmv_identity)
  {
    Compressed_row_matrix<double> I{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{1, 1}, 1.0},
      {Index{2, 2}, 1.0}
    }};

    std::vector<double> x{4.0, 5.0, 6.0};
    auto y = multiply(I, std::span<double const>{x});

    ASSERT_EQ(std::ssize(y), 3);
    EXPECT_DOUBLE_EQ(y[0], 4.0);
    EXPECT_DOUBLE_EQ(y[1], 5.0);
    EXPECT_DOUBLE_EQ(y[2], 6.0);
  }

  // -- Phase 2: Transpose SpMV --

  TEST(sparse_blas, transpose_spmv_known_3x3)
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

    ASSERT_EQ(std::ssize(y), 3);
    EXPECT_DOUBLE_EQ(y[0], 17.0);
    EXPECT_DOUBLE_EQ(y[1], 8.0);
    EXPECT_DOUBLE_EQ(y[2], 21.0);
  }

  TEST(sparse_blas, transpose_spmv_rectangular)
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

    ASSERT_EQ(std::ssize(y), 3);
    EXPECT_DOUBLE_EQ(y[0], 1.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
    EXPECT_DOUBLE_EQ(y[2], 2.0);
  }

  TEST(sparse_blas, transpose_spmv_symmetric_equals_spmv)
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

    ASSERT_EQ(std::ssize(y1), std::ssize(y2));
    for (std::ptrdiff_t i = 0; i < std::ssize(y1); ++i) {
      EXPECT_DOUBLE_EQ(y1[static_cast<std::size_t>(i)],
                        y2[static_cast<std::size_t>(i)]);
    }
  }

  TEST(sparse_blas, transpose_spmv_empty_matrix)
  {
    Compressed_row_matrix<double> A{Shape{3, 4}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    ASSERT_EQ(std::ssize(y), 4);
    for (auto val : y) {
      EXPECT_DOUBLE_EQ(val, 0.0);
    }
  }

  // -- Phase 3: Diagonal operations --

  TEST(sparse_blas, left_diagonal_scales_rows)
  {
    // A = [[2,3],[4,5]], d = {10, 100}
    // diag(d)*A = [[20,30],[400,500]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{10.0, 100.0};
    auto C = multiply_left_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_DOUBLE_EQ(C(0, 0), 20.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 30.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 400.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 500.0);
  }

  TEST(sparse_blas, left_diagonal_identity)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{1.0, 1.0};
    auto C = multiply_left_diagonal(A, std::span<double const>{d});

    EXPECT_DOUBLE_EQ(C(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 5.0);
  }

  TEST(sparse_blas, right_diagonal_scales_columns)
  {
    // A = [[2,3],[4,5]], d = {10, 100}
    // A*diag(d) = [[20,300],[40,500]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{10.0, 100.0};
    auto C = multiply_right_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_DOUBLE_EQ(C(0, 0), 20.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 300.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 40.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 500.0);
  }

  TEST(sparse_blas, right_diagonal_identity)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    std::vector<double> d{1.0, 1.0};
    auto C = multiply_right_diagonal(A, std::span<double const>{d});

    EXPECT_DOUBLE_EQ(C(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 5.0);
  }

  TEST(sparse_blas, add_diagonal_present)
  {
    // A = [[1,2],[3,4]], d = {10, 20}
    // A + diag(d) = [[11,2],[3,24]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_DOUBLE_EQ(C(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 24.0);
  }

  TEST(sparse_blas, add_diagonal_absent)
  {
    // A has no diagonal entries: A = [[0,2],[3,0]], d = {10, 20}
    // A + diag(d) = [[10,2],[3,20]]
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_EQ(C.size(), 4);
    EXPECT_DOUBLE_EQ(C(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 20.0);
  }

  TEST(sparse_blas, add_diagonal_empty_matrix)
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> d{1.0, 2.0, 3.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(3, 3));
    EXPECT_EQ(C.size(), 3);
    EXPECT_DOUBLE_EQ(C(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(C(2, 2), 3.0);
  }

  TEST(sparse_blas, add_diagonal_rectangular)
  {
    // 2x3 matrix, d has min(2,3)=2 entries
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    std::vector<double> d{10.0, 20.0};
    auto C = add_diagonal(A, std::span<double const>{d});

    EXPECT_EQ(C.shape(), Shape(2, 3));
    EXPECT_DOUBLE_EQ(C(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(C(0, 2), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 23.0);
  }

  // -- Phase 4: Sparse addition --

  TEST(sparse_blas, add_disjoint_patterns)
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

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_EQ(C.size(), 4);
    EXPECT_DOUBLE_EQ(C(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 2.0);
  }

  TEST(sparse_blas, add_identical_patterns)
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

    EXPECT_EQ(C.size(), 4);
    EXPECT_DOUBLE_EQ(C(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 33.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 44.0);
  }

  TEST(sparse_blas, add_partial_overlap)
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

    EXPECT_EQ(C.shape(), Shape(2, 3));
    EXPECT_DOUBLE_EQ(C(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 20.0);
    EXPECT_DOUBLE_EQ(C(0, 2), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 2), 30.0);
  }

  TEST(sparse_blas, add_scaled)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C = add(A, 2.0, B);

    EXPECT_DOUBLE_EQ(C(0, 0), 7.0);  // 1 + 2*3
    EXPECT_DOUBLE_EQ(C(1, 1), 10.0); // 2 + 2*4
  }

  TEST(sparse_blas, add_one_empty)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}
    }};
    Compressed_row_matrix<double> empty{Shape{2, 2}, {}};

    auto C = add(A, empty);

    EXPECT_EQ(C.size(), 2);
    EXPECT_DOUBLE_EQ(C(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 2.0);
  }

  TEST(sparse_blas, add_commutativity)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C1 = add(A, B);
    auto C2 = add(B, A);

    EXPECT_DOUBLE_EQ(C1(0, 0), C2(0, 0));
    EXPECT_DOUBLE_EQ(C1(0, 1), C2(0, 1));
    EXPECT_DOUBLE_EQ(C1(1, 1), C2(1, 1));
  }

  TEST(sparse_blas, add_negation)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 5.0}, {Index{0, 1}, 7.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 9.0}
    }};

    auto C = add(A, -1.0, A);

    for (config::size_type i = 0; i < 2; ++i) {
      for (config::size_type j = 0; j < 2; ++j) {
        EXPECT_DOUBLE_EQ(C(i, j), 0.0);
      }
    }
  }

  // -- Phase 5: Sparse matrix-matrix multiply --

  TEST(sparse_blas, matmul_known_2x2)
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

    EXPECT_EQ(C.shape(), Shape(2, 2));
    EXPECT_DOUBLE_EQ(C(0, 0), 36.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 41.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 64.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 73.0);
  }

  TEST(sparse_blas, matmul_identity_left)
  {
    Compressed_row_matrix<double> I{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 1.0}
    }};
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};

    auto C = multiply(I, A);

    EXPECT_DOUBLE_EQ(C(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 5.0);
  }

  TEST(sparse_blas, matmul_identity_right)
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 2.0}, {Index{0, 1}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}
    }};
    Compressed_row_matrix<double> I{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 1.0}
    }};

    auto C = multiply(A, I);

    EXPECT_DOUBLE_EQ(C(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 5.0);
  }

  TEST(sparse_blas, matmul_result_shape)
  {
    // A(2,3) * B(3,4) = C(2,4)
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{1, 2}, 1.0}
    }};
    Compressed_row_matrix<double> B{Shape{3, 4}, {
      {Index{0, 0}, 1.0}, {Index{2, 3}, 1.0}
    }};

    auto C = multiply(A, B);

    EXPECT_EQ(C.shape(), Shape(2, 4));
  }

  TEST(sparse_blas, matmul_associativity_with_vector)
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

    ASSERT_EQ(std::ssize(lhs), std::ssize(rhs));
    for (std::ptrdiff_t i = 0; i < std::ssize(lhs); ++i) {
      EXPECT_DOUBLE_EQ(lhs[static_cast<std::size_t>(i)],
                        rhs[static_cast<std::size_t>(i)]);
    }
  }

  TEST(sparse_blas, matmul_empty_row_propagation)
  {
    // Row 1 of A is empty, so row 1 of C should also be empty
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}
    }};
    Compressed_row_matrix<double> B{Shape{2, 2}, {
      {Index{0, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    auto C = multiply(A, B);

    EXPECT_DOUBLE_EQ(C(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 0.0);
  }

} // end of namespace sparkit::testing
