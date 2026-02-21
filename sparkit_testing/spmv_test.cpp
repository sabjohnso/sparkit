//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Block_sparse_row_matrix.hpp>
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Diagonal_matrix.hpp>
#include <sparkit/data/Ellpack_matrix.hpp>
#include <sparkit/data/Jagged_diagonal_matrix.hpp>
#include <sparkit/data/Modified_sparse_row_matrix.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/spmv.hpp>
#include <sparkit/data/Symmetric_compressed_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Block_sparse_row_matrix;
  using sparkit::data::detail::Compressed_column_matrix;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Diagonal_matrix;
  using sparkit::data::detail::Ellpack_matrix;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Jagged_diagonal_matrix;
  using sparkit::data::detail::Modified_sparse_row_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::multiply_transpose;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_compressed_row_matrix;

  // ================================================================
  // CSC SpMV
  // ================================================================

  TEST_CASE("spmv - csc_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {2*1+3*3, 4*2, 5*1+6*3} = {11, 8, 23}
    Compressed_column_matrix<double> A{Shape{3, 3},
                                       {{Index{0, 0}, 2.0},
                                        {Index{0, 2}, 3.0},
                                        {Index{1, 1}, 4.0},
                                        {Index{2, 0}, 5.0},
                                        {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(11.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - csc_empty_matrix", "[spmv]") {
    Compressed_column_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - csc_rectangular", "[spmv]") {
    // 2x3: A = [[1,0,2],[0,3,0]], x = {1,2,3}
    // y = {1+6, 6} = {7, 6}
    Compressed_column_matrix<double> A{
        Shape{2, 3},
        {{Index{0, 0}, 1.0}, {Index{0, 2}, 2.0}, {Index{1, 1}, 3.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 2);
    CHECK(y[0] == Catch::Approx(7.0));
    CHECK(y[1] == Catch::Approx(6.0));
  }

  TEST_CASE("spmv - csc_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Compressed_column_matrix<double> csc{Shape{3, 3},
                                         {{Index{0, 0}, 2.0},
                                          {Index{0, 2}, 3.0},
                                          {Index{1, 1}, 4.0},
                                          {Index{2, 0}, 5.0},
                                          {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_csc = multiply(csc, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_csc));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_csc[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // MSR SpMV
  // ================================================================

  TEST_CASE("spmv - msr_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {11, 8, 23}
    Modified_sparse_row_matrix<double> A{Shape{3, 3},
                                         {{Index{0, 0}, 2.0},
                                          {Index{0, 2}, 3.0},
                                          {Index{1, 1}, 4.0},
                                          {Index{2, 0}, 5.0},
                                          {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(11.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - msr_empty_matrix", "[spmv]") {
    Modified_sparse_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - msr_diagonal_only", "[spmv]") {
    Modified_sparse_row_matrix<double> A{
        Shape{3, 3},
        {{Index{0, 0}, 2.0}, {Index{1, 1}, 3.0}, {Index{2, 2}, 5.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(2.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(15.0));
  }

  TEST_CASE("spmv - msr_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Modified_sparse_row_matrix<double> msr{Shape{3, 3},
                                           {{Index{0, 0}, 2.0},
                                            {Index{0, 2}, 3.0},
                                            {Index{1, 1}, 4.0},
                                            {Index{2, 0}, 5.0},
                                            {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_msr = multiply(msr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_msr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_msr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // DIA SpMV
  // ================================================================

  TEST_CASE("spmv - dia_tridiagonal", "[spmv]") {
    // Tridiagonal: A = [[2,1,0],[1,3,1],[0,1,4]], x = {1,2,3}
    // y = {2+2, 1+6+3, 2+12} = {4, 10, 14}
    Diagonal_matrix<double> A{Shape{3, 3},
                              {{Index{0, 0}, 2.0},
                               {Index{0, 1}, 1.0},
                               {Index{1, 0}, 1.0},
                               {Index{1, 1}, 3.0},
                               {Index{1, 2}, 1.0},
                               {Index{2, 1}, 1.0},
                               {Index{2, 2}, 4.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(4.0));
    CHECK(y[1] == Catch::Approx(10.0));
    CHECK(y[2] == Catch::Approx(14.0));
  }

  TEST_CASE("spmv - dia_main_diagonal_only", "[spmv]") {
    Diagonal_matrix<double> A{
        Shape{3, 3},
        {{Index{0, 0}, 2.0}, {Index{1, 1}, 3.0}, {Index{2, 2}, 5.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(2.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(15.0));
  }

  TEST_CASE("spmv - dia_empty_matrix", "[spmv]") {
    Diagonal_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - dia_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 1}, 1.0},
                                       {Index{1, 0}, 1.0},
                                       {Index{1, 1}, 3.0},
                                       {Index{1, 2}, 1.0},
                                       {Index{2, 1}, 1.0},
                                       {Index{2, 2}, 4.0}}};
    Diagonal_matrix<double> dia{Shape{3, 3},
                                {{Index{0, 0}, 2.0},
                                 {Index{0, 1}, 1.0},
                                 {Index{1, 0}, 1.0},
                                 {Index{1, 1}, 3.0},
                                 {Index{1, 2}, 1.0},
                                 {Index{2, 1}, 1.0},
                                 {Index{2, 2}, 4.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_dia = multiply(dia, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_dia));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_dia[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // ELL SpMV
  // ================================================================

  TEST_CASE("spmv - ell_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {11, 8, 23}
    Ellpack_matrix<double> A{Shape{3, 3},
                             {{Index{0, 0}, 2.0},
                              {Index{0, 2}, 3.0},
                              {Index{1, 1}, 4.0},
                              {Index{2, 0}, 5.0},
                              {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(11.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - ell_varied_row_lengths", "[spmv]") {
    // Row 0: 3 entries, Row 1: 1 entry, Row 2: 2 entries
    // A = [[1,2,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {1+4+9, 8, 5+18} = {14, 8, 23}
    Ellpack_matrix<double> A{Shape{3, 3},
                             {{Index{0, 0}, 1.0},
                              {Index{0, 1}, 2.0},
                              {Index{0, 2}, 3.0},
                              {Index{1, 1}, 4.0},
                              {Index{2, 0}, 5.0},
                              {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(14.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - ell_empty_matrix", "[spmv]") {
    Ellpack_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - ell_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Ellpack_matrix<double> ell{Shape{3, 3},
                               {{Index{0, 0}, 2.0},
                                {Index{0, 2}, 3.0},
                                {Index{1, 1}, 4.0},
                                {Index{2, 0}, 5.0},
                                {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_ell = multiply(ell, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_ell));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_ell[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // BSR SpMV
  // ================================================================

  TEST_CASE("spmv - bsr_known_4x4", "[spmv]") {
    // 4x4 with 2x2 blocks:
    // A = [[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]]
    // x = {1,2,3,4}
    // y = {1+4, 3+8, 15+24, 21+32} = {5, 11, 39, 53}
    Block_sparse_row_matrix<double> A{Shape{4, 4},
                                      2,
                                      2,
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 2}, 5.0},
                                       {Index{2, 3}, 6.0},
                                       {Index{3, 2}, 7.0},
                                       {Index{3, 3}, 8.0}}};

    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 4);
    CHECK(y[0] == Catch::Approx(5.0));
    CHECK(y[1] == Catch::Approx(11.0));
    CHECK(y[2] == Catch::Approx(39.0));
    CHECK(y[3] == Catch::Approx(53.0));
  }

  TEST_CASE("spmv - bsr_mixed_blocks", "[spmv]") {
    // 4x4 with 2x2 blocks, off-diagonal blocks present:
    // A = [[1,2,5,6],[3,4,7,8],[0,0,9,10],[0,0,11,12]]
    // x = {1,2,3,4}
    // y = {1+4+15+24, 3+8+21+32, 27+40, 33+48} = {44, 64, 67, 81}
    Block_sparse_row_matrix<double> A{Shape{4, 4},
                                      2,
                                      2,
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{0, 2}, 5.0},
                                       {Index{0, 3}, 6.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{1, 2}, 7.0},
                                       {Index{1, 3}, 8.0},
                                       {Index{2, 2}, 9.0},
                                       {Index{2, 3}, 10.0},
                                       {Index{3, 2}, 11.0},
                                       {Index{3, 3}, 12.0}}};

    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 4);
    CHECK(y[0] == Catch::Approx(44.0));
    CHECK(y[1] == Catch::Approx(64.0));
    CHECK(y[2] == Catch::Approx(67.0));
    CHECK(y[3] == Catch::Approx(81.0));
  }

  TEST_CASE("spmv - bsr_empty_matrix", "[spmv]") {
    Block_sparse_row_matrix<double> A{Shape{4, 4}, 2, 2, {}};

    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 4);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
    CHECK(y[3] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - bsr_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{4, 4},
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 2}, 5.0},
                                       {Index{2, 3}, 6.0},
                                       {Index{3, 2}, 7.0},
                                       {Index{3, 3}, 8.0}}};
    Block_sparse_row_matrix<double> bsr{Shape{4, 4},
                                        2,
                                        2,
                                        {{Index{0, 0}, 1.0},
                                         {Index{0, 1}, 2.0},
                                         {Index{1, 0}, 3.0},
                                         {Index{1, 1}, 4.0},
                                         {Index{2, 2}, 5.0},
                                         {Index{2, 3}, 6.0},
                                         {Index{3, 2}, 7.0},
                                         {Index{3, 3}, 8.0}}};

    std::vector<double> x{7.0, 11.0, 13.0, 17.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_bsr = multiply(bsr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_bsr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_bsr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // JAD SpMV
  // ================================================================

  TEST_CASE("spmv - jad_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {11, 8, 23}
    Jagged_diagonal_matrix<double> A{Shape{3, 3},
                                     {{Index{0, 0}, 2.0},
                                      {Index{0, 2}, 3.0},
                                      {Index{1, 1}, 4.0},
                                      {Index{2, 0}, 5.0},
                                      {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(11.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - jad_permuted_rows", "[spmv]") {
    // Row nnz: row0=3, row1=1, row2=2 => perm puts row0 first, row2 second
    // A = [[1,2,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // y = {14, 8, 23}
    Jagged_diagonal_matrix<double> A{Shape{3, 3},
                                     {{Index{0, 0}, 1.0},
                                      {Index{0, 1}, 2.0},
                                      {Index{0, 2}, 3.0},
                                      {Index{1, 1}, 4.0},
                                      {Index{2, 0}, 5.0},
                                      {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(14.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(23.0));
  }

  TEST_CASE("spmv - jad_empty_matrix", "[spmv]") {
    Jagged_diagonal_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - jad_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Jagged_diagonal_matrix<double> jad{Shape{3, 3},
                                       {{Index{0, 0}, 2.0},
                                        {Index{0, 2}, 3.0},
                                        {Index{1, 1}, 4.0},
                                        {Index{2, 0}, 5.0},
                                        {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_jad = multiply(jad, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_jad));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_jad[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // sCSR SpMV
  // ================================================================

  TEST_CASE("spmv - scsr_known_3x3", "[spmv]") {
    // Symmetric: A = [[4,1,0],[1,5,2],[0,2,6]], x = {1,2,3}
    // y = {4+2, 1+10+6, 4+18} = {6, 17, 22}
    Symmetric_compressed_row_matrix<double> A{Shape{3, 3},
                                              {{Index{0, 0}, 4.0},
                                               {Index{0, 1}, 1.0},
                                               {Index{1, 1}, 5.0},
                                               {Index{1, 2}, 2.0},
                                               {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(6.0));
    CHECK(y[1] == Catch::Approx(17.0));
    CHECK(y[2] == Catch::Approx(22.0));
  }

  TEST_CASE("spmv - scsr_diagonal_only", "[spmv]") {
    Symmetric_compressed_row_matrix<double> A{
        Shape{3, 3},
        {{Index{0, 0}, 2.0}, {Index{1, 1}, 3.0}, {Index{2, 2}, 5.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(2.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(15.0));
  }

  TEST_CASE("spmv - scsr_off_diagonal_symmetry", "[spmv]") {
    // Full symmetric: A = [[1,2,3],[2,4,5],[3,5,6]], x = {1,2,3}
    // y = {1+4+9, 2+8+15, 3+10+18} = {14, 25, 31}
    Symmetric_compressed_row_matrix<double> A{Shape{3, 3},
                                              {{Index{0, 0}, 1.0},
                                               {Index{0, 1}, 2.0},
                                               {Index{0, 2}, 3.0},
                                               {Index{1, 1}, 4.0},
                                               {Index{1, 2}, 5.0},
                                               {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(14.0));
    CHECK(y[1] == Catch::Approx(25.0));
    CHECK(y[2] == Catch::Approx(31.0));
  }

  TEST_CASE("spmv - scsr_empty_matrix", "[spmv]") {
    Symmetric_compressed_row_matrix<double> A{Shape{3, 3}, {}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(0.0));
    CHECK(y[1] == Catch::Approx(0.0));
    CHECK(y[2] == Catch::Approx(0.0));
  }

  TEST_CASE("spmv - scsr_cross_validate_with_csr", "[spmv]") {
    // Build equivalent full CSR from symmetric input
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 4.0},
                                       {Index{0, 1}, 1.0},
                                       {Index{1, 0}, 1.0},
                                       {Index{1, 1}, 5.0},
                                       {Index{1, 2}, 2.0},
                                       {Index{2, 1}, 2.0},
                                       {Index{2, 2}, 6.0}}};
    Symmetric_compressed_row_matrix<double> scsr{Shape{3, 3},
                                                 {{Index{0, 0}, 4.0},
                                                  {Index{0, 1}, 1.0},
                                                  {Index{1, 1}, 5.0},
                                                  {Index{1, 2}, 2.0},
                                                  {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply(csr, std::span<double const>{x});
    auto y_scsr = multiply(scsr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_scsr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_scsr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // CSC transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - csc_transpose_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // A^T = [[2,0,5],[0,4,0],[3,0,6]]
    // A^T*x = {2+15, 8, 3+18} = {17, 8, 21}
    Compressed_column_matrix<double> A{Shape{3, 3},
                                       {{Index{0, 0}, 2.0},
                                        {Index{0, 2}, 3.0},
                                        {Index{1, 1}, 4.0},
                                        {Index{2, 0}, 5.0},
                                        {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("spmv - csc_transpose_rectangular", "[spmv]") {
    // A is 2x3: [[1,0,2],[0,3,0]], x = {1,2}
    // A^T is 3x2: A^T*x = {1, 6, 2} — output length 3
    Compressed_column_matrix<double> A{
        Shape{2, 3},
        {{Index{0, 0}, 1.0}, {Index{0, 2}, 2.0}, {Index{1, 1}, 3.0}}};

    std::vector<double> x{1.0, 2.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(1.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(2.0));
  }

  TEST_CASE("spmv - csc_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Compressed_column_matrix<double> csc{Shape{3, 3},
                                         {{Index{0, 0}, 2.0},
                                          {Index{0, 2}, 3.0},
                                          {Index{1, 1}, 4.0},
                                          {Index{2, 0}, 5.0},
                                          {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_csc = multiply_transpose(csc, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_csc));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_csc[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // MSR transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - msr_transpose_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // A^T*x = {17, 8, 21}
    Modified_sparse_row_matrix<double> A{Shape{3, 3},
                                         {{Index{0, 0}, 2.0},
                                          {Index{0, 2}, 3.0},
                                          {Index{1, 1}, 4.0},
                                          {Index{2, 0}, 5.0},
                                          {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("spmv - msr_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Modified_sparse_row_matrix<double> msr{Shape{3, 3},
                                           {{Index{0, 0}, 2.0},
                                            {Index{0, 2}, 3.0},
                                            {Index{1, 1}, 4.0},
                                            {Index{2, 0}, 5.0},
                                            {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_msr = multiply_transpose(msr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_msr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_msr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // DIA transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - dia_transpose_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // A^T*x = {17, 8, 21}
    Diagonal_matrix<double> A{Shape{3, 3},
                              {{Index{0, 0}, 2.0},
                               {Index{0, 2}, 3.0},
                               {Index{1, 1}, 4.0},
                               {Index{2, 0}, 5.0},
                               {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("spmv - dia_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 1}, 1.0},
                                       {Index{1, 0}, 1.0},
                                       {Index{1, 1}, 3.0},
                                       {Index{1, 2}, 1.0},
                                       {Index{2, 1}, 1.0},
                                       {Index{2, 2}, 4.0}}};
    Diagonal_matrix<double> dia{Shape{3, 3},
                                {{Index{0, 0}, 2.0},
                                 {Index{0, 1}, 1.0},
                                 {Index{1, 0}, 1.0},
                                 {Index{1, 1}, 3.0},
                                 {Index{1, 2}, 1.0},
                                 {Index{2, 1}, 1.0},
                                 {Index{2, 2}, 4.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_dia = multiply_transpose(dia, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_dia));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_dia[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // ELL transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - ell_transpose_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // A^T*x = {17, 8, 21}
    Ellpack_matrix<double> A{Shape{3, 3},
                             {{Index{0, 0}, 2.0},
                              {Index{0, 2}, 3.0},
                              {Index{1, 1}, 4.0},
                              {Index{2, 0}, 5.0},
                              {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("spmv - ell_transpose_rectangular", "[spmv]") {
    // A is 2x3: [[1,0,2],[0,3,0]], x = {1,2}
    // A^T*x = {1, 6, 2} — output length 3
    Ellpack_matrix<double> A{
        Shape{2, 3},
        {{Index{0, 0}, 1.0}, {Index{0, 2}, 2.0}, {Index{1, 1}, 3.0}}};

    std::vector<double> x{1.0, 2.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(1.0));
    CHECK(y[1] == Catch::Approx(6.0));
    CHECK(y[2] == Catch::Approx(2.0));
  }

  TEST_CASE("spmv - ell_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Ellpack_matrix<double> ell{Shape{3, 3},
                               {{Index{0, 0}, 2.0},
                                {Index{0, 2}, 3.0},
                                {Index{1, 1}, 4.0},
                                {Index{2, 0}, 5.0},
                                {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_ell = multiply_transpose(ell, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_ell));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_ell[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // BSR transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - bsr_transpose_known_4x4", "[spmv]") {
    // A = [[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]], x = {1,2,3,4}
    // A^T = [[1,3,0,0],[2,4,0,0],[0,0,5,7],[0,0,6,8]]
    // A^T*x = {7, 10, 43, 50}
    Block_sparse_row_matrix<double> A{Shape{4, 4},
                                      2,
                                      2,
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 2}, 5.0},
                                       {Index{2, 3}, 6.0},
                                       {Index{3, 2}, 7.0},
                                       {Index{3, 3}, 8.0}}};

    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 4);
    CHECK(y[0] == Catch::Approx(7.0));
    CHECK(y[1] == Catch::Approx(10.0));
    CHECK(y[2] == Catch::Approx(43.0));
    CHECK(y[3] == Catch::Approx(50.0));
  }

  TEST_CASE("spmv - bsr_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{4, 4},
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 2}, 5.0},
                                       {Index{2, 3}, 6.0},
                                       {Index{3, 2}, 7.0},
                                       {Index{3, 3}, 8.0}}};
    Block_sparse_row_matrix<double> bsr{Shape{4, 4},
                                        2,
                                        2,
                                        {{Index{0, 0}, 1.0},
                                         {Index{0, 1}, 2.0},
                                         {Index{1, 0}, 3.0},
                                         {Index{1, 1}, 4.0},
                                         {Index{2, 2}, 5.0},
                                         {Index{2, 3}, 6.0},
                                         {Index{3, 2}, 7.0},
                                         {Index{3, 3}, 8.0}}};

    std::vector<double> x{7.0, 11.0, 13.0, 17.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_bsr = multiply_transpose(bsr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_bsr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_bsr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // JAD transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - jad_transpose_known_3x3", "[spmv]") {
    // A = [[2,0,3],[0,4,0],[5,0,6]], x = {1,2,3}
    // A^T*x = {17, 8, 21}
    Jagged_diagonal_matrix<double> A{Shape{3, 3},
                                     {{Index{0, 0}, 2.0},
                                      {Index{0, 2}, 3.0},
                                      {Index{1, 1}, 4.0},
                                      {Index{2, 0}, 5.0},
                                      {Index{2, 2}, 6.0}}};

    std::vector<double> x{1.0, 2.0, 3.0};
    auto y = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y) == 3);
    CHECK(y[0] == Catch::Approx(17.0));
    CHECK(y[1] == Catch::Approx(8.0));
    CHECK(y[2] == Catch::Approx(21.0));
  }

  TEST_CASE("spmv - jad_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 2}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 0}, 5.0},
                                       {Index{2, 2}, 6.0}}};
    Jagged_diagonal_matrix<double> jad{Shape{3, 3},
                                       {{Index{0, 0}, 2.0},
                                        {Index{0, 2}, 3.0},
                                        {Index{1, 1}, 4.0},
                                        {Index{2, 0}, 5.0},
                                        {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_jad = multiply_transpose(jad, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_jad));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_jad[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

  // ================================================================
  // sCSR transpose-SpMV
  // ================================================================

  TEST_CASE("spmv - scsr_transpose_equals_forward", "[spmv]") {
    // For symmetric A, A^T = A, so multiply_transpose == multiply
    Symmetric_compressed_row_matrix<double> A{Shape{3, 3},
                                              {{Index{0, 0}, 4.0},
                                               {Index{0, 1}, 1.0},
                                               {Index{1, 1}, 5.0},
                                               {Index{1, 2}, 2.0},
                                               {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_fwd = multiply(A, std::span<double const>{x});
    auto y_trans = multiply_transpose(A, std::span<double const>{x});

    REQUIRE(std::ssize(y_fwd) == std::ssize(y_trans));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_fwd); ++i) {
      CHECK(y_trans[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_fwd[static_cast<std::size_t>(i)]));
    }
  }

  TEST_CASE("spmv - scsr_transpose_cross_validate_with_csr", "[spmv]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 4.0},
                                       {Index{0, 1}, 1.0},
                                       {Index{1, 0}, 1.0},
                                       {Index{1, 1}, 5.0},
                                       {Index{1, 2}, 2.0},
                                       {Index{2, 1}, 2.0},
                                       {Index{2, 2}, 6.0}}};
    Symmetric_compressed_row_matrix<double> scsr{Shape{3, 3},
                                                 {{Index{0, 0}, 4.0},
                                                  {Index{0, 1}, 1.0},
                                                  {Index{1, 1}, 5.0},
                                                  {Index{1, 2}, 2.0},
                                                  {Index{2, 2}, 6.0}}};

    std::vector<double> x{7.0, 11.0, 13.0};
    auto y_csr = multiply_transpose(csr, std::span<double const>{x});
    auto y_scsr = multiply_transpose(scsr, std::span<double const>{x});

    REQUIRE(std::ssize(y_csr) == std::ssize(y_scsr));
    for (std::ptrdiff_t i = 0; i < std::ssize(y_csr); ++i) {
      CHECK(y_scsr[static_cast<std::size_t>(i)] ==
            Catch::Approx(y_csr[static_cast<std::size_t>(i)]));
    }
  }

} // end of namespace sparkit::testing
