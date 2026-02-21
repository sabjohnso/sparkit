//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Symmetric_block_sparse_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_block_sparse_row_matrix;
  using sparkit::data::detail::Symmetric_block_sparse_row_sparsity;

  // -- sBSR matrix core --

  TEST_CASE("symmetric_block_sparse_row_matrix - construction_from_entries",
            "[symmetric_block_sparse_row_matrix]") {
    // 4x4 symmetric: blocks (0,0), (1,0), (1,1)
    Symmetric_block_sparse_row_matrix<double> mat{Shape{4, 4},
                                                  2,
                                                  2,
                                                  {{Index{0, 0}, 1.0},
                                                   {Index{0, 1}, 2.0},
                                                   {Index{1, 0}, 2.0},
                                                   {Index{1, 1}, 3.0},
                                                   {Index{2, 0}, 4.0},
                                                   {Index{2, 1}, 5.0},
                                                   {Index{3, 0}, 6.0},
                                                   {Index{3, 1}, 7.0},
                                                   {Index{2, 2}, 8.0},
                                                   {Index{2, 3}, 9.0},
                                                   {Index{3, 2}, 9.0},
                                                   {Index{3, 3}, 10.0}}};

    CHECK(mat.shape() == Shape(4, 4));
    // 3 blocks * 4 positions = 12
    CHECK(mat.size() == 12);
  }

  TEST_CASE("symmetric_block_sparse_row_matrix - element_access_lower_triangle",
            "[symmetric_block_sparse_row_matrix]") {
    Symmetric_block_sparse_row_matrix<double> mat{Shape{4, 4},
                                                  2,
                                                  2,
                                                  {{Index{0, 0}, 1.0},
                                                   {Index{0, 1}, 2.0},
                                                   {Index{1, 0}, 2.0},
                                                   {Index{1, 1}, 3.0},
                                                   {Index{2, 0}, 4.0},
                                                   {Index{2, 1}, 5.0},
                                                   {Index{3, 0}, 6.0},
                                                   {Index{3, 1}, 7.0},
                                                   {Index{2, 2}, 8.0},
                                                   {Index{2, 3}, 9.0},
                                                   {Index{3, 2}, 9.0},
                                                   {Index{3, 3}, 10.0}}};

    // Block (0,0) — diagonal
    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(1, 1) == Catch::Approx(3.0));

    // Block (1,0) — off-diagonal, stored in lower triangle
    CHECK(mat(2, 0) == Catch::Approx(4.0));
    CHECK(mat(2, 1) == Catch::Approx(5.0));
    CHECK(mat(3, 0) == Catch::Approx(6.0));
    CHECK(mat(3, 1) == Catch::Approx(7.0));

    // Block (1,1) — diagonal
    CHECK(mat(2, 2) == Catch::Approx(8.0));
    CHECK(mat(3, 3) == Catch::Approx(10.0));
  }

  TEST_CASE("symmetric_block_sparse_row_matrix - element_access_upper_triangle",
            "[symmetric_block_sparse_row_matrix]") {
    Symmetric_block_sparse_row_matrix<double> mat{Shape{4, 4},
                                                  2,
                                                  2,
                                                  {{Index{0, 0}, 1.0},
                                                   {Index{0, 1}, 2.0},
                                                   {Index{1, 0}, 2.0},
                                                   {Index{1, 1}, 3.0},
                                                   {Index{2, 0}, 4.0},
                                                   {Index{2, 1}, 5.0},
                                                   {Index{3, 0}, 6.0},
                                                   {Index{3, 1}, 7.0},
                                                   {Index{2, 2}, 8.0},
                                                   {Index{2, 3}, 9.0},
                                                   {Index{3, 2}, 9.0},
                                                   {Index{3, 3}, 10.0}}};

    // Block (0,1) — mirror of block (1,0), read with transposed local indices
    // (0,1) stores block (1,0) which has:
    //   local(0,0)=4, local(0,1)=5
    //   local(1,0)=6, local(1,1)=7
    // Accessing (0,2) -> block(0,1) -> read from block(1,0) with swapped local
    // indices scalar(0,2): block_row=0, block_col=1 -> stored block(1,0),
    // local_row=0, local_col=0
    //   but transposed: local_row=0, local_col=0 -> stored(0,0) in block(1,0) =
    //   4
    CHECK(mat(0, 2) == Catch::Approx(4.0));
    CHECK(mat(0, 3) == Catch::Approx(6.0));
    CHECK(mat(1, 2) == Catch::Approx(5.0));
    CHECK(mat(1, 3) == Catch::Approx(7.0));
  }

  // -- sBSR matrix <-> CSR matrix conversions --

  TEST_CASE("conversions - sbsr_matrix_to_csr_matrix_basic", "[conversions]") {
    Symmetric_block_sparse_row_matrix<double> sbsr{Shape{4, 4},
                                                   2,
                                                   2,
                                                   {{Index{0, 0}, 1.0},
                                                    {Index{0, 1}, 2.0},
                                                    {Index{1, 0}, 2.0},
                                                    {Index{1, 1}, 3.0},
                                                    {Index{2, 0}, 4.0},
                                                    {Index{2, 1}, 5.0},
                                                    {Index{3, 0}, 6.0},
                                                    {Index{3, 1}, 7.0},
                                                    {Index{2, 2}, 8.0},
                                                    {Index{2, 3}, 9.0},
                                                    {Index{3, 2}, 9.0},
                                                    {Index{3, 3}, 10.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(sbsr);

    CHECK(csr.shape() == Shape(4, 4));
    CHECK(csr(0, 0) == Catch::Approx(1.0));
    CHECK(csr(2, 0) == Catch::Approx(4.0));
    CHECK(csr(0, 2) == Catch::Approx(4.0)); // mirror
    CHECK(csr(3, 3) == Catch::Approx(10.0));
  }

  TEST_CASE("conversions - csr_matrix_to_sbsr_matrix_basic", "[conversions]") {
    // Full symmetric 4x4 matrix
    Compressed_row_matrix<double> csr{Shape{4, 4},
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{0, 2}, 4.0},
                                       {Index{0, 3}, 6.0},
                                       {Index{1, 0}, 2.0},
                                       {Index{1, 1}, 3.0},
                                       {Index{1, 2}, 5.0},
                                       {Index{1, 3}, 7.0},
                                       {Index{2, 0}, 4.0},
                                       {Index{2, 1}, 5.0},
                                       {Index{2, 2}, 8.0},
                                       {Index{2, 3}, 9.0},
                                       {Index{3, 0}, 6.0},
                                       {Index{3, 1}, 7.0},
                                       {Index{3, 2}, 9.0},
                                       {Index{3, 3}, 10.0}}};

    auto sbsr = sparkit::data::detail::to_symmetric_block_sparse_row(csr, 2, 2);

    CHECK(sbsr.shape() == Shape(4, 4));
    CHECK(sbsr(0, 0) == Catch::Approx(1.0));
    CHECK(sbsr(2, 0) == Catch::Approx(4.0));
    CHECK(sbsr(0, 2) == Catch::Approx(4.0));
    CHECK(sbsr(3, 3) == Catch::Approx(10.0));
  }

  TEST_CASE("conversions - sbsr_matrix_csr_matrix_roundtrip", "[conversions]") {
    Symmetric_block_sparse_row_matrix<double> original{Shape{4, 4},
                                                       2,
                                                       2,
                                                       {{Index{0, 0}, 1.0},
                                                        {Index{0, 1}, 2.0},
                                                        {Index{1, 0}, 2.0},
                                                        {Index{1, 1}, 3.0},
                                                        {Index{2, 0}, 4.0},
                                                        {Index{2, 1}, 5.0},
                                                        {Index{3, 0}, 6.0},
                                                        {Index{3, 1}, 7.0},
                                                        {Index{2, 2}, 8.0},
                                                        {Index{2, 3}, 9.0},
                                                        {Index{3, 2}, 9.0},
                                                        {Index{3, 3}, 10.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(original);
    auto roundtrip =
        sparkit::data::detail::to_symmetric_block_sparse_row(csr, 2, 2);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip(0, 0) == Catch::Approx(1.0));
    CHECK(roundtrip(2, 0) == Catch::Approx(4.0));
    CHECK(roundtrip(0, 2) == Catch::Approx(4.0));
    CHECK(roundtrip(1, 3) == Catch::Approx(7.0));
    CHECK(roundtrip(3, 3) == Catch::Approx(10.0));
  }

} // end of namespace sparkit::testing
