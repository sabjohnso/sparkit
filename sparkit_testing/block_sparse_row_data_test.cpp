//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Block_sparse_row_sparsity.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Block_sparse_row_sparsity;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using size_type = sparkit::config::size_type;

  // -- BSR construction --

  TEST_CASE("block_sparse_row_sparsity - construction_basic",
            "[block_sparse_row_sparsity]") {
    // 4x4 matrix, 2x2 blocks, entries in block (0,0) and block (1,1)
    Block_sparse_row_sparsity bsr{Shape{4, 4},
                                  2,
                                  2,
                                  {Index{0, 0}, Index{0, 1}, Index{1, 0},
                                   Index{1, 1}, Index{2, 2}, Index{2, 3},
                                   Index{3, 2}, Index{3, 3}}};

    CHECK(bsr.shape() == Shape(4, 4));
    CHECK(bsr.block_rows() == 2);
    CHECK(bsr.block_cols() == 2);
    CHECK(bsr.num_block_rows() == 2);
    CHECK(bsr.num_block_cols() == 2);
    CHECK(bsr.num_blocks() == 2);
    CHECK(bsr.size() == 8);
  }

  TEST_CASE("block_sparse_row_sparsity - construction_empty",
            "[block_sparse_row_sparsity]") {
    Block_sparse_row_sparsity bsr{Shape{4, 6}, 2, 3, {}};
    CHECK(bsr.shape() == Shape(4, 6));
    CHECK(bsr.num_blocks() == 0);
    CHECK(bsr.size() == 0);
  }

  // -- Block structure accessors --

  TEST_CASE("block_sparse_row_sparsity - row_ptr_and_col_ind_structure",
            "[block_sparse_row_sparsity]") {
    // 4x6 matrix, 2x2 blocks. Blocks at (0,0), (0,2), (1,1)
    // Scalar indices: block (0,0) -> rows 0-1, cols 0-1
    //                 block (0,2) -> rows 0-1, cols 4-5
    //                 block (1,1) -> rows 2-3, cols 2-3
    Block_sparse_row_sparsity bsr{Shape{4, 6},
                                  2,
                                  2,
                                  {Index{0, 0}, Index{0, 1}, Index{1, 0},
                                   Index{1, 1}, Index{0, 4}, Index{0, 5},
                                   Index{1, 4}, Index{1, 5}, Index{2, 2},
                                   Index{2, 3}, Index{3, 2}, Index{3, 3}}};

    CHECK(bsr.num_blocks() == 3);

    auto rp = bsr.row_ptr();
    REQUIRE(std::ssize(rp) == 3); // num_block_rows + 1
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 2); // block row 0 has 2 blocks
    CHECK(rp[2] == 3); // block row 1 has 1 block

    auto ci = bsr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 0); // block column 0
    CHECK(ci[1] == 2); // block column 2
    CHECK(ci[2] == 1); // block column 1
  }

  // -- Duplicate handling --

  TEST_CASE("block_sparse_row_sparsity - duplicate_indices_collapsed",
            "[block_sparse_row_sparsity]") {
    // Same block touched twice
    Block_sparse_row_sparsity bsr{
        Shape{4, 4}, 2, 2, {Index{0, 0}, Index{0, 1}, Index{0, 0}}};

    CHECK(bsr.num_blocks() == 1);
  }

  // -- Non-divisible dimensions --

  TEST_CASE("block_sparse_row_sparsity - non_divisible_dimensions",
            "[block_sparse_row_sparsity]") {
    // 5x5 matrix with 2x2 blocks: 3 block rows, 3 block cols
    // (last block row/col only partially filled)
    Block_sparse_row_sparsity bsr{
        Shape{5, 5}, 2, 2, {Index{0, 0}, Index{4, 4}}};

    CHECK(bsr.num_block_rows() == 3);
    CHECK(bsr.num_block_cols() == 3);
    CHECK(bsr.num_blocks() == 2);
  }

  // -- Copy/move --

  TEST_CASE("block_sparse_row_sparsity - copy_construction",
            "[block_sparse_row_sparsity]") {
    Block_sparse_row_sparsity original{
        Shape{4, 4}, 2, 2, {Index{0, 0}, Index{2, 2}}};
    Block_sparse_row_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.num_blocks() == original.num_blocks());
    CHECK(copy.col_ind().data() != original.col_ind().data());
  }

  TEST_CASE("block_sparse_row_sparsity - move_construction",
            "[block_sparse_row_sparsity]") {
    Block_sparse_row_sparsity original{
        Shape{4, 4}, 2, 2, {Index{0, 0}, Index{2, 2}}};
    auto original_blocks = original.num_blocks();

    Block_sparse_row_sparsity moved{std::move(original)};
    CHECK(moved.num_blocks() == original_blocks);
  }

  // -- CSR <-> BSR conversions --

  TEST_CASE("conversions - csr_to_bsr_basic", "[conversions]") {
    // 4x4 identity matrix -> 2x2 block diagonal
    Compressed_row_sparsity csr{
        Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto bsr = sparkit::data::detail::to_block_sparse_row(csr, 2, 2);

    CHECK(bsr.shape() == Shape(4, 4));
    CHECK(bsr.num_blocks() == 2);
    CHECK(bsr.block_rows() == 2);
    CHECK(bsr.block_cols() == 2);
  }

  TEST_CASE("conversions - bsr_to_csr_basic", "[conversions]") {
    Block_sparse_row_sparsity bsr{
        Shape{4, 4},
        2,
        2,
        {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto csr = sparkit::data::detail::to_compressed_row(bsr);

    CHECK(csr.shape() == Shape(4, 4));
    // BSR stores entire blocks, so all positions in occupied blocks
    // appear in the CSR. 2 blocks x 4 positions = 8 structural entries.
    CHECK(csr.size() == 8);
  }

  TEST_CASE("conversions - csr_bsr_roundtrip", "[conversions]") {
    // Dense 4x4 matrix (all entries present) -> single 4x4 block
    Compressed_row_sparsity original{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{1, 0},
         Index{1, 1}, Index{1, 2}, Index{1, 3}, Index{2, 0}, Index{2, 1},
         Index{2, 2}, Index{2, 3}, Index{3, 0}, Index{3, 1}, Index{3, 2},
         Index{3, 3}}};

    auto bsr = sparkit::data::detail::to_block_sparse_row(original, 2, 2);
    auto roundtrip = sparkit::data::detail::to_compressed_row(bsr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
  }

} // end of namespace sparkit::testing
