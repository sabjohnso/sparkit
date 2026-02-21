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
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Symmetric_block_sparse_row_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_block_sparse_row_sparsity;
  using size_type = sparkit::config::size_type;

  // -- sBSR construction --

  TEST_CASE("symmetric_block_sparse_row_sparsity - construction_basic",
            "[symmetric_block_sparse_row_sparsity]") {
    // 4x4 symmetric matrix, 2x2 blocks
    // Diagonal blocks (0,0) and (1,1) + off-diagonal block (1,0)
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{4, 4},
        2,
        2,
        {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}, Index{2, 0},
         Index{2, 1}, Index{3, 0}, Index{3, 1}, Index{2, 2}, Index{2, 3},
         Index{3, 2}, Index{3, 3}}};

    CHECK(sbsr.shape() == Shape(4, 4));
    CHECK(sbsr.block_rows() == 2);
    CHECK(sbsr.block_cols() == 2);
    CHECK(sbsr.num_block_rows() == 2);
    CHECK(sbsr.num_block_cols() == 2);
    // Blocks: (0,0), (1,0), (1,1) = 3 lower-triangle blocks
    CHECK(sbsr.num_blocks() == 3);
  }

  TEST_CASE("symmetric_block_sparse_row_sparsity - "
            "construction_normalizes_upper_blocks",
            "[symmetric_block_sparse_row_sparsity]") {
    // Provide entries in upper-triangle block (0,1) -> normalized to block
    // (1,0)
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{4, 4},
        2,
        2,
        {Index{0, 0}, Index{1, 1},   // block (0,0)
         Index{0, 2}, Index{0, 3},   // block (0,1) -> (1,0)
         Index{2, 2}, Index{3, 3}}}; // block (1,1)

    // Should have 3 lower-triangle blocks: (0,0), (1,0), (1,1)
    CHECK(sbsr.num_blocks() == 3);
  }

  TEST_CASE("symmetric_block_sparse_row_sparsity - construction_empty",
            "[symmetric_block_sparse_row_sparsity]") {
    Symmetric_block_sparse_row_sparsity sbsr{Shape{4, 4}, 2, 2, {}};
    CHECK(sbsr.shape() == Shape(4, 4));
    CHECK(sbsr.num_blocks() == 0);
    CHECK(sbsr.size() == 0);
  }

  // -- Block structure --

  TEST_CASE("symmetric_block_sparse_row_sparsity - lower_triangle_blocks_only",
            "[symmetric_block_sparse_row_sparsity]") {
    // 6x6, 2x2 blocks -> 3x3 block grid
    // Entries touching blocks (0,0), (1,0), (1,1), (2,0), (2,2)
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{6, 6},
        2,
        2,
        {Index{0, 0}, Index{1, 1},   // block (0,0)
         Index{2, 0}, Index{3, 1},   // block (1,0)
         Index{2, 2}, Index{3, 3},   // block (1,1)
         Index{4, 0}, Index{5, 1},   // block (2,0)
         Index{4, 4}, Index{5, 5}}}; // block (2,2)

    CHECK(sbsr.num_blocks() == 5);

    auto rp = sbsr.row_ptr();
    REQUIRE(std::ssize(rp) == 4); // 3 block rows + 1
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 1); // block row 0: 1 block (0,0)
    CHECK(rp[2] == 3); // block row 1: 2 blocks (1,0), (1,1)
    CHECK(rp[3] == 5); // block row 2: 2 blocks (2,0), (2,2)

    auto ci = sbsr.col_ind();
    REQUIRE(std::ssize(ci) == 5);
    CHECK(ci[0] == 0); // block (0,0)
    CHECK(ci[1] == 0); // block (1,0)
    CHECK(ci[2] == 1); // block (1,1)
    CHECK(ci[3] == 0); // block (2,0)
    CHECK(ci[4] == 2); // block (2,2)
  }

  // -- Duplicate handling --

  TEST_CASE("symmetric_block_sparse_row_sparsity - duplicate_blocks_collapsed",
            "[symmetric_block_sparse_row_sparsity]") {
    // Multiple entries in same block
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{4, 4},
        2,
        2,
        {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}}};

    CHECK(sbsr.num_blocks() == 1);
  }

  // -- Non-divisible dimensions --

  TEST_CASE("symmetric_block_sparse_row_sparsity - non_divisible_dimensions",
            "[symmetric_block_sparse_row_sparsity]") {
    // 5x5 with 2x2 blocks: 3 block rows, 3 block cols
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{5, 5}, 2, 2, {Index{0, 0}, Index{4, 4}}};

    CHECK(sbsr.num_block_rows() == 3);
    CHECK(sbsr.num_block_cols() == 3);
    CHECK(sbsr.num_blocks() == 2);
  }

  // -- Diagonal blocks only --

  TEST_CASE("symmetric_block_sparse_row_sparsity - diagonal_blocks_only",
            "[symmetric_block_sparse_row_sparsity]") {
    Symmetric_block_sparse_row_sparsity sbsr{
        Shape{4, 4},
        2,
        2,
        {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    CHECK(sbsr.num_blocks() == 2);
    auto ci = sbsr.col_ind();
    CHECK(ci[0] == 0);
    CHECK(ci[1] == 1);
  }

  // -- Copy/move --

  TEST_CASE("symmetric_block_sparse_row_sparsity - copy_construction",
            "[symmetric_block_sparse_row_sparsity]") {
    Symmetric_block_sparse_row_sparsity original{
        Shape{4, 4}, 2, 2, {Index{0, 0}, Index{2, 0}, Index{2, 2}}};
    Symmetric_block_sparse_row_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.num_blocks() == original.num_blocks());
    CHECK(copy.col_ind().data() != original.col_ind().data());
  }

  TEST_CASE("symmetric_block_sparse_row_sparsity - move_construction",
            "[symmetric_block_sparse_row_sparsity]") {
    Symmetric_block_sparse_row_sparsity original{
        Shape{4, 4}, 2, 2, {Index{0, 0}, Index{2, 0}, Index{2, 2}}};
    auto original_blocks = original.num_blocks();

    Symmetric_block_sparse_row_sparsity moved{std::move(original)};
    CHECK(moved.num_blocks() == original_blocks);
  }

  // -- CSR <-> sBSR conversions --

  TEST_CASE("conversions - sbsr_to_csr_expands", "[conversions]") {
    // 4x4 with block (0,0), (1,0), (1,1) — all lower-triangle blocks
    Symmetric_block_sparse_row_sparsity sbsr{Shape{4, 4},
                                             2,
                                             2,
                                             {Index{0, 0}, Index{1, 1},
                                              Index{2, 0}, Index{3, 1},
                                              Index{2, 2}, Index{3, 3}}};

    auto csr = sparkit::data::detail::to_compressed_row(sbsr);

    CHECK(csr.shape() == Shape(4, 4));
    // 3 blocks of 2x2 = 12 lower-triangle positions
    // off-diagonal block (1,0) also generates mirror block (0,1) = 4 more
    // Total: 12 + 4 = 16 scalar entries
    CHECK(csr.size() == 16);
  }

  TEST_CASE("conversions - csr_to_sbsr_filters_lower_blocks", "[conversions]") {
    // Full symmetric CSR (4x4 dense)
    Compressed_row_sparsity csr{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{1, 0},
         Index{1, 1}, Index{1, 2}, Index{1, 3}, Index{2, 0}, Index{2, 1},
         Index{2, 2}, Index{2, 3}, Index{3, 0}, Index{3, 1}, Index{3, 2},
         Index{3, 3}}};

    auto sbsr = sparkit::data::detail::to_symmetric_block_sparse_row(csr, 2, 2);

    CHECK(sbsr.shape() == Shape(4, 4));
    // 2x2 block grid, lower triangle: (0,0), (1,0), (1,1) = 3 blocks
    CHECK(sbsr.num_blocks() == 3);
  }

  TEST_CASE("conversions - csr_sbsr_roundtrip", "[conversions]") {
    // 4x4 symmetric with blocks (0,0), (1,0), (1,1)
    Compressed_row_sparsity original{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{1, 0},
         Index{1, 1}, Index{1, 2}, Index{1, 3}, Index{2, 0}, Index{2, 1},
         Index{2, 2}, Index{2, 3}, Index{3, 0}, Index{3, 1}, Index{3, 2},
         Index{3, 3}}};

    auto sbsr =
        sparkit::data::detail::to_symmetric_block_sparse_row(original, 2, 2);
    auto roundtrip = sparkit::data::detail::to_compressed_row(sbsr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
  }

} // end of namespace sparkit::testing
