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
#include <sparkit/data/Modified_sparse_row_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Modified_sparse_row_sparsity;
  using sparkit::data::detail::Shape;

  // -- MSR construction --

  TEST_CASE("modified_sparse_row_sparsity - construction_from_initializer_list",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity msr{Shape{4, 5},
                                     {Index{2, 2}, Index{2, 3}, Index{3, 4}}};
    CHECK(msr.shape() == Shape(4, 5));
    CHECK(msr.size() == 3);
  }

  TEST_CASE("modified_sparse_row_sparsity - construction_empty",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity msr{Shape{3, 3}, {}};
    CHECK(msr.shape() == Shape(3, 3));
    CHECK(msr.size() == 0);
  }

  // -- Diagonal / off-diagonal separation --

  TEST_CASE("modified_sparse_row_sparsity - diagonal_detection",
            "[modified_sparse_row_sparsity]") {
    // 4x4 matrix with diag at (0,0), (2,2) and off-diag at (1,3), (2,0)
    Modified_sparse_row_sparsity msr{
        Shape{4, 4}, {Index{0, 0}, Index{1, 3}, Index{2, 0}, Index{2, 2}}};

    CHECK(msr.diagonal_length() == 4);
    CHECK(msr.has_diagonal(0) == true);
    CHECK(msr.has_diagonal(1) == false);
    CHECK(msr.has_diagonal(2) == true);
    CHECK(msr.has_diagonal(3) == false);
  }

  TEST_CASE("modified_sparse_row_sparsity - off_diagonal_structure",
            "[modified_sparse_row_sparsity]") {
    // 4x4 matrix with diag at (0,0), (2,2) and off-diag at (1,3), (2,0)
    Modified_sparse_row_sparsity msr{
        Shape{4, 4}, {Index{0, 0}, Index{1, 3}, Index{2, 0}, Index{2, 2}}};

    auto rp = msr.off_diagonal_row_ptr();
    auto ci = msr.off_diagonal_col_ind();

    // 4 rows + 1 = 5 entries in row_ptr
    REQUIRE(std::ssize(rp) == 5);
    CHECK(rp[0] == 0); // row 0: no off-diag
    CHECK(rp[1] == 0); // row 1 starts at 0
    CHECK(rp[2] == 1); // row 1 has 1 off-diag (col 3)
    CHECK(rp[3] == 2); // row 2 has 1 off-diag (col 0)
    CHECK(rp[4] == 2); // row 3: no off-diag

    REQUIRE(std::ssize(ci) == 2);
    CHECK(ci[0] == 3); // row 1, off-diag col 3
    CHECK(ci[1] == 0); // row 2, off-diag col 0
  }

  TEST_CASE("modified_sparse_row_sparsity - all_diagonal",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity msr{Shape{3, 3},
                                     {Index{0, 0}, Index{1, 1}, Index{2, 2}}};

    CHECK(msr.has_diagonal(0) == true);
    CHECK(msr.has_diagonal(1) == true);
    CHECK(msr.has_diagonal(2) == true);

    auto ci = msr.off_diagonal_col_ind();
    CHECK(ci.empty());
  }

  TEST_CASE("modified_sparse_row_sparsity - no_diagonal",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity msr{Shape{3, 3},
                                     {Index{0, 1}, Index{1, 0}, Index{2, 0}}};

    CHECK(msr.has_diagonal(0) == false);
    CHECK(msr.has_diagonal(1) == false);
    CHECK(msr.has_diagonal(2) == false);

    CHECK(std::ssize(msr.off_diagonal_col_ind()) == 3);
  }

  TEST_CASE("modified_sparse_row_sparsity - rectangular_matrix",
            "[modified_sparse_row_sparsity]") {
    // 3x5 matrix — diagonal length is min(3,5) = 3
    Modified_sparse_row_sparsity msr{Shape{3, 5},
                                     {Index{0, 0}, Index{1, 4}, Index{2, 2}}};

    CHECK(msr.diagonal_length() == 3);
    CHECK(msr.has_diagonal(0) == true);
    CHECK(msr.has_diagonal(1) == false);
    CHECK(msr.has_diagonal(2) == true);
  }

  // -- Duplicate handling --

  TEST_CASE("modified_sparse_row_sparsity - duplicates_collapsed",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity msr{Shape{3, 3},
                                     {Index{1, 1}, Index{1, 1}, Index{0, 2}}};

    CHECK(msr.size() == 2);
  }

  // -- Copy/move --

  TEST_CASE("modified_sparse_row_sparsity - copy_construction",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity original{
        Shape{4, 4}, {Index{0, 0}, Index{1, 3}, Index{2, 0}}};
    Modified_sparse_row_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());

    auto orig_ci = original.off_diagonal_col_ind();
    auto copy_ci = copy.off_diagonal_col_ind();
    REQUIRE(std::ssize(copy_ci) == std::ssize(orig_ci));
    CHECK(copy_ci.data() != orig_ci.data());
  }

  TEST_CASE("modified_sparse_row_sparsity - move_construction",
            "[modified_sparse_row_sparsity]") {
    Modified_sparse_row_sparsity original{Shape{4, 4},
                                          {Index{0, 0}, Index{1, 3}}};
    auto original_size = original.size();

    Modified_sparse_row_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- CSR <-> MSR conversions --

  TEST_CASE("conversions - csr_to_msr_basic", "[conversions]") {
    Compressed_row_sparsity csr{
        Shape{4, 4}, {Index{0, 0}, Index{1, 3}, Index{2, 0}, Index{2, 2}}};

    auto msr = sparkit::data::detail::to_modified_sparse_row(csr);

    CHECK(msr.shape() == Shape(4, 4));
    CHECK(msr.size() == 4);

    CHECK(msr.has_diagonal(0) == true);
    CHECK(msr.has_diagonal(2) == true);

    auto ci = msr.off_diagonal_col_ind();
    REQUIRE(std::ssize(ci) == 2);
    CHECK(ci[0] == 3);
    CHECK(ci[1] == 0);
  }

  TEST_CASE("conversions - msr_to_csr_basic", "[conversions]") {
    Modified_sparse_row_sparsity msr{
        Shape{4, 4}, {Index{0, 0}, Index{1, 3}, Index{2, 0}, Index{2, 2}}};

    auto csr = sparkit::data::detail::to_compressed_row(msr);

    CHECK(csr.shape() == Shape(4, 4));
    CHECK(csr.size() == 4);

    auto rp = csr.row_ptr();
    REQUIRE(std::ssize(rp) == 5);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 1);
    CHECK(rp[2] == 2);
    CHECK(rp[3] == 4);
    CHECK(rp[4] == 4);
  }

  TEST_CASE("conversions - csr_msr_roundtrip", "[conversions]") {
    Compressed_row_sparsity original{Shape{5, 5},
                                     {Index{0, 0}, Index{0, 3}, Index{1, 1},
                                      Index{2, 0}, Index{2, 2}, Index{3, 4},
                                      Index{4, 4}}};

    auto msr = sparkit::data::detail::to_modified_sparse_row(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(msr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());

    auto orig_rp = original.row_ptr();
    auto rt_rp = roundtrip.row_ptr();
    REQUIRE(std::ssize(rt_rp) == std::ssize(orig_rp));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_rp); ++i) {
      CHECK(rt_rp[i] == orig_rp[i]);
    }

    auto orig_ci = original.col_ind();
    auto rt_ci = roundtrip.col_ind();
    REQUIRE(std::ssize(rt_ci) == std::ssize(orig_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      CHECK(rt_ci[i] == orig_ci[i]);
    }
  }

} // end of namespace sparkit::testing
