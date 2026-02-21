//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_column_sparsity.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_column_sparsity;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- CSC construction --

  TEST_CASE("compressed_column_sparsity - construction_from_initializer_list",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity csc{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    CHECK(csc.shape() == Shape(4, 5));
    CHECK(csc.size() == 2);
  }

  TEST_CASE("compressed_column_sparsity - construction_empty",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity csc{Shape{3, 3}, {}};
    CHECK(csc.shape() == Shape(3, 3));
    CHECK(csc.size() == 0);
  }

  // -- CSC structure accessors --

  TEST_CASE("compressed_column_sparsity - col_ptr_and_row_ind_structure",
            "[compressed_column_sparsity]") {
    // 6x5 matrix with entries at (2,2), (4,2), (5,3)
    Compressed_column_sparsity csc{Shape{6, 5},
                                   {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    auto cp = csc.col_ptr();
    auto ri = csc.row_ind();

    // col_ptr has shape.column()+1 = 6 entries
    REQUIRE(std::ssize(cp) == 6);

    // columns 0,1 are empty
    CHECK(cp[0] == 0);
    CHECK(cp[1] == 0);
    CHECK(cp[2] == 0);
    // column 2 has 2 entries: col_ptr[3]=2
    CHECK(cp[3] == 2);
    // column 3 has 1 entry: col_ptr[4]=3
    CHECK(cp[4] == 3);
    // column 4 is empty: col_ptr[5]=3
    CHECK(cp[5] == 3);

    // row_ind has nnz=3 entries
    REQUIRE(std::ssize(ri) == 3);
    CHECK(ri[0] == 2);
    CHECK(ri[1] == 4);
    CHECK(ri[2] == 5);
  }

  TEST_CASE("compressed_column_sparsity - row_ind_sorted_within_columns",
            "[compressed_column_sparsity]") {
    // Indices given out of order within a column
    Compressed_column_sparsity csc{Shape{6, 4},
                                   {Index{5, 2}, Index{2, 2}, Index{4, 2}}};

    auto ri = csc.row_ind();
    REQUIRE(std::ssize(ri) == 3);
    CHECK(ri[0] == 2);
    CHECK(ri[1] == 4);
    CHECK(ri[2] == 5);
  }

  TEST_CASE("compressed_column_sparsity - duplicate_indices_are_collapsed",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity csc{Shape{5, 4},
                                   {Index{2, 3}, Index{2, 3}, Index{3, 3}}};

    CHECK(csc.size() == 2);

    auto cp = csc.col_ptr();
    CHECK(cp[3] == 0); // columns 0-2 empty
    CHECK(cp[4] == 2); // column 3 has 2 unique entries
  }

  TEST_CASE("compressed_column_sparsity - col_ptr_empty_matrix",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity csc{Shape{3, 3}, {}};

    auto cp = csc.col_ptr();
    REQUIRE(std::ssize(cp) == 4);
    for (auto v : cp) {
      CHECK(v == 0);
    }

    CHECK(std::ssize(csc.row_ind()) == 0);
  }

  TEST_CASE("compressed_column_sparsity - single_column_multiple_entries",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity csc{
        Shape{8, 4}, {Index{7, 2}, Index{3, 2}, Index{2, 2}, Index{5, 2}}};

    CHECK(csc.size() == 4);

    auto cp = csc.col_ptr();
    CHECK(cp[0] == 0);
    CHECK(cp[1] == 0);
    CHECK(cp[2] == 0);
    CHECK(cp[3] == 4);
    CHECK(cp[4] == 4);

    auto ri = csc.row_ind();
    CHECK(ri[0] == 2);
    CHECK(ri[1] == 3);
    CHECK(ri[2] == 5);
    CHECK(ri[3] == 7);
  }

  // -- Iterator range constructor --

  TEST_CASE("compressed_column_sparsity - construction_from_iterator_range",
            "[compressed_column_sparsity]") {
    std::vector<Index> indices{Index{3, 2}, Index{2, 4}, Index{2, 2}};
    Compressed_column_sparsity csc{Shape{5, 6}, begin(indices), end(indices)};

    CHECK(csc.shape() == Shape(5, 6));
    CHECK(csc.size() == 3);

    auto ri = csc.row_ind();
    REQUIRE(std::ssize(ri) == 3);
    // column 2: rows 2, 3; column 4: row 2
    CHECK(ri[0] == 2);
    CHECK(ri[1] == 3);
    CHECK(ri[2] == 2);
  }

  // -- Copy/move semantics --

  TEST_CASE("compressed_column_sparsity - copy_construction",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity original{Shape{4, 5},
                                        {Index{2, 3}, Index{3, 4}}};
    Compressed_column_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());

    auto orig_ri = original.row_ind();
    auto copy_ri = copy.row_ind();
    REQUIRE(std::ssize(copy_ri) == std::ssize(orig_ri));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ri); ++i) {
      CHECK(copy_ri[i] == orig_ri[i]);
    }

    // Verify independent storage (different addresses)
    CHECK(copy_ri.data() != orig_ri.data());
  }

  TEST_CASE("compressed_column_sparsity - move_construction",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity original{Shape{4, 5},
                                        {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_column_sparsity moved{std::move(original)};

    CHECK(moved.shape() == original_shape);
    CHECK(moved.size() == original_size);
  }

  TEST_CASE("compressed_column_sparsity - copy_assignment",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity original{Shape{4, 5},
                                        {Index{2, 3}, Index{3, 4}}};
    Compressed_column_sparsity target{Shape{3, 3}, {}};

    target = original;

    CHECK(target.shape() == original.shape());
    CHECK(target.size() == original.size());

    auto orig_ri = original.row_ind();
    auto tgt_ri = target.row_ind();
    REQUIRE(std::ssize(tgt_ri) == std::ssize(orig_ri));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ri); ++i) {
      CHECK(tgt_ri[i] == orig_ri[i]);
    }

    CHECK(tgt_ri.data() != orig_ri.data());
  }

  TEST_CASE("compressed_column_sparsity - move_assignment",
            "[compressed_column_sparsity]") {
    Compressed_column_sparsity original{Shape{4, 5},
                                        {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_column_sparsity target{Shape{3, 3}, {}};
    target = std::move(original);

    CHECK(target.shape() == original_shape);
    CHECK(target.size() == original_size);
  }

  // -- CSR to CSC conversion --

  TEST_CASE("conversions - csr_to_csc_basic", "[conversions]") {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Compressed_row_sparsity csr{Shape{5, 6},
                                {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    auto csc = sparkit::data::detail::to_compressed_column(csr);

    CHECK(csc.shape() == Shape(5, 6));
    CHECK(csc.size() == 3);

    auto cp = csc.col_ptr();
    REQUIRE(std::ssize(cp) == 7);
    CHECK(cp[0] == 0);
    CHECK(cp[1] == 0);
    CHECK(cp[2] == 0);
    CHECK(cp[3] == 1); // column 2 has 1 entry (row 2)
    CHECK(cp[4] == 1); // column 3 has 0 entries
    CHECK(cp[5] == 2); // column 4 has 1 entry (row 2)
    CHECK(cp[6] == 3); // column 5 has 1 entry (row 3)

    auto ri = csc.row_ind();
    REQUIRE(std::ssize(ri) == 3);
    CHECK(ri[0] == 2); // col 2, row 2
    CHECK(ri[1] == 2); // col 4, row 2
    CHECK(ri[2] == 3); // col 5, row 3
  }

  TEST_CASE("conversions - csr_to_csc_empty", "[conversions]") {
    Compressed_row_sparsity csr{Shape{3, 3}, {}};
    auto csc = sparkit::data::detail::to_compressed_column(csr);

    CHECK(csc.shape() == Shape(3, 3));
    CHECK(csc.size() == 0);
    CHECK(std::ssize(csc.row_ind()) == 0);
  }

  TEST_CASE("conversions - csc_to_csr_basic", "[conversions]") {
    // Build CSC directly, then convert back to CSR
    Compressed_column_sparsity csc{Shape{5, 6},
                                   {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    auto csr = sparkit::data::detail::to_compressed_row(csc);

    CHECK(csr.shape() == Shape(5, 6));
    CHECK(csr.size() == 3);

    auto rp = csr.row_ptr();
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    CHECK(rp[3] == 2); // row 2 has 2 entries
    CHECK(rp[4] == 3); // row 3 has 1 entry
    CHECK(rp[5] == 3);

    auto ci = csr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);
  }

  // -- CSR <-> CSC roundtrip --

  TEST_CASE("conversions - csr_csc_roundtrip", "[conversions]") {
    Compressed_row_sparsity original{Shape{5, 6},
                                     {Index{0, 1}, Index{1, 0}, Index{1, 3},
                                      Index{2, 2}, Index{2, 4}, Index{3, 5},
                                      Index{4, 0}}};

    auto csc = sparkit::data::detail::to_compressed_column(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(csc);

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
