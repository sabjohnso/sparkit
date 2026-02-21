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
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Coordinate_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Coordinate_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- Index construction --

  TEST_CASE("index - zero_based_indices_are_valid", "[index]") {
    Index idx{0, 0};
    CHECK(idx.row() == 0);
    CHECK(idx.column() == 0);
  }

  TEST_CASE("index - row_zero_column_nonzero", "[index]") {
    Index idx{0, 5};
    CHECK(idx.row() == 0);
    CHECK(idx.column() == 5);
  }

  TEST_CASE("index - row_nonzero_column_zero", "[index]") {
    Index idx{3, 0};
    CHECK(idx.row() == 3);
    CHECK(idx.column() == 0);
  }

  TEST_CASE("index - row_one_column_one", "[index]") {
    Index idx{1, 1};
    CHECK(idx.row() == 1);
    CHECK(idx.column() == 1);
  }

  TEST_CASE("index - negative_row_throws", "[index]") {
    CHECK_THROWS_AS(Index(-1, 3), std::logic_error);
  }

  TEST_CASE("index - negative_column_throws", "[index]") {
    CHECK_THROWS_AS(Index(3, -1), std::logic_error);
  }

  TEST_CASE("index - both_negative_throws", "[index]") {
    CHECK_THROWS_AS(Index(-1, -1), std::logic_error);
  }

  // -- COO indices() accessor --

  TEST_CASE("coordinate_sparsity - indices_empty", "[coordinate_sparsity]") {
    Coordinate_sparsity coo{Shape{3, 3}, {}};
    auto idx = coo.indices();
    CHECK(idx.empty());
  }

  TEST_CASE("coordinate_sparsity - indices_returns_all",
            "[coordinate_sparsity]") {
    Coordinate_sparsity coo{Shape{5, 6},
                            {Index{2, 3}, Index{3, 4}, Index{4, 5}}};

    auto idx = coo.indices();
    REQUIRE(std::ssize(idx) == 3);

    // Order-independent check
    auto by_row_col = [](Index const& a, Index const& b) {
      return a.row() < b.row() ||
             (a.row() == b.row() && a.column() < b.column());
    };
    std::sort(begin(idx), end(idx), by_row_col);
    CHECK(idx[0] == Index(2, 3));
    CHECK(idx[1] == Index(3, 4));
    CHECK(idx[2] == Index(4, 5));
  }

  // -- CSR tests --

  TEST_CASE("compressed_row_sparsity - construction_from_initializer_list",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity csr{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    CHECK(csr.shape() == Shape(4, 5));
    CHECK(csr.size() == 2);
  }

  TEST_CASE("compressed_row_sparsity - construction_empty",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity csr{Shape{3, 3}, {}};
    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr.size() == 0);
  }

  // -- Red 2: CSR structure accessors --

  TEST_CASE("compressed_row_sparsity - row_ptr_and_col_ind_structure",
            "[compressed_row_sparsity]") {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Compressed_row_sparsity csr{Shape{5, 6},
                                {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();

    // row_ptr has shape.row()+1 = 6 entries
    REQUIRE(std::ssize(rp) == 6);

    // rows 0,1 are empty
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    // row 2 has 2 entries: row_ptr[3]=2
    CHECK(rp[3] == 2);
    // row 3 has 1 entry: row_ptr[4]=3
    CHECK(rp[4] == 3);
    // row 4 is empty: row_ptr[5]=3
    CHECK(rp[5] == 3);

    // col_ind has nnz=3 entries
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);
  }

  TEST_CASE("compressed_row_sparsity - col_ind_sorted_within_rows",
            "[compressed_row_sparsity]") {
    // Indices given out of order within a row
    Compressed_row_sparsity csr{Shape{4, 6},
                                {Index{2, 5}, Index{2, 2}, Index{2, 4}}};

    auto ci = csr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);
  }

  TEST_CASE("compressed_row_sparsity - duplicate_indices_are_collapsed",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity csr{Shape{4, 5},
                                {Index{2, 3}, Index{2, 3}, Index{3, 4}}};

    CHECK(csr.size() == 2);

    auto rp = csr.row_ptr();
    CHECK(rp[3] == 1); // row 2 has 1 unique entry
    CHECK(rp[4] == 2); // row 3 has 1 entry, cumulative = 2
  }

  TEST_CASE("compressed_row_sparsity - row_ptr_empty_matrix",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity csr{Shape{3, 3}, {}};

    auto rp = csr.row_ptr();
    REQUIRE(std::ssize(rp) == 4);
    for (auto v : rp) {
      CHECK(v == 0);
    }

    CHECK(std::ssize(csr.col_ind()) == 0);
  }

  TEST_CASE("compressed_row_sparsity - single_row_multiple_entries",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity csr{
        Shape{4, 8}, {Index{2, 7}, Index{2, 3}, Index{2, 2}, Index{2, 5}}};

    CHECK(csr.size() == 4);

    auto rp = csr.row_ptr();
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    CHECK(rp[3] == 4);
    CHECK(rp[4] == 4);

    auto ci = csr.col_ind();
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 3);
    CHECK(ci[2] == 5);
    CHECK(ci[3] == 7);
  }

  // -- Red 3: Iterator range constructor --

  TEST_CASE("compressed_row_sparsity - construction_from_iterator_range",
            "[compressed_row_sparsity]") {
    std::vector<Index> indices{Index{3, 2}, Index{2, 4}, Index{2, 2}};
    Compressed_row_sparsity csr{Shape{5, 6}, begin(indices), end(indices)};

    CHECK(csr.shape() == Shape(5, 6));
    CHECK(csr.size() == 3);

    auto ci = csr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    // row 2: columns 2, 4; row 3: column 2
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 2);
  }

  // -- Red 4: Copy/move semantics --

  TEST_CASE("compressed_row_sparsity - copy_construction",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    Compressed_row_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());

    auto orig_ci = original.col_ind();
    auto copy_ci = copy.col_ind();
    REQUIRE(std::ssize(copy_ci) == std::ssize(orig_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      CHECK(copy_ci[i] == orig_ci[i]);
    }

    // Verify independent storage (different addresses)
    CHECK(copy_ci.data() != orig_ci.data());
  }

  TEST_CASE("compressed_row_sparsity - move_construction",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_sparsity moved{std::move(original)};

    CHECK(moved.shape() == original_shape);
    CHECK(moved.size() == original_size);
  }

  TEST_CASE("compressed_row_sparsity - copy_assignment",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    Compressed_row_sparsity target{Shape{3, 3}, {}};

    target = original;

    CHECK(target.shape() == original.shape());
    CHECK(target.size() == original.size());

    auto orig_ci = original.col_ind();
    auto tgt_ci = target.col_ind();
    REQUIRE(std::ssize(tgt_ci) == std::ssize(orig_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      CHECK(tgt_ci[i] == orig_ci[i]);
    }

    CHECK(tgt_ci.data() != orig_ci.data());
  }

  TEST_CASE("compressed_row_sparsity - move_assignment",
            "[compressed_row_sparsity]") {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_sparsity target{Shape{3, 3}, {}};
    target = std::move(original);

    CHECK(target.shape() == original_shape);
    CHECK(target.size() == original_size);
  }

  // -- COO to CSR conversion --

  TEST_CASE("conversions - coo_to_csr_empty", "[conversions]") {
    Coordinate_sparsity coo{Shape{3, 3}, {}};
    auto csr = sparkit::data::detail::to_compressed_row(coo);

    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr.size() == 0);
    CHECK(std::ssize(csr.col_ind()) == 0);
  }

  TEST_CASE("conversions - coo_to_csr_basic", "[conversions]") {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Coordinate_sparsity coo{Shape{5, 6},
                            {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    auto csr = sparkit::data::detail::to_compressed_row(coo);

    CHECK(csr.size() == 3);

    auto rp = csr.row_ptr();
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    CHECK(rp[3] == 2);
    CHECK(rp[4] == 3);
    CHECK(rp[5] == 3);

    auto ci = csr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);
  }

  TEST_CASE("conversions - coo_to_csr_preserves_shape", "[conversions]") {
    Coordinate_sparsity coo{Shape{7, 9}, {Index{3, 4}}};
    auto csr = sparkit::data::detail::to_compressed_row(coo);
    CHECK(csr.shape() == Shape(7, 9));
  }

  TEST_CASE("conversions - coo_to_csr_deduplicates", "[conversions]") {
    Coordinate_sparsity coo{Shape{5, 6}, {}};
    coo.add(Index{2, 3});
    coo.add(Index{2, 3}); // duplicate
    coo.add(Index{3, 4});

    auto csr = sparkit::data::detail::to_compressed_row(coo);

    // COO's unordered_set already deduplicates, but CSR constructor
    // also deduplicates — either way, result has 2 unique entries.
    CHECK(csr.size() == 2);
  }

} // end of namespace sparkit::testing
