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
#include <sparkit/data/Jagged_diagonal_sparsity.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Jagged_diagonal_sparsity;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;

  // -- JAD construction --

  TEST_CASE("jagged_diagonal_sparsity - construction_from_initializer_list", "[jagged_diagonal_sparsity]")
  {
    // Row 0: cols [1, 3]       (2 entries)
    // Row 1: cols [0, 2, 4]    (3 entries)
    // Row 2: cols [1]           (1 entry)
    // Row 3: cols [0, 3]        (2 entries)
    Jagged_diagonal_sparsity jad{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    CHECK(jad.shape() == Shape(4, 5));
    CHECK(jad.size() == 8);
  }

  TEST_CASE("jagged_diagonal_sparsity - construction_empty", "[jagged_diagonal_sparsity]")
  {
    Jagged_diagonal_sparsity jad{Shape{3, 3}, {}};
    CHECK(jad.shape() == Shape(3, 3));
    CHECK(jad.size() == 0);
  }

  // -- Permutation order --

  TEST_CASE("jagged_diagonal_sparsity - perm_sorted_by_decreasing_nnz", "[jagged_diagonal_sparsity]")
  {
    // Row 0: 2 entries, Row 1: 3 entries, Row 2: 1 entry, Row 3: 2 entries
    Jagged_diagonal_sparsity jad{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    auto perm = jad.perm();
    REQUIRE(std::ssize(perm) == 4);

    // Row 1 (3 entries) should be first
    CHECK(perm[0] == 1);

    // Next rows have 2 entries each (0 and 3), then row 2 with 1 entry last
    // Among ties, original order is preserved (stable sort)
    CHECK(perm[1] == 0);
    CHECK(perm[2] == 3);
    CHECK(perm[3] == 2);
  }

  // -- Jagged diagonal pointers --

  TEST_CASE("jagged_diagonal_sparsity - jdiag_pointers", "[jagged_diagonal_sparsity]")
  {
    // Row 0: 2, Row 1: 3, Row 2: 1, Row 3: 2
    // Sorted nnz: [3, 2, 2, 1] → jdiag has 4 entries (max_nnz+1)
    // JD 0: 4 rows → width 4
    // JD 1: 3 rows → width 3
    // JD 2: 1 row  → width 1
    // jdiag = [0, 4, 7, 8]
    Jagged_diagonal_sparsity jad{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    auto jd = jad.jdiag();
    REQUIRE(std::ssize(jd) == 4);  // max_nnz_per_row + 1
    CHECK(jd[0] == 0);
    CHECK(jd[1] == 4);
    CHECK(jd[2] == 7);
    CHECK(jd[3] == 8);
  }

  // -- Column indices in JD order --

  TEST_CASE("jagged_diagonal_sparsity - col_ind_in_jagged_diagonal_order", "[jagged_diagonal_sparsity]")
  {
    // Permuted order: row 1, row 0, row 3, row 2
    // Row 1 sorted cols: [0, 2, 4]
    // Row 0 sorted cols: [1, 3]
    // Row 3 sorted cols: [0, 3]
    // Row 2 sorted cols: [1]
    //
    // JD 0 (1st entry from each): [0, 1, 0, 1]
    // JD 1 (2nd entry from top 3): [2, 3, 3]
    // JD 2 (3rd entry from top 1): [4]
    // col_ind = [0, 1, 0, 1, 2, 3, 3, 4]
    Jagged_diagonal_sparsity jad{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    auto ci = jad.col_ind();
    REQUIRE(std::ssize(ci) == 8);

    // JD 0
    CHECK(ci[0] == 0);  // row 1's 1st
    CHECK(ci[1] == 1);  // row 0's 1st
    CHECK(ci[2] == 0);  // row 3's 1st
    CHECK(ci[3] == 1);  // row 2's 1st

    // JD 1
    CHECK(ci[4] == 2);  // row 1's 2nd
    CHECK(ci[5] == 3);  // row 0's 2nd
    CHECK(ci[6] == 3);  // row 3's 2nd

    // JD 2
    CHECK(ci[7] == 4);  // row 1's 3rd
  }

  // -- Duplicate handling --

  TEST_CASE("jagged_diagonal_sparsity - duplicates_collapsed", "[jagged_diagonal_sparsity]")
  {
    Jagged_diagonal_sparsity jad{Shape{3, 3},
      {Index{0, 1}, Index{0, 1}, Index{1, 2}}};
    CHECK(jad.size() == 2);
  }

  // -- Copy/move --

  TEST_CASE("jagged_diagonal_sparsity - copy_construction", "[jagged_diagonal_sparsity]")
  {
    Jagged_diagonal_sparsity original{Shape{4, 5},
      {Index{0, 1}, Index{1, 2}, Index{1, 3}}};
    Jagged_diagonal_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());
    CHECK(copy.perm().data() != original.perm().data());
  }

  TEST_CASE("jagged_diagonal_sparsity - move_construction", "[jagged_diagonal_sparsity]")
  {
    Jagged_diagonal_sparsity original{Shape{4, 5},
      {Index{0, 1}, Index{1, 2}}};
    auto original_size = original.size();

    Jagged_diagonal_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- CSR <-> JAD conversions --

  TEST_CASE("conversions - csr_to_jad_basic", "[conversions]")
  {
    Compressed_row_sparsity csr{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    auto jad = sparkit::data::detail::to_jagged_diagonal(csr);

    CHECK(jad.shape() == Shape(4, 5));
    CHECK(jad.size() == 8);
  }

  TEST_CASE("conversions - jad_to_csr_basic", "[conversions]")
  {
    Jagged_diagonal_sparsity jad{Shape{4, 5},
      {Index{0, 1}, Index{0, 3}, Index{1, 0}, Index{1, 2}, Index{1, 4},
       Index{2, 1}, Index{3, 0}, Index{3, 3}}};

    auto csr = sparkit::data::detail::to_compressed_row(jad);

    CHECK(csr.shape() == Shape(4, 5));
    CHECK(csr.size() == 8);
  }

  TEST_CASE("conversions - csr_jad_roundtrip", "[conversions]")
  {
    Compressed_row_sparsity original{Shape{5, 6},
      {Index{0, 1}, Index{0, 3}, Index{1, 1}, Index{2, 0},
       Index{2, 2}, Index{2, 4}, Index{3, 5}, Index{4, 0}}};

    auto jad = sparkit::data::detail::to_jagged_diagonal(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(jad);

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
