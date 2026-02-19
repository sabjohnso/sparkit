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
#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Symmetric_compressed_row_sparsity;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using size_type = sparkit::config::size_type;

  // -- sCSR construction --

  TEST_CASE("symmetric_compressed_row_sparsity - construction_from_lower_triangle",
            "[symmetric_compressed_row_sparsity]")
  {
    // 4x4 symmetric matrix with lower-triangle entries:
    // (0,0), (1,0), (1,1), (2,1), (3,3)
    Symmetric_compressed_row_sparsity scsr{Shape{4, 4},
      {Index{0, 0}, Index{1, 0}, Index{1, 1}, Index{2, 1}, Index{3, 3}}};

    CHECK(scsr.shape() == Shape(4, 4));
    CHECK(scsr.size() == 5);
  }

  TEST_CASE("symmetric_compressed_row_sparsity - construction_normalizes_upper_to_lower",
            "[symmetric_compressed_row_sparsity]")
  {
    // Provide upper-triangle indices — should be normalized to lower triangle
    // (0,1) -> (1,0), (1,2) -> (2,1)
    Symmetric_compressed_row_sparsity scsr{Shape{4, 4},
      {Index{0, 0}, Index{0, 1}, Index{1, 1}, Index{1, 2}, Index{3, 3}}};

    CHECK(scsr.size() == 5);

    // Verify row_ptr structure matches lower triangle
    auto rp = scsr.row_ptr();
    auto ci = scsr.col_ind();

    // Row 0: (0,0)
    CHECK(rp[1] - rp[0] == 1);
    CHECK(ci[rp[0]] == 0);

    // Row 1: (1,0), (1,1)
    CHECK(rp[2] - rp[1] == 2);
    CHECK(ci[rp[1]] == 0);
    CHECK(ci[rp[1] + 1] == 1);

    // Row 2: (2,1)
    CHECK(rp[3] - rp[2] == 1);
    CHECK(ci[rp[2]] == 1);

    // Row 3: (3,3)
    CHECK(rp[4] - rp[3] == 1);
    CHECK(ci[rp[3]] == 3);
  }

  TEST_CASE("symmetric_compressed_row_sparsity - construction_empty",
            "[symmetric_compressed_row_sparsity]")
  {
    Symmetric_compressed_row_sparsity scsr{Shape{3, 3}, {}};
    CHECK(scsr.shape() == Shape(3, 3));
    CHECK(scsr.size() == 0);
  }

  // -- Row pointer and col_ind structure --

  TEST_CASE("symmetric_compressed_row_sparsity - row_ptr_and_col_ind_structure",
            "[symmetric_compressed_row_sparsity]")
  {
    // 5x5 tridiagonal symmetric matrix (lower triangle only)
    // Row 0: (0,0)
    // Row 1: (1,0), (1,1)
    // Row 2: (2,1), (2,2)
    // Row 3: (3,2), (3,3)
    // Row 4: (4,3), (4,4)
    Symmetric_compressed_row_sparsity scsr{Shape{5, 5},
      {Index{0, 0}, Index{1, 0}, Index{1, 1},
       Index{2, 1}, Index{2, 2}, Index{3, 2},
       Index{3, 3}, Index{4, 3}, Index{4, 4}}};

    auto rp = scsr.row_ptr();
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 1);
    CHECK(rp[2] == 3);
    CHECK(rp[3] == 5);
    CHECK(rp[4] == 7);
    CHECK(rp[5] == 9);

    auto ci = scsr.col_ind();
    REQUIRE(std::ssize(ci) == 9);
    CHECK(ci[0] == 0);  // row 0: col 0
    CHECK(ci[1] == 0);  // row 1: col 0
    CHECK(ci[2] == 1);  // row 1: col 1
    CHECK(ci[3] == 1);  // row 2: col 1
    CHECK(ci[4] == 2);  // row 2: col 2
    CHECK(ci[5] == 2);  // row 3: col 2
    CHECK(ci[6] == 3);  // row 3: col 3
    CHECK(ci[7] == 3);  // row 4: col 3
    CHECK(ci[8] == 4);  // row 4: col 4
  }

  // -- Duplicate handling --

  TEST_CASE("symmetric_compressed_row_sparsity - duplicate_indices_collapsed",
            "[symmetric_compressed_row_sparsity]")
  {
    // (0,1) and (1,0) both normalize to (1,0) — should yield 1 entry
    Symmetric_compressed_row_sparsity scsr{Shape{3, 3},
      {Index{0, 1}, Index{1, 0}, Index{2, 2}}};

    CHECK(scsr.size() == 2);
  }

  // -- Diagonal only --

  TEST_CASE("symmetric_compressed_row_sparsity - diagonal_only",
            "[symmetric_compressed_row_sparsity]")
  {
    Symmetric_compressed_row_sparsity scsr{Shape{4, 4},
      {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    CHECK(scsr.size() == 4);

    auto rp = scsr.row_ptr();
    for (size_type r = 0; r < 4; ++r) {
      CHECK(rp[r + 1] - rp[r] == 1);
    }

    auto ci = scsr.col_ind();
    for (size_type i = 0; i < 4; ++i) {
      CHECK(ci[i] == i);
    }
  }

  // -- Copy/move --

  TEST_CASE("symmetric_compressed_row_sparsity - copy_construction",
            "[symmetric_compressed_row_sparsity]")
  {
    Symmetric_compressed_row_sparsity original{Shape{4, 4},
      {Index{0, 0}, Index{1, 0}, Index{1, 1}}};
    Symmetric_compressed_row_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());
    CHECK(copy.col_ind().data() != original.col_ind().data());
  }

  TEST_CASE("symmetric_compressed_row_sparsity - move_construction",
            "[symmetric_compressed_row_sparsity]")
  {
    Symmetric_compressed_row_sparsity original{Shape{4, 4},
      {Index{0, 0}, Index{1, 0}, Index{1, 1}}};
    auto original_size = original.size();

    Symmetric_compressed_row_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- CSR <-> sCSR conversions --

  TEST_CASE("conversions - scsr_to_csr_expands_both_triangles",
            "[conversions]")
  {
    // Lower triangle: (0,0), (1,0), (1,1), (2,1), (2,2)
    Symmetric_compressed_row_sparsity scsr{Shape{3, 3},
      {Index{0, 0}, Index{1, 0}, Index{1, 1}, Index{2, 1}, Index{2, 2}}};

    auto csr = sparkit::data::detail::to_compressed_row(scsr);

    CHECK(csr.shape() == Shape(3, 3));
    // Expanded: (0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2) = 7 entries
    CHECK(csr.size() == 7);
  }

  TEST_CASE("conversions - csr_to_scsr_filters_lower_triangle",
            "[conversions]")
  {
    // Full symmetric CSR: 7 entries
    Compressed_row_sparsity csr{Shape{3, 3},
      {Index{0, 0}, Index{0, 1},
       Index{1, 0}, Index{1, 1}, Index{1, 2},
       Index{2, 1}, Index{2, 2}}};

    auto scsr = sparkit::data::detail::to_symmetric_compressed_row(csr);

    CHECK(scsr.shape() == Shape(3, 3));
    CHECK(scsr.size() == 5);
  }

  TEST_CASE("conversions - scsr_csr_roundtrip",
            "[conversions]")
  {
    Symmetric_compressed_row_sparsity original{Shape{4, 4},
      {Index{0, 0}, Index{1, 0}, Index{1, 1},
       Index{2, 1}, Index{2, 2}, Index{3, 2}, Index{3, 3}}};

    auto csr = sparkit::data::detail::to_compressed_row(original);
    auto roundtrip = sparkit::data::detail::to_symmetric_compressed_row(csr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());

    auto orig_rp = original.row_ptr();
    auto rt_rp = roundtrip.row_ptr();
    REQUIRE(std::ssize(orig_rp) == std::ssize(rt_rp));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_rp); ++i) {
      CHECK(orig_rp[i] == rt_rp[i]);
    }

    auto orig_ci = original.col_ind();
    auto rt_ci = roundtrip.col_ind();
    REQUIRE(std::ssize(orig_ci) == std::ssize(rt_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      CHECK(orig_ci[i] == rt_ci[i]);
    }
  }

} // end of namespace sparkit::testing
