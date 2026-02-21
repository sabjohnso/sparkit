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
#include <sparkit/data/Ellpack_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Ellpack_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- ELL construction --

  TEST_CASE("ellpack_sparsity - construction_from_initializer_list",
            "[ellpack_sparsity]") {
    Ellpack_sparsity ell{Shape{4, 5},
                         {Index{0, 1}, Index{1, 2}, Index{1, 3}, Index{3, 4}}};
    CHECK(ell.shape() == Shape(4, 5));
    CHECK(ell.size() == 4);
  }

  TEST_CASE("ellpack_sparsity - construction_empty", "[ellpack_sparsity]") {
    Ellpack_sparsity ell{Shape{3, 3}, {}};
    CHECK(ell.shape() == Shape(3, 3));
    CHECK(ell.size() == 0);
    CHECK(ell.max_nnz_per_row() == 0);
  }

  // -- Max nnz per row --

  TEST_CASE("ellpack_sparsity - max_nnz_per_row", "[ellpack_sparsity]") {
    // Row 0: 1 entry, Row 1: 2 entries, Row 2: 0, Row 3: 1
    Ellpack_sparsity ell{Shape{4, 5},
                         {Index{0, 1}, Index{1, 2}, Index{1, 3}, Index{3, 4}}};

    CHECK(ell.max_nnz_per_row() == 2);
  }

  // -- Padded col_ind structure --

  TEST_CASE("ellpack_sparsity - col_ind_padded_with_sentinel",
            "[ellpack_sparsity]") {
    // Row 0: col 1, Row 1: cols 2, 3, Row 2: empty, Row 3: col 4
    Ellpack_sparsity ell{Shape{4, 5},
                         {Index{0, 1}, Index{1, 2}, Index{1, 3}, Index{3, 4}}};

    auto ci = ell.col_ind();
    auto max_nnz = ell.max_nnz_per_row();

    // nrow * max_nnz = 4 * 2 = 8 entries
    REQUIRE(std::ssize(ci) == 8);

    // Row 0: [1, -1]
    CHECK(ci[0 * max_nnz + 0] == 1);
    CHECK(ci[0 * max_nnz + 1] == -1);

    // Row 1: [2, 3]
    CHECK(ci[1 * max_nnz + 0] == 2);
    CHECK(ci[1 * max_nnz + 1] == 3);

    // Row 2: [-1, -1]
    CHECK(ci[2 * max_nnz + 0] == -1);
    CHECK(ci[2 * max_nnz + 1] == -1);

    // Row 3: [4, -1]
    CHECK(ci[3 * max_nnz + 0] == 4);
    CHECK(ci[3 * max_nnz + 1] == -1);
  }

  // -- Duplicate handling --

  TEST_CASE("ellpack_sparsity - duplicates_collapsed", "[ellpack_sparsity]") {
    Ellpack_sparsity ell{Shape{3, 3}, {Index{0, 1}, Index{0, 1}, Index{1, 2}}};
    CHECK(ell.size() == 2);
  }

  // -- Copy/move --

  TEST_CASE("ellpack_sparsity - copy_construction", "[ellpack_sparsity]") {
    Ellpack_sparsity original{Shape{4, 5},
                              {Index{0, 1}, Index{1, 2}, Index{1, 3}}};
    Ellpack_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());
    CHECK(copy.max_nnz_per_row() == original.max_nnz_per_row());
    CHECK(copy.col_ind().data() != original.col_ind().data());
  }

  TEST_CASE("ellpack_sparsity - move_construction", "[ellpack_sparsity]") {
    Ellpack_sparsity original{Shape{4, 5}, {Index{0, 1}, Index{1, 2}}};
    auto original_size = original.size();

    Ellpack_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- CSR <-> ELL conversions --

  TEST_CASE("conversions - csr_to_ell_basic", "[conversions]") {
    Compressed_row_sparsity csr{
        Shape{4, 5}, {Index{0, 1}, Index{1, 2}, Index{1, 3}, Index{3, 4}}};

    auto ell = sparkit::data::detail::to_ellpack(csr);

    CHECK(ell.shape() == Shape(4, 5));
    CHECK(ell.size() == 4);
    CHECK(ell.max_nnz_per_row() == 2);
  }

  TEST_CASE("conversions - ell_to_csr_basic", "[conversions]") {
    Ellpack_sparsity ell{Shape{4, 5},
                         {Index{0, 1}, Index{1, 2}, Index{1, 3}, Index{3, 4}}};

    auto csr = sparkit::data::detail::to_compressed_row(ell);

    CHECK(csr.shape() == Shape(4, 5));
    CHECK(csr.size() == 4);
  }

  TEST_CASE("conversions - csr_ell_roundtrip", "[conversions]") {
    Compressed_row_sparsity original{Shape{5, 6},
                                     {Index{0, 1}, Index{0, 3}, Index{1, 1},
                                      Index{2, 0}, Index{2, 2}, Index{2, 4},
                                      Index{3, 5}, Index{4, 0}}};

    auto ell = sparkit::data::detail::to_ellpack(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(ell);

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
