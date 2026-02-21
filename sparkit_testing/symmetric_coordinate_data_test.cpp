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
#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>
#include <sparkit/data/Symmetric_coordinate_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_compressed_row_sparsity;
  using sparkit::data::detail::Symmetric_coordinate_sparsity;
  using size_type = sparkit::config::size_type;

  // -- sCOO construction --

  TEST_CASE("symmetric_coordinate_sparsity - construction_from_lower_triangle",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{
        Shape{3, 3},
        {Index{0, 0}, Index{1, 0}, Index{1, 1}, Index{2, 1}, Index{2, 2}}};

    CHECK(scoo.shape() == Shape(3, 3));
    CHECK(scoo.size() == 5);
  }

  TEST_CASE(
      "symmetric_coordinate_sparsity - construction_normalizes_upper_to_lower",
      "[symmetric_coordinate_sparsity]") {
    // (0,1) -> (1,0), (1,2) -> (2,1)
    Symmetric_coordinate_sparsity scoo{
        Shape{3, 3},
        {Index{0, 0}, Index{0, 1}, Index{1, 1}, Index{1, 2}, Index{2, 2}}};

    CHECK(scoo.size() == 5);

    // All stored indices should have row >= col
    auto idx = scoo.indices();
    for (auto const& i : idx) {
      CHECK(i.row() >= i.column());
    }
  }

  TEST_CASE("symmetric_coordinate_sparsity - construction_empty",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{Shape{4, 4}, {}};
    CHECK(scoo.shape() == Shape(4, 4));
    CHECK(scoo.size() == 0);
  }

  // -- Add/remove with normalization --

  TEST_CASE("symmetric_coordinate_sparsity - add_normalizes_to_lower",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{Shape{3, 3}, {}};

    scoo.add(Index{0, 1}); // upper -> stored as (1,0)
    CHECK(scoo.size() == 1);

    auto idx = scoo.indices();
    REQUIRE(std::ssize(idx) == 1);
    CHECK(idx[0].row() == 1);
    CHECK(idx[0].column() == 0);
  }

  TEST_CASE("symmetric_coordinate_sparsity - add_duplicate_via_normalization",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{Shape{3, 3}, {}};

    scoo.add(Index{0, 1}); // stored as (1,0)
    scoo.add(Index{1, 0}); // same after normalization
    CHECK(scoo.size() == 1);
  }

  TEST_CASE("symmetric_coordinate_sparsity - remove_normalizes",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{Shape{3, 3}, {Index{1, 0}, Index{2, 2}}};
    CHECK(scoo.size() == 2);

    scoo.remove(Index{0, 1}); // normalized to (1,0) — should remove
    CHECK(scoo.size() == 1);
  }

  // -- Indices accessor --

  TEST_CASE("symmetric_coordinate_sparsity - indices_all_lower_triangle",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity scoo{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{1, 1}, Index{3, 3}}};

    auto idx = scoo.indices();
    CHECK(std::ssize(idx) == 5);

    for (auto const& i : idx) {
      CHECK(i.row() >= i.column());
    }
  }

  // -- Copy/move --

  TEST_CASE("symmetric_coordinate_sparsity - copy_construction",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity original{Shape{3, 3},
                                           {Index{0, 0}, Index{1, 0}}};
    Symmetric_coordinate_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());
  }

  TEST_CASE("symmetric_coordinate_sparsity - move_construction",
            "[symmetric_coordinate_sparsity]") {
    Symmetric_coordinate_sparsity original{Shape{3, 3},
                                           {Index{0, 0}, Index{1, 0}}};
    auto original_size = original.size();

    Symmetric_coordinate_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- sCOO -> sCSR conversion --

  TEST_CASE("conversions - scoo_to_scsr_basic", "[conversions]") {
    Symmetric_coordinate_sparsity scoo{
        Shape{3, 3},
        {Index{0, 0}, Index{1, 0}, Index{1, 1}, Index{2, 1}, Index{2, 2}}};

    auto scsr = sparkit::data::detail::to_symmetric_compressed_row(scoo);

    CHECK(scsr.shape() == Shape(3, 3));
    CHECK(scsr.size() == 5);
  }

  // -- sCOO -> CSR conversion --

  TEST_CASE("conversions - scoo_to_csr_expands", "[conversions]") {
    Symmetric_coordinate_sparsity scoo{
        Shape{3, 3},
        {Index{0, 0}, Index{1, 0}, Index{1, 1}, Index{2, 1}, Index{2, 2}}};

    auto csr = sparkit::data::detail::to_compressed_row(scoo);

    CHECK(csr.shape() == Shape(3, 3));
    // 5 stored, 2 off-diagonal -> 7 expanded
    CHECK(csr.size() == 7);
  }

} // end of namespace sparkit::testing
