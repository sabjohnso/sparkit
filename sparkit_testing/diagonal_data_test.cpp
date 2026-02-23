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
#include <sparkit/data/Diagonal_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Diagonal_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using size_type = sparkit::config::size_type;

  // -- DIA construction from offsets --

  TEST_CASE(
    "diagonal_sparsity - construction_from_offsets_tridiagonal",
    "[diagonal_sparsity]") {
    // 4x4 tridiagonal: offsets -1, 0, 1
    Diagonal_sparsity dia{Shape{4, 4}, {-1, 0, 1}};
    CHECK(dia.shape() == Shape(4, 4));
    CHECK(dia.num_diagonals() == 3);
    // Tridiagonal 4x4: main=4, sub=3, super=3 => 10 entries
    CHECK(dia.size() == 10);
  }

  TEST_CASE(
    "diagonal_sparsity - construction_from_offsets_main_diagonal_only",
    "[diagonal_sparsity]") {
    Diagonal_sparsity dia{Shape{5, 5}, {0}};
    CHECK(dia.num_diagonals() == 1);
    CHECK(dia.size() == 5);
  }

  TEST_CASE("diagonal_sparsity - construction_empty", "[diagonal_sparsity]") {
    Diagonal_sparsity dia{Shape{3, 3}, std::initializer_list<size_type>{}};
    CHECK(dia.size() == 0);
    CHECK(dia.num_diagonals() == 0);
  }

  // -- Offset accessor --

  TEST_CASE("diagonal_sparsity - offsets_sorted", "[diagonal_sparsity]") {
    Diagonal_sparsity dia{Shape{4, 4}, {1, -1, 0}};

    auto off = dia.offsets();
    REQUIRE(std::ssize(off) == 3);
    CHECK(off[0] == -1);
    CHECK(off[1] == 0);
    CHECK(off[2] == 1);
  }

  // -- Construction from indices --

  TEST_CASE(
    "diagonal_sparsity - construction_from_indices", "[diagonal_sparsity]") {
    Diagonal_sparsity dia{
      Shape{4, 4},
      {Index{0, 0},
       Index{1, 1},
       Index{2, 2}, // main diagonal
       Index{0, 1},
       Index{1, 2},
       Index{2, 3}}}; // super diagonal

    CHECK(dia.num_diagonals() == 2);
    auto off = dia.offsets();
    REQUIRE(std::ssize(off) == 2);
    CHECK(off[0] == 0); // main
    CHECK(off[1] == 1); // super

    // main=4, super=3 => 7 entries
    CHECK(dia.size() == 7);
  }

  // -- Rectangular matrix --

  TEST_CASE("diagonal_sparsity - rectangular_matrix", "[diagonal_sparsity]") {
    // 3x5 matrix with main diagonal (offset 0): min(3,5-0)=3 entries
    // and offset 2: valid positions are (0,2),(1,3),(2,4) => 3 entries
    Diagonal_sparsity dia{Shape{3, 5}, {0, 2}};
    CHECK(dia.size() == 6);
  }

  // -- Banded matrix --

  TEST_CASE("diagonal_sparsity - banded_matrix", "[diagonal_sparsity]") {
    // 5x5 pentadiagonal: offsets -2,-1,0,1,2
    Diagonal_sparsity dia{Shape{5, 5}, {-2, -1, 0, 1, 2}};
    CHECK(dia.num_diagonals() == 5);
    // main=5, sub1=4, sub2=3, super1=4, super2=3 => 19
    CHECK(dia.size() == 19);
  }

  // -- Duplicate offsets collapsed --

  TEST_CASE(
    "diagonal_sparsity - duplicate_offsets_collapsed", "[diagonal_sparsity]") {
    Diagonal_sparsity dia{Shape{4, 4}, {0, 1, 0, 1}};
    CHECK(dia.num_diagonals() == 2);
  }

  // -- Copy/move --

  TEST_CASE("diagonal_sparsity - copy_construction", "[diagonal_sparsity]") {
    Diagonal_sparsity original{Shape{4, 4}, {-1, 0, 1}};
    Diagonal_sparsity copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());
    CHECK(copy.num_diagonals() == original.num_diagonals());

    auto orig_off = original.offsets();
    auto copy_off = copy.offsets();
    CHECK(copy_off.data() != orig_off.data());
  }

  TEST_CASE("diagonal_sparsity - move_construction", "[diagonal_sparsity]") {
    Diagonal_sparsity original{Shape{4, 4}, {-1, 0, 1}};
    auto original_size = original.size();

    Diagonal_sparsity moved{std::move(original)};
    CHECK(moved.size() == original_size);
  }

  // -- CSR <-> DIA conversions --

  TEST_CASE("conversions - csr_to_dia_basic", "[conversions]") {
    // 4x4 tridiagonal
    Compressed_row_sparsity csr{
      Shape{4, 4},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3}}};

    auto dia = sparkit::data::detail::to_diagonal(csr);

    CHECK(dia.shape() == Shape(4, 4));
    CHECK(dia.num_diagonals() == 3);
    CHECK(dia.size() == 10);

    auto off = dia.offsets();
    CHECK(off[0] == -1);
    CHECK(off[1] == 0);
    CHECK(off[2] == 1);
  }

  TEST_CASE("conversions - dia_to_csr_basic", "[conversions]") {
    Diagonal_sparsity dia{Shape{4, 4}, {-1, 0, 1}};
    auto csr = sparkit::data::detail::to_compressed_row(dia);

    CHECK(csr.shape() == Shape(4, 4));
    CHECK(csr.size() == 10);
  }

  TEST_CASE("conversions - csr_dia_roundtrip", "[conversions]") {
    // 4x4 tridiagonal
    Compressed_row_sparsity original{
      Shape{4, 4},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3}}};

    auto dia = sparkit::data::detail::to_diagonal(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(dia);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());

    auto orig_rp = original.row_ptr();
    auto rt_rp = roundtrip.row_ptr();
    REQUIRE(std::ssize(rt_rp) == std::ssize(orig_rp));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_rp); ++i) {
      CHECK(rt_rp[i] == orig_rp[i]);
    }
  }

} // end of namespace sparkit::testing
