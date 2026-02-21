//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Block_sparse_row_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Block_sparse_row_matrix;
  using sparkit::data::detail::Block_sparse_row_sparsity;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- BSR matrix core --

  TEST_CASE("block_sparse_row_matrix - construction_from_entries",
            "[block_sparse_row_matrix]") {
    Block_sparse_row_matrix<double> mat{Shape{4, 4},
                                        2,
                                        2,
                                        {{Index{0, 0}, 1.0},
                                         {Index{0, 1}, 2.0},
                                         {Index{1, 0}, 3.0},
                                         {Index{1, 1}, 4.0},
                                         {Index{2, 2}, 5.0},
                                         {Index{2, 3}, 6.0},
                                         {Index{3, 2}, 7.0},
                                         {Index{3, 3}, 8.0}}};

    CHECK(mat.shape() == Shape(4, 4));
    CHECK(mat.size() == 8);
  }

  TEST_CASE("block_sparse_row_matrix - element_access",
            "[block_sparse_row_matrix]") {
    Block_sparse_row_matrix<double> mat{Shape{4, 4},
                                        2,
                                        2,
                                        {{Index{0, 0}, 1.0},
                                         {Index{0, 1}, 2.0},
                                         {Index{1, 0}, 3.0},
                                         {Index{1, 1}, 4.0},
                                         {Index{2, 2}, 5.0},
                                         {Index{2, 3}, 6.0},
                                         {Index{3, 2}, 7.0},
                                         {Index{3, 3}, 8.0}}};

    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(1, 0) == Catch::Approx(3.0));
    CHECK(mat(1, 1) == Catch::Approx(4.0));
    CHECK(mat(2, 2) == Catch::Approx(5.0));
    CHECK(mat(3, 3) == Catch::Approx(8.0));
    CHECK(mat(0, 2) == Catch::Approx(0.0)); // block not stored
    CHECK(mat(2, 0) == Catch::Approx(0.0));
  }

  // -- CSR matrix <-> BSR matrix conversions --

  TEST_CASE("conversions - csr_matrix_to_bsr_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{Shape{4, 4},
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 1}, 4.0},
                                       {Index{2, 2}, 5.0},
                                       {Index{2, 3}, 6.0},
                                       {Index{3, 2}, 7.0},
                                       {Index{3, 3}, 8.0}}};

    auto bsr = sparkit::data::detail::to_block_sparse_row(csr, 2, 2);

    CHECK(bsr.shape() == Shape(4, 4));
    CHECK(bsr(0, 0) == Catch::Approx(1.0));
    CHECK(bsr(1, 1) == Catch::Approx(4.0));
    CHECK(bsr(2, 2) == Catch::Approx(5.0));
    CHECK(bsr(3, 3) == Catch::Approx(8.0));
  }

  TEST_CASE("conversions - bsr_matrix_to_csr_matrix_basic", "[conversions]") {
    Block_sparse_row_matrix<double> bsr{Shape{4, 4},
                                        2,
                                        2,
                                        {{Index{0, 0}, 1.0},
                                         {Index{0, 1}, 2.0},
                                         {Index{1, 0}, 3.0},
                                         {Index{1, 1}, 4.0},
                                         {Index{2, 2}, 5.0},
                                         {Index{2, 3}, 6.0},
                                         {Index{3, 2}, 7.0},
                                         {Index{3, 3}, 8.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(bsr);

    CHECK(csr.shape() == Shape(4, 4));
    CHECK(csr(0, 0) == Catch::Approx(1.0));
    CHECK(csr(1, 1) == Catch::Approx(4.0));
    CHECK(csr(2, 2) == Catch::Approx(5.0));
    CHECK(csr(3, 3) == Catch::Approx(8.0));
  }

  TEST_CASE("conversions - csr_matrix_bsr_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{Shape{4, 4},
                                           {{Index{0, 0}, 1.0},
                                            {Index{0, 1}, 2.0},
                                            {Index{1, 0}, 3.0},
                                            {Index{1, 1}, 4.0},
                                            {Index{2, 2}, 5.0},
                                            {Index{2, 3}, 6.0},
                                            {Index{3, 2}, 7.0},
                                            {Index{3, 3}, 8.0}}};

    auto bsr = sparkit::data::detail::to_block_sparse_row(original, 2, 2);
    auto roundtrip = sparkit::data::detail::to_compressed_row(bsr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip(0, 0) == Catch::Approx(1.0));
    CHECK(roundtrip(0, 1) == Catch::Approx(2.0));
    CHECK(roundtrip(1, 0) == Catch::Approx(3.0));
    CHECK(roundtrip(1, 1) == Catch::Approx(4.0));
    CHECK(roundtrip(2, 2) == Catch::Approx(5.0));
    CHECK(roundtrip(3, 3) == Catch::Approx(8.0));
  }

} // end of namespace sparkit::testing
