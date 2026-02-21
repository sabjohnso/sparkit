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
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Jagged_diagonal_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Jagged_diagonal_matrix;
  using sparkit::data::detail::Jagged_diagonal_sparsity;
  using sparkit::data::detail::Shape;

  // -- JAD matrix core --

  TEST_CASE("jagged_diagonal_matrix - construction_from_entries",
            "[jagged_diagonal_matrix]") {
    Jagged_diagonal_matrix<double> mat{Shape{4, 5},
                                       {{Index{0, 1}, 1.0},
                                        {Index{0, 3}, 2.0},
                                        {Index{1, 0}, 3.0},
                                        {Index{1, 2}, 4.0},
                                        {Index{1, 4}, 5.0},
                                        {Index{2, 1}, 6.0},
                                        {Index{3, 0}, 7.0},
                                        {Index{3, 3}, 8.0}}};

    CHECK(mat.shape() == Shape(4, 5));
    CHECK(mat.size() == 8);
  }

  TEST_CASE("jagged_diagonal_matrix - element_access",
            "[jagged_diagonal_matrix]") {
    Jagged_diagonal_matrix<double> mat{Shape{4, 5},
                                       {{Index{0, 1}, 1.0},
                                        {Index{0, 3}, 2.0},
                                        {Index{1, 0}, 3.0},
                                        {Index{1, 2}, 4.0},
                                        {Index{1, 4}, 5.0},
                                        {Index{2, 1}, 6.0},
                                        {Index{3, 0}, 7.0},
                                        {Index{3, 3}, 8.0}}};

    CHECK(mat(0, 1) == Catch::Approx(1.0));
    CHECK(mat(0, 3) == Catch::Approx(2.0));
    CHECK(mat(1, 0) == Catch::Approx(3.0));
    CHECK(mat(1, 2) == Catch::Approx(4.0));
    CHECK(mat(1, 4) == Catch::Approx(5.0));
    CHECK(mat(2, 1) == Catch::Approx(6.0));
    CHECK(mat(3, 0) == Catch::Approx(7.0));
    CHECK(mat(3, 3) == Catch::Approx(8.0));

    // Zero entries
    CHECK(mat(0, 0) == Catch::Approx(0.0));
    CHECK(mat(2, 2) == Catch::Approx(0.0));
    CHECK(mat(3, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("jagged_diagonal_matrix - empty_matrix",
            "[jagged_diagonal_matrix]") {
    Jagged_diagonal_sparsity sp{Shape{3, 3}, {}};
    Jagged_diagonal_matrix<double> mat{sp, {}};

    CHECK(mat.size() == 0);
    CHECK(mat(0, 0) == Catch::Approx(0.0));
  }

  // -- CSR matrix <-> JAD matrix conversions --

  TEST_CASE("conversions - csr_matrix_to_jad_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{Shape{4, 5},
                                      {{Index{0, 1}, 1.0},
                                       {Index{0, 3}, 2.0},
                                       {Index{1, 0}, 3.0},
                                       {Index{1, 2}, 4.0},
                                       {Index{1, 4}, 5.0},
                                       {Index{2, 1}, 6.0},
                                       {Index{3, 0}, 7.0},
                                       {Index{3, 3}, 8.0}}};

    auto jad = sparkit::data::detail::to_jagged_diagonal(csr);

    CHECK(jad.shape() == Shape(4, 5));
    CHECK(jad(0, 1) == Catch::Approx(1.0));
    CHECK(jad(0, 3) == Catch::Approx(2.0));
    CHECK(jad(1, 0) == Catch::Approx(3.0));
    CHECK(jad(1, 2) == Catch::Approx(4.0));
    CHECK(jad(1, 4) == Catch::Approx(5.0));
    CHECK(jad(2, 1) == Catch::Approx(6.0));
    CHECK(jad(3, 0) == Catch::Approx(7.0));
    CHECK(jad(3, 3) == Catch::Approx(8.0));
  }

  TEST_CASE("conversions - jad_matrix_to_csr_matrix_basic", "[conversions]") {
    Jagged_diagonal_matrix<double> jad{Shape{4, 5},
                                       {{Index{0, 1}, 1.0},
                                        {Index{1, 2}, 2.0},
                                        {Index{1, 3}, 3.0},
                                        {Index{3, 4}, 4.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(jad);

    CHECK(csr.shape() == Shape(4, 5));
    CHECK(csr(0, 1) == Catch::Approx(1.0));
    CHECK(csr(1, 2) == Catch::Approx(2.0));
    CHECK(csr(1, 3) == Catch::Approx(3.0));
    CHECK(csr(3, 4) == Catch::Approx(4.0));
  }

  TEST_CASE("conversions - csr_matrix_jad_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{Shape{4, 5},
                                           {{Index{0, 1}, 10.0},
                                            {Index{1, 2}, 20.0},
                                            {Index{1, 3}, 30.0},
                                            {Index{2, 0}, 40.0},
                                            {Index{3, 4}, 50.0}}};

    auto jad = sparkit::data::detail::to_jagged_diagonal(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(jad);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 1) == Catch::Approx(10.0));
    CHECK(roundtrip(1, 2) == Catch::Approx(20.0));
    CHECK(roundtrip(1, 3) == Catch::Approx(30.0));
    CHECK(roundtrip(2, 0) == Catch::Approx(40.0));
    CHECK(roundtrip(3, 4) == Catch::Approx(50.0));
  }

} // end of namespace sparkit::testing
