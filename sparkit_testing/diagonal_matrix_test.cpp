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
#include <sparkit/data/Diagonal_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Diagonal_matrix;
  using sparkit::data::detail::Diagonal_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using size_type = sparkit::config::size_type;

  // -- DIA matrix core --

  TEST_CASE("diagonal_matrix - construction_from_entries",
            "[diagonal_matrix]") {
    Diagonal_matrix<double> mat{Shape{3, 3},
                                {{Index{0, 0}, 1.0},
                                 {Index{1, 1}, 2.0},
                                 {Index{2, 2}, 3.0},
                                 {Index{0, 1}, 4.0},
                                 {Index{1, 2}, 5.0}}};

    CHECK(mat.shape() == Shape(3, 3));
    CHECK(mat.size() == 5);
  }

  TEST_CASE("diagonal_matrix - element_access", "[diagonal_matrix]") {
    Diagonal_matrix<double> mat{Shape{3, 3},
                                {{Index{0, 0}, 1.0},
                                 {Index{1, 1}, 2.0},
                                 {Index{2, 2}, 3.0},
                                 {Index{0, 1}, 4.0},
                                 {Index{1, 2}, 5.0}}};

    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 1) == Catch::Approx(2.0));
    CHECK(mat(2, 2) == Catch::Approx(3.0));
    CHECK(mat(0, 1) == Catch::Approx(4.0));
    CHECK(mat(1, 2) == Catch::Approx(5.0));
    CHECK(mat(0, 2) == Catch::Approx(0.0)); // not stored
    CHECK(mat(2, 0) == Catch::Approx(0.0));
  }

  TEST_CASE("diagonal_matrix - tridiagonal_element_access",
            "[diagonal_matrix]") {
    // 4x4 tridiagonal
    Diagonal_matrix<double> mat{Shape{4, 4},
                                {{Index{0, 0}, 2.0},
                                 {Index{0, 1}, -1.0},
                                 {Index{1, 0}, -1.0},
                                 {Index{1, 1}, 2.0},
                                 {Index{1, 2}, -1.0},
                                 {Index{2, 1}, -1.0},
                                 {Index{2, 2}, 2.0},
                                 {Index{2, 3}, -1.0},
                                 {Index{3, 2}, -1.0},
                                 {Index{3, 3}, 2.0}}};

    CHECK(mat(0, 0) == Catch::Approx(2.0));
    CHECK(mat(0, 1) == Catch::Approx(-1.0));
    CHECK(mat(1, 0) == Catch::Approx(-1.0));
    CHECK(mat(3, 3) == Catch::Approx(2.0));
    CHECK(mat(0, 2) == Catch::Approx(0.0));
  }

  TEST_CASE("diagonal_matrix - empty_matrix", "[diagonal_matrix]") {
    Diagonal_sparsity sp{Shape{3, 3}, std::initializer_list<size_type>{}};
    Diagonal_matrix<double> mat{sp, {}};

    CHECK(mat.size() == 0);
    CHECK(mat(0, 0) == Catch::Approx(0.0));
  }

  // -- CSR matrix <-> DIA matrix conversions --

  TEST_CASE("conversions - csr_matrix_to_dia_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 2.0},
                                       {Index{0, 1}, -1.0},
                                       {Index{1, 0}, -1.0},
                                       {Index{1, 1}, 2.0},
                                       {Index{1, 2}, -1.0},
                                       {Index{2, 1}, -1.0},
                                       {Index{2, 2}, 2.0}}};

    auto dia = sparkit::data::detail::to_diagonal(csr);

    CHECK(dia.shape() == Shape(3, 3));
    CHECK(dia(0, 0) == Catch::Approx(2.0));
    CHECK(dia(0, 1) == Catch::Approx(-1.0));
    CHECK(dia(1, 0) == Catch::Approx(-1.0));
    CHECK(dia(2, 2) == Catch::Approx(2.0));
  }

  TEST_CASE("conversions - dia_matrix_to_csr_matrix_basic", "[conversions]") {
    Diagonal_matrix<double> dia{Shape{3, 3},
                                {{Index{0, 0}, 2.0},
                                 {Index{0, 1}, -1.0},
                                 {Index{1, 0}, -1.0},
                                 {Index{1, 1}, 2.0},
                                 {Index{1, 2}, -1.0},
                                 {Index{2, 1}, -1.0},
                                 {Index{2, 2}, 2.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(dia);

    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr(0, 0) == Catch::Approx(2.0));
    CHECK(csr(0, 1) == Catch::Approx(-1.0));
    CHECK(csr(1, 0) == Catch::Approx(-1.0));
    CHECK(csr(2, 2) == Catch::Approx(2.0));
  }

  TEST_CASE("conversions - csr_matrix_dia_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{Shape{4, 4},
                                           {{Index{0, 0}, 2.0},
                                            {Index{0, 1}, -1.0},
                                            {Index{1, 0}, -1.0},
                                            {Index{1, 1}, 2.0},
                                            {Index{1, 2}, -1.0},
                                            {Index{2, 1}, -1.0},
                                            {Index{2, 2}, 2.0},
                                            {Index{2, 3}, -1.0},
                                            {Index{3, 2}, -1.0},
                                            {Index{3, 3}, 2.0}}};

    auto dia = sparkit::data::detail::to_diagonal(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(dia);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 0) == Catch::Approx(2.0));
    CHECK(roundtrip(0, 1) == Catch::Approx(-1.0));
    CHECK(roundtrip(1, 0) == Catch::Approx(-1.0));
    CHECK(roundtrip(3, 3) == Catch::Approx(2.0));
  }

} // end of namespace sparkit::testing
