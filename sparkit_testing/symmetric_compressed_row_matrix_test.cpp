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
#include <sparkit/data/Symmetric_compressed_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_compressed_row_matrix;
  using sparkit::data::detail::Symmetric_compressed_row_sparsity;

  // -- sCSR matrix core --

  TEST_CASE("symmetric_compressed_row_matrix - construction_from_entries",
            "[symmetric_compressed_row_matrix]") {
    // 4x4 symmetric tridiagonal:
    // [[1, 2, 0, 0],
    //  [2, 3, 4, 0],
    //  [0, 4, 5, 6],
    //  [0, 0, 6, 7]]
    Symmetric_compressed_row_matrix<double> mat{Shape{4, 4},
                                                {{Index{0, 0}, 1.0},
                                                 {Index{1, 0}, 2.0},
                                                 {Index{1, 1}, 3.0},
                                                 {Index{2, 1}, 4.0},
                                                 {Index{2, 2}, 5.0},
                                                 {Index{3, 2}, 6.0},
                                                 {Index{3, 3}, 7.0}}};

    CHECK(mat.shape() == Shape(4, 4));
    CHECK(mat.size() == 7);
  }

  TEST_CASE("symmetric_compressed_row_matrix - element_access_lower_triangle",
            "[symmetric_compressed_row_matrix]") {
    Symmetric_compressed_row_matrix<double> mat{Shape{4, 4},
                                                {{Index{0, 0}, 1.0},
                                                 {Index{1, 0}, 2.0},
                                                 {Index{1, 1}, 3.0},
                                                 {Index{2, 1}, 4.0},
                                                 {Index{2, 2}, 5.0},
                                                 {Index{3, 2}, 6.0},
                                                 {Index{3, 3}, 7.0}}};

    // Lower triangle stored directly
    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(1, 1) == Catch::Approx(3.0));
    CHECK(mat(2, 1) == Catch::Approx(4.0));
    CHECK(mat(2, 2) == Catch::Approx(5.0));
    CHECK(mat(3, 2) == Catch::Approx(6.0));
    CHECK(mat(3, 3) == Catch::Approx(7.0));
  }

  TEST_CASE("symmetric_compressed_row_matrix - element_access_upper_triangle",
            "[symmetric_compressed_row_matrix]") {
    Symmetric_compressed_row_matrix<double> mat{Shape{4, 4},
                                                {{Index{0, 0}, 1.0},
                                                 {Index{1, 0}, 2.0},
                                                 {Index{1, 1}, 3.0},
                                                 {Index{2, 1}, 4.0},
                                                 {Index{2, 2}, 5.0},
                                                 {Index{3, 2}, 6.0},
                                                 {Index{3, 3}, 7.0}}};

    // Upper triangle via symmetry: A(i,j) = A(j,i)
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(1, 2) == Catch::Approx(4.0));
    CHECK(mat(2, 3) == Catch::Approx(6.0));
  }

  TEST_CASE("symmetric_compressed_row_matrix - element_access_zeros",
            "[symmetric_compressed_row_matrix]") {
    Symmetric_compressed_row_matrix<double> mat{Shape{4, 4},
                                                {{Index{0, 0}, 1.0},
                                                 {Index{1, 0}, 2.0},
                                                 {Index{1, 1}, 3.0},
                                                 {Index{2, 1}, 4.0},
                                                 {Index{2, 2}, 5.0},
                                                 {Index{3, 2}, 6.0},
                                                 {Index{3, 3}, 7.0}}};

    // Structural zeros
    CHECK(mat(0, 2) == Catch::Approx(0.0));
    CHECK(mat(0, 3) == Catch::Approx(0.0));
    CHECK(mat(2, 0) == Catch::Approx(0.0));
    CHECK(mat(3, 0) == Catch::Approx(0.0));
    CHECK(mat(3, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("symmetric_compressed_row_matrix - construction_from_upper_entries",
            "[symmetric_compressed_row_matrix]") {
    // Provide upper-triangle entries — should be normalized
    Symmetric_compressed_row_matrix<double> mat{Shape{3, 3},
                                                {{Index{0, 0}, 1.0},
                                                 {Index{0, 1}, 2.0},
                                                 {Index{1, 1}, 3.0},
                                                 {Index{0, 2}, 4.0},
                                                 {Index{2, 2}, 5.0}}};

    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(1, 1) == Catch::Approx(3.0));
    CHECK(mat(2, 0) == Catch::Approx(4.0));
    CHECK(mat(0, 2) == Catch::Approx(4.0));
    CHECK(mat(2, 2) == Catch::Approx(5.0));
  }

  // -- sCSR matrix <-> CSR matrix conversions --

  TEST_CASE("conversions - scsr_matrix_to_csr_matrix_basic", "[conversions]") {
    Symmetric_compressed_row_matrix<double> scsr{Shape{3, 3},
                                                 {{Index{0, 0}, 1.0},
                                                  {Index{1, 0}, 2.0},
                                                  {Index{1, 1}, 3.0},
                                                  {Index{2, 1}, 4.0},
                                                  {Index{2, 2}, 5.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(scsr);

    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr(0, 0) == Catch::Approx(1.0));
    CHECK(csr(0, 1) == Catch::Approx(2.0));
    CHECK(csr(1, 0) == Catch::Approx(2.0));
    CHECK(csr(1, 1) == Catch::Approx(3.0));
    CHECK(csr(1, 2) == Catch::Approx(4.0));
    CHECK(csr(2, 1) == Catch::Approx(4.0));
    CHECK(csr(2, 2) == Catch::Approx(5.0));
  }

  TEST_CASE("conversions - csr_matrix_to_scsr_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{Shape{3, 3},
                                      {{Index{0, 0}, 1.0},
                                       {Index{0, 1}, 2.0},
                                       {Index{1, 0}, 2.0},
                                       {Index{1, 1}, 3.0},
                                       {Index{1, 2}, 4.0},
                                       {Index{2, 1}, 4.0},
                                       {Index{2, 2}, 5.0}}};

    auto scsr = sparkit::data::detail::to_symmetric_compressed_row(csr);

    CHECK(scsr.shape() == Shape(3, 3));
    CHECK(scsr(0, 0) == Catch::Approx(1.0));
    CHECK(scsr(1, 0) == Catch::Approx(2.0));
    CHECK(scsr(0, 1) == Catch::Approx(2.0));
    CHECK(scsr(1, 1) == Catch::Approx(3.0));
    CHECK(scsr(2, 1) == Catch::Approx(4.0));
    CHECK(scsr(1, 2) == Catch::Approx(4.0));
    CHECK(scsr(2, 2) == Catch::Approx(5.0));
  }

  TEST_CASE("conversions - scsr_matrix_csr_matrix_roundtrip", "[conversions]") {
    Symmetric_compressed_row_matrix<double> original{Shape{4, 4},
                                                     {{Index{0, 0}, 1.0},
                                                      {Index{1, 0}, 2.0},
                                                      {Index{1, 1}, 3.0},
                                                      {Index{2, 1}, 4.0},
                                                      {Index{2, 2}, 5.0},
                                                      {Index{3, 2}, 6.0},
                                                      {Index{3, 3}, 7.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(original);
    auto roundtrip = sparkit::data::detail::to_symmetric_compressed_row(csr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 0) == Catch::Approx(1.0));
    CHECK(roundtrip(1, 0) == Catch::Approx(2.0));
    CHECK(roundtrip(0, 1) == Catch::Approx(2.0));
    CHECK(roundtrip(1, 1) == Catch::Approx(3.0));
    CHECK(roundtrip(2, 1) == Catch::Approx(4.0));
    CHECK(roundtrip(1, 2) == Catch::Approx(4.0));
    CHECK(roundtrip(3, 3) == Catch::Approx(7.0));
  }

} // end of namespace sparkit::testing
