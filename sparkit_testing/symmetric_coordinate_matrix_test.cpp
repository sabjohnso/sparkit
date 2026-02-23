//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Symmetric_compressed_row_matrix.hpp>
#include <sparkit/data/Symmetric_coordinate_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Symmetric_compressed_row_matrix;
  using sparkit::data::detail::Symmetric_coordinate_matrix;

  // -- sCOO matrix core --

  TEST_CASE(
    "symmetric_coordinate_matrix - construction_from_entries",
    "[symmetric_coordinate_matrix]") {
    Symmetric_coordinate_matrix<double> mat{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{1, 0}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{2, 1}, 4.0},
       {Index{2, 2}, 5.0}}};

    CHECK(mat.shape() == Shape(3, 3));
    CHECK(mat.size() == 5);
  }

  TEST_CASE(
    "symmetric_coordinate_matrix - element_access_both_triangles",
    "[symmetric_coordinate_matrix]") {
    Symmetric_coordinate_matrix<double> mat{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{1, 0}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{2, 1}, 4.0},
       {Index{2, 2}, 5.0}}};

    // Lower triangle
    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(1, 1) == Catch::Approx(3.0));
    CHECK(mat(2, 1) == Catch::Approx(4.0));
    CHECK(mat(2, 2) == Catch::Approx(5.0));

    // Upper triangle via symmetry
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(1, 2) == Catch::Approx(4.0));

    // Structural zeros
    CHECK(mat(0, 2) == Catch::Approx(0.0));
    CHECK(mat(2, 0) == Catch::Approx(0.0));
  }

  TEST_CASE(
    "symmetric_coordinate_matrix - add_normalizes",
    "[symmetric_coordinate_matrix]") {
    Symmetric_coordinate_matrix<double> mat{Shape{3, 3}};

    mat.add(Index{0, 1}, 7.0); // upper -> stored as (1,0)
    CHECK(mat(0, 1) == Catch::Approx(7.0));
    CHECK(mat(1, 0) == Catch::Approx(7.0));
    CHECK(mat.size() == 1);
  }

  TEST_CASE(
    "symmetric_coordinate_matrix - remove_normalizes",
    "[symmetric_coordinate_matrix]") {
    Symmetric_coordinate_matrix<double> mat{
      Shape{3, 3}, {{Index{1, 0}, 2.0}, {Index{2, 2}, 5.0}}};
    CHECK(mat.size() == 2);

    mat.remove(Index{0, 1}); // normalized to (1,0)
    CHECK(mat.size() == 1);
    CHECK(mat(1, 0) == Catch::Approx(0.0));
  }

  TEST_CASE(
    "symmetric_coordinate_matrix - construction_from_upper_entries",
    "[symmetric_coordinate_matrix]") {
    Symmetric_coordinate_matrix<double> mat{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{0, 1}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{0, 2}, 4.0},
       {Index{2, 2}, 5.0}}};

    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(2, 0) == Catch::Approx(4.0));
    CHECK(mat(0, 2) == Catch::Approx(4.0));
  }

  // -- sCOO matrix -> sCSR matrix conversion --

  TEST_CASE("conversions - scoo_matrix_to_scsr_matrix_basic", "[conversions]") {
    Symmetric_coordinate_matrix<double> scoo{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{1, 0}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{2, 1}, 4.0},
       {Index{2, 2}, 5.0}}};

    auto scsr = sparkit::data::detail::to_symmetric_compressed_row(scoo);

    CHECK(scsr.shape() == Shape(3, 3));
    CHECK(scsr(0, 0) == Catch::Approx(1.0));
    CHECK(scsr(1, 0) == Catch::Approx(2.0));
    CHECK(scsr(0, 1) == Catch::Approx(2.0));
    CHECK(scsr(2, 2) == Catch::Approx(5.0));
  }

  // -- sCOO matrix -> CSR matrix conversion --

  TEST_CASE(
    "conversions - scoo_matrix_to_csr_matrix_expands", "[conversions]") {
    Symmetric_coordinate_matrix<double> scoo{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{1, 0}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{2, 1}, 4.0},
       {Index{2, 2}, 5.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(scoo);

    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr.size() == 7);
    CHECK(csr(0, 0) == Catch::Approx(1.0));
    CHECK(csr(0, 1) == Catch::Approx(2.0));
    CHECK(csr(1, 0) == Catch::Approx(2.0));
  }

} // end of namespace sparkit::testing
