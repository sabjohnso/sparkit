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
#include <sparkit/data/Ellpack_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Ellpack_matrix;
  using sparkit::data::detail::Ellpack_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- ELL matrix core --

  TEST_CASE("ellpack_matrix - construction_from_entries", "[ellpack_matrix]") {
    Ellpack_matrix<double> mat{
      Shape{4, 5},
      {{Index{0, 1}, 1.0},
       {Index{1, 2}, 2.0},
       {Index{1, 3}, 3.0},
       {Index{3, 4}, 4.0}}};

    CHECK(mat.shape() == Shape(4, 5));
    CHECK(mat.size() == 4);
  }

  TEST_CASE("ellpack_matrix - element_access", "[ellpack_matrix]") {
    Ellpack_matrix<double> mat{
      Shape{4, 5},
      {{Index{0, 1}, 1.0},
       {Index{1, 2}, 2.0},
       {Index{1, 3}, 3.0},
       {Index{3, 4}, 4.0}}};

    CHECK(mat(0, 1) == Catch::Approx(1.0));
    CHECK(mat(1, 2) == Catch::Approx(2.0));
    CHECK(mat(1, 3) == Catch::Approx(3.0));
    CHECK(mat(3, 4) == Catch::Approx(4.0));
    CHECK(mat(0, 0) == Catch::Approx(0.0));
    CHECK(mat(2, 2) == Catch::Approx(0.0));
  }

  TEST_CASE("ellpack_matrix - empty_matrix", "[ellpack_matrix]") {
    Ellpack_sparsity sp{Shape{3, 3}, {}};
    Ellpack_matrix<double> mat{sp, {}};

    CHECK(mat.size() == 0);
    CHECK(mat(0, 0) == Catch::Approx(0.0));
  }

  // -- CSR matrix <-> ELL matrix conversions --

  TEST_CASE("conversions - csr_matrix_to_ell_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{
      Shape{4, 5},
      {{Index{0, 1}, 1.0},
       {Index{1, 2}, 2.0},
       {Index{1, 3}, 3.0},
       {Index{3, 4}, 4.0}}};

    auto ell = sparkit::data::detail::to_ellpack(csr);

    CHECK(ell.shape() == Shape(4, 5));
    CHECK(ell(0, 1) == Catch::Approx(1.0));
    CHECK(ell(1, 2) == Catch::Approx(2.0));
    CHECK(ell(1, 3) == Catch::Approx(3.0));
    CHECK(ell(3, 4) == Catch::Approx(4.0));
  }

  TEST_CASE("conversions - ell_matrix_to_csr_matrix_basic", "[conversions]") {
    Ellpack_matrix<double> ell{
      Shape{4, 5},
      {{Index{0, 1}, 1.0},
       {Index{1, 2}, 2.0},
       {Index{1, 3}, 3.0},
       {Index{3, 4}, 4.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(ell);

    CHECK(csr.shape() == Shape(4, 5));
    CHECK(csr(0, 1) == Catch::Approx(1.0));
    CHECK(csr(1, 2) == Catch::Approx(2.0));
    CHECK(csr(1, 3) == Catch::Approx(3.0));
    CHECK(csr(3, 4) == Catch::Approx(4.0));
  }

  TEST_CASE("conversions - csr_matrix_ell_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{
      Shape{4, 5},
      {{Index{0, 1}, 10.0},
       {Index{1, 2}, 20.0},
       {Index{1, 3}, 30.0},
       {Index{2, 0}, 40.0},
       {Index{3, 4}, 50.0}}};

    auto ell = sparkit::data::detail::to_ellpack(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(ell);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 1) == Catch::Approx(10.0));
    CHECK(roundtrip(1, 2) == Catch::Approx(20.0));
    CHECK(roundtrip(1, 3) == Catch::Approx(30.0));
    CHECK(roundtrip(2, 0) == Catch::Approx(40.0));
    CHECK(roundtrip(3, 4) == Catch::Approx(50.0));
  }

} // end of namespace sparkit::testing
