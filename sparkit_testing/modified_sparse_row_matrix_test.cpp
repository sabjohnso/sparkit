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
#include <sparkit/data/Modified_sparse_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Modified_sparse_row_matrix;
  using sparkit::data::detail::Modified_sparse_row_sparsity;
  using sparkit::data::detail::Shape;

  // -- MSR matrix core --

  TEST_CASE(
    "modified_sparse_row_matrix - construction_and_accessors",
    "[modified_sparse_row_matrix]") {
    Modified_sparse_row_matrix<double> mat{
      Shape{4, 4},
      {{Index{0, 0}, 1.0},
       {Index{1, 3}, 2.0},
       {Index{2, 0}, 3.0},
       {Index{2, 2}, 4.0}}};

    CHECK(mat.shape() == Shape(4, 4));
    CHECK(mat.size() == 4);
  }

  TEST_CASE(
    "modified_sparse_row_matrix - element_access",
    "[modified_sparse_row_matrix]") {
    Modified_sparse_row_matrix<double> mat{
      Shape{4, 4},
      {{Index{0, 0}, 1.0},
       {Index{1, 3}, 2.0},
       {Index{2, 0}, 3.0},
       {Index{2, 2}, 4.0}}};

    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 3) == Catch::Approx(2.0));
    CHECK(mat(2, 0) == Catch::Approx(3.0));
    CHECK(mat(2, 2) == Catch::Approx(4.0));
    CHECK(mat(3, 3) == Catch::Approx(0.0));
    CHECK(mat(0, 1) == Catch::Approx(0.0));
  }

  TEST_CASE(
    "modified_sparse_row_matrix - diagonal_accessor",
    "[modified_sparse_row_matrix]") {
    Modified_sparse_row_matrix<double> mat{
      Shape{3, 3},
      {{Index{0, 0}, 10.0},
       {Index{0, 2}, 5.0},
       {Index{1, 1}, 20.0},
       {Index{2, 2}, 30.0}}};

    auto diag = mat.diagonal();
    REQUIRE(std::ssize(diag) == 3);
    CHECK(diag[0] == Catch::Approx(10.0));
    CHECK(diag[1] == Catch::Approx(20.0));
    CHECK(diag[2] == Catch::Approx(30.0));
  }

  TEST_CASE(
    "modified_sparse_row_matrix - empty_matrix",
    "[modified_sparse_row_matrix]") {
    Modified_sparse_row_sparsity sp{Shape{3, 3}, {}};
    Modified_sparse_row_matrix<double> mat{sp, {}, {}};

    CHECK(mat.size() == 0);
    CHECK(mat.shape() == Shape(3, 3));
  }

  // -- CSR matrix <-> MSR matrix conversion --

  TEST_CASE("conversions - csr_matrix_to_msr_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{
      Shape{4, 4},
      {{Index{0, 0}, 1.0},
       {Index{1, 3}, 2.0},
       {Index{2, 0}, 3.0},
       {Index{2, 2}, 4.0}}};

    auto msr = sparkit::data::detail::to_modified_sparse_row(csr);

    CHECK(msr.shape() == Shape(4, 4));
    CHECK(msr(0, 0) == Catch::Approx(1.0));
    CHECK(msr(1, 3) == Catch::Approx(2.0));
    CHECK(msr(2, 0) == Catch::Approx(3.0));
    CHECK(msr(2, 2) == Catch::Approx(4.0));
  }

  TEST_CASE("conversions - msr_matrix_to_csr_matrix_basic", "[conversions]") {
    Modified_sparse_row_matrix<double> msr{
      Shape{4, 4},
      {{Index{0, 0}, 1.0},
       {Index{1, 3}, 2.0},
       {Index{2, 0}, 3.0},
       {Index{2, 2}, 4.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(msr);

    CHECK(csr.shape() == Shape(4, 4));
    CHECK(csr(0, 0) == Catch::Approx(1.0));
    CHECK(csr(1, 3) == Catch::Approx(2.0));
    CHECK(csr(2, 0) == Catch::Approx(3.0));
    CHECK(csr(2, 2) == Catch::Approx(4.0));
  }

  TEST_CASE("conversions - csr_matrix_msr_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{
      Shape{4, 4},
      {{Index{0, 0}, 10.0},
       {Index{0, 3}, 20.0},
       {Index{1, 1}, 30.0},
       {Index{2, 0}, 40.0},
       {Index{2, 2}, 50.0},
       {Index{3, 3}, 60.0}}};

    auto msr = sparkit::data::detail::to_modified_sparse_row(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(msr);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 0) == Catch::Approx(10.0));
    CHECK(roundtrip(0, 3) == Catch::Approx(20.0));
    CHECK(roundtrip(1, 1) == Catch::Approx(30.0));
    CHECK(roundtrip(2, 0) == Catch::Approx(40.0));
    CHECK(roundtrip(2, 2) == Catch::Approx(50.0));
    CHECK(roundtrip(3, 3) == Catch::Approx(60.0));
  }

} // end of namespace sparkit::testing
