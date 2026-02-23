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
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_column_matrix;
  using sparkit::data::detail::Compressed_column_sparsity;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  // -- Compressed_column_matrix core --

  TEST_CASE(
    "compressed_column_matrix - construction_and_accessors",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{
      Shape{6, 5}, {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat.shape() == Shape(6, 5));
    CHECK(mat.size() == 3);

    auto vals = mat.values();
    REQUIRE(std::ssize(vals) == 3);
    CHECK(vals[0] == Catch::Approx(1.0));
    CHECK(vals[1] == Catch::Approx(2.0));
    CHECK(vals[2] == Catch::Approx(3.0));
  }

  TEST_CASE(
    "compressed_column_matrix - structural_accessors_delegate",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{
      Shape{6, 5}, {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    auto cp = mat.col_ptr();
    auto ri = mat.row_ind();

    REQUIRE(std::ssize(cp) == 6);
    CHECK(cp[0] == 0);
    CHECK(cp[3] == 2);
    CHECK(cp[4] == 3);

    REQUIRE(std::ssize(ri) == 3);
    CHECK(ri[0] == 2);
    CHECK(ri[1] == 4);
    CHECK(ri[2] == 5);
  }

  TEST_CASE(
    "compressed_column_matrix - sparsity_accessor",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{Shape{6, 5}, {Index{2, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0}};

    auto const& sp_ref = mat.sparsity();
    CHECK(sp_ref.shape() == Shape(6, 5));
    CHECK(sp_ref.size() == 2);
  }

  TEST_CASE(
    "compressed_column_matrix - empty_matrix", "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{Shape{3, 3}, {}};
    Compressed_column_matrix<double> mat{sp, {}};

    CHECK(mat.size() == 0);
    CHECK(mat.shape() == Shape(3, 3));
    CHECK(mat.values().empty());
  }

  // -- Element access --

  TEST_CASE(
    "compressed_column_matrix - element_access_existing",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{
      Shape{6, 5}, {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat(2, 2) == Catch::Approx(1.0));
    CHECK(mat(4, 2) == Catch::Approx(2.0));
    CHECK(mat(5, 3) == Catch::Approx(3.0));
  }

  TEST_CASE(
    "compressed_column_matrix - element_access_absent_in_populated_column",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{
      Shape{6, 5}, {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat(3, 2) == Catch::Approx(0.0));
  }

  TEST_CASE(
    "compressed_column_matrix - element_access_empty_column",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{Shape{6, 5}, {Index{2, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{sp, {1.0, 2.0}};

    CHECK(mat(3, 4) == Catch::Approx(0.0));
  }

  // -- Construction from entries --

  TEST_CASE(
    "compressed_column_matrix - construction_from_entry_initializer_list",
    "[compressed_column_matrix]") {
    Compressed_column_matrix<double> mat{
      Shape{6, 5},
      {{Index{2, 2}, 1.0}, {Index{4, 2}, 2.0}, {Index{5, 3}, 3.0}}};

    CHECK(mat.shape() == Shape(6, 5));
    CHECK(mat.size() == 3);

    CHECK(mat(2, 2) == Catch::Approx(1.0));
    CHECK(mat(4, 2) == Catch::Approx(2.0));
    CHECK(mat(5, 3) == Catch::Approx(3.0));
  }

  // -- Construction from sparsity + function --

  TEST_CASE(
    "compressed_column_matrix - construction_from_sparsity_and_function",
    "[compressed_column_matrix]") {
    Compressed_column_sparsity sp{
      Shape{6, 5}, {Index{2, 2}, Index{4, 2}, Index{5, 3}}};

    Compressed_column_matrix<double> mat{
      sp,
      [](auto row, auto col) { return static_cast<double>(row * 10 + col); }};

    CHECK(mat(2, 2) == Catch::Approx(22.0));
    CHECK(mat(4, 2) == Catch::Approx(42.0));
    CHECK(mat(5, 3) == Catch::Approx(53.0));
  }

  // -- CSR matrix <-> CSC matrix conversion --

  TEST_CASE("conversions - csr_matrix_to_csc_matrix_basic", "[conversions]") {
    Compressed_row_matrix<double> csr{
      Shape{5, 6},
      {{Index{2, 2}, 1.0}, {Index{2, 4}, 2.0}, {Index{3, 5}, 3.0}}};

    auto csc = sparkit::data::detail::to_compressed_column(csr);

    CHECK(csc.shape() == Shape(5, 6));
    CHECK(csc.size() == 3);
    CHECK(csc(2, 2) == Catch::Approx(1.0));
    CHECK(csc(2, 4) == Catch::Approx(2.0));
    CHECK(csc(3, 5) == Catch::Approx(3.0));
  }

  TEST_CASE("conversions - csc_matrix_to_csr_matrix_basic", "[conversions]") {
    Compressed_column_matrix<double> csc{
      Shape{5, 6},
      {{Index{2, 2}, 1.0}, {Index{2, 4}, 2.0}, {Index{3, 5}, 3.0}}};

    auto csr = sparkit::data::detail::to_compressed_row(csc);

    CHECK(csr.shape() == Shape(5, 6));
    CHECK(csr.size() == 3);
    CHECK(csr(2, 2) == Catch::Approx(1.0));
    CHECK(csr(2, 4) == Catch::Approx(2.0));
    CHECK(csr(3, 5) == Catch::Approx(3.0));
  }

  TEST_CASE("conversions - csr_matrix_csc_matrix_roundtrip", "[conversions]") {
    Compressed_row_matrix<double> original{
      Shape{5, 6},
      {{Index{0, 1}, 10.0},
       {Index{1, 0}, 20.0},
       {Index{1, 3}, 30.0},
       {Index{2, 2}, 40.0},
       {Index{3, 5}, 50.0}}};

    auto csc = sparkit::data::detail::to_compressed_column(original);
    auto roundtrip = sparkit::data::detail::to_compressed_row(csc);

    CHECK(roundtrip.shape() == original.shape());
    CHECK(roundtrip.size() == original.size());
    CHECK(roundtrip(0, 1) == Catch::Approx(10.0));
    CHECK(roundtrip(1, 0) == Catch::Approx(20.0));
    CHECK(roundtrip(1, 3) == Catch::Approx(30.0));
    CHECK(roundtrip(2, 2) == Catch::Approx(40.0));
    CHECK(roundtrip(3, 5) == Catch::Approx(50.0));
  }

} // end of namespace sparkit::testing
