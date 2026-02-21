//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <sstream>

//
// ... sparkit header files
//
#include <sparkit/data/matrix_market.hpp>
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Matrix_market_banner;
  using sparkit::data::detail::parse_banner;
  using sparkit::data::detail::read_matrix_market;
  using sparkit::data::detail::write_matrix_market;

  // -- Banner parsing --

  TEST_CASE("matrix_market - parse_banner", "[matrix_market]")
  {
    auto banner = parse_banner(
      "%%MatrixMarket matrix coordinate real general");

    CHECK(banner.format == Matrix_market_banner::Format::coordinate);
    CHECK(banner.field == Matrix_market_banner::Field::real);
    CHECK(banner.symmetry == Matrix_market_banner::Symmetry::general);
  }

  TEST_CASE("matrix_market - parse_banner_case_insensitive", "[matrix_market]")
  {
    auto banner = parse_banner(
      "%%MatrixMarket MATRIX COORDINATE INTEGER SYMMETRIC");

    CHECK(banner.format == Matrix_market_banner::Format::coordinate);
    CHECK(banner.field == Matrix_market_banner::Field::integer);
    CHECK(banner.symmetry == Matrix_market_banner::Symmetry::symmetric);
  }

  TEST_CASE("matrix_market - parse_banner_invalid", "[matrix_market]")
  {
    CHECK_THROWS_AS(
      parse_banner("not a valid banner"),
      std::runtime_error);

    CHECK_THROWS_AS(
      parse_banner("%%MatrixMarket matrix coordinate badfield general"),
      std::runtime_error);

    CHECK_THROWS_AS(
      parse_banner("%%MatrixMarket matrix coordinate real badsymmetry"),
      std::runtime_error);
  }

  // -- Reading --

  TEST_CASE("matrix_market - read_general_real", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix coordinate real general\n"
      "3 4 3\n"
      "1 2 4.5\n"
      "2 3 7.0\n"
      "3 1 2.5\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat.shape() == Shape(3, 4));
    CHECK(mat.size() == 3);
    CHECK(mat(0, 1) == Catch::Approx(4.5));
    CHECK(mat(1, 2) == Catch::Approx(7.0));
    CHECK(mat(2, 0) == Catch::Approx(2.5));
  }

  TEST_CASE("matrix_market - read_with_comments", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix coordinate real general\n"
      "% This is a comment\n"
      "% Another comment\n"
      "3 3 2\n"
      "1 1 5.0\n"
      "3 3 9.0\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat.shape() == Shape(3, 3));
    CHECK(mat.size() == 2);
    CHECK(mat(0, 0) == Catch::Approx(5.0));
    CHECK(mat(2, 2) == Catch::Approx(9.0));
  }

  TEST_CASE("matrix_market - read_symmetric", "[matrix_market]")
  {
    // Symmetric stores lower triangle only (i >= j).
    // Off-diagonal entries should be mirrored.
    std::istringstream input{
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "3 3 4\n"
      "1 1 1.0\n"
      "2 1 2.0\n"
      "3 1 3.0\n"
      "3 3 4.0\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat.shape() == Shape(3, 3));
    // 4 stored + 2 mirrored off-diag = 6
    CHECK(mat.size() == 6);

    // Diagonal entries
    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(2, 2) == Catch::Approx(4.0));

    // Off-diagonal: original and mirrored
    CHECK(mat(1, 0) == Catch::Approx(2.0));
    CHECK(mat(0, 1) == Catch::Approx(2.0));
    CHECK(mat(2, 0) == Catch::Approx(3.0));
    CHECK(mat(0, 2) == Catch::Approx(3.0));
  }

  TEST_CASE("matrix_market - read_integer", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix coordinate integer general\n"
      "3 3 2\n"
      "1 2 7\n"
      "3 1 3\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat.size() == 2);
    CHECK(mat(0, 1) == Catch::Approx(7.0));
    CHECK(mat(2, 0) == Catch::Approx(3.0));
  }

  TEST_CASE("matrix_market - read_pattern", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix coordinate pattern general\n"
      "3 3 3\n"
      "1 1\n"
      "2 2\n"
      "3 3\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat.size() == 3);
    CHECK(mat(0, 0) == Catch::Approx(1.0));
    CHECK(mat(1, 1) == Catch::Approx(1.0));
    CHECK(mat(2, 2) == Catch::Approx(1.0));
    CHECK(mat(0, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("matrix_market - read_1based_indexing", "[matrix_market]")
  {
    // Entry at (1,1) in file should become (0,0) internally
    std::istringstream input{
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "1 1 42.0\n"
    };

    auto mat = read_matrix_market<double>(input);

    CHECK(mat(0, 0) == Catch::Approx(42.0));
    CHECK(mat(1, 1) == Catch::Approx(0.0));
  }

  // -- Writing --

  TEST_CASE("matrix_market - write_known_matrix", "[matrix_market]")
  {
    Compressed_row_matrix<double> mat{Shape{3, 4}, {
      {Index{0, 1}, 4.5},
      {Index{1, 2}, 7.0},
      {Index{2, 0}, 2.5}
    }};

    std::ostringstream output;
    write_matrix_market(output, mat);

    std::string expected =
      "%%MatrixMarket matrix coordinate real general\n"
      "3 4 3\n"
      "1 2 4.5\n"
      "2 3 7\n"
      "3 1 2.5\n";

    CHECK(output.str() == expected);
  }

  TEST_CASE("matrix_market - round_trip", "[matrix_market]")
  {
    // Build a CSR matrix, write it, read it back as COO, convert to CSR
    Compressed_row_matrix<double> original{Shape{4, 5}, {
      {Index{0, 0}, 1.0},
      {Index{0, 3}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 4}, 4.0},
      {Index{3, 2}, 5.0}
    }};

    std::ostringstream out;
    write_matrix_market(out, original);

    std::istringstream in{out.str()};
    auto coo = read_matrix_market<double>(in);
    auto result = sparkit::data::detail::to_compressed_row(coo);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    for (config::size_type r = 0; r < 4; ++r) {
      for (config::size_type c = 0; c < 5; ++c) {
        CHECK(result(r, c) == Catch::Approx(original(r, c)));
      }
    }
  }

  TEST_CASE("matrix_market - write_empty_matrix", "[matrix_market]")
  {
    Compressed_row_sparsity sp{Shape{3, 3}, {}};
    Compressed_row_matrix<double> mat{sp, {}};

    std::ostringstream output;
    write_matrix_market(output, mat);

    std::string expected =
      "%%MatrixMarket matrix coordinate real general\n"
      "3 3 0\n";

    CHECK(output.str() == expected);
  }

  // -- Unsupported formats --

  TEST_CASE("matrix_market - read_unsupported_format", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix array real general\n"
      "3 3\n"
    };

    CHECK_THROWS_AS(
      read_matrix_market<double>(input),
      std::runtime_error);
  }

  TEST_CASE("matrix_market - read_unsupported_field", "[matrix_market]")
  {
    std::istringstream input{
      "%%MatrixMarket matrix coordinate complex general\n"
      "3 3 1\n"
      "1 1 1.0 2.0\n"
    };

    CHECK_THROWS_AS(
      read_matrix_market<double>(input),
      std::runtime_error);
  }

} // end of namespace sparkit::testing
