//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <sstream>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/harwell_boeing.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_column_matrix;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Fortran_format;
  using sparkit::data::detail::Hb_header;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::parse_fortran_format;
  using sparkit::data::detail::parse_hb_header;
  using sparkit::data::detail::read_fortran_integers;
  using sparkit::data::detail::read_fortran_reals;
  using sparkit::data::detail::read_harwell_boeing;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::write_harwell_boeing;
  using sparkit::data::detail::write_hb_header;

  // -- Fortran format parsing --

  TEST_CASE("harwell_boeing - parse_integer_format", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(3I14)");

    CHECK(fmt.repeat == 3);
    CHECK(fmt.type == 'I');
    CHECK(fmt.width == 14);
    CHECK(fmt.decimals == 0);
  }

  TEST_CASE("harwell_boeing - parse_real_format", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(5E26.18)");

    CHECK(fmt.repeat == 5);
    CHECK(fmt.type == 'E');
    CHECK(fmt.width == 26);
    CHECK(fmt.decimals == 18);
  }

  TEST_CASE("harwell_boeing - parse_double_format", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(2D26.18)");

    CHECK(fmt.repeat == 2);
    CHECK(fmt.type == 'D');
    CHECK(fmt.width == 26);
    CHECK(fmt.decimals == 18);
  }

  TEST_CASE("harwell_boeing - parse_float_format", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(4F16.8)");

    CHECK(fmt.repeat == 4);
    CHECK(fmt.type == 'F');
    CHECK(fmt.width == 16);
    CHECK(fmt.decimals == 8);
  }

  TEST_CASE("harwell_boeing - parse_general_format", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(1G25.17)");

    CHECK(fmt.repeat == 1);
    CHECK(fmt.type == 'G');
    CHECK(fmt.width == 25);
    CHECK(fmt.decimals == 17);
  }

  TEST_CASE("harwell_boeing - parse_with_scale_factor", "[harwell_boeing]") {
    auto fmt = parse_fortran_format("(1P2D26.18)");

    CHECK(fmt.repeat == 2);
    CHECK(fmt.type == 'D');
    CHECK(fmt.width == 26);
    CHECK(fmt.decimals == 18);
  }

  TEST_CASE("harwell_boeing - parse_invalid_format", "[harwell_boeing]") {
    CHECK_THROWS_AS(parse_fortran_format("not a format"), std::runtime_error);
  }

  // -- Reading fixed-width fields --

  TEST_CASE("harwell_boeing - read_fortran_integers", "[harwell_boeing]") {
    // 3 integers in (3I5) format: 5-char wide fields
    std::istringstream input{"    1    2    3"};
    auto fmt = parse_fortran_format("(3I5)");
    auto values = read_fortran_integers(input, fmt, 3);

    REQUIRE(values.size() == 3);
    CHECK(values[0] == 1);
    CHECK(values[1] == 2);
    CHECK(values[2] == 3);
  }

  TEST_CASE("harwell_boeing - read_fortran_reals", "[harwell_boeing]") {
    // 2 reals in (2E15.8) format
    std::istringstream input{" 1.50000000E+00 2.70000000E+00"};
    auto fmt = parse_fortran_format("(2E15.8)");
    auto values = read_fortran_reals<double>(input, fmt, 2);

    REQUIRE(values.size() == 2);
    CHECK(values[0] == Catch::Approx(1.5));
    CHECK(values[1] == Catch::Approx(2.7));
  }

  TEST_CASE(
    "harwell_boeing - read_fortran_reals_d_exponent", "[harwell_boeing]") {
    // D-format exponent: 1.5D+02 should parse as 150.0
    std::istringstream input{"  1.50000000D+02  3.00000000D+00"};
    auto fmt = parse_fortran_format("(2E16.8)");
    auto values = read_fortran_reals<double>(input, fmt, 2);

    REQUIRE(values.size() == 2);
    CHECK(values[0] == Catch::Approx(150.0));
    CHECK(values[1] == Catch::Approx(3.0));
  }

  // -- Header parsing --

  TEST_CASE("harwell_boeing - parse_header", "[harwell_boeing]") {
    std::istringstream input{
      "Test Matrix                                                           "
      "  TESTKEY \n"
      "             5             2             1             2             "
      "0\n"
      "RUA                    3             4             5             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"};

    auto header = parse_hb_header(input);

    CHECK(header.title == "Test Matrix");
    CHECK(header.key == "TESTKEY");
    CHECK(header.totcrd == 5);
    CHECK(header.ptrcrd == 2);
    CHECK(header.indcrd == 1);
    CHECK(header.valcrd == 2);
    CHECK(header.rhscrd == 0);
    CHECK(header.value_type == 'R');
    CHECK(header.structure == 'U');
    CHECK(header.assembly == 'A');
    CHECK(header.nrow == 3);
    CHECK(header.ncol == 4);
    CHECK(header.nnzero == 5);
    CHECK(header.neltvl == 0);
  }

  TEST_CASE("harwell_boeing - parse_header_with_rhs_line", "[harwell_boeing]") {
    std::istringstream input{
      "Test Matrix with RHS                                                  "
      "  RHSKEY  \n"
      "             6             2             1             2             "
      "1\n"
      "RUA                    3             4             5             0\n"
      "(8I10)          (8I10)          (3E26.18)           (3E26.18)         "
      "\n"
      "F                    1             0\n"};

    auto header = parse_hb_header(input);

    CHECK(header.title == "Test Matrix with RHS");
    CHECK(header.key == "RHSKEY");
    CHECK(header.rhscrd == 1);
    CHECK(header.nrow == 3);
    CHECK(header.ncol == 4);
  }

  // -- Reading --

  TEST_CASE("harwell_boeing - read_unsymmetric_real", "[harwell_boeing]") {
    // A 3x3 matrix with 4 nonzeros, stored in CSC (native HB format)
    // Column pointers: 1 3 4 5 (1-based)
    // Row indices:     1 3 2 1 (1-based)
    // Values:          1.0 3.0 2.0 4.0
    //
    // This represents:
    //   col 0: rows 0,2 -> values 1.0, 3.0
    //   col 1: row 1    -> value 2.0
    //   col 2: row 0    -> value 4.0
    std::istringstream input{
      "Small unsymmetric matrix                                              "
      "  SMALLUNS\n"
      "             4             1             1             1             "
      "0\n"
      "RUA                    3             3             4             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"
      "         1         3         4         5\n"
      "         1         3         2         1\n"
      "  1.000000000000000000E+00  3.000000000000000000E+00  "
      "2.000000000000000000E+00\n"
      "  4.000000000000000000E+00\n"};

    auto csc = read_harwell_boeing<double>(input);

    CHECK(csc.shape() == Shape(3, 3));
    CHECK(csc.size() == 4);
    CHECK(csc(0, 0) == Catch::Approx(1.0));
    CHECK(csc(2, 0) == Catch::Approx(3.0));
    CHECK(csc(1, 1) == Catch::Approx(2.0));
    CHECK(csc(0, 2) == Catch::Approx(4.0));
  }

  TEST_CASE("harwell_boeing - read_pattern_matrix", "[harwell_boeing]") {
    // Pattern matrix: no values section, all values should be 1.0
    // 3x3, 3 nonzeros (diagonal)
    std::istringstream input{
      "Pattern matrix                                                        "
      "  PATTERN \n"
      "             2             1             1             0             "
      "0\n"
      "PUA                    3             3             3             0\n"
      "(8I10)          (8I10)                                            \n"
      "         1         2         3         4\n"
      "         1         2         3\n"};

    auto csc = read_harwell_boeing<double>(input);

    CHECK(csc.shape() == Shape(3, 3));
    CHECK(csc.size() == 3);
    CHECK(csc(0, 0) == Catch::Approx(1.0));
    CHECK(csc(1, 1) == Catch::Approx(1.0));
    CHECK(csc(2, 2) == Catch::Approx(1.0));
    CHECK(csc(0, 1) == Catch::Approx(0.0));
  }

  TEST_CASE("harwell_boeing - read_symmetric_matrix", "[harwell_boeing]") {
    // Symmetric 3x3, lower triangle stored in CSC
    // Lower triangle entries: (0,0)=1.0, (1,0)=2.0, (1,1)=3.0, (2,2)=4.0
    // After expansion: also (0,1)=2.0
    std::istringstream input{
      "Symmetric test matrix                                                 "
      "  SYMTEST \n"
      "             4             1             1             1             "
      "0\n"
      "RSA                    3             3             4             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"
      "         1         3         4         5\n"
      "         1         2         2         3\n"
      "  1.000000000000000000E+00  2.000000000000000000E+00  "
      "3.000000000000000000E+00\n"
      "  4.000000000000000000E+00\n"};

    auto csc = read_harwell_boeing<double>(input);

    CHECK(csc.shape() == Shape(3, 3));

    // Diagonal entries
    CHECK(csc(0, 0) == Catch::Approx(1.0));
    CHECK(csc(1, 1) == Catch::Approx(3.0));
    CHECK(csc(2, 2) == Catch::Approx(4.0));

    // Off-diagonal: original and mirrored
    CHECK(csc(1, 0) == Catch::Approx(2.0));
    CHECK(csc(0, 1) == Catch::Approx(2.0));
  }

  TEST_CASE("harwell_boeing - read_1based_to_0based", "[harwell_boeing]") {
    // Verify 1-based indices in file become 0-based internally
    // 2x2 identity matrix
    std::istringstream input{
      "Identity 2x2                                                          "
      "  IDENT2  \n"
      "             3             1             1             1             "
      "0\n"
      "RUA                    2             2             2             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"
      "         1         2         3\n"
      "         1         2\n"
      "  1.000000000000000000E+00  1.000000000000000000E+00\n"};

    auto csc = read_harwell_boeing<double>(input);

    CHECK(csc(0, 0) == Catch::Approx(1.0));
    CHECK(csc(1, 1) == Catch::Approx(1.0));
    CHECK(csc(0, 1) == Catch::Approx(0.0));
    CHECK(csc(1, 0) == Catch::Approx(0.0));
  }

  TEST_CASE("harwell_boeing - read_empty_stream", "[harwell_boeing]") {
    std::istringstream input{""};

    CHECK_THROWS_AS(read_harwell_boeing<double>(input), std::runtime_error);
  }

  // -- Writing --

  TEST_CASE("harwell_boeing - write_known_matrix", "[harwell_boeing]") {
    // 3x3 matrix with known entries
    Compressed_row_matrix<double> mat{
      Shape{3, 3},
      {{Index{0, 0}, 1.0},
       {Index{0, 2}, 4.0},
       {Index{1, 1}, 2.0},
       {Index{2, 0}, 3.0}}};

    std::ostringstream output;
    write_harwell_boeing(output, mat);

    std::string result = output.str();

    // Verify header line 3 contains correct dimensions and type
    CHECK(result.find("RUA") != std::string::npos);

    // Verify it can be parsed back (structural check)
    std::istringstream input{result};
    auto header = parse_hb_header(input);
    CHECK(header.nrow == 3);
    CHECK(header.ncol == 3);
    CHECK(header.nnzero == 4);
    CHECK(header.value_type == 'R');
    CHECK(header.structure == 'U');
    CHECK(header.assembly == 'A');
  }

  TEST_CASE(
    "harwell_boeing - write_produces_valid_header", "[harwell_boeing]") {
    Compressed_row_matrix<double> mat{
      Shape{4, 5},
      {{Index{0, 0}, 1.0}, {Index{1, 3}, 2.0}, {Index{3, 4}, 3.0}}};

    std::ostringstream output;
    write_harwell_boeing(output, mat);

    std::istringstream input{output.str()};
    auto header = parse_hb_header(input);

    CHECK(header.nrow == 4);
    CHECK(header.ncol == 5);
    CHECK(header.nnzero == 3);
  }

  // -- Round-trip --

  TEST_CASE("harwell_boeing - round_trip", "[harwell_boeing]") {
    Compressed_row_matrix<double> original{
      Shape{4, 5},
      {{Index{0, 0}, 1.0},
       {Index{0, 3}, 2.0},
       {Index{1, 1}, 3.0},
       {Index{2, 4}, 4.0},
       {Index{3, 2}, 5.0}}};

    // Write CSR as HB
    std::ostringstream out;
    write_harwell_boeing(out, original);

    // Read back as CSC (native HB format)
    std::istringstream in{out.str()};
    auto csc = read_harwell_boeing<double>(in);

    // Convert CSC -> CSR for comparison
    auto result = sparkit::data::detail::to_compressed_row(csc);

    CHECK(result.shape() == original.shape());
    CHECK(result.size() == original.size());

    for (config::size_type r = 0; r < 4; ++r) {
      for (config::size_type c = 0; c < 5; ++c) {
        CHECK(result(r, c) == Catch::Approx(original(r, c)));
      }
    }
  }

  // -- Error handling --

  TEST_CASE("harwell_boeing - read_unsupported_complex", "[harwell_boeing]") {
    std::istringstream input{
      "Complex matrix                                                        "
      "  COMPLEX \n"
      "             4             1             1             1             "
      "0\n"
      "CUA                    3             3             4             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"};

    CHECK_THROWS_AS(read_harwell_boeing<double>(input), std::runtime_error);
  }

  TEST_CASE("harwell_boeing - read_unsupported_elemental", "[harwell_boeing]") {
    std::istringstream input{
      "Elemental matrix                                                      "
      "  ELEMENT \n"
      "             4             1             1             1             "
      "0\n"
      "RUE                    3             3             4             0\n"
      "(8I10)          (8I10)          (3E26.18)                        \n"};

    CHECK_THROWS_AS(read_harwell_boeing<double>(input), std::runtime_error);
  }

} // end of namespace sparkit::testing
