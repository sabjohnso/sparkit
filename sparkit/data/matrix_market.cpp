//
// ... Standard header files
//
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

//
// ... sparkit header files
//
#include <sparkit/data/matrix_market.hpp>

namespace sparkit::data::detail {

  namespace {

    std::string
    to_lower(std::string s)
    {
      std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return s;
    }

  } // end of anonymous namespace

  Matrix_market_banner
  parse_banner(std::string const& line)
  {
    std::istringstream iss{line};

    std::string header;
    std::string object;
    std::string format_str;
    std::string field_str;
    std::string symmetry_str;

    iss >> header >> object >> format_str >> field_str >> symmetry_str;

    if (to_lower(header) != "%%matrixmarket") {
      throw std::runtime_error(
        "matrix market: invalid banner header: " + header);
    }

    format_str = to_lower(format_str);
    field_str = to_lower(field_str);
    symmetry_str = to_lower(symmetry_str);

    Matrix_market_banner banner{};

    if (format_str == "coordinate") {
      banner.format = Matrix_market_banner::Format::coordinate;
    } else if (format_str == "array") {
      banner.format = Matrix_market_banner::Format::array;
    } else {
      throw std::runtime_error(
        "matrix market: unknown format: " + format_str);
    }

    if (field_str == "real") {
      banner.field = Matrix_market_banner::Field::real;
    } else if (field_str == "integer") {
      banner.field = Matrix_market_banner::Field::integer;
    } else if (field_str == "complex") {
      banner.field = Matrix_market_banner::Field::complex;
    } else if (field_str == "pattern") {
      banner.field = Matrix_market_banner::Field::pattern;
    } else {
      throw std::runtime_error(
        "matrix market: unknown field: " + field_str);
    }

    if (symmetry_str == "general") {
      banner.symmetry = Matrix_market_banner::Symmetry::general;
    } else if (symmetry_str == "symmetric") {
      banner.symmetry = Matrix_market_banner::Symmetry::symmetric;
    } else if (symmetry_str == "skew-symmetric") {
      banner.symmetry = Matrix_market_banner::Symmetry::skew_symmetric;
    } else if (symmetry_str == "hermitian") {
      banner.symmetry = Matrix_market_banner::Symmetry::hermitian;
    } else {
      throw std::runtime_error(
        "matrix market: unknown symmetry: " + symmetry_str);
    }

    return banner;
  }

  std::string
  format_banner(Matrix_market_banner const& banner)
  {
    std::string result = "%%MatrixMarket matrix";

    switch (banner.format) {
    case Matrix_market_banner::Format::coordinate:
      result += " coordinate"; break;
    case Matrix_market_banner::Format::array:
      result += " array"; break;
    }

    switch (banner.field) {
    case Matrix_market_banner::Field::real:
      result += " real"; break;
    case Matrix_market_banner::Field::integer:
      result += " integer"; break;
    case Matrix_market_banner::Field::complex:
      result += " complex"; break;
    case Matrix_market_banner::Field::pattern:
      result += " pattern"; break;
    }

    switch (banner.symmetry) {
    case Matrix_market_banner::Symmetry::general:
      result += " general"; break;
    case Matrix_market_banner::Symmetry::symmetric:
      result += " symmetric"; break;
    case Matrix_market_banner::Symmetry::skew_symmetric:
      result += " skew-symmetric"; break;
    case Matrix_market_banner::Symmetry::hermitian:
      result += " hermitian"; break;
    }

    return result;
  }

} // end of namespace sparkit::data::detail
