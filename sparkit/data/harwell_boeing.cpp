//
// ... Standard header files
//
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

//
// ... sparkit header files
//
#include <sparkit/data/harwell_boeing.hpp>

namespace sparkit::data::detail {

  namespace {

    std::string
    trim(std::string const& s)
    {
      auto start = s.find_first_not_of(' ');
      if (start == std::string::npos) return "";
      auto end = s.find_last_not_of(' ');
      return s.substr(start, end - start + 1);
    }

  } // end of anonymous namespace

  Hb_header
  parse_hb_header(std::istream& is)
  {
    Hb_header header;
    std::string line;

    // Line 1: Title (A72) + Key (A8)
    if (!std::getline(is, line)) {
      throw std::runtime_error(
        "harwell boeing: unexpected end of input reading header line 1");
    }

    if (line.size() >= 72) {
      header.title = trim(line.substr(0, 72));
      header.key = trim(line.substr(72));
    } else {
      header.title = trim(line);
      header.key = "";
    }

    // Line 2: TOTCRD, PTRCRD, INDCRD, VALCRD, RHSCRD (5I14)
    if (!std::getline(is, line)) {
      throw std::runtime_error(
        "harwell boeing: unexpected end of input reading header line 2");
    }
    {
      std::istringstream iss{line};
      iss >> header.totcrd >> header.ptrcrd >> header.indcrd
          >> header.valcrd >> header.rhscrd;
    }

    // Line 3: MXTYPE (A3) + NROW, NCOL, NNZERO, NELTVL (4I14)
    if (!std::getline(is, line)) {
      throw std::runtime_error(
        "harwell boeing: unexpected end of input reading header line 3");
    }
    {
      // MXTYPE is first 3 characters
      std::string mxtype = line.substr(0, 3);
      header.value_type = mxtype[0];   // R/C/P
      header.structure  = mxtype[1];   // U/S/H/Z/R
      header.assembly   = mxtype[2];   // A/E

      std::istringstream iss{line.substr(3)};
      iss >> header.nrow >> header.ncol >> header.nnzero >> header.neltvl;
    }

    // Line 4: Format specs (2A16, 2A20)
    if (!std::getline(is, line)) {
      throw std::runtime_error(
        "harwell boeing: unexpected end of input reading header line 4");
    }
    {
      // Pad line to at least 72 characters
      while (line.size() < 72) {
        line += ' ';
      }
      header.ptrfmt = trim(line.substr(0, 16));
      header.indfmt = trim(line.substr(16, 16));
      header.valfmt = trim(line.substr(32, 20));
      header.rhsfmt = trim(line.substr(52, 20));
    }

    // Line 5 (optional): RHS info — skip if present
    if (header.rhscrd > 0) {
      std::getline(is, line);
    }

    return header;
  }

  void
  write_hb_header(std::ostream& os, Hb_header const& header)
  {
    // Line 1: Title (A72) + Key (A8)
    os << std::left << std::setw(72) << header.title
       << std::setw(8) << header.key << '\n';

    // Line 2: TOTCRD, PTRCRD, INDCRD, VALCRD, RHSCRD (5I14)
    os << std::right
       << std::setw(14) << header.totcrd
       << std::setw(14) << header.ptrcrd
       << std::setw(14) << header.indcrd
       << std::setw(14) << header.valcrd
       << std::setw(14) << header.rhscrd << '\n';

    // Line 3: MXTYPE (A3) + NROW, NCOL, NNZERO, NELTVL (4I14)
    os << header.value_type << header.structure << header.assembly
       << std::setw(14) << header.nrow
       << std::setw(14) << header.ncol
       << std::setw(14) << header.nnzero
       << std::setw(14) << header.neltvl << '\n';

    // Line 4: Format specs (2A16, 2A20)
    os << std::left
       << std::setw(16) << header.ptrfmt
       << std::setw(16) << header.indfmt
       << std::setw(20) << header.valfmt
       << std::setw(20) << header.rhsfmt << '\n';
  }

  Fortran_format
  parse_fortran_format(std::string const& fmt)
  {
    // Expected forms: (nIw), (nFw.d), (nEw.d), (nDw.d), (nGw.d)
    // Optional 1P prefix: (1PnDw.d)

    auto open = fmt.find('(');
    auto close = fmt.find(')');
    if (open == std::string::npos || close == std::string::npos) {
      throw std::runtime_error(
        "harwell boeing: invalid Fortran format: " + fmt);
    }

    std::string inner = fmt.substr(open + 1, close - open - 1);

    // Skip optional scale factor (e.g., "1P")
    std::size_t pos = 0;
    while (pos < inner.size() && std::isdigit(static_cast<unsigned char>(inner[pos]))) {
      ++pos;
    }
    if (pos < inner.size() && (inner[pos] == 'P' || inner[pos] == 'p')) {
      inner = inner.substr(pos + 1);
      pos = 0;
    } else {
      pos = 0;
    }

    // Parse repeat count
    std::size_t repeat_end = pos;
    while (repeat_end < inner.size() &&
           std::isdigit(static_cast<unsigned char>(inner[repeat_end]))) {
      ++repeat_end;
    }

    if (repeat_end == pos || repeat_end >= inner.size()) {
      throw std::runtime_error(
        "harwell boeing: invalid Fortran format: " + fmt);
    }

    config::size_type repeat =
      std::stol(inner.substr(pos, repeat_end - pos));

    // Parse type character
    char type = static_cast<char>(
      std::toupper(static_cast<unsigned char>(inner[repeat_end])));
    if (type != 'I' && type != 'F' && type != 'E' &&
        type != 'D' && type != 'G') {
      throw std::runtime_error(
        "harwell boeing: invalid Fortran format type: " + fmt);
    }

    // Parse width (and optional .decimals)
    std::string rest = inner.substr(repeat_end + 1);
    config::size_type width = 0;
    config::size_type decimals = 0;

    auto dot = rest.find('.');
    if (dot != std::string::npos) {
      width = std::stol(rest.substr(0, dot));
      decimals = std::stol(rest.substr(dot + 1));
    } else {
      width = std::stol(rest);
    }

    return Fortran_format{repeat, type, width, decimals};
  }

  std::vector<config::size_type>
  read_fortran_integers(
    std::istream& is,
    Fortran_format const& fmt,
    config::size_type count)
  {
    using size_type = config::size_type;

    std::vector<size_type> result;
    result.reserve(static_cast<std::size_t>(count));

    std::string line;
    size_type fields_per_line = fmt.repeat;
    size_type field_width = fmt.width;
    size_type read_so_far = 0;

    while (read_so_far < count) {
      if (!std::getline(is, line)) {
        throw std::runtime_error(
          "harwell boeing: unexpected end of input reading integers");
      }

      size_type fields_on_line =
        std::min(fields_per_line, count - read_so_far);

      for (size_type i = 0; i < fields_on_line; ++i) {
        auto start = static_cast<std::size_t>(i * field_width);
        if (start >= line.size()) break;

        auto len = std::min(
          static_cast<std::size_t>(field_width),
          line.size() - start);
        std::string field = line.substr(start, len);

        result.push_back(static_cast<size_type>(std::stol(field)));
        ++read_so_far;
      }
    }

    return result;
  }

} // end of namespace sparkit::data::detail
