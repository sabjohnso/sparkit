#pragma once

//
// ... Standard header files
//
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  struct Matrix_market_banner {
    enum class Format { coordinate, array };
    enum class Field { real, integer, complex, pattern };
    enum class Symmetry { general, symmetric, skew_symmetric, hermitian };

    Format format;
    Field field;
    Symmetry symmetry;
  };

  Matrix_market_banner
  parse_banner(std::string const& line);

  std::string
  format_banner(Matrix_market_banner const& banner);

  // -- Read --

  template <typename T>
  Coordinate_matrix<T>
  read_matrix_market(std::istream& is) {
    using size_type = config::size_type;

    std::string line;
    if (!std::getline(is, line)) {
      throw std::runtime_error("matrix market: unexpected end of input");
    }

    auto banner = parse_banner(line);

    if (banner.format != Matrix_market_banner::Format::coordinate) {
      throw std::runtime_error(
        "matrix market: only coordinate format is supported");
    }

    if (banner.field == Matrix_market_banner::Field::complex) {
      throw std::runtime_error(
        "matrix market: complex field is not yet supported");
    }

    while (std::getline(is, line)) {
      if (!line.empty() && line[0] != '%') { break; }
    }

    size_type rows{};
    size_type cols{};
    size_type nnz{};
    {
      std::istringstream size_line{line};
      size_line >> rows >> cols >> nnz;
    }

    bool const is_pattern =
      (banner.field == Matrix_market_banner::Field::pattern);
    bool const is_symmetric =
      (banner.symmetry == Matrix_market_banner::Symmetry::symmetric);

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(is_symmetric ? nnz * 2 : nnz));

    for (size_type k = 0; k < nnz; ++k) {
      std::getline(is, line);
      std::istringstream entry_line{line};

      size_type row{};
      size_type col{};
      entry_line >> row >> col;

      row -= 1;
      col -= 1;

      T value{};
      if (is_pattern) {
        value = T{1};
      } else {
        entry_line >> value;
      }

      entries.push_back(Entry<T>{Index{row, col}, value});

      if (is_symmetric && row != col) {
        entries.push_back(Entry<T>{Index{col, row}, value});
      }
    }

    return Coordinate_matrix<T>{
      Shape{rows, cols}, entries.begin(), entries.end()};
  }

  template <typename T>
  Coordinate_matrix<T>
  read_matrix_market(std::filesystem::path const& path) {
    std::ifstream file{path};
    if (!file) {
      throw std::runtime_error(
        "matrix market: cannot open file: " + path.string());
    }
    return read_matrix_market<T>(file);
  }

  // -- Write --

  template <typename T>
  void
  write_matrix_market(std::ostream& os, Compressed_row_matrix<T> const& A) {
    using size_type = config::size_type;

    os << format_banner(
            Matrix_market_banner{
              Matrix_market_banner::Format::coordinate,
              Matrix_market_banner::Field::real,
              Matrix_market_banner::Symmetry::general})
       << '\n';

    os << A.shape().row() << ' ' << A.shape().column() << ' ' << A.size()
       << '\n';

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    for (size_type row = 0; row < A.shape().row(); ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        os << (row + 1) << ' ' << (ci[j] + 1) << ' ' << vals[j] << '\n';
      }
    }
  }

  template <typename T>
  void
  write_matrix_market(
    std::filesystem::path const& path, Compressed_row_matrix<T> const& A) {
    std::ofstream file{path};
    if (!file) {
      throw std::runtime_error(
        "matrix market: cannot open file: " + path.string());
    }
    write_matrix_market(file, A);
  }

} // end of namespace sparkit::data::detail
