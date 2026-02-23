#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/conversions.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  struct Hb_header {
    std::string title;
    std::string key;
    config::size_type totcrd{};
    config::size_type ptrcrd{};
    config::size_type indcrd{};
    config::size_type valcrd{};
    config::size_type rhscrd{};
    char value_type{}; // R/C/P
    char structure{};  // U/S/H/Z/R
    char assembly{};   // A/E
    config::size_type nrow{};
    config::size_type ncol{};
    config::size_type nnzero{};
    config::size_type neltvl{};
    std::string ptrfmt;
    std::string indfmt;
    std::string valfmt;
    std::string rhsfmt;
  };

  struct Fortran_format {
    config::size_type repeat{};
    char type{}; // I/F/E/D/G
    config::size_type width{};
    config::size_type decimals{}; // 0 for integer
  };

  Hb_header
  parse_hb_header(std::istream& is);

  void
  write_hb_header(std::ostream& os, Hb_header const& header);

  Fortran_format
  parse_fortran_format(std::string const& fmt);

  std::vector<config::size_type>
  read_fortran_integers(
    std::istream& is, Fortran_format const& fmt, config::size_type count);

  // -- Template functions --

  template <typename T>
  std::vector<T>
  read_fortran_reals(
    std::istream& is, Fortran_format const& fmt, config::size_type count) {
    using size_type = config::size_type;

    std::vector<T> result;
    result.reserve(static_cast<std::size_t>(count));

    std::string line;
    size_type fields_per_line = fmt.repeat;
    size_type field_width = fmt.width;
    size_type read_so_far = 0;

    while (read_so_far < count) {
      if (!std::getline(is, line)) {
        throw std::runtime_error(
          "harwell boeing: unexpected end of input reading reals");
      }

      size_type fields_on_line = std::min(fields_per_line, count - read_so_far);

      for (size_type i = 0; i < fields_on_line; ++i) {
        auto start = static_cast<std::size_t>(i * field_width);
        if (start >= line.size()) break;

        auto len =
          std::min(static_cast<std::size_t>(field_width), line.size() - start);
        std::string field = line.substr(start, len);

        // Replace D/d exponent with E for std::stod
        for (auto& ch : field) {
          if (ch == 'D' || ch == 'd') { ch = 'E'; }
        }

        result.push_back(static_cast<T>(std::stod(field)));
        ++read_so_far;
      }
    }

    return result;
  }

  template <typename T>
  Compressed_column_matrix<T>
  read_harwell_boeing(std::istream& is) {
    using size_type = config::size_type;

    auto header = parse_hb_header(is);

    if (header.value_type == 'C') {
      throw std::runtime_error(
        "harwell boeing: complex value type is not yet supported");
    }

    if (header.assembly == 'E') {
      throw std::runtime_error(
        "harwell boeing: elemental assembly is not yet supported");
    }

    bool const is_pattern = (header.value_type == 'P');
    bool const is_symmetric = (header.structure == 'S');

    // Parse format descriptors
    auto ptr_fmt = parse_fortran_format(header.ptrfmt);
    auto ind_fmt = parse_fortran_format(header.indfmt);

    // Read column pointers (ncol + 1 values, 1-based)
    auto col_ptr = read_fortran_integers(is, ptr_fmt, header.ncol + 1);

    // Convert to 0-based
    for (auto& v : col_ptr) {
      v -= 1;
    }

    // Read row indices (nnzero values, 1-based)
    auto row_ind = read_fortran_integers(is, ind_fmt, header.nnzero);

    // Convert to 0-based
    for (auto& v : row_ind) {
      v -= 1;
    }

    // Read values
    std::vector<T> values;
    if (is_pattern) {
      values.assign(static_cast<std::size_t>(header.nnzero), T{1});
    } else {
      auto val_fmt = parse_fortran_format(header.valfmt);
      values = read_fortran_reals<T>(is, val_fmt, header.nnzero);
    }

    if (is_symmetric) {
      // Expand lower triangle to full matrix
      std::vector<Entry<T>> entries;
      entries.reserve(static_cast<std::size_t>(header.nnzero * 2));

      for (size_type col = 0; col < header.ncol; ++col) {
        for (auto j = col_ptr[static_cast<std::size_t>(col)];
             j < col_ptr[static_cast<std::size_t>(col + 1)];
             ++j) {
          auto row = row_ind[static_cast<std::size_t>(j)];
          auto val = values[static_cast<std::size_t>(j)];
          entries.push_back(Entry<T>{Index{row, col}, val});
          if (row != col) { entries.push_back(Entry<T>{Index{col, row}, val}); }
        }
      }

      // Sort by (column, row) for CSC construction
      auto by_col_row = [](auto const& a, auto const& b) {
        return a.index.column() < b.index.column() ||
               (a.index.column() == b.index.column() &&
                a.index.row() < b.index.row());
      };
      std::sort(entries.begin(), entries.end(), by_col_row);

      std::vector<Index> indices;
      std::vector<T> expanded_values;
      indices.reserve(entries.size());
      expanded_values.reserve(entries.size());

      for (auto const& e : entries) {
        indices.push_back(e.index);
        expanded_values.push_back(e.value);
      }

      Compressed_column_sparsity sparsity{
        Shape{header.nrow, header.ncol}, indices.begin(), indices.end()};
      return Compressed_column_matrix<T>{
        std::move(sparsity), std::move(expanded_values)};
    }

    // Build CSC directly from raw arrays
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(header.nnzero));

    for (size_type col = 0; col < header.ncol; ++col) {
      for (auto j = col_ptr[static_cast<std::size_t>(col)];
           j < col_ptr[static_cast<std::size_t>(col + 1)];
           ++j) {
        indices.push_back(Index{row_ind[static_cast<std::size_t>(j)], col});
      }
    }

    Compressed_column_sparsity sparsity{
      Shape{header.nrow, header.ncol}, indices.begin(), indices.end()};
    return Compressed_column_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template <typename T>
  Compressed_column_matrix<T>
  read_harwell_boeing(std::filesystem::path const& path) {
    std::ifstream file{path};
    if (!file) {
      throw std::runtime_error(
        "harwell boeing: cannot open file: " + path.string());
    }
    return read_harwell_boeing<T>(file);
  }

  template <typename T>
  void
  write_harwell_boeing(std::ostream& os, Compressed_row_matrix<T> const& A) {
    using size_type = config::size_type;

    // Convert CSR -> CSC
    auto csc = to_compressed_column(A);

    auto cp = csc.col_ptr();
    auto ri = csc.row_ind();
    auto sv = csc.values();

    size_type ncol = csc.shape().column();
    size_type nnz = csc.size();

    // Compute line counts for our fixed output formats:
    // Pointers: (8I10) -> 8 per line
    // Indices:  (8I10) -> 8 per line
    // Values:   (3E26.18) -> 3 per line
    size_type ptrcrd = (ncol + 1 + 7) / 8;
    size_type indcrd = (nnz + 7) / 8;
    size_type valcrd = (nnz + 2) / 3;
    size_type totcrd = ptrcrd + indcrd + valcrd;

    Hb_header header;
    header.title = "sparkit";
    header.key = "SPARKIT";
    header.totcrd = totcrd;
    header.ptrcrd = ptrcrd;
    header.indcrd = indcrd;
    header.valcrd = valcrd;
    header.rhscrd = 0;
    header.value_type = 'R';
    header.structure = 'U';
    header.assembly = 'A';
    header.nrow = csc.shape().row();
    header.ncol = ncol;
    header.nnzero = nnz;
    header.neltvl = 0;
    header.ptrfmt = "(8I10)";
    header.indfmt = "(8I10)";
    header.valfmt = "(3E26.18)";
    header.rhsfmt = "";

    write_hb_header(os, header);

    // Write column pointers (1-based, 8I10)
    for (size_type i = 0; i <= ncol; ++i) {
      os << std::setw(10) << (cp[i] + 1);
      if ((i + 1) % 8 == 0 || i == ncol) { os << '\n'; }
    }

    // Write row indices (1-based, 8I10)
    for (size_type i = 0; i < nnz; ++i) {
      os << std::setw(10) << (ri[i] + 1);
      if ((i + 1) % 8 == 0 || i == nnz - 1) { os << '\n'; }
    }

    // Write values (3E26.18)
    os << std::scientific << std::setprecision(18);
    for (size_type i = 0; i < nnz; ++i) {
      os << std::setw(26) << sv[i];
      if ((i + 1) % 3 == 0 || i == nnz - 1) { os << '\n'; }
    }
  }

  template <typename T>
  void
  write_harwell_boeing(
    std::filesystem::path const& path, Compressed_row_matrix<T> const& A) {
    std::ofstream file{path};
    if (!file) {
      throw std::runtime_error(
        "harwell boeing: cannot open file: " + path.string());
    }
    write_harwell_boeing(file, A);
  }

} // end of namespace sparkit::data::detail
