#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/info.hpp>
#include <sparkit/data/numeric_cholesky.hpp>

namespace sparkit::data::detail {

  // --- Symmetry checks ---

  // Structural symmetry: does (i,j) exist iff (j,i) exists?
  //
  // Collects all (col, row) transpose pairs, sorts them, and compares
  // with the original sorted (row, col) pairs. O(nnz log nnz).
  template <typename T>
  bool
  is_structurally_symmetric(Compressed_row_matrix<T> const& A) {
    auto m = A.shape().row();
    auto n = A.shape().column();
    if (m != n) { return false; }

    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    auto nnz = static_cast<std::size_t>(A.size());

    // Collect (row, col) pairs sorted lexicographically
    using Pair = std::pair<config::size_type, config::size_type>;
    std::vector<Pair> forward;
    std::vector<Pair> transpose;
    forward.reserve(nnz);
    transpose.reserve(nnz);

    for (config::size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        forward.emplace_back(i, j);
        transpose.emplace_back(j, i);
      }
    }

    std::sort(forward.begin(), forward.end());
    std::sort(transpose.begin(), transpose.end());

    return forward == transpose;
  }

  // Numerical symmetry: |A[i,j] - A[j,i]| <= tol * max(|A[i,j]|, |A[j,i]|)
  // for all (i,j). Default tol = 0 means exact equality.
  //
  // For each off-diagonal entry (i, j, v), binary-searches row j of the
  // CSR for column i. Returns false if any entry has no symmetric
  // counterpart or violates the tolerance. O(nnz log(max_row_nnz)).
  template <typename T>
  bool
  is_numerically_symmetric(Compressed_row_matrix<T> const& A, T tol = T{0}) {
    auto m = A.shape().row();
    auto n = A.shape().column();
    if (m != n) { return false; }

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    for (config::size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j == i) { continue; } // diagonal is always "symmetric" with itself

        auto v = vals[static_cast<std::size_t>(p)];

        // Binary-search row j for column i
        auto row_begin = ci.data() + rp[j];
        auto row_end = ci.data() + rp[j + 1];
        auto it = std::lower_bound(row_begin, row_end, i);

        if (it == row_end || *it != i) { return false; }

        auto q = static_cast<std::size_t>(it - ci.data());
        auto w = vals[q];

        if (tol == T{0}) {
          if (v != w) { return false; }
        } else {
          T diff = std::abs(v - w);
          T scale = std::max(std::abs(v), std::abs(w));
          if (diff > tol * scale) { return false; }
        }
      }
    }
    return true;
  }

  // --- Diagonal dominance ---

  // Per-row ratio: |a_ii| / (sum_{j≠i} |a_ij|).
  // Ratio >= 1 => diagonally dominant. Infinite ratio if off-diagonal
  // sum is zero (trivially dominant).
  template <typename T>
  std::vector<T>
  row_dominance_ratios(Compressed_row_matrix<T> const& A) {
    auto m = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> ratios(static_cast<std::size_t>(m), T{0});

    for (config::size_type i = 0; i < m; ++i) {
      T diag{0};
      Neumaier_sum<T> off;
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        auto v = std::abs(vals[static_cast<std::size_t>(p)]);
        if (j == i) {
          diag = v;
        } else {
          off.add(v);
        }
      }
      T off_sum = off.result();
      if (off_sum == T{0}) {
        ratios[static_cast<std::size_t>(i)] =
          std::numeric_limits<T>::infinity();
      } else {
        ratios[static_cast<std::size_t>(i)] = diag / off_sum;
      }
    }
    return ratios;
  }

  // True if every row ratio >= 1 (non-strict diagonal dominance by row)
  template <typename T>
  bool
  is_row_diagonally_dominant(Compressed_row_matrix<T> const& A) {
    auto ratios = row_dominance_ratios(A);
    return std::all_of(
      ratios.begin(), ratios.end(), [](T r) { return r >= T{1}; });
  }

  // True if every row ratio > 1
  template <typename T>
  bool
  is_strictly_row_diagonally_dominant(Compressed_row_matrix<T> const& A) {
    auto ratios = row_dominance_ratios(A);
    return std::all_of(
      ratios.begin(), ratios.end(), [](T r) { return r > T{1}; });
  }

  // Per-column ratio: |a_jj| / (sum_{i≠j} |a_ij|).
  //
  // Uses column_norms_1 from info.hpp, then subtracts the diagonal.
  template <typename T>
  std::vector<T>
  column_dominance_ratios(Compressed_row_matrix<T> const& A) {
    auto m = A.shape().row();
    auto n = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    // Diagonal values indexed by column
    std::vector<T> diag(static_cast<std::size_t>(n), T{0});
    for (config::size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j == i) {
          diag[static_cast<std::size_t>(j)] =
            std::abs(vals[static_cast<std::size_t>(p)]);
        }
      }
    }

    auto col_l1 = column_norms_1(A);

    std::vector<T> ratios(static_cast<std::size_t>(n));
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
      T d = diag[j];
      T off_sum = col_l1[j] - d;
      if (off_sum <= T{0}) {
        ratios[j] = std::numeric_limits<T>::infinity();
      } else {
        ratios[j] = d / off_sum;
      }
    }
    return ratios;
  }

  template <typename T>
  bool
  is_column_diagonally_dominant(Compressed_row_matrix<T> const& A) {
    auto ratios = column_dominance_ratios(A);
    return std::all_of(
      ratios.begin(), ratios.end(), [](T r) { return r >= T{1}; });
  }

  template <typename T>
  bool
  is_strictly_column_diagonally_dominant(Compressed_row_matrix<T> const& A) {
    auto ratios = column_dominance_ratios(A);
    return std::all_of(
      ratios.begin(), ratios.end(), [](T r) { return r > T{1}; });
  }

  // --- Positive definiteness ---

  // Attempts numeric Cholesky. Returns true iff it succeeds without
  // throwing std::domain_error. Throws std::invalid_argument if A is not
  // square.
  template <typename T>
  bool
  is_positive_definite(Compressed_row_matrix<T> const& A) {
    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument(
        "is_positive_definite requires a square matrix");
    }
    try {
      cholesky(A);
      return true;
    } catch (std::domain_error const&) { return false; }
  }

  // --- Sparsity visualization ---

  // ASCII spy-plot. Maps matrix to a grid of at most width x height
  // characters. Occupied cells print '#', empty cells print '.'.
  // Defaults: 80 wide x 40 tall.
  template <typename T>
  std::string
  spy(
    Compressed_row_matrix<T> const& A,
    config::size_type width = 80,
    config::size_type height = 40) {
    auto m = A.shape().row();
    auto n = A.shape().column();

    auto gw = std::min(width, n);
    auto gh = std::min(height, m);

    auto sgw = static_cast<std::size_t>(gw);
    auto sgh = static_cast<std::size_t>(gh);

    std::vector<std::vector<bool>> grid(sgh, std::vector<bool>(sgw, false));

    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    for (config::size_type i = 0; i < m; ++i) {
      auto gi = i * gh / m;
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        auto gj = j * gw / n;
        grid[static_cast<std::size_t>(gi)][static_cast<std::size_t>(gj)] = true;
      }
    }

    std::string result;
    result.reserve(static_cast<std::size_t>((gw + 1) * gh));
    for (std::size_t gi = 0; gi < sgh; ++gi) {
      for (std::size_t gj = 0; gj < sgw; ++gj) {
        result += grid[gi][gj] ? '#' : '.';
      }
      result += '\n';
    }
    return result;
  }

  // SVG spy-plot. Each nonzero cell is a px_per_cell x px_per_cell black
  // square. Returns a complete SVG document. Default cell size 4px.
  template <typename T>
  std::string
  spy_svg(
    Compressed_row_matrix<T> const& A, config::size_type px_per_cell = 4) {
    auto m = A.shape().row();
    auto n = A.shape().column();

    auto svg_w = n * px_per_cell;
    auto svg_h = m * px_per_cell;

    std::ostringstream out;
    out << R"(<?xml version="1.0" encoding="UTF-8"?>)"
        << "\n"
        << R"(<svg xmlns="http://www.w3.org/2000/svg" width=")" << svg_w
        << R"(" height=")" << svg_h << R"(">)"
        << "\n";

    auto rp = A.row_ptr();
    auto ci = A.col_ind();

    for (config::size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        auto x = j * px_per_cell;
        auto y = i * px_per_cell;
        out << R"(<rect x=")" << x << R"(" y=")" << y << R"(" width=")"
            << px_per_cell << R"(" height=")" << px_per_cell
            << R"(" fill="black"/>)"
            << "\n";
      }
    }

    out << "</svg>\n";
    return out.str();
  }

} // end of namespace sparkit::data::detail
