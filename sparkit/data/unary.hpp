#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  template<typename T>
  std::vector<T>
  extract_diagonal(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto diag_len = std::min(rows, cols);
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> d(static_cast<std::size_t>(diag_len), T{0});
    for (config::size_type i = 0; i < diag_len; ++i) {
      auto begin = ci.begin() + rp[i];
      auto end = ci.begin() + rp[i + 1];
      auto it = std::lower_bound(begin, end, i);
      if (it != end && *it == i) {
        auto idx = static_cast<std::size_t>(std::distance(ci.begin(), it));
        d[static_cast<std::size_t>(i)] = vals[idx];
      }
    }
    return d;
  }

  template<typename T>
  Compressed_row_matrix<T>
  extract_lower_triangle(
    Compressed_row_matrix<T> const& A,
    bool include_diagonal = false)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<Index> indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        bool keep = include_diagonal ? (ci[j] <= i) : (ci[j] < i);
        if (keep) {
          indices.push_back(Index{i, ci[j]});
          new_vals.push_back(vals[j]);
        }
      }
    }

    Compressed_row_sparsity sparsity{
      A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  extract_upper_triangle(
    Compressed_row_matrix<T> const& A,
    bool include_diagonal = false)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<Index> indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        bool keep = include_diagonal ? (ci[j] >= i) : (ci[j] > i);
        if (keep) {
          indices.push_back(Index{i, ci[j]});
          new_vals.push_back(vals[j]);
        }
      }
    }

    Compressed_row_sparsity sparsity{
      A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  transpose(Compressed_row_matrix<T> const& A)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<Index> indices;
    std::vector<T> new_vals;
    indices.reserve(static_cast<std::size_t>(A.size()));
    new_vals.reserve(static_cast<std::size_t>(A.size()));

    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        indices.push_back(Index{ci[j], i});
        new_vals.push_back(vals[j]);
      }
    }

    // Sort by (row, col) in transpose — need to reorder values with indices
    std::vector<std::size_t> perm(indices.size());
    std::iota(perm.begin(), perm.end(), std::size_t{0});

    auto by_row_col = [&](std::size_t a, std::size_t b) {
      if (indices[a].row() != indices[b].row())
        return indices[a].row() < indices[b].row();
      return indices[a].column() < indices[b].column();
    };
    std::sort(perm.begin(), perm.end(), by_row_col);

    std::vector<Index> sorted_indices;
    std::vector<T> sorted_vals;
    sorted_indices.reserve(perm.size());
    sorted_vals.reserve(perm.size());
    for (std::size_t k = 0; k < perm.size(); ++k) {
      sorted_indices.push_back(indices[perm[k]]);
      sorted_vals.push_back(new_vals[perm[k]]);
    }

    Compressed_row_sparsity sparsity{
      Shape{cols, rows}, sorted_indices.begin(), sorted_indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(sorted_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  filter(Compressed_row_matrix<T> const& A, T tolerance)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<Index> indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        if (std::abs(vals[j]) > tolerance) {
          indices.push_back(Index{i, ci[j]});
          new_vals.push_back(vals[j]);
        }
      }
    }

    Compressed_row_sparsity sparsity{
      A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  submatrix(
    Compressed_row_matrix<T> const& A,
    config::size_type row_start,
    config::size_type row_end,
    config::size_type col_start,
    config::size_type col_end)
  {
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<Index> indices;
    std::vector<T> new_vals;

    for (auto i = row_start; i < row_end; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        if (ci[j] >= col_start && ci[j] < col_end) {
          indices.push_back(Index{i - row_start, ci[j] - col_start});
          new_vals.push_back(vals[j]);
        }
      }
    }

    Compressed_row_sparsity sparsity{
      Shape{row_end - row_start, col_end - col_start},
      indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

} // end of namespace sparkit::data::detail
