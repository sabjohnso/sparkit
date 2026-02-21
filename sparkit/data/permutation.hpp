#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // -- Permutation utilities (implemented in permutation.cpp) --

  bool
  is_valid_permutation(std::span<config::size_type const> perm);

  std::vector<config::size_type>
  inverse_permutation(std::span<config::size_type const> perm);

  // -- Sparsity permutations (implemented in permutation.cpp) --

  Compressed_row_sparsity
  rperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm);

  Compressed_row_sparsity
  cperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm);

  Compressed_row_sparsity
  dperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm);

  // -- Matrix permutations (template, header-only) --

  template <typename T>
  Compressed_row_matrix<T>
  rperm(Compressed_row_matrix<T> const& A,
        std::span<config::size_type const> perm) {
    auto inv = inverse_permutation(perm);
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto sv = A.values();
    auto nrow = A.shape().row();

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(static_cast<std::size_t>(A.size()));
    values.reserve(static_cast<std::size_t>(A.size()));

    for (config::size_type new_row = 0; new_row < nrow; ++new_row) {
      auto old_row = inv[static_cast<std::size_t>(new_row)];
      for (auto j = rp[old_row]; j < rp[old_row + 1]; ++j) {
        indices.push_back(Index{new_row, ci[j]});
        values.push_back(sv[j]);
      }
    }

    Compressed_row_sparsity sp{A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sp), std::move(values)};
  }

  template <typename T>
  Compressed_row_matrix<T>
  cperm(Compressed_row_matrix<T> const& A,
        std::span<config::size_type const> perm) {
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto sv = A.values();
    auto nrow = A.shape().row();

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(static_cast<std::size_t>(A.size()));
    values.reserve(static_cast<std::size_t>(A.size()));

    for (config::size_type row = 0; row < nrow; ++row) {
      // Collect (new_col, value) pairs for this row
      std::vector<std::pair<config::size_type, T>> row_entries;
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        auto new_col = perm[static_cast<std::size_t>(ci[j])];
        row_entries.push_back({new_col, sv[j]});
      }

      // Sort by new column index
      std::sort(row_entries.begin(), row_entries.end(),
                [](auto const& a, auto const& b) { return a.first < b.first; });

      for (auto const& [col, val] : row_entries) {
        indices.push_back(Index{row, col});
        values.push_back(val);
      }
    }

    Compressed_row_sparsity sp{A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sp), std::move(values)};
  }

  template <typename T>
  Compressed_row_matrix<T>
  dperm(Compressed_row_matrix<T> const& A,
        std::span<config::size_type const> perm) {
    auto inv = inverse_permutation(perm);
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto sv = A.values();
    auto nrow = A.shape().row();

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(static_cast<std::size_t>(A.size()));
    values.reserve(static_cast<std::size_t>(A.size()));

    for (config::size_type new_row = 0; new_row < nrow; ++new_row) {
      auto old_row = inv[static_cast<std::size_t>(new_row)];

      // Collect (new_col, value) pairs for this row
      std::vector<std::pair<config::size_type, T>> row_entries;
      for (auto j = rp[old_row]; j < rp[old_row + 1]; ++j) {
        auto new_col = perm[static_cast<std::size_t>(ci[j])];
        row_entries.push_back({new_col, sv[j]});
      }

      // Sort by new column index
      std::sort(row_entries.begin(), row_entries.end(),
                [](auto const& a, auto const& b) { return a.first < b.first; });

      for (auto const& [col, val] : row_entries) {
        indices.push_back(Index{new_row, col});
        values.push_back(val);
      }
    }

    Compressed_row_sparsity sp{A.shape(), indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sp), std::move(values)};
  }

} // end of namespace sparkit::data::detail
