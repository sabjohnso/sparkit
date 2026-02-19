#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <initializer_list>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Modified_sparse_row_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  class Modified_sparse_row_matrix final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from sparsity, diagonal values, and off-diagonal values.
     */
    Modified_sparse_row_matrix(
      Modified_sparse_row_sparsity sparsity,
      std::vector<T> diagonal,
      std::vector<T> off_diagonal_values)
      : sparsity_(std::move(sparsity))
      , diagonal_(std::move(diagonal))
      , off_diagonal_values_(std::move(off_diagonal_values))
    {}

    /**
     * @brief Construct from a shape and an entry list.
     */
    Modified_sparse_row_matrix(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
      : Modified_sparse_row_matrix(from_entries(shape, input))
    {}

    size_type
    size() const
    {
      return sparsity_.size();
    }

    Shape
    shape() const
    {
      return sparsity_.shape();
    }

    std::span<T const>
    diagonal() const
    {
      return {diagonal_.data(), diagonal_.size()};
    }

    T
    operator()(size_type row, size_type col) const
    {
      auto diag_len = sparsity_.diagonal_length();

      // Check diagonal
      if (row == col && row < diag_len) {
        if (sparsity_.has_diagonal(row)) {
          return diagonal_[static_cast<std::size_t>(row)];
        }
        return T{0};
      }

      // Check off-diagonal
      auto rp = sparsity_.off_diagonal_row_ptr();
      auto ci = sparsity_.off_diagonal_col_ind();
      auto begin = ci.begin() + rp[row];
      auto end = ci.begin() + rp[row + 1];
      for (auto it = begin; it != end; ++it) {
        if (*it == col) {
          auto idx = static_cast<std::size_t>(std::distance(ci.begin(), it));
          return off_diagonal_values_[idx];
        }
      }
      return T{0};
    }

    Modified_sparse_row_sparsity const&
    sparsity() const
    {
      return sparsity_;
    }

  private:

    static
    Modified_sparse_row_matrix
    from_entries(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
    {
      std::vector<Entry<T>> sorted(input.begin(), input.end());

      auto by_row_col = [](auto const& a, auto const& b) {
        return a.index.row() < b.index.row()
          || (a.index.row() == b.index.row()
              && a.index.column() < b.index.column());
      };
      std::sort(sorted.begin(), sorted.end(), by_row_col);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(
        std::unique(sorted.begin(), sorted.end(), same_index),
        sorted.end());

      auto diag_len = std::min(shape.row(), shape.column());

      std::vector<Index> indices;
      std::vector<T> diagonal(static_cast<std::size_t>(diag_len), T{0});
      std::vector<T> off_diag_values;

      indices.reserve(sorted.size());
      off_diag_values.reserve(sorted.size());

      for (auto const& entry : sorted) {
        indices.push_back(entry.index);
      }

      Modified_sparse_row_sparsity sparsity{shape, indices.begin(), indices.end()};

      // Fill diagonal values
      for (auto const& entry : sorted) {
        if (entry.index.row() == entry.index.column()
            && entry.index.row() < diag_len) {
          diagonal[static_cast<std::size_t>(entry.index.row())] = entry.value;
        }
      }

      // Fill off-diagonal values in the order they appear in the sparsity
      auto rp = sparsity.off_diagonal_row_ptr();
      auto ci = sparsity.off_diagonal_col_ind();
      off_diag_values.resize(static_cast<std::size_t>(ci.size()), T{0});

      for (auto const& entry : sorted) {
        if (entry.index.row() == entry.index.column()
            && entry.index.row() < diag_len) {
          continue;
        }
        // Find position in off-diagonal col_ind
        auto row = entry.index.row();
        auto begin_it = ci.begin() + rp[row];
        auto end_it = ci.begin() + rp[row + 1];
        for (auto it = begin_it; it != end_it; ++it) {
          if (*it == entry.index.column()) {
            auto idx = static_cast<std::size_t>(std::distance(ci.begin(), it));
            off_diag_values[idx] = entry.value;
            break;
          }
        }
      }

      return Modified_sparse_row_matrix{
        std::move(sparsity),
        std::move(diagonal),
        std::move(off_diag_values)};
    }

    Modified_sparse_row_sparsity sparsity_;
    std::vector<T> diagonal_;
    std::vector<T> off_diagonal_values_;

  }; // end of class Modified_sparse_row_matrix

} // end of namespace sparkit::data::detail
