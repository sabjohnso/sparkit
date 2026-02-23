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
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/Symmetric_block_sparse_row_sparsity.hpp>

namespace sparkit::data::detail {

  template <typename T = config::value_type>
  class Symmetric_block_sparse_row_matrix final {
  public:
    using size_type = config::size_type;

    Symmetric_block_sparse_row_matrix(
      Symmetric_block_sparse_row_sparsity sparsity, std::vector<T> values)
        : sparsity_(std::move(sparsity))
        , values_(std::move(values)) {}

    Symmetric_block_sparse_row_matrix(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::initializer_list<Entry<T>> const& input)
        : Symmetric_block_sparse_row_matrix(
            from_entries(shape, block_rows, block_cols, input)) {}

    size_type
    size() const {
      return sparsity_.size();
    }

    Shape
    shape() const {
      return sparsity_.shape();
    }

    T
    operator()(size_type row, size_type col) const {
      auto br = sparsity_.block_rows();
      auto bc = sparsity_.block_cols();
      auto block_row = row / br;
      auto block_col = col / bc;
      auto local_row = row % br;
      auto local_col = col % bc;

      auto rp = sparsity_.row_ptr();
      auto ci = sparsity_.col_ind();

      if (block_row >= block_col) {
        // Lower triangle — direct lookup
        for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
          if (ci[j] == block_col) {
            auto offset = j * br * bc + local_row * bc + local_col;
            return values_[static_cast<std::size_t>(offset)];
          }
        }
      } else {
        // Upper triangle — look up transposed block with swapped local indices
        for (auto j = rp[block_col]; j < rp[block_col + 1]; ++j) {
          if (ci[j] == block_row) {
            auto offset = j * br * bc + local_col * bc + local_row;
            return values_[static_cast<std::size_t>(offset)];
          }
        }
      }
      return T{0};
    }

    Symmetric_block_sparse_row_sparsity const&
    sparsity() const {
      return sparsity_;
    }

  private:
    static Symmetric_block_sparse_row_matrix
    from_entries(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::initializer_list<Entry<T>> const& input) {
      // Normalize entries to lower-triangle blocks
      std::vector<Entry<T>> sorted;
      sorted.reserve(input.size());
      for (auto const& e : input) {
        auto br_idx = e.index.row() / block_rows;
        auto bc_idx = e.index.column() / block_cols;
        if (br_idx < bc_idx) {
          // Swap block position and local indices
          auto local_row = e.index.row() % block_rows;
          auto local_col = e.index.column() % block_cols;
          auto new_row = bc_idx * block_rows + local_col;
          auto new_col = br_idx * block_cols + local_row;
          sorted.push_back({Index{new_row, new_col}, e.value});
        } else {
          sorted.push_back(e);
        }
      }

      auto by_row_col = [](auto const& a, auto const& b) {
        return a.index.row() < b.index.row() ||
               (a.index.row() == b.index.row() &&
                a.index.column() < b.index.column());
      };
      std::sort(sorted.begin(), sorted.end(), by_row_col);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(
        std::unique(sorted.begin(), sorted.end(), same_index), sorted.end());

      std::vector<Index> indices;
      indices.reserve(sorted.size());
      for (auto const& e : sorted) {
        indices.push_back(e.index);
      }

      Symmetric_block_sparse_row_sparsity sparsity{
        shape, block_rows, block_cols, indices.begin(), indices.end()};

      // Allocate values for all blocks
      auto num_blocks = sparsity.num_blocks();
      std::vector<T> values(
        static_cast<std::size_t>(num_blocks * block_rows * block_cols), T{0});

      // Fill values
      auto rp = sparsity.row_ptr();
      auto ci = sparsity.col_ind();

      for (auto const& entry : sorted) {
        auto br_idx = entry.index.row() / block_rows;
        auto bc_idx = entry.index.column() / block_cols;
        auto local_row = entry.index.row() % block_rows;
        auto local_col = entry.index.column() % block_cols;

        for (auto j = rp[br_idx]; j < rp[br_idx + 1]; ++j) {
          if (ci[j] == bc_idx) {
            auto offset =
              j * block_rows * block_cols + local_row * block_cols + local_col;
            values[static_cast<std::size_t>(offset)] = entry.value;
            break;
          }
        }
      }

      return Symmetric_block_sparse_row_matrix{
        std::move(sparsity), std::move(values)};
    }

    Symmetric_block_sparse_row_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Symmetric_block_sparse_row_matrix

} // end of namespace sparkit::data::detail
