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
#include <sparkit/data/Block_sparse_row_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template <typename T = config::value_type>
  class Block_sparse_row_matrix final {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from sparsity and flat values array.
     *
     * Values are stored contiguously: for each block (in block-CSR order),
     * the dense block is stored row-major (br x bc values).
     */
    Block_sparse_row_matrix(Block_sparse_row_sparsity sparsity,
                            std::vector<T> values)
        : sparsity_(std::move(sparsity)), values_(std::move(values)) {}

    Block_sparse_row_matrix(Shape shape, size_type block_rows,
                            size_type block_cols,
                            std::initializer_list<Entry<T>> const& input)
        : Block_sparse_row_matrix(
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

      // Find block column in this block row
      for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
        if (ci[j] == block_col) {
          auto block_idx = j;
          auto offset = block_idx * br * bc + local_row * bc + local_col;
          return values_[static_cast<std::size_t>(offset)];
        }
      }
      return T{0};
    }

    std::span<T const>
    values() const {
      return {values_.data(), values_.size()};
    }

    Block_sparse_row_sparsity const&
    sparsity() const {
      return sparsity_;
    }

  private:
    static Block_sparse_row_matrix
    from_entries(Shape shape, size_type block_rows, size_type block_cols,
                 std::initializer_list<Entry<T>> const& input) {
      std::vector<Entry<T>> sorted(input.begin(), input.end());

      auto by_row_col = [](auto const& a, auto const& b) {
        return a.index.row() < b.index.row() ||
               (a.index.row() == b.index.row() &&
                a.index.column() < b.index.column());
      };
      std::sort(sorted.begin(), sorted.end(), by_row_col);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(std::unique(sorted.begin(), sorted.end(), same_index),
                   sorted.end());

      std::vector<Index> indices;
      indices.reserve(sorted.size());
      for (auto const& e : sorted) {
        indices.push_back(e.index);
      }

      Block_sparse_row_sparsity sparsity{shape, block_rows, block_cols,
                                         indices.begin(), indices.end()};

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

        // Find block position
        for (auto j = rp[br_idx]; j < rp[br_idx + 1]; ++j) {
          if (ci[j] == bc_idx) {
            auto offset = j * block_rows * block_cols + local_row * block_cols +
                          local_col;
            values[static_cast<std::size_t>(offset)] = entry.value;
            break;
          }
        }
      }

      return Block_sparse_row_matrix{std::move(sparsity), std::move(values)};
    }

    Block_sparse_row_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Block_sparse_row_matrix

} // end of namespace sparkit::data::detail
