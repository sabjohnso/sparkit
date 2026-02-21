#include <sparkit/data/Block_sparse_row_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <set>

namespace sparkit::data::detail {

  Block_sparse_row_sparsity::Impl::Impl(Shape shape, size_type block_rows,
                                        size_type block_cols,
                                        std::vector<Index> indices)
      : shape_(shape), block_rows_(block_rows), block_cols_(block_cols),
        num_block_rows_((shape.row() + block_rows - 1) / block_rows),
        num_block_cols_((shape.column() + block_cols - 1) / block_cols),
        row_ptr_(static_cast<std::size_t>(num_block_rows_ + 1), 0), col_ind_() {
    // Map scalar indices to block indices and deduplicate
    std::set<std::pair<size_type, size_type>> block_set;
    for (auto const& idx : indices) {
      auto br = idx.row() / block_rows_;
      auto bc = idx.column() / block_cols_;
      block_set.insert({br, bc});
    }

    // Count blocks per block-row
    for (auto const& [br, bc] : block_set) {
      ++row_ptr_[static_cast<std::size_t>(br)];
    }

    // Prefix sum
    size_type running = 0;
    for (std::size_t r = 0; r < row_ptr_.size(); ++r) {
      size_type count = row_ptr_[r];
      row_ptr_[r] = running;
      running += count;
    }

    // Fill col_ind (already sorted by set ordering)
    col_ind_.resize(block_set.size());
    std::vector<size_type> work(row_ptr_.begin(), row_ptr_.end());
    for (auto const& [br, bc] : block_set) {
      auto dest = work[static_cast<std::size_t>(br)]++;
      col_ind_[static_cast<std::size_t>(dest)] = bc;
    }
  }

  Shape
  Block_sparse_row_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::size() const {
    auto nblocks = static_cast<size_type>(col_ind_.size());
    return nblocks * block_rows_ * block_cols_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::block_rows() const {
    return block_rows_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::block_cols() const {
    return block_cols_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::num_block_rows() const {
    return num_block_rows_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::num_block_cols() const {
    return num_block_cols_;
  }

  size_type
  Block_sparse_row_sparsity::Impl::num_blocks() const {
    return static_cast<size_type>(col_ind_.size());
  }

  std::span<size_type const>
  Block_sparse_row_sparsity::Impl::row_ptr() const {
    return row_ptr_;
  }

  std::span<size_type const>
  Block_sparse_row_sparsity::Impl::col_ind() const {
    return col_ind_;
  }

} // end of namespace sparkit::data::detail
