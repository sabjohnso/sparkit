#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Block_sparse_row_sparsity.hpp>

namespace sparkit::data::detail {

  class Block_sparse_row_sparsity::Impl {
  public:
    Impl(Shape shape, size_type block_rows, size_type block_cols,
         std::vector<Index> indices);

    Shape
    shape() const;
    size_type
    size() const;
    size_type
    block_rows() const;
    size_type
    block_cols() const;
    size_type
    num_block_rows() const;
    size_type
    num_block_cols() const;
    size_type
    num_blocks() const;
    std::span<size_type const>
    row_ptr() const;
    std::span<size_type const>
    col_ind() const;

  private:
    Shape shape_;
    size_type block_rows_;
    size_type block_cols_;
    size_type num_block_rows_;
    size_type num_block_cols_;
    std::vector<size_type> row_ptr_;
    std::vector<size_type> col_ind_;

  }; // end of class Block_sparse_row_sparsity::Impl

} // end of namespace sparkit::data::detail
