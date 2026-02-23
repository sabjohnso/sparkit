#pragma once

//
// ... Standard header files
//
#include <initializer_list>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Index.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Immutable symmetric block sparse row (sBSR) sparsity pattern.
   *
   * Stores a block-level CSR structure where only lower-triangle blocks
   * (block_row >= block_col) are stored.  Scalar indices are mapped to
   * block positions and normalized to lower triangle.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Block_sparse_row_sparsity
   */
  class Symmetric_block_sparse_row_sparsity final {
  public:
    using size_type = config::size_type;

    Symmetric_block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::initializer_list<Index> const& input);

    template <typename Iter>
    Symmetric_block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      Iter first,
      Iter last)
        : Symmetric_block_sparse_row_sparsity(
            shape, block_rows, block_cols, std::vector<Index>(first, last)) {}

    Symmetric_block_sparse_row_sparsity(
      Symmetric_block_sparse_row_sparsity const& input);
    Symmetric_block_sparse_row_sparsity(
      Symmetric_block_sparse_row_sparsity&& input);

    Symmetric_block_sparse_row_sparsity&
    operator=(Symmetric_block_sparse_row_sparsity const& input);

    Symmetric_block_sparse_row_sparsity&
    operator=(Symmetric_block_sparse_row_sparsity&& input);

    ~Symmetric_block_sparse_row_sparsity();

    /**
     * @brief Total number of scalar positions in all stored blocks.
     */
    size_type
    size() const;

    Shape
    shape() const;

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

    /**
     * @brief Block-level row pointer array (length num_block_rows + 1).
     */
    std::span<size_type const>
    row_ptr() const;

    /**
     * @brief Block-level column index array (length num_blocks).
     *
     * All column indices satisfy col <= row for the corresponding block row.
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Symmetric_block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Symmetric_block_sparse_row_sparsity

} // end of namespace sparkit::data::detail
