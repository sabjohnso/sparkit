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

namespace sparkit::data::detail
{

  /**
   * @brief Immutable block sparse row (BSR) sparsity pattern.
   *
   * Stores a block-level CSR structure where each structural entry
   * represents a dense block of size block_rows x block_cols.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Block_sparse_row_sparsity final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from scalar shape, block dimensions, and scalar indices.
     *
     * Scalar indices are mapped to block positions, deduplicated.
     */
    Block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::initializer_list<Index> const& input);

    template<typename Iter>
    Block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      Iter first, Iter last)
      : Block_sparse_row_sparsity(shape, block_rows, block_cols,
          std::vector<Index>(first, last))
    {}

    Block_sparse_row_sparsity(Block_sparse_row_sparsity const& input);
    Block_sparse_row_sparsity(Block_sparse_row_sparsity&& input);

    Block_sparse_row_sparsity&
    operator=(Block_sparse_row_sparsity const& input);

    Block_sparse_row_sparsity&
    operator=(Block_sparse_row_sparsity&& input);

    ~Block_sparse_row_sparsity();

    /**
     * @brief Total number of scalar positions in all blocks.
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
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Block_sparse_row_sparsity(
      Shape shape,
      size_type block_rows,
      size_type block_cols,
      std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Block_sparse_row_sparsity

} // end of namespace sparkit::data::detail
