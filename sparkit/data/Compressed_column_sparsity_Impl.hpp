#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_column_sparsity.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Hidden implementation of Compressed_column_sparsity.
   *
   * Owns the CSC data arrays and implements the construction algorithm:
   *   1. Sort indices by (column, row).
   *   2. Remove duplicate entries.
   *   3. Build the row-index array from sorted unique indices.
   *   4. Build the column-pointer array via a shifted count + prefix sum.
   *
   * @note Implicitly copyable (compiler-generated copy is correct
   *       because all members are value types / vectors).
   */
  class Compressed_column_sparsity::Impl {
  public:
    /**
     * @brief Construct the CSC arrays from a shape and a collection of indices.
     *
     * @param shape    Matrix dimensions; determines the length of col_ptr_.
     * @param indices  Nonzero positions (consumed by value; sorted and
     *                 deduplicated internally).
     */
    Impl(Shape shape, std::vector<Index> indices);

    /// @brief Return the matrix dimensions.
    Shape
    shape() const;

    /// @brief Return the number of structural nonzeros.
    size_type
    size() const;

    /// @brief Return a read-only view of the column-pointer array.
    std::span<size_type const>
    col_ptr() const;

    /// @brief Return a read-only view of the row-index array.
    std::span<size_type const>
    row_ind() const;

  private:
    Shape shape_; ///< Matrix dimensions.
    std::vector<size_type>
      col_ptr_; ///< Column offsets, length shape_.column()+1.
    std::vector<size_type> row_ind_; ///< Row indices, length nnz.

  }; // end of class Compressed_column_sparsity::Impl

} // end of namespace sparkit::data::detail
