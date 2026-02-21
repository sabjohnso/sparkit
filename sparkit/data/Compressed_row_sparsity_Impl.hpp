#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Hidden implementation of Compressed_row_sparsity.
   *
   * Owns the CSR data arrays and implements the construction algorithm:
   *   1. Sort indices by (row, column).
   *   2. Remove duplicate entries.
   *   3. Build the column-index array from sorted unique indices.
   *   4. Build the row-pointer array via a shifted count + prefix sum.
   *
   * @note Implicitly copyable (compiler-generated copy is correct
   *       because all members are value types / vectors).
   */
  class Compressed_row_sparsity::Impl {
  public:
    /**
     * @brief Construct the CSR arrays from a shape and a collection of indices.
     *
     * @param shape    Matrix dimensions; determines the length of row_ptr_.
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

    /// @brief Return a read-only view of the row-pointer array.
    std::span<size_type const>
    row_ptr() const;

    /// @brief Return a read-only view of the column-index array.
    std::span<size_type const>
    col_ind() const;

  private:
    Shape shape_;                    ///< Matrix dimensions.
    std::vector<size_type> row_ptr_; ///< Row offsets, length shape_.row()+1.
    std::vector<size_type> col_ind_; ///< Column indices, length nnz.

  }; // end of class Compressed_row_sparsity::Impl

} // end of namespace sparkit::data::detail
