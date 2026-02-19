#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Modified_sparse_row_sparsity.hpp>

namespace sparkit::data::detail
{

  /**
   * @brief Hidden implementation of Modified_sparse_row_sparsity.
   *
   * Separates diagonal from off-diagonal entries during construction:
   *   1. Sort indices by (row, column) and deduplicate.
   *   2. For each entry where row == column, mark has_diagonal_[row].
   *   3. For off-diagonal entries, build CSR-like row_ptr/col_ind.
   */
  class Modified_sparse_row_sparsity::Impl
  {
  public:
    Impl(Shape shape, std::vector<Index> indices);

    Shape
    shape() const;

    size_type
    size() const;

    bool
    has_diagonal(size_type i) const;

    size_type
    diagonal_length() const;

    std::span<size_type const>
    off_diagonal_row_ptr() const;

    std::span<size_type const>
    off_diagonal_col_ind() const;

  private:
    Shape shape_;
    size_type total_size_;
    std::vector<bool> has_diagonal_;
    std::vector<size_type> off_diagonal_row_ptr_;
    std::vector<size_type> off_diagonal_col_ind_;

  }; // end of class Modified_sparse_row_sparsity::Impl

} // end of namespace sparkit::data::detail
