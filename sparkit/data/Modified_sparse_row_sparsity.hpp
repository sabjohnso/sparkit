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
   * @brief Immutable modified sparse row (MSR) sparsity pattern.
   *
   * Separates diagonal entries from off-diagonal entries. The diagonal
   * is tracked per-element, while off-diagonal entries are stored in
   * CSR-like format.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Modified_sparse_row_sparsity final {
  public:
    using size_type = config::size_type;

    Modified_sparse_row_sparsity(
      Shape shape, std::initializer_list<Index> const& input);

    template <typename Iter>
    Modified_sparse_row_sparsity(Shape shape, Iter first, Iter last)
        : Modified_sparse_row_sparsity(shape, std::vector<Index>(first, last)) {
    }

    Modified_sparse_row_sparsity(Modified_sparse_row_sparsity const& input);
    Modified_sparse_row_sparsity(Modified_sparse_row_sparsity&& input);

    Modified_sparse_row_sparsity&
    operator=(Modified_sparse_row_sparsity const& input);

    Modified_sparse_row_sparsity&
    operator=(Modified_sparse_row_sparsity&& input);

    ~Modified_sparse_row_sparsity();

    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return whether diagonal position i exists.
     *
     * Valid for i in [0, min(nrow, ncol)).
     */
    bool
    has_diagonal(size_type i) const;

    /**
     * @brief Return the length of the diagonal: min(nrow, ncol).
     */
    size_type
    diagonal_length() const;

    /**
     * @brief Return the row-pointer array for off-diagonal entries.
     *
     * Length is nrow + 1.
     */
    std::span<size_type const>
    off_diagonal_row_ptr() const;

    /**
     * @brief Return the column-index array for off-diagonal entries.
     */
    std::span<size_type const>
    off_diagonal_col_ind() const;

  private:
    Modified_sparse_row_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Modified_sparse_row_sparsity

} // end of namespace sparkit::data::detail
