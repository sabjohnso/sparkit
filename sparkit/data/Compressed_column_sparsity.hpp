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
   * @brief Immutable compressed sparse column (CSC) sparsity pattern.
   *
   * Stores the structure of a sparse matrix in CSC format: a column-pointer
   * array and a row-index array. This is the column-oriented mirror of CSR.
   *
   * Duplicate indices provided at construction are collapsed
   * and row indices are sorted within each column.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Compressed_column_sparsity final {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from a shape and an initializer list of indices.
     *
     * Indices are sorted by (column, row) and duplicates are removed.
     *
     * @param shape  Matrix dimensions.
     * @param input  Nonzero positions.
     */
    Compressed_column_sparsity(Shape shape,
                               std::initializer_list<Index> const& input);

    /**
     * @brief Construct from a shape and an iterator range of indices.
     *
     * Indices are sorted by (column, row) and duplicates are removed.
     *
     * @tparam Iter  An input iterator whose value type is Index.
     * @param shape  Matrix dimensions.
     * @param first  Beginning of the index range.
     * @param last   End of the index range.
     */
    template <typename Iter>
    Compressed_column_sparsity(Shape shape, Iter first, Iter last)
        : Compressed_column_sparsity(shape, std::vector<Index>(first, last)) {}

    Compressed_column_sparsity(Compressed_column_sparsity const& input);
    Compressed_column_sparsity(Compressed_column_sparsity&& input);

    Compressed_column_sparsity&
    operator=(Compressed_column_sparsity const& input);

    Compressed_column_sparsity&
    operator=(Compressed_column_sparsity&& input);

    ~Compressed_column_sparsity();

    /**
     * @brief Return the number of structural nonzeros.
     */
    size_type
    size() const;

    /**
     * @brief Return the matrix dimensions.
     */
    Shape
    shape() const;

    /**
     * @brief Return the column-pointer array.
     *
     * The returned span has @c shape().column()+1 elements.
     * For column @c c, the row indices are in
     * @c row_ind()[col_ptr()[c] .. col_ptr()[c+1]).
     */
    std::span<size_type const>
    col_ptr() const;

    /**
     * @brief Return the row-index array.
     *
     * The returned span has @c size() elements. Row indices are
     * sorted in ascending order within each column.
     */
    std::span<size_type const>
    row_ind() const;

  private:
    Compressed_column_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Compressed_column_sparsity

} // end of namespace sparkit::data::detail
