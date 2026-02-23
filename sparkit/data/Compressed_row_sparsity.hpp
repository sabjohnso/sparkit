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
   * @brief Immutable compressed sparse row (CSR) sparsity pattern.
   *
   * Stores the structure of a sparse matrix in CSR format: a row-pointer
   * array and a column-index array. This is the sparsity pattern only —
   * no values are stored.
   *
   * CSR is the hub format in sparkit. Most format conversions route
   * through CSR, following the SPARSKIT2 convention.
   *
   * The typical workflow is to build a pattern in COO
   * (Coordinate_sparsity), then convert to CSR for efficient row-wise
   * access. Duplicate indices provided at construction are collapsed
   * and column indices are sorted within each row.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Coordinate_sparsity
   */
  class Compressed_row_sparsity final {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from a shape and an initializer list of indices.
     *
     * Indices are sorted by (row, column) and duplicates are removed.
     *
     * @param shape  Matrix dimensions.
     * @param input  Nonzero positions.
     */
    Compressed_row_sparsity(
      Shape shape, std::initializer_list<Index> const& input);

    /**
     * @brief Construct from a shape and an iterator range of indices.
     *
     * Indices are sorted by (row, column) and duplicates are removed.
     *
     * @tparam Iter  An input iterator whose value type is Index.
     * @param shape  Matrix dimensions.
     * @param first  Beginning of the index range.
     * @param last   End of the index range.
     */
    template <typename Iter>
    Compressed_row_sparsity(Shape shape, Iter first, Iter last)
        : Compressed_row_sparsity(shape, std::vector<Index>(first, last)) {}

    Compressed_row_sparsity(Compressed_row_sparsity const& input);
    Compressed_row_sparsity(Compressed_row_sparsity&& input);

    Compressed_row_sparsity&
    operator=(Compressed_row_sparsity const& input);

    Compressed_row_sparsity&
    operator=(Compressed_row_sparsity&& input);

    ~Compressed_row_sparsity();

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
     * @brief Return the row-pointer array.
     *
     * The returned span has @c shape().row()+1 elements.
     * For row @c r, the column indices are in
     * @c col_ind()[row_ptr()[r] .. row_ptr()[r+1]).
     */
    std::span<size_type const>
    row_ptr() const;

    /**
     * @brief Return the column-index array.
     *
     * The returned span has @c size() elements. Column indices are
     * sorted in ascending order within each row.
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Compressed_row_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Compressed_row_sparsity

} // end of namespace sparkit::data::detail
