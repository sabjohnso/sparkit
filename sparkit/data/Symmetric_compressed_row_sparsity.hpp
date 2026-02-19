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
   * @brief Immutable symmetric compressed sparse row (sCSR) sparsity pattern.
   *
   * Stores only the lower triangle (row >= col) of a symmetric matrix
   * in CSR format.  Indices provided from either triangle are
   * automatically normalized to lower triangle by swapping (i,j) to
   * (j,i) when i < j.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Symmetric_compressed_row_sparsity final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from a shape and an initializer list of indices.
     *
     * Indices are normalized to lower triangle, sorted, and deduplicated.
     *
     * @param shape  Matrix dimensions (must be square).
     * @param input  Nonzero positions (from either triangle).
     */
    Symmetric_compressed_row_sparsity(
      Shape shape,
      std::initializer_list<Index> const& input);

    template<typename Iter>
    Symmetric_compressed_row_sparsity(Shape shape, Iter first, Iter last)
      : Symmetric_compressed_row_sparsity(shape, std::vector<Index>(first, last))
    {}

    Symmetric_compressed_row_sparsity(
      Symmetric_compressed_row_sparsity const& input);
    Symmetric_compressed_row_sparsity(
      Symmetric_compressed_row_sparsity&& input);

    Symmetric_compressed_row_sparsity&
    operator=(Symmetric_compressed_row_sparsity const& input);

    Symmetric_compressed_row_sparsity&
    operator=(Symmetric_compressed_row_sparsity&& input);

    ~Symmetric_compressed_row_sparsity();

    /**
     * @brief Return the number of stored entries (lower triangle only).
     */
    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return the row-pointer array (length shape().row()+1).
     */
    std::span<size_type const>
    row_ptr() const;

    /**
     * @brief Return the column-index array (length size()).
     *
     * All column indices satisfy col <= row for the corresponding row.
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Symmetric_compressed_row_sparsity(
      Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Symmetric_compressed_row_sparsity

} // end of namespace sparkit::data::detail
