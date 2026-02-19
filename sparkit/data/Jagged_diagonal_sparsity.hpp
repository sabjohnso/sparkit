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
   * @brief Immutable jagged diagonal (JAD) sparsity pattern.
   *
   * Stores a row permutation sorted by decreasing number of nonzeros
   * per row, jagged diagonal pointers, and column indices in jagged
   * diagonal order. This format is suited for vector and parallel
   * architectures because the jagged diagonals can be processed as
   * independent, densely packed segments.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Jagged_diagonal_sparsity final
  {
  public:
    using size_type = config::size_type;

    Jagged_diagonal_sparsity(Shape shape, std::initializer_list<Index> const& input);

    template<typename Iter>
    Jagged_diagonal_sparsity(Shape shape, Iter first, Iter last)
      : Jagged_diagonal_sparsity(shape, std::vector<Index>(first, last))
    {}

    Jagged_diagonal_sparsity(Jagged_diagonal_sparsity const& input);
    Jagged_diagonal_sparsity(Jagged_diagonal_sparsity&& input);

    Jagged_diagonal_sparsity&
    operator=(Jagged_diagonal_sparsity const& input);

    Jagged_diagonal_sparsity&
    operator=(Jagged_diagonal_sparsity&& input);

    ~Jagged_diagonal_sparsity();

    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return the row permutation (sorted by decreasing nnz).
     *
     * perm[i] is the original row index of the i-th permuted row.
     */
    std::span<size_type const>
    perm() const;

    /**
     * @brief Return the jagged diagonal pointers.
     *
     * Has max_nnz_per_row + 1 entries. Jagged diagonal k spans
     * col_ind[jdiag[k] .. jdiag[k+1]).
     */
    std::span<size_type const>
    jdiag() const;

    /**
     * @brief Return the column-index array in jagged diagonal order.
     *
     * Has size() entries.
     */
    std::span<size_type const>
    col_ind() const;

  private:
    Jagged_diagonal_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Jagged_diagonal_sparsity

} // end of namespace sparkit::data::detail
