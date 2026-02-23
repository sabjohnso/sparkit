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
   * @brief Immutable diagonal (DIA) sparsity pattern.
   *
   * Stores which diagonals are present. An offset of 0 is the main
   * diagonal, positive offsets are super-diagonals, negative offsets
   * are sub-diagonals.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Compressed_row_sparsity
   */
  class Diagonal_sparsity final {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from a shape and a list of diagonal offsets.
     */
    Diagonal_sparsity(
      Shape shape, std::initializer_list<size_type> const& offsets);

    /**
     * @brief Construct from a shape and an initializer list of indices.
     *
     * Diagonal offsets are deduced from the indices (offset = col - row).
     */
    Diagonal_sparsity(Shape shape, std::initializer_list<Index> const& input);

    template <typename Iter>
    Diagonal_sparsity(Shape shape, Iter first, Iter last)
        : Diagonal_sparsity(shape, std::vector<Index>(first, last)) {}

    Diagonal_sparsity(Diagonal_sparsity const& input);
    Diagonal_sparsity(Diagonal_sparsity&& input);

    Diagonal_sparsity&
    operator=(Diagonal_sparsity const& input);

    Diagonal_sparsity&
    operator=(Diagonal_sparsity&& input);

    ~Diagonal_sparsity();

    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return the sorted diagonal offsets.
     */
    std::span<size_type const>
    offsets() const;

    /**
     * @brief Return the number of stored diagonals.
     */
    size_type
    num_diagonals() const;

  private:
    Diagonal_sparsity(Shape shape, std::vector<size_type> offsets);
    Diagonal_sparsity(Shape shape, std::vector<Index> indices);

    class Impl;
    Impl* pimpl;

  }; // end of class Diagonal_sparsity

} // end of namespace sparkit::data::detail
