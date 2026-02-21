#pragma once

//
// ... Standard header files
//
#include <initializer_list>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Index.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data::detail {

  /**
   * @brief Mutable symmetric coordinate (sCOO) sparsity pattern.
   *
   * Stores only lower-triangle entries (row >= col) in an unordered set.
   * All indices are normalized to lower triangle on insertion.
   *
   * @note This class uses the PImpl idiom for ABI stability.
   *
   * @see Coordinate_sparsity
   */
  class Symmetric_coordinate_sparsity {
  public:
    using size_type = config::size_type;

    template <typename Iter>
    Symmetric_coordinate_sparsity(Shape shape, Iter first, Iter last)
        : pimpl(nullptr) {
      init(shape);
      std::for_each(first, last, [this](Index index) { add(index); });
    }

    Symmetric_coordinate_sparsity(Shape shape,
                                  std::initializer_list<Index> const& input);

    Symmetric_coordinate_sparsity(Symmetric_coordinate_sparsity const& input);
    Symmetric_coordinate_sparsity(Symmetric_coordinate_sparsity&& input);

    Symmetric_coordinate_sparsity&
    operator=(Symmetric_coordinate_sparsity const& input);

    Symmetric_coordinate_sparsity&
    operator=(Symmetric_coordinate_sparsity&& input);

    ~Symmetric_coordinate_sparsity();

    /**
     * @brief Add an index, normalizing to lower triangle.
     */
    void
    add(Index index);

    /**
     * @brief Remove an index, normalizing to lower triangle.
     */
    void
    remove(Index index);

    /**
     * @brief Return the number of stored entries (lower triangle only).
     */
    size_type
    size() const;

    Shape
    shape() const;

    /**
     * @brief Return all stored indices (lower triangle only).
     */
    std::vector<Index>
    indices() const;

  private:
    void
    init(Shape shape);

    class Impl;
    Impl* pimpl;

  }; // end of class Symmetric_coordinate_sparsity

} // end of namespace sparkit::data::detail
