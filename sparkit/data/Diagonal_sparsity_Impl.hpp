#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Diagonal_sparsity.hpp>

namespace sparkit::data::detail
{

  /**
   * @brief Hidden implementation of Diagonal_sparsity.
   *
   * Stores sorted unique diagonal offsets and computes total size
   * (number of valid positions on all stored diagonals).
   */
  class Diagonal_sparsity::Impl
  {
  public:
    Impl(Shape shape, std::vector<size_type> offsets);

    Shape
    shape() const;

    size_type
    size() const;

    std::span<size_type const>
    offsets() const;

    size_type
    num_diagonals() const;

  private:
    Shape shape_;
    std::vector<size_type> offsets_;
    size_type total_size_;

  }; // end of class Diagonal_sparsity::Impl

} // end of namespace sparkit::data::detail
