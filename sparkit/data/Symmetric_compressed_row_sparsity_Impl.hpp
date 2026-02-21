#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  class Symmetric_compressed_row_sparsity::Impl {
  public:
    Impl(Shape shape, std::vector<Index> indices);

    Shape
    shape() const;

    size_type
    size() const;

    std::span<size_type const>
    row_ptr() const;

    std::span<size_type const>
    col_ind() const;

  private:
    Shape shape_;
    std::vector<size_type> row_ptr_;
    std::vector<size_type> col_ind_;

  }; // end of class Symmetric_compressed_row_sparsity::Impl

} // end of namespace sparkit::data::detail
