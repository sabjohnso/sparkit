#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Ellpack_sparsity.hpp>

namespace sparkit::data::detail
{

  class Ellpack_sparsity::Impl
  {
  public:
    Impl(Shape shape, std::vector<Index> indices);

    Shape
    shape() const;

    size_type
    size() const;

    size_type
    max_nnz_per_row() const;

    std::span<size_type const>
    col_ind() const;

  private:
    Shape shape_;
    size_type total_size_;
    size_type max_nnz_per_row_;
    std::vector<size_type> col_ind_;

  }; // end of class Ellpack_sparsity::Impl

} // end of namespace sparkit::data::detail
