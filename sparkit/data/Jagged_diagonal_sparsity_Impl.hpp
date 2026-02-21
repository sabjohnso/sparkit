#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Jagged_diagonal_sparsity.hpp>

namespace sparkit::data::detail {

  class Jagged_diagonal_sparsity::Impl {
  public:
    Impl(Shape shape, std::vector<Index> indices);

    Shape
    shape() const;

    size_type
    size() const;

    std::span<size_type const>
    perm() const;

    std::span<size_type const>
    jdiag() const;

    std::span<size_type const>
    col_ind() const;

  private:
    Shape shape_;
    size_type total_size_;
    std::vector<size_type> perm_;
    std::vector<size_type> jdiag_;
    std::vector<size_type> col_ind_;

  }; // end of class Jagged_diagonal_sparsity::Impl

} // end of namespace sparkit::data::detail
