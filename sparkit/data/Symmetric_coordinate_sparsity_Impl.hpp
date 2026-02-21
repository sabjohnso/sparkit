#pragma once

//
// ... Standard header files
//
#include <unordered_set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Index.hpp>
#include <sparkit/data/Symmetric_coordinate_sparsity.hpp>

namespace sparkit::data::detail {

  class Symmetric_coordinate_sparsity::Impl {
  public:
    Impl(Shape shape);

    Impl(Impl const& input);

    Impl&
    operator=(Impl const& input);

    void
    add(Index index);

    void
    remove(Index index);

    Shape
    shape() const;

    size_type
    size() const;

    std::vector<Index>
    indices() const;

  private:
    Shape shape_;
    std::unordered_set<Index, IndexHash> nonzeros_;

  }; // end of class Symmetric_coordinate_sparsity::Impl

} // end of namespace sparkit::data::detail
