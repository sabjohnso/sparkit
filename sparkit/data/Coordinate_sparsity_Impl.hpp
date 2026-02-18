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
#include <sparkit/data/Coordinate_sparsity.hpp>

namespace sparkit::data::detail{

  class Coordinate_sparsity::Impl
  {
  public:
    Impl(Shape shape);

    Impl(const Impl& input);

    Impl&
    operator =(const Impl& input);

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

    void
    copy_nonzeros(const Impl& input);


    Shape shape_;
    std::unordered_set<Index, IndexHash> nonzeros_;
  }; // end of class Coordinate_sparsity::Impl

} // end of namespace sparkit::data::detail
