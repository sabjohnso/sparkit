#include <sparkit/data/Coordinate_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Coordinate_sparsity::Impl::Impl(Shape shape)
    : shape_(shape)
    , nonzeros_{}
  {}

  Coordinate_sparsity::Impl::Impl(Impl const& input)
    : shape_(input.shape_)
  {
    std::copy(
      begin(input.nonzeros_),
      end(input.nonzeros_),
      std::inserter(nonzeros_, end(nonzeros_)));
  }


  Coordinate_sparsity::Impl&
  Coordinate_sparsity::Impl::operator=(const Coordinate_sparsity::Impl& input){
    shape_ = input.shape_;
    nonzeros_.clear();
    std::copy(begin(input.nonzeros_), end(input.nonzeros_),
              std::inserter(nonzeros_, std::end(nonzeros_)));
    return *this;
  }

  void
  Coordinate_sparsity::Impl::add(Index index)
  {
    nonzeros_.insert(index);
  }

  void
  Coordinate_sparsity::Impl::remove(Index index)
  {
    nonzeros_.erase(index);
  }

  Shape
  Coordinate_sparsity::Impl::shape() const { return shape_; }

  size_type
  Coordinate_sparsity::Impl::size() const { return std::size(nonzeros_); }

  std::vector<Index>
  Coordinate_sparsity::Impl::indices() const
  {
    return {begin(nonzeros_), end(nonzeros_)};
  }

} // end of namespace sparkit::data::detail
