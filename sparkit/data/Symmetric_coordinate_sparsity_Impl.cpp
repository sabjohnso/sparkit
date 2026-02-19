#include <sparkit/data/Symmetric_coordinate_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Symmetric_coordinate_sparsity::Impl::Impl(Shape shape)
    : shape_(shape)
    , nonzeros_{}
  {}

  Symmetric_coordinate_sparsity::Impl::Impl(Impl const& input)
    : shape_(input.shape_)
  {
    std::copy(
      begin(input.nonzeros_),
      end(input.nonzeros_),
      std::inserter(nonzeros_, end(nonzeros_)));
  }

  Symmetric_coordinate_sparsity::Impl&
  Symmetric_coordinate_sparsity::Impl::operator=(Impl const& input)
  {
    shape_ = input.shape_;
    nonzeros_.clear();
    std::copy(begin(input.nonzeros_), end(input.nonzeros_),
              std::inserter(nonzeros_, end(nonzeros_)));
    return *this;
  }

  void
  Symmetric_coordinate_sparsity::Impl::add(Index index)
  {
    // Normalize to lower triangle
    if (index.row() < index.column()) {
      index = Index{index.column(), index.row()};
    }
    nonzeros_.insert(index);
  }

  void
  Symmetric_coordinate_sparsity::Impl::remove(Index index)
  {
    // Normalize to lower triangle
    if (index.row() < index.column()) {
      index = Index{index.column(), index.row()};
    }
    nonzeros_.erase(index);
  }

  Shape
  Symmetric_coordinate_sparsity::Impl::shape() const { return shape_; }

  size_type
  Symmetric_coordinate_sparsity::Impl::size() const
  {
    return static_cast<size_type>(nonzeros_.size());
  }

  std::vector<Index>
  Symmetric_coordinate_sparsity::Impl::indices() const
  {
    return {begin(nonzeros_), end(nonzeros_)};
  }

} // end of namespace sparkit::data::detail
