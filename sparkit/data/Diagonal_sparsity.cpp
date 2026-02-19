#include <sparkit/data/Diagonal_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Diagonal_sparsity_Impl.hpp>

namespace sparkit::data::detail
{

  Diagonal_sparsity::Diagonal_sparsity(
    Shape shape,
    std::vector<size_type> offsets)
    : pimpl(new Impl(shape, std::move(offsets)))
  {}

  Diagonal_sparsity::Diagonal_sparsity(
    Shape shape,
    std::initializer_list<size_type> const& offsets)
    : Diagonal_sparsity(shape, std::vector<size_type>(offsets.begin(), offsets.end()))
  {}

  Diagonal_sparsity::Diagonal_sparsity(
    Shape shape,
    std::vector<Index> indices)
    : pimpl(nullptr)
  {
    // Deduce offsets from indices
    std::set<size_type> offset_set;
    for (auto const& idx : indices) {
      offset_set.insert(idx.column() - idx.row());
    }
    std::vector<size_type> offsets(offset_set.begin(), offset_set.end());
    pimpl = new Impl(shape, std::move(offsets));
  }

  Diagonal_sparsity::Diagonal_sparsity(
    Shape shape,
    std::initializer_list<Index> const& input)
    : Diagonal_sparsity(shape, std::vector<Index>(input.begin(), input.end()))
  {}

  Diagonal_sparsity::Diagonal_sparsity(
    Diagonal_sparsity const& input)
    : pimpl(new Impl(*input.pimpl))
  {}

  Diagonal_sparsity::Diagonal_sparsity(
    Diagonal_sparsity&& input)
    : pimpl(input.pimpl)
  {
    input.pimpl = nullptr;
  }

  Diagonal_sparsity::~Diagonal_sparsity()
  {
    delete pimpl;
  }

  Diagonal_sparsity&
  Diagonal_sparsity::operator=(Diagonal_sparsity const& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Diagonal_sparsity&
  Diagonal_sparsity::operator=(Diagonal_sparsity&& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Diagonal_sparsity::size() const
  {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Diagonal_sparsity::shape() const
  {
    assert(pimpl);
    return pimpl->shape();
  }

  std::span<size_type const>
  Diagonal_sparsity::offsets() const
  {
    assert(pimpl);
    return pimpl->offsets();
  }

  size_type
  Diagonal_sparsity::num_diagonals() const
  {
    assert(pimpl);
    return pimpl->num_diagonals();
  }

} // end of namespace sparkit::data::detail
