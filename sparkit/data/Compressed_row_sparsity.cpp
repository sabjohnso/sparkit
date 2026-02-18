#include <sparkit/data/Compressed_row_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity_Impl.hpp>

namespace sparkit::data::detail
{

  Compressed_row_sparsity::Compressed_row_sparsity(
    Shape shape,
    std::vector<Index> indices)
    : pimpl(new Impl(shape, std::move(indices)))
  {}

  Compressed_row_sparsity::Compressed_row_sparsity(
    Shape shape,
    std::initializer_list<Index> const& input)
    : Compressed_row_sparsity(shape, std::vector<Index>(begin(input), end(input)))
  {}

  Compressed_row_sparsity::Compressed_row_sparsity(
    Compressed_row_sparsity const& input)
    : pimpl(new Impl(*input.pimpl))
  {}

  Compressed_row_sparsity::Compressed_row_sparsity(
    Compressed_row_sparsity&& input)
    : pimpl(input.pimpl)
  {
    input.pimpl = nullptr;
  }

  Compressed_row_sparsity::~Compressed_row_sparsity()
  {
    delete pimpl;
  }

  Compressed_row_sparsity&
  Compressed_row_sparsity::operator=(Compressed_row_sparsity const& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Compressed_row_sparsity&
  Compressed_row_sparsity::operator=(Compressed_row_sparsity&& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Compressed_row_sparsity::size() const
  {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Compressed_row_sparsity::shape() const
  {
    assert(pimpl);
    return pimpl->shape();
  }

  std::span<size_type const>
  Compressed_row_sparsity::row_ptr() const
  {
    assert(pimpl);
    return pimpl->row_ptr();
  }

  std::span<size_type const>
  Compressed_row_sparsity::col_ind() const
  {
    assert(pimpl);
    return pimpl->col_ind();
  }

} // end of namespace sparkit::data::detail
