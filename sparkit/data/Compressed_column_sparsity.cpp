#include <sparkit/data/Compressed_column_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_column_sparsity_Impl.hpp>

namespace sparkit::data::detail
{

  Compressed_column_sparsity::Compressed_column_sparsity(
    Shape shape,
    std::vector<Index> indices)
    : pimpl(new Impl(shape, std::move(indices)))
  {}

  Compressed_column_sparsity::Compressed_column_sparsity(
    Shape shape,
    std::initializer_list<Index> const& input)
    : Compressed_column_sparsity(shape, std::vector<Index>(begin(input), end(input)))
  {}

  Compressed_column_sparsity::Compressed_column_sparsity(
    Compressed_column_sparsity const& input)
    : pimpl(new Impl(*input.pimpl))
  {}

  Compressed_column_sparsity::Compressed_column_sparsity(
    Compressed_column_sparsity&& input)
    : pimpl(input.pimpl)
  {
    input.pimpl = nullptr;
  }

  Compressed_column_sparsity::~Compressed_column_sparsity()
  {
    delete pimpl;
  }

  Compressed_column_sparsity&
  Compressed_column_sparsity::operator=(Compressed_column_sparsity const& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Compressed_column_sparsity&
  Compressed_column_sparsity::operator=(Compressed_column_sparsity&& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Compressed_column_sparsity::size() const
  {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Compressed_column_sparsity::shape() const
  {
    assert(pimpl);
    return pimpl->shape();
  }

  std::span<size_type const>
  Compressed_column_sparsity::col_ptr() const
  {
    assert(pimpl);
    return pimpl->col_ptr();
  }

  std::span<size_type const>
  Compressed_column_sparsity::row_ind() const
  {
    assert(pimpl);
    return pimpl->row_ind();
  }

} // end of namespace sparkit::data::detail
