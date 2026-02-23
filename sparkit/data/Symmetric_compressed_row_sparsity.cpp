#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Symmetric_compressed_row_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Symmetric_compressed_row_sparsity::Symmetric_compressed_row_sparsity(
    Shape shape, std::vector<Index> indices)
      : pimpl(new Impl(shape, std::move(indices))) {}

  Symmetric_compressed_row_sparsity::Symmetric_compressed_row_sparsity(
    Shape shape, std::initializer_list<Index> const& input)
      : Symmetric_compressed_row_sparsity(
          shape, std::vector<Index>(begin(input), end(input))) {}

  Symmetric_compressed_row_sparsity::Symmetric_compressed_row_sparsity(
    Symmetric_compressed_row_sparsity const& input)
      : pimpl(new Impl(*input.pimpl)) {}

  Symmetric_compressed_row_sparsity::Symmetric_compressed_row_sparsity(
    Symmetric_compressed_row_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Symmetric_compressed_row_sparsity::~Symmetric_compressed_row_sparsity() {
    delete pimpl;
  }

  Symmetric_compressed_row_sparsity&
  Symmetric_compressed_row_sparsity::operator=(
    Symmetric_compressed_row_sparsity const& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Symmetric_compressed_row_sparsity&
  Symmetric_compressed_row_sparsity::operator=(
    Symmetric_compressed_row_sparsity&& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Symmetric_compressed_row_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Symmetric_compressed_row_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }

  std::span<size_type const>
  Symmetric_compressed_row_sparsity::row_ptr() const {
    assert(pimpl);
    return pimpl->row_ptr();
  }

  std::span<size_type const>
  Symmetric_compressed_row_sparsity::col_ind() const {
    assert(pimpl);
    return pimpl->col_ind();
  }

} // end of namespace sparkit::data::detail
