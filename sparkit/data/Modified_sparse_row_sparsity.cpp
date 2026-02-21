#include <sparkit/data/Modified_sparse_row_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Modified_sparse_row_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Modified_sparse_row_sparsity::Modified_sparse_row_sparsity(
      Shape shape, std::vector<Index> indices)
      : pimpl(new Impl(shape, std::move(indices))) {}

  Modified_sparse_row_sparsity::Modified_sparse_row_sparsity(
      Shape shape, std::initializer_list<Index> const& input)
      : Modified_sparse_row_sparsity(
            shape, std::vector<Index>(begin(input), end(input))) {}

  Modified_sparse_row_sparsity::Modified_sparse_row_sparsity(
      Modified_sparse_row_sparsity const& input)
      : pimpl(new Impl(*input.pimpl)) {}

  Modified_sparse_row_sparsity::Modified_sparse_row_sparsity(
      Modified_sparse_row_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Modified_sparse_row_sparsity::~Modified_sparse_row_sparsity() {
    delete pimpl;
  }

  Modified_sparse_row_sparsity&
  Modified_sparse_row_sparsity::operator=(
      Modified_sparse_row_sparsity const& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Modified_sparse_row_sparsity&
  Modified_sparse_row_sparsity::operator=(
      Modified_sparse_row_sparsity&& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Modified_sparse_row_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Modified_sparse_row_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }

  bool
  Modified_sparse_row_sparsity::has_diagonal(size_type i) const {
    assert(pimpl);
    return pimpl->has_diagonal(i);
  }

  size_type
  Modified_sparse_row_sparsity::diagonal_length() const {
    assert(pimpl);
    return pimpl->diagonal_length();
  }

  std::span<size_type const>
  Modified_sparse_row_sparsity::off_diagonal_row_ptr() const {
    assert(pimpl);
    return pimpl->off_diagonal_row_ptr();
  }

  std::span<size_type const>
  Modified_sparse_row_sparsity::off_diagonal_col_ind() const {
    assert(pimpl);
    return pimpl->off_diagonal_col_ind();
  }

} // end of namespace sparkit::data::detail
