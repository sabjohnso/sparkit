#include <sparkit/data/Symmetric_block_sparse_row_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Symmetric_block_sparse_row_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Symmetric_block_sparse_row_sparsity::Symmetric_block_sparse_row_sparsity(
    Shape shape,
    size_type block_rows,
    size_type block_cols,
    std::vector<Index> indices)
      : pimpl(new Impl(shape, block_rows, block_cols, std::move(indices))) {}

  Symmetric_block_sparse_row_sparsity::Symmetric_block_sparse_row_sparsity(
    Shape shape,
    size_type block_rows,
    size_type block_cols,
    std::initializer_list<Index> const& input)
      : Symmetric_block_sparse_row_sparsity(
          shape,
          block_rows,
          block_cols,
          std::vector<Index>(begin(input), end(input))) {}

  Symmetric_block_sparse_row_sparsity::Symmetric_block_sparse_row_sparsity(
    Symmetric_block_sparse_row_sparsity const& input)
      : pimpl(new Impl(*input.pimpl)) {}

  Symmetric_block_sparse_row_sparsity::Symmetric_block_sparse_row_sparsity(
    Symmetric_block_sparse_row_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Symmetric_block_sparse_row_sparsity::~Symmetric_block_sparse_row_sparsity() {
    delete pimpl;
  }

  Symmetric_block_sparse_row_sparsity&
  Symmetric_block_sparse_row_sparsity::operator=(
    Symmetric_block_sparse_row_sparsity const& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Symmetric_block_sparse_row_sparsity&
  Symmetric_block_sparse_row_sparsity::operator=(
    Symmetric_block_sparse_row_sparsity&& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Symmetric_block_sparse_row_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }
  Shape
  Symmetric_block_sparse_row_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }
  size_type
  Symmetric_block_sparse_row_sparsity::block_rows() const {
    assert(pimpl);
    return pimpl->block_rows();
  }
  size_type
  Symmetric_block_sparse_row_sparsity::block_cols() const {
    assert(pimpl);
    return pimpl->block_cols();
  }
  size_type
  Symmetric_block_sparse_row_sparsity::num_block_rows() const {
    assert(pimpl);
    return pimpl->num_block_rows();
  }
  size_type
  Symmetric_block_sparse_row_sparsity::num_block_cols() const {
    assert(pimpl);
    return pimpl->num_block_cols();
  }
  size_type
  Symmetric_block_sparse_row_sparsity::num_blocks() const {
    assert(pimpl);
    return pimpl->num_blocks();
  }

  std::span<size_type const>
  Symmetric_block_sparse_row_sparsity::row_ptr() const {
    assert(pimpl);
    return pimpl->row_ptr();
  }

  std::span<size_type const>
  Symmetric_block_sparse_row_sparsity::col_ind() const {
    assert(pimpl);
    return pimpl->col_ind();
  }

} // end of namespace sparkit::data::detail
