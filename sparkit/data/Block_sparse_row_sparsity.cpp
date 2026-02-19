#include <sparkit/data/Block_sparse_row_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Block_sparse_row_sparsity_Impl.hpp>

namespace sparkit::data::detail
{

  Block_sparse_row_sparsity::Block_sparse_row_sparsity(
    Shape shape,
    size_type block_rows,
    size_type block_cols,
    std::vector<Index> indices)
    : pimpl(new Impl(shape, block_rows, block_cols, std::move(indices)))
  {}

  Block_sparse_row_sparsity::Block_sparse_row_sparsity(
    Shape shape,
    size_type block_rows,
    size_type block_cols,
    std::initializer_list<Index> const& input)
    : Block_sparse_row_sparsity(shape, block_rows, block_cols,
        std::vector<Index>(begin(input), end(input)))
  {}

  Block_sparse_row_sparsity::Block_sparse_row_sparsity(
    Block_sparse_row_sparsity const& input)
    : pimpl(new Impl(*input.pimpl))
  {}

  Block_sparse_row_sparsity::Block_sparse_row_sparsity(
    Block_sparse_row_sparsity&& input)
    : pimpl(input.pimpl)
  {
    input.pimpl = nullptr;
  }

  Block_sparse_row_sparsity::~Block_sparse_row_sparsity()
  {
    delete pimpl;
  }

  Block_sparse_row_sparsity&
  Block_sparse_row_sparsity::operator=(Block_sparse_row_sparsity const& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Block_sparse_row_sparsity&
  Block_sparse_row_sparsity::operator=(Block_sparse_row_sparsity&& input)
  {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type Block_sparse_row_sparsity::size() const { assert(pimpl); return pimpl->size(); }
  Shape Block_sparse_row_sparsity::shape() const { assert(pimpl); return pimpl->shape(); }
  size_type Block_sparse_row_sparsity::block_rows() const { assert(pimpl); return pimpl->block_rows(); }
  size_type Block_sparse_row_sparsity::block_cols() const { assert(pimpl); return pimpl->block_cols(); }
  size_type Block_sparse_row_sparsity::num_block_rows() const { assert(pimpl); return pimpl->num_block_rows(); }
  size_type Block_sparse_row_sparsity::num_block_cols() const { assert(pimpl); return pimpl->num_block_cols(); }
  size_type Block_sparse_row_sparsity::num_blocks() const { assert(pimpl); return pimpl->num_blocks(); }

  std::span<size_type const>
  Block_sparse_row_sparsity::row_ptr() const { assert(pimpl); return pimpl->row_ptr(); }

  std::span<size_type const>
  Block_sparse_row_sparsity::col_ind() const { assert(pimpl); return pimpl->col_ind(); }

} // end of namespace sparkit::data::detail
