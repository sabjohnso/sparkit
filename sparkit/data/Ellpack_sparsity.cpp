#include <sparkit/data/Ellpack_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Ellpack_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Ellpack_sparsity::Ellpack_sparsity(Shape shape, std::vector<Index> indices)
      : pimpl(new Impl(shape, std::move(indices))) {}

  Ellpack_sparsity::Ellpack_sparsity(Shape shape,
                                     std::initializer_list<Index> const& input)
      : Ellpack_sparsity(shape, std::vector<Index>(begin(input), end(input))) {}

  Ellpack_sparsity::Ellpack_sparsity(Ellpack_sparsity const& input)
      : pimpl(new Impl(*input.pimpl)) {}

  Ellpack_sparsity::Ellpack_sparsity(Ellpack_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Ellpack_sparsity::~Ellpack_sparsity() { delete pimpl; }

  Ellpack_sparsity&
  Ellpack_sparsity::operator=(Ellpack_sparsity const& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Ellpack_sparsity&
  Ellpack_sparsity::operator=(Ellpack_sparsity&& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Ellpack_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Ellpack_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }

  size_type
  Ellpack_sparsity::max_nnz_per_row() const {
    assert(pimpl);
    return pimpl->max_nnz_per_row();
  }

  std::span<size_type const>
  Ellpack_sparsity::col_ind() const {
    assert(pimpl);
    return pimpl->col_ind();
  }

} // end of namespace sparkit::data::detail
