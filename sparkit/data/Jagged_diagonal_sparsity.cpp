#include <sparkit/data/Jagged_diagonal_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Jagged_diagonal_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Jagged_diagonal_sparsity::Jagged_diagonal_sparsity(
    Shape shape, std::vector<Index> indices)
      : pimpl(new Impl(shape, std::move(indices))) {}

  Jagged_diagonal_sparsity::Jagged_diagonal_sparsity(
    Shape shape, std::initializer_list<Index> const& input)
      : Jagged_diagonal_sparsity(
          shape, std::vector<Index>(begin(input), end(input))) {}

  Jagged_diagonal_sparsity::Jagged_diagonal_sparsity(
    Jagged_diagonal_sparsity const& input)
      : pimpl(new Impl(*input.pimpl)) {}

  Jagged_diagonal_sparsity::Jagged_diagonal_sparsity(
    Jagged_diagonal_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Jagged_diagonal_sparsity::~Jagged_diagonal_sparsity() { delete pimpl; }

  Jagged_diagonal_sparsity&
  Jagged_diagonal_sparsity::operator=(Jagged_diagonal_sparsity const& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = new Impl(*input.pimpl);
    }
    return *this;
  }

  Jagged_diagonal_sparsity&
  Jagged_diagonal_sparsity::operator=(Jagged_diagonal_sparsity&& input) {
    if (this != &input) {
      delete pimpl;
      pimpl = input.pimpl;
      input.pimpl = nullptr;
    }
    return *this;
  }

  size_type
  Jagged_diagonal_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Jagged_diagonal_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::perm() const {
    assert(pimpl);
    return pimpl->perm();
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::jdiag() const {
    assert(pimpl);
    return pimpl->jdiag();
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::col_ind() const {
    assert(pimpl);
    return pimpl->col_ind();
  }

} // end of namespace sparkit::data::detail
