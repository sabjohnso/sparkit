#include <sparkit/data/Symmetric_coordinate_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>

//
// ... sparkit header files
//
#include <sparkit/data/Symmetric_coordinate_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Symmetric_coordinate_sparsity::Symmetric_coordinate_sparsity(
      Shape shape, std::initializer_list<Index> const& input)
      : pimpl(nullptr) {
    init(shape);
    assert(pimpl);
    std::for_each(begin(input), end(input),
                  [this](Index index) { add(index); });
  }

  Symmetric_coordinate_sparsity::Symmetric_coordinate_sparsity(
      Symmetric_coordinate_sparsity const& input)
      : pimpl(nullptr) {
    init(input.shape());
    *pimpl = *input.pimpl;
  }

  Symmetric_coordinate_sparsity::Symmetric_coordinate_sparsity(
      Symmetric_coordinate_sparsity&& input)
      : pimpl(input.pimpl) {
    input.pimpl = nullptr;
  }

  Symmetric_coordinate_sparsity::~Symmetric_coordinate_sparsity() {
    delete pimpl;
  }

  Symmetric_coordinate_sparsity&
  Symmetric_coordinate_sparsity::operator=(
      Symmetric_coordinate_sparsity const& input) {
    init(input.shape());
    *pimpl = *input.pimpl;
    return *this;
  }

  Symmetric_coordinate_sparsity&
  Symmetric_coordinate_sparsity::operator=(
      Symmetric_coordinate_sparsity&& input) {
    delete pimpl;
    pimpl = input.pimpl;
    input.pimpl = nullptr;
    return *this;
  }

  void
  Symmetric_coordinate_sparsity::init(Shape shape) {
    delete pimpl;
    pimpl = new Impl(shape);
  }

  void
  Symmetric_coordinate_sparsity::add(Index index) {
    assert(pimpl);
    pimpl->add(index);
  }

  void
  Symmetric_coordinate_sparsity::remove(Index index) {
    assert(pimpl);
    pimpl->remove(index);
  }

  size_type
  Symmetric_coordinate_sparsity::size() const {
    assert(pimpl);
    return pimpl->size();
  }

  Shape
  Symmetric_coordinate_sparsity::shape() const {
    assert(pimpl);
    return pimpl->shape();
  }

  std::vector<Index>
  Symmetric_coordinate_sparsity::indices() const {
    assert(pimpl);
    return pimpl->indices();
  }

} // end of namespace sparkit::data::detail
