#include <sparkit/data/Coordinate_sparsity.hpp>

//
// ... Standard header files
//
#include <cassert>

//
// ... sparkit header files
//
#include <sparkit/data/Coordinate_sparsity_Impl.hpp>

namespace sparkit::data::detail {

  Coordinate_sparsity::Coordinate_sparsity(
    Shape shape, std::initializer_list<Index> const& input)
      : pimpl(nullptr) {
    init(shape);
    assert(pimpl);
    std::for_each(
      begin(input), end(input), [this](Index index) { add(index); });
  }

  Coordinate_sparsity::Coordinate_sparsity(const Coordinate_sparsity& input)
      : pimpl(nullptr) {
    init(input.shape());
    *pimpl = *input.pimpl;
  }

  Coordinate_sparsity::Coordinate_sparsity(Coordinate_sparsity&& input)
      : pimpl(nullptr) {
    pimpl = input.pimpl;
    input.pimpl = nullptr;
  }

  Coordinate_sparsity::~Coordinate_sparsity() {
    if (pimpl) { delete pimpl; }
  }

  Coordinate_sparsity&
  Coordinate_sparsity::operator=(const Coordinate_sparsity& input) {
    init(input.shape());
    *pimpl = *input.pimpl;
    return *this;
  }

  Coordinate_sparsity&
  Coordinate_sparsity::operator=(Coordinate_sparsity&& input) {
    if (pimpl) { delete pimpl; }
    pimpl = input.pimpl;
    input.pimpl = nullptr;
    return *this;
  }

  void
  Coordinate_sparsity::init(Shape shape) {
    if (pimpl) { delete pimpl; }

    pimpl = new Impl(shape);
  }

  void
  Coordinate_sparsity::add(Index index) {
    assert(pimpl);
    pimpl->add(index);
  }

  void
  Coordinate_sparsity::remove(Index index) {
    assert(pimpl);
    pimpl->remove(index);
  }

  size_type
  Coordinate_sparsity::size() const {
    return pimpl->size();
  }

  Shape
  Coordinate_sparsity::shape() const {
    return pimpl->shape();
  }

  std::vector<Index>
  Coordinate_sparsity::indices() const {
    assert(pimpl);
    return pimpl->indices();
  }

} // end of namespace sparkit::data::detail
