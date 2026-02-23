#pragma once

//
// ... Standard header files
//
#include <initializer_list>
#include <vector>
//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Index.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data::detail {

  class Coordinate_sparsity {
  public:
    using size_type = config::size_type;

    template <typename Iter>
    Coordinate_sparsity(Shape shape, Iter first, Iter last)
        : pimpl(nullptr) {
      init(shape);
      std::for_each(first, last, [this](Index index) { add(index); });
    }

    Coordinate_sparsity(Shape shape, std::initializer_list<Index> const& input);
    Coordinate_sparsity(Coordinate_sparsity const& input);
    Coordinate_sparsity(Coordinate_sparsity&& input);

    Coordinate_sparsity&
    operator=(Coordinate_sparsity const& input);

    Coordinate_sparsity&
    operator=(Coordinate_sparsity&& input);

    void
    add(Index index);

    void
    remove(Index index);

    size_type
    size() const;

    Shape
    shape() const;

    std::vector<Index>
    indices() const;

    virtual ~Coordinate_sparsity();

  private:
    void
    init(Shape shape);

    class Impl;
    Impl* pimpl;
  }; // end of class Coordinate_sparsity

} // namespace sparkit::data::detail
