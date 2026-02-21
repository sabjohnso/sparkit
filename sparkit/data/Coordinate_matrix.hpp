#pragma once

//
// ... Standard header files
//
#include <initializer_list>
#include <unordered_map>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Coordinate_sparsity.hpp>
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/Shape.hpp>

namespace sparkit::data::detail {

  template <typename T = config::value_type>
  class Coordinate_matrix final {
  public:
    using size_type = config::size_type;

    explicit Coordinate_matrix(Shape shape) : shape_(shape) {}

    Coordinate_matrix(Shape shape, std::initializer_list<Entry<T>> const& input)
        : shape_(shape) {
      for (auto const& entry : input) {
        entries_.insert_or_assign(entry.index, entry.value);
      }
    }

    template <typename Iter>
    Coordinate_matrix(Shape shape, Iter first, Iter last) : shape_(shape) {
      for (auto it = first; it != last; ++it) {
        entries_.insert_or_assign(it->index, it->value);
      }
    }

    template <typename F>
    Coordinate_matrix(Coordinate_sparsity sparsity, F f)
        : shape_(sparsity.shape()) {
      for (auto const& index : sparsity.indices()) {
        entries_.insert_or_assign(index, f(index.row(), index.column()));
      }
    }

    void
    add(Index index, T value) {
      entries_.insert_or_assign(index, value);
    }

    void
    remove(Index index) {
      entries_.erase(index);
    }

    size_type
    size() const {
      return static_cast<size_type>(entries_.size());
    }

    Shape
    shape() const {
      return shape_;
    }

    std::vector<std::pair<Index, T>>
    entries() const {
      return {entries_.begin(), entries_.end()};
    }

    T
    operator()(size_type row, size_type col) const {
      auto it = entries_.find(Index{row, col});
      if (it != entries_.end()) { return it->second; }
      return T{0};
    }

    Coordinate_sparsity
    sparsity() const {
      std::vector<Index> indices;
      indices.reserve(entries_.size());
      for (auto const& [index, value] : entries_) {
        indices.push_back(index);
      }
      return Coordinate_sparsity(shape_, indices.begin(), indices.end());
    }

  private:
    Shape shape_;
    std::unordered_map<Index, T, IndexHash> entries_;

  }; // end of class Coordinate_matrix

} // end of namespace sparkit::data::detail
