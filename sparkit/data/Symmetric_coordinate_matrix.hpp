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
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/Shape.hpp>
#include <sparkit/data/Symmetric_coordinate_sparsity.hpp>

namespace sparkit::data::detail {

  template <typename T = config::value_type>
  class Symmetric_coordinate_matrix final {
  public:
    using size_type = config::size_type;

    explicit Symmetric_coordinate_matrix(Shape shape)
        : shape_(shape) {}

    Symmetric_coordinate_matrix(
      Shape shape, std::initializer_list<Entry<T>> const& input)
        : shape_(shape) {
      for (auto const& entry : input) {
        add(entry.index, entry.value);
      }
    }

    template <typename Iter>
    Symmetric_coordinate_matrix(Shape shape, Iter first, Iter last)
        : shape_(shape) {
      for (auto it = first; it != last; ++it) {
        add(it->index, it->value);
      }
    }

    void
    add(Index index, T value) {
      entries_.insert_or_assign(normalize(index), value);
    }

    void
    remove(Index index) {
      entries_.erase(normalize(index));
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
      auto it = entries_.find(normalize(Index{row, col}));
      if (it != entries_.end()) { return it->second; }
      return T{0};
    }

    Symmetric_coordinate_sparsity
    sparsity() const {
      std::vector<Index> indices;
      indices.reserve(entries_.size());
      for (auto const& [index, value] : entries_) {
        indices.push_back(index);
      }
      return Symmetric_coordinate_sparsity(
        shape_, indices.begin(), indices.end());
    }

  private:
    static Index
    normalize(Index index) {
      if (index.row() < index.column()) {
        return Index{index.column(), index.row()};
      }
      return index;
    }

    Shape shape_;
    std::unordered_map<Index, T, IndexHash> entries_;

  }; // end of class Symmetric_coordinate_matrix

} // end of namespace sparkit::data::detail
