#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <initializer_list>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  template <typename T = config::value_type>
  class Symmetric_compressed_row_matrix final {
  public:
    using size_type = config::size_type;

    Symmetric_compressed_row_matrix(Symmetric_compressed_row_sparsity sparsity,
                                    std::vector<T> values)
        : sparsity_(std::move(sparsity)), values_(std::move(values)) {}

    Symmetric_compressed_row_matrix(
        Shape shape, std::initializer_list<Entry<T>> const& input)
        : Symmetric_compressed_row_matrix(from_entries(shape, input)) {}

    size_type
    size() const {
      return sparsity_.size();
    }

    Shape
    shape() const {
      return sparsity_.shape();
    }

    std::span<size_type const>
    row_ptr() const {
      return sparsity_.row_ptr();
    }

    std::span<size_type const>
    col_ind() const {
      return sparsity_.col_ind();
    }

    std::span<T const>
    values() const {
      return {values_.data(), values_.size()};
    }

    T
    operator()(size_type row, size_type col) const {
      // Normalize to lower triangle
      auto r = std::max(row, col);
      auto c = std::min(row, col);

      auto rp = sparsity_.row_ptr();
      auto ci = sparsity_.col_ind();
      auto begin = ci.begin() + rp[r];
      auto end = ci.begin() + rp[r + 1];
      auto it = std::lower_bound(begin, end, c);
      if (it != end && *it == c) {
        return values_[static_cast<std::size_t>(std::distance(ci.begin(), it))];
      }
      return T{0};
    }

    Symmetric_compressed_row_sparsity const&
    sparsity() const {
      return sparsity_;
    }

  private:
    static Symmetric_compressed_row_matrix
    from_entries(Shape shape, std::initializer_list<Entry<T>> const& input) {
      // Normalize entries to lower triangle
      std::vector<Entry<T>> sorted;
      sorted.reserve(input.size());
      for (auto const& e : input) {
        auto r = std::max(e.index.row(), e.index.column());
        auto c = std::min(e.index.row(), e.index.column());
        sorted.push_back({Index{r, c}, e.value});
      }

      auto by_row_col = [](auto const& a, auto const& b) {
        return a.index.row() < b.index.row() ||
               (a.index.row() == b.index.row() &&
                a.index.column() < b.index.column());
      };
      std::sort(sorted.begin(), sorted.end(), by_row_col);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(std::unique(sorted.begin(), sorted.end(), same_index),
                   sorted.end());

      std::vector<Index> indices;
      std::vector<T> values;
      indices.reserve(sorted.size());
      values.reserve(sorted.size());

      for (auto const& entry : sorted) {
        indices.push_back(entry.index);
        values.push_back(entry.value);
      }

      return Symmetric_compressed_row_matrix{
          Symmetric_compressed_row_sparsity{shape, indices.begin(),
                                            indices.end()},
          std::move(values)};
    }

    Symmetric_compressed_row_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Symmetric_compressed_row_matrix

} // end of namespace sparkit::data::detail
