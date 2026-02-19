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
#include <sparkit/data/Compressed_column_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  class Compressed_column_matrix final
  {
  public:
    using size_type = config::size_type;

    Compressed_column_matrix(
      Compressed_column_sparsity sparsity,
      std::vector<T> values)
      : sparsity_(std::move(sparsity))
      , values_(std::move(values))
    {}

    Compressed_column_matrix(
      Compressed_column_sparsity sparsity,
      std::initializer_list<T> const& values)
      : sparsity_(std::move(sparsity))
      , values_(values)
    {}

    template<typename F>
    Compressed_column_matrix(Compressed_column_sparsity sparsity, F f)
      : sparsity_(std::move(sparsity))
    {
      auto cp = sparsity_.col_ptr();
      auto ri = sparsity_.row_ind();
      values_.resize(static_cast<std::size_t>(sparsity_.size()));

      for (size_type col = 0; col < sparsity_.shape().column(); ++col) {
        for (auto j = cp[col]; j < cp[col + 1]; ++j) {
          values_[static_cast<std::size_t>(j)] = f(ri[j], col);
        }
      }
    }

    Compressed_column_matrix(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
      : Compressed_column_matrix(from_entries(shape, input))
    {}

    size_type
    size() const
    {
      return sparsity_.size();
    }

    Shape
    shape() const
    {
      return sparsity_.shape();
    }

    std::span<size_type const>
    col_ptr() const
    {
      return sparsity_.col_ptr();
    }

    std::span<size_type const>
    row_ind() const
    {
      return sparsity_.row_ind();
    }

    std::span<T const>
    values() const
    {
      return {values_.data(), values_.size()};
    }

    T
    operator()(size_type row, size_type col) const
    {
      auto cp = sparsity_.col_ptr();
      auto ri = sparsity_.row_ind();
      auto begin = ri.begin() + cp[col];
      auto end = ri.begin() + cp[col + 1];
      auto it = std::lower_bound(begin, end, row);
      if (it != end && *it == row) {
        return values_[static_cast<std::size_t>(std::distance(ri.begin(), it))];
      }
      return T{0};
    }

    Compressed_column_sparsity const&
    sparsity() const
    {
      return sparsity_;
    }

  private:

    static
    Compressed_column_matrix
    from_entries(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
    {
      std::vector<Entry<T>> sorted(input.begin(), input.end());

      auto by_col_row = [](auto const& a, auto const& b) {
        return a.index.column() < b.index.column()
          || (a.index.column() == b.index.column()
              && a.index.row() < b.index.row());
      };
      std::sort(sorted.begin(), sorted.end(), by_col_row);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(
        std::unique(sorted.begin(), sorted.end(), same_index),
        sorted.end());

      std::vector<Index> indices;
      std::vector<T> values;
      indices.reserve(sorted.size());
      values.reserve(sorted.size());

      for (auto const& entry : sorted) {
        indices.push_back(entry.index);
        values.push_back(entry.value);
      }

      return Compressed_column_matrix{
        Compressed_column_sparsity{shape, indices.begin(), indices.end()},
        std::move(values)};
    }

    Compressed_column_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Compressed_column_matrix

} // end of namespace sparkit::data::detail
