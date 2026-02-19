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
#include <sparkit/data/Ellpack_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  class Ellpack_matrix final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from sparsity and padded values array.
     *
     * Values layout matches col_ind: nrow x max_nnz_per_row, row-major.
     */
    Ellpack_matrix(
      Ellpack_sparsity sparsity,
      std::vector<T> values)
      : sparsity_(std::move(sparsity))
      , values_(std::move(values))
    {}

    Ellpack_matrix(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
      : Ellpack_matrix(from_entries(shape, input))
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

    T
    operator()(size_type row, size_type col) const
    {
      auto max_nnz = sparsity_.max_nnz_per_row();
      if (max_nnz == 0) return T{0};

      auto ci = sparsity_.col_ind();
      auto base = static_cast<std::size_t>(row * max_nnz);

      for (size_type k = 0; k < max_nnz; ++k) {
        auto c = ci[base + static_cast<std::size_t>(k)];
        if (c == -1) break;
        if (c == col) {
          return values_[base + static_cast<std::size_t>(k)];
        }
      }
      return T{0};
    }

    Ellpack_sparsity const&
    sparsity() const
    {
      return sparsity_;
    }

  private:

    static
    Ellpack_matrix
    from_entries(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
    {
      std::vector<Entry<T>> sorted(input.begin(), input.end());

      auto by_row_col = [](auto const& a, auto const& b) {
        return a.index.row() < b.index.row()
          || (a.index.row() == b.index.row()
              && a.index.column() < b.index.column());
      };
      std::sort(sorted.begin(), sorted.end(), by_row_col);

      auto same_index = [](auto const& a, auto const& b) {
        return a.index == b.index;
      };
      sorted.erase(
        std::unique(sorted.begin(), sorted.end(), same_index),
        sorted.end());

      std::vector<Index> indices;
      indices.reserve(sorted.size());
      for (auto const& e : sorted) {
        indices.push_back(e.index);
      }

      Ellpack_sparsity sparsity{shape, indices.begin(), indices.end()};
      auto max_nnz = sparsity.max_nnz_per_row();
      auto nrow = shape.row();

      // Allocate padded values
      std::vector<T> values(
        static_cast<std::size_t>(nrow * max_nnz), T{0});

      // Fill values at correct positions
      auto ci = sparsity.col_ind();
      for (auto const& entry : sorted) {
        auto base = static_cast<std::size_t>(entry.index.row() * max_nnz);
        for (size_type k = 0; k < max_nnz; ++k) {
          if (ci[base + static_cast<std::size_t>(k)] == entry.index.column()) {
            values[base + static_cast<std::size_t>(k)] = entry.value;
            break;
          }
        }
      }

      return Ellpack_matrix{std::move(sparsity), std::move(values)};
    }

    Ellpack_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Ellpack_matrix

} // end of namespace sparkit::data::detail
