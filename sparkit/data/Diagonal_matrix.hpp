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
#include <sparkit/data/Diagonal_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  class Diagonal_matrix final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from sparsity and flat values array.
     *
     * Values are stored in diagonal order: for each diagonal d (in
     * sorted offset order), values for all valid positions on that
     * diagonal are stored contiguously.
     */
    Diagonal_matrix(
      Diagonal_sparsity sparsity,
      std::vector<T> values)
      : sparsity_(std::move(sparsity))
      , values_(std::move(values))
    {}

    /**
     * @brief Construct from a shape and an entry list.
     */
    Diagonal_matrix(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
      : Diagonal_matrix(from_entries(shape, input))
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
      auto offset = col - row;
      auto off = sparsity_.offsets();

      // Binary search for the offset
      auto it = std::lower_bound(off.begin(), off.end(), offset);
      if (it == off.end() || *it != offset) {
        return T{0};
      }

      // Compute position in values array
      auto diag_idx = static_cast<std::size_t>(std::distance(off.begin(), it));
      size_type pos = 0;
      auto nrow = sparsity_.shape().row();
      auto ncol = sparsity_.shape().column();

      for (std::size_t d = 0; d < diag_idx; ++d) {
        auto o = off[d];
        if (o >= 0) {
          pos += std::min(nrow, ncol - o);
        } else {
          pos += std::min(nrow + o, ncol);
        }
      }

      // Index within this diagonal
      size_type within;
      if (offset >= 0) {
        within = row;  // row is the index along super/main diagonal
      } else {
        within = col;  // col is the index along sub-diagonal
      }

      return values_[static_cast<std::size_t>(pos + within)];
    }

    std::span<T const>
    values() const
    {
      return {values_.data(), values_.size()};
    }

    Diagonal_sparsity const&
    sparsity() const
    {
      return sparsity_;
    }

  private:

    static
    Diagonal_matrix
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

      // Build sparsity from indices
      std::vector<Index> indices;
      indices.reserve(sorted.size());
      for (auto const& entry : sorted) {
        indices.push_back(entry.index);
      }
      Diagonal_sparsity sparsity{shape, indices.begin(), indices.end()};

      // Allocate values array: one slot per valid position on each diagonal
      auto off = sparsity.offsets();
      auto nrow = shape.row();
      auto ncol = shape.column();

      std::vector<T> values(static_cast<std::size_t>(sparsity.size()), T{0});

      // Fill in values at the correct positions
      for (auto const& entry : sorted) {
        auto offset = entry.index.column() - entry.index.row();
        auto it = std::lower_bound(off.begin(), off.end(), offset);
        auto diag_idx = static_cast<std::size_t>(std::distance(off.begin(), it));

        size_type pos = 0;
        for (std::size_t d = 0; d < diag_idx; ++d) {
          auto o = off[d];
          if (o >= 0) {
            pos += std::min(nrow, ncol - o);
          } else {
            pos += std::min(nrow + o, ncol);
          }
        }

        size_type within;
        if (offset >= 0) {
          within = entry.index.row();
        } else {
          within = entry.index.column();
        }

        values[static_cast<std::size_t>(pos + within)] = entry.value;
      }

      return Diagonal_matrix{std::move(sparsity), std::move(values)};
    }

    Diagonal_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Diagonal_matrix

} // end of namespace sparkit::data::detail
