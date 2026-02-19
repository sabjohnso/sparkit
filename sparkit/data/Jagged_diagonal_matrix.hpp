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
#include <sparkit/data/Jagged_diagonal_sparsity.hpp>
#include <sparkit/data/Entry.hpp>

namespace sparkit::data::detail {

  template<typename T = config::value_type>
  class Jagged_diagonal_matrix final
  {
  public:
    using size_type = config::size_type;

    /**
     * @brief Construct from sparsity and values in jagged diagonal order.
     *
     * Values layout matches col_ind: organized by jagged diagonals.
     */
    Jagged_diagonal_matrix(
      Jagged_diagonal_sparsity sparsity,
      std::vector<T> values)
      : sparsity_(std::move(sparsity))
      , values_(std::move(values))
    {}

    Jagged_diagonal_matrix(
      Shape shape,
      std::initializer_list<Entry<T>> const& input)
      : Jagged_diagonal_matrix(from_entries(shape, input))
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
      auto pm = sparsity_.perm();
      auto jd = sparsity_.jdiag();
      auto ci = sparsity_.col_ind();
      auto num_jdiags = std::ssize(jd) - 1;

      // Find position of row in perm
      size_type pos = -1;
      for (std::size_t i = 0; i < pm.size(); ++i) {
        if (pm[i] == row) {
          pos = static_cast<size_type>(i);
          break;
        }
      }
      if (pos == -1) return T{0};

      // Scan jagged diagonals
      for (size_type k = 0; k < num_jdiags; ++k) {
        auto width = jd[k + 1] - jd[k];
        if (pos >= width) break;
        if (ci[jd[k] + pos] == col) {
          return values_[static_cast<std::size_t>(jd[k] + pos)];
        }
      }
      return T{0};
    }

    std::span<T const>
    values() const
    {
      return {values_.data(), values_.size()};
    }

    Jagged_diagonal_sparsity const&
    sparsity() const
    {
      return sparsity_;
    }

  private:

    static
    Jagged_diagonal_matrix
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

      Jagged_diagonal_sparsity sparsity{shape, indices.begin(), indices.end()};

      // Build per-row value lists (sorted by column)
      auto nrow = shape.row();
      std::vector<std::vector<T>> row_vals(static_cast<std::size_t>(nrow));
      for (auto const& entry : sorted) {
        row_vals[static_cast<std::size_t>(entry.index.row())].push_back(entry.value);
      }

      // Fill values in jagged diagonal order
      auto pm = sparsity.perm();
      auto jd = sparsity.jdiag();
      auto num_jdiags = std::ssize(jd) - 1;

      std::vector<T> values(static_cast<std::size_t>(sparsity.size()), T{0});

      for (size_type k = 0; k < num_jdiags; ++k) {
        auto width = jd[k + 1] - jd[k];
        for (size_type i = 0; i < width; ++i) {
          auto orig_row = pm[i];
          values[static_cast<std::size_t>(jd[k] + i)]
            = row_vals[static_cast<std::size_t>(orig_row)][static_cast<std::size_t>(k)];
        }
      }

      return Jagged_diagonal_matrix{std::move(sparsity), std::move(values)};
    }

    Jagged_diagonal_sparsity sparsity_;
    std::vector<T> values_;

  }; // end of class Jagged_diagonal_matrix

} // end of namespace sparkit::data::detail
