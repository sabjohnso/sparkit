#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Coordinate_sparsity.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  to_compressed_row(Coordinate_sparsity const& coo);

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Coordinate_matrix<T> const& coo)
  {
    auto entries = coo.entries();

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.first.row() < b.first.row()
        || (a.first.row() == b.first.row()
            && a.first.column() < b.first.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& [index, value] : entries) {
      indices.push_back(index);
      values.push_back(value);
    }

    Compressed_row_sparsity sparsity{
      coo.shape(), indices.begin(), indices.end()};

    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

} // end of namespace sparkit::data::detail
