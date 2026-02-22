#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  std::vector<config::size_type>
  elimination_tree(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  tree_postorder(std::span<config::size_type const> parent);

  std::vector<config::size_type>
  cholesky_column_counts(Compressed_row_sparsity const& sp,
                         std::span<config::size_type const> parent);

} // end of namespace sparkit::data::detail
