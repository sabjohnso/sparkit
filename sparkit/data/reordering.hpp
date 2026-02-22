#pragma once

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  Compressed_row_sparsity
  symmetrize_pattern(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  adjacency_degree(Compressed_row_sparsity const& sp);

  config::size_type
  pseudo_peripheral_node(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  reverse_cuthill_mckee(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  approximate_minimum_degree(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  column_approximate_minimum_degree(Compressed_row_sparsity const& sp);

  std::vector<config::size_type>
  nested_dissection(Compressed_row_sparsity const& sp);

} // end of namespace sparkit::data::detail
