#pragma once

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::data::detail {

  struct Supernode_partition {
    std::vector<config::size_type> snode_start; // length n_supernodes + 1
    std::vector<config::size_type> membership; // length n (column -> supernode)
    config::size_type n_supernodes;
  };

  Supernode_partition
  find_supernodes(
    Compressed_row_sparsity const& L_pattern,
    std::span<config::size_type const> parent);

} // end of namespace sparkit::data::detail
