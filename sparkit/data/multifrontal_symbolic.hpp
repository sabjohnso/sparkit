#pragma once

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/assembly_tree.hpp>
#include <sparkit/data/supernode.hpp>

namespace sparkit::data::detail {

  struct Supernode_map {
    std::vector<config::size_type> row_indices; // all row indices in the front
    config::size_type snode_size;               // columns in supernode
    config::size_type front_size;               // total frontal dimension
  };

  struct Multifrontal_symbolic {
    config::size_type n;
    Supernode_partition partition;
    Assembly_tree tree;
    std::vector<Supernode_map> maps;
    std::vector<std::vector<config::size_type>> relative_maps;
  };

  Multifrontal_symbolic
  multifrontal_analyze(
    Compressed_row_sparsity const& L_pattern,
    Supernode_partition const& partition,
    Assembly_tree const& tree);

} // end of namespace sparkit::data::detail
