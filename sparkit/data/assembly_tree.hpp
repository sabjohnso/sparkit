#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/supernode.hpp>

namespace sparkit::data::detail {

  struct Assembly_tree {
    std::vector<config::size_type> snode_parent;
    std::vector<std::vector<config::size_type>> snode_children;
    std::vector<config::size_type> postorder;
  };

  Assembly_tree
  build_assembly_tree(
    Supernode_partition const& partition,
    std::span<config::size_type const> parent);

} // end of namespace sparkit::data::detail
