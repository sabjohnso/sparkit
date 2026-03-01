//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/assembly_tree.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  Assembly_tree
  build_assembly_tree(
    Supernode_partition const& partition, std::span<size_type const> parent) {
    auto ns = partition.n_supernodes;

    // snode_parent[s] = membership[parent[last_col_of_s]], or -1 if root
    std::vector<size_type> snode_parent(static_cast<std::size_t>(ns), -1);

    for (size_type s = 0; s < ns; ++s) {
      auto last_col =
        partition.snode_start[static_cast<std::size_t>(s + 1)] - 1;
      auto col_parent = parent[static_cast<std::size_t>(last_col)];
      if (col_parent != -1) {
        snode_parent[static_cast<std::size_t>(s)] =
          partition.membership[static_cast<std::size_t>(col_parent)];
      }
    }

    // Build children lists by inverting snode_parent
    std::vector<std::vector<size_type>> snode_children(
      static_cast<std::size_t>(ns));
    for (size_type s = 0; s < ns; ++s) {
      auto p = snode_parent[static_cast<std::size_t>(s)];
      if (p != -1) { snode_children[static_cast<std::size_t>(p)].push_back(s); }
    }

    // Postorder via iterative DFS (same algorithm as tree_postorder)
    std::vector<size_type> head(static_cast<std::size_t>(ns), -1);
    std::vector<size_type> next(static_cast<std::size_t>(ns), -1);

    for (size_type s = 0; s < ns; ++s) {
      auto p = snode_parent[static_cast<std::size_t>(s)];
      if (p != -1) {
        next[static_cast<std::size_t>(s)] = head[static_cast<std::size_t>(p)];
        head[static_cast<std::size_t>(p)] = s;
      }
    }

    std::vector<size_type> postorder;
    postorder.reserve(static_cast<std::size_t>(ns));
    std::vector<size_type> stack;

    for (size_type root = 0; root < ns; ++root) {
      if (snode_parent[static_cast<std::size_t>(root)] != -1) { continue; }

      stack.push_back(root);
      while (!stack.empty()) {
        auto node = stack.back();
        auto child = head[static_cast<std::size_t>(node)];

        if (child != -1) {
          head[static_cast<std::size_t>(node)] =
            next[static_cast<std::size_t>(child)];
          stack.push_back(child);
        } else {
          stack.pop_back();
          postorder.push_back(node);
        }
      }
    }

    return Assembly_tree{
      std::move(snode_parent), std::move(snode_children), std::move(postorder)};
  }

} // end of namespace sparkit::data::detail
