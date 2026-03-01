//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/assembly_tree.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Assembly_tree;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Supernode_partition;

  using sparkit::data::detail::build_assembly_tree;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::find_supernodes;
  using sparkit::data::detail::is_valid_permutation;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // Helper: full symbolic analysis
  struct Analysis {
    Compressed_row_sparsity L;
    std::vector<size_type> parent;
    Supernode_partition partition;
  };

  static Analysis
  full_analyze(Compressed_row_sparsity const& sp) {
    auto parent = elimination_tree(sp);
    auto L = symbolic_cholesky(sp);
    auto part = find_supernodes(L, parent);
    return {std::move(L), std::move(parent), std::move(part)};
  }

  TEST_CASE(
    "assembly tree - single supernode yields one-node tree",
    "[assembly_tree]") {
    // Dense 4x4: one supernode containing all columns
    std::vector<Index> indices;
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        indices.push_back(Index{i, j});
      }
    }
    Compressed_row_sparsity sp{Shape{4, 4}, indices.begin(), indices.end()};

    auto a = full_analyze(sp);
    auto tree = build_assembly_tree(a.partition, a.parent);

    REQUIRE(std::ssize(tree.snode_parent) == 1);
    CHECK(tree.snode_parent[0] == -1);
    CHECK(tree.snode_children[0].empty());
    REQUIRE(std::ssize(tree.postorder) == 1);
    CHECK(tree.postorder[0] == 0);
  }

  TEST_CASE(
    "assembly tree - all singletons matches etree structure",
    "[assembly_tree]") {
    // 4x4 diagonal: 4 singleton supernodes, each a root
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto a = full_analyze(sp);
    auto tree = build_assembly_tree(a.partition, a.parent);

    REQUIRE(std::ssize(tree.snode_parent) == 4);
    for (size_type s = 0; s < 4; ++s) {
      CHECK(tree.snode_parent[static_cast<std::size_t>(s)] == -1);
      CHECK(tree.snode_children[static_cast<std::size_t>(s)].empty());
    }
  }

  TEST_CASE(
    "assembly tree - two disconnected blocks yield two roots",
    "[assembly_tree]") {
    // Two 2x2 dense blocks
    Compressed_row_sparsity sp{
      Shape{4, 4},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3}}};

    auto a = full_analyze(sp);
    auto tree = build_assembly_tree(a.partition, a.parent);

    REQUIRE(a.partition.n_supernodes == 2);
    REQUIRE(std::ssize(tree.snode_parent) == 2);

    // Both supernodes are roots
    CHECK(tree.snode_parent[0] == -1);
    CHECK(tree.snode_parent[1] == -1);
  }

  TEST_CASE(
    "assembly tree - postorder is valid permutation", "[assembly_tree]") {
    // 4x4 grid graph (16 nodes)
    std::vector<Index> indices;
    for (size_type r = 0; r < 4; ++r) {
      for (size_type c = 0; c < 4; ++c) {
        auto node = r * 4 + c;
        indices.push_back(Index{node, node});
        if (c + 1 < 4) {
          indices.push_back(Index{node, node + 1});
          indices.push_back(Index{node + 1, node});
        }
        if (r + 1 < 4) {
          indices.push_back(Index{node, node + 4});
          indices.push_back(Index{node + 4, node});
        }
      }
    }
    Compressed_row_sparsity sp{Shape{16, 16}, indices.begin(), indices.end()};

    auto a = full_analyze(sp);
    auto tree = build_assembly_tree(a.partition, a.parent);

    REQUIRE(std::ssize(tree.postorder) == a.partition.n_supernodes);
    CHECK(is_valid_permutation(tree.postorder));
  }

  TEST_CASE(
    "assembly tree - children before parents in postorder", "[assembly_tree]") {
    // 5x5 arrow matrix: chain etree -> chain assembly tree
    Compressed_row_sparsity sp{
      Shape{5, 5},
      {Index{0, 0},
       Index{0, 1},
       Index{0, 2},
       Index{0, 3},
       Index{0, 4},
       Index{1, 0},
       Index{1, 1},
       Index{2, 0},
       Index{2, 2},
       Index{3, 0},
       Index{3, 3},
       Index{4, 0},
       Index{4, 4}}};

    auto a = full_analyze(sp);
    auto tree = build_assembly_tree(a.partition, a.parent);

    auto ns = a.partition.n_supernodes;
    REQUIRE(std::ssize(tree.postorder) == ns);

    // Build inverse postorder
    std::vector<size_type> position(static_cast<std::size_t>(ns));
    for (size_type k = 0; k < ns; ++k) {
      position[static_cast<std::size_t>(
        tree.postorder[static_cast<std::size_t>(k)])] = k;
    }

    // Children must appear before their parents
    for (size_type s = 0; s < ns; ++s) {
      auto p = tree.snode_parent[static_cast<std::size_t>(s)];
      if (p != -1) {
        CHECK(
          position[static_cast<std::size_t>(s)] <
          position[static_cast<std::size_t>(p)]);
      }
    }
  }

} // end of namespace sparkit::testing
