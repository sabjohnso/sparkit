//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/assembly_tree.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/multifrontal_symbolic.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Assembly_tree;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Multifrontal_symbolic;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Supernode_partition;

  using sparkit::data::detail::build_assembly_tree;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::find_supernodes;
  using sparkit::data::detail::multifrontal_analyze;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // Helper: full analysis pipeline
  static Multifrontal_symbolic
  full_symbolic(Compressed_row_sparsity const& sp) {
    auto parent = elimination_tree(sp);
    auto L = symbolic_cholesky(sp);
    auto part = find_supernodes(L, parent);
    auto tree = build_assembly_tree(part, parent);
    return multifrontal_analyze(L, part, tree);
  }

  TEST_CASE(
    "multifrontal symbolic - diagonal has size-1 fronts",
    "[multifrontal_symbolic]") {
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto sym = full_symbolic(sp);

    REQUIRE(sym.n == 4);
    REQUIRE(std::ssize(sym.maps) == sym.partition.n_supernodes);

    // Each supernode: front_size == 1, snode_size == 1
    for (size_type s = 0; s < sym.partition.n_supernodes; ++s) {
      auto const& m = sym.maps[static_cast<std::size_t>(s)];
      CHECK(m.snode_size == 1);
      CHECK(m.front_size == 1);
      REQUIRE(std::ssize(m.row_indices) == 1);
    }
  }

  TEST_CASE(
    "multifrontal symbolic - tridiagonal has small fronts",
    "[multifrontal_symbolic]") {
    Compressed_row_sparsity sp{
      Shape{4, 4},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3}}};

    auto sym = full_symbolic(sp);

    // Every front has front_size >= snode_size
    for (size_type s = 0; s < sym.partition.n_supernodes; ++s) {
      auto const& m = sym.maps[static_cast<std::size_t>(s)];
      CHECK(m.front_size >= m.snode_size);
      CHECK(std::ssize(m.row_indices) == m.front_size);
    }
  }

  TEST_CASE(
    "multifrontal symbolic - arrow yields single large front",
    "[multifrontal_symbolic]") {
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

    auto sym = full_symbolic(sp);

    // One supernode with front_size == 5
    REQUIRE(sym.partition.n_supernodes == 1);
    CHECK(sym.maps[0].snode_size == 5);
    CHECK(sym.maps[0].front_size == 5);
  }

  TEST_CASE(
    "multifrontal symbolic - row indices sorted and start with supernode cols",
    "[multifrontal_symbolic]") {
    // 4x4 grid
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

    auto sym = full_symbolic(sp);

    for (size_type s = 0; s < sym.partition.n_supernodes; ++s) {
      auto const& m = sym.maps[static_cast<std::size_t>(s)];
      auto sstart = sym.partition.snode_start[static_cast<std::size_t>(s)];
      auto send = sym.partition.snode_start[static_cast<std::size_t>(s + 1)];

      // First snode_size entries must be the supernode columns
      for (size_type k = 0; k < m.snode_size; ++k) {
        CHECK(m.row_indices[static_cast<std::size_t>(k)] == sstart + k);
      }

      // Remaining update rows must be sorted and > last supernode col
      for (size_type k = m.snode_size; k < m.front_size; ++k) {
        CHECK(m.row_indices[static_cast<std::size_t>(k)] >= send);
        if (k > m.snode_size) {
          CHECK(
            m.row_indices[static_cast<std::size_t>(k)] >
            m.row_indices[static_cast<std::size_t>(k - 1)]);
        }
      }
    }
  }

  TEST_CASE(
    "multifrontal symbolic - relative maps correctly map child to parent",
    "[multifrontal_symbolic]") {
    // 4x4 grid
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

    auto sym = full_symbolic(sp);

    // For each supernode s with children, relative_maps[c] maps
    // child c's update indices into parent s's row_indices
    for (size_type s = 0; s < sym.partition.n_supernodes; ++s) {
      auto const& children =
        sym.tree.snode_children[static_cast<std::size_t>(s)];
      auto const& parent_rows =
        sym.maps[static_cast<std::size_t>(s)].row_indices;

      for (auto c : children) {
        auto const& child_map = sym.relative_maps[static_cast<std::size_t>(c)];
        auto const& child_rows =
          sym.maps[static_cast<std::size_t>(c)].row_indices;
        auto child_snode_size =
          sym.maps[static_cast<std::size_t>(c)].snode_size;

        // relative_map has update_size entries
        auto update_size =
          child_rows.size() - static_cast<std::size_t>(child_snode_size);
        REQUIRE(child_map.size() == update_size);

        // Each mapped index refers to a valid position in parent's row_indices
        // and the row values match
        for (std::size_t k = 0; k < update_size; ++k) {
          auto parent_pos = child_map[k];
          CHECK(parent_pos >= 0);
          CHECK(parent_pos < std::ssize(parent_rows));
          auto child_row =
            child_rows[static_cast<std::size_t>(child_snode_size) + k];
          CHECK(parent_rows[static_cast<std::size_t>(parent_pos)] == child_row);
        }
      }
    }
  }

} // end of namespace sparkit::testing
