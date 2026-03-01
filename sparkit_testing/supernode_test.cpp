//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <numeric>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::find_supernodes;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // Helper: build L_pattern and etree for a given sparsity
  static std::pair<Compressed_row_sparsity, std::vector<size_type>>
  analyze(Compressed_row_sparsity const& sp) {
    auto parent = elimination_tree(sp);
    auto L = symbolic_cholesky(sp);
    return {std::move(L), std::move(parent)};
  }

  TEST_CASE("supernode - diagonal matrix yields n singletons", "[supernode]") {
    // 4x4 diagonal: no off-diagonal, every column is its own supernode
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    REQUIRE(part.n_supernodes == 4);
    REQUIRE(std::ssize(part.snode_start) == 5);
    REQUIRE(std::ssize(part.membership) == 4);

    // Each supernode contains exactly one column
    for (size_type s = 0; s < 4; ++s) {
      CHECK(
        part.snode_start[static_cast<std::size_t>(s + 1)] -
          part.snode_start[static_cast<std::size_t>(s)] ==
        1);
    }

    // Every column has a distinct membership
    for (size_type j = 0; j < 4; ++j) {
      CHECK(part.membership[static_cast<std::size_t>(j)] == j);
    }
  }

  TEST_CASE("supernode - dense matrix yields one supernode", "[supernode]") {
    // 4x4 fully connected: L is dense lower triangle, one supernode
    std::vector<Index> indices;
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        indices.push_back(Index{i, j});
      }
    }
    Compressed_row_sparsity sp{Shape{4, 4}, indices.begin(), indices.end()};

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    REQUIRE(part.n_supernodes == 1);
    CHECK(part.snode_start[0] == 0);
    CHECK(part.snode_start[1] == 4);

    for (size_type j = 0; j < 4; ++j) {
      CHECK(part.membership[static_cast<std::size_t>(j)] == 0);
    }
  }

  TEST_CASE("supernode - tridiagonal matrix supernodes", "[supernode]") {
    // 4x4 tridiagonal: L is lower bidiagonal.
    // col_count = [2,2,2,1], parent = [1,2,3,-1]
    // Column 1: parent[0]=1, col_count[0]=2 != col_count[1]+1=3 -> new
    // Column 2: parent[1]=2, col_count[1]=2 != col_count[2]+1=3 -> new
    // Column 3: parent[2]=3, col_count[2]=2 == col_count[3]+1=2 -> merge
    // Supernodes: {0}, {1}, {2,3}
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

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    REQUIRE(part.n_supernodes == 3);
    CHECK(part.snode_start[0] == 0);
    CHECK(part.snode_start[1] == 1);
    CHECK(part.snode_start[2] == 2);
    CHECK(part.snode_start[3] == 4);
  }

  TEST_CASE("supernode - arrow matrix yields one supernode", "[supernode]") {
    // 5x5 arrow: hub at node 0 -> L is dense lower triangle, one supernode
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

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    REQUIRE(part.n_supernodes == 1);
    CHECK(part.snode_start[0] == 0);
    CHECK(part.snode_start[1] == 5);
  }

  TEST_CASE("supernode - block diagonal yields two supernodes", "[supernode]") {
    // Two 2x2 dense blocks: L has two supernodes
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

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    REQUIRE(part.n_supernodes == 2);
    CHECK(part.snode_start[0] == 0);
    CHECK(part.snode_start[1] == 2);
    CHECK(part.snode_start[2] == 4);
  }

  TEST_CASE("supernode - membership covers all columns", "[supernode]") {
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

    auto [L, parent] = analyze(sp);
    auto part = find_supernodes(L, parent);

    // Every column is assigned to some supernode in [0, n_supernodes)
    REQUIRE(std::ssize(part.membership) == 16);
    for (size_type j = 0; j < 16; ++j) {
      auto s = part.membership[static_cast<std::size_t>(j)];
      CHECK(s >= 0);
      CHECK(s < part.n_supernodes);
    }

    // snode_start partitions [0, n) completely
    CHECK(part.snode_start[0] == 0);
    CHECK(part.snode_start[static_cast<std::size_t>(part.n_supernodes)] == 16);

    // Each column's membership is consistent with snode_start
    for (size_type j = 0; j < 16; ++j) {
      auto s = part.membership[static_cast<std::size_t>(j)];
      CHECK(j >= part.snode_start[static_cast<std::size_t>(s)]);
      CHECK(j < part.snode_start[static_cast<std::size_t>(s + 1)]);
    }
  }

} // end of namespace sparkit::testing
