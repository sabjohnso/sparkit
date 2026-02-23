//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <numeric>
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/reordering.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::is_valid_permutation;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::symmetrize_pattern;

  using sparkit::data::detail::cholesky_column_counts;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::tree_postorder;

  using size_type = sparkit::config::size_type;

  // Right-looking symbolic Cholesky on a symmetric pattern.
  // Returns the total number of nonzeros in the lower-triangular
  // factor L (including diagonal). Used to measure fill-in.
  static size_type
  symbolic_cholesky_nnz(Compressed_row_sparsity const& sp) {
    auto sym = symmetrize_pattern(sp);
    auto n = sym.shape().row();
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    // Initialize L with the lower triangle of the symmetric pattern
    std::vector<std::set<size_type>> rows(static_cast<std::size_t>(n));
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] <= i) { rows[static_cast<std::size_t>(i)].insert(ci[p]); }
      }
    }

    // Simulate elimination: when pivot k is eliminated, all pairs
    // (i,j) with i,j > k and both in column k of L create fill.
    for (size_type k = 0; k < n; ++k) {
      std::vector<size_type> col_k;
      for (size_type i = k + 1; i < n; ++i) {
        if (rows[static_cast<std::size_t>(i)].count(k)) { col_k.push_back(i); }
      }
      for (auto i : col_k) {
        for (auto j : col_k) {
          if (j <= i) { rows[static_cast<std::size_t>(i)].insert(j); }
        }
      }
    }

    size_type total = 0;
    for (size_type i = 0; i < n; ++i) {
      total += static_cast<size_type>(rows[static_cast<std::size_t>(i)].size());
    }
    return total;
  }

  // ================================================================
  // elimination_tree
  // ================================================================

  TEST_CASE("elimination tree - etree diagonal", "[elimination_tree]") {
    // 4x4 diagonal matrix: no off-diagonal entries
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 4);
    // All nodes are roots (no dependencies)
    for (size_type i = 0; i < 4; ++i) {
      CHECK(parent[static_cast<std::size_t>(i)] == -1);
    }
  }

  TEST_CASE("elimination tree - etree tridiagonal", "[elimination_tree]") {
    // 4x4 tridiagonal: 0-1-2-3
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

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 4);
    // Chain: parent[i] == i+1 for i < n-1, parent[n-1] == -1
    CHECK(parent[0] == 1);
    CHECK(parent[1] == 2);
    CHECK(parent[2] == 3);
    CHECK(parent[3] == -1);
  }

  TEST_CASE("elimination tree - etree arrow", "[elimination_tree]") {
    // 5x5 arrow: node 0 is hub, connected to all
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

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 5);
    // Upper triangle has entries (0,1),(0,2),(0,3),(0,4)
    // Processing column 1: row 0 -> parent[0] = 1
    // Processing column 2: row 0 -> walk 0->1 (ancestor), parent[1] = 2
    // Processing column 3: row 0 -> walk 0->...->2 (ancestor), parent[2] = 3
    // Processing column 4: row 0 -> walk 0->...->3 (ancestor), parent[3] = 4
    // Result: chain 0->1->2->3->4 (root)
    CHECK(parent[0] == 1);
    CHECK(parent[1] == 2);
    CHECK(parent[2] == 3);
    CHECK(parent[3] == 4);
    CHECK(parent[4] == -1);
  }

  TEST_CASE(
    "elimination tree - etree rectangular rejected", "[elimination_tree]") {
    Compressed_row_sparsity sp{
      Shape{3, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}}};

    CHECK_THROWS_AS(elimination_tree(sp), std::invalid_argument);
  }

  TEST_CASE("elimination tree - etree disconnected", "[elimination_tree]") {
    // Two 2x2 blocks: {0,1} and {2,3}
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

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 4);
    // Component 1: parent[0] = 1, parent[1] = -1
    // Component 2: parent[2] = 3, parent[3] = -1
    CHECK(parent[0] == 1);
    CHECK(parent[1] == -1);
    CHECK(parent[2] == 3);
    CHECK(parent[3] == -1);
  }

  TEST_CASE("elimination tree - etree dense", "[elimination_tree]") {
    // 4x4 fully connected
    std::vector<Index> indices;
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        indices.push_back(Index{i, j});
      }
    }
    Compressed_row_sparsity sp{Shape{4, 4}, indices.begin(), indices.end()};

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 4);
    // Dense matrix: parent[i] == i+1 (chain)
    CHECK(parent[0] == 1);
    CHECK(parent[1] == 2);
    CHECK(parent[2] == 3);
    CHECK(parent[3] == -1);
  }

  TEST_CASE("elimination tree - etree small", "[elimination_tree]") {
    // 2x2 with off-diagonal
    Compressed_row_sparsity sp{
      Shape{2, 2}, {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}}};

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 2);
    CHECK(parent[0] == 1);
    CHECK(parent[1] == -1);
  }

  TEST_CASE("elimination tree - etree parent greater", "[elimination_tree]") {
    // 6x6 tridiagonal
    Compressed_row_sparsity sp{
      Shape{6, 6},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3},
       Index{3, 4},
       Index{4, 3},
       Index{4, 4},
       Index{4, 5},
       Index{5, 4},
       Index{5, 5}}};

    auto parent = elimination_tree(sp);

    REQUIRE(std::ssize(parent) == 6);
    // parent[i] > i for all non-roots
    for (size_type i = 0; i < 6; ++i) {
      auto p = parent[static_cast<std::size_t>(i)];
      CHECK((p == -1 || p > i));
    }
  }

  // ================================================================
  // tree_postorder
  // ================================================================

  TEST_CASE(
    "elimination tree - postorder valid permutation", "[elimination_tree]") {
    // 6x6 tridiagonal -> chain etree
    Compressed_row_sparsity sp{
      Shape{6, 6},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3},
       Index{3, 4},
       Index{4, 3},
       Index{4, 4},
       Index{4, 5},
       Index{5, 4},
       Index{5, 5}}};

    auto parent = elimination_tree(sp);
    auto post = tree_postorder(parent);

    REQUIRE(std::ssize(post) == 6);
    CHECK(is_valid_permutation(post));
  }

  TEST_CASE(
    "elimination tree - postorder children before parents",
    "[elimination_tree]") {
    // Arrow matrix: star tree (all nodes parent to a chain ending at root)
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

    auto parent = elimination_tree(sp);
    auto post = tree_postorder(parent);

    REQUIRE(std::ssize(post) == 5);
    CHECK(is_valid_permutation(post));

    // Build inverse: position[node] = k means post[k] = node
    std::vector<size_type> position(5);
    for (size_type k = 0; k < 5; ++k) {
      position[static_cast<std::size_t>(post[static_cast<std::size_t>(k)])] = k;
    }

    // For every node with a parent, the child's position < parent's position
    for (size_type i = 0; i < 5; ++i) {
      auto p = parent[static_cast<std::size_t>(i)];
      if (p != -1) {
        CHECK(
          position[static_cast<std::size_t>(i)] <
          position[static_cast<std::size_t>(p)]);
      }
    }
  }

  TEST_CASE("elimination tree - postorder chain", "[elimination_tree]") {
    // Chain etree (tridiagonal): postorder is identity [0,1,...,n-1]
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

    auto parent = elimination_tree(sp);
    auto post = tree_postorder(parent);

    REQUIRE(std::ssize(post) == 4);
    // Chain: each node has exactly one child, postorder is identity
    for (size_type i = 0; i < 4; ++i) {
      CHECK(post[static_cast<std::size_t>(i)] == i);
    }
  }

  TEST_CASE("elimination tree - postorder all roots", "[elimination_tree]") {
    // Diagonal matrix: all nodes are roots (forest of singletons)
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto parent = elimination_tree(sp);
    auto post = tree_postorder(parent);

    REQUIRE(std::ssize(post) == 4);
    CHECK(is_valid_permutation(post));
  }

  // ================================================================
  // cholesky_column_counts
  // ================================================================

  TEST_CASE("elimination tree - counts diagonal", "[elimination_tree]") {
    // 4x4 diagonal: each column has only the diagonal
    Compressed_row_sparsity sp{
      Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    REQUIRE(std::ssize(counts) == 4);
    for (size_type i = 0; i < 4; ++i) {
      CHECK(counts[static_cast<std::size_t>(i)] == 1);
    }
  }

  TEST_CASE("elimination tree - counts tridiagonal", "[elimination_tree]") {
    // 4x4 tridiagonal
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

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    REQUIRE(std::ssize(counts) == 4);
    // Tridiagonal Cholesky: L is lower bidiagonal
    // Column 0: diagonal + entry from row 1 -> count = 2
    // Column 1: diagonal + entry from row 2 -> count = 2
    // Column 2: diagonal + entry from row 3 -> count = 2
    // Column 3: diagonal only -> count = 1
    CHECK(counts[0] == 2);
    CHECK(counts[1] == 2);
    CHECK(counts[2] == 2);
    CHECK(counts[3] == 1);
  }

  TEST_CASE("elimination tree - counts arrow", "[elimination_tree]") {
    // 5x5 arrow: hub at node 0 connected to all
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

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    REQUIRE(std::ssize(counts) == 5);
    // Eliminating node 0 fills in all pairs -> L is dense lower triangle
    // Column 0: entries in rows 0,1,2,3,4 -> count = 5
    // Column 1: entries in rows 1,2,3,4 -> count = 4
    // Column 2: entries in rows 2,3,4 -> count = 3
    // Column 3: entries in rows 3,4 -> count = 2
    // Column 4: entry in row 4 -> count = 1
    CHECK(counts[0] == 5);
    CHECK(counts[1] == 4);
    CHECK(counts[2] == 3);
    CHECK(counts[3] == 2);
    CHECK(counts[4] == 1);
  }

  TEST_CASE(
    "elimination tree - counts sum equals symbolic nnz tridiagonal",
    "[elimination_tree]") {
    // 6x6 tridiagonal
    Compressed_row_sparsity sp{
      Shape{6, 6},
      {Index{0, 0},
       Index{0, 1},
       Index{1, 0},
       Index{1, 1},
       Index{1, 2},
       Index{2, 1},
       Index{2, 2},
       Index{2, 3},
       Index{3, 2},
       Index{3, 3},
       Index{3, 4},
       Index{4, 3},
       Index{4, 4},
       Index{4, 5},
       Index{5, 4},
       Index{5, 5}}};

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    auto sum = std::accumulate(counts.begin(), counts.end(), size_type{0});
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(sum == expected);
  }

  TEST_CASE("elimination tree - counts sum arrow", "[elimination_tree]") {
    // 5x5 arrow
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

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    auto sum = std::accumulate(counts.begin(), counts.end(), size_type{0});
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(sum == expected);
  }

  TEST_CASE("elimination tree - counts sum grid", "[elimination_tree]") {
    // 4x4 grid graph (16 nodes)
    std::vector<Index> indices;
    for (size_type r = 0; r < 4; ++r) {
      for (size_type c = 0; c < 4; ++c) {
        auto node = r * 4 + c;
        indices.push_back(Index{node, node}); // diagonal
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

    auto parent = elimination_tree(sp);
    auto counts = cholesky_column_counts(sp, parent);

    auto sum = std::accumulate(counts.begin(), counts.end(), size_type{0});
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(sum == expected);
  }

} // end of namespace sparkit::testing
