//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/amg_aggregation.hpp>
#include <sparkit/data/amg_strength.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::aggregate;
  using sparkit::data::detail::strength_of_connection;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // Aggregation tests
  // ================================================================

  TEST_CASE("aggregate - fully connected 3 nodes", "[amg_aggregation]") {
    // Full strength graph: all connected to all.
    // Should produce 1 aggregate containing all 3 nodes.
    std::vector<Index> indices = {
      Index{0, 1},
      Index{0, 2},
      Index{1, 0},
      Index{1, 2},
      Index{2, 0},
      Index{2, 1}};
    Compressed_row_sparsity S{Shape{3, 3}, indices.begin(), indices.end()};

    auto [agg_ids, n_agg] = aggregate(S, 3);

    REQUIRE(agg_ids.size() == 3);
    CHECK(n_agg == 1);
    CHECK(agg_ids[0] == agg_ids[1]);
    CHECK(agg_ids[1] == agg_ids[2]);
  }

  TEST_CASE("aggregate - disconnected components", "[amg_aggregation]") {
    // Two disconnected pairs: {0,1} and {2,3}.
    std::vector<Index> indices = {
      Index{0, 1}, Index{1, 0}, Index{2, 3}, Index{3, 2}};
    Compressed_row_sparsity S{Shape{4, 4}, indices.begin(), indices.end()};

    auto [agg_ids, n_agg] = aggregate(S, 4);

    REQUIRE(agg_ids.size() == 4);
    CHECK(n_agg >= 2);

    // Nodes 0 and 1 should be in the same aggregate
    CHECK(agg_ids[0] == agg_ids[1]);
    // Nodes 2 and 3 should be in the same aggregate
    CHECK(agg_ids[2] == agg_ids[3]);
    // Different components should be in different aggregates
    CHECK(agg_ids[0] != agg_ids[2]);
  }

  TEST_CASE("aggregate - linear chain", "[amg_aggregation]") {
    // 0 -- 1 -- 2 -- 3 -- 4
    std::vector<Index> indices = {
      Index{0, 1},
      Index{1, 0},
      Index{1, 2},
      Index{2, 1},
      Index{2, 3},
      Index{3, 2},
      Index{3, 4},
      Index{4, 3}};
    Compressed_row_sparsity S{Shape{5, 5}, indices.begin(), indices.end()};

    auto [agg_ids, n_agg] = aggregate(S, 5);

    REQUIRE(agg_ids.size() == 5);
    CHECK(n_agg >= 1);
    CHECK(n_agg <= 5);

    // Full coverage: every node assigned to a valid aggregate
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(agg_ids[i] >= 0);
      CHECK(agg_ids[i] < n_agg);
    }
  }

  TEST_CASE("aggregate - single node", "[amg_aggregation]") {
    // Isolated node: no connections.
    std::vector<Index> indices;
    Compressed_row_sparsity S{Shape{2, 2}, indices.begin(), indices.end()};

    auto [agg_ids, n_agg] = aggregate(S, 2);

    REQUIRE(agg_ids.size() == 2);
    // Each isolated node becomes its own aggregate
    CHECK(n_agg == 2);
    CHECK(agg_ids[0] != agg_ids[1]);
  }

  TEST_CASE("aggregate - full coverage invariant", "[amg_aggregation]") {
    // 4x4 grid Laplacian: all connections strong at theta=0.25
    size_type const grid = 4;
    size_type const n = grid * grid;

    std::vector<Entry<double>> entries;
    for (size_type r = 0; r < grid; ++r) {
      for (size_type c = 0; c < grid; ++c) {
        auto node = r * grid + c;
        entries.push_back(Entry<double>{Index{node, node}, 4.0});
        if (c > 0) {
          entries.push_back(Entry<double>{Index{node, node - 1}, -1.0});
        }
        if (c + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + 1}, -1.0});
        }
        if (r > 0) {
          entries.push_back(Entry<double>{Index{node, node - grid}, -1.0});
        }
        if (r + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + grid}, -1.0});
        }
      }
    }

    std::vector<Index> indices;
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }
    Compressed_row_sparsity sp{Shape{n, n}, indices.begin(), indices.end()};

    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    std::vector<double> vals(static_cast<std::size_t>(sp.size()), 0.0);
    for (auto const& e : entries) {
      auto row = e.index.row();
      auto col = e.index.column();
      for (auto p = rp[row]; p < rp[row + 1]; ++p) {
        if (ci[p] == col) {
          vals[static_cast<std::size_t>(p)] = e.value;
          break;
        }
      }
    }
    Compressed_row_matrix<double> A{std::move(sp), std::move(vals)};

    auto S = strength_of_connection(A, 0.25);
    auto [agg_ids, n_agg] = aggregate(S, n);

    REQUIRE(agg_ids.size() == static_cast<std::size_t>(n));

    // Every node is assigned
    for (size_type i = 0; i < n; ++i) {
      CHECK(agg_ids[static_cast<std::size_t>(i)] >= 0);
      CHECK(agg_ids[static_cast<std::size_t>(i)] < n_agg);
    }

    // Coarsening: fewer aggregates than fine nodes
    CHECK(n_agg < n);
    CHECK(n_agg >= 1);

    // Each aggregate ID is used at least once
    std::set<size_type> used;
    for (auto id : agg_ids) {
      used.insert(id);
    }
    CHECK(static_cast<size_type>(used.size()) == n_agg);
  }

} // end of namespace sparkit::testing
