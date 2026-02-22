//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/reordering.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::symmetrize_pattern;

  using sparkit::data::detail::cholesky_column_counts;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // Right-looking symbolic Cholesky on a symmetric pattern.
  // Returns the total number of nonzeros in the lower-triangular
  // factor L (including diagonal). Used as reference oracle.
  static size_type
  symbolic_cholesky_nnz(Compressed_row_sparsity const& sp) {
    auto sym = symmetrize_pattern(sp);
    auto n = sym.shape().row();
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    std::vector<std::set<size_type>> rows(static_cast<std::size_t>(n));
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] <= i) { rows[static_cast<std::size_t>(i)].insert(ci[p]); }
      }
    }

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
  // symbolic_cholesky
  // ================================================================

  TEST_CASE("symbolic cholesky - diagonal", "[symbolic_cholesky]") {
    // 4x4 diagonal matrix: L == A, each row has 1 entry
    Compressed_row_sparsity sp{
        Shape{4, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}}};

    auto L = symbolic_cholesky(sp);

    REQUIRE(L.shape().row() == 4);
    REQUIRE(L.shape().column() == 4);
    REQUIRE(L.size() == 4);

    auto rp = L.row_ptr();
    for (size_type i = 0; i < 4; ++i) {
      CHECK(rp[i + 1] - rp[i] == 1);
    }
  }

  TEST_CASE("symbolic cholesky - tridiagonal", "[symbolic_cholesky]") {
    // 4x4 tridiagonal: L is lower bidiagonal, nnz = 4 + 3 = 7
    Compressed_row_sparsity sp{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}, Index{1, 2},
         Index{2, 1}, Index{2, 2}, Index{2, 3}, Index{3, 2}, Index{3, 3}}};

    auto L = symbolic_cholesky(sp);

    REQUIRE(L.shape().row() == 4);
    REQUIRE(L.shape().column() == 4);
    CHECK(L.size() == 7);
  }

  TEST_CASE("symbolic cholesky - arrow", "[symbolic_cholesky]") {
    // 5x5 arrow: hub at node 0 connected to all -> L is dense lower triangle
    Compressed_row_sparsity sp{
        Shape{5, 5},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
         Index{1, 0}, Index{1, 1}, Index{2, 0}, Index{2, 2}, Index{3, 0},
         Index{3, 3}, Index{4, 0}, Index{4, 4}}};

    auto L = symbolic_cholesky(sp);

    // Dense lower triangle: n*(n+1)/2 = 5*6/2 = 15
    CHECK(L.size() == 15);
  }

  TEST_CASE("symbolic cholesky - lower triangular", "[symbolic_cholesky]") {
    // Every entry of L must have row >= col
    Compressed_row_sparsity sp{
        Shape{5, 5},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
         Index{1, 0}, Index{1, 1}, Index{2, 0}, Index{2, 2}, Index{3, 0},
         Index{3, 3}, Index{4, 0}, Index{4, 4}}};

    auto L = symbolic_cholesky(sp);

    auto rp = L.row_ptr();
    auto ci = L.col_ind();

    for (size_type i = 0; i < L.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] <= i);
      }
    }
  }

  TEST_CASE("symbolic cholesky - nnz tridiagonal", "[symbolic_cholesky]") {
    // 6x6 tridiagonal
    Compressed_row_sparsity sp{
        Shape{6, 6},
        {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}, Index{1, 2},
         Index{2, 1}, Index{2, 2}, Index{2, 3}, Index{3, 2}, Index{3, 3},
         Index{3, 4}, Index{4, 3}, Index{4, 4}, Index{4, 5}, Index{5, 4},
         Index{5, 5}}};

    auto L = symbolic_cholesky(sp);
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(L.size() == expected);
  }

  TEST_CASE("symbolic cholesky - nnz arrow", "[symbolic_cholesky]") {
    // 5x5 arrow
    Compressed_row_sparsity sp{
        Shape{5, 5},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
         Index{1, 0}, Index{1, 1}, Index{2, 0}, Index{2, 2}, Index{3, 0},
         Index{3, 3}, Index{4, 0}, Index{4, 4}}};

    auto L = symbolic_cholesky(sp);
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(L.size() == expected);
  }

  TEST_CASE("symbolic cholesky - nnz grid", "[symbolic_cholesky]") {
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

    auto L = symbolic_cholesky(sp);
    auto expected = symbolic_cholesky_nnz(sp);

    CHECK(L.size() == expected);
  }

  TEST_CASE("symbolic cholesky - column counts match", "[symbolic_cholesky]") {
    // 5x5 arrow: per-column counts of L must match cholesky_column_counts
    Compressed_row_sparsity sp{
        Shape{5, 5},
        {Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
         Index{1, 0}, Index{1, 1}, Index{2, 0}, Index{2, 2}, Index{3, 0},
         Index{3, 3}, Index{4, 0}, Index{4, 4}}};

    auto L = symbolic_cholesky(sp);
    auto parent = elimination_tree(sp);
    auto expected_counts = cholesky_column_counts(sp, parent);

    auto n = L.shape().row();
    auto rp = L.row_ptr();
    auto ci = L.col_ind();

    // Count entries per column from the CSR representation
    std::vector<size_type> actual_counts(static_cast<std::size_t>(n), 0);
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        ++actual_counts[static_cast<std::size_t>(ci[p])];
      }
    }

    for (size_type j = 0; j < n; ++j) {
      CHECK(actual_counts[static_cast<std::size_t>(j)] ==
            expected_counts[static_cast<std::size_t>(j)]);
    }
  }

  TEST_CASE("symbolic cholesky - includes original entries",
            "[symbolic_cholesky]") {
    // All lower-triangle entries of A must appear in L
    Compressed_row_sparsity sp{
        Shape{4, 4},
        {Index{0, 0}, Index{0, 1}, Index{1, 0}, Index{1, 1}, Index{1, 2},
         Index{2, 1}, Index{2, 2}, Index{2, 3}, Index{3, 2}, Index{3, 3}}};

    auto L = symbolic_cholesky(sp);
    auto sym = symmetrize_pattern(sp);

    auto L_rp = L.row_ptr();
    auto L_ci = L.col_ind();

    auto A_rp = sym.row_ptr();
    auto A_ci = sym.col_ind();

    // For each lower-triangle entry in A, check it exists in L
    for (size_type i = 0; i < sym.shape().row(); ++i) {
      // Collect L's columns for row i into a set
      std::set<size_type> L_cols;
      for (auto p = L_rp[i]; p < L_rp[i + 1]; ++p) {
        L_cols.insert(L_ci[p]);
      }

      for (auto p = A_rp[i]; p < A_rp[i + 1]; ++p) {
        if (A_ci[p] <= i) { CHECK(L_cols.count(A_ci[p]) == 1); }
      }
    }
  }

  TEST_CASE("symbolic cholesky - rectangular rejected", "[symbolic_cholesky]") {
    Compressed_row_sparsity sp{Shape{3, 4},
                               {Index{0, 0}, Index{1, 1}, Index{2, 2}}};

    CHECK_THROWS_AS(symbolic_cholesky(sp), std::invalid_argument);
  }

  TEST_CASE("symbolic cholesky - disconnected", "[symbolic_cholesky]") {
    // Two 2x2 blocks: {0,1} and {2,3}
    Compressed_row_sparsity sp{Shape{4, 4},
                               {Index{0, 0}, Index{0, 1}, Index{1, 0},
                                Index{1, 1}, Index{2, 2}, Index{2, 3},
                                Index{3, 2}, Index{3, 3}}};

    auto L = symbolic_cholesky(sp);
    auto expected = symbolic_cholesky_nnz(sp);

    // Each 2x2 block produces a 2x2 lower triangle (3 entries each) = 6 total
    CHECK(L.size() == expected);
    CHECK(L.size() == 6);
  }

} // end of namespace sparkit::testing
