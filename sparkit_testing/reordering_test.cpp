//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/info.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/reordering.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::symmetrize_pattern;
  using sparkit::data::detail::adjacency_degree;
  using sparkit::data::detail::pseudo_peripheral_node;
  using sparkit::data::detail::reverse_cuthill_mckee;
  using sparkit::data::detail::is_valid_permutation;
  using sparkit::data::detail::dperm;
  using sparkit::data::detail::bandwidth;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // symmetrize_pattern
  // ================================================================

  TEST_CASE("reordering - symmetrize already symmetric", "[reordering]")
  {
    // Symmetric pattern: (0,1),(1,0),(0,0),(1,1)
    Compressed_row_sparsity sp{Shape{2, 2}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}
    }};

    auto sym = symmetrize_pattern(sp);

    CHECK(sym.shape() == Shape(2, 2));
    CHECK(sym.size() == 4);

    auto rp = sym.row_ptr();
    CHECK(rp[1] - rp[0] == 2);
    CHECK(rp[2] - rp[1] == 2);
  }

  TEST_CASE("reordering - symmetrize asymmetric", "[reordering]")
  {
    // Lower triangular: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 1}, Index{2, 2}
    }};

    auto sym = symmetrize_pattern(sp);

    CHECK(sym.shape() == Shape(3, 3));
    // Should have upper entries added: (0,1),(0,2),(1,2)
    // Full symmetric: diagonal (3) + lower (3) + upper (3) = 9
    CHECK(sym.size() == 9);

    auto rp = sym.row_ptr();
    // Row 0 should now have cols {0, 1, 2}
    CHECK(rp[1] - rp[0] == 3);
  }

  TEST_CASE("reordering - symmetrize preserves diagonal", "[reordering]")
  {
    // Just diagonal entries
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0}, Index{1, 1}, Index{2, 2}
    }};

    auto sym = symmetrize_pattern(sp);

    CHECK(sym.size() == 3);
  }

  // ================================================================
  // adjacency_degree
  // ================================================================

  TEST_CASE("reordering - adjacency degree known", "[reordering]")
  {
    // Path graph: 0-1-2
    // Adjacency: (0,1),(1,0),(1,2),(2,1) + diag
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}
    }};

    auto deg = adjacency_degree(sp);

    REQUIRE(std::ssize(deg) == 3);
    CHECK(deg[0] == 1);  // node 0: neighbor {1}
    CHECK(deg[1] == 2);  // node 1: neighbors {0, 2}
    CHECK(deg[2] == 1);  // node 2: neighbor {1}
  }

  // ================================================================
  // pseudo_peripheral_node
  // ================================================================

  TEST_CASE("reordering - pseudo peripheral node path graph", "[reordering]")
  {
    // Path graph: 0-1-2-3
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto node = pseudo_peripheral_node(sp);

    // Should be one of the endpoints (0 or 3)
    CHECK((node == 0 || node == 3));
  }

  // ================================================================
  // reverse_cuthill_mckee
  // ================================================================

  TEST_CASE("reordering - rcm valid permutation", "[reordering]")
  {
    // Arrow matrix: row 0 connects to all, others only to 0 and self
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 2},
      Index{3, 0}, Index{3, 3}
    }};

    auto perm = reverse_cuthill_mckee(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - rcm identity permutation tridiagonal", "[reordering]")
  {
    // Tridiagonal: already optimal bandwidth
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = reverse_cuthill_mckee(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));

    // Create matrix and check bandwidth is still 1
    Compressed_row_matrix<double> A{sp, [](auto, auto) { return 1.0; }};
    auto B = dperm(A, perm);
    auto [lo, hi] = bandwidth(B);
    CHECK(lo <= 1);
    CHECK(hi <= 1);
  }

  TEST_CASE("reordering - rcm arrow matrix", "[reordering]")
  {
    // Arrow pattern: node 0 is hub connected to all others
    // Original bandwidth = n-1
    Compressed_row_sparsity sp{Shape{5, 5}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 2},
      Index{3, 0}, Index{3, 3},
      Index{4, 0}, Index{4, 4}
    }};

    Compressed_row_matrix<double> A{sp, [](auto, auto) { return 1.0; }};
    auto [orig_lo, orig_hi] = bandwidth(A);

    auto perm = reverse_cuthill_mckee(sp);
    auto B = dperm(A, perm);
    auto [new_lo, new_hi] = bandwidth(B);

    // RCM should reduce bandwidth from 4 to at most 4
    // (for arrow, RCM moves hub to end)
    CHECK(new_lo <= orig_lo);
    CHECK(new_hi <= orig_hi);
  }

  TEST_CASE("reordering - rcm reduces bandwidth", "[reordering]")
  {
    // Scrambled tridiagonal: 0-3, 1-2, 2-0, 3-1
    // Actual adjacency: {0,2}, {1,3}, {2,0}, {2,1}, {3,1}, {3,2} (not symmetric yet)
    // Use a bad ordering that has large bandwidth
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 3},
      Index{1, 1}, Index{1, 2},
      Index{2, 2}, Index{2, 3},
      Index{3, 0}, Index{3, 3}
    }};

    Compressed_row_matrix<double> A{sp, [](auto, auto) { return 1.0; }};
    auto [orig_lo, orig_hi] = bandwidth(A);

    auto perm = reverse_cuthill_mckee(sp);
    auto B = dperm(A, perm);
    auto [new_lo, new_hi] = bandwidth(B);

    CHECK(std::max(new_lo, new_hi) <= std::max(orig_lo, orig_hi));
  }

  TEST_CASE("reordering - rcm disconnected graph", "[reordering]")
  {
    // Two disconnected components: {0,1} and {2,3}
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1},
      Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = reverse_cuthill_mckee(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - rcm single diagonal", "[reordering]")
  {
    // Minimal 2x2 diagonal matrix (no off-diagonal entries)
    Compressed_row_sparsity sp{Shape{2, 2}, {
      Index{0, 0}, Index{1, 1}
    }};

    auto perm = reverse_cuthill_mckee(sp);

    REQUIRE(std::ssize(perm) == 2);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - rcm round trip values", "[reordering]")
  {
    // Create matrix, apply RCM permutation, verify all entries preserved
    Compressed_row_matrix<double> A{Shape{4, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}, {Index{1, 2}, 5.0},
      {Index{2, 1}, 6.0}, {Index{2, 2}, 7.0}, {Index{2, 3}, 8.0},
      {Index{3, 2}, 9.0}, {Index{3, 3}, 10.0}
    }};

    auto perm = reverse_cuthill_mckee(A.sparsity());
    auto B = dperm(A, perm);

    // B should have the same number of nonzeros
    CHECK(B.size() == A.size());

    // Apply inverse permutation to get back original
    auto inv = sparkit::data::detail::inverse_permutation(perm);
    auto C = dperm(B, inv);

    // C should match A element-by-element
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        CHECK(C(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  // ================================================================
  // approximate_minimum_degree
  // ================================================================

  using sparkit::data::detail::approximate_minimum_degree;
  using sparkit::data::detail::inverse_permutation;

  // Right-looking symbolic Cholesky on a symmetric pattern.
  // Returns the total number of nonzeros in the lower-triangular
  // factor L (including diagonal). Used to measure fill-in.
  static size_type
  symbolic_cholesky_nnz(Compressed_row_sparsity const& sp)
  {
    auto sym = symmetrize_pattern(sp);
    auto n = sym.shape().row();
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    // Initialize L with the lower triangle of the symmetric pattern
    std::vector<std::set<size_type>> rows(static_cast<std::size_t>(n));
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] <= i) {
          rows[static_cast<std::size_t>(i)].insert(ci[p]);
        }
      }
    }

    // Simulate elimination: when pivot k is eliminated, all pairs
    // (i,j) with i,j > k and both in column k of L create fill.
    for (size_type k = 0; k < n; ++k) {
      std::vector<size_type> col_k;
      for (size_type i = k + 1; i < n; ++i) {
        if (rows[static_cast<std::size_t>(i)].count(k)) {
          col_k.push_back(i);
        }
      }
      for (auto i : col_k) {
        for (auto j : col_k) {
          if (j <= i) {
            rows[static_cast<std::size_t>(i)].insert(j);
          }
        }
      }
    }

    size_type total = 0;
    for (size_type i = 0; i < n; ++i) {
      total += static_cast<size_type>(rows[static_cast<std::size_t>(i)].size());
    }
    return total;
  }

  // -- Validation --

  TEST_CASE("reordering - amd valid permutation", "[reordering]")
  {
    // Tridiagonal 4x4
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - amd rectangular rejected", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{3, 4}, {
      Index{0, 0}, Index{1, 1}, Index{2, 2}
    }};

    CHECK_THROWS(approximate_minimum_degree(sp));
  }

  // -- Small known examples --

  TEST_CASE("reordering - amd diagonal matrix", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - amd tridiagonal", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));

    // Tridiagonal already has minimal fill; bandwidth should stay small
    Compressed_row_matrix<double> A{sp, [](auto, auto) { return 1.0; }};
    auto B = dperm(A, perm);
    auto [lo, hi] = bandwidth(B);
    CHECK(lo <= 2);
    CHECK(hi <= 2);
  }

  TEST_CASE("reordering - amd arrow matrix", "[reordering]")
  {
    // Node 0 is the hub, connected to all others
    Compressed_row_sparsity sp{Shape{5, 5}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 2},
      Index{3, 0}, Index{3, 3},
      Index{4, 0}, Index{4, 4}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 5);
    CHECK(is_valid_permutation(perm));

    // The hub node (degree 4) should be among the last eliminated
    CHECK(perm[0] >= 3);
  }

  TEST_CASE("reordering - amd path graph", "[reordering]")
  {
    // Path: 0-1-2-3-4
    Compressed_row_sparsity sp{Shape{5, 5}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}, Index{3, 4},
      Index{4, 3}, Index{4, 4}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 5);
    CHECK(is_valid_permutation(perm));

    // An endpoint (degree 1) should be eliminated first
    CHECK((perm[0] == 0 || perm[4] == 0));
  }

  // -- Properties --

  TEST_CASE("reordering - amd disconnected components", "[reordering]")
  {
    // Two disconnected components: {0,1} and {2,3}
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1},
      Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - amd fill reduction", "[reordering]")
  {
    // Arrow matrix: natural ordering creates massive fill
    Compressed_row_sparsity sp{Shape{5, 5}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 2},
      Index{3, 0}, Index{3, 3},
      Index{4, 0}, Index{4, 4}
    }};

    auto natural_fill = symbolic_cholesky_nnz(sp);

    auto perm = approximate_minimum_degree(sp);
    auto reordered = dperm(sp, perm);
    auto amd_fill = symbolic_cholesky_nnz(reordered);

    CHECK(amd_fill <= natural_fill);
  }

  TEST_CASE("reordering - amd symmetric input", "[reordering]")
  {
    // Already-symmetric 3x3 tridiagonal
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}, Index{1, 2},
      Index{2, 1}, Index{2, 2}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 3);
    CHECK(is_valid_permutation(perm));
  }

  // -- Comparison --

  TEST_CASE("reordering - amd vs rcm arrow", "[reordering]")
  {
    // Arrow matrix: AMD should produce less or equal fill than RCM
    Compressed_row_sparsity sp{Shape{5, 5}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3}, Index{0, 4},
      Index{1, 0}, Index{1, 1},
      Index{2, 0}, Index{2, 2},
      Index{3, 0}, Index{3, 3},
      Index{4, 0}, Index{4, 4}
    }};

    auto amd_perm = approximate_minimum_degree(sp);
    auto rcm_perm = reverse_cuthill_mckee(sp);

    auto amd_reordered = dperm(sp, amd_perm);
    auto rcm_reordered = dperm(sp, rcm_perm);

    auto amd_fill = symbolic_cholesky_nnz(amd_reordered);
    auto rcm_fill = symbolic_cholesky_nnz(rcm_reordered);

    CHECK(amd_fill <= rcm_fill);
  }

  TEST_CASE("reordering - amd round trip values", "[reordering]")
  {
    Compressed_row_matrix<double> A{Shape{4, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}, {Index{1, 2}, 5.0},
      {Index{2, 1}, 6.0}, {Index{2, 2}, 7.0}, {Index{2, 3}, 8.0},
      {Index{3, 2}, 9.0}, {Index{3, 3}, 10.0}
    }};

    auto perm = approximate_minimum_degree(A.sparsity());
    auto B = dperm(A, perm);

    CHECK(B.size() == A.size());

    auto inv = inverse_permutation(perm);
    auto C = dperm(B, inv);

    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        CHECK(C(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  // -- Edge cases --

  TEST_CASE("reordering - amd single node", "[reordering]")
  {
    // Minimal 2x2 diagonal matrix
    Compressed_row_sparsity sp{Shape{2, 2}, {
      Index{0, 0}, Index{1, 1}
    }};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 2);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - amd dense block", "[reordering]")
  {
    // Fully connected 4x4
    std::vector<Index> indices;
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        indices.push_back(Index{i, j});
      }
    }
    Compressed_row_sparsity sp{Shape{4, 4}, indices.begin(), indices.end()};

    auto perm = approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  // ================================================================
  // column_approximate_minimum_degree (COLAMD)
  // ================================================================

  using sparkit::data::detail::column_approximate_minimum_degree;
  using sparkit::data::detail::cperm;

  // Build the sparsity pattern of A^T*A from A's CSR sparsity.
  // Used for measuring fill in COLAMD tests.
  static Compressed_row_sparsity
  test_form_ata(Compressed_row_sparsity const& sp)
  {
    auto nrow = sp.shape().row();
    auto ncol = sp.shape().column();
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();

    std::vector<Index> indices;
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        for (auto k = rp[row]; k < rp[row + 1]; ++k) {
          indices.push_back(Index{ci[j], ci[k]});
        }
      }
    }

    return Compressed_row_sparsity{Shape{ncol, ncol},
                                    indices.begin(), indices.end()};
  }

  // -- Validation --

  TEST_CASE("reordering - colamd valid permutation", "[reordering]")
  {
    // Unsymmetric 4x4
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 1}, Index{1, 2},
      Index{2, 0}, Index{2, 2}, Index{2, 3},
      Index{3, 1}, Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - colamd rectangular", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{4, 5}, {
      Index{0, 0}, Index{0, 2},
      Index{1, 1}, Index{1, 3},
      Index{2, 2}, Index{2, 4},
      Index{3, 0}, Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 5);
    CHECK(is_valid_permutation(perm));
  }

  // -- Known examples --

  TEST_CASE("reordering - colamd diagonal", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{1, 1}, Index{2, 2}, Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - colamd arrow columns", "[reordering]")
  {
    // Column 0 appears in every row — A^T*A is an arrow with node 0 as hub
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 2},
      Index{2, 0}, Index{2, 3},
      Index{3, 0}, Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
    // The dense column (col 0) should be ordered late
    // (not first — AMD reduces its degree as leaves are eliminated)
    CHECK(perm[0] >= 2);
  }

  TEST_CASE("reordering - colamd square unsymmetric", "[reordering]")
  {
    // Upper triangular 4x4
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1}, Index{0, 2}, Index{0, 3},
      Index{1, 1}, Index{1, 2}, Index{1, 3},
      Index{2, 2}, Index{2, 3},
      Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  // -- Properties --

  TEST_CASE("reordering - colamd fill reduction", "[reordering]")
  {
    // Column 0 appears in every row (arrow in A^T*A)
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 2},
      Index{2, 0}, Index{2, 3},
      Index{3, 0}, Index{3, 3}
    }};

    auto ata = test_form_ata(sp);
    auto natural_fill = symbolic_cholesky_nnz(ata);

    auto perm = column_approximate_minimum_degree(sp);
    auto reordered_ata = dperm(ata, perm);
    auto colamd_fill = symbolic_cholesky_nnz(reordered_ata);

    CHECK(colamd_fill <= natural_fill);
  }

  TEST_CASE("reordering - colamd round trip cperm", "[reordering]")
  {
    Compressed_row_matrix<double> A{Shape{4, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}, {Index{2, 3}, 7.0},
      {Index{3, 1}, 8.0}, {Index{3, 3}, 9.0}
    }};

    auto perm = column_approximate_minimum_degree(A.sparsity());
    auto B = cperm(A, perm);
    auto inv = inverse_permutation(perm);
    auto C = cperm(B, inv);

    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        CHECK(C(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("reordering - colamd disconnected columns", "[reordering]")
  {
    // Block diagonal: cols {0,1} and {2,3} never share a row
    Compressed_row_sparsity sp{Shape{4, 4}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1},
      Index{2, 2}, Index{2, 3},
      Index{3, 2}, Index{3, 3}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

  // -- Edge cases --

  TEST_CASE("reordering - colamd minimal", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{2, 2}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 0}, Index{1, 1}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 2);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - colamd tall thin", "[reordering]")
  {
    Compressed_row_sparsity sp{Shape{5, 2}, {
      Index{0, 0},
      Index{1, 0}, Index{1, 1},
      Index{2, 1},
      Index{3, 0},
      Index{4, 0}, Index{4, 1}
    }};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 2);
    CHECK(is_valid_permutation(perm));
  }

  TEST_CASE("reordering - colamd dense block", "[reordering]")
  {
    std::vector<Index> indices;
    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        indices.push_back(Index{i, j});
      }
    }
    Compressed_row_sparsity sp{Shape{4, 4}, indices.begin(), indices.end()};

    auto perm = column_approximate_minimum_degree(sp);

    REQUIRE(std::ssize(perm) == 4);
    CHECK(is_valid_permutation(perm));
  }

} // end of namespace sparkit::testing
