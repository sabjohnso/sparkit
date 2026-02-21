//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <algorithm>
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

} // end of namespace sparkit::testing
