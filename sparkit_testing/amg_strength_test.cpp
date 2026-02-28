//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/amg_strength.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::strength_of_connection;

  using size_type = sparkit::config::size_type;

  static Compressed_row_matrix<double>
  make_matrix(Shape shape, std::vector<Entry<double>> const& entries) {
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Compressed_row_sparsity sp{shape, indices.begin(), indices.end()};

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

    return Compressed_row_matrix<double>{std::move(sp), std::move(vals)};
  }

  // ================================================================
  // Strength of connection tests
  // ================================================================

  TEST_CASE(
    "strength - diagonal matrix has no strong connections", "[amg_strength]") {
    // Pure diagonal: no off-diagonal entries, so no strong connections.
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 3}, 4.0}});

    auto S = strength_of_connection(A, 0.25);
    CHECK(S.size() == 0);
  }

  TEST_CASE(
    "strength - uniform tridiag all strong at low theta", "[amg_strength]") {
    // Tridiag with equal off-diag: all off-diag are strong at any theta <= 1.
    // |a_ij| = 1, max_{k!=i} |a_ik| = 1, so |a_ij| >= theta * 1 for theta <= 1.
    std::vector<Entry<double>> entries;
    size_type const n = 4;
    for (size_type i = 0; i < n; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < n) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    auto A = make_matrix(Shape{n, n}, entries);

    auto S = strength_of_connection(A, 0.25);

    // 6 off-diag entries, all strong. Symmetric: both (i,j) and (j,i).
    CHECK(S.size() == 6);

    // Check symmetry: for each (i,j) in S, (j,i) is also in S.
    auto rp = S.row_ptr();
    auto ci = S.col_ind();
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        // j should have i as a neighbor
        bool found = false;
        for (auto q = rp[j]; q < rp[j + 1]; ++q) {
          if (ci[q] == i) {
            found = true;
            break;
          }
        }
        CHECK(found);
      }
    }
  }

  TEST_CASE("strength - mixed strength 4x4", "[amg_strength]") {
    // Row 0: diag=4, a01=-1, a02=-0.01
    // max_off_diag(row 0) = 1.0
    // At theta=0.25: |a01|=1 >= 0.25 (strong), |a02|=0.01 < 0.25 (weak)
    auto A = make_matrix(
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{0, 2}, -0.01},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 3}, -1.0},
       Entry<double>{Index{2, 0}, -0.01},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{2, 3}, -1.0},
       Entry<double>{Index{3, 1}, -1.0},
       Entry<double>{Index{3, 2}, -1.0},
       Entry<double>{Index{3, 3}, 4.0}});

    auto S = strength_of_connection(A, 0.25);

    // Strong connections: (0,1),(1,0),(1,3),(3,1),(2,3),(3,2) = 6
    // Weak: (0,2),(2,0) since 0.01 < 0.25*1.0
    CHECK(S.size() == 6);

    // Verify (0,1) is strong
    auto rp = S.row_ptr();
    auto ci = S.col_ind();
    bool found_01 = false;
    for (auto p = rp[0]; p < rp[0 + 1]; ++p) {
      if (ci[p] == 1) found_01 = true;
    }
    CHECK(found_01);

    // Verify (0,2) is NOT strong
    bool found_02 = false;
    for (auto p = rp[0]; p < rp[0 + 1]; ++p) {
      if (ci[p] == 2) found_02 = true;
    }
    CHECK_FALSE(found_02);
  }

  TEST_CASE("strength - theta=0 makes all off-diag strong", "[amg_strength]") {
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -1.0},
       Entry<double>{Index{0, 2}, -0.001},
       Entry<double>{Index{1, 0}, -1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 0}, -0.001},
       Entry<double>{Index{2, 2}, 4.0}});

    auto S = strength_of_connection(A, 0.0);

    // All 4 off-diag entries should be strong
    CHECK(S.size() == 4);
  }

  TEST_CASE(
    "strength - diagonal excluded from strong connections", "[amg_strength]") {
    auto A = make_matrix(
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, -2.0},
       Entry<double>{Index{1, 0}, -2.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 2}, -2.0},
       Entry<double>{Index{2, 1}, -2.0},
       Entry<double>{Index{2, 2}, 4.0}});

    auto S = strength_of_connection(A, 0.25);

    // No diagonal entries should appear in strength graph
    auto rp = S.row_ptr();
    auto ci = S.col_ind();
    for (size_type i = 0; i < 3; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] != i);
      }
    }
  }

  TEST_CASE("strength - 4x4 grid Laplacian", "[amg_strength]") {
    // 2D Laplacian on 4x4 grid: diag=4, off-diag=-1
    // All off-diag equally strong: all pass at theta=0.25
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
    auto A = make_matrix(Shape{n, n}, entries);

    auto S = strength_of_connection(A, 0.25);

    // Count total off-diagonal entries in A
    size_type off_diag_count = 0;
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] != i) ++off_diag_count;
      }
    }

    // All off-diag are equally strong, so all should be in S
    CHECK(S.size() == off_diag_count);
  }

} // end of namespace sparkit::testing
