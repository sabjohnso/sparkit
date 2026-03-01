//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/assembly_tree.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/multifrontal_numeric.hpp>
#include <sparkit/data/multifrontal_symbolic.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Multifrontal_symbolic;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Supernode_factor;

  using sparkit::data::detail::build_assembly_tree;
  using sparkit::data::detail::dense_cholesky;
  using sparkit::data::detail::dense_syrk;
  using sparkit::data::detail::dense_trsm;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::find_supernodes;
  using sparkit::data::detail::multifrontal_analyze;
  using sparkit::data::detail::multifrontal_factorize;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // Dense kernel tests
  // ================================================================

  TEST_CASE("dense cholesky - 1x1", "[multifrontal_numeric]") {
    std::vector<double> L = {4.0};
    dense_cholesky<double>(L, 1);
    CHECK_THAT(L[0], Catch::Matchers::WithinRel(2.0, 1e-14));
  }

  TEST_CASE("dense cholesky - 2x2", "[multifrontal_numeric]") {
    // A = [4 2; 2 5], column-major: [4, 2, 2, 5]
    // L = [2 0; 1 2], column-major: [2, 1, 0, 2]
    std::vector<double> L = {4.0, 2.0, 2.0, 5.0};
    dense_cholesky<double>(L, 2);
    CHECK_THAT(L[0], Catch::Matchers::WithinRel(2.0, 1e-14));
    CHECK_THAT(L[1], Catch::Matchers::WithinRel(1.0, 1e-14));
    // L[2] is upper triangle, don't care
    CHECK_THAT(L[3], Catch::Matchers::WithinRel(2.0, 1e-14));
  }

  TEST_CASE("dense cholesky - 3x3 reconstruction", "[multifrontal_numeric]") {
    // A = [4 2 0; 2 5 1; 0 1 3], column-major
    std::vector<double> A = {4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0};
    auto L = A;
    dense_cholesky<double>(L, 3);

    // Verify L * L^T = A (lower triangle)
    for (size_type i = 0; i < 3; ++i) {
      for (size_type j = 0; j <= i; ++j) {
        double sum = 0;
        for (size_type k = 0; k <= std::min(i, j); ++k) {
          sum += L[static_cast<std::size_t>(k * 3 + i)] *
                 L[static_cast<std::size_t>(k * 3 + j)];
        }
        CHECK_THAT(
          sum,
          Catch::Matchers::WithinRel(
            A[static_cast<std::size_t>(j * 3 + i)], 1e-12));
      }
    }
  }

  TEST_CASE("dense cholesky - non-SPD throws", "[multifrontal_numeric]") {
    std::vector<double> L = {1.0, 2.0, 2.0, 1.0}; // not pos def
    CHECK_THROWS_AS(dense_cholesky<double>(L, 2), std::domain_error);
  }

  TEST_CASE("dense trsm - known answer", "[multifrontal_numeric]") {
    // L = [2 0; 1 2], column-major: [2, 1, 0, 2]
    // B = 1 row (q=1), 2 cols (p=2): B^T = [6; 5]
    // Solve L * x = [6; 5]: x = [3; 1]
    // B column-major (1x2): [6, 5]
    std::vector<double> L = {2.0, 1.0, 0.0, 2.0};
    std::vector<double> B = {6.0, 5.0};
    dense_trsm<double>(L, 2, B, 1);
    CHECK_THAT(B[0], Catch::Matchers::WithinRel(3.0, 1e-14));
    CHECK_THAT(B[1], Catch::Matchers::WithinRel(1.0, 1e-14));
  }

  TEST_CASE("dense syrk - known answer", "[multifrontal_numeric]") {
    // A = [1 3; 2 4] (2x2, column-major: [1,2,3,4])
    // C = [10 0; 0 10] initially
    // C -= A * A^T = [10-1*1-3*3, _; 0-2*1-4*3, 10-2*2-4*4]
    //             = [0, _; -14, -10]
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> C = {10.0, 0.0, 0.0, 10.0};
    dense_syrk<double>(A, 2, 2, C);
    // C is 2x2 column-major: C[0]=C(0,0), C[1]=C(1,0), C[2]=C(0,1), C[3]=C(1,1)
    // Only lower triangle updated: C(0,0)-=10, C(1,0)-=14, C(1,1)-=20
    CHECK_THAT(C[0], Catch::Matchers::WithinAbs(0.0, 1e-14));
    CHECK_THAT(C[1], Catch::Matchers::WithinRel(-14.0, 1e-14));
    CHECK_THAT(C[3], Catch::Matchers::WithinRel(-10.0, 1e-14));
  }

  // ================================================================
  // Numeric factorization tests
  // ================================================================

  // Helper: full multifrontal pipeline returning factors
  static auto
  factor(Compressed_row_matrix<double> const& A) {
    auto sp = A.sparsity();
    auto parent = elimination_tree(sp);
    auto L_pattern = symbolic_cholesky(sp);
    auto part = find_supernodes(L_pattern, parent);
    auto tree = build_assembly_tree(part, parent);
    auto sym = multifrontal_analyze(L_pattern, part, tree);
    auto factors = multifrontal_factorize(A, sym);
    return std::make_pair(std::move(sym), std::move(factors));
  }

  // Helper: extract L from multifrontal factors as a dense n×n matrix
  // (column-major) for verification
  static std::vector<double>
  extract_dense_L(
    Multifrontal_symbolic const& sym,
    std::vector<Supernode_factor<double>> const& factors) {
    auto n = sym.n;
    std::vector<double> L(static_cast<std::size_t>(n * n), 0.0);

    for (size_type s = 0; s < sym.partition.n_supernodes; ++s) {
      auto const& map = sym.maps[static_cast<std::size_t>(s)];
      auto const& fac = factors[static_cast<std::size_t>(s)];

      // Fill diagonal block
      for (size_type j = 0; j < fac.snode_size; ++j) {
        for (size_type i = j; i < fac.snode_size; ++i) {
          auto global_i = map.row_indices[static_cast<std::size_t>(i)];
          auto global_j = map.row_indices[static_cast<std::size_t>(j)];
          L[static_cast<std::size_t>(global_j * n + global_i)] =
            fac.L_diag[static_cast<std::size_t>(j * fac.snode_size + i)];
        }
      }

      // Fill sub-diagonal block
      for (size_type j = 0; j < fac.snode_size; ++j) {
        for (size_type i = 0; i < fac.update_size; ++i) {
          auto global_i =
            map.row_indices[static_cast<std::size_t>(fac.snode_size + i)];
          auto global_j = map.row_indices[static_cast<std::size_t>(j)];
          L[static_cast<std::size_t>(global_j * n + global_i)] =
            fac.L_sub[static_cast<std::size_t>(j * fac.update_size + i)];
        }
      }
    }

    return L;
  }

  // Helper: check L * L^T ≈ A for dense L (column-major)
  static void
  check_reconstruction(
    std::vector<double> const& L,
    Compressed_row_matrix<double> const& A,
    size_type n) {
    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();

    for (size_type i = 0; i < n; ++i) {
      for (auto p = a_rp[i]; p < a_rp[i + 1]; ++p) {
        auto j = a_ci[p];
        if (j > i) { continue; } // only check lower triangle

        double sum = 0;
        for (size_type k = 0; k <= std::min(i, j); ++k) {
          sum += L[static_cast<std::size_t>(k * n + i)] *
                 L[static_cast<std::size_t>(k * n + j)];
        }

        CHECK_THAT(sum, Catch::Matchers::WithinRel(a_vals[p], 1e-10));
      }
    }
  }

  TEST_CASE("multifrontal factorize - diagonal", "[multifrontal_numeric]") {
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{{0, 0}, 4.0},
       Entry<double>{{1, 1}, 9.0},
       Entry<double>{{2, 2}, 16.0},
       Entry<double>{{3, 3}, 25.0}}};

    auto [sym, factors] = factor(A);
    auto L = extract_dense_L(sym, factors);
    check_reconstruction(L, A, 4);
  }

  TEST_CASE("multifrontal factorize - 2x2", "[multifrontal_numeric]") {
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{{0, 0}, 4.0},
       Entry<double>{{0, 1}, 2.0},
       Entry<double>{{1, 0}, 2.0},
       Entry<double>{{1, 1}, 5.0}}};

    auto [sym, factors] = factor(A);
    auto L = extract_dense_L(sym, factors);
    check_reconstruction(L, A, 2);
  }

  TEST_CASE("multifrontal factorize - tridiagonal", "[multifrontal_numeric]") {
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{{0, 0}, 4.0},
       Entry<double>{{0, 1}, 1.0},
       Entry<double>{{1, 0}, 1.0},
       Entry<double>{{1, 1}, 4.0},
       Entry<double>{{1, 2}, 1.0},
       Entry<double>{{2, 1}, 1.0},
       Entry<double>{{2, 2}, 4.0},
       Entry<double>{{2, 3}, 1.0},
       Entry<double>{{3, 2}, 1.0},
       Entry<double>{{3, 3}, 4.0}}};

    auto [sym, factors] = factor(A);
    auto L = extract_dense_L(sym, factors);
    check_reconstruction(L, A, 4);
  }

  TEST_CASE("multifrontal factorize - arrow", "[multifrontal_numeric]") {
    // Arrow: diagonally dominant
    Compressed_row_matrix<double> A{
      Shape{5, 5},
      {Entry<double>{{0, 0}, 10.0},
       Entry<double>{{0, 1}, 1.0},
       Entry<double>{{0, 2}, 1.0},
       Entry<double>{{0, 3}, 1.0},
       Entry<double>{{0, 4}, 1.0},
       Entry<double>{{1, 0}, 1.0},
       Entry<double>{{1, 1}, 10.0},
       Entry<double>{{2, 0}, 1.0},
       Entry<double>{{2, 2}, 10.0},
       Entry<double>{{3, 0}, 1.0},
       Entry<double>{{3, 3}, 10.0},
       Entry<double>{{4, 0}, 1.0},
       Entry<double>{{4, 4}, 10.0}}};

    auto [sym, factors] = factor(A);
    auto L = extract_dense_L(sym, factors);
    check_reconstruction(L, A, 5);
  }

  // Helper: build a grid Laplacian as CSR matrix
  static Compressed_row_matrix<double>
  grid_laplacian(size_type grid_size) {
    auto n = grid_size * grid_size;
    std::vector<Index> indices;
    for (size_type r = 0; r < grid_size; ++r) {
      for (size_type c = 0; c < grid_size; ++c) {
        auto node = r * grid_size + c;
        indices.push_back(Index{node, node});
        if (c + 1 < grid_size) {
          indices.push_back(Index{node, node + 1});
          indices.push_back(Index{node + 1, node});
        }
        if (r + 1 < grid_size) {
          indices.push_back(Index{node, node + grid_size});
          indices.push_back(Index{node + grid_size, node});
        }
      }
    }
    Compressed_row_sparsity sp{Shape{n, n}, indices.begin(), indices.end()};
    return Compressed_row_matrix<double>{
      sp, [](size_type row, size_type col) { return row == col ? 4.0 : -1.0; }};
  }

  TEST_CASE(
    "multifrontal factorize - grid Laplacian", "[multifrontal_numeric]") {
    auto A = grid_laplacian(3);
    auto n = A.shape().row();

    auto [sym, factors] = factor(A);
    auto L = extract_dense_L(sym, factors);
    check_reconstruction(L, A, n);
  }

  TEST_CASE(
    "multifrontal factorize - non-SPD throws", "[multifrontal_numeric]") {
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{{0, 0}, 1.0},
       Entry<double>{{0, 1}, 2.0},
       Entry<double>{{1, 0}, 2.0},
       Entry<double>{{1, 1}, 1.0}}};

    CHECK_THROWS_AS(factor(A), std::domain_error);
  }

} // end of namespace sparkit::testing
