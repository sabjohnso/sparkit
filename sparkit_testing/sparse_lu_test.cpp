//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/sparse_lu.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::cperm;
  using sparkit::data::detail::lu_apply;
  using sparkit::data::detail::lu_solve;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::rperm;
  using sparkit::data::detail::sparse_lu;

  using size_type = sparkit::config::size_type;

  // Build a CSR matrix from a list of (row, col, value) entries.
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

  // Dense reconstruction: compute product of two sparse matrices as dense.
  static std::vector<std::vector<double>>
  dense_product(
    Compressed_row_matrix<double> const& A,
    Compressed_row_matrix<double> const& B) {
    auto m = A.shape().row();
    auto n = B.shape().column();
    auto un = static_cast<std::size_t>(n);
    auto um = static_cast<std::size_t>(m);
    std::vector<std::vector<double>> C(um, std::vector<double>(un, 0.0));

    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_v = A.values();
    auto b_rp = B.row_ptr();
    auto b_ci = B.col_ind();
    auto b_v = B.values();

    for (size_type i = 0; i < m; ++i) {
      for (auto pa = a_rp[i]; pa < a_rp[i + 1]; ++pa) {
        auto k = a_ci[pa];
        for (auto pb = b_rp[k]; pb < b_rp[k + 1]; ++pb) {
          C[static_cast<std::size_t>(i)][static_cast<std::size_t>(b_ci[pb])] +=
            a_v[pa] * b_v[pb];
        }
      }
    }
    return C;
  }

  // Dense matrix from sparse (for permutation reconstruction checks).
  static std::vector<std::vector<double>>
  to_dense(Compressed_row_matrix<double> const& A) {
    auto m = A.shape().row();
    auto n = A.shape().column();
    auto um = static_cast<std::size_t>(m);
    auto un = static_cast<std::size_t>(n);
    std::vector<std::vector<double>> D(um, std::vector<double>(un, 0.0));

    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto v = A.values();

    for (size_type i = 0; i < m; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        D[static_cast<std::size_t>(i)][static_cast<std::size_t>(ci[p])] = v[p];
      }
    }
    return D;
  }

  // --- Test 1: Non-square throws ---

  TEST_CASE("sparse_lu: non-square throws", "[sparse_lu]") {
    auto A =
      make_matrix(Shape{3, 4}, {{{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}});

    REQUIRE_THROWS_AS(sparse_lu(A, false), std::invalid_argument);
  }

  // --- Test 2: Diagonal matrix ---

  TEST_CASE("sparse_lu: diagonal matrix", "[sparse_lu]") {
    auto A =
      make_matrix(Shape{3, 3}, {{{0, 0}, 2.0}, {{1, 1}, 5.0}, {{2, 2}, 3.0}});

    auto factors = sparse_lu(A, false);

    // L should be identity
    auto l_rp = factors.L.row_ptr();
    auto l_ci = factors.L.col_ind();
    auto l_v = factors.L.values();

    for (size_type i = 0; i < 3; ++i) {
      REQUIRE(l_rp[i + 1] - l_rp[i] == 1);
      REQUIRE(l_ci[l_rp[i]] == i);
      REQUIRE(l_v[l_rp[i]] == Catch::Approx(1.0));
    }

    // U should be diag(A)
    auto u_rp = factors.U.row_ptr();
    auto u_ci = factors.U.col_ind();
    auto u_v = factors.U.values();

    for (size_type i = 0; i < 3; ++i) {
      REQUIRE(u_rp[i + 1] - u_rp[i] == 1);
      REQUIRE(u_ci[u_rp[i]] == i);
    }
    REQUIRE(u_v[u_rp[0]] == Catch::Approx(2.0));
    REQUIRE(u_v[u_rp[1]] == Catch::Approx(5.0));
    REQUIRE(u_v[u_rp[2]] == Catch::Approx(3.0));
  }

  // --- Test 3: 2x2 reconstruction (L*U == P_r * A * P_c) ---

  TEST_CASE("sparse_lu: 2x2 reconstruction", "[sparse_lu]") {
    // A = [1 2; 3 4]
    auto A = make_matrix(
      Shape{2, 2},
      {{{0, 0}, 1.0}, {{0, 1}, 2.0}, {{1, 0}, 3.0}, {{1, 1}, 4.0}});

    auto factors = sparse_lu(A, false);

    // L*U should equal P_r*A (no column perm since apply_colamd=false)
    auto LU = dense_product(factors.L, factors.U);
    auto PA = to_dense(rperm(A, std::span<size_type const>{factors.row_perm}));

    for (size_type i = 0; i < 2; ++i) {
      for (size_type j = 0; j < 2; ++j) {
        REQUIRE(
          LU[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] ==
          Catch::Approx(
            PA[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));
      }
    }
  }

  // --- Test 4: Tridiagonal reconstruction ---

  TEST_CASE("sparse_lu: tridiagonal reconstruction", "[sparse_lu]") {
    // 4x4 tridiagonal: diag=4, off-diag=-1
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 4.0},
       {{0, 1}, -1.0},
       {{1, 0}, -1.0},
       {{1, 1}, 4.0},
       {{1, 2}, -1.0},
       {{2, 1}, -1.0},
       {{2, 2}, 4.0},
       {{2, 3}, -1.0},
       {{3, 2}, -1.0},
       {{3, 3}, 4.0}});

    auto factors = sparse_lu(A, false);
    auto LU = dense_product(factors.L, factors.U);
    auto PA = to_dense(rperm(A, std::span<size_type const>{factors.row_perm}));

    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        REQUIRE(
          LU[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] ==
          Catch::Approx(
            PA[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)])
            .margin(1e-12));
      }
    }
  }

  // --- Test 5: L is unit lower triangular ---

  TEST_CASE("sparse_lu: L is unit lower triangular", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 2.0},
       {{0, 1}, 1.0},
       {{1, 0}, 4.0},
       {{1, 1}, 3.0},
       {{1, 2}, 1.0},
       {{2, 1}, 2.0},
       {{2, 2}, 5.0}});

    auto factors = sparse_lu(A, false);

    auto l_rp = factors.L.row_ptr();
    auto l_ci = factors.L.col_ind();
    auto l_v = factors.L.values();
    auto n = factors.L.shape().row();

    for (size_type i = 0; i < n; ++i) {
      for (auto p = l_rp[i]; p < l_rp[i + 1]; ++p) {
        if (l_ci[p] == i) {
          // Diagonal must be 1
          REQUIRE(l_v[p] == Catch::Approx(1.0));
        } else {
          // Must be strictly below diagonal
          REQUIRE(l_ci[p] < i);
        }
      }
      // Last entry in row must be diagonal
      REQUIRE(l_ci[l_rp[i + 1] - 1] == i);
    }
  }

  // --- Test 6: U is upper triangular ---

  TEST_CASE("sparse_lu: U is upper triangular", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 2.0},
       {{0, 1}, 1.0},
       {{1, 0}, 4.0},
       {{1, 1}, 3.0},
       {{1, 2}, 1.0},
       {{2, 1}, 2.0},
       {{2, 2}, 5.0}});

    auto factors = sparse_lu(A, false);

    auto u_rp = factors.U.row_ptr();
    auto u_ci = factors.U.col_ind();
    auto n = factors.U.shape().row();

    for (size_type i = 0; i < n; ++i) {
      for (auto p = u_rp[i]; p < u_rp[i + 1]; ++p) {
        REQUIRE(u_ci[p] >= i);
      }
      // First entry in row must be diagonal
      REQUIRE(u_ci[u_rp[i]] == i);
    }
  }

  // --- Test 7: lu_solve diagonal ---

  TEST_CASE("sparse_lu: lu_solve diagonal", "[sparse_lu]") {
    auto A =
      make_matrix(Shape{3, 3}, {{{0, 0}, 2.0}, {{1, 1}, 5.0}, {{2, 2}, 3.0}});

    auto factors = sparse_lu(A, false);
    std::vector<double> b = {4.0, 10.0, 9.0};
    auto x = lu_solve(factors, std::span<double const>{b});

    REQUIRE(x[0] == Catch::Approx(2.0));
    REQUIRE(x[1] == Catch::Approx(2.0));
    REQUIRE(x[2] == Catch::Approx(3.0));
  }

  // --- Test 8: lu_solve tridiagonal ---

  TEST_CASE("sparse_lu: lu_solve tridiagonal", "[sparse_lu]") {
    // 4x4 diagonally dominant tridiagonal
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 4.0},
       {{0, 1}, -1.0},
       {{1, 0}, -1.0},
       {{1, 1}, 4.0},
       {{1, 2}, -1.0},
       {{2, 1}, -1.0},
       {{2, 2}, 4.0},
       {{2, 3}, -1.0},
       {{3, 2}, -1.0},
       {{3, 3}, 4.0}});

    std::vector<double> b = {3.0, 2.0, 2.0, 3.0};
    auto factors = sparse_lu(A, false);
    auto x = lu_solve(factors, std::span<double const>{b});

    // Verify A*x = b
    auto Ax = multiply(A, std::span<double const>{x});
    for (size_type i = 0; i < 4; ++i) {
      REQUIRE(
        Ax[static_cast<std::size_t>(i)] ==
        Catch::Approx(b[static_cast<std::size_t>(i)]).margin(1e-12));
    }
  }

  // --- Test 9: lu_solve nonsymmetric 4x4 ---

  TEST_CASE("sparse_lu: lu_solve nonsymmetric 4x4", "[sparse_lu]") {
    // Nonsymmetric: A(0,1) != A(1,0), etc.
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 5.0},
       {{0, 1}, 1.0},
       {{0, 3}, -1.0},
       {{1, 0}, -2.0},
       {{1, 1}, 6.0},
       {{1, 2}, 1.0},
       {{2, 1}, -1.0},
       {{2, 2}, 4.0},
       {{2, 3}, 2.0},
       {{3, 0}, 1.0},
       {{3, 2}, -1.0},
       {{3, 3}, 7.0}});

    std::vector<double> b = {5.0, 5.0, 5.0, 7.0};
    auto factors = sparse_lu(A, false);
    auto x = lu_solve(factors, std::span<double const>{b});

    auto Ax = multiply(A, std::span<double const>{x});
    for (size_type i = 0; i < 4; ++i) {
      REQUIRE(
        Ax[static_cast<std::size_t>(i)] ==
        Catch::Approx(b[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // --- Test 10: lu_solve zero RHS ---

  TEST_CASE("sparse_lu: lu_solve zero RHS", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 2.0},
       {{0, 1}, 1.0},
       {{1, 0}, 1.0},
       {{1, 1}, 3.0},
       {{1, 2}, 1.0},
       {{2, 1}, 1.0},
       {{2, 2}, 4.0}});

    std::vector<double> b = {0.0, 0.0, 0.0};
    auto factors = sparse_lu(A, false);
    auto x = lu_solve(factors, std::span<double const>{b});

    for (auto xi : x) {
      REQUIRE(xi == Catch::Approx(0.0).margin(1e-15));
    }
  }

  // --- Test 11: lu_solve convection-diffusion grid ---

  TEST_CASE("sparse_lu: lu_solve convdiff grid", "[sparse_lu]") {
    // 4x4 grid (16 unknowns), nonsymmetric convection-diffusion
    // -Laplacian + upwind convection in x-direction
    // 5-point stencil: center=4+h, west=-1-h, east=-1, south=-1, north=-1
    // where h = 0.5 (convection strength)
    size_type grid = 4;
    size_type n = grid * grid;
    double h = 0.5;

    std::vector<Entry<double>> entries;
    for (size_type iy = 0; iy < grid; ++iy) {
      for (size_type ix = 0; ix < grid; ++ix) {
        auto node = iy * grid + ix;
        entries.push_back({{node, node}, 4.0 + h});
        if (ix > 0) { entries.push_back({{node, node - 1}, -1.0 - h}); }
        if (ix < grid - 1) { entries.push_back({{node, node + 1}, -1.0}); }
        if (iy > 0) { entries.push_back({{node, node - grid}, -1.0}); }
        if (iy < grid - 1) { entries.push_back({{node, node + grid}, -1.0}); }
      }
    }

    auto A = make_matrix(Shape{n, n}, entries);
    std::vector<double> b(static_cast<std::size_t>(n), 1.0);

    auto factors = sparse_lu(A, false);
    auto x = lu_solve(factors, std::span<double const>{b});

    auto Ax = multiply(A, std::span<double const>{x});
    for (size_type i = 0; i < n; ++i) {
      REQUIRE(
        Ax[static_cast<std::size_t>(i)] ==
        Catch::Approx(b[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // --- Test 12: lu_solve needs pivoting ---

  TEST_CASE("sparse_lu: lu_solve needs pivoting", "[sparse_lu]") {
    // A = [0 1; 1 0] -- zero diagonal requires row swap
    auto A = make_matrix(Shape{2, 2}, {{{0, 1}, 1.0}, {{1, 0}, 1.0}});

    auto factors = sparse_lu(A, false);
    std::vector<double> b = {3.0, 7.0};
    auto x = lu_solve(factors, std::span<double const>{b});

    // x should be [7, 3]
    REQUIRE(x[0] == Catch::Approx(7.0));
    REQUIRE(x[1] == Catch::Approx(3.0));
  }

  // --- Test 13: lu_solve manufactured solution ---

  TEST_CASE("sparse_lu: lu_solve manufactured solution", "[sparse_lu]") {
    // Known exact solution x_exact = [1, 2, 3, 4]
    // Nonsymmetric A, compute b = A * x_exact
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 3.0},
       {{0, 1}, -1.0},
       {{0, 3}, 2.0},
       {{1, 0}, 1.0},
       {{1, 1}, 4.0},
       {{1, 2}, -1.0},
       {{2, 1}, 2.0},
       {{2, 2}, 5.0},
       {{2, 3}, -1.0},
       {{3, 0}, -1.0},
       {{3, 2}, 1.0},
       {{3, 3}, 6.0}});

    std::vector<double> x_exact = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_exact});

    auto factors = sparse_lu(A, false);
    auto x = lu_solve(factors, std::span<double const>{b});

    for (size_type i = 0; i < 4; ++i) {
      REQUIRE(
        x[static_cast<std::size_t>(i)] ==
        Catch::Approx(x_exact[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // --- Test 14: lu_apply preconditioner interface ---

  TEST_CASE("sparse_lu: lu_apply preconditioner", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 4.0},
       {{0, 1}, 1.0},
       {{1, 0}, 2.0},
       {{1, 1}, 5.0},
       {{1, 2}, 1.0},
       {{2, 1}, 1.0},
       {{2, 2}, 3.0}});

    auto factors = sparse_lu(A, false);
    std::vector<double> b = {5.0, 8.0, 4.0};
    std::vector<double> x(3);

    lu_apply(factors, b.begin(), b.end(), x.begin());

    // Should match lu_solve
    auto x_ref = lu_solve(factors, std::span<double const>{b});
    for (size_type i = 0; i < 3; ++i) {
      REQUIRE(
        x[static_cast<std::size_t>(i)] ==
        Catch::Approx(x_ref[static_cast<std::size_t>(i)]));
    }
  }

  // --- Test 15: Singular matrix throws ---

  TEST_CASE("sparse_lu: singular matrix throws", "[sparse_lu]") {
    // Row 0 and row 1 are identical: [1 2]
    auto A = make_matrix(
      Shape{2, 2},
      {{{0, 0}, 1.0}, {{0, 1}, 2.0}, {{1, 0}, 1.0}, {{1, 1}, 2.0}});

    REQUIRE_THROWS_AS(sparse_lu(A, false), std::domain_error);
  }

  // --- Test 16: Without COLAMD ---

  TEST_CASE("sparse_lu: without COLAMD", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 3.0},
       {{0, 1}, 1.0},
       {{1, 0}, 1.0},
       {{1, 1}, 4.0},
       {{1, 2}, 1.0},
       {{2, 1}, 1.0},
       {{2, 2}, 5.0}});

    auto factors = sparse_lu(A, false);

    // col_perm should be empty
    REQUIRE(factors.col_perm.empty());

    // Solve should still work
    std::vector<double> b = {4.0, 6.0, 6.0};
    auto x = lu_solve(factors, std::span<double const>{b});
    auto Ax = multiply(A, std::span<double const>{x});

    for (size_type i = 0; i < 3; ++i) {
      REQUIRE(
        Ax[static_cast<std::size_t>(i)] ==
        Catch::Approx(b[static_cast<std::size_t>(i)]).margin(1e-12));
    }
  }

  // --- Test 17: With COLAMD same solution ---

  TEST_CASE("sparse_lu: with COLAMD same solution", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 5.0},
       {{0, 1}, 1.0},
       {{0, 3}, -1.0},
       {{1, 0}, -2.0},
       {{1, 1}, 6.0},
       {{1, 2}, 1.0},
       {{2, 1}, -1.0},
       {{2, 2}, 4.0},
       {{2, 3}, 2.0},
       {{3, 0}, 1.0},
       {{3, 2}, -1.0},
       {{3, 3}, 7.0}});

    std::vector<double> b = {5.0, 5.0, 5.0, 7.0};

    auto factors_no = sparse_lu(A, false);
    auto factors_yes = sparse_lu(A, true);

    auto x_no = lu_solve(factors_no, std::span<double const>{b});
    auto x_yes = lu_solve(factors_yes, std::span<double const>{b});

    for (size_type i = 0; i < 4; ++i) {
      REQUIRE(
        x_yes[static_cast<std::size_t>(i)] ==
        Catch::Approx(x_no[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // --- Test 18: Reconstruction with permutations ---

  TEST_CASE("sparse_lu: reconstruction with permutations", "[sparse_lu]") {
    // P_r * A * P_c = L * U
    auto A = make_matrix(
      Shape{4, 4},
      {{{0, 0}, 5.0},
       {{0, 1}, 1.0},
       {{0, 3}, -1.0},
       {{1, 0}, -2.0},
       {{1, 1}, 6.0},
       {{1, 2}, 1.0},
       {{2, 1}, -1.0},
       {{2, 2}, 4.0},
       {{2, 3}, 2.0},
       {{3, 0}, 1.0},
       {{3, 2}, -1.0},
       {{3, 3}, 7.0}});

    auto factors = sparse_lu(A, true);

    // Compute P_r * A * P_c
    auto PA = rperm(A, std::span<size_type const>{factors.row_perm});
    Compressed_row_matrix<double> PAP =
      factors.col_perm.empty()
        ? PA
        : cperm(PA, std::span<size_type const>{factors.col_perm});

    auto LU = dense_product(factors.L, factors.U);
    auto PAP_dense = to_dense(PAP);

    for (size_type i = 0; i < 4; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        REQUIRE(
          LU[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] ==
          Catch::Approx(
            PAP_dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)])
            .margin(1e-12));
      }
    }
  }

  // --- Test 19: Multiple RHS ---

  TEST_CASE("sparse_lu: multiple RHS", "[sparse_lu]") {
    auto A = make_matrix(
      Shape{3, 3},
      {{{0, 0}, 4.0},
       {{0, 1}, 1.0},
       {{1, 0}, 2.0},
       {{1, 1}, 5.0},
       {{1, 2}, 1.0},
       {{2, 1}, 1.0},
       {{2, 2}, 3.0}});

    auto factors = sparse_lu(A, false);

    std::vector<double> b1 = {5.0, 7.0, 4.0};
    std::vector<double> b2 = {1.0, 2.0, 3.0};

    auto x1 = lu_solve(factors, std::span<double const>{b1});
    auto x2 = lu_solve(factors, std::span<double const>{b2});

    auto Ax1 = multiply(A, std::span<double const>{x1});
    auto Ax2 = multiply(A, std::span<double const>{x2});

    for (size_type i = 0; i < 3; ++i) {
      REQUIRE(
        Ax1[static_cast<std::size_t>(i)] ==
        Catch::Approx(b1[static_cast<std::size_t>(i)]).margin(1e-12));
      REQUIRE(
        Ax2[static_cast<std::size_t>(i)] ==
        Catch::Approx(b2[static_cast<std::size_t>(i)]).margin(1e-12));
    }
  }

} // end of namespace sparkit::testing
