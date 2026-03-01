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
#include <sparkit/data/ldlt.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>
#include <sparkit/data/triangular_solve.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Ldl_factors;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::cholesky;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
  using sparkit::data::detail::ldlt;
  using sparkit::data::detail::ldlt_apply;
  using sparkit::data::detail::ldlt_solve;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::numeric_ldlt;
  using sparkit::data::detail::symbolic_cholesky;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  // Check that L * D * L^T == A entry-by-entry with tolerance.
  // Reconstructs the full dense product and compares.
  static void
  check_ldlt_reconstruction(
    Ldl_factors<double> const& factors,
    Compressed_row_matrix<double> const& A) {
    auto n = A.shape().row();
    auto un = static_cast<std::size_t>(n);

    auto const& L = factors.L;
    auto const& D_diag = factors.D_diag;
    auto const& D_subdiag = factors.D_subdiag;

    // Compute L * D * L^T via dense intermediate:
    // First compute D * L^T column by column, then L * (D*L^T).
    // Simpler: compute (L*D) then (L*D)*L^T.
    // Simplest: element-wise dense reconstruction.

    // Dense L
    std::vector<double> Ld(un * un, 0.0);
    auto rp = L.row_ptr();
    auto ci = L.col_ind();
    auto vals = L.values();
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        Ld[static_cast<std::size_t>(i) * un + static_cast<std::size_t>(ci[p])] =
          vals[p];
      }
    }

    // Dense D (block diagonal)
    std::vector<double> Dd(un * un, 0.0);
    for (size_type k = 0; k < n; ++k) {
      Dd[static_cast<std::size_t>(k) * un + static_cast<std::size_t>(k)] =
        D_diag[static_cast<std::size_t>(k)];
      if (static_cast<std::size_t>(k) + 1 < un) {
        auto s = D_subdiag[static_cast<std::size_t>(k)];
        if (s != 0.0) {
          Dd
            [static_cast<std::size_t>(k + 1) * un +
             static_cast<std::size_t>(k)] = s;
          Dd
            [static_cast<std::size_t>(k) * un +
             static_cast<std::size_t>(k + 1)] = s;
        }
      }
    }

    // Compute L * D
    std::vector<double> LD(un * un, 0.0);
    for (std::size_t i = 0; i < un; ++i) {
      for (std::size_t j = 0; j < un; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < un; ++k) {
          sum += Ld[i * un + k] * Dd[k * un + j];
        }
        LD[i * un + j] = sum;
      }
    }

    // Compute (L*D) * L^T
    for (size_type i = 0; i < n; ++i) {
      for (size_type j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < un; ++k) {
          sum += LD[static_cast<std::size_t>(i) * un + k] *
                 Ld[static_cast<std::size_t>(j) * un + k]; // L^T(k,j) = L(j,k)
        }
        CHECK(sum == Catch::Approx(A(i, j)).margin(1e-12));
      }
    }
  }

  // ================================================================
  // LDL^T factorization tests
  // ================================================================

  TEST_CASE("ldlt - non-square throws", "[ldlt]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    CHECK_THROWS_AS(ldlt(A), std::invalid_argument);
  }

  TEST_CASE("ldlt - diagonal matrix", "[ldlt]") {
    // A = diag(4, -3, 7, 2) -> L = I, D = diag(4, -3, 7, 2)
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, -3.0},
       Entry<double>{Index{2, 2}, 7.0},
       Entry<double>{Index{3, 3}, 2.0}}};

    auto factors = ldlt(A);

    // L should be identity
    REQUIRE(factors.L.shape().row() == 4);
    for (size_type i = 0; i < 4; ++i) {
      CHECK(factors.L(i, i) == Catch::Approx(1.0));
    }

    // D should match diagonal (including negative)
    CHECK(factors.D_diag[0] == Catch::Approx(4.0));
    CHECK(factors.D_diag[1] == Catch::Approx(-3.0));
    CHECK(factors.D_diag[2] == Catch::Approx(7.0));
    CHECK(factors.D_diag[3] == Catch::Approx(2.0));

    // All pivots should be 1x1
    for (size_type i = 0; i < 4; ++i) {
      CHECK(factors.pivot_size[static_cast<std::size_t>(i)] == 1);
    }

    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt - 2x2 SPD reconstruction", "[ldlt]") {
    // A = [[4, 2], [2, 5]]
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 5.0}}};

    auto factors = ldlt(A);
    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt - tridiag SPD reconstruction", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);
    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt - L is unit lower triangular", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);
    auto const& L = factors.L;

    auto rp = L.row_ptr();
    auto ci = L.col_ind();
    auto vals = L.values();

    for (size_type i = 0; i < L.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] <= i); // lower triangular
      }
      // Diagonal entry (last in row) should be 1.0
      auto diag_pos = rp[i + 1] - 1;
      CHECK(vals[diag_pos] == Catch::Approx(1.0));
    }
  }

  TEST_CASE("ldlt - all pivot_size=1 for SPD", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);

    for (size_type i = 0; i < 4; ++i) {
      CHECK(factors.pivot_size[static_cast<std::size_t>(i)] == 1);
    }
  }

  TEST_CASE("ldlt - 2x2 indefinite forces 2x2 pivot", "[ldlt]") {
    // A = [[eps, 1], [1, 0]] with eps very small
    // Diagonal is nearly zero so |a_kk| < alpha * lambda => 2x2 pivot
    double eps = 1e-15;
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, eps},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 0.0}}};

    auto factors = ldlt(A);

    // Should use a 2x2 pivot
    CHECK(factors.pivot_size[0] == 2);
    CHECK(factors.pivot_size[1] == 0);

    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt - 3x3 indefinite with mixed pivots", "[ldlt]") {
    // A = [[1e-16, 1, 0],
    //      [1,     0, 0],
    //      [0,     0, 5]]
    // Columns 0-1 should be a 2x2 pivot, column 2 a 1x1 pivot
    double eps = 1e-16;
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, eps},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 0.0},
       Entry<double>{Index{2, 2}, 5.0}}};

    auto factors = ldlt(A);

    CHECK(factors.pivot_size[0] == 2);
    CHECK(factors.pivot_size[1] == 0);
    CHECK(factors.pivot_size[2] == 1);

    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt - pivot_size pattern for 2x2", "[ldlt]") {
    // For a matrix where first 2 columns form a 2x2 pivot:
    // pivot_size should be {2, 0, ...}
    double eps = 1e-16;
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, eps},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 0.0},
       Entry<double>{Index{2, 2}, 5.0}}};

    auto factors = ldlt(A);

    REQUIRE(factors.pivot_size.size() == 3);
    CHECK(factors.pivot_size[0] == 2);
    CHECK(factors.pivot_size[1] == 0);
    CHECK(factors.pivot_size[2] == 1);
  }

  TEST_CASE("ldlt_solve - diagonal", "[ldlt]") {
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{1, 1}, -4.0},
       Entry<double>{Index{2, 2}, 5.0}}};

    auto factors = ldlt(A);
    std::vector<double> b = {4.0, -8.0, 15.0};
    auto x = ldlt_solve(factors, std::span<double const>{b});

    CHECK(x[0] == Catch::Approx(2.0));
    CHECK(x[1] == Catch::Approx(2.0));
    CHECK(x[2] == Catch::Approx(3.0));
  }

  TEST_CASE("ldlt_solve - SPD tridiag", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto x_ldlt = ldlt_solve(factors, std::span<double const>{b});

    // Compare with Cholesky solve
    auto L_chol = cholesky(A);
    auto y = forward_solve(L_chol, std::span<double const>{b});
    auto x_chol = forward_solve_transpose(L_chol, std::span<double const>{y});

    REQUIRE(x_ldlt.size() == x_chol.size());
    for (std::size_t i = 0; i < x_ldlt.size(); ++i) {
      CHECK(x_ldlt[i] == Catch::Approx(x_chol[i]).margin(1e-12));
    }
  }

  TEST_CASE("ldlt_solve - indefinite", "[ldlt]") {
    // A = [[0, 1], [1, 0]] -> eigenvalues +1, -1
    // A * [1, 1]^T = [1, 1]^T
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 0.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 0.0}}};

    auto factors = ldlt(A);
    std::vector<double> b = {1.0, 1.0};
    auto x = ldlt_solve(factors, std::span<double const>{b});

    CHECK(x[0] == Catch::Approx(1.0).margin(1e-12));
    CHECK(x[1] == Catch::Approx(1.0).margin(1e-12));
  }

  TEST_CASE("ldlt_solve - zero RHS", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);

    std::vector<double> b = {0.0, 0.0, 0.0, 0.0};
    auto x = ldlt_solve(factors, std::span<double const>{b});

    for (std::size_t i = 0; i < x.size(); ++i) {
      CHECK(x[i] == Catch::Approx(0.0).margin(1e-15));
    }
  }

  TEST_CASE("ldlt_solve - grid Laplacian", "[ldlt]") {
    // 4x4 grid Laplacian + 5*I (16 nodes), SPD
    size_type const grid = 4;
    size_type const n = grid * grid;

    std::vector<Entry<double>> entries;
    for (size_type r = 0; r < grid; ++r) {
      for (size_type c = 0; c < grid; ++c) {
        auto node = r * grid + c;
        size_type degree = 0;
        if (c > 0) {
          entries.push_back(Entry<double>{Index{node, node - 1}, -1.0});
          ++degree;
        }
        if (c + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + 1}, -1.0});
          ++degree;
        }
        if (r > 0) {
          entries.push_back(Entry<double>{Index{node, node - grid}, -1.0});
          ++degree;
        }
        if (r + 1 < grid) {
          entries.push_back(Entry<double>{Index{node, node + grid}, -1.0});
          ++degree;
        }
        entries.push_back(
          Entry<double>{Index{node, node}, static_cast<double>(degree) + 5.0});
      }
    }

    auto A = make_matrix(Shape{n, n}, entries);
    auto factors = ldlt(A);

    // Manufactured solution: x_true = [1, 2, ..., n]
    auto un = static_cast<std::size_t>(n);
    std::vector<double> x_true(un);
    for (std::size_t i = 0; i < un; ++i) {
      x_true[i] = static_cast<double>(i + 1);
    }

    // b = A * x_true
    auto b = multiply(A, std::span<double const>{x_true});
    auto x = ldlt_solve(factors, std::span<double const>{b});

    for (std::size_t i = 0; i < un; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-10));
    }
  }

  TEST_CASE("ldlt - saddle-point reconstruction", "[ldlt]") {
    // KKT / saddle-point matrix:
    // A = [[ 4, 1, 1, 0],
    //      [ 1, 4, 0, 1],
    //      [ 1, 0, 0, 0],
    //      [ 0, 1, 0, 0]]
    // Top-left 2x2 is SPD, bottom-right 2x2 is zero.
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{0, 2}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 3}, 1.0},
       Entry<double>{Index{2, 0}, 1.0},
       Entry<double>{Index{2, 2}, 0.0},
       Entry<double>{Index{3, 1}, 1.0},
       Entry<double>{Index{3, 3}, 0.0}}};

    auto factors = ldlt(A);
    check_ldlt_reconstruction(factors, A);
  }

  TEST_CASE("ldlt_solve - saddle point", "[ldlt]") {
    // Same KKT matrix as above
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{0, 2}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{1, 3}, 1.0},
       Entry<double>{Index{2, 0}, 1.0},
       Entry<double>{Index{2, 2}, 0.0},
       Entry<double>{Index{3, 1}, 1.0},
       Entry<double>{Index{3, 3}, 0.0}}};

    auto factors = ldlt(A);

    // Manufactured solution
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});
    auto x = ldlt_solve(factors, std::span<double const>{b});

    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-12));
    }
  }

  TEST_CASE("ldlt_apply - preconditioner", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto factors = ldlt(A);

    std::vector<double> r = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> z(4, 0.0);

    ldlt_apply(factors, r.begin(), r.end(), z.begin());

    // z should satisfy A * z ≈ r (since LDL^T is exact factorization)
    auto Az = multiply(A, std::span<double const>{z});
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(Az[i] == Catch::Approx(r[i]).margin(1e-12));
    }
  }

  TEST_CASE("ldlt - singular zero pivot throws", "[ldlt]") {
    // A = [[0, 0], [0, 1]]
    // Column 0 has zero diagonal and zero off-diagonal: singular
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 0.0}, Entry<double>{Index{1, 1}, 1.0}}};

    CHECK_THROWS_AS(ldlt(A), std::domain_error);
  }

  TEST_CASE("ldlt - pattern reuse", "[ldlt]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L_pattern = symbolic_cholesky(A.sparsity());

    auto factors_separate = numeric_ldlt(A, L_pattern);
    auto factors_combined = ldlt(A);

    // Both should produce the same D
    REQUIRE(factors_separate.D_diag.size() == factors_combined.D_diag.size());
    for (std::size_t i = 0; i < factors_separate.D_diag.size(); ++i) {
      CHECK(
        factors_separate.D_diag[i] ==
        Catch::Approx(factors_combined.D_diag[i]));
    }

    // Both should produce the same L values
    auto sep_vals = factors_separate.L.values();
    auto com_vals = factors_combined.L.values();
    REQUIRE(sep_vals.size() == com_vals.size());
    for (std::size_t i = 0; i < sep_vals.size(); ++i) {
      CHECK(sep_vals[i] == Catch::Approx(com_vals[i]));
    }
  }

} // end of namespace sparkit::testing
