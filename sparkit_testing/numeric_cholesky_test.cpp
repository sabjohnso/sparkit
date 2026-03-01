//
// ... Test header files
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/matgen.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::arrow_matrix;
  using sparkit::data::detail::cholesky;
  using sparkit::data::detail::make_matrix;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::numeric_cholesky;
  using sparkit::data::detail::symbolic_cholesky;
  using sparkit::data::detail::transpose;
  using sparkit::data::detail::tridiagonal_matrix;

  using size_type = sparkit::config::size_type;

  // Check that L * L^T == A entry-by-entry with tolerance.
  static void
  check_reconstruction(
    Compressed_row_matrix<double> const& L,
    Compressed_row_matrix<double> const& A) {
    auto Lt = transpose(L);
    auto LLt = multiply(L, Lt);

    auto rows = A.shape().row();
    auto cols = A.shape().column();
    for (size_type i = 0; i < rows; ++i) {
      for (size_type j = 0; j < cols; ++j) {
        CHECK(LLt(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
      }
    }
  }

  // ================================================================
  // numeric_cholesky
  // ================================================================

  TEST_CASE("numeric cholesky - diagonal", "[numeric_cholesky]") {
    // A = diag(4, 9, 16, 25)  ->  L = diag(2, 3, 4, 5)
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0},
       Entry<double>{Index{3, 3}, 25.0}}};

    auto L = cholesky(A);

    REQUIRE(L.shape().row() == 4);
    REQUIRE(L.shape().column() == 4);
    REQUIRE(L.size() == 4);

    CHECK(L(0, 0) == Catch::Approx(2.0));
    CHECK(L(1, 1) == Catch::Approx(3.0));
    CHECK(L(2, 2) == Catch::Approx(4.0));
    CHECK(L(3, 3) == Catch::Approx(5.0));
  }

  TEST_CASE("numeric cholesky - small 2x2", "[numeric_cholesky]") {
    // A = [[4, 2], [2, 5]]  ->  L = [[2, 0], [1, 2]]
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 5.0}}};

    auto L = cholesky(A);

    REQUIRE(L.shape().row() == 2);
    REQUIRE(L.shape().column() == 2);
    REQUIRE(L.size() == 3); // lower triangle: (0,0), (1,0), (1,1)

    CHECK(L(0, 0) == Catch::Approx(2.0));
    CHECK(L(1, 0) == Catch::Approx(1.0));
    CHECK(L(1, 1) == Catch::Approx(2.0));
  }

  TEST_CASE(
    "numeric cholesky - tridiagonal reconstruction", "[numeric_cholesky]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L = cholesky(A);
    check_reconstruction(L, A);
  }

  TEST_CASE("numeric cholesky - arrow reconstruction", "[numeric_cholesky]") {
    auto A = arrow_matrix<double>(5, 10.0, 1.0);
    auto L = cholesky(A);
    check_reconstruction(L, A);
  }

  TEST_CASE("numeric cholesky - lower triangular", "[numeric_cholesky]") {
    auto A = arrow_matrix<double>(5, 10.0, 1.0);
    auto L = cholesky(A);

    auto rp = L.row_ptr();
    auto ci = L.col_ind();
    for (size_type i = 0; i < L.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] <= i);
      }
    }
  }

  TEST_CASE(
    "numeric cholesky - pattern matches symbolic", "[numeric_cholesky]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L_pattern = symbolic_cholesky(A.sparsity());
    auto L = cholesky(A);

    REQUIRE(L.sparsity().size() == L_pattern.size());

    auto L_rp = L.row_ptr();
    auto P_rp = L_pattern.row_ptr();
    for (size_type i = 0; i <= L.shape().row(); ++i) {
      CHECK(L_rp[i] == P_rp[i]);
    }

    auto L_ci = L.col_ind();
    auto P_ci = L_pattern.col_ind();
    for (size_type i = 0; i < L.sparsity().size(); ++i) {
      CHECK(L_ci[i] == P_ci[i]);
    }
  }

  TEST_CASE(
    "numeric cholesky - separate symbolic numeric", "[numeric_cholesky]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L_pattern = symbolic_cholesky(A.sparsity());
    auto L_separate = numeric_cholesky(A, L_pattern);
    auto L_combined = cholesky(A);

    REQUIRE(L_separate.size() == L_combined.size());
    auto sep_vals = L_separate.values();
    auto com_vals = L_combined.values();
    for (size_type i = 0; i < L_separate.size(); ++i) {
      CHECK(sep_vals[i] == Catch::Approx(com_vals[i]));
    }
  }

  TEST_CASE("numeric cholesky - grid reconstruction", "[numeric_cholesky]") {
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
    auto L = cholesky(A);
    check_reconstruction(L, A);
  }

  TEST_CASE("numeric cholesky - rectangular rejected", "[numeric_cholesky]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    CHECK_THROWS_AS(cholesky(A), std::invalid_argument);
  }

  TEST_CASE("numeric cholesky - not positive definite", "[numeric_cholesky]") {
    // A = [[1, 2], [2, 1]] is symmetric but not positive definite
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 1.0}}};

    CHECK_THROWS_AS(cholesky(A), std::domain_error);
  }

  TEST_CASE(
    "numeric cholesky - diagonal values positive", "[numeric_cholesky]") {
    auto A = tridiagonal_matrix<double>(4, -1.0, 4.0, -1.0);
    auto L = cholesky(A);

    auto rp = L.row_ptr();
    for (size_type i = 0; i < L.shape().row(); ++i) {
      // Diagonal is last entry in each row (symbolic_cholesky convention)
      auto diag_pos = rp[i + 1] - 1;
      auto diag_val = L.values()[diag_pos];
      CHECK(diag_val > 0.0);
    }
  }

} // end of namespace sparkit::testing
