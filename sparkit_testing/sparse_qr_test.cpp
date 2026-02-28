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
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/sparse_qr.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::multiply;
  using sparkit::data::detail::qr;
  using sparkit::data::detail::qr_apply_q;
  using sparkit::data::detail::qr_apply_qt;
  using sparkit::data::detail::qr_solve;
  using sparkit::data::detail::transpose;

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

  // ================================================================
  // Phase 1: QR factorization structure and values
  // ================================================================

  TEST_CASE("sparse QR - diagonal matrix", "[sparse_qr]") {
    // A = diag(3, -4, 5)  ->  |R(i,i)| = |A(i,i)|
    Compressed_row_matrix<double> A{
      Shape{3, 3},
      {Entry<double>{Index{0, 0}, 3.0},
       Entry<double>{Index{1, 1}, -4.0},
       Entry<double>{Index{2, 2}, 5.0}}};

    auto factors = qr(A, false);
    auto const& R = factors.R;

    REQUIRE(R.shape().row() == 3);
    REQUIRE(R.shape().column() == 3);
    REQUIRE(R.size() == 3);

    CHECK(std::abs(R(0, 0)) == Catch::Approx(3.0));
    CHECK(std::abs(R(1, 1)) == Catch::Approx(4.0));
    CHECK(std::abs(R(2, 2)) == Catch::Approx(5.0));
  }

  TEST_CASE("sparse QR - 2x2 hand computed", "[sparse_qr]") {
    // A = [[3, 0], [4, 5]]
    // ||col0|| = 5, so R(0,0) = ±5
    // Q^T * col1 gives R(0,1) and R(1,1)
    Compressed_row_matrix<double> A{
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 3.0},
       Entry<double>{Index{1, 0}, 4.0},
       Entry<double>{Index{1, 1}, 5.0}}};

    auto factors = qr(A, false);
    auto const& R = factors.R;

    REQUIRE(R.shape().row() == 2);
    REQUIRE(R.shape().column() == 2);

    // |R(0,0)| = sqrt(3^2 + 4^2) = 5
    CHECK(std::abs(R(0, 0)) == Catch::Approx(5.0));

    // |R(1,1)| = 3 (from Frobenius norm preservation: 9+16+25 = 25+R01^2+R11^2)
    // R(0,1) = (3*0 + 4*5)/5 = 4, R(1,1) = ±3
    CHECK(std::abs(R(1, 1)) == Catch::Approx(3.0));
  }

  TEST_CASE("sparse QR - R is upper triangular", "[sparse_qr]") {
    // 4x4 tridiagonal
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, 1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, 1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);
    auto factors = qr(A, false);
    auto const& R = factors.R;

    auto rp = R.row_ptr();
    auto ci = R.col_ind();
    for (size_type i = 0; i < R.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] >= i);
      }
    }
  }

  TEST_CASE("sparse QR - Frobenius norm preserved", "[sparse_qr]") {
    // A = [[1, 2], [3, 4], [5, 6]]
    Compressed_row_matrix<double> A{
      Shape{3, 2},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 3.0},
       Entry<double>{Index{1, 1}, 4.0},
       Entry<double>{Index{2, 0}, 5.0},
       Entry<double>{Index{2, 1}, 6.0}}};

    auto factors = qr(A, false);
    auto const& R = factors.R;

    // ||A||_F^2 = 1+4+9+16+25+36 = 91
    double a_norm_sq = 0.0;
    auto a_vals = A.values();
    for (size_type k = 0; k < A.size(); ++k) {
      a_norm_sq += a_vals[k] * a_vals[k];
    }

    double r_norm_sq = 0.0;
    auto r_vals = R.values();
    for (size_type k = 0; k < R.size(); ++k) {
      r_norm_sq += r_vals[k] * r_vals[k];
    }

    CHECK(r_norm_sq == Catch::Approx(a_norm_sq).epsilon(1e-12));
  }

  TEST_CASE("sparse QR - rectangular 6x4 dimensions", "[sparse_qr]") {
    // Build a 6x4 matrix with some structure
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 6; ++i) {
      size_type col = i % 4;
      entries.push_back(Entry<double>{Index{i, col}, 2.0 + i});
      if (col + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, col + 1}, 1.0});
      }
    }
    auto A = make_matrix(Shape{6, 4}, entries);
    auto factors = qr(A, false);

    CHECK(factors.R.shape().row() == 4);
    CHECK(factors.R.shape().column() == 4);
    CHECK(factors.V.shape().row() == 6);
    CHECK(factors.V.shape().column() == 4);
    CHECK(static_cast<size_type>(factors.beta.size()) == 4);
  }

  TEST_CASE("sparse QR - A = QR reconstruction", "[sparse_qr]") {
    // Verify Q*R = A (or A*P = Q*R) by checking Q^T*A columns = R columns
    // Build a well-conditioned 4x3 matrix
    Compressed_row_matrix<double> A{
      Shape{4, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 1.0},
       Entry<double>{Index{2, 1}, 1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 0}, 1.0},
       Entry<double>{Index{3, 2}, 1.0}}};

    auto factors = qr(A, false);
    auto const& R = factors.R;

    // Reconstruct: for each column j of A, Q^T * A(:,j) should give R(:,j)
    auto At = transpose(A);
    auto n = A.shape().column();
    auto m = A.shape().row();

    for (size_type j = 0; j < n; ++j) {
      // Extract column j of A as a dense vector
      std::vector<double> col_j(static_cast<std::size_t>(m), 0.0);
      auto at_rp = At.row_ptr();
      auto at_ci = At.col_ind();
      auto at_vals = At.values();
      for (auto p = at_rp[j]; p < at_rp[j + 1]; ++p) {
        col_j[static_cast<std::size_t>(at_ci[p])] = at_vals[p];
      }

      auto qt_col = qr_apply_qt(factors, std::span<double const>{col_j});

      // First n entries should match R(:,j)
      for (size_type i = 0; i < n; ++i) {
        CHECK(
          qt_col[static_cast<std::size_t>(i)] ==
          Catch::Approx(R(i, j)).margin(1e-10));
      }
    }
  }

  // ================================================================
  // Phase 2: Apply and solve
  // ================================================================

  TEST_CASE("sparse QR - Q^T Q roundtrip", "[sparse_qr]") {
    // Q^T * (Q * e_i) should return e_i (for each standard basis vector)
    Compressed_row_matrix<double> A{
      Shape{4, 3},
      {Entry<double>{Index{0, 0}, 2.0},
       Entry<double>{Index{0, 1}, 1.0},
       Entry<double>{Index{1, 0}, 1.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 1.0},
       Entry<double>{Index{2, 1}, 1.0},
       Entry<double>{Index{2, 2}, 4.0},
       Entry<double>{Index{3, 0}, 1.0},
       Entry<double>{Index{3, 2}, 1.0}}};

    auto factors = qr(A, false);
    auto m = A.shape().row();

    for (size_type i = 0; i < m; ++i) {
      std::vector<double> e_i(static_cast<std::size_t>(m), 0.0);
      e_i[static_cast<std::size_t>(i)] = 1.0;

      auto q_ei = qr_apply_q(factors, std::span<double const>{e_i});
      auto roundtrip = qr_apply_qt(factors, std::span<double const>{q_ei});

      for (size_type k = 0; k < m; ++k) {
        CHECK(
          roundtrip[static_cast<std::size_t>(k)] ==
          Catch::Approx(e_i[static_cast<std::size_t>(k)]).margin(1e-12));
      }
    }
  }

  TEST_CASE("sparse QR - solve square system", "[sparse_qr]") {
    // 4x4 tridiagonal: diag=4, off-diag=1
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, 1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, 1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);

    // b = A * [1, 2, 3, 4]^T
    std::vector<double> x_exact = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_exact});

    auto factors = qr(A, false);
    auto x = qr_solve(factors, std::span<double const>{b});

    // Check ||Ax - b|| is small
    auto Ax = multiply(A, std::span<double const>{x});
    double residual = 0.0;
    for (size_type i = 0; i < 4; ++i) {
      auto diff =
        Ax[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
      residual += diff * diff;
    }
    CHECK(std::sqrt(residual) < 1e-10);
  }

  TEST_CASE("sparse QR - solve overdetermined", "[sparse_qr]") {
    // 6x4 least-squares: check normal equations A^T*(Ax-b) ≈ 0
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 6; ++i) {
      size_type col = i % 4;
      entries.push_back(
        Entry<double>{Index{i, col}, 3.0 + static_cast<double>(i)});
      if (col + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, col + 1}, 1.0});
      }
    }
    auto A = make_matrix(Shape{6, 4}, entries);

    // b = A * [1,1,1,1] + noise (so it's not exactly in range)
    std::vector<double> x_init = {1.0, 1.0, 1.0, 1.0};
    auto b = multiply(A, std::span<double const>{x_init});
    b[0] += 0.5;
    b[2] -= 0.3;

    auto factors = qr(A, false);
    auto x = qr_solve(factors, std::span<double const>{b});

    // Normal equations: A^T * (A*x - b) should be ≈ 0
    auto Ax = multiply(A, std::span<double const>{x});
    std::vector<double> residual(6);
    for (size_type i = 0; i < 6; ++i) {
      residual[static_cast<std::size_t>(i)] =
        Ax[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
    }

    auto At = transpose(A);
    auto normal_res = multiply(At, std::span<double const>{residual});

    double norm = 0.0;
    for (size_type i = 0; i < 4; ++i) {
      norm += normal_res[static_cast<std::size_t>(i)] *
              normal_res[static_cast<std::size_t>(i)];
    }
    CHECK(std::sqrt(norm) < 1e-10);
  }

  TEST_CASE("sparse QR - solve with COLAMD", "[sparse_qr]") {
    // Same tridiagonal system but with COLAMD enabled
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, 1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, 1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);

    std::vector<double> x_exact = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_exact});

    auto factors = qr(A, true); // COLAMD enabled
    CHECK(!factors.column_perm.empty());

    auto x = qr_solve(factors, std::span<double const>{b});

    for (size_type i = 0; i < 4; ++i) {
      CHECK(
        x[static_cast<std::size_t>(i)] ==
        Catch::Approx(x_exact[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  // ================================================================
  // Phase 3: Integration and edge cases
  // ================================================================

  TEST_CASE("sparse QR - m < n rejected", "[sparse_qr]") {
    Compressed_row_matrix<double> A{
      Shape{2, 3},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 1}, 3.0},
       Entry<double>{Index{1, 2}, 4.0}}};

    CHECK_THROWS_AS(qr(A, false), std::invalid_argument);
  }

  TEST_CASE("sparse QR - compare with Cholesky on SPD system", "[sparse_qr]") {
    // For SPD A, both QR and Cholesky should give the same solution
    // A = 4x4 tridiagonal SPD: diag=4, off-diag=-1
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);

    std::vector<double> b = {1.0, 0.0, 0.0, 1.0};

    // QR solve
    auto factors = qr(A, false);
    auto x_qr = qr_solve(factors, std::span<double const>{b});

    // Cholesky solve (via L*L^T = A, then forward/backward)
    using sparkit::data::detail::cholesky;
    using sparkit::data::detail::forward_solve;
    using sparkit::data::detail::forward_solve_transpose;
    auto L = cholesky(A);
    auto y = forward_solve(L, std::span<double const>{b});
    auto x_chol = forward_solve_transpose(L, std::span<double const>{y});

    for (size_type i = 0; i < 4; ++i) {
      CHECK(
        x_qr[static_cast<std::size_t>(i)] ==
        Catch::Approx(x_chol[static_cast<std::size_t>(i)]).margin(1e-10));
    }
  }

  TEST_CASE("sparse QR - column permutation dimensions", "[sparse_qr]") {
    // With COLAMD, permutation should have length n
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(
        Entry<double>{Index{i, i}, 3.0 + static_cast<double>(i)});
      if (i + 1 < 5) {
        entries.push_back(Entry<double>{Index{i, i + 1}, 1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);
    auto factors = qr(A, true);

    auto n = A.shape().column();
    CHECK(static_cast<size_type>(factors.column_perm.size()) == n);

    // Verify it is a valid permutation
    using sparkit::data::detail::is_valid_permutation;
    CHECK(
      is_valid_permutation(std::span<size_type const>{factors.column_perm}));
  }

} // end of namespace sparkit::testing
