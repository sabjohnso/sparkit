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
#include <sparkit/data/incomplete_cholesky.hpp>
#include <sparkit/data/numeric_cholesky.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::cholesky;
  using sparkit::data::detail::extract_lower_triangle;
  using sparkit::data::detail::incomplete_cholesky;
  using sparkit::data::detail::multiply;
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
  // incomplete_cholesky tests
  // ================================================================

  TEST_CASE("incomplete cholesky - diagonal", "[incomplete_cholesky]") {
    // A = diag(4, 9, 16, 25)  ->  IC(0) == full Cholesky = diag(2, 3, 4, 5)
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{1, 1}, 9.0},
       Entry<double>{Index{2, 2}, 16.0},
       Entry<double>{Index{3, 3}, 25.0}}};

    auto L = incomplete_cholesky(A);

    REQUIRE(L.shape().row() == 4);
    REQUIRE(L.shape().column() == 4);
    REQUIRE(L.size() == 4);

    CHECK(L(0, 0) == Catch::Approx(2.0));
    CHECK(L(1, 1) == Catch::Approx(3.0));
    CHECK(L(2, 2) == Catch::Approx(4.0));
    CHECK(L(3, 3) == Catch::Approx(5.0));
  }

  TEST_CASE("incomplete cholesky - 2x2 SPD", "[incomplete_cholesky]") {
    // A = [[4, 2], [2, 5]]  ->  IC(0) == full Cholesky (no fill possible)
    auto A = make_matrix(
      Shape{2, 2},
      {Entry<double>{Index{0, 0}, 4.0},
       Entry<double>{Index{0, 1}, 2.0},
       Entry<double>{Index{1, 0}, 2.0},
       Entry<double>{Index{1, 1}, 5.0}});

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    REQUIRE(L_ic.size() == L_full.size());
    auto ic_vals = L_ic.values();
    auto full_vals = L_full.values();
    for (size_type i = 0; i < L_ic.size(); ++i) {
      CHECK(ic_vals[i] == Catch::Approx(full_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE("incomplete cholesky - tridiag 4x4", "[incomplete_cholesky]") {
    // Tridiagonal has no fill, so IC(0) == full Cholesky
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 4; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 4.0});
      if (i + 1 < 4) {
        entries.push_back(Entry<double>{Index{i, i + 1}, -1.0});
        entries.push_back(Entry<double>{Index{i + 1, i}, -1.0});
      }
    }
    auto A = make_matrix(Shape{4, 4}, entries);

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    REQUIRE(L_ic.size() == L_full.size());
    auto ic_vals = L_ic.values();
    auto full_vals = L_full.values();
    for (size_type i = 0; i < L_ic.size(); ++i) {
      CHECK(ic_vals[i] == Catch::Approx(full_vals[i]).margin(1e-12));
    }
  }

  TEST_CASE(
    "incomplete cholesky - arrow 5x5 approximation", "[incomplete_cholesky]") {
    // Arrow matrix has fill in full Cholesky, so IC(0) != full Cholesky.
    // But L*L^T should approximate A well.
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L_ic = incomplete_cholesky(A);
    auto L_full = cholesky(A);

    // IC(0) should have fewer nonzeros than full Cholesky (arrow has fill)
    CHECK(L_ic.size() <= L_full.size());

    // L*L^T should approximate A: check diagonal entries match reasonably
    auto Lt = transpose(L_ic);
    auto LLt = multiply(L_ic, Lt);
    for (size_type i = 0; i < 5; ++i) {
      CHECK(LLt(i, i) == Catch::Approx(A(i, i)).margin(0.5));
    }
  }

  TEST_CASE(
    "incomplete cholesky - L is lower triangular", "[incomplete_cholesky]") {
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L = incomplete_cholesky(A);
    auto rp = L.row_ptr();
    auto ci = L.col_ind();

    for (size_type i = 0; i < L.shape().row(); ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        CHECK(ci[p] <= i);
      }
    }
  }

  TEST_CASE(
    "incomplete cholesky - same nnz as lower triangle",
    "[incomplete_cholesky]") {
    // IC(0) factor should have exactly the same nnz as lower(A)
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);

    auto L = incomplete_cholesky(A);
    auto lower_A = extract_lower_triangle(A, true);

    CHECK(L.size() == lower_A.size());
  }

  TEST_CASE(
    "incomplete cholesky - rectangular rejected", "[incomplete_cholesky]") {
    Compressed_row_matrix<double> A{
      Shape{3, 4},
      {Entry<double>{Index{0, 0}, 1.0},
       Entry<double>{Index{1, 1}, 1.0},
       Entry<double>{Index{2, 2}, 1.0}}};

    CHECK_THROWS_AS(incomplete_cholesky(A), std::invalid_argument);
  }

} // end of namespace sparkit::testing
