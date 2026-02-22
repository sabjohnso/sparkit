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
#include <sparkit/data/triangular_solve.hpp>
#include <sparkit/data/unary.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::backward_solve;
  using sparkit::data::detail::cholesky;
  using sparkit::data::detail::forward_solve;
  using sparkit::data::detail::forward_solve_transpose;
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
  // forward_solve tests
  // ================================================================

  TEST_CASE("forward solve - identity", "[triangular_solve]") {
    // L = 4x4 identity, b = [1,2,3,4] -> x = [1,2,3,4]
    auto L = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 1.0},
                                       Entry<double>{Index{1, 1}, 1.0},
                                       Entry<double>{Index{2, 2}, 1.0},
                                       Entry<double>{Index{3, 3}, 1.0}});

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto x = forward_solve(L, std::span<double const>{b});

    REQUIRE(x.size() == 4);
    CHECK(x[0] == Catch::Approx(1.0));
    CHECK(x[1] == Catch::Approx(2.0));
    CHECK(x[2] == Catch::Approx(3.0));
    CHECK(x[3] == Catch::Approx(4.0));
  }

  TEST_CASE("forward solve - diagonal", "[triangular_solve]") {
    // L = diag(2,3,4,5), b = [6,12,20,30] -> x = [3,4,5,6]
    auto L = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 2.0},
                                       Entry<double>{Index{1, 1}, 3.0},
                                       Entry<double>{Index{2, 2}, 4.0},
                                       Entry<double>{Index{3, 3}, 5.0}});

    std::vector<double> b = {6.0, 12.0, 20.0, 30.0};
    auto x = forward_solve(L, std::span<double const>{b});

    REQUIRE(x.size() == 4);
    CHECK(x[0] == Catch::Approx(3.0));
    CHECK(x[1] == Catch::Approx(4.0));
    CHECK(x[2] == Catch::Approx(5.0));
    CHECK(x[3] == Catch::Approx(6.0));
  }

  TEST_CASE("forward solve - 2x2", "[triangular_solve]") {
    // L = [[2, 0], [1, 2]], b = [4, 5]
    // Row 0: x[0] = 4/2 = 2
    // Row 1: x[1] = (5 - 1*2)/2 = 3/2 = 1.5
    auto L = make_matrix(Shape{2, 2}, {Entry<double>{Index{0, 0}, 2.0},
                                       Entry<double>{Index{1, 0}, 1.0},
                                       Entry<double>{Index{1, 1}, 2.0}});

    std::vector<double> b = {4.0, 5.0};
    auto x = forward_solve(L, std::span<double const>{b});

    REQUIRE(x.size() == 2);
    CHECK(x[0] == Catch::Approx(2.0));
    CHECK(x[1] == Catch::Approx(1.5));
  }

  TEST_CASE("forward solve - tridiag cholesky", "[triangular_solve]") {
    // Factor tridiag SPD, then verify L*x == b
    auto A = make_matrix(
        Shape{4, 4},
        {Entry<double>{Index{0, 0}, 4.0}, Entry<double>{Index{0, 1}, -1.0},
         Entry<double>{Index{1, 0}, -1.0}, Entry<double>{Index{1, 1}, 4.0},
         Entry<double>{Index{1, 2}, -1.0}, Entry<double>{Index{2, 1}, -1.0},
         Entry<double>{Index{2, 2}, 4.0}, Entry<double>{Index{2, 3}, -1.0},
         Entry<double>{Index{3, 2}, -1.0}, Entry<double>{Index{3, 3}, 4.0}});

    auto L = cholesky(A);

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto x = forward_solve(L, std::span<double const>{b});

    // Verify: L*x should equal b
    auto Lx = multiply(L, std::span<double const>{x});
    REQUIRE(Lx.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(Lx[i] == Catch::Approx(b[i]).margin(1e-12));
    }
  }

  TEST_CASE("forward solve - arrow cholesky", "[triangular_solve]") {
    // 5x5 arrow SPD: diag=10, off-diag=1 from row 0
    std::vector<Entry<double>> entries;
    for (size_type i = 0; i < 5; ++i) {
      entries.push_back(Entry<double>{Index{i, i}, 10.0});
      if (i > 0) {
        entries.push_back(Entry<double>{Index{0, i}, 1.0});
        entries.push_back(Entry<double>{Index{i, 0}, 1.0});
      }
    }
    auto A = make_matrix(Shape{5, 5}, entries);
    auto L = cholesky(A);

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto x = forward_solve(L, std::span<double const>{b});

    auto Lx = multiply(L, std::span<double const>{x});
    REQUIRE(Lx.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(Lx[i] == Catch::Approx(b[i]).margin(1e-12));
    }
  }

  // ================================================================
  // backward_solve tests
  // ================================================================

  TEST_CASE("backward solve - identity", "[triangular_solve]") {
    auto U = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 1.0},
                                       Entry<double>{Index{1, 1}, 1.0},
                                       Entry<double>{Index{2, 2}, 1.0},
                                       Entry<double>{Index{3, 3}, 1.0}});

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto x = backward_solve(U, std::span<double const>{b});

    REQUIRE(x.size() == 4);
    CHECK(x[0] == Catch::Approx(1.0));
    CHECK(x[1] == Catch::Approx(2.0));
    CHECK(x[2] == Catch::Approx(3.0));
    CHECK(x[3] == Catch::Approx(4.0));
  }

  TEST_CASE("backward solve - diagonal", "[triangular_solve]") {
    auto U = make_matrix(Shape{4, 4}, {Entry<double>{Index{0, 0}, 2.0},
                                       Entry<double>{Index{1, 1}, 3.0},
                                       Entry<double>{Index{2, 2}, 4.0},
                                       Entry<double>{Index{3, 3}, 5.0}});

    std::vector<double> b = {6.0, 12.0, 20.0, 30.0};
    auto x = backward_solve(U, std::span<double const>{b});

    REQUIRE(x.size() == 4);
    CHECK(x[0] == Catch::Approx(3.0));
    CHECK(x[1] == Catch::Approx(4.0));
    CHECK(x[2] == Catch::Approx(5.0));
    CHECK(x[3] == Catch::Approx(6.0));
  }

  TEST_CASE("backward solve - 2x2", "[triangular_solve]") {
    // U = [[2, 1], [0, 2]], b = [5, 4]
    // Row 1: x[1] = 4/2 = 2
    // Row 0: x[0] = (5 - 1*2)/2 = 3/2 = 1.5
    auto U = make_matrix(Shape{2, 2}, {Entry<double>{Index{0, 0}, 2.0},
                                       Entry<double>{Index{0, 1}, 1.0},
                                       Entry<double>{Index{1, 1}, 2.0}});

    std::vector<double> b = {5.0, 4.0};
    auto x = backward_solve(U, std::span<double const>{b});

    REQUIRE(x.size() == 2);
    CHECK(x[0] == Catch::Approx(1.5));
    CHECK(x[1] == Catch::Approx(2.0));
  }

  // ================================================================
  // forward_solve_transpose tests
  // ================================================================

  TEST_CASE("forward solve transpose - matches backward on L^T",
            "[triangular_solve]") {
    // Factor tridiag, then:
    //   forward_solve_transpose(L, b) == backward_solve(transpose(L), b)
    auto A = make_matrix(
        Shape{4, 4},
        {Entry<double>{Index{0, 0}, 4.0}, Entry<double>{Index{0, 1}, -1.0},
         Entry<double>{Index{1, 0}, -1.0}, Entry<double>{Index{1, 1}, 4.0},
         Entry<double>{Index{1, 2}, -1.0}, Entry<double>{Index{2, 1}, -1.0},
         Entry<double>{Index{2, 2}, 4.0}, Entry<double>{Index{2, 3}, -1.0},
         Entry<double>{Index{3, 2}, -1.0}, Entry<double>{Index{3, 3}, 4.0}});

    auto L = cholesky(A);
    auto Lt = transpose(L);

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto x_transpose = forward_solve_transpose(L, std::span<double const>{b});
    auto x_explicit = backward_solve(Lt, std::span<double const>{b});

    REQUIRE(x_transpose.size() == x_explicit.size());
    for (std::size_t i = 0; i < x_transpose.size(); ++i) {
      CHECK(x_transpose[i] == Catch::Approx(x_explicit[i]).margin(1e-12));
    }
  }

  // ================================================================
  // Full Cholesky solve: A = L*L^T, solve Ax = b
  // ================================================================

  TEST_CASE("full cholesky solve - tridiag", "[triangular_solve]") {
    auto A = make_matrix(
        Shape{4, 4},
        {Entry<double>{Index{0, 0}, 4.0}, Entry<double>{Index{0, 1}, -1.0},
         Entry<double>{Index{1, 0}, -1.0}, Entry<double>{Index{1, 1}, 4.0},
         Entry<double>{Index{1, 2}, -1.0}, Entry<double>{Index{2, 1}, -1.0},
         Entry<double>{Index{2, 2}, 4.0}, Entry<double>{Index{2, 3}, -1.0},
         Entry<double>{Index{3, 2}, -1.0}, Entry<double>{Index{3, 3}, 4.0}});

    auto L = cholesky(A);

    // Known x, compute b = A*x, then solve and recover x
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = multiply(A, std::span<double const>{x_true});

    // Solve: L*y = b, then L^T*x = y
    auto y = forward_solve(L, std::span<double const>{b});
    auto x = forward_solve_transpose(L, std::span<double const>{y});

    REQUIRE(x.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-10));
    }
  }

  TEST_CASE("full cholesky solve - grid", "[triangular_solve]") {
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
        entries.push_back(Entry<double>{Index{node, node},
                                        static_cast<double>(degree) + 5.0});
      }
    }

    auto A = make_matrix(Shape{n, n}, entries);
    auto L = cholesky(A);

    // Known x = [1, 2, ..., 16]
    std::vector<double> x_true;
    for (size_type i = 0; i < n; ++i) {
      x_true.push_back(static_cast<double>(i + 1));
    }
    auto b = multiply(A, std::span<double const>{x_true});

    auto y = forward_solve(L, std::span<double const>{b});
    auto x = forward_solve_transpose(L, std::span<double const>{y});

    REQUIRE(x.size() == static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      CHECK(x[i] == Catch::Approx(x_true[i]).margin(1e-9));
    }
  }

  // ================================================================
  // Error handling
  // ================================================================

  TEST_CASE("forward solve - rectangular rejected", "[triangular_solve]") {
    auto L = make_matrix(Shape{3, 4}, {Entry<double>{Index{0, 0}, 1.0},
                                       Entry<double>{Index{1, 1}, 1.0},
                                       Entry<double>{Index{2, 2}, 1.0}});

    std::vector<double> b = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS(forward_solve(L, std::span<double const>{b}),
                    std::invalid_argument);
  }

} // end of namespace sparkit::testing
