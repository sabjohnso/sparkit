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
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/multifrontal.hpp>
#include <sparkit/data/sparse_blas.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::multifrontal_cholesky;
  using sparkit::data::detail::multifrontal_solve;
  using sparkit::data::detail::multiply;

  using size_type = sparkit::config::size_type;

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

  // Helper: check relative residual ||Ax-b||/||b|| < tol
  static void
  check_residual(
    Compressed_row_matrix<double> const& A,
    std::vector<double> const& x,
    std::vector<double> const& b,
    double tol = 1e-12) {
    auto Ax = multiply(A, std::span<double const>{x});
    double norm_r = 0, norm_b = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      auto r = Ax[i] - b[i];
      norm_r += r * r;
      norm_b += b[i] * b[i];
    }
    CHECK(std::sqrt(norm_r) / std::sqrt(norm_b) < tol);
  }

  TEST_CASE("multifrontal convenience - with AMD", "[multifrontal]") {
    auto A = grid_laplacian(4);
    auto n = static_cast<std::size_t>(A.shape().row());

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }

    auto factor = multifrontal_cholesky(A, true);
    auto x = multifrontal_solve(factor, std::span<double const>{b});

    check_residual(A, x, b);
  }

  TEST_CASE("multifrontal convenience - without AMD", "[multifrontal]") {
    auto A = grid_laplacian(4);
    auto n = static_cast<std::size_t>(A.shape().row());

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }

    auto factor = multifrontal_cholesky(A, false);
    auto x = multifrontal_solve(factor, std::span<double const>{b});

    check_residual(A, x, b);
  }

  TEST_CASE("multifrontal convenience - end-to-end small", "[multifrontal]") {
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

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto factor = multifrontal_cholesky(A);
    auto x = multifrontal_solve(factor, std::span<double const>{b});
    check_residual(A, x, b);
  }

  TEST_CASE(
    "multifrontal convenience - rectangular rejected", "[multifrontal]") {
    Compressed_row_sparsity sp{
      Shape{3, 4}, {Index{0, 0}, Index{1, 1}, Index{2, 2}}};
    Compressed_row_matrix<double> A{
      sp, [](size_type, size_type) { return 1.0; }};

    CHECK_THROWS_AS(multifrontal_cholesky(A), std::invalid_argument);
  }

  TEST_CASE(
    "multifrontal convenience - symbolic reuse with two value sets",
    "[multifrontal]") {
    // Factor the same pattern with two different value sets
    auto A1 = grid_laplacian(3);
    auto n = static_cast<std::size_t>(A1.shape().row());

    // Second matrix: same pattern, different diagonal
    Compressed_row_matrix<double> A2{
      A1.sparsity(),
      [](size_type row, size_type col) { return row == col ? 8.0 : -1.0; }};

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }

    auto factor1 = multifrontal_cholesky(A1);
    auto x1 = multifrontal_solve(factor1, std::span<double const>{b});
    check_residual(A1, x1, b);

    auto factor2 = multifrontal_cholesky(A2);
    auto x2 = multifrontal_solve(factor2, std::span<double const>{b});
    check_residual(A2, x2, b);
  }

} // end of namespace sparkit::testing
