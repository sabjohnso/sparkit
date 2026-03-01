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
#include <sparkit/data/multifrontal_solve.hpp>
#include <sparkit/data/multifrontal_symbolic.hpp>
#include <sparkit/data/sparse_blas.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Entry;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Multifrontal_factor;
  using sparkit::data::detail::Shape;

  using sparkit::data::detail::build_assembly_tree;
  using sparkit::data::detail::elimination_tree;
  using sparkit::data::detail::find_supernodes;
  using sparkit::data::detail::multifrontal_analyze;
  using sparkit::data::detail::multifrontal_backward_solve;
  using sparkit::data::detail::multifrontal_factorize;
  using sparkit::data::detail::multifrontal_forward_solve;
  using sparkit::data::detail::multiply;
  using sparkit::data::detail::symbolic_cholesky;

  using size_type = sparkit::config::size_type;

  // Helper: full factorization pipeline
  static Multifrontal_factor<double>
  full_factor(Compressed_row_matrix<double> const& A) {
    auto sp = A.sparsity();
    auto parent = elimination_tree(sp);
    auto L_pattern = symbolic_cholesky(sp);
    auto part = find_supernodes(L_pattern, parent);
    auto tree = build_assembly_tree(part, parent);
    auto sym = multifrontal_analyze(L_pattern, part, tree);
    auto factors = multifrontal_factorize(A, sym);
    return Multifrontal_factor<double>{
      std::move(sym), std::move(factors), {}, {}};
  }

  // Helper: solve A*x = b and check ||A*x - b|| / ||b|| < tol
  static void
  check_solve(
    Compressed_row_matrix<double> const& A,
    std::vector<double> const& b,
    double tol = 1e-12) {
    auto factor = full_factor(A);
    auto y = multifrontal_forward_solve(factor, std::span<double const>{b});
    auto x = multifrontal_backward_solve(factor, std::span<double const>{y});

    // Compute residual
    auto Ax = multiply(A, std::span<double const>{x});
    double norm_r = 0, norm_b = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      auto r = Ax[i] - b[i];
      norm_r += r * r;
      norm_b += b[i] * b[i];
    }
    norm_r = std::sqrt(norm_r);
    norm_b = std::sqrt(norm_b);

    CHECK(norm_r / norm_b < tol);
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

  TEST_CASE("multifrontal solve - diagonal", "[multifrontal_solve]") {
    Compressed_row_matrix<double> A{
      Shape{4, 4},
      {Entry<double>{{0, 0}, 4.0},
       Entry<double>{{1, 1}, 9.0},
       Entry<double>{{2, 2}, 16.0},
       Entry<double>{{3, 3}, 25.0}}};

    check_solve(A, {1.0, 2.0, 3.0, 4.0});
  }

  TEST_CASE("multifrontal solve - tridiagonal", "[multifrontal_solve]") {
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

    check_solve(A, {1.0, 2.0, 3.0, 4.0});
  }

  TEST_CASE("multifrontal solve - arrow", "[multifrontal_solve]") {
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

    check_solve(A, {1.0, 2.0, 3.0, 4.0, 5.0});
  }

  TEST_CASE("multifrontal solve - grid Laplacian", "[multifrontal_solve]") {
    auto A = grid_laplacian(4);
    auto n = static_cast<std::size_t>(A.shape().row());

    // RHS: b_i = i + 1
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }

    check_solve(A, b);
  }

} // end of namespace sparkit::testing
