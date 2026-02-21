//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/permutation.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::is_valid_permutation;
  using sparkit::data::detail::inverse_permutation;
  using sparkit::data::detail::rperm;
  using sparkit::data::detail::cperm;
  using sparkit::data::detail::dperm;

  using size_type = sparkit::config::size_type;

  // ================================================================
  // is_valid_permutation
  // ================================================================

  TEST_CASE("permutation - is_valid_permutation valid", "[permutation]")
  {
    std::vector<size_type> p{2, 0, 1};
    CHECK(is_valid_permutation(p));
  }

  TEST_CASE("permutation - is_valid_permutation empty", "[permutation]")
  {
    std::vector<size_type> p{};
    CHECK(is_valid_permutation(p));
  }

  TEST_CASE("permutation - is_valid_permutation duplicate", "[permutation]")
  {
    std::vector<size_type> p{0, 0, 1};
    CHECK_FALSE(is_valid_permutation(p));
  }

  TEST_CASE("permutation - is_valid_permutation out of range", "[permutation]")
  {
    std::vector<size_type> p{0, 3, 1};
    CHECK_FALSE(is_valid_permutation(p));
  }

  // ================================================================
  // inverse_permutation
  // ================================================================

  TEST_CASE("permutation - inverse_permutation identity", "[permutation]")
  {
    std::vector<size_type> p{0, 1, 2};
    auto inv = inverse_permutation(p);

    REQUIRE(std::ssize(inv) == 3);
    CHECK(inv[0] == 0);
    CHECK(inv[1] == 1);
    CHECK(inv[2] == 2);
  }

  TEST_CASE("permutation - inverse_permutation known", "[permutation]")
  {
    // perm[old] = new: 0->2, 1->0, 2->1
    // inv[new] = old:  0->1, 1->2, 2->0
    std::vector<size_type> p{2, 0, 1};
    auto inv = inverse_permutation(p);

    REQUIRE(std::ssize(inv) == 3);
    CHECK(inv[0] == 1);
    CHECK(inv[1] == 2);
    CHECK(inv[2] == 0);
  }

  TEST_CASE("permutation - inverse_permutation round trip", "[permutation]")
  {
    std::vector<size_type> p{2, 0, 1};
    auto inv = inverse_permutation(p);
    auto inv_inv = inverse_permutation(inv);

    REQUIRE(std::ssize(inv_inv) == 3);
    CHECK(inv_inv[0] == p[0]);
    CHECK(inv_inv[1] == p[1]);
    CHECK(inv_inv[2] == p[2]);
  }

  // ================================================================
  // rperm (row permutation)
  // ================================================================

  TEST_CASE("permutation - rperm identity", "[permutation]")
  {
    // A = [[1,2],[0,3],[4,0]]
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}
    }};

    std::vector<size_type> p{0, 1, 2};
    auto B = rperm(A, p);

    CHECK(B.shape() == A.shape());
    CHECK(B.size() == A.size());
    for (size_type i = 0; i < 3; ++i) {
      for (size_type j = 0; j < 3; ++j) {
        CHECK(B(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("permutation - rperm known matrix", "[permutation]")
  {
    // A = [[1,2,0],[0,3,4],[5,0,6]]
    // perm = {2,0,1}: row 0->2, row 1->0, row 2->1
    // Result: row 0 of B = old row inv[0] = old row 1 = [0,3,4]
    //         row 1 of B = old row inv[1] = old row 2 = [5,0,6]
    //         row 2 of B = old row inv[2] = old row 0 = [1,2,0]
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<size_type> p{2, 0, 1};
    auto B = rperm(A, p);

    CHECK(B.shape() == Shape(3, 3));
    CHECK(B.size() == 6);

    // Row 0 of B = old row 1: (0,1)=3, (0,2)=4
    CHECK(B(0, 0) == Catch::Approx(0.0));
    CHECK(B(0, 1) == Catch::Approx(3.0));
    CHECK(B(0, 2) == Catch::Approx(4.0));

    // Row 1 of B = old row 2: (1,0)=5, (1,2)=6
    CHECK(B(1, 0) == Catch::Approx(5.0));
    CHECK(B(1, 1) == Catch::Approx(0.0));
    CHECK(B(1, 2) == Catch::Approx(6.0));

    // Row 2 of B = old row 0: (2,0)=1, (2,1)=2
    CHECK(B(2, 0) == Catch::Approx(1.0));
    CHECK(B(2, 1) == Catch::Approx(2.0));
    CHECK(B(2, 2) == Catch::Approx(0.0));
  }

  TEST_CASE("permutation - rperm sparsity", "[permutation]")
  {
    // Test sparsity-only rperm
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 1}, Index{1, 2},
      Index{2, 0}, Index{2, 2}
    }};

    std::vector<size_type> p{2, 0, 1};
    auto result = rperm(sp, p);

    CHECK(result.shape() == Shape(3, 3));
    CHECK(result.size() == 6);

    // Row 0 of result = old row 1: cols {1, 2}
    auto rp = result.row_ptr();
    auto ci = result.col_ind();

    CHECK(rp[1] - rp[0] == 2);
    CHECK(ci[rp[0]] == 1);
    CHECK(ci[rp[0] + 1] == 2);

    // Row 1 of result = old row 2: cols {0, 2}
    CHECK(rp[2] - rp[1] == 2);
    CHECK(ci[rp[1]] == 0);
    CHECK(ci[rp[1] + 1] == 2);
  }

  // ================================================================
  // cperm (column permutation)
  // ================================================================

  TEST_CASE("permutation - cperm identity", "[permutation]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}
    }};

    std::vector<size_type> p{0, 1, 2};
    auto B = cperm(A, p);

    CHECK(B.shape() == A.shape());
    CHECK(B.size() == A.size());
    for (size_type i = 0; i < 3; ++i) {
      for (size_type j = 0; j < 3; ++j) {
        CHECK(B(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("permutation - cperm known matrix", "[permutation]")
  {
    // A = [[1,2,0],[0,3,4],[5,0,6]]
    // perm = {2,0,1}: col 0->2, col 1->0, col 2->1
    // Result: B(i, perm[j]) = A(i, j)
    //   B(0,2)=1, B(0,0)=2, B(0,1)=0
    //   B(1,2)=0, B(1,0)=3, B(1,1)=4
    //   B(2,2)=5, B(2,0)=0, B(2,1)=6  wait... no
    // cperm replaces col j with perm[j]:
    //   A(0,0)=1 -> B(0, perm[0])=B(0,2)=1
    //   A(0,1)=2 -> B(0, perm[1])=B(0,0)=2
    //   A(1,1)=3 -> B(1, perm[1])=B(1,0)=3
    //   A(1,2)=4 -> B(1, perm[2])=B(1,1)=4
    //   A(2,0)=5 -> B(2, perm[0])=B(2,2)=5
    //   A(2,2)=6 -> B(2, perm[2])=B(2,1)=6
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<size_type> p{2, 0, 1};
    auto B = cperm(A, p);

    CHECK(B.shape() == Shape(3, 3));
    CHECK(B.size() == 6);

    CHECK(B(0, 0) == Catch::Approx(2.0));
    CHECK(B(0, 1) == Catch::Approx(0.0));
    CHECK(B(0, 2) == Catch::Approx(1.0));
    CHECK(B(1, 0) == Catch::Approx(3.0));
    CHECK(B(1, 1) == Catch::Approx(4.0));
    CHECK(B(1, 2) == Catch::Approx(0.0));
    CHECK(B(2, 0) == Catch::Approx(0.0));
    CHECK(B(2, 1) == Catch::Approx(6.0));
    CHECK(B(2, 2) == Catch::Approx(5.0));
  }

  TEST_CASE("permutation - cperm sparsity", "[permutation]")
  {
    Compressed_row_sparsity sp{Shape{3, 3}, {
      Index{0, 0}, Index{0, 1},
      Index{1, 1}, Index{1, 2},
      Index{2, 0}, Index{2, 2}
    }};

    // perm = {2,0,1}: col 0->2, col 1->0, col 2->1
    std::vector<size_type> p{2, 0, 1};
    auto result = cperm(sp, p);

    CHECK(result.shape() == Shape(3, 3));
    CHECK(result.size() == 6);

    auto rp = result.row_ptr();
    auto ci = result.col_ind();

    // Row 0: old cols {0,1} -> new cols {perm[0]=2, perm[1]=0} -> sorted {0,2}
    CHECK(rp[1] - rp[0] == 2);
    CHECK(ci[rp[0]] == 0);
    CHECK(ci[rp[0] + 1] == 2);

    // Row 1: old cols {1,2} -> new cols {perm[1]=0, perm[2]=1} -> sorted {0,1}
    CHECK(rp[2] - rp[1] == 2);
    CHECK(ci[rp[1]] == 0);
    CHECK(ci[rp[1] + 1] == 1);
  }

  // ================================================================
  // dperm (symmetric/double permutation)
  // ================================================================

  TEST_CASE("permutation - dperm identity", "[permutation]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<size_type> p{0, 1, 2};
    auto B = dperm(A, p);

    CHECK(B.shape() == A.shape());
    CHECK(B.size() == A.size());
    for (size_type i = 0; i < 3; ++i) {
      for (size_type j = 0; j < 3; ++j) {
        CHECK(B(i, j) == Catch::Approx(A(i, j)));
      }
    }
  }

  TEST_CASE("permutation - dperm equals rperm of cperm", "[permutation]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, 6.0}
    }};

    std::vector<size_type> p{2, 0, 1};

    auto B = dperm(A, p);
    auto C = rperm(cperm(A, p), p);

    CHECK(B.shape() == C.shape());
    CHECK(B.size() == C.size());
    for (size_type i = 0; i < 3; ++i) {
      for (size_type j = 0; j < 3; ++j) {
        CHECK(B(i, j) == Catch::Approx(C(i, j)));
      }
    }
  }

} // end of namespace sparkit::testing
