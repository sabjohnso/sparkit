//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <cmath>
#include <limits>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/info.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::row_norms_1;
  using sparkit::data::detail::row_norms_inf;
  using sparkit::data::detail::column_norms_1;
  using sparkit::data::detail::column_norms_inf;
  using sparkit::data::detail::frobenius_norm;
  using sparkit::data::detail::norm_1;
  using sparkit::data::detail::norm_inf;
  using sparkit::data::detail::bandwidth;
  using sparkit::data::detail::diagonal_occupancy;
  using sparkit::data::detail::diagonal_positions;
  using sparkit::data::detail::detect_block_size;

  // ================================================================
  // row_norms_1
  // ================================================================

  TEST_CASE("info - row_norms_1 known values", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // row sums: |1|+|-2| = 3, |3|+|4| = 7, |5|+|-6| = 11
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    auto norms = row_norms_1(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(3.0));
    CHECK(norms[1] == Catch::Approx(7.0));
    CHECK(norms[2] == Catch::Approx(11.0));
  }

  TEST_CASE("info - row_norms_1 empty rows", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{2, 2}, 2.0}
    }};

    auto norms = row_norms_1(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(1.0));
    CHECK(norms[1] == Catch::Approx(0.0));
    CHECK(norms[2] == Catch::Approx(2.0));
  }

  TEST_CASE("info - row_norms_1 single entry rows", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 1}, -7.0},
      {Index{1, 0}, 3.0}
    }};

    auto norms = row_norms_1(A);

    REQUIRE(std::ssize(norms) == 2);
    CHECK(norms[0] == Catch::Approx(7.0));
    CHECK(norms[1] == Catch::Approx(3.0));
  }

  TEST_CASE("info - row_norms_1 compensated summation", "[info]")
  {
    // Classic summation precision test: 1.0 + many small values.
    // Row 0: 1e16 + 1.0 + 1.0 + ... (many 1.0s)
    // Naive summation loses the small terms when added to 1e16.
    // With 10 entries of 1.0, exact row sum = 1e16 + 10.
    Compressed_row_matrix<double> A{Shape{2, 12}, {
      {Index{0, 0}, 1e16},
      {Index{0, 1}, 1.0}, {Index{0, 2}, 1.0}, {Index{0, 3}, 1.0},
      {Index{0, 4}, 1.0}, {Index{0, 5}, 1.0}, {Index{0, 6}, 1.0},
      {Index{0, 7}, 1.0}, {Index{0, 8}, 1.0}, {Index{0, 9}, 1.0},
      {Index{0, 10}, 1.0},
      {Index{1, 11}, 1.0}
    }};

    auto norms = row_norms_1(A);

    // Exact answer is 1e16 + 10.  Naive sum gives 1e16 (loses the 10).
    CHECK(norms[0] == Catch::Approx(1e16 + 10.0).epsilon(0.0));
  }

  // ================================================================
  // row_norms_inf
  // ================================================================

  TEST_CASE("info - row_norms_inf known values", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // row maxes: max(|1|,|-2|)=2, max(|3|,|4|)=4, max(|5|,|-6|)=6
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    auto norms = row_norms_inf(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(2.0));
    CHECK(norms[1] == Catch::Approx(4.0));
    CHECK(norms[2] == Catch::Approx(6.0));
  }

  TEST_CASE("info - row_norms_inf empty rows", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 5.0},
      {Index{2, 2}, 3.0}
    }};

    auto norms = row_norms_inf(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(5.0));
    CHECK(norms[1] == Catch::Approx(0.0));
    CHECK(norms[2] == Catch::Approx(3.0));
  }

  // ================================================================
  // column_norms_1
  // ================================================================

  TEST_CASE("info - column_norms_1 known values", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // col sums: |1|+|5|=6, |-2|+|3|=5, |4|+|-6|=10
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    auto norms = column_norms_1(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(6.0));
    CHECK(norms[1] == Catch::Approx(5.0));
    CHECK(norms[2] == Catch::Approx(10.0));
  }

  TEST_CASE("info - column_norms_1 compensated summation", "[info]")
  {
    // Column 0 receives: 1e16 then 1.0, 1.0, ..., 1.0  (10 ones).
    // Exact column sum = 1e16 + 10.
    Compressed_row_matrix<double> A{Shape{12, 2}, {
      {Index{0, 0}, 1e16},
      {Index{1, 0}, 1.0}, {Index{2, 0}, 1.0}, {Index{3, 0}, 1.0},
      {Index{4, 0}, 1.0}, {Index{5, 0}, 1.0}, {Index{6, 0}, 1.0},
      {Index{7, 0}, 1.0}, {Index{8, 0}, 1.0}, {Index{9, 0}, 1.0},
      {Index{10, 0}, 1.0},
      {Index{11, 1}, 1.0}
    }};

    auto norms = column_norms_1(A);

    CHECK(norms[0] == Catch::Approx(1e16 + 10.0).epsilon(0.0));
  }

  // ================================================================
  // column_norms_inf
  // ================================================================

  TEST_CASE("info - column_norms_inf known values", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // col maxes: max(|1|,|5|)=5, max(|-2|,|3|)=3, max(|4|,|-6|)=6
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    auto norms = column_norms_inf(A);

    REQUIRE(std::ssize(norms) == 3);
    CHECK(norms[0] == Catch::Approx(5.0));
    CHECK(norms[1] == Catch::Approx(3.0));
    CHECK(norms[2] == Catch::Approx(6.0));
  }

  // ================================================================
  // frobenius_norm
  // ================================================================

  TEST_CASE("info - frobenius_norm known result", "[info]")
  {
    // A = [[1,2],[3,4]]
    // ||A||_F = sqrt(1+4+9+16) = sqrt(30)
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}
    }};

    CHECK(frobenius_norm(A) == Catch::Approx(std::sqrt(30.0)));
  }

  TEST_CASE("info - frobenius_norm single entry", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{1, 2}, 5.0}
    }};

    CHECK(frobenius_norm(A) == Catch::Approx(5.0));
  }

  TEST_CASE("info - frobenius_norm identity-like", "[info]")
  {
    // 3x3 identity: ||I||_F = sqrt(3)
    Compressed_row_matrix<double> I{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 1.0}, {Index{2, 2}, 1.0}
    }};

    CHECK(frobenius_norm(I) == Catch::Approx(std::sqrt(3.0)));
  }

  TEST_CASE("info - frobenius_norm near overflow", "[info]")
  {
    // Two entries near sqrt(DBL_MAX) ≈ 1.34e154.
    // Naive v*v overflows, but the norm itself is representable.
    auto big = 1e200;
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, big}, {Index{1, 1}, big}
    }};

    // ||A||_F = sqrt(2) * 1e200
    auto expected = std::sqrt(2.0) * big;
    CHECK(frobenius_norm(A) == Catch::Approx(expected).epsilon(1e-14));
  }

  TEST_CASE("info - frobenius_norm near underflow", "[info]")
  {
    // Two entries near sqrt(DBL_MIN) ≈ 1.49e-162.
    // Naive v*v underflows to zero, but the norm is nonzero.
    auto tiny = 1e-200;
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 0}, tiny}, {Index{1, 1}, tiny}
    }};

    // ||A||_F = sqrt(2) * 1e-200
    auto expected = std::sqrt(2.0) * tiny;
    CHECK(frobenius_norm(A) == Catch::Approx(expected).epsilon(1e-14));
  }

  TEST_CASE("info - frobenius_norm compensated summation", "[info]")
  {
    // 1e16 and many 1.0 entries.  Exact sum of squares =
    // 1e32 + 10.  sqrt(1e32 + 10) ≈ 1e16 + 5e-17.
    // This primarily tests that the summation doesn't lose the small terms.
    Compressed_row_matrix<double> A{Shape{2, 12}, {
      {Index{0, 0}, 1e8},
      {Index{0, 1}, 1.0}, {Index{0, 2}, 1.0}, {Index{0, 3}, 1.0},
      {Index{0, 4}, 1.0}, {Index{0, 5}, 1.0}, {Index{0, 6}, 1.0},
      {Index{0, 7}, 1.0}, {Index{0, 8}, 1.0}, {Index{0, 9}, 1.0},
      {Index{0, 10}, 1.0},
      {Index{1, 11}, 1.0}
    }};

    // sum of squares = 1e16 + 11
    auto expected = std::sqrt(1e16 + 11.0);
    CHECK(frobenius_norm(A) == Catch::Approx(expected).epsilon(1e-14));
  }

  // ================================================================
  // norm_1
  // ================================================================

  TEST_CASE("info - norm_1 known result", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // col sums: 6, 5, 10 → norm_1 = 10
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    CHECK(norm_1(A) == Catch::Approx(10.0));
  }

  TEST_CASE("info - norm_1 equals max column_norms_1", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, -3.0},
      {Index{1, 0}, 2.0}, {Index{1, 1}, 4.0}
    }};

    auto cn = column_norms_1(A);
    auto expected = *std::max_element(cn.begin(), cn.end());

    CHECK(norm_1(A) == Catch::Approx(expected));
  }

  // ================================================================
  // norm_inf
  // ================================================================

  TEST_CASE("info - norm_inf known result", "[info]")
  {
    // A = [[1,-2,0],[0,3,4],[5,0,-6]]
    // row sums: 3, 7, 11 → norm_inf = 11
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, -2.0},
      {Index{1, 1}, 3.0}, {Index{1, 2}, 4.0},
      {Index{2, 0}, 5.0}, {Index{2, 2}, -6.0}
    }};

    CHECK(norm_inf(A) == Catch::Approx(11.0));
  }

  TEST_CASE("info - norm_inf equals max row_norms_1", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{2, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, -3.0},
      {Index{1, 0}, 2.0}, {Index{1, 1}, 4.0}
    }};

    auto rn = row_norms_1(A);
    auto expected = *std::max_element(rn.begin(), rn.end());

    CHECK(norm_inf(A) == Catch::Approx(expected));
  }

  // ================================================================
  // bandwidth
  // ================================================================

  TEST_CASE("info - bandwidth diagonal matrix", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{1, 1}, 2.0}, {Index{2, 2}, 3.0}
    }};

    auto [lower, upper] = bandwidth(A);

    CHECK(lower == 0);
    CHECK(upper == 0);
  }

  TEST_CASE("info - bandwidth tridiagonal", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0}, {Index{1, 2}, 5.0},
      {Index{2, 1}, 6.0}, {Index{2, 2}, 7.0}
    }};

    auto [lower, upper] = bandwidth(A);

    CHECK(lower == 1);
    CHECK(upper == 1);
  }

  TEST_CASE("info - bandwidth full band", "[info]")
  {
    // 3x3 full matrix: lower=2, upper=2
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0}, {Index{0, 2}, 3.0},
      {Index{1, 0}, 4.0}, {Index{1, 1}, 5.0}, {Index{1, 2}, 6.0},
      {Index{2, 0}, 7.0}, {Index{2, 1}, 8.0}, {Index{2, 2}, 9.0}
    }};

    auto [lower, upper] = bandwidth(A);

    CHECK(lower == 2);
    CHECK(upper == 2);
  }

  TEST_CASE("info - bandwidth rectangular", "[info]")
  {
    // 2x4 matrix: entries at (0,0), (0,3), (1,1)
    // lower: max(i-j) where i>=j = max(0,0) = 0
    // upper: max(j-i) where j>=i = max(0,3,0) = 3
    Compressed_row_matrix<double> A{Shape{2, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 3}, 2.0},
      {Index{1, 1}, 3.0}
    }};

    auto [lower, upper] = bandwidth(A);

    CHECK(lower == 0);
    CHECK(upper == 3);
  }

  TEST_CASE("info - bandwidth empty matrix", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {}};

    auto [lower, upper] = bandwidth(A);

    CHECK(lower == 0);
    CHECK(upper == 0);
  }

  // ================================================================
  // diagonal_occupancy
  // ================================================================

  TEST_CASE("info - diagonal_occupancy full diagonal", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}, {Index{2, 2}, 5.0}
    }};

    CHECK(diagonal_occupancy(A) == 3);
  }

  TEST_CASE("info - diagonal_occupancy partial diagonal", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{2, 2}, 2.0}
    }};

    CHECK(diagonal_occupancy(A) == 2);
  }

  TEST_CASE("info - diagonal_occupancy no diagonal", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{2, 2}, {
      {Index{0, 1}, 1.0},
      {Index{1, 0}, 2.0}
    }};

    CHECK(diagonal_occupancy(A) == 0);
  }

  // ================================================================
  // diagonal_positions
  // ================================================================

  TEST_CASE("info - diagonal_positions matches occupancy count", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0},
      {Index{2, 2}, 2.0}
    }};

    auto pos = diagonal_positions(A);
    config::size_type count = 0;
    for (auto b : pos) { if (b) ++count; }

    CHECK(count == diagonal_occupancy(A));
  }

  TEST_CASE("info - diagonal_positions specific positions", "[info]")
  {
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 2}, 3.0},
      {Index{2, 2}, 4.0}
    }};

    auto pos = diagonal_positions(A);

    REQUIRE(std::ssize(pos) == 3);
    CHECK(pos[0] == true);
    CHECK(pos[1] == false);
    CHECK(pos[2] == true);
  }

  // ================================================================
  // detect_block_size
  // ================================================================

  TEST_CASE("info - detect_block_size 2x2 blocked", "[info]")
  {
    // 4x4 matrix with perfect 2x2 block structure
    Compressed_row_matrix<double> A{Shape{4, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0},
      {Index{2, 2}, 5.0}, {Index{2, 3}, 6.0},
      {Index{3, 2}, 7.0}, {Index{3, 3}, 8.0}
    }};

    auto [br, bc] = detect_block_size(A);

    CHECK(br == 2);
    CHECK(bc == 2);
  }

  TEST_CASE("info - detect_block_size non-blocked", "[info]")
  {
    // Irregular pattern — no block structure
    Compressed_row_matrix<double> A{Shape{3, 3}, {
      {Index{0, 0}, 1.0}, {Index{0, 2}, 2.0},
      {Index{1, 1}, 3.0},
      {Index{2, 0}, 4.0}, {Index{2, 2}, 5.0}
    }};

    auto [br, bc] = detect_block_size(A);

    CHECK(br == 1);
    CHECK(bc == 1);
  }

  TEST_CASE("info - detect_block_size mixed", "[info]")
  {
    // 4x4 with some 2x2 blocks but not all — should detect (1,1) or
    // a valid block size
    Compressed_row_matrix<double> A{Shape{4, 4}, {
      {Index{0, 0}, 1.0}, {Index{0, 1}, 2.0},
      {Index{1, 0}, 3.0}, {Index{1, 1}, 4.0},
      {Index{2, 2}, 5.0},
      {Index{3, 3}, 6.0}
    }};

    auto [br, bc] = detect_block_size(A);

    // Block (2,2) at (1,1) is missing entry (2,3), (3,2), (3,3 missing one)
    // so no 2x2 block structure
    CHECK(br == 1);
    CHECK(bc == 1);
  }

} // end of namespace sparkit::testing
