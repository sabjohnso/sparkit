//
// ... Test header files
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Entry.hpp>
#include <sparkit/data/conversions.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Coordinate_matrix;
  using sparkit::data::detail::Coordinate_sparsity;
  using sparkit::data::detail::Compressed_row_matrix;
  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;
  using sparkit::data::detail::Entry;

  // -- Coordinate_matrix core --

  TEST_CASE("coordinate_matrix - construction_empty", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{3, 4}};
    CHECK(mat.shape() == Shape(3, 4));
    CHECK(mat.size() == 0);
  }

  TEST_CASE("coordinate_matrix - add_and_entries", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{6, 7}};
    mat.add(Index{2, 3}, 3.0);
    mat.add(Index{4, 5}, 7.0);

    CHECK(mat.size() == 2);

    auto ents = mat.entries();
    REQUIRE(std::ssize(ents) == 2);

    // Sort for order-independent comparison
    auto by_row_col = [](auto const& a, auto const& b) {
      return a.first.row() < b.first.row()
        || (a.first.row() == b.first.row()
            && a.first.column() < b.first.column());
    };
    std::sort(begin(ents), end(ents), by_row_col);

    CHECK(ents[0].first == Index(2, 3));
    CHECK(ents[0].second == Catch::Approx(3.0));
    CHECK(ents[1].first == Index(4, 5));
    CHECK(ents[1].second == Catch::Approx(7.0));
  }

  TEST_CASE("coordinate_matrix - add_replaces_duplicate", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{3, 3}, 5.0);
    mat.add(Index{3, 3}, 9.0);

    CHECK(mat.size() == 1);

    auto ents = mat.entries();
    REQUIRE(std::ssize(ents) == 1);
    CHECK(ents[0].first == Index(3, 3));
    CHECK(ents[0].second == Catch::Approx(9.0));
  }

  TEST_CASE("coordinate_matrix - remove_entry", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{2, 3}, 2.0);
    mat.add(Index{3, 4}, 4.0);

    mat.remove(Index{2, 3});
    CHECK(mat.size() == 1);

    auto ents = mat.entries();
    REQUIRE(std::ssize(ents) == 1);
    CHECK(ents[0].first == Index(3, 4));
  }

  TEST_CASE("coordinate_matrix - remove_absent_is_noop", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{2, 2}, 1.0);

    mat.remove(Index{4, 4});
    CHECK(mat.size() == 1);
  }

  TEST_CASE("coordinate_matrix - construction_from_entry_initializer_list", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 2.0},
      {Index{3, 4}, 4.0},
      {Index{4, 5}, 6.0}
    }};

    CHECK(mat.size() == 3);
    CHECK(mat.shape() == Shape(6, 7));
    CHECK(mat(2, 3) == Catch::Approx(2.0));
    CHECK(mat(3, 4) == Catch::Approx(4.0));
    CHECK(mat(4, 5) == Catch::Approx(6.0));
  }

  TEST_CASE("coordinate_matrix - construction_from_entry_iterator_range", "[coordinate_matrix]")
  {
    std::vector<Entry<double>> data{
      {Index{3, 2}, 1.5},
      {Index{4, 3}, 2.5}
    };

    Coordinate_matrix<double> mat{Shape{6, 6}, begin(data), end(data)};

    CHECK(mat.size() == 2);
    CHECK(mat.shape() == Shape(6, 6));
    CHECK(mat(3, 2) == Catch::Approx(1.5));
    CHECK(mat(4, 3) == Catch::Approx(2.5));
  }

  TEST_CASE("coordinate_matrix - sparsity_extraction", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{5, 6}, {
      {Index{2, 3}, 1.0},
      {Index{3, 4}, 2.0},
      {Index{4, 5}, 3.0}
    }};

    Coordinate_sparsity sp = mat.sparsity();
    CHECK(sp.shape() == Shape(5, 6));
    CHECK(sp.size() == 3);

    auto idx = sp.indices();
    auto by_row_col = [](Index const& a, Index const& b) {
      return a.row() < b.row()
        || (a.row() == b.row() && a.column() < b.column());
    };
    std::sort(begin(idx), end(idx), by_row_col);

    CHECK(idx[0] == Index(2, 3));
    CHECK(idx[1] == Index(3, 4));
    CHECK(idx[2] == Index(4, 5));
  }

  // -- Coordinate_matrix sparsity + function construction --

  TEST_CASE("coordinate_matrix - construction_from_sparsity_and_function", "[coordinate_matrix]")
  {
    Coordinate_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}, Index{4, 5}}};

    Coordinate_matrix<double> mat{sp, [](auto row, auto col) {
      return static_cast<double>(row * 10 + col);
    }};

    CHECK(mat.shape() == Shape(5, 6));
    CHECK(mat.size() == 3);
    CHECK(mat(2, 3) == Catch::Approx(23.0));
    CHECK(mat(3, 4) == Catch::Approx(34.0));
    CHECK(mat(4, 5) == Catch::Approx(45.0));
    CHECK(mat(2, 2) == Catch::Approx(0.0));
  }

  // -- Coordinate_matrix element access --

  TEST_CASE("coordinate_matrix - element_access_existing", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 3.0},
      {Index{4, 5}, 7.0}
    }};

    CHECK(mat(2, 3) == Catch::Approx(3.0));
    CHECK(mat(4, 5) == Catch::Approx(7.0));
  }

  TEST_CASE("coordinate_matrix - element_access_absent_returns_zero", "[coordinate_matrix]")
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 3.0}
    }};

    CHECK(mat(3, 4) == Catch::Approx(0.0));
    CHECK(mat(5, 6) == Catch::Approx(0.0));
  }

  // -- Compressed_row_matrix core --

  TEST_CASE("compressed_row_matrix - construction_and_accessors", "[compressed_row_matrix]")
  {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat.shape() == Shape(5, 6));
    CHECK(mat.size() == 3);

    auto vals = mat.values();
    REQUIRE(std::ssize(vals) == 3);
    CHECK(vals[0] == Catch::Approx(1.0));
    CHECK(vals[1] == Catch::Approx(2.0));
    CHECK(vals[2] == Catch::Approx(3.0));
  }

  TEST_CASE("compressed_row_matrix - structural_accessors_delegate", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    auto rp = mat.row_ptr();
    auto ci = mat.col_ind();

    // These must match the sparsity pattern
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[3] == 2);
    CHECK(rp[4] == 3);

    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);
  }

  TEST_CASE("compressed_row_matrix - sparsity_accessor", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0}};

    auto const& sp_ref = mat.sparsity();
    CHECK(sp_ref.shape() == Shape(5, 6));
    CHECK(sp_ref.size() == 2);
  }

  TEST_CASE("compressed_row_matrix - empty_matrix", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{3, 3}, {}};
    Compressed_row_matrix<double> mat{sp, {}};

    CHECK(mat.size() == 0);
    CHECK(mat.shape() == Shape(3, 3));
    CHECK(mat.values().empty());
  }

  TEST_CASE("compressed_row_matrix - copy_construction", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}}};

    Compressed_row_matrix<double> original{sp, {5.0, 7.0}};
    Compressed_row_matrix<double> copy{original};

    CHECK(copy.shape() == original.shape());
    CHECK(copy.size() == original.size());

    auto orig_vals = original.values();
    auto copy_vals = copy.values();
    REQUIRE(std::ssize(copy_vals) == std::ssize(orig_vals));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_vals); ++i) {
      CHECK(copy_vals[i] == Catch::Approx(orig_vals[i]));
    }

    // Verify independent storage
    CHECK(copy_vals.data() != orig_vals.data());
  }

  TEST_CASE("compressed_row_matrix - move_construction", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}}};

    Compressed_row_matrix<double> original{sp, {5.0, 7.0}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_matrix<double> moved{std::move(original)};

    CHECK(moved.shape() == original_shape);
    CHECK(moved.size() == original_size);
  }

  // -- Compressed_row_matrix element access --

  TEST_CASE("compressed_row_matrix - element_access_existing", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat(2, 2) == Catch::Approx(1.0));
    CHECK(mat(2, 4) == Catch::Approx(2.0));
    CHECK(mat(3, 5) == Catch::Approx(3.0));
  }

  TEST_CASE("compressed_row_matrix - element_access_absent_in_populated_row", "[compressed_row_matrix]")
  {
    // Row 2 has entries at columns 2 and 4, but not column 3
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    CHECK(mat(2, 3) == Catch::Approx(0.0));
  }

  TEST_CASE("compressed_row_matrix - element_access_empty_row", "[compressed_row_matrix]")
  {
    // Row 4 has no entries
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0}};

    CHECK(mat(4, 3) == Catch::Approx(0.0));
  }

  // -- Compressed_row_matrix from entries --

  TEST_CASE("compressed_row_matrix - construction_from_entry_initializer_list", "[compressed_row_matrix]")
  {
    Compressed_row_matrix<double> mat{Shape{5, 6}, {
      {Index{2, 2}, 1.0},
      {Index{2, 4}, 2.0},
      {Index{3, 5}, 3.0}
    }};

    CHECK(mat.shape() == Shape(5, 6));
    CHECK(mat.size() == 3);

    auto rp = mat.row_ptr();
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    CHECK(rp[3] == 2);
    CHECK(rp[4] == 3);
    CHECK(rp[5] == 3);

    auto ci = mat.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);

    auto vals = mat.values();
    REQUIRE(std::ssize(vals) == 3);
    CHECK(vals[0] == Catch::Approx(1.0));
    CHECK(vals[1] == Catch::Approx(2.0));
    CHECK(vals[2] == Catch::Approx(3.0));
  }

  TEST_CASE("compressed_row_matrix - construction_from_entries_sorts", "[compressed_row_matrix]")
  {
    // Provide entries out of order — constructor must sort by (row, col)
    Compressed_row_matrix<double> mat{Shape{6, 6}, {
      {Index{4, 5}, 30.0},
      {Index{2, 4}, 20.0},
      {Index{2, 2}, 10.0}
    }};

    CHECK(mat.size() == 3);

    auto ci = mat.col_ind();
    auto vals = mat.values();

    // Sorted order: (2,2)=10, (2,4)=20, (4,5)=30
    REQUIRE(std::ssize(vals) == 3);
    CHECK(ci[0] == 2);
    CHECK(vals[0] == Catch::Approx(10.0));
    CHECK(ci[1] == 4);
    CHECK(vals[1] == Catch::Approx(20.0));
    CHECK(ci[2] == 5);
    CHECK(vals[2] == Catch::Approx(30.0));
  }

  // -- Compressed_row_matrix sparsity + function construction --

  TEST_CASE("compressed_row_matrix - construction_from_sparsity_and_function", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, [](auto row, auto col) {
      return static_cast<double>(row + col);
    }};

    CHECK(mat.shape() == Shape(5, 6));
    CHECK(mat.size() == 3);

    auto rp = mat.row_ptr();
    auto ci = mat.col_ind();
    REQUIRE(std::ssize(rp) == 6);
    REQUIRE(std::ssize(ci) == 3);

    CHECK(mat(2, 2) == Catch::Approx(4.0));
    CHECK(mat(2, 4) == Catch::Approx(6.0));
    CHECK(mat(3, 5) == Catch::Approx(8.0));
  }

  TEST_CASE("compressed_row_matrix - construction_from_sparsity_and_function_values_match_structure", "[compressed_row_matrix]")
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}, Index{4, 5}}};

    Compressed_row_matrix<double> mat{sp, [](auto row, auto col) {
      return static_cast<double>(row * 10 + col);
    }};

    auto vals = mat.values();

    // Each value encodes its (row, col) position
    REQUIRE(std::ssize(vals) == 3);
    CHECK(vals[0] == Catch::Approx(23.0));  // row=2, col=3
    CHECK(vals[1] == Catch::Approx(34.0));  // row=3, col=4
    CHECK(vals[2] == Catch::Approx(45.0));  // row=4, col=5
  }

  // -- COO matrix to CSR matrix conversion --

  TEST_CASE("conversions - coo_matrix_to_csr_matrix_basic", "[conversions]")
  {
    // 5x6 matrix with entries at (2,2)=1.0, (2,4)=2.0, (3,5)=3.0
    Coordinate_matrix<double> coo{Shape{5, 6}, {
      {Index{2, 2}, 1.0},
      {Index{2, 4}, 2.0},
      {Index{3, 5}, 3.0}
    }};

    auto csr = sparkit::data::detail::to_compressed_row(coo);

    CHECK(csr.size() == 3);
    CHECK(csr.shape() == Shape(5, 6));

    auto rp = csr.row_ptr();
    REQUIRE(std::ssize(rp) == 6);
    CHECK(rp[0] == 0);
    CHECK(rp[1] == 0);
    CHECK(rp[2] == 0);
    CHECK(rp[3] == 2);
    CHECK(rp[4] == 3);
    CHECK(rp[5] == 3);

    auto ci = csr.col_ind();
    REQUIRE(std::ssize(ci) == 3);
    CHECK(ci[0] == 2);
    CHECK(ci[1] == 4);
    CHECK(ci[2] == 5);

    auto vals = csr.values();
    REQUIRE(std::ssize(vals) == 3);
    CHECK(vals[0] == Catch::Approx(1.0));
    CHECK(vals[1] == Catch::Approx(2.0));
    CHECK(vals[2] == Catch::Approx(3.0));
  }

  TEST_CASE("conversions - coo_matrix_to_csr_matrix_empty", "[conversions]")
  {
    Coordinate_matrix<double> coo{Shape{3, 3}};
    auto csr = sparkit::data::detail::to_compressed_row(coo);

    CHECK(csr.size() == 0);
    CHECK(csr.shape() == Shape(3, 3));
    CHECK(csr.values().empty());
  }

  TEST_CASE("conversions - coo_matrix_to_csr_matrix_values_follow_sort_order", "[conversions]")
  {
    // Add entries in reverse order — conversion must sort by (row, col)
    Coordinate_matrix<double> coo{Shape{6, 6}};
    coo.add(Index{4, 5}, 30.0);
    coo.add(Index{2, 4}, 20.0);
    coo.add(Index{2, 2}, 10.0);

    auto csr = sparkit::data::detail::to_compressed_row(coo);

    auto ci = csr.col_ind();
    auto vals = csr.values();

    // Sorted order: (2,2)=10, (2,4)=20, (4,5)=30
    REQUIRE(std::ssize(vals) == 3);
    CHECK(ci[0] == 2);
    CHECK(vals[0] == Catch::Approx(10.0));
    CHECK(ci[1] == 4);
    CHECK(vals[1] == Catch::Approx(20.0));
    CHECK(ci[2] == 5);
    CHECK(vals[2] == Catch::Approx(30.0));
  }

  TEST_CASE("conversions - coo_matrix_to_csr_matrix_preserves_shape", "[conversions]")
  {
    Coordinate_matrix<double> coo{Shape{7, 9}, {
      {Index{3, 4}, 1.0}
    }};

    auto csr = sparkit::data::detail::to_compressed_row(coo);
    CHECK(csr.shape() == Shape(7, 9));
  }

} // end of namespace sparkit::testing
