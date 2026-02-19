//
// ... Test header files
//
#include <gtest/gtest.h>

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

  TEST(coordinate_matrix, construction_empty)
  {
    Coordinate_matrix<double> mat{Shape{3, 4}};
    EXPECT_EQ(mat.shape(), Shape(3, 4));
    EXPECT_EQ(mat.size(), 0);
  }

  TEST(coordinate_matrix, add_and_entries)
  {
    Coordinate_matrix<double> mat{Shape{6, 7}};
    mat.add(Index{2, 3}, 3.0);
    mat.add(Index{4, 5}, 7.0);

    EXPECT_EQ(mat.size(), 2);

    auto ents = mat.entries();
    ASSERT_EQ(std::ssize(ents), 2);

    // Sort for order-independent comparison
    auto by_row_col = [](auto const& a, auto const& b) {
      return a.first.row() < b.first.row()
        || (a.first.row() == b.first.row()
            && a.first.column() < b.first.column());
    };
    std::sort(begin(ents), end(ents), by_row_col);

    EXPECT_EQ(ents[0].first, Index(2, 3));
    EXPECT_DOUBLE_EQ(ents[0].second, 3.0);
    EXPECT_EQ(ents[1].first, Index(4, 5));
    EXPECT_DOUBLE_EQ(ents[1].second, 7.0);
  }

  TEST(coordinate_matrix, add_replaces_duplicate)
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{3, 3}, 5.0);
    mat.add(Index{3, 3}, 9.0);

    EXPECT_EQ(mat.size(), 1);

    auto ents = mat.entries();
    ASSERT_EQ(std::ssize(ents), 1);
    EXPECT_EQ(ents[0].first, Index(3, 3));
    EXPECT_DOUBLE_EQ(ents[0].second, 9.0);
  }

  TEST(coordinate_matrix, remove_entry)
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{2, 3}, 2.0);
    mat.add(Index{3, 4}, 4.0);

    mat.remove(Index{2, 3});
    EXPECT_EQ(mat.size(), 1);

    auto ents = mat.entries();
    ASSERT_EQ(std::ssize(ents), 1);
    EXPECT_EQ(ents[0].first, Index(3, 4));
  }

  TEST(coordinate_matrix, remove_absent_is_noop)
  {
    Coordinate_matrix<double> mat{Shape{5, 5}};
    mat.add(Index{2, 2}, 1.0);

    mat.remove(Index{4, 4});
    EXPECT_EQ(mat.size(), 1);
  }

  TEST(coordinate_matrix, construction_from_entry_initializer_list)
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 2.0},
      {Index{3, 4}, 4.0},
      {Index{4, 5}, 6.0}
    }};

    EXPECT_EQ(mat.size(), 3);
    EXPECT_EQ(mat.shape(), Shape(6, 7));
    EXPECT_DOUBLE_EQ(mat(2, 3), 2.0);
    EXPECT_DOUBLE_EQ(mat(3, 4), 4.0);
    EXPECT_DOUBLE_EQ(mat(4, 5), 6.0);
  }

  TEST(coordinate_matrix, construction_from_entry_iterator_range)
  {
    std::vector<Entry<double>> data{
      {Index{3, 2}, 1.5},
      {Index{4, 3}, 2.5}
    };

    Coordinate_matrix<double> mat{Shape{6, 6}, begin(data), end(data)};

    EXPECT_EQ(mat.size(), 2);
    EXPECT_EQ(mat.shape(), Shape(6, 6));
    EXPECT_DOUBLE_EQ(mat(3, 2), 1.5);
    EXPECT_DOUBLE_EQ(mat(4, 3), 2.5);
  }

  TEST(coordinate_matrix, sparsity_extraction)
  {
    Coordinate_matrix<double> mat{Shape{5, 6}, {
      {Index{2, 3}, 1.0},
      {Index{3, 4}, 2.0},
      {Index{4, 5}, 3.0}
    }};

    Coordinate_sparsity sp = mat.sparsity();
    EXPECT_EQ(sp.shape(), Shape(5, 6));
    EXPECT_EQ(sp.size(), 3);

    auto idx = sp.indices();
    auto by_row_col = [](Index const& a, Index const& b) {
      return a.row() < b.row()
        || (a.row() == b.row() && a.column() < b.column());
    };
    std::sort(begin(idx), end(idx), by_row_col);

    EXPECT_EQ(idx[0], Index(2, 3));
    EXPECT_EQ(idx[1], Index(3, 4));
    EXPECT_EQ(idx[2], Index(4, 5));
  }

  // -- Coordinate_matrix element access --

  TEST(coordinate_matrix, element_access_existing)
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 3.0},
      {Index{4, 5}, 7.0}
    }};

    EXPECT_DOUBLE_EQ(mat(2, 3), 3.0);
    EXPECT_DOUBLE_EQ(mat(4, 5), 7.0);
  }

  TEST(coordinate_matrix, element_access_absent_returns_zero)
  {
    Coordinate_matrix<double> mat{Shape{6, 7}, {
      {Index{2, 3}, 3.0}
    }};

    EXPECT_DOUBLE_EQ(mat(3, 4), 0.0);
    EXPECT_DOUBLE_EQ(mat(5, 6), 0.0);
  }

  // -- Compressed_row_matrix core --

  TEST(compressed_row_matrix, construction_and_accessors)
  {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    EXPECT_EQ(mat.shape(), Shape(5, 6));
    EXPECT_EQ(mat.size(), 3);

    auto vals = mat.values();
    ASSERT_EQ(std::ssize(vals), 3);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 2.0);
    EXPECT_DOUBLE_EQ(vals[2], 3.0);
  }

  TEST(compressed_row_matrix, structural_accessors_delegate)
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    auto rp = mat.row_ptr();
    auto ci = mat.col_ind();

    // These must match the sparsity pattern
    ASSERT_EQ(std::ssize(rp), 6);
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[3], 2);
    EXPECT_EQ(rp[4], 3);

    ASSERT_EQ(std::ssize(ci), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 5);
  }

  TEST(compressed_row_matrix, sparsity_accessor)
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0}};

    auto const& sp_ref = mat.sparsity();
    EXPECT_EQ(sp_ref.shape(), Shape(5, 6));
    EXPECT_EQ(sp_ref.size(), 2);
  }

  TEST(compressed_row_matrix, empty_matrix)
  {
    Compressed_row_sparsity sp{Shape{3, 3}, {}};
    Compressed_row_matrix<double> mat{sp, {}};

    EXPECT_EQ(mat.size(), 0);
    EXPECT_EQ(mat.shape(), Shape(3, 3));
    EXPECT_TRUE(mat.values().empty());
  }

  TEST(compressed_row_matrix, copy_construction)
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}}};

    Compressed_row_matrix<double> original{sp, {5.0, 7.0}};
    Compressed_row_matrix<double> copy{original};

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());

    auto orig_vals = original.values();
    auto copy_vals = copy.values();
    ASSERT_EQ(std::ssize(copy_vals), std::ssize(orig_vals));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_vals); ++i) {
      EXPECT_DOUBLE_EQ(copy_vals[i], orig_vals[i]);
    }

    // Verify independent storage
    EXPECT_NE(copy_vals.data(), orig_vals.data());
  }

  TEST(compressed_row_matrix, move_construction)
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 3}, Index{3, 4}}};

    Compressed_row_matrix<double> original{sp, {5.0, 7.0}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_matrix<double> moved{std::move(original)};

    EXPECT_EQ(moved.shape(), original_shape);
    EXPECT_EQ(moved.size(), original_size);
  }

  // -- Compressed_row_matrix element access --

  TEST(compressed_row_matrix, element_access_existing)
  {
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    EXPECT_DOUBLE_EQ(mat(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(mat(2, 4), 2.0);
    EXPECT_DOUBLE_EQ(mat(3, 5), 3.0);
  }

  TEST(compressed_row_matrix, element_access_absent_in_populated_row)
  {
    // Row 2 has entries at columns 2 and 4, but not column 3
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0, 3.0}};

    EXPECT_DOUBLE_EQ(mat(2, 3), 0.0);
  }

  TEST(compressed_row_matrix, element_access_empty_row)
  {
    // Row 4 has no entries
    Compressed_row_sparsity sp{Shape{5, 6},
      {Index{2, 2}, Index{3, 5}}};

    Compressed_row_matrix<double> mat{sp, {1.0, 2.0}};

    EXPECT_DOUBLE_EQ(mat(4, 3), 0.0);
  }

  // -- Compressed_row_matrix from entries --

  TEST(compressed_row_matrix, construction_from_entry_initializer_list)
  {
    Compressed_row_matrix<double> mat{Shape{5, 6}, {
      {Index{2, 2}, 1.0},
      {Index{2, 4}, 2.0},
      {Index{3, 5}, 3.0}
    }};

    EXPECT_EQ(mat.shape(), Shape(5, 6));
    EXPECT_EQ(mat.size(), 3);

    auto rp = mat.row_ptr();
    ASSERT_EQ(std::ssize(rp), 6);
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[1], 0);
    EXPECT_EQ(rp[2], 0);
    EXPECT_EQ(rp[3], 2);
    EXPECT_EQ(rp[4], 3);
    EXPECT_EQ(rp[5], 3);

    auto ci = mat.col_ind();
    ASSERT_EQ(std::ssize(ci), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 5);

    auto vals = mat.values();
    ASSERT_EQ(std::ssize(vals), 3);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 2.0);
    EXPECT_DOUBLE_EQ(vals[2], 3.0);
  }

  TEST(compressed_row_matrix, construction_from_entries_sorts)
  {
    // Provide entries out of order — constructor must sort by (row, col)
    Compressed_row_matrix<double> mat{Shape{6, 6}, {
      {Index{4, 5}, 30.0},
      {Index{2, 4}, 20.0},
      {Index{2, 2}, 10.0}
    }};

    EXPECT_EQ(mat.size(), 3);

    auto ci = mat.col_ind();
    auto vals = mat.values();

    // Sorted order: (2,2)=10, (2,4)=20, (4,5)=30
    ASSERT_EQ(std::ssize(vals), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_DOUBLE_EQ(vals[0], 10.0);
    EXPECT_EQ(ci[1], 4);
    EXPECT_DOUBLE_EQ(vals[1], 20.0);
    EXPECT_EQ(ci[2], 5);
    EXPECT_DOUBLE_EQ(vals[2], 30.0);
  }

  // -- COO matrix to CSR matrix conversion --

  TEST(conversions, coo_matrix_to_csr_matrix_basic)
  {
    // 5x6 matrix with entries at (2,2)=1.0, (2,4)=2.0, (3,5)=3.0
    Coordinate_matrix<double> coo{Shape{5, 6}, {
      {Index{2, 2}, 1.0},
      {Index{2, 4}, 2.0},
      {Index{3, 5}, 3.0}
    }};

    auto csr = sparkit::data::detail::to_compressed_row(coo);

    EXPECT_EQ(csr.size(), 3);
    EXPECT_EQ(csr.shape(), Shape(5, 6));

    auto rp = csr.row_ptr();
    ASSERT_EQ(std::ssize(rp), 6);
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[1], 0);
    EXPECT_EQ(rp[2], 0);
    EXPECT_EQ(rp[3], 2);
    EXPECT_EQ(rp[4], 3);
    EXPECT_EQ(rp[5], 3);

    auto ci = csr.col_ind();
    ASSERT_EQ(std::ssize(ci), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 5);

    auto vals = csr.values();
    ASSERT_EQ(std::ssize(vals), 3);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 2.0);
    EXPECT_DOUBLE_EQ(vals[2], 3.0);
  }

  TEST(conversions, coo_matrix_to_csr_matrix_empty)
  {
    Coordinate_matrix<double> coo{Shape{3, 3}};
    auto csr = sparkit::data::detail::to_compressed_row(coo);

    EXPECT_EQ(csr.size(), 0);
    EXPECT_EQ(csr.shape(), Shape(3, 3));
    EXPECT_TRUE(csr.values().empty());
  }

  TEST(conversions, coo_matrix_to_csr_matrix_values_follow_sort_order)
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
    ASSERT_EQ(std::ssize(vals), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_DOUBLE_EQ(vals[0], 10.0);
    EXPECT_EQ(ci[1], 4);
    EXPECT_DOUBLE_EQ(vals[1], 20.0);
    EXPECT_EQ(ci[2], 5);
    EXPECT_DOUBLE_EQ(vals[2], 30.0);
  }

  TEST(conversions, coo_matrix_to_csr_matrix_preserves_shape)
  {
    Coordinate_matrix<double> coo{Shape{7, 9}, {
      {Index{3, 4}, 1.0}
    }};

    auto csr = sparkit::data::detail::to_compressed_row(coo);
    EXPECT_EQ(csr.shape(), Shape(7, 9));
  }

} // end of namespace sparkit::testing
