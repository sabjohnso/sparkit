//
// ... Test header files
//
#include <gtest/gtest.h>

//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_sparsity.hpp>

namespace sparkit::testing {

  using sparkit::data::detail::Compressed_row_sparsity;
  using sparkit::data::detail::Shape;
  using sparkit::data::detail::Index;

  TEST(compressed_row_sparsity, construction_from_initializer_list)
  {
    Compressed_row_sparsity csr{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    EXPECT_EQ(csr.shape(), Shape(4, 5));
    EXPECT_EQ(csr.size(), 2);
  }

  TEST(compressed_row_sparsity, construction_empty)
  {
    Compressed_row_sparsity csr{Shape{3, 3}, {}};
    EXPECT_EQ(csr.shape(), Shape(3, 3));
    EXPECT_EQ(csr.size(), 0);
  }

  // -- Red 2: CSR structure accessors --

  TEST(compressed_row_sparsity, row_ptr_and_col_ind_structure)
  {
    // 5x6 matrix with entries at (2,2), (2,4), (3,5)
    Compressed_row_sparsity csr{Shape{5, 6},
      {Index{2, 2}, Index{2, 4}, Index{3, 5}}};

    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();

    // row_ptr has shape.row()+1 = 6 entries
    ASSERT_EQ(std::ssize(rp), 6);

    // rows 0,1 are empty
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[1], 0);
    EXPECT_EQ(rp[2], 0);
    // row 2 has 2 entries: row_ptr[3]=2
    EXPECT_EQ(rp[3], 2);
    // row 3 has 1 entry: row_ptr[4]=3
    EXPECT_EQ(rp[4], 3);
    // row 4 is empty: row_ptr[5]=3
    EXPECT_EQ(rp[5], 3);

    // col_ind has nnz=3 entries
    ASSERT_EQ(std::ssize(ci), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 5);
  }

  TEST(compressed_row_sparsity, col_ind_sorted_within_rows)
  {
    // Indices given out of order within a row
    Compressed_row_sparsity csr{Shape{4, 6},
      {Index{2, 5}, Index{2, 2}, Index{2, 4}}};

    auto ci = csr.col_ind();
    ASSERT_EQ(std::ssize(ci), 3);
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 5);
  }

  TEST(compressed_row_sparsity, duplicate_indices_are_collapsed)
  {
    Compressed_row_sparsity csr{Shape{4, 5},
      {Index{2, 3}, Index{2, 3}, Index{3, 4}}};

    EXPECT_EQ(csr.size(), 2);

    auto rp = csr.row_ptr();
    EXPECT_EQ(rp[3], 1);  // row 2 has 1 unique entry
    EXPECT_EQ(rp[4], 2);  // row 3 has 1 entry, cumulative = 2
  }

  TEST(compressed_row_sparsity, row_ptr_empty_matrix)
  {
    Compressed_row_sparsity csr{Shape{3, 3}, {}};

    auto rp = csr.row_ptr();
    ASSERT_EQ(std::ssize(rp), 4);
    for (auto v : rp) {
      EXPECT_EQ(v, 0);
    }

    EXPECT_EQ(std::ssize(csr.col_ind()), 0);
  }

  TEST(compressed_row_sparsity, single_row_multiple_entries)
  {
    Compressed_row_sparsity csr{Shape{4, 8},
      {Index{2, 7}, Index{2, 3}, Index{2, 2}, Index{2, 5}}};

    EXPECT_EQ(csr.size(), 4);

    auto rp = csr.row_ptr();
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[1], 0);
    EXPECT_EQ(rp[2], 0);
    EXPECT_EQ(rp[3], 4);
    EXPECT_EQ(rp[4], 4);

    auto ci = csr.col_ind();
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 3);
    EXPECT_EQ(ci[2], 5);
    EXPECT_EQ(ci[3], 7);
  }

  // -- Red 3: Iterator range constructor --

  TEST(compressed_row_sparsity, construction_from_iterator_range)
  {
    std::vector<Index> indices{Index{3, 2}, Index{2, 4}, Index{2, 2}};
    Compressed_row_sparsity csr{Shape{5, 6}, begin(indices), end(indices)};

    EXPECT_EQ(csr.shape(), Shape(5, 6));
    EXPECT_EQ(csr.size(), 3);

    auto ci = csr.col_ind();
    ASSERT_EQ(std::ssize(ci), 3);
    // row 2: columns 2, 4; row 3: column 2
    EXPECT_EQ(ci[0], 2);
    EXPECT_EQ(ci[1], 4);
    EXPECT_EQ(ci[2], 2);
  }

  // -- Red 4: Copy/move semantics --

  TEST(compressed_row_sparsity, copy_construction)
  {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    Compressed_row_sparsity copy{original};

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());

    auto orig_ci = original.col_ind();
    auto copy_ci = copy.col_ind();
    ASSERT_EQ(std::ssize(copy_ci), std::ssize(orig_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      EXPECT_EQ(copy_ci[i], orig_ci[i]);
    }

    // Verify independent storage (different addresses)
    EXPECT_NE(copy_ci.data(), orig_ci.data());
  }

  TEST(compressed_row_sparsity, move_construction)
  {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_sparsity moved{std::move(original)};

    EXPECT_EQ(moved.shape(), original_shape);
    EXPECT_EQ(moved.size(), original_size);
  }

  TEST(compressed_row_sparsity, copy_assignment)
  {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    Compressed_row_sparsity target{Shape{3, 3}, {}};

    target = original;

    EXPECT_EQ(target.shape(), original.shape());
    EXPECT_EQ(target.size(), original.size());

    auto orig_ci = original.col_ind();
    auto tgt_ci = target.col_ind();
    ASSERT_EQ(std::ssize(tgt_ci), std::ssize(orig_ci));
    for (std::ptrdiff_t i = 0; i < std::ssize(orig_ci); ++i) {
      EXPECT_EQ(tgt_ci[i], orig_ci[i]);
    }

    EXPECT_NE(tgt_ci.data(), orig_ci.data());
  }

  TEST(compressed_row_sparsity, move_assignment)
  {
    Compressed_row_sparsity original{Shape{4, 5}, {Index{2, 3}, Index{3, 4}}};
    auto original_size = original.size();
    auto original_shape = original.shape();

    Compressed_row_sparsity target{Shape{3, 3}, {}};
    target = std::move(original);

    EXPECT_EQ(target.shape(), original_shape);
    EXPECT_EQ(target.size(), original_size);
  }

} // end of namespace sparkit::testing
