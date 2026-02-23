#include <sparkit/data/Ellpack_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Ellpack_sparsity::Impl::Impl(Shape shape, std::vector<Index> indices)
      : shape_(shape)
      , total_size_(0)
      , max_nnz_per_row_(0)
      , col_ind_() {
    auto nrow = shape.row();

    // Sort by (row, column) and deduplicate
    std::sort(begin(indices), end(indices), [](Index const& a, Index const& b) {
      return a.row() != b.row() ? a.row() < b.row() : a.column() < b.column();
    });
    auto last = std::unique(begin(indices), end(indices));
    indices.erase(last, end(indices));

    total_size_ = static_cast<size_type>(indices.size());

    if (total_size_ == 0) { return; }

    // Count entries per row to find max
    std::vector<size_type> row_counts(static_cast<std::size_t>(nrow), 0);
    for (auto const& idx : indices) {
      ++row_counts[static_cast<std::size_t>(idx.row())];
    }
    max_nnz_per_row_ = *std::max_element(row_counts.begin(), row_counts.end());

    // Allocate padded array with sentinel -1
    col_ind_.assign(
      static_cast<std::size_t>(nrow * max_nnz_per_row_), size_type{-1});

    // Fill in column indices
    std::vector<size_type> pos(static_cast<std::size_t>(nrow), 0);
    for (auto const& idx : indices) {
      auto r = static_cast<std::size_t>(idx.row());
      col_ind_
        [r * static_cast<std::size_t>(max_nnz_per_row_) +
         static_cast<std::size_t>(pos[r])] = idx.column();
      ++pos[r];
    }
  }

  Shape
  Ellpack_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Ellpack_sparsity::Impl::size() const {
    return total_size_;
  }

  size_type
  Ellpack_sparsity::Impl::max_nnz_per_row() const {
    return max_nnz_per_row_;
  }

  std::span<size_type const>
  Ellpack_sparsity::Impl::col_ind() const {
    return col_ind_;
  }

} // end of namespace sparkit::data::detail
