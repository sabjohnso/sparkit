#include <sparkit/data/Compressed_column_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Compressed_column_sparsity::Impl::Impl(Shape shape,
                                         std::vector<Index> indices)
      : shape_(shape),
        col_ptr_(static_cast<std::size_t>(shape.column() + 1), 0), row_ind_() {
    // Sort by (column, row)
    std::sort(begin(indices), end(indices), [](Index const& a, Index const& b) {
      return a.column() != b.column() ? a.column() < b.column()
                                      : a.row() < b.row();
    });

    // Remove duplicates
    auto last = std::unique(begin(indices), end(indices));
    indices.erase(last, end(indices));

    // Build row_ind_ and count entries per column
    row_ind_.reserve(indices.size());
    for (auto const& idx : indices) {
      row_ind_.push_back(idx.row());
      ++col_ptr_[static_cast<std::size_t>(idx.column())];
    }

    // Convert counts to offsets via prefix sum
    size_type running = 0;
    for (std::size_t c = 0; c < col_ptr_.size(); ++c) {
      size_type count = col_ptr_[c];
      col_ptr_[c] = running;
      running += count;
    }
  }

  Shape
  Compressed_column_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Compressed_column_sparsity::Impl::size() const {
    return static_cast<size_type>(row_ind_.size());
  }

  std::span<size_type const>
  Compressed_column_sparsity::Impl::col_ptr() const {
    return col_ptr_;
  }

  std::span<size_type const>
  Compressed_column_sparsity::Impl::row_ind() const {
    return row_ind_;
  }

} // end of namespace sparkit::data::detail
