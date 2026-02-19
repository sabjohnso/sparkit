#include <sparkit/data/Symmetric_compressed_row_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail
{

  Symmetric_compressed_row_sparsity::Impl::Impl(
    Shape shape, std::vector<Index> indices)
    : shape_(shape)
    , row_ptr_(static_cast<std::size_t>(shape.row() + 1), 0)
    , col_ind_()
  {
    // Normalize all indices to lower triangle: row >= col
    for (auto& idx : indices) {
      if (idx.row() < idx.column()) {
        idx = Index{idx.column(), idx.row()};
      }
    }

    // Sort by (row, column)
    std::sort(begin(indices), end(indices),
      [](Index const& a, Index const& b) {
        return a.row() != b.row()
          ? a.row() < b.row()
          : a.column() < b.column();
      });

    // Remove duplicates
    auto last = std::unique(begin(indices), end(indices));
    indices.erase(last, end(indices));

    // Build col_ind_ and count entries per row
    col_ind_.reserve(indices.size());
    for (auto const& idx : indices) {
      col_ind_.push_back(idx.column());
      ++row_ptr_[static_cast<std::size_t>(idx.row())];
    }

    // Convert counts to offsets via prefix sum
    size_type running = 0;
    for (std::size_t r = 0; r < row_ptr_.size(); ++r) {
      size_type count = row_ptr_[r];
      row_ptr_[r] = running;
      running += count;
    }
  }

  Shape
  Symmetric_compressed_row_sparsity::Impl::shape() const { return shape_; }

  size_type
  Symmetric_compressed_row_sparsity::Impl::size() const
  {
    return static_cast<size_type>(col_ind_.size());
  }

  std::span<size_type const>
  Symmetric_compressed_row_sparsity::Impl::row_ptr() const { return row_ptr_; }

  std::span<size_type const>
  Symmetric_compressed_row_sparsity::Impl::col_ind() const { return col_ind_; }

} // end of namespace sparkit::data::detail
