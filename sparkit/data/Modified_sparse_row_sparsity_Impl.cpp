#include <sparkit/data/Modified_sparse_row_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Modified_sparse_row_sparsity::Impl::Impl(
    Shape shape, std::vector<Index> indices)
      : shape_(shape)
      , total_size_(0)
      , has_diagonal_(
          static_cast<std::size_t>(std::min(shape.row(), shape.column())),
          false)
      , off_diagonal_row_ptr_(static_cast<std::size_t>(shape.row() + 1), 0)
      , off_diagonal_col_ind_() {
    // Sort by (row, column)
    std::sort(begin(indices), end(indices), [](Index const& a, Index const& b) {
      return a.row() != b.row() ? a.row() < b.row() : a.column() < b.column();
    });

    // Remove duplicates
    auto last = std::unique(begin(indices), end(indices));
    indices.erase(last, end(indices));

    total_size_ = static_cast<size_type>(indices.size());

    auto diag_len = std::min(shape.row(), shape.column());

    // Separate diagonal from off-diagonal, count off-diag per row
    for (auto const& idx : indices) {
      if (idx.row() == idx.column() && idx.row() < diag_len) {
        has_diagonal_[static_cast<std::size_t>(idx.row())] = true;
      } else {
        ++off_diagonal_row_ptr_[static_cast<std::size_t>(idx.row())];
      }
    }

    // Convert counts to offsets via prefix sum
    size_type running = 0;
    for (std::size_t r = 0; r < off_diagonal_row_ptr_.size(); ++r) {
      size_type count = off_diagonal_row_ptr_[r];
      off_diagonal_row_ptr_[r] = running;
      running += count;
    }

    // Build off-diagonal col_ind
    off_diagonal_col_ind_.resize(static_cast<std::size_t>(running));
    std::vector<size_type> work(
      off_diagonal_row_ptr_.begin(), off_diagonal_row_ptr_.end());

    for (auto const& idx : indices) {
      if (idx.row() == idx.column() && idx.row() < diag_len) {
        continue; // skip diagonal
      }
      auto dest = work[static_cast<std::size_t>(idx.row())]++;
      off_diagonal_col_ind_[static_cast<std::size_t>(dest)] = idx.column();
    }
  }

  Shape
  Modified_sparse_row_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Modified_sparse_row_sparsity::Impl::size() const {
    return total_size_;
  }

  bool
  Modified_sparse_row_sparsity::Impl::has_diagonal(size_type i) const {
    return has_diagonal_[static_cast<std::size_t>(i)];
  }

  size_type
  Modified_sparse_row_sparsity::Impl::diagonal_length() const {
    return static_cast<size_type>(has_diagonal_.size());
  }

  std::span<size_type const>
  Modified_sparse_row_sparsity::Impl::off_diagonal_row_ptr() const {
    return off_diagonal_row_ptr_;
  }

  std::span<size_type const>
  Modified_sparse_row_sparsity::Impl::off_diagonal_col_ind() const {
    return off_diagonal_col_ind_;
  }

} // end of namespace sparkit::data::detail
