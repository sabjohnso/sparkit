#include <sparkit/data/Jagged_diagonal_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>
#include <numeric>

namespace sparkit::data::detail {

  Jagged_diagonal_sparsity::Impl::Impl(Shape shape, std::vector<Index> indices)
      : shape_(shape), total_size_(0), perm_(), jdiag_(), col_ind_() {
    auto nrow = shape.row();

    // Sort by (row, column) and deduplicate
    std::sort(begin(indices), end(indices), [](Index const& a, Index const& b) {
      return a.row() != b.row() ? a.row() < b.row() : a.column() < b.column();
    });
    auto last = std::unique(begin(indices), end(indices));
    indices.erase(last, end(indices));

    total_size_ = static_cast<size_type>(indices.size());

    if (total_size_ == 0) {
      jdiag_.push_back(0);
      return;
    }

    // Count entries per row
    std::vector<size_type> nnz_per_row(static_cast<std::size_t>(nrow), 0);
    for (auto const& idx : indices) {
      ++nnz_per_row[static_cast<std::size_t>(idx.row())];
    }

    // Build permutation: argsort by decreasing nnz, breaking ties by
    // original row index (equivalent to stable_sort without using the
    // deprecated std::get_temporary_buffer)
    perm_.resize(static_cast<std::size_t>(nrow));
    std::iota(perm_.begin(), perm_.end(), size_type{0});
    std::sort(perm_.begin(), perm_.end(), [&](size_type a, size_type b) {
      auto na = nnz_per_row[static_cast<std::size_t>(a)];
      auto nb = nnz_per_row[static_cast<std::size_t>(b)];
      return na != nb ? na > nb : a < b;
    });

    // Find max nnz per row
    auto max_nnz = nnz_per_row[static_cast<std::size_t>(perm_[0])];

    // Build jdiag pointers
    // jdiag[k] = start of jagged diagonal k in col_ind
    // Width of JD k = number of rows with >= k+1 entries
    jdiag_.resize(static_cast<std::size_t>(max_nnz + 1));
    jdiag_[0] = 0;
    for (size_type k = 0; k < max_nnz; ++k) {
      // Count rows with >= k+1 entries
      size_type width = 0;
      for (std::size_t i = 0; i < perm_.size(); ++i) {
        if (nnz_per_row[static_cast<std::size_t>(perm_[i])] >= k + 1) {
          ++width;
        } else {
          break; // perm is sorted by decreasing nnz
        }
      }
      jdiag_[static_cast<std::size_t>(k + 1)] =
          jdiag_[static_cast<std::size_t>(k)] + width;
    }

    // Build CSR-like row pointers and sorted column indices per row
    // for efficient filling of col_ind in JD order
    std::vector<std::vector<size_type>> row_cols(
        static_cast<std::size_t>(nrow));
    for (auto const& idx : indices) {
      row_cols[static_cast<std::size_t>(idx.row())].push_back(idx.column());
    }

    // Fill col_ind in jagged diagonal order
    col_ind_.resize(static_cast<std::size_t>(total_size_));
    for (size_type k = 0; k < max_nnz; ++k) {
      auto width = jdiag_[static_cast<std::size_t>(k + 1)] -
                   jdiag_[static_cast<std::size_t>(k)];
      for (size_type i = 0; i < width; ++i) {
        auto orig_row = perm_[static_cast<std::size_t>(i)];
        col_ind_[static_cast<std::size_t>(jdiag_[static_cast<std::size_t>(k)] +
                                          i)] =
            row_cols[static_cast<std::size_t>(orig_row)]
                    [static_cast<std::size_t>(k)];
      }
    }
  }

  Shape
  Jagged_diagonal_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Jagged_diagonal_sparsity::Impl::size() const {
    return total_size_;
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::Impl::perm() const {
    return perm_;
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::Impl::jdiag() const {
    return jdiag_;
  }

  std::span<size_type const>
  Jagged_diagonal_sparsity::Impl::col_ind() const {
    return col_ind_;
  }

} // end of namespace sparkit::data::detail
