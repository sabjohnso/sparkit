#include <sparkit/data/Diagonal_sparsity_Impl.hpp>

//
// ... Standard header files
//
#include <algorithm>

namespace sparkit::data::detail {

  Diagonal_sparsity::Impl::Impl(Shape shape, std::vector<size_type> offsets)
      : shape_(shape)
      , offsets_(std::move(offsets))
      , total_size_(0) {
    // Sort and deduplicate offsets
    std::sort(offsets_.begin(), offsets_.end());
    offsets_.erase(
      std::unique(offsets_.begin(), offsets_.end()), offsets_.end());

    // Compute total size: number of valid positions on each diagonal
    auto nrow = shape_.row();
    auto ncol = shape_.column();

    for (auto offset : offsets_) {
      if (offset >= 0) {
        // Super-diagonal or main: length = min(nrow, ncol - offset)
        auto len = std::min(nrow, ncol - offset);
        if (len > 0) { total_size_ += len; }
      } else {
        // Sub-diagonal: length = min(nrow + offset, ncol)
        auto len = std::min(nrow + offset, ncol);
        if (len > 0) { total_size_ += len; }
      }
    }
  }

  Shape
  Diagonal_sparsity::Impl::shape() const {
    return shape_;
  }

  size_type
  Diagonal_sparsity::Impl::size() const {
    return total_size_;
  }

  std::span<size_type const>
  Diagonal_sparsity::Impl::offsets() const {
    return offsets_;
  }

  size_type
  Diagonal_sparsity::Impl::num_diagonals() const {
    return static_cast<size_type>(offsets_.size());
  }

} // end of namespace sparkit::data::detail
