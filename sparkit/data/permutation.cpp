//
// ... Standard header files
//
#include <algorithm>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/permutation.hpp>

namespace sparkit::data::detail {

  bool
  is_valid_permutation(std::span<config::size_type const> perm)
  {
    auto n = static_cast<config::size_type>(perm.size());
    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (auto val : perm) {
      if (val < 0 || val >= n) {
        return false;
      }
      if (seen[static_cast<std::size_t>(val)]) {
        return false;
      }
      seen[static_cast<std::size_t>(val)] = true;
    }
    return true;
  }

  std::vector<config::size_type>
  inverse_permutation(std::span<config::size_type const> perm)
  {
    auto n = perm.size();
    std::vector<config::size_type> inv(n);
    for (std::size_t old_idx = 0; old_idx < n; ++old_idx) {
      inv[static_cast<std::size_t>(perm[old_idx])] =
        static_cast<config::size_type>(old_idx);
    }
    return inv;
  }

  Compressed_row_sparsity
  rperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm)
  {
    auto inv = inverse_permutation(perm);
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sp.size()));

    for (config::size_type new_row = 0; new_row < nrow; ++new_row) {
      auto old_row = inv[static_cast<std::size_t>(new_row)];
      for (auto j = rp[old_row]; j < rp[old_row + 1]; ++j) {
        indices.push_back(Index{new_row, ci[j]});
      }
    }

    return Compressed_row_sparsity{sp.shape(), indices.begin(), indices.end()};
  }

  Compressed_row_sparsity
  cperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm)
  {
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sp.size()));

    for (config::size_type row = 0; row < nrow; ++row) {
      // Collect new column indices for this row
      std::vector<config::size_type> new_cols;
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        new_cols.push_back(perm[static_cast<std::size_t>(ci[j])]);
      }
      std::sort(new_cols.begin(), new_cols.end());

      for (auto col : new_cols) {
        indices.push_back(Index{row, col});
      }
    }

    return Compressed_row_sparsity{sp.shape(), indices.begin(), indices.end()};
  }

  Compressed_row_sparsity
  dperm(Compressed_row_sparsity const& sp,
        std::span<config::size_type const> perm)
  {
    auto inv = inverse_permutation(perm);
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sp.size()));

    for (config::size_type new_row = 0; new_row < nrow; ++new_row) {
      auto old_row = inv[static_cast<std::size_t>(new_row)];

      std::vector<config::size_type> new_cols;
      for (auto j = rp[old_row]; j < rp[old_row + 1]; ++j) {
        new_cols.push_back(perm[static_cast<std::size_t>(ci[j])]);
      }
      std::sort(new_cols.begin(), new_cols.end());

      for (auto col : new_cols) {
        indices.push_back(Index{new_row, col});
      }
    }

    return Compressed_row_sparsity{sp.shape(), indices.begin(), indices.end()};
  }

} // end of namespace sparkit::data::detail
