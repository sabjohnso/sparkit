//
// ... Standard header files
//
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/supernode.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  // Fundamental supernodes via Liu's characterisation:
  // Column j merges with j-1 iff parent[j-1] == j AND
  // col_count[j-1] == col_count[j] + 1.
  //
  // col_count[j] is the number of nonzeros in column j of L (including
  // diagonal). We compute it by scanning L_pattern rows and counting
  // occurrences of each column index.

  Supernode_partition
  find_supernodes(
    Compressed_row_sparsity const& L_pattern,
    std::span<size_type const> parent) {
    auto n = L_pattern.shape().row();
    auto rp = L_pattern.row_ptr();
    auto ci = L_pattern.col_ind();

    // Compute column counts from L_pattern (lower triangular)
    std::vector<size_type> col_count(static_cast<std::size_t>(n), 0);
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        ++col_count[static_cast<std::size_t>(ci[p])];
      }
    }

    // Determine supernode boundaries
    // Column 0 always starts a new supernode.
    // Column j merges with j-1 iff parent[j-1] == j
    // and col_count[j-1] == col_count[j] + 1.
    std::vector<size_type> snode_start;
    snode_start.push_back(0);

    for (size_type j = 1; j < n; ++j) {
      auto prev_parent = parent[static_cast<std::size_t>(j - 1)];
      auto prev_count = col_count[static_cast<std::size_t>(j - 1)];
      auto curr_count = col_count[static_cast<std::size_t>(j)];

      if (prev_parent != j || prev_count != curr_count + 1) {
        snode_start.push_back(j);
      }
    }
    snode_start.push_back(n);

    auto n_supernodes = static_cast<size_type>(snode_start.size()) - 1;

    // Build membership array
    std::vector<size_type> membership(static_cast<std::size_t>(n));
    for (size_type s = 0; s < n_supernodes; ++s) {
      for (auto j = snode_start[static_cast<std::size_t>(s)];
           j < snode_start[static_cast<std::size_t>(s + 1)];
           ++j) {
        membership[static_cast<std::size_t>(j)] = s;
      }
    }

    return Supernode_partition{
      std::move(snode_start), std::move(membership), n_supernodes};
  }

} // end of namespace sparkit::data::detail
