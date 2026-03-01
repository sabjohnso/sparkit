//
// ... Standard header files
//
#include <set>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/multifrontal_symbolic.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  Multifrontal_symbolic
  multifrontal_analyze(
    Compressed_row_sparsity const& L_pattern,
    Supernode_partition const& partition,
    Assembly_tree const& tree) {
    auto n = L_pattern.shape().row();
    auto ns = partition.n_supernodes;
    auto rp = L_pattern.row_ptr();
    auto ci = L_pattern.col_ind();

    // Build supernode maps: for each supernode s, collect all row indices
    // that appear in any column of s in L_pattern.
    //
    // L_pattern is lower triangular CSR. For supernode s with columns
    // [col_start, col_end), we need rows i >= col_end that have an entry
    // in any of those columns.
    std::vector<Supernode_map> maps(static_cast<std::size_t>(ns));

    for (size_type s = 0; s < ns; ++s) {
      auto col_start = partition.snode_start[static_cast<std::size_t>(s)];
      auto col_end = partition.snode_start[static_cast<std::size_t>(s + 1)];
      auto snode_size = col_end - col_start;

      // Scan rows beyond the supernode to find update rows
      std::set<size_type> update_set;
      for (size_type i = col_end; i < n; ++i) {
        for (auto p = rp[i]; p < rp[i + 1]; ++p) {
          if (ci[p] >= col_start && ci[p] < col_end) {
            update_set.insert(i);
            break;
          }
        }
      }

      // Row indices: supernode columns first, then sorted update rows
      std::vector<size_type> row_indices;
      row_indices.reserve(
        static_cast<std::size_t>(snode_size) + update_set.size());
      for (auto j = col_start; j < col_end; ++j) {
        row_indices.push_back(j);
      }
      for (auto r : update_set) {
        row_indices.push_back(r);
      }

      auto front_size = static_cast<size_type>(row_indices.size());
      maps[static_cast<std::size_t>(s)] =
        Supernode_map{std::move(row_indices), snode_size, front_size};
    }

    // Build relative maps: for each child c, map child's update indices
    // into parent s's row_indices via merge scan (both sorted)
    std::vector<std::vector<size_type>> relative_maps(
      static_cast<std::size_t>(ns));

    for (size_type s = 0; s < ns; ++s) {
      auto const& children = tree.snode_children[static_cast<std::size_t>(s)];
      auto const& parent_rows = maps[static_cast<std::size_t>(s)].row_indices;

      for (auto c : children) {
        auto const& child_rows = maps[static_cast<std::size_t>(c)].row_indices;
        auto child_snode_size = maps[static_cast<std::size_t>(c)].snode_size;

        std::vector<size_type> rmap;
        auto update_begin = static_cast<std::size_t>(child_snode_size);

        size_type pi = 0;
        for (auto k = update_begin; k < child_rows.size(); ++k) {
          auto row = child_rows[k];
          while (pi < std::ssize(parent_rows) &&
                 parent_rows[static_cast<std::size_t>(pi)] < row) {
            ++pi;
          }
          rmap.push_back(pi);
        }

        relative_maps[static_cast<std::size_t>(c)] = std::move(rmap);
      }
    }

    return Multifrontal_symbolic{
      n, partition, tree, std::move(maps), std::move(relative_maps)};
  }

} // end of namespace sparkit::data::detail
