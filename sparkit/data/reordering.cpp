//
// ... Standard header files
//
#include <algorithm>
#include <queue>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/reordering.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  Compressed_row_sparsity
  symmetrize_pattern(Compressed_row_sparsity const& sp)
  {
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    // Collect all (i,j) pairs including transposed entries
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(sp.size() * 2));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
        if (row != ci[j]) {
          indices.push_back(Index{ci[j], row});
        }
      }
    }

    // The CSR constructor handles sorting and deduplication
    return Compressed_row_sparsity{sp.shape(), indices.begin(), indices.end()};
  }

  std::vector<size_type>
  adjacency_degree(Compressed_row_sparsity const& sp)
  {
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    std::vector<size_type> deg(static_cast<std::size_t>(nrow));
    for (size_type row = 0; row < nrow; ++row) {
      size_type count = 0;
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        if (ci[j] != row) {
          ++count;
        }
      }
      deg[static_cast<std::size_t>(row)] = count;
    }
    return deg;
  }

  // BFS from a given start node. Returns the eccentricity (max level)
  // and the last level set. Operates on the given symmetric pattern.
  static
  std::pair<size_type, std::vector<size_type>>
  bfs_levels(Compressed_row_sparsity const& sp,
             size_type start,
             std::vector<size_type> const& deg)
  {
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto nrow = sp.shape().row();

    std::vector<size_type> level(static_cast<std::size_t>(nrow), -1);
    std::queue<size_type> queue;

    level[static_cast<std::size_t>(start)] = 0;
    queue.push(start);

    size_type max_level = 0;
    std::vector<size_type> last_level;

    while (!queue.empty()) {
      auto node = queue.front();
      queue.pop();
      auto node_level = level[static_cast<std::size_t>(node)];

      if (node_level > max_level) {
        max_level = node_level;
        last_level.clear();
      }
      if (node_level == max_level) {
        last_level.push_back(node);
      }

      // Visit neighbors sorted by degree (for determinism)
      std::vector<size_type> neighbors;
      for (auto j = rp[node]; j < rp[node + 1]; ++j) {
        if (ci[j] != node && level[static_cast<std::size_t>(ci[j])] == -1) {
          neighbors.push_back(ci[j]);
        }
      }
      std::sort(neighbors.begin(), neighbors.end(),
        [&deg](size_type a, size_type b) {
          return deg[static_cast<std::size_t>(a)]
               < deg[static_cast<std::size_t>(b)];
        });

      for (auto nbr : neighbors) {
        if (level[static_cast<std::size_t>(nbr)] == -1) {
          level[static_cast<std::size_t>(nbr)] = node_level + 1;
          queue.push(nbr);
        }
      }
    }

    return {max_level, last_level};
  }

  size_type
  pseudo_peripheral_node(Compressed_row_sparsity const& sp)
  {
    auto sym = symmetrize_pattern(sp);
    auto deg = adjacency_degree(sym);
    auto nrow = sp.shape().row();

    // Start from the node with minimum degree
    size_type start = 0;
    size_type min_deg = deg[0];
    for (size_type i = 1; i < nrow; ++i) {
      if (deg[static_cast<std::size_t>(i)] < min_deg) {
        min_deg = deg[static_cast<std::size_t>(i)];
        start = i;
      }
    }

    // George-Liu algorithm
    auto [eccentricity, last_level] = bfs_levels(sym, start, deg);

    for (;;) {
      // Pick node in last level set with minimum degree
      size_type candidate = last_level[0];
      size_type cand_deg = deg[static_cast<std::size_t>(candidate)];
      for (std::size_t k = 1; k < last_level.size(); ++k) {
        auto d = deg[static_cast<std::size_t>(last_level[k])];
        if (d < cand_deg) {
          candidate = last_level[k];
          cand_deg = d;
        }
      }

      auto [new_ecc, new_last] = bfs_levels(sym, candidate, deg);
      if (new_ecc <= eccentricity) {
        return start;
      }

      start = candidate;
      eccentricity = new_ecc;
      last_level = std::move(new_last);
    }
  }

  std::vector<size_type>
  reverse_cuthill_mckee(Compressed_row_sparsity const& sp)
  {
    auto sym = symmetrize_pattern(sp);
    auto deg = adjacency_degree(sym);
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();
    auto nrow = sp.shape().row();

    // ordering[position] = old_node (BFS order)
    std::vector<size_type> ordering;
    ordering.reserve(static_cast<std::size_t>(nrow));

    std::vector<bool> visited(static_cast<std::size_t>(nrow), false);

    // Handle potentially disconnected components
    while (static_cast<size_type>(ordering.size()) < nrow) {
      // Find starting node for this component
      size_type start;
      if (ordering.empty()) {
        start = pseudo_peripheral_node(sp);
      } else {
        // Pick unvisited node with minimum degree
        start = -1;
        size_type min_d = nrow + 1;
        for (size_type i = 0; i < nrow; ++i) {
          if (!visited[static_cast<std::size_t>(i)]) {
            if (deg[static_cast<std::size_t>(i)] < min_d) {
              min_d = deg[static_cast<std::size_t>(i)];
              start = i;
            }
          }
        }
      }

      // BFS from start, sorting neighbors by increasing degree
      visited[static_cast<std::size_t>(start)] = true;
      ordering.push_back(start);

      for (std::size_t head = ordering.size() - 1;
           head < ordering.size(); ++head) {
        auto node = ordering[head];

        // Gather unvisited neighbors
        std::vector<size_type> neighbors;
        for (auto j = rp[node]; j < rp[node + 1]; ++j) {
          if (ci[j] != node && !visited[static_cast<std::size_t>(ci[j])]) {
            neighbors.push_back(ci[j]);
          }
        }

        // Sort by increasing degree
        std::sort(neighbors.begin(), neighbors.end(),
          [&deg](size_type a, size_type b) {
            return deg[static_cast<std::size_t>(a)]
                 < deg[static_cast<std::size_t>(b)];
          });

        for (auto nbr : neighbors) {
          if (!visited[static_cast<std::size_t>(nbr)]) {
            visited[static_cast<std::size_t>(nbr)] = true;
            ordering.push_back(nbr);
          }
        }
      }
    }

    // Reverse the ordering (Cuthill-McKee -> Reverse Cuthill-McKee)
    std::reverse(ordering.begin(), ordering.end());

    // Convert from ordering[new_pos] = old_node
    // to perm[old_node] = new_pos
    std::vector<size_type> perm(static_cast<std::size_t>(nrow));
    for (std::size_t new_pos = 0; new_pos < ordering.size(); ++new_pos) {
      perm[static_cast<std::size_t>(ordering[new_pos])] =
        static_cast<size_type>(new_pos);
    }

    return perm;
  }

} // end of namespace sparkit::data::detail
