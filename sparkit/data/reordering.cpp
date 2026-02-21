//
// ... Standard header files
//
#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
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

  // ================================================================
  // Approximate Minimum Degree (AMD) ordering
  //
  // Reference: Amestoy, Davis, Duff — "An Approximate Minimum Degree
  // Ordering Algorithm" (SIAM J. Matrix Anal. Appl., 1996).
  //
  // Uses a quotient graph to simulate elimination without building
  // the filled graph explicitly.
  // ================================================================

  namespace {

    enum class Amd_status : char { variable, element, absorbed };

    struct Amd_state
    {
      size_type n;

      // Flat adjacency storage with per-node head/length
      std::vector<size_type> adj;
      std::vector<size_type> head;
      std::vector<size_type> len;
      size_type adj_capacity;

      // Per-node metadata
      std::vector<Amd_status> status;
      std::vector<size_type> degree;
      std::vector<size_type> weight;
      std::vector<size_type> representative;

      // Degree bucket structure (doubly-linked lists)
      std::vector<size_type> bucket_head;
      std::vector<size_type> bucket_next;
      std::vector<size_type> bucket_prev;
      size_type min_degree;

      // Elimination tracking
      size_type n_eliminated;
      std::vector<size_type> order;

      // Reusable marker (generation counter pattern)
      std::vector<size_type> marker;
      size_type marker_gen;
    };

    constexpr size_type none = -1;

    void
    mark(Amd_state& s, size_type i)
    {
      s.marker[static_cast<std::size_t>(i)] = s.marker_gen;
    }

    bool
    is_marked(Amd_state const& s, size_type i)
    {
      return s.marker[static_cast<std::size_t>(i)] == s.marker_gen;
    }

    void
    next_generation(Amd_state& s)
    {
      ++s.marker_gen;
    }

    // Remove node from its degree bucket
    void
    bucket_remove(Amd_state& s, size_type node)
    {
      auto prev = s.bucket_prev[static_cast<std::size_t>(node)];
      auto next = s.bucket_next[static_cast<std::size_t>(node)];

      if (prev != none) {
        s.bucket_next[static_cast<std::size_t>(prev)] = next;
      } else {
        auto d = s.degree[static_cast<std::size_t>(node)];
        s.bucket_head[static_cast<std::size_t>(d)] = next;
      }

      if (next != none) {
        s.bucket_prev[static_cast<std::size_t>(next)] = prev;
      }

      s.bucket_next[static_cast<std::size_t>(node)] = none;
      s.bucket_prev[static_cast<std::size_t>(node)] = none;
    }

    // Insert node into the bucket for its current degree
    void
    bucket_insert(Amd_state& s, size_type node)
    {
      auto d = s.degree[static_cast<std::size_t>(node)];
      auto old_head = s.bucket_head[static_cast<std::size_t>(d)];

      s.bucket_next[static_cast<std::size_t>(node)] = old_head;
      s.bucket_prev[static_cast<std::size_t>(node)] = none;

      if (old_head != none) {
        s.bucket_prev[static_cast<std::size_t>(old_head)] = node;
      }

      s.bucket_head[static_cast<std::size_t>(d)] = node;

      if (d < s.min_degree) {
        s.min_degree = d;
      }
    }

    // Ensure adj has room for additional entries
    void
    ensure_adj_capacity(Amd_state& s, size_type needed)
    {
      auto required = s.adj_capacity + needed;
      if (static_cast<std::size_t>(required) > s.adj.size()) {
        auto new_cap = std::max(
          static_cast<std::size_t>(required),
          s.adj.size() * 2);
        s.adj.resize(new_cap, none);
      }
    }

    // Allocate a contiguous block in adj[], returns the start index
    size_type
    alloc_adj(Amd_state& s, size_type count)
    {
      ensure_adj_capacity(s, count);
      auto start = s.adj_capacity;
      s.adj_capacity += count;
      return start;
    }

    // Set node's adjacency list to the given entries
    void
    set_adj(Amd_state& s, size_type node,
            std::vector<size_type> const& entries)
    {
      auto count = static_cast<size_type>(entries.size());
      if (count == 0) {
        s.head[static_cast<std::size_t>(node)] = 0;
        s.len[static_cast<std::size_t>(node)] = 0;
        return;
      }

      // Try to reuse existing slot if it fits
      auto old_len = s.len[static_cast<std::size_t>(node)];
      size_type start;
      if (count <= old_len) {
        start = s.head[static_cast<std::size_t>(node)];
      } else {
        start = alloc_adj(s, count);
      }

      for (size_type k = 0; k < count; ++k) {
        s.adj[static_cast<std::size_t>(start + k)] = entries[
          static_cast<std::size_t>(k)];
      }
      s.head[static_cast<std::size_t>(node)] = start;
      s.len[static_cast<std::size_t>(node)] = count;
    }

    Amd_state
    amd_initialize(Compressed_row_sparsity const& sym)
    {
      auto n = sym.shape().row();
      auto rp = sym.row_ptr();
      auto ci = sym.col_ind();

      Amd_state s;
      s.n = n;

      // Count total adjacency entries (excluding diagonal)
      size_type total_adj = 0;
      for (size_type i = 0; i < n; ++i) {
        for (auto j = rp[i]; j < rp[i + 1]; ++j) {
          if (ci[j] != i) ++total_adj;
        }
      }

      // Allocate with headroom for fill during elimination
      auto initial_cap = std::max(total_adj * 2, n * 2);
      s.adj.resize(static_cast<std::size_t>(initial_cap), none);
      s.head.resize(static_cast<std::size_t>(n));
      s.len.resize(static_cast<std::size_t>(n));

      // Fill adjacency lists (exclude diagonal, exclude self)
      size_type pos = 0;
      for (size_type i = 0; i < n; ++i) {
        s.head[static_cast<std::size_t>(i)] = pos;
        size_type count = 0;
        for (auto j = rp[i]; j < rp[i + 1]; ++j) {
          if (ci[j] != i) {
            s.adj[static_cast<std::size_t>(pos++)] = ci[j];
            ++count;
          }
        }
        s.len[static_cast<std::size_t>(i)] = count;
      }
      s.adj_capacity = pos;

      // Per-node metadata
      auto un = static_cast<std::size_t>(n);
      s.status.assign(un, Amd_status::variable);
      s.degree.resize(un);
      s.weight.assign(un, 1);
      s.representative.resize(un);
      std::iota(s.representative.begin(), s.representative.end(),
                size_type{0});

      // Initialize degrees = adjacency degree
      for (size_type i = 0; i < n; ++i) {
        s.degree[static_cast<std::size_t>(i)] =
          s.len[static_cast<std::size_t>(i)];
      }

      // Degree buckets
      s.bucket_head.assign(un, none);
      s.bucket_next.assign(un, none);
      s.bucket_prev.assign(un, none);
      s.min_degree = n;

      for (size_type i = 0; i < n; ++i) {
        bucket_insert(s, i);
      }

      s.n_eliminated = 0;
      s.order.resize(un, none);

      s.marker.assign(un, size_type{0});
      s.marker_gen = 1;

      return s;
    }

    // Select the pivot: lowest-index node in the minimum degree bucket
    size_type
    amd_select_pivot(Amd_state& s)
    {
      while (s.min_degree < s.n &&
             s.bucket_head[static_cast<std::size_t>(s.min_degree)] == none) {
        ++s.min_degree;
      }
      if (s.min_degree >= s.n) return none;

      // Find lowest-index node in this bucket for determinism
      auto best = s.bucket_head[static_cast<std::size_t>(s.min_degree)];
      for (auto cur = s.bucket_next[static_cast<std::size_t>(best)];
           cur != none;
           cur = s.bucket_next[static_cast<std::size_t>(cur)]) {
        if (cur < best) best = cur;
      }
      return best;
    }

    // Eliminate pivot: mark as element, compute reach (the set of
    // variable neighbors reachable through the quotient graph),
    // set element's adjacency to the reach, and return affected set.
    std::vector<size_type>
    amd_eliminate(Amd_state& s, size_type pivot)
    {
      bucket_remove(s, pivot);
      s.status[static_cast<std::size_t>(pivot)] = Amd_status::element;

      // Record elimination order (expand supervariable)
      auto w = s.weight[static_cast<std::size_t>(pivot)];
      for (size_type k = 0; k < w; ++k) {
        s.order[static_cast<std::size_t>(s.n_eliminated++)] = pivot;
      }

      // Gather the reach: union of variable neighbors of pivot
      // and variables reachable through adjacent elements.
      next_generation(s);
      mark(s, pivot);

      std::vector<size_type> reach;
      auto p_head = s.head[static_cast<std::size_t>(pivot)];
      auto p_len = s.len[static_cast<std::size_t>(pivot)];

      // Collect adjacent elements and direct variable neighbors
      std::vector<size_type> adj_elements;
      for (size_type k = 0; k < p_len; ++k) {
        auto nbr = s.adj[static_cast<std::size_t>(p_head + k)];
        if (nbr == none) continue;
        if (s.status[static_cast<std::size_t>(nbr)] == Amd_status::absorbed) {
          continue;
        }
        if (is_marked(s, nbr)) continue;
        mark(s, nbr);

        if (s.status[static_cast<std::size_t>(nbr)] == Amd_status::variable) {
          reach.push_back(nbr);
        } else {
          // Element: traverse its adjacency for variables
          adj_elements.push_back(nbr);
          auto e_head = s.head[static_cast<std::size_t>(nbr)];
          auto e_len = s.len[static_cast<std::size_t>(nbr)];
          for (size_type j = 0; j < e_len; ++j) {
            auto v = s.adj[static_cast<std::size_t>(e_head + j)];
            if (v == none) continue;
            if (s.status[static_cast<std::size_t>(v)] !=
                Amd_status::variable) continue;
            if (is_marked(s, v)) continue;
            mark(s, v);
            reach.push_back(v);
          }
        }
      }

      // Set the new element's adjacency to the reach
      set_adj(s, pivot, reach);

      // Aggressive absorption: absorb adjacent elements whose
      // variable set is a subset of the new element's reach
      for (auto e : adj_elements) {
        bool subset = true;
        auto e_head = s.head[static_cast<std::size_t>(e)];
        auto e_len = s.len[static_cast<std::size_t>(e)];
        for (size_type j = 0; j < e_len && subset; ++j) {
          auto v = s.adj[static_cast<std::size_t>(e_head + j)];
          if (v == none) continue;
          if (s.status[static_cast<std::size_t>(v)] !=
              Amd_status::variable) continue;
          if (!is_marked(s, v)) subset = false;
        }
        if (subset) {
          s.status[static_cast<std::size_t>(e)] = Amd_status::absorbed;
          s.representative[static_cast<std::size_t>(e)] = pivot;
        }
      }

      return reach;
    }

    // For each affected variable, rebuild adjacency: keep variable
    // neighbors, replace element references with the new element,
    // then compute exact external degree.
    void
    amd_update_degrees(Amd_state& s, size_type pivot,
                       std::vector<size_type> const& affected)
    {
      for (auto v : affected) {
        auto v_head = s.head[static_cast<std::size_t>(v)];
        auto v_len = s.len[static_cast<std::size_t>(v)];

        // Rebuild: keep live variables and non-absorbed elements,
        // ensure the new pivot element is referenced exactly once
        std::vector<size_type> new_adj;
        bool has_pivot = false;

        for (size_type k = 0; k < v_len; ++k) {
          auto nbr = s.adj[static_cast<std::size_t>(v_head + k)];
          if (nbr == none) continue;

          auto st = s.status[static_cast<std::size_t>(nbr)];
          if (st == Amd_status::absorbed) continue;
          if (nbr == pivot) {
            if (!has_pivot) {
              new_adj.push_back(pivot);
              has_pivot = true;
            }
            continue;
          }
          if (st == Amd_status::variable || st == Amd_status::element) {
            new_adj.push_back(nbr);
          }
        }

        if (!has_pivot) {
          new_adj.push_back(pivot);
        }

        set_adj(s, v, new_adj);

        // Compute exact external degree via marker-based set union.
        // The degree of v is |{variables reachable through v's
        // adjacency (elements + direct variables)} \ {v}|.
        next_generation(s);
        mark(s, v);

        size_type ext_degree = 0;
        auto nv_head = s.head[static_cast<std::size_t>(v)];
        auto nv_len = s.len[static_cast<std::size_t>(v)];

        for (size_type k = 0; k < nv_len; ++k) {
          auto nbr = s.adj[static_cast<std::size_t>(nv_head + k)];
          if (nbr == none) continue;
          if (is_marked(s, nbr)) continue;

          auto st = s.status[static_cast<std::size_t>(nbr)];
          if (st == Amd_status::variable) {
            mark(s, nbr);
            ext_degree += s.weight[static_cast<std::size_t>(nbr)];
          } else if (st == Amd_status::element) {
            mark(s, nbr);
            auto e_head = s.head[static_cast<std::size_t>(nbr)];
            auto e_len = s.len[static_cast<std::size_t>(nbr)];
            for (size_type j = 0; j < e_len; ++j) {
              auto u = s.adj[static_cast<std::size_t>(e_head + j)];
              if (u == none) continue;
              if (s.status[static_cast<std::size_t>(u)] !=
                  Amd_status::variable) continue;
              if (is_marked(s, u)) continue;
              mark(s, u);
              ext_degree += s.weight[static_cast<std::size_t>(u)];
            }
          }
        }

        // Update degree and bucket position
        auto old_deg = s.degree[static_cast<std::size_t>(v)];
        if (ext_degree != old_deg) {
          bucket_remove(s, v);
          s.degree[static_cast<std::size_t>(v)] = ext_degree;
          bucket_insert(s, v);
        }
      }
    }

    // Detect supervariables among the affected set: variables with
    // identical adjacency are merged into a single representative.
    void
    amd_detect_supervariables(Amd_state& s,
                              std::vector<size_type> const& affected)
    {
      if (affected.size() < 2) return;

      // Hash adjacency lists to find candidates
      auto hash_adj = [&](size_type v) -> std::size_t {
        auto h = s.head[static_cast<std::size_t>(v)];
        auto l = s.len[static_cast<std::size_t>(v)];
        std::size_t hash = static_cast<std::size_t>(l);
        for (size_type k = 0; k < l; ++k) {
          hash ^= static_cast<std::size_t>(
            s.adj[static_cast<std::size_t>(h + k)]) * 2654435761u;
        }
        return hash;
      };

      // Group by (degree, hash)
      std::vector<std::pair<std::size_t, size_type>> keyed;
      keyed.reserve(affected.size());
      for (auto v : affected) {
        if (s.status[static_cast<std::size_t>(v)] != Amd_status::variable) {
          continue;
        }
        auto key = (static_cast<std::size_t>(
          s.degree[static_cast<std::size_t>(v)]) << 32) | hash_adj(v);
        keyed.push_back({key, v});
      }
      std::sort(keyed.begin(), keyed.end());

      for (std::size_t i = 0; i < keyed.size(); ++i) {
        auto vi = keyed[i].second;
        if (s.status[static_cast<std::size_t>(vi)] != Amd_status::variable) {
          continue;
        }

        for (std::size_t j = i + 1;
             j < keyed.size() && keyed[j].first == keyed[i].first;
             ++j) {
          auto vj = keyed[j].second;
          if (s.status[static_cast<std::size_t>(vj)] !=
              Amd_status::variable) continue;

          // Full comparison of adjacency lists
          auto hi = s.head[static_cast<std::size_t>(vi)];
          auto li = s.len[static_cast<std::size_t>(vi)];
          auto hj = s.head[static_cast<std::size_t>(vj)];
          auto lj = s.len[static_cast<std::size_t>(vj)];

          if (li != lj) continue;

          bool same = true;
          for (size_type k = 0; k < li && same; ++k) {
            if (s.adj[static_cast<std::size_t>(hi + k)] !=
                s.adj[static_cast<std::size_t>(hj + k)]) {
              same = false;
            }
          }

          if (same) {
            // Absorb vj into vi
            bucket_remove(s, vj);
            s.status[static_cast<std::size_t>(vj)] = Amd_status::absorbed;
            s.representative[static_cast<std::size_t>(vj)] = vi;
            s.weight[static_cast<std::size_t>(vi)] +=
              s.weight[static_cast<std::size_t>(vj)];
          }
        }
      }
    }

    // Convert elimination order to perm[old] = new
    std::vector<size_type>
    amd_build_permutation(Amd_state const& s)
    {
      std::vector<size_type> perm(static_cast<std::size_t>(s.n), none);

      // order[pos] = supervariable representative.
      // A supervariable rep appears weight times; assign only once.
      size_type new_pos = 0;
      for (size_type pos = 0; pos < s.n; ++pos) {
        auto node = s.order[static_cast<std::size_t>(pos)];
        if (node == none) break;
        if (perm[static_cast<std::size_t>(node)] != none) continue;
        perm[static_cast<std::size_t>(node)] = new_pos++;
      }

      // Assign absorbed nodes (supervariable non-principals)
      for (size_type i = 0; i < s.n; ++i) {
        if (perm[static_cast<std::size_t>(i)] == none) {
          perm[static_cast<std::size_t>(i)] = new_pos++;
        }
      }

      return perm;
    }

  } // end of anonymous namespace

  std::vector<size_type>
  approximate_minimum_degree(Compressed_row_sparsity const& sp)
  {
    auto nrow = sp.shape().row();
    auto ncol = sp.shape().column();
    if (nrow != ncol) {
      throw std::invalid_argument(
        "approximate_minimum_degree: matrix must be square");
    }

    auto sym = symmetrize_pattern(sp);
    auto s = amd_initialize(sym);

    while (s.n_eliminated < s.n) {
      auto pivot = amd_select_pivot(s);
      if (pivot == none) break;

      auto affected = amd_eliminate(s, pivot);
      amd_update_degrees(s, pivot, affected);
      amd_detect_supervariables(s, affected);
    }

    return amd_build_permutation(s);
  }

  // ================================================================
  // Column Approximate Minimum Degree (COLAMD) ordering
  //
  // Reference: Davis, Gilbert, Larimore, Ng — "Algorithm 836: COLAMD,
  // a Column Approximate Minimum Degree Ordering Algorithm" (ACM
  // TOMS, 2004).
  //
  // COLAMD on A is equivalent to AMD on A^T*A. We form the sparsity
  // pattern of A^T*A and delegate to approximate_minimum_degree.
  // ================================================================

  // Build the sparsity pattern of A^T*A from A's CSR, without values.
  //
  // Phase 1: Build a column-to-row mapping (CSC-like structure).
  // Phase 2: For each column j (= row j of A^T*A), find all rows
  //   containing j, then for each such row find all columns — those
  //   are j's neighbors in A^T*A. A marker array avoids duplicates.
  static Compressed_row_sparsity
  form_ata_pattern(Compressed_row_sparsity const& sp)
  {
    auto nrow = sp.shape().row();
    auto ncol = sp.shape().column();
    auto rp = sp.row_ptr();
    auto ci = sp.col_ind();
    auto un = static_cast<std::size_t>(ncol);

    // Phase 1: Build column-to-row mapping (col_ptr, row_ind)
    std::vector<size_type> col_count(un, 0);
    for (size_type i = 0; i < nrow; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        ++col_count[static_cast<std::size_t>(ci[p])];
      }
    }

    std::vector<size_type> col_ptr(un + 1, 0);
    for (size_type j = 0; j < ncol; ++j) {
      col_ptr[static_cast<std::size_t>(j + 1)] =
        col_ptr[static_cast<std::size_t>(j)] +
        col_count[static_cast<std::size_t>(j)];
    }

    auto total_entries = col_ptr[un];
    std::vector<size_type> row_ind(static_cast<std::size_t>(total_entries));
    std::vector<size_type> col_pos(col_ptr.begin(), col_ptr.end() - 1);

    for (size_type i = 0; i < nrow; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        row_ind[static_cast<std::size_t>(col_pos[
          static_cast<std::size_t>(j)]++)] = i;
      }
    }

    // Phase 2: Collect A^T*A entries as Index pairs
    std::vector<Index> indices;
    std::vector<size_type> marker(un, -1);

    for (size_type j = 0; j < ncol; ++j) {
      auto uj = static_cast<std::size_t>(j);

      // Always include diagonal
      marker[uj] = j;
      indices.push_back(Index{j, j});

      // For each row containing column j
      for (auto r = col_ptr[uj]; r < col_ptr[uj + 1]; ++r) {
        auto row = row_ind[static_cast<std::size_t>(r)];
        // For each column k in that row
        for (auto p = rp[row]; p < rp[row + 1]; ++p) {
          auto k = ci[p];
          if (marker[static_cast<std::size_t>(k)] != j) {
            marker[static_cast<std::size_t>(k)] = j;
            indices.push_back(Index{j, k});
          }
        }
      }
    }

    return Compressed_row_sparsity{
      Shape{ncol, ncol}, indices.begin(), indices.end()};
  }

  std::vector<size_type>
  column_approximate_minimum_degree(Compressed_row_sparsity const& sp)
  {
    auto ata = form_ata_pattern(sp);
    return approximate_minimum_degree(ata);
  }

} // end of namespace sparkit::data::detail
