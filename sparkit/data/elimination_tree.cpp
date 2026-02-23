//
// ... Standard header files
//
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/reordering.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  // Build a column-oriented view of the upper triangle from a symmetric
  // CSR pattern. Returns (col_ptr, row_ind) where row_ind[col_ptr[j]]
  // through row_ind[col_ptr[j+1]-1] are the row indices i < j in the
  // upper triangle of column j. Uses counting-sort (two-pass transpose).
  static std::pair<std::vector<size_type>, std::vector<size_type>>
  upper_triangle_by_column(Compressed_row_sparsity const& sym) {
    auto n = sym.shape().row();
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    // Pass 1: count entries per column in the upper triangle
    std::vector<size_type> count(static_cast<std::size_t>(n), 0);
    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j > i) { ++count[static_cast<std::size_t>(j)]; }
      }
    }

    // Build col_ptr via exclusive prefix sum of counts
    std::vector<size_type> col_ptr(static_cast<std::size_t>(n + 1), 0);
    for (size_type j = 0; j < n; ++j) {
      col_ptr[static_cast<std::size_t>(j + 1)] =
        col_ptr[static_cast<std::size_t>(j)] +
        count[static_cast<std::size_t>(j)];
    }

    // Pass 2: fill row indices using work copy of col_ptr
    auto nnz_upper = col_ptr[static_cast<std::size_t>(n)];
    std::vector<size_type> row_ind(static_cast<std::size_t>(nnz_upper));
    auto work = col_ptr;

    for (size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto j = ci[p];
        if (j > i) {
          auto pos = work[static_cast<std::size_t>(j)]++;
          row_ind[static_cast<std::size_t>(pos)] = i;
        }
      }
    }

    return {std::move(col_ptr), std::move(row_ind)};
  }

  std::vector<size_type>
  elimination_tree(Compressed_row_sparsity const& sp) {
    if (sp.shape().row() != sp.shape().column()) {
      throw std::invalid_argument("elimination_tree requires a square matrix");
    }

    auto n = sp.shape().row();
    auto sym = symmetrize_pattern(sp);
    auto [col_ptr, row_ind] = upper_triangle_by_column(sym);

    std::vector<size_type> parent(static_cast<std::size_t>(n), -1);
    std::vector<size_type> ancestor(static_cast<std::size_t>(n), -1);

    for (size_type j = 0; j < n; ++j) {
      for (auto p = col_ptr[static_cast<std::size_t>(j)];
           p < col_ptr[static_cast<std::size_t>(j + 1)];
           ++p) {
        auto i = row_ind[static_cast<std::size_t>(p)];

        // Walk from i toward j following ancestor pointers
        auto node = i;
        while (node != -1 && node != j) {
          auto next = ancestor[static_cast<std::size_t>(node)];
          ancestor[static_cast<std::size_t>(node)] = j; // path compression
          if (next == -1) { parent[static_cast<std::size_t>(node)] = j; }
          node = next;
        }
      }
    }

    return parent;
  }

  std::vector<size_type>
  tree_postorder(std::span<size_type const> parent) {
    auto n = static_cast<size_type>(parent.size());

    // Build child linked lists using head/next arrays
    std::vector<size_type> head(static_cast<std::size_t>(n), -1);
    std::vector<size_type> next(static_cast<std::size_t>(n), -1);

    for (size_type j = 0; j < n; ++j) {
      auto p = parent[static_cast<std::size_t>(j)];
      if (p != -1) {
        next[static_cast<std::size_t>(j)] = head[static_cast<std::size_t>(p)];
        head[static_cast<std::size_t>(p)] = j;
      }
    }

    // Iterative DFS postorder
    std::vector<size_type> post;
    post.reserve(static_cast<std::size_t>(n));
    std::vector<size_type> stack;

    for (size_type root = 0; root < n; ++root) {
      if (parent[static_cast<std::size_t>(root)] != -1) { continue; }

      stack.push_back(root);
      while (!stack.empty()) {
        auto node = stack.back();
        auto child = head[static_cast<std::size_t>(node)];

        if (child != -1) {
          // Advance to next child for next visit
          head[static_cast<std::size_t>(node)] =
            next[static_cast<std::size_t>(child)];
          stack.push_back(child);
        } else {
          // All children visited; emit this node
          stack.pop_back();
          post.push_back(node);
        }
      }
    }

    return post;
  }

  std::vector<size_type>
  cholesky_column_counts(
    Compressed_row_sparsity const& sp, std::span<size_type const> parent) {
    auto n = sp.shape().row();
    auto sym = symmetrize_pattern(sp);
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    // counts[j] = number of nonzeros in column j of L (including diagonal)
    std::vector<size_type> counts(static_cast<std::size_t>(n), 1);
    // marker[node] = i means node was already counted for row i
    std::vector<size_type> marker(static_cast<std::size_t>(n), -1);

    for (size_type i = 0; i < n; ++i) {
      // For each lower-triangle entry (i, k) with k < i, walk from k
      // toward the root through the etree, incrementing counts[node]
      // for each new node encountered. This counts L(i, node) != 0.
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto k = ci[p];
        if (k >= i) { continue; }

        auto node = k;
        while (node != -1 && node != i &&
               marker[static_cast<std::size_t>(node)] != i) {
          marker[static_cast<std::size_t>(node)] = i;
          ++counts[static_cast<std::size_t>(node)];
          node = parent[static_cast<std::size_t>(node)];
        }
      }
    }

    return counts;
  }

} // end of namespace sparkit::data::detail
