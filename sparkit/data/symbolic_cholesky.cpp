//
// ... Standard header files
//
#include <algorithm>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/reordering.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::data::detail {

  using size_type = config::size_type;

  Compressed_row_sparsity
  symbolic_cholesky(Compressed_row_sparsity const& sp) {
    if (sp.shape().row() != sp.shape().column()) {
      throw std::invalid_argument("symbolic_cholesky requires a square matrix");
    }

    auto sym = symmetrize_pattern(sp);
    auto n = sym.shape().row();
    auto rp = sym.row_ptr();
    auto ci = sym.col_ind();

    auto parent = elimination_tree(sp);

    // marker[node] == i means node was already visited for row i
    std::vector<size_type> marker(static_cast<std::size_t>(n), -1);

    // Accumulate all indices for L row by row
    std::vector<Index> indices;
    std::vector<size_type> row_cols;

    for (size_type i = 0; i < n; ++i) {
      marker[static_cast<std::size_t>(i)] = i;
      row_cols.clear();

      // Walk etree paths from each lower-triangle neighbor k < i
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        auto k = ci[p];
        if (k >= i) { continue; }

        auto node = k;
        while (node != -1 && node != i &&
               marker[static_cast<std::size_t>(node)] != i) {
          marker[static_cast<std::size_t>(node)] = i;
          row_cols.push_back(node);
          node = parent[static_cast<std::size_t>(node)];
        }
      }

      std::sort(row_cols.begin(), row_cols.end());

      for (auto j : row_cols) {
        indices.push_back(Index{i, j});
      }
      indices.push_back(Index{i, i}); // diagonal
    }

    return Compressed_row_sparsity{Shape{n, n}, indices.begin(), indices.end()};
  }

} // end of namespace sparkit::data::detail
