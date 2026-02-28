#pragma once

//
// ... Standard header files
//
#include <cmath>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // Strength of connection for algebraic multigrid.
  //
  // Entry (i,j) is strong if |a_ij| >= theta * max_{k != i} |a_ik|.
  // Diagonal entries are excluded. Returns a symmetric sparsity pattern
  // containing all strong connections (both (i,j) and (j,i)).

  template <typename T>
  Compressed_row_sparsity
  strength_of_connection(Compressed_row_matrix<T> const& A, T theta) {
    auto n = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    // Phase 1: find max off-diagonal magnitude per row.
    std::vector<T> max_off_diag(static_cast<std::size_t>(n), T{0});
    for (config::size_type i = 0; i < n; ++i) {
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] != i) {
          auto mag = std::abs(vals[p]);
          if (mag > max_off_diag[static_cast<std::size_t>(i)]) {
            max_off_diag[static_cast<std::size_t>(i)] = mag;
          }
        }
      }
    }

    // Phase 2: collect strong connections (asymmetric).
    std::vector<std::vector<config::size_type>> strong(
      static_cast<std::size_t>(n));

    for (config::size_type i = 0; i < n; ++i) {
      auto threshold = theta * max_off_diag[static_cast<std::size_t>(i)];
      for (auto p = rp[i]; p < rp[i + 1]; ++p) {
        if (ci[p] != i && std::abs(vals[p]) >= threshold) {
          strong[static_cast<std::size_t>(i)].push_back(ci[p]);
        }
      }
    }

    // Phase 3: symmetrize — if (i,j) is strong, ensure (j,i) is too.
    for (config::size_type i = 0; i < n; ++i) {
      for (auto j : strong[static_cast<std::size_t>(i)]) {
        auto& sj = strong[static_cast<std::size_t>(j)];
        bool found = false;
        for (auto k : sj) {
          if (k == i) {
            found = true;
            break;
          }
        }
        if (!found) { sj.push_back(i); }
      }
    }

    // Phase 4: build sparsity pattern from strong lists.
    std::vector<Index> indices;
    for (config::size_type i = 0; i < n; ++i) {
      auto& si = strong[static_cast<std::size_t>(i)];
      std::sort(si.begin(), si.end());
      for (auto j : si) {
        indices.push_back(Index{i, j});
      }
    }

    return Compressed_row_sparsity{Shape{n, n}, indices.begin(), indices.end()};
  }

} // end of namespace sparkit::data::detail
