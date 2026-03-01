#pragma once

//
// ... Standard header files
//
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/multifrontal_numeric.hpp>

namespace sparkit::data::detail {

  // Forward solve: L * y = b, postorder traversal.
  // For each supernode in postorder:
  //   1. Dense forward solve within L_diag
  //   2. Scatter L_sub * y_s into update rows
  template <typename T>
  std::vector<T>
  multifrontal_forward_solve(
    Multifrontal_factor<T> const& factor, std::span<T const> b) {
    auto n = factor.symbolic.n;
    auto ns = factor.symbolic.partition.n_supernodes;
    auto const& postorder = factor.symbolic.tree.postorder;

    std::vector<T> y(b.begin(), b.end());

    for (config::size_type step = 0; step < ns; ++step) {
      auto s = postorder[static_cast<std::size_t>(step)];
      auto const& map = factor.symbolic.maps[static_cast<std::size_t>(s)];
      auto const& fac = factor.factors[static_cast<std::size_t>(s)];
      auto snode_size = fac.snode_size;
      auto update_size = fac.update_size;

      // Dense forward solve on diagonal block: L_diag * y_s = b_s
      for (config::size_type j = 0; j < snode_size; ++j) {
        auto global_j = map.row_indices[static_cast<std::size_t>(j)];
        for (config::size_type k = 0; k < j; ++k) {
          auto global_k = map.row_indices[static_cast<std::size_t>(k)];
          y[static_cast<std::size_t>(global_j)] -=
            fac.L_diag[static_cast<std::size_t>(k * snode_size + j)] *
            y[static_cast<std::size_t>(global_k)];
        }
        y[static_cast<std::size_t>(global_j)] /=
          fac.L_diag[static_cast<std::size_t>(j * snode_size + j)];
      }

      // Scatter: y_update -= L_sub * y_s
      for (config::size_type i = 0; i < update_size; ++i) {
        auto global_i =
          map.row_indices[static_cast<std::size_t>(snode_size + i)];
        for (config::size_type j = 0; j < snode_size; ++j) {
          auto global_j = map.row_indices[static_cast<std::size_t>(j)];
          y[static_cast<std::size_t>(global_i)] -=
            fac.L_sub[static_cast<std::size_t>(j * update_size + i)] *
            y[static_cast<std::size_t>(global_j)];
        }
      }
    }

    (void)n;
    return y;
  }

  // Backward solve: L^T * x = y, reverse postorder traversal.
  // For each supernode in reverse postorder:
  //   1. Gather L_sub^T * x_update into supernode rows
  //   2. Dense backward solve within L_diag^T
  template <typename T>
  std::vector<T>
  multifrontal_backward_solve(
    Multifrontal_factor<T> const& factor, std::span<T const> y) {
    auto n = factor.symbolic.n;
    auto ns = factor.symbolic.partition.n_supernodes;
    auto const& postorder = factor.symbolic.tree.postorder;

    std::vector<T> x(y.begin(), y.end());

    for (auto step_rev = ns; step_rev > 0; --step_rev) {
      auto s = postorder[static_cast<std::size_t>(step_rev - 1)];
      auto const& map = factor.symbolic.maps[static_cast<std::size_t>(s)];
      auto const& fac = factor.factors[static_cast<std::size_t>(s)];
      auto snode_size = fac.snode_size;
      auto update_size = fac.update_size;

      // Gather: x_s -= L_sub^T * x_update
      for (config::size_type j = 0; j < snode_size; ++j) {
        auto global_j = map.row_indices[static_cast<std::size_t>(j)];
        for (config::size_type i = 0; i < update_size; ++i) {
          auto global_i =
            map.row_indices[static_cast<std::size_t>(snode_size + i)];
          x[static_cast<std::size_t>(global_j)] -=
            fac.L_sub[static_cast<std::size_t>(j * update_size + i)] *
            x[static_cast<std::size_t>(global_i)];
        }
      }

      // Dense backward solve on diagonal block: L_diag^T * x_s = rhs_s
      for (auto jj = snode_size; jj > 0; --jj) {
        auto j = jj - 1;
        auto global_j = map.row_indices[static_cast<std::size_t>(j)];
        x[static_cast<std::size_t>(global_j)] /=
          fac.L_diag[static_cast<std::size_t>(j * snode_size + j)];

        for (config::size_type k = 0; k < j; ++k) {
          auto global_k = map.row_indices[static_cast<std::size_t>(k)];
          x[static_cast<std::size_t>(global_k)] -=
            fac.L_diag[static_cast<std::size_t>(k * snode_size + j)] *
            x[static_cast<std::size_t>(global_j)];
        }
      }
    }

    (void)n;
    return x;
  }

} // end of namespace sparkit::data::detail
