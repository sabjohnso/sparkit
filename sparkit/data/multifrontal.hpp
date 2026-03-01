#pragma once

//
// ... Standard header files
//
#include <span>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/assembly_tree.hpp>
#include <sparkit/data/elimination_tree.hpp>
#include <sparkit/data/multifrontal_numeric.hpp>
#include <sparkit/data/multifrontal_solve.hpp>
#include <sparkit/data/multifrontal_symbolic.hpp>
#include <sparkit/data/permutation.hpp>
#include <sparkit/data/reordering.hpp>
#include <sparkit/data/supernode.hpp>
#include <sparkit/data/symbolic_cholesky.hpp>

namespace sparkit::data::detail {

  // Convenience wrapper: multifrontal Cholesky factorization.
  // Optional AMD reordering for fill reduction.
  template <typename T>
  Multifrontal_factor<T>
  multifrontal_cholesky(
    Compressed_row_matrix<T> const& A, bool apply_amd = true) {
    if (A.shape().row() != A.shape().column()) {
      throw std::invalid_argument(
        "multifrontal_cholesky requires a square matrix");
    }

    // Optionally apply AMD reordering
    std::vector<config::size_type> perm;
    std::vector<config::size_type> inv_perm;

    if (apply_amd) {
      perm = approximate_minimum_degree(A.sparsity());
      inv_perm = inverse_permutation(perm);
      auto A_perm = dperm(A, perm);

      auto parent = elimination_tree(A_perm.sparsity());
      auto L_pattern = symbolic_cholesky(A_perm.sparsity());
      auto partition = find_supernodes(L_pattern, parent);
      auto tree = build_assembly_tree(partition, parent);
      auto symbolic = multifrontal_analyze(L_pattern, partition, tree);
      auto factors = multifrontal_factorize(A_perm, symbolic);

      return Multifrontal_factor<T>{
        std::move(symbolic),
        std::move(factors),
        std::move(perm),
        std::move(inv_perm)};
    }

    // No reordering
    auto parent = elimination_tree(A.sparsity());
    auto L_pattern = symbolic_cholesky(A.sparsity());
    auto partition = find_supernodes(L_pattern, parent);
    auto tree = build_assembly_tree(partition, parent);
    auto symbolic = multifrontal_analyze(L_pattern, partition, tree);
    auto factors = multifrontal_factorize(A, symbolic);

    return Multifrontal_factor<T>{
      std::move(symbolic),
      std::move(factors),
      std::move(perm),
      std::move(inv_perm)};
  }

  // Convenience wrapper: solve A*x = b using a precomputed factorization.
  // Handles permutation if AMD was applied.
  template <typename T>
  std::vector<T>
  multifrontal_solve(
    Multifrontal_factor<T> const& factor, std::span<T const> b) {
    auto n = factor.symbolic.n;
    bool has_perm = !factor.perm.empty();

    // Apply permutation to b if needed
    std::vector<T> b_work;
    if (has_perm) {
      b_work.resize(static_cast<std::size_t>(n));
      for (config::size_type i = 0; i < n; ++i) {
        b_work[static_cast<std::size_t>(
          factor.perm[static_cast<std::size_t>(i)])] =
          b[static_cast<std::size_t>(i)];
      }
    } else {
      b_work.assign(b.begin(), b.end());
    }

    // Forward and backward solve
    auto y = multifrontal_forward_solve(factor, std::span<T const>{b_work});
    auto x_perm = multifrontal_backward_solve(factor, std::span<T const>{y});

    // Apply inverse permutation if needed
    if (has_perm) {
      std::vector<T> x(static_cast<std::size_t>(n));
      for (config::size_type i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = x_perm[static_cast<std::size_t>(
          factor.perm[static_cast<std::size_t>(i)])];
      }
      return x;
    }

    return x_perm;
  }

} // end of namespace sparkit::data::detail
