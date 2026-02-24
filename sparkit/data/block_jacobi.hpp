#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // Precomputed LU-factorized diagonal blocks for Block Jacobi.

  template <typename T>
  struct Block_jacobi_factors {
    config::size_type block_size;
    config::size_type n;
    config::size_type num_blocks;
    std::vector<T> lu_blocks;
    std::vector<config::size_type> pivots;
  };

  // Dense LU factorization with partial pivoting.
  //
  // Operates in-place on a row-major bs x bs block.
  // Returns pivots. Throws std::invalid_argument if singular.

  template <typename T>
  static void
  dense_lu(T* block, config::size_type* piv, config::size_type bs) {
    for (config::size_type i = 0; i < bs; ++i) {
      piv[i] = i;
    }

    for (config::size_type k = 0; k < bs; ++k) {
      // Find pivot
      config::size_type max_row = k;
      T max_val = std::abs(block[k * bs + k]);
      for (config::size_type i = k + 1; i < bs; ++i) {
        T val = std::abs(block[i * bs + k]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }

      if (max_val == T{0}) {
        throw std::invalid_argument("block_jacobi: singular diagonal block");
      }

      // Swap rows
      if (max_row != k) {
        std::swap(piv[k], piv[max_row]);
        for (config::size_type j = 0; j < bs; ++j) {
          std::swap(block[k * bs + j], block[max_row * bs + j]);
        }
      }

      // Eliminate
      T pivot = block[k * bs + k];
      for (config::size_type i = k + 1; i < bs; ++i) {
        T factor = block[i * bs + k] / pivot;
        block[i * bs + k] = factor;
        for (config::size_type j = k + 1; j < bs; ++j) {
          block[i * bs + j] -= factor * block[k * bs + j];
        }
      }
    }
  }

  // Solve LU system with pre-computed pivots.
  //
  // Solves Ax = b in-place where A is stored as an LU factorization.
  // The result is written to x (which initially contains b permuted by pivots).

  template <typename T>
  static void
  dense_lu_solve(
    T const* lu,
    config::size_type const* piv,
    config::size_type bs,
    T* x,
    T const* rhs) {
    // Apply permutation
    for (config::size_type i = 0; i < bs; ++i) {
      x[i] = rhs[piv[i]];
    }

    // Forward solve (L, unit diagonal)
    for (config::size_type i = 1; i < bs; ++i) {
      for (config::size_type j = 0; j < i; ++j) {
        x[i] -= lu[i * bs + j] * x[j];
      }
    }

    // Backward solve (U)
    for (config::size_type i = bs - 1; i >= 0; --i) {
      for (config::size_type j = i + 1; j < bs; ++j) {
        x[i] -= lu[i * bs + j] * x[j];
      }
      x[i] /= lu[i * bs + i];
      if (i == 0) { break; }
    }
  }

  // Block Jacobi preconditioner setup.
  //
  // Extracts diagonal blocks of size block_size from A and factorizes
  // each with dense LU + partial pivoting. The last block may be smaller
  // if n is not divisible by block_size.
  //
  // Throws std::invalid_argument if any diagonal block is singular.

  template <typename T>
  Block_jacobi_factors<T>
  block_jacobi(
    Compressed_row_matrix<T> const& A, config::size_type block_size) {
    auto n = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    auto num_blocks = (n + block_size - 1) / block_size;

    std::vector<T> lu_blocks(
      static_cast<std::size_t>(num_blocks * block_size * block_size), T{0});
    std::vector<config::size_type> pivots(
      static_cast<std::size_t>(num_blocks * block_size), 0);

    for (config::size_type k = 0; k < num_blocks; ++k) {
      auto row_start = k * block_size;
      auto bs_k = std::min(block_size, n - row_start);

      auto block_offset = static_cast<std::size_t>(k * block_size * block_size);

      // Extract dense block from CSR
      for (config::size_type i = 0; i < bs_k; ++i) {
        auto global_row = row_start + i;
        for (auto p = rp[global_row]; p < rp[global_row + 1]; ++p) {
          auto col = ci[p];
          auto local_col = col - row_start;
          if (local_col >= 0 && local_col < bs_k) {
            lu_blocks
              [block_offset +
               static_cast<std::size_t>(i * block_size + local_col)] = vals[p];
          }
        }
      }

      // Factorize
      dense_lu(
        lu_blocks.data() + block_offset,
        pivots.data() + static_cast<std::size_t>(k * block_size),
        bs_k);
    }

    return Block_jacobi_factors<T>{
      block_size, n, num_blocks, std::move(lu_blocks), std::move(pivots)};
  }

  // Block Jacobi preconditioner apply.
  //
  // For each block, solves the dense LU system to compute z = M^{-1} r.

  template <typename T, typename Iter, typename OutIter>
  void
  block_jacobi_apply(
    Block_jacobi_factors<T> const& factors,
    Iter first,
    Iter last,
    OutIter out) {
    auto n = factors.n;
    auto bs = factors.block_size;
    auto num_blocks = factors.num_blocks;

    // Copy input to a contiguous buffer
    std::vector<T> rhs(first, last);
    std::vector<T> result(static_cast<std::size_t>(n), T{0});

    for (config::size_type k = 0; k < num_blocks; ++k) {
      auto row_start = k * bs;
      auto bs_k = std::min(bs, n - row_start);

      auto block_offset = static_cast<std::size_t>(k * bs * bs);
      auto piv_offset = static_cast<std::size_t>(k * bs);

      dense_lu_solve(
        factors.lu_blocks.data() + block_offset,
        factors.pivots.data() + piv_offset,
        bs_k,
        result.data() + static_cast<std::size_t>(row_start),
        rhs.data() + static_cast<std::size_t>(row_start));
    }

    std::copy(result.begin(), result.end(), out);
  }

} // end of namespace sparkit::data::detail
