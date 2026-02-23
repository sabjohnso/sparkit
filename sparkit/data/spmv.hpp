#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/Block_sparse_row_matrix.hpp>
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Diagonal_matrix.hpp>
#include <sparkit/data/Ellpack_matrix.hpp>
#include <sparkit/data/Jagged_diagonal_matrix.hpp>
#include <sparkit/data/Modified_sparse_row_matrix.hpp>
#include <sparkit/data/Symmetric_compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  // ================================================================
  // CSC — column-wise scatter
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Compressed_column_matrix<T> const& A, std::span<T const> x) {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto cp = A.col_ptr();
    auto ri = A.row_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(rows), T{0});
    for (config::size_type j = 0; j < cols; ++j) {
      auto xj = x[j];
      for (auto k = cp[j]; k < cp[j + 1]; ++k) {
        y[static_cast<std::size_t>(ri[k])] += vals[k] * xj;
      }
    }
    return y;
  }

  // ================================================================
  // MSR — diagonal + off-diagonal
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Modified_sparse_row_matrix<T> const& A, std::span<T const> x) {
    auto rows = A.shape().row();
    auto diag = A.diagonal();
    auto diag_len = A.sparsity().diagonal_length();
    auto rp = A.sparsity().off_diagonal_row_ptr();
    auto ci = A.sparsity().off_diagonal_col_ind();
    auto od_vals = A.off_diagonal_values();

    std::vector<T> y(static_cast<std::size_t>(rows), T{0});

    // Diagonal contribution
    for (config::size_type i = 0; i < diag_len; ++i) {
      if (A.sparsity().has_diagonal(i)) {
        y[static_cast<std::size_t>(i)] = diag[i] * x[i];
      }
    }

    // Off-diagonal contribution
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto k = rp[i]; k < rp[i + 1]; ++k) {
        y[static_cast<std::size_t>(i)] += od_vals[k] * x[ci[k]];
      }
    }
    return y;
  }

  // ================================================================
  // DIA — diagonal-by-diagonal (SPARSKIT2 AMUXD)
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Diagonal_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto ncol = A.shape().column();
    auto offsets = A.sparsity().offsets();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(nrow), T{0});

    config::size_type pos = 0;
    for (std::size_t d = 0; d < offsets.size(); ++d) {
      auto off = offsets[d];
      config::size_type diag_len;
      if (off >= 0) {
        diag_len = std::min(nrow, ncol - off);
      } else {
        diag_len = std::min(nrow + off, ncol);
      }

      for (config::size_type k = 0; k < diag_len; ++k) {
        auto row = (off >= 0) ? k : k - off;
        auto col = (off >= 0) ? k + off : k;
        y[static_cast<std::size_t>(row)] += vals[pos + k] * x[col];
      }
      pos += diag_len;
    }
    return y;
  }

  // ================================================================
  // ELL — row-by-row padded (SPARSKIT2 AMUXE)
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Ellpack_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto max_nnz = A.sparsity().max_nnz_per_row();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(nrow), T{0});
    if (max_nnz == 0) return y;

    for (config::size_type i = 0; i < nrow; ++i) {
      T sum{0};
      auto base = static_cast<std::size_t>(i * max_nnz);
      for (config::size_type k = 0; k < max_nnz; ++k) {
        auto idx = base + static_cast<std::size_t>(k);
        auto col = ci[idx];
        if (col == -1) break;
        sum += vals[idx] * x[col];
      }
      y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
  }

  // ================================================================
  // BSR — block-level CSR with dense block multiply
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Block_sparse_row_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto br = A.sparsity().block_rows();
    auto bc = A.sparsity().block_cols();
    auto rp = A.sparsity().row_ptr();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();
    auto num_block_rows = nrow / br;

    std::vector<T> y(static_cast<std::size_t>(nrow), T{0});

    for (config::size_type I = 0; I < num_block_rows; ++I) {
      for (auto bj = rp[I]; bj < rp[I + 1]; ++bj) {
        auto J = ci[bj];
        auto block_base = static_cast<std::size_t>(bj * br * bc);
        for (config::size_type r = 0; r < br; ++r) {
          T sum{0};
          for (config::size_type c = 0; c < bc; ++c) {
            sum += vals[block_base + static_cast<std::size_t>(r * bc + c)] *
                   x[J * bc + c];
          }
          y[static_cast<std::size_t>(I * br + r)] += sum;
        }
      }
    }
    return y;
  }

  // ================================================================
  // JAD — jagged diagonal with permutation (SPARSKIT2 AMUXJ)
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Jagged_diagonal_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto pm = A.sparsity().perm();
    auto jd = A.sparsity().jdiag();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();
    auto num_jdiags = std::ssize(jd) - 1;

    std::vector<T> y(static_cast<std::size_t>(nrow), T{0});

    for (config::size_type k = 0; k < num_jdiags; ++k) {
      auto width = jd[k + 1] - jd[k];
      for (config::size_type i = 0; i < width; ++i) {
        auto idx = static_cast<std::size_t>(jd[k] + i);
        y[static_cast<std::size_t>(pm[i])] += vals[idx] * x[ci[idx]];
      }
    }
    return y;
  }

  // ================================================================
  // sCSR — symmetric lower-triangle with scatter
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply(Symmetric_compressed_row_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(nrow), T{0});

    for (config::size_type i = 0; i < nrow; ++i) {
      for (auto k = rp[i]; k < rp[i + 1]; ++k) {
        auto j = ci[k];
        auto v = vals[k];
        if (j == i) {
          y[static_cast<std::size_t>(i)] += v * x[i];
        } else {
          // Lower triangle: j < i
          y[static_cast<std::size_t>(i)] += v * x[j];
          y[static_cast<std::size_t>(j)] += v * x[i];
        }
      }
    }
    return y;
  }

  // ================================================================
  // CSC — transpose: column-gather (dual of column-scatter)
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(
    Compressed_column_matrix<T> const& A, std::span<T const> x) {
    auto cols = A.shape().column();
    auto cp = A.col_ptr();
    auto ri = A.row_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(cols), T{0});
    for (config::size_type j = 0; j < cols; ++j) {
      T sum{0};
      for (auto k = cp[j]; k < cp[j + 1]; ++k) {
        sum += vals[k] * x[ri[k]];
      }
      y[static_cast<std::size_t>(j)] = sum;
    }
    return y;
  }

  // ================================================================
  // MSR — transpose: diagonal + off-diagonal scatter
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(
    Modified_sparse_row_matrix<T> const& A, std::span<T const> x) {
    auto cols = A.shape().column();
    auto diag = A.diagonal();
    auto diag_len = A.sparsity().diagonal_length();
    auto rp = A.sparsity().off_diagonal_row_ptr();
    auto ci = A.sparsity().off_diagonal_col_ind();
    auto od_vals = A.off_diagonal_values();
    auto rows = A.shape().row();

    std::vector<T> y(static_cast<std::size_t>(cols), T{0});

    // Diagonal contribution (symmetric under transpose)
    for (config::size_type i = 0; i < diag_len; ++i) {
      if (A.sparsity().has_diagonal(i)) {
        y[static_cast<std::size_t>(i)] = diag[i] * x[i];
      }
    }

    // Off-diagonal: scatter into y[col] instead of y[row]
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto k = rp[i]; k < rp[i + 1]; ++k) {
        y[static_cast<std::size_t>(ci[k])] += od_vals[k] * x[i];
      }
    }
    return y;
  }

  // ================================================================
  // DIA — transpose: swap row/col roles
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(Diagonal_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto ncol = A.shape().column();
    auto offsets = A.sparsity().offsets();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(ncol), T{0});

    config::size_type pos = 0;
    for (std::size_t d = 0; d < offsets.size(); ++d) {
      auto off = offsets[d];
      config::size_type diag_len;
      if (off >= 0) {
        diag_len = std::min(nrow, ncol - off);
      } else {
        diag_len = std::min(nrow + off, ncol);
      }

      for (config::size_type k = 0; k < diag_len; ++k) {
        auto row = (off >= 0) ? k : k - off;
        auto col = (off >= 0) ? k + off : k;
        y[static_cast<std::size_t>(col)] += vals[pos + k] * x[row];
      }
      pos += diag_len;
    }
    return y;
  }

  // ================================================================
  // ELL — transpose: row-scatter (dual of row-gather)
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(Ellpack_matrix<T> const& A, std::span<T const> x) {
    auto nrow = A.shape().row();
    auto ncol = A.shape().column();
    auto max_nnz = A.sparsity().max_nnz_per_row();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(ncol), T{0});
    if (max_nnz == 0) return y;

    for (config::size_type i = 0; i < nrow; ++i) {
      auto base = static_cast<std::size_t>(i * max_nnz);
      for (config::size_type k = 0; k < max_nnz; ++k) {
        auto idx = base + static_cast<std::size_t>(k);
        auto col = ci[idx];
        if (col == -1) break;
        y[static_cast<std::size_t>(col)] += vals[idx] * x[i];
      }
    }
    return y;
  }

  // ================================================================
  // BSR — transpose: transpose the dense block contribution
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(
    Block_sparse_row_matrix<T> const& A, std::span<T const> x) {
    auto ncol = A.shape().column();
    auto nrow = A.shape().row();
    auto br = A.sparsity().block_rows();
    auto bc = A.sparsity().block_cols();
    auto rp = A.sparsity().row_ptr();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();
    auto num_block_rows = nrow / br;

    std::vector<T> y(static_cast<std::size_t>(ncol), T{0});

    for (config::size_type I = 0; I < num_block_rows; ++I) {
      for (auto bj = rp[I]; bj < rp[I + 1]; ++bj) {
        auto J = ci[bj];
        auto block_base = static_cast<std::size_t>(bj * br * bc);
        for (config::size_type r = 0; r < br; ++r) {
          for (config::size_type c = 0; c < bc; ++c) {
            y[static_cast<std::size_t>(J * bc + c)] +=
              vals[block_base + static_cast<std::size_t>(r * bc + c)] *
              x[I * br + r];
          }
        }
      }
    }
    return y;
  }

  // ================================================================
  // JAD — transpose: gather from x[perm], scatter into y[col]
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(Jagged_diagonal_matrix<T> const& A, std::span<T const> x) {
    auto ncol = A.shape().column();
    auto pm = A.sparsity().perm();
    auto jd = A.sparsity().jdiag();
    auto ci = A.sparsity().col_ind();
    auto vals = A.values();
    auto num_jdiags = std::ssize(jd) - 1;

    std::vector<T> y(static_cast<std::size_t>(ncol), T{0});

    for (config::size_type k = 0; k < num_jdiags; ++k) {
      auto width = jd[k + 1] - jd[k];
      for (config::size_type i = 0; i < width; ++i) {
        auto idx = static_cast<std::size_t>(jd[k] + i);
        y[static_cast<std::size_t>(ci[idx])] += vals[idx] * x[pm[i]];
      }
    }
    return y;
  }

  // ================================================================
  // sCSR — transpose: A^T = A for symmetric matrices
  // ================================================================

  template <typename T>
  std::vector<T>
  multiply_transpose(
    Symmetric_compressed_row_matrix<T> const& A, std::span<T const> x) {
    return multiply(A, x);
  }

} // end of namespace sparkit::data::detail
