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
#include <sparkit/data/Compressed_row_matrix.hpp>

namespace sparkit::data::detail {

  template<typename T>
  std::vector<T>
  multiply(Compressed_row_matrix<T> const& A, std::span<T const> x)
  {
    auto rows = A.shape().row();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(rows), T{0});
    for (config::size_type i = 0; i < rows; ++i) {
      T sum{0};
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        sum += vals[j] * x[ci[j]];
      }
      y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
  }

  template<typename T>
  std::vector<T>
  multiply_transpose(Compressed_row_matrix<T> const& A, std::span<T const> x)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> y(static_cast<std::size_t>(cols), T{0});
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        y[static_cast<std::size_t>(ci[j])] += vals[j] * x[i];
      }
    }
    return y;
  }

  template<typename T>
  Compressed_row_matrix<T>
  multiply_left_diagonal(
    Compressed_row_matrix<T> const& A,
    std::span<T const> d)
  {
    auto rp = A.row_ptr();
    auto vals = A.values();
    auto rows = A.shape().row();

    std::vector<T> new_vals(vals.begin(), vals.end());
    for (config::size_type i = 0; i < rows; ++i) {
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        new_vals[static_cast<std::size_t>(j)] *= d[i];
      }
    }
    return Compressed_row_matrix<T>{A.sparsity(), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  multiply_right_diagonal(
    Compressed_row_matrix<T> const& A,
    std::span<T const> d)
  {
    auto ci = A.col_ind();
    auto vals = A.values();

    std::vector<T> new_vals(vals.begin(), vals.end());
    for (std::size_t j = 0; j < new_vals.size(); ++j) {
      new_vals[j] *= d[ci[static_cast<config::size_type>(j)]];
    }
    return Compressed_row_matrix<T>{A.sparsity(), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  add_diagonal(
    Compressed_row_matrix<T> const& A,
    std::span<T const> d)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vals = A.values();
    auto diag_len = std::min(rows, cols);

    std::vector<Index> new_indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      bool diag_added = false;
      for (auto j = rp[i]; j < rp[i + 1]; ++j) {
        if (!diag_added && i < diag_len && ci[j] >= i) {
          if (ci[j] == i) {
            new_indices.push_back(Index{i, i});
            new_vals.push_back(vals[j] + d[i]);
            diag_added = true;
            continue;
          }
          new_indices.push_back(Index{i, i});
          new_vals.push_back(d[i]);
          diag_added = true;
        }
        new_indices.push_back(Index{i, ci[j]});
        new_vals.push_back(vals[j]);
      }
      if (!diag_added && i < diag_len) {
        new_indices.push_back(Index{i, i});
        new_vals.push_back(d[i]);
      }
    }

    Compressed_row_sparsity sparsity{
      A.shape(), new_indices.begin(), new_indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  add(
    Compressed_row_matrix<T> const& A,
    T s,
    Compressed_row_matrix<T> const& B)
  {
    auto rows = A.shape().row();
    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();
    auto b_rp = B.row_ptr();
    auto b_ci = B.col_ind();
    auto b_vals = B.values();

    std::vector<Index> new_indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      auto a = a_rp[i];
      auto a_end = a_rp[i + 1];
      auto b = b_rp[i];
      auto b_end = b_rp[i + 1];

      while (a < a_end && b < b_end) {
        if (a_ci[a] < b_ci[b]) {
          new_indices.push_back(Index{i, a_ci[a]});
          new_vals.push_back(a_vals[a]);
          ++a;
        } else if (b_ci[b] < a_ci[a]) {
          new_indices.push_back(Index{i, b_ci[b]});
          new_vals.push_back(s * b_vals[b]);
          ++b;
        } else {
          new_indices.push_back(Index{i, a_ci[a]});
          new_vals.push_back(a_vals[a] + s * b_vals[b]);
          ++a;
          ++b;
        }
      }
      while (a < a_end) {
        new_indices.push_back(Index{i, a_ci[a]});
        new_vals.push_back(a_vals[a]);
        ++a;
      }
      while (b < b_end) {
        new_indices.push_back(Index{i, b_ci[b]});
        new_vals.push_back(s * b_vals[b]);
        ++b;
      }
    }

    Compressed_row_sparsity sparsity{
      A.shape(), new_indices.begin(), new_indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  add(
    Compressed_row_matrix<T> const& A,
    Compressed_row_matrix<T> const& B)
  {
    return add(A, T{1}, B);
  }

  // C = A + s * B^T (SPARSKIT2 APLSBT)
  //
  // Fused transpose-add that avoids materializing the full transpose.
  // Phase 1 builds a CSC view of B via counting sort — column i of CSC(B)
  // equals row i of B^T.  Phase 2 merges each row of A with the
  // corresponding column of CSC(B) using a sorted two-pointer merge,
  // identical to the merge in add().
  //
  // Total cost: O(nnz(A) + nnz(B) + rows + cols).

  template<typename T>
  Compressed_row_matrix<T>
  add_transpose(
    Compressed_row_matrix<T> const& A,
    T s,
    Compressed_row_matrix<T> const& B)
  {
    auto rows = A.shape().row();
    auto cols = A.shape().column();
    auto b_rows = B.shape().row();
    auto b_cols = B.shape().column();

    // Phase 1: Build CSC representation of B via counting sort.
    // CSC column i of B = row i of B^T.
    auto b_nnz = static_cast<std::size_t>(B.size());
    auto b_rp = B.row_ptr();
    auto b_ci = B.col_ind();
    auto b_vals = B.values();

    // Count entries per column of B
    std::vector<config::size_type> col_ptr(
      static_cast<std::size_t>(b_cols + 1), config::size_type{0});
    for (std::size_t k = 0; k < b_nnz; ++k) {
      ++col_ptr[static_cast<std::size_t>(b_ci[static_cast<config::size_type>(k)] + 1)];
    }

    // Prefix sum
    for (config::size_type j = 0; j < b_cols; ++j) {
      col_ptr[static_cast<std::size_t>(j + 1)]
        += col_ptr[static_cast<std::size_t>(j)];
    }

    // Place entries into CSC arrays
    std::vector<config::size_type> row_ind(b_nnz);
    std::vector<T> csc_vals(b_nnz);
    // Work copy of col_ptr for placement
    std::vector<config::size_type> work(col_ptr.begin(), col_ptr.end());

    for (config::size_type i = 0; i < b_rows; ++i) {
      for (auto j = b_rp[i]; j < b_rp[i + 1]; ++j) {
        auto col = b_ci[j];
        auto pos = static_cast<std::size_t>(work[static_cast<std::size_t>(col)]);
        row_ind[pos] = i;
        csc_vals[pos] = b_vals[j];
        ++work[static_cast<std::size_t>(col)];
      }
    }

    // Phase 2: Merge row-by-row.
    // Row i of A: a_ci[a_rp[i]..a_rp[i+1]]
    // Row i of B^T = column i of CSC(B): row_ind[col_ptr[i]..col_ptr[i+1]]
    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();

    std::vector<Index> new_indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < rows; ++i) {
      auto a = a_rp[i];
      auto a_end = a_rp[i + 1];

      // Column i of CSC(B) — only valid if i < b_cols
      auto bt_begin = (i < b_cols)
        ? col_ptr[static_cast<std::size_t>(i)]
        : config::size_type{0};
      auto bt_end = (i < b_cols)
        ? col_ptr[static_cast<std::size_t>(i + 1)]
        : config::size_type{0};
      auto bt = bt_begin;

      while (a < a_end && bt < bt_end) {
        auto a_col = a_ci[a];
        auto bt_col = row_ind[static_cast<std::size_t>(bt)];
        if (a_col < bt_col) {
          new_indices.push_back(Index{i, a_col});
          new_vals.push_back(a_vals[a]);
          ++a;
        } else if (bt_col < a_col) {
          new_indices.push_back(Index{i, bt_col});
          new_vals.push_back(s * csc_vals[static_cast<std::size_t>(bt)]);
          ++bt;
        } else {
          new_indices.push_back(Index{i, a_col});
          new_vals.push_back(
            a_vals[a] + s * csc_vals[static_cast<std::size_t>(bt)]);
          ++a;
          ++bt;
        }
      }
      while (a < a_end) {
        new_indices.push_back(Index{i, a_ci[a]});
        new_vals.push_back(a_vals[a]);
        ++a;
      }
      while (bt < bt_end) {
        new_indices.push_back(
          Index{i, row_ind[static_cast<std::size_t>(bt)]});
        new_vals.push_back(s * csc_vals[static_cast<std::size_t>(bt)]);
        ++bt;
      }
    }

    Compressed_row_sparsity sparsity{
      Shape{rows, cols}, new_indices.begin(), new_indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  add_transpose(
    Compressed_row_matrix<T> const& A,
    Compressed_row_matrix<T> const& B)
  {
    return add_transpose(A, T{1}, B);
  }

  template<typename T>
  Compressed_row_matrix<T>
  multiply(
    Compressed_row_matrix<T> const& A,
    Compressed_row_matrix<T> const& B)
  {
    auto a_rows = A.shape().row();
    auto b_cols = B.shape().column();
    auto a_rp = A.row_ptr();
    auto a_ci = A.col_ind();
    auto a_vals = A.values();
    auto b_rp = B.row_ptr();
    auto b_ci = B.col_ind();
    auto b_vals = B.values();

    auto n = static_cast<std::size_t>(b_cols);
    std::vector<T> w(n, T{0});
    std::vector<bool> occupied(n, false);

    std::vector<Index> new_indices;
    std::vector<T> new_vals;

    for (config::size_type i = 0; i < a_rows; ++i) {
      std::vector<config::size_type> col_list;

      for (auto ja = a_rp[i]; ja < a_rp[i + 1]; ++ja) {
        auto k = a_ci[ja];
        auto a_ik = a_vals[ja];
        for (auto jb = b_rp[k]; jb < b_rp[k + 1]; ++jb) {
          auto col = static_cast<std::size_t>(b_ci[jb]);
          if (!occupied[col]) {
            occupied[col] = true;
            col_list.push_back(b_ci[jb]);
          }
          w[col] += a_ik * b_vals[jb];
        }
      }

      std::sort(col_list.begin(), col_list.end());

      for (auto col : col_list) {
        auto uc = static_cast<std::size_t>(col);
        new_indices.push_back(Index{i, col});
        new_vals.push_back(w[uc]);
        w[uc] = T{0};
        occupied[uc] = false;
      }
    }

    Compressed_row_sparsity sparsity{
      Shape{a_rows, b_cols}, new_indices.begin(), new_indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(new_vals)};
  }

} // end of namespace sparkit::data::detail
