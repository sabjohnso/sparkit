#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <utility>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/data/Coordinate_matrix.hpp>
#include <sparkit/data/Coordinate_sparsity.hpp>
#include <sparkit/data/Compressed_column_matrix.hpp>
#include <sparkit/data/Compressed_column_sparsity.hpp>
#include <sparkit/data/Compressed_row_matrix.hpp>
#include <sparkit/data/Compressed_row_sparsity.hpp>
#include <sparkit/data/Modified_sparse_row_matrix.hpp>
#include <sparkit/data/Modified_sparse_row_sparsity.hpp>
#include <sparkit/data/Diagonal_matrix.hpp>
#include <sparkit/data/Diagonal_sparsity.hpp>
#include <sparkit/data/Ellpack_matrix.hpp>
#include <sparkit/data/Ellpack_sparsity.hpp>
#include <sparkit/data/Block_sparse_row_matrix.hpp>
#include <sparkit/data/Block_sparse_row_sparsity.hpp>
#include <sparkit/data/Jagged_diagonal_matrix.hpp>
#include <sparkit/data/Jagged_diagonal_sparsity.hpp>
#include <sparkit/data/Symmetric_compressed_row_matrix.hpp>
#include <sparkit/data/Symmetric_compressed_row_sparsity.hpp>
#include <sparkit/data/Symmetric_coordinate_matrix.hpp>
#include <sparkit/data/Symmetric_coordinate_sparsity.hpp>
#include <sparkit/data/Symmetric_block_sparse_row_matrix.hpp>
#include <sparkit/data/Symmetric_block_sparse_row_sparsity.hpp>

namespace sparkit::data::detail {

  // -- COO to CSR --

  Compressed_row_sparsity
  to_compressed_row(Coordinate_sparsity const& coo);

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Coordinate_matrix<T> const& coo)
  {
    auto entries = coo.entries();

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.first.row() < b.first.row()
        || (a.first.row() == b.first.row()
            && a.first.column() < b.first.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& [index, value] : entries) {
      indices.push_back(index);
      values.push_back(value);
    }

    Compressed_row_sparsity sparsity{
      coo.shape(), indices.begin(), indices.end()};

    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  // -- CSR <-> CSC sparsity --

  Compressed_column_sparsity
  to_compressed_column(Compressed_row_sparsity const& csr);

  Compressed_row_sparsity
  to_compressed_row(Compressed_column_sparsity const& csc);

  // -- CSR <-> CSC matrix --

  template<typename T>
  Compressed_column_matrix<T>
  to_compressed_column(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();

    // Count entries per column
    std::vector<size_type> col_ptr(static_cast<std::size_t>(ncol + 1), 0);
    for (auto c : ci) {
      ++col_ptr[static_cast<std::size_t>(c)];
    }

    // Prefix sum
    size_type running = 0;
    for (std::size_t c = 0; c < col_ptr.size(); ++c) {
      size_type count = col_ptr[c];
      col_ptr[c] = running;
      running += count;
    }

    // Place entries
    std::vector<size_type> row_ind(static_cast<std::size_t>(csr.size()));
    std::vector<T> values(static_cast<std::size_t>(csr.size()));
    std::vector<size_type> work(col_ptr.begin(), col_ptr.end());

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        auto col = ci[j];
        auto dest = work[static_cast<std::size_t>(col)]++;
        row_ind[static_cast<std::size_t>(dest)] = row;
        values[static_cast<std::size_t>(dest)] = sv[j];
      }
    }

    // Build CSC from raw arrays — use indices to go through constructor
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type col = 0; col < ncol; ++col) {
      for (auto j = col_ptr[static_cast<std::size_t>(col)];
           j < col_ptr[static_cast<std::size_t>(col + 1)]; ++j) {
        indices.push_back(Index{row_ind[static_cast<std::size_t>(j)], col});
      }
    }

    Compressed_column_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_column_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Compressed_column_matrix<T> const& csc)
  {
    auto cp = csc.col_ptr();
    auto ri = csc.row_ind();
    auto sv = csc.values();
    auto shape = csc.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();

    // Count entries per row
    std::vector<size_type> row_ptr(static_cast<std::size_t>(nrow + 1), 0);
    for (auto r : ri) {
      ++row_ptr[static_cast<std::size_t>(r)];
    }

    // Prefix sum
    size_type running = 0;
    for (std::size_t r = 0; r < row_ptr.size(); ++r) {
      size_type count = row_ptr[r];
      row_ptr[r] = running;
      running += count;
    }

    // Place entries
    std::vector<size_type> col_ind(static_cast<std::size_t>(csc.size()));
    std::vector<T> values(static_cast<std::size_t>(csc.size()));
    std::vector<size_type> work(row_ptr.begin(), row_ptr.end());

    for (size_type col = 0; col < ncol; ++col) {
      for (auto j = cp[col]; j < cp[col + 1]; ++j) {
        auto row = ri[j];
        auto dest = work[static_cast<std::size_t>(row)]++;
        col_ind[static_cast<std::size_t>(dest)] = col;
        values[static_cast<std::size_t>(dest)] = sv[j];
      }
    }

    // Build CSR from indices
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csc.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = row_ptr[static_cast<std::size_t>(row)];
           j < row_ptr[static_cast<std::size_t>(row + 1)]; ++j) {
        indices.push_back(Index{row, col_ind[static_cast<std::size_t>(j)]});
      }
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  // -- CSR <-> MSR sparsity --

  Modified_sparse_row_sparsity
  to_modified_sparse_row(Compressed_row_sparsity const& csr);

  Compressed_row_sparsity
  to_compressed_row(Modified_sparse_row_sparsity const& msr);

  // -- CSR <-> MSR matrix --

  template<typename T>
  Modified_sparse_row_matrix<T>
  to_modified_sparse_row(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();
    auto diag_len = std::min(shape.row(), shape.column());

    // Collect indices for sparsity
    std::vector<Index> indices;
    indices.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        indices.push_back(Index{row, ci[j]});
      }
    }

    Modified_sparse_row_sparsity sparsity{shape, indices.begin(), indices.end()};

    // Fill diagonal
    std::vector<T> diagonal(static_cast<std::size_t>(diag_len), T{0});
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        if (ci[j] == row && row < diag_len) {
          diagonal[static_cast<std::size_t>(row)] = sv[j];
        }
      }
    }

    // Fill off-diagonal values
    auto od_rp = sparsity.off_diagonal_row_ptr();
    auto od_ci = sparsity.off_diagonal_col_ind();
    std::vector<T> off_diag(static_cast<std::size_t>(od_ci.size()), T{0});

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        if (ci[j] == row && row < diag_len) {
          continue;
        }
        // Find in off-diagonal col_ind
        for (auto k = od_rp[row]; k < od_rp[row + 1]; ++k) {
          if (od_ci[k] == ci[j]) {
            off_diag[static_cast<std::size_t>(k)] = sv[j];
            break;
          }
        }
      }
    }

    return Modified_sparse_row_matrix<T>{
      std::move(sparsity), std::move(diagonal), std::move(off_diag)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Modified_sparse_row_matrix<T> const& msr)
  {
    auto shape = msr.shape();
    auto nrow = shape.row();
    auto const& sp = msr.sparsity();
    auto diag_len = sp.diagonal_length();
    auto od_rp = sp.off_diagonal_row_ptr();
    auto od_ci = sp.off_diagonal_col_ind();
    auto diag = msr.diagonal();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(msr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      // Diagonal
      if (row < diag_len && sp.has_diagonal(row)) {
        entries.push_back({Index{row, row}, diag[row]});
      }
      // Off-diagonal
      for (auto j = od_rp[row]; j < od_rp[row + 1]; ++j) {
        entries.push_back({Index{row, od_ci[j]},
          msr(row, od_ci[j])});
      }
    }

    // Sort entries by (row, col) to match CSR constructor ordering
    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity csr_sp{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(csr_sp), std::move(values)};
  }

  // -- CSR <-> DIA sparsity --

  Diagonal_sparsity
  to_diagonal(Compressed_row_sparsity const& csr);

  Compressed_row_sparsity
  to_compressed_row(Diagonal_sparsity const& dia);

  // -- CSR <-> DIA matrix --

  template<typename T>
  Diagonal_matrix<T>
  to_diagonal(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Collect entries
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        entries.push_back({Index{row, ci[j]}, sv[j]});
      }
    }

    // Build DIA matrix from entries via shape + entry list
    // First, build sparsity from indices
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }
    Diagonal_sparsity sparsity{shape, indices.begin(), indices.end()};

    // Allocate and fill values
    auto off = sparsity.offsets();
    auto ncol = shape.column();
    std::vector<T> values(static_cast<std::size_t>(sparsity.size()), T{0});

    for (auto const& entry : entries) {
      auto offset = entry.index.column() - entry.index.row();
      auto it = std::lower_bound(off.begin(), off.end(), offset);
      auto diag_idx = static_cast<std::size_t>(std::distance(off.begin(), it));

      size_type pos = 0;
      for (std::size_t d = 0; d < diag_idx; ++d) {
        auto o = off[d];
        if (o >= 0) {
          pos += std::min(nrow, ncol - o);
        } else {
          pos += std::min(nrow + o, ncol);
        }
      }

      size_type within = (offset >= 0) ? entry.index.row() : entry.index.column();
      values[static_cast<std::size_t>(pos + within)] = entry.value;
    }

    return Diagonal_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Diagonal_matrix<T> const& dia)
  {
    auto shape = dia.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto off = dia.sparsity().offsets();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(dia.size()));

    for (auto offset : off) {
      size_type start_row, start_col, len;
      if (offset >= 0) {
        start_row = 0;
        start_col = offset;
        len = std::min(nrow, ncol - offset);
      } else {
        start_row = -offset;
        start_col = 0;
        len = std::min(nrow + offset, ncol);
      }
      for (size_type k = 0; k < len; ++k) {
        auto row = start_row + k;
        auto col = start_col + k;
        auto val = dia(row, col);
        entries.push_back({Index{row, col}, val});
      }
    }

    // Sort by (row, col)
    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  // -- CSR <-> ELL sparsity --

  Ellpack_sparsity
  to_ellpack(Compressed_row_sparsity const& csr);

  Compressed_row_sparsity
  to_compressed_row(Ellpack_sparsity const& ell);

  // -- CSR <-> ELL matrix --

  template<typename T>
  Ellpack_matrix<T>
  to_ellpack(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        entries.push_back({Index{row, ci[j]}, sv[j]});
      }
    }

    // Build sparsity
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }
    Ellpack_sparsity sparsity{shape, indices.begin(), indices.end()};

    auto max_nnz = sparsity.max_nnz_per_row();
    std::vector<T> values(
      static_cast<std::size_t>(nrow * max_nnz), T{0});

    auto ell_ci = sparsity.col_ind();
    for (auto const& entry : entries) {
      auto base = static_cast<std::size_t>(entry.index.row() * max_nnz);
      for (size_type k = 0; k < max_nnz; ++k) {
        if (ell_ci[base + static_cast<std::size_t>(k)] == entry.index.column()) {
          values[base + static_cast<std::size_t>(k)] = entry.value;
          break;
        }
      }
    }

    return Ellpack_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Ellpack_matrix<T> const& ell)
  {
    auto shape = ell.shape();
    auto nrow = shape.row();
    auto max_nnz = ell.sparsity().max_nnz_per_row();
    auto ci = ell.sparsity().col_ind();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(ell.size()));

    for (size_type row = 0; row < nrow; ++row) {
      auto base = static_cast<std::size_t>(row * max_nnz);
      for (size_type k = 0; k < max_nnz; ++k) {
        auto col = ci[base + static_cast<std::size_t>(k)];
        if (col == -1) break;
        entries.push_back({Index{row, col}, ell(row, col)});
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  // -- CSR <-> BSR sparsity --

  Block_sparse_row_sparsity
  to_block_sparse_row(Compressed_row_sparsity const& csr,
                      size_type block_rows, size_type block_cols);

  Compressed_row_sparsity
  to_compressed_row(Block_sparse_row_sparsity const& bsr);

  // -- CSR <-> BSR matrix --

  template<typename T>
  Block_sparse_row_matrix<T>
  to_block_sparse_row(Compressed_row_matrix<T> const& csr,
                      size_type block_rows, size_type block_cols)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Collect entries
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        entries.push_back({Index{row, ci[j]}, sv[j]});
      }
    }

    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Block_sparse_row_sparsity sparsity{
      shape, block_rows, block_cols, indices.begin(), indices.end()};

    auto num_blocks = sparsity.num_blocks();
    std::vector<T> values(
      static_cast<std::size_t>(num_blocks * block_rows * block_cols), T{0});

    auto bsr_rp = sparsity.row_ptr();
    auto bsr_ci = sparsity.col_ind();

    for (auto const& entry : entries) {
      auto br_idx = entry.index.row() / block_rows;
      auto bc_idx = entry.index.column() / block_cols;
      auto local_row = entry.index.row() % block_rows;
      auto local_col = entry.index.column() % block_cols;

      for (auto j = bsr_rp[br_idx]; j < bsr_rp[br_idx + 1]; ++j) {
        if (bsr_ci[j] == bc_idx) {
          auto offset = j * block_rows * block_cols
                      + local_row * block_cols + local_col;
          values[static_cast<std::size_t>(offset)] = entry.value;
          break;
        }
      }
    }

    return Block_sparse_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Block_sparse_row_matrix<T> const& bsr)
  {
    auto shape = bsr.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto br = bsr.sparsity().block_rows();
    auto bc = bsr.sparsity().block_cols();
    auto rp = bsr.sparsity().row_ptr();
    auto ci = bsr.sparsity().col_ind();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(bsr.size()));

    auto nbr = bsr.sparsity().num_block_rows();
    for (size_type block_row = 0; block_row < nbr; ++block_row) {
      for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
        auto block_col = ci[j];
        for (size_type lr = 0; lr < br; ++lr) {
          for (size_type lc = 0; lc < bc; ++lc) {
            auto row = block_row * br + lr;
            auto col = block_col * bc + lc;
            if (row < nrow && col < ncol) {
              entries.push_back({Index{row, col}, bsr(row, col)});
            }
          }
        }
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  // -- CSR <-> JAD sparsity --

  Jagged_diagonal_sparsity
  to_jagged_diagonal(Compressed_row_sparsity const& csr);

  Compressed_row_sparsity
  to_compressed_row(Jagged_diagonal_sparsity const& jad);

  // -- CSR <-> JAD matrix --

  template<typename T>
  Jagged_diagonal_matrix<T>
  to_jagged_diagonal(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Collect entries
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        entries.push_back({Index{row, ci[j]}, sv[j]});
      }
    }

    // Build sparsity from indices
    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }
    Jagged_diagonal_sparsity sparsity{shape, indices.begin(), indices.end()};

    // Build per-row value lists (sorted by column, matching CSR order)
    std::vector<std::vector<T>> row_vals(static_cast<std::size_t>(nrow));
    for (auto const& entry : entries) {
      row_vals[static_cast<std::size_t>(entry.index.row())].push_back(entry.value);
    }

    // Fill values in jagged diagonal order
    auto pm = sparsity.perm();
    auto jd = sparsity.jdiag();
    auto num_jdiags = std::ssize(jd) - 1;

    std::vector<T> values(static_cast<std::size_t>(sparsity.size()), T{0});

    for (size_type k = 0; k < num_jdiags; ++k) {
      auto width = jd[k + 1] - jd[k];
      for (size_type i = 0; i < width; ++i) {
        auto orig_row = pm[i];
        values[static_cast<std::size_t>(jd[k] + i)]
          = row_vals[static_cast<std::size_t>(orig_row)][static_cast<std::size_t>(k)];
      }
    }

    return Jagged_diagonal_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Jagged_diagonal_matrix<T> const& jad)
  {
    auto shape = jad.shape();
    auto const& sp = jad.sparsity();
    auto pm = sp.perm();
    auto jd = sp.jdiag();
    auto jad_ci = sp.col_ind();
    auto num_jdiags = std::ssize(jd) - 1;

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(jad.size()));

    for (size_type k = 0; k < num_jdiags; ++k) {
      auto width = jd[k + 1] - jd[k];
      for (size_type i = 0; i < width; ++i) {
        auto orig_row = pm[i];
        auto pos = jd[k] + i;
        entries.push_back({
          Index{orig_row, jad_ci[pos]},
          jad(orig_row, jad_ci[pos])});
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity csr_sp{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(csr_sp), std::move(values)};
  }

  // -- CSR <-> sCSR sparsity --

  Compressed_row_sparsity
  to_compressed_row(Symmetric_compressed_row_sparsity const& scsr);

  Symmetric_compressed_row_sparsity
  to_symmetric_compressed_row(Compressed_row_sparsity const& csr);

  // -- CSR <-> sCSR matrix --

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Symmetric_compressed_row_matrix<T> const& scsr)
  {
    auto rp = scsr.row_ptr();
    auto ci = scsr.col_ind();
    auto sv = scsr.values();
    auto shape = scsr.shape();
    auto nrow = shape.row();

    // Expand lower triangle to full matrix:
    // for each stored (i,j): emit (i,j); if i != j also emit (j,i)
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(scsr.size() * 2));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        auto col = ci[j];
        auto val = sv[j];
        entries.push_back({Index{row, col}, val});
        if (row != col) {
          entries.push_back({Index{col, row}, val});
        }
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Symmetric_compressed_row_matrix<T>
  to_symmetric_compressed_row(Compressed_row_matrix<T> const& csr)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Extract lower triangle entries (row >= col)
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));

    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        if (row >= ci[j]) {
          entries.push_back({Index{row, ci[j]}, sv[j]});
        }
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Symmetric_compressed_row_sparsity sparsity{
      shape, indices.begin(), indices.end()};
    return Symmetric_compressed_row_matrix<T>{
      std::move(sparsity), std::move(values)};
  }

  // -- sCOO -> sCSR sparsity --

  Symmetric_compressed_row_sparsity
  to_symmetric_compressed_row(Symmetric_coordinate_sparsity const& scoo);

  // -- sCOO -> CSR sparsity --

  Compressed_row_sparsity
  to_compressed_row(Symmetric_coordinate_sparsity const& scoo);

  // -- sCOO -> sCSR matrix --

  template<typename T>
  Symmetric_compressed_row_matrix<T>
  to_symmetric_compressed_row(Symmetric_coordinate_matrix<T> const& scoo)
  {
    auto entries = scoo.entries();

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.first.row() < b.first.row()
        || (a.first.row() == b.first.row()
            && a.first.column() < b.first.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& [index, value] : entries) {
      indices.push_back(index);
      values.push_back(value);
    }

    Symmetric_compressed_row_sparsity sparsity{
      scoo.shape(), indices.begin(), indices.end()};

    return Symmetric_compressed_row_matrix<T>{
      std::move(sparsity), std::move(values)};
  }

  // -- sCOO -> CSR matrix --

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Symmetric_coordinate_matrix<T> const& scoo)
  {
    // Route through sCSR as intermediate
    auto scsr = to_symmetric_compressed_row(scoo);
    return to_compressed_row(scsr);
  }

  // -- CSR <-> sBSR sparsity --

  Symmetric_block_sparse_row_sparsity
  to_symmetric_block_sparse_row(
    Compressed_row_sparsity const& csr,
    size_type block_rows, size_type block_cols);

  Compressed_row_sparsity
  to_compressed_row(Symmetric_block_sparse_row_sparsity const& sbsr);

  // -- CSR <-> sBSR matrix --

  template<typename T>
  Symmetric_block_sparse_row_matrix<T>
  to_symmetric_block_sparse_row(
    Compressed_row_matrix<T> const& csr,
    size_type block_rows, size_type block_cols)
  {
    auto rp = csr.row_ptr();
    auto ci = csr.col_ind();
    auto sv = csr.values();
    auto shape = csr.shape();
    auto nrow = shape.row();

    // Collect entries and normalize to lower-triangle blocks
    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(csr.size()));
    for (size_type row = 0; row < nrow; ++row) {
      for (auto j = rp[row]; j < rp[row + 1]; ++j) {
        auto br_idx = row / block_rows;
        auto bc_idx = ci[j] / block_cols;
        if (br_idx < bc_idx) {
          // Swap block and local indices
          auto local_row = row % block_rows;
          auto local_col = ci[j] % block_cols;
          auto new_row = bc_idx * block_rows + local_col;
          auto new_col = br_idx * block_cols + local_row;
          entries.push_back({Index{new_row, new_col}, sv[j]});
        } else {
          entries.push_back({Index{row, ci[j]}, sv[j]});
        }
      }
    }

    std::vector<Index> indices;
    indices.reserve(entries.size());
    for (auto const& e : entries) {
      indices.push_back(e.index);
    }

    Symmetric_block_sparse_row_sparsity sparsity{
      shape, block_rows, block_cols, indices.begin(), indices.end()};

    auto num_blocks = sparsity.num_blocks();
    std::vector<T> values(
      static_cast<std::size_t>(num_blocks * block_rows * block_cols), T{0});

    auto sbsr_rp = sparsity.row_ptr();
    auto sbsr_ci = sparsity.col_ind();

    for (auto const& entry : entries) {
      auto br_idx = entry.index.row() / block_rows;
      auto bc_idx = entry.index.column() / block_cols;
      auto local_row = entry.index.row() % block_rows;
      auto local_col = entry.index.column() % block_cols;

      for (auto j = sbsr_rp[br_idx]; j < sbsr_rp[br_idx + 1]; ++j) {
        if (sbsr_ci[j] == bc_idx) {
          auto offset = j * block_rows * block_cols
                      + local_row * block_cols + local_col;
          values[static_cast<std::size_t>(offset)] = entry.value;
          break;
        }
      }
    }

    return Symmetric_block_sparse_row_matrix<T>{
      std::move(sparsity), std::move(values)};
  }

  template<typename T>
  Compressed_row_matrix<T>
  to_compressed_row(Symmetric_block_sparse_row_matrix<T> const& sbsr)
  {
    auto shape = sbsr.shape();
    auto nrow = shape.row();
    auto ncol = shape.column();
    auto br = sbsr.sparsity().block_rows();
    auto bc = sbsr.sparsity().block_cols();
    auto rp = sbsr.sparsity().row_ptr();
    auto ci = sbsr.sparsity().col_ind();

    std::vector<Entry<T>> entries;
    entries.reserve(static_cast<std::size_t>(sbsr.size() * 2));

    auto nbr = sbsr.sparsity().num_block_rows();
    for (size_type block_row = 0; block_row < nbr; ++block_row) {
      for (auto j = rp[block_row]; j < rp[block_row + 1]; ++j) {
        auto block_col = ci[j];
        for (size_type lr = 0; lr < br; ++lr) {
          for (size_type lc = 0; lc < bc; ++lc) {
            auto row = block_row * br + lr;
            auto col = block_col * bc + lc;
            if (row < nrow && col < ncol) {
              auto val = sbsr(row, col);
              entries.push_back({Index{row, col}, val});
              if (block_row != block_col) {
                entries.push_back({Index{col, row}, sbsr(col, row)});
              }
            }
          }
        }
      }
    }

    auto by_row_col = [](auto const& a, auto const& b) {
      return a.index.row() < b.index.row()
        || (a.index.row() == b.index.row()
            && a.index.column() < b.index.column());
    };
    std::sort(entries.begin(), entries.end(), by_row_col);

    // Deduplicate
    auto same_index = [](auto const& a, auto const& b) {
      return a.index == b.index;
    };
    entries.erase(
      std::unique(entries.begin(), entries.end(), same_index),
      entries.end());

    std::vector<Index> indices;
    std::vector<T> values;
    indices.reserve(entries.size());
    values.reserve(entries.size());

    for (auto const& e : entries) {
      indices.push_back(e.index);
      values.push_back(e.value);
    }

    Compressed_row_sparsity sparsity{shape, indices.begin(), indices.end()};
    return Compressed_row_matrix<T>{std::move(sparsity), std::move(values)};
  }

} // end of namespace sparkit::data::detail
